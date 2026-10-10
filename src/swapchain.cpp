#include "internal.hpp"

#include "noorrhi/device.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <thread>

namespace noorrhi {
namespace detail {
namespace {

// Presentation images live in GENERAL like every other image this library
// owns, which a render pass, a blit and a compute write all accept. Only the handoff to the presentation
// engine needs a specific layout, and that transition is what this helper
// exists for.
void transition(const vk::CommandBuffer command, const vk::Image image,
    const vk::ImageLayout from, const vk::ImageLayout to) {
    vk::ImageMemoryBarrier2 barrier{};
    barrier.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
        .setSrcAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
        .setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
        .setDstAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
        .setOldLayout(from)
        .setNewLayout(to)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(image)
        .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    command.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
}

ImageFormat to_public_format(const vk::Format format) {
    switch (format) {
    case vk::Format::eB8G8R8A8Unorm: return ImageFormat::Bgra8Unorm;
    case vk::Format::eR8G8B8A8Unorm: return ImageFormat::Rgba8Unorm;
    case vk::Format::eR32G32B32A32Sfloat: return ImageFormat::Rgba32Float;
    case vk::Format::eR16G16B16A16Sfloat: return ImageFormat::Rgba16Float;
    case vk::Format::eR32Uint: return ImageFormat::R32Uint;
    case vk::Format::eR32Sfloat: return ImageFormat::R32Float;
    default: return ImageFormat::Bgra8Unorm;
    }
}

// Pick the caller's preferred format when the surface offers it, and fall back
// to whatever the surface does offer rather than failing: a swapchain format
// is a negotiation, not a requirement the caller can insist on.
vk::SurfaceFormatKHR choose_format(const std::vector<vk::SurfaceFormatKHR>& available,
    const ImageFormat preferred) {
    if (available.empty())
        throw Error(ErrorCode::UnsupportedFeature, "surface reports no formats");
    if (preferred != ImageFormat::Auto) {
        const vk::Format wanted = to_vulkan_format(preferred);
        for (const auto& candidate : available)
            if (candidate.format == wanted
                && candidate.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
                return candidate;
    }
    for (const auto& candidate : available)
        if ((candidate.format == vk::Format::eB8G8R8A8Unorm
                || candidate.format == vk::Format::eR8G8B8A8Unorm)
            && candidate.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            return candidate;
    return available.front();
}

// Walk the policy's preference order and take the first mode the surface
// offers. FIFO is the only mode the spec guarantees, so every order ends there.
vk::PresentModeKHR choose_present_mode(const std::vector<vk::PresentModeKHR>& available,
    const PresentMode policy, const bool fifo_latest_ready) {
    using Mode = vk::PresentModeKHR;
    static constexpr std::array low_latency{Mode::eMailbox, Mode::eFifoLatestReadyEXT, Mode::eFifo};
    static constexpr std::array vsync{Mode::eFifo};
    static constexpr std::array immediate{
        Mode::eImmediate, Mode::eMailbox, Mode::eFifoLatestReadyEXT, Mode::eFifo};
    const std::span<const Mode> order = policy == PresentMode::Immediate ? std::span<const Mode>(immediate)
        : policy == PresentMode::Vsync ? std::span<const Mode>(vsync)
        : std::span<const Mode>(low_latency);
    for (const Mode mode : order) {
        // The mode is only valid with its device extension enabled, even when
        // the surface lists it.
        if (mode == Mode::eFifoLatestReadyEXT && !fifo_latest_ready)
            continue;
        if (std::ranges::find(available, mode) != available.end())
            return mode;
    }
    return Mode::eFifo;
}

ActivePresentMode to_active(const vk::PresentModeKHR mode) {
    switch (mode) {
    case vk::PresentModeKHR::eImmediate: return ActivePresentMode::Immediate;
    case vk::PresentModeKHR::eMailbox: return ActivePresentMode::Mailbox;
    case vk::PresentModeKHR::eFifoLatestReadyEXT: return ActivePresentMode::FifoLatestReady;
    default: return ActivePresentMode::Fifo;
    }
}

// MAILBOX and IMMEDIATE replace a waiting image instead of queueing behind it,
// so they need one image for the display, one waiting and one being rendered
// for acquire never to block. The FIFO modes queue every image, where extra
// images only add latency.
std::uint32_t choose_image_count(const vk::SurfaceCapabilitiesKHR& capabilities,
    const vk::PresentModeKHR mode) {
    std::uint32_t count = capabilities.minImageCount + 1;
    if (mode == vk::PresentModeKHR::eMailbox || mode == vk::PresentModeKHR::eImmediate)
        count = std::max(count, 3u);
    if (capabilities.maxImageCount > 0)
        count = std::min(count, capabilities.maxImageCount);
    return count;
}

vk::CompositeAlphaFlagBitsKHR choose_composite_alpha(const vk::SurfaceCapabilitiesKHR& capabilities,
    const bool transparent) {
    const auto supported = capabilities.supportedCompositeAlpha;
    const auto supports = [supported](const vk::CompositeAlphaFlagBitsKHR candidate) {
        return (static_cast<VkCompositeAlphaFlagsKHR>(supported)
            & static_cast<VkCompositeAlphaFlagsKHR>(candidate)) != 0;
    };

    static constexpr std::array transparent_modes{
        vk::CompositeAlphaFlagBitsKHR::eInherit,
        vk::CompositeAlphaFlagBitsKHR::ePreMultiplied,
        vk::CompositeAlphaFlagBitsKHR::ePostMultiplied,
        vk::CompositeAlphaFlagBitsKHR::eOpaque,
    };
    static constexpr std::array opaque_modes{
        vk::CompositeAlphaFlagBitsKHR::eOpaque,
        vk::CompositeAlphaFlagBitsKHR::eInherit,
        vk::CompositeAlphaFlagBitsKHR::ePostMultiplied,
        vk::CompositeAlphaFlagBitsKHR::ePreMultiplied,
    };
    const auto& modes = transparent ? transparent_modes : opaque_modes;
    for (const auto mode : modes)
        if (supports(mode)) return mode;

    throw Error(ErrorCode::UnsupportedFeature, "surface reports no supported composite-alpha mode");
}

} // namespace

SwapchainImpl::~SwapchainImpl() {
    // No wait: work in flight may still present to these images, and the
    // retire queue releases them once it has finished.
    if (device)
        device->retire_chain(*this);
}

std::shared_ptr<ImageImpl> DeviceImpl::wrap_presentation_image(const vk::Image image,
    const vk::Format format, const ImageFormat public_format,
    const std::uint32_t width, const std::uint32_t height) {
    auto result = std::make_shared<ImageImpl>();
    result->device = self_.lock();
    result->image = image;
    result->allocation = VK_NULL_HANDLE;
    result->owns_image = false;
    result->format = format;
    result->public_format = public_format;
    result->aspect = vk::ImageAspectFlagBits::eColor;
    result->width = width;
    result->height = height;
    result->byte_size = format_byte_size(public_format, width, height);
    // A presentation image is a render target and a blit destination, never a
    // shader resource, so it needs an identity handle but no texture slot.
    result->handle = ImageHandle{result};
    const vk::ImageViewCreateInfo view_info({}, image, vk::ImageViewType::e2D, format, {},
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    result->view = vk_device().createImageViewUnique(view_info);
    return result;
}

void DeviceImpl::retire_chain(SwapchainImpl& chain) {
    if (!chain.swapchain && chain.images.empty())
        return;
    struct Retirement {
        std::shared_ptr<SurfaceImpl> surface;
        vk::UniqueSwapchainKHR swapchain;
        std::vector<std::shared_ptr<ImageImpl>> images;
        std::vector<SwapchainImpl::Acquire> acquire_semaphores;
        std::vector<vk::UniqueSemaphore> present_semaphores;
    };
    auto retirement = std::make_shared<Retirement>();
    retirement->surface = chain.surface;
    retirement->swapchain = std::move(chain.swapchain);
    retirement->images = std::move(chain.images);
    retirement->acquire_semaphores = std::move(chain.acquire_semaphores);
    retirement->present_semaphores = std::move(chain.present_semaphores);
    chain.images.clear();
    chain.acquire_semaphores.clear();
    chain.present_semaphores.clear();
    chain.presented.clear();
    retire([retirement] {
        // Views belong to images the swapchain owns, so they must go first,
        // here, rather than through each image's own deferred release.
        for (const auto& image : retirement->images)
            if (image)
                image->view.reset();
        retirement->images.clear();
        retirement->acquire_semaphores.clear();
        retirement->present_semaphores.clear();
        retirement->swapchain.reset();
        retirement->surface.reset();
    });
}

void DeviceImpl::rebuild_swapchain(SwapchainImpl& chain) {
    const vk::SurfaceKHR surface = chain.surface->surface.get();
    const auto capabilities = physical_device_.getSurfaceCapabilitiesKHR(surface);
    std::uint32_t width = chain.provider->width();
    std::uint32_t height = chain.provider->height();
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
        width = capabilities.currentExtent.width;
        height = capabilities.currentExtent.height;
    }
    width = std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    height = std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    if (width == 0 || height == 0) {
        // A minimised window has a zero-sized surface. Leave the chain stale
        // so begin_frame keeps returning nothing until it comes back.
        chain.width = chain.height = 0;
        chain.stale = true;
        return;
    }

    const auto surface_format = choose_format(physical_device_.getSurfaceFormatsKHR(surface),
        chain.format == vk::Format::eUndefined ? chain.public_format : ImageFormat::Auto);
    chain.format = surface_format.format;
    chain.color_space = surface_format.colorSpace;
    chain.public_format = to_public_format(surface_format.format);
    chain.present_mode = choose_present_mode(physical_device_.getSurfacePresentModesKHR(surface),
        chain.requested_mode, fifo_latest_ready_enabled_);

    vk::SwapchainCreateInfoKHR info{};
    info.setSurface(surface)
        .setMinImageCount(choose_image_count(capabilities, chain.present_mode))
        .setImageFormat(chain.format)
        .setImageColorSpace(chain.color_space)
        .setImageExtent({width, height})
        .setImageArrayLayers(1)
        // Colour attachment for render scopes, transfer destination for the
        // blit that composites a rendered image onto the frame.
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment
            | vk::ImageUsageFlagBits::eTransferDst
            | vk::ImageUsageFlagBits::eTransferSrc)
        .setImageSharingMode(vk::SharingMode::eExclusive)
        .setPreTransform(capabilities.currentTransform)
        .setCompositeAlpha(choose_composite_alpha(capabilities, chain.transparent))
        .setPresentMode(chain.present_mode)
        .setClipped(VK_TRUE)
        // Handing over the old chain lets the driver reuse its resources and
        // keep presenting its queued images while the new one takes over.
        .setOldSwapchain(chain.swapchain.get());

    auto replacement = vk_device().createSwapchainKHRUnique(info);
    // The old chain may still be referenced by submitted frames. Retire it
    // rather than waiting for the GPU: a resize must never stall the queue.
    retire_chain(chain);
    chain.swapchain = std::move(replacement);

    const auto raw_images = vk_device().getSwapchainImagesKHR(chain.swapchain.get());
    chain.images.reserve(raw_images.size());
    for (const vk::Image image : raw_images)
        chain.images.push_back(
            wrap_presentation_image(image, chain.format, chain.public_format, width, height));
    chain.presented.assign(raw_images.size(), false);
    for (std::size_t i = 0; i < raw_images.size(); ++i) {
        chain.acquire_semaphores.push_back({vk_device().createSemaphoreUnique({}), {}});
        chain.present_semaphores.push_back(vk_device().createSemaphoreUnique({}));
    }
    chain.semaphore_cursor = 0;
    chain.width = width;
    chain.height = height;
    chain.stale = false;
}

std::shared_ptr<SwapchainImpl> DeviceImpl::create_swapchain(SurfaceProvider& provider,
    const SwapchainDesc& desc) {
    if (!presenting())
        throw Error(ErrorCode::InvalidState,
            "noorrhi::Swapchain requires a Device created with DeviceConfig::presentation");
    auto surface = std::make_shared<SurfaceImpl>();
    const std::uintptr_t raw = provider.create_surface(
        reinterpret_cast<std::uintptr_t>(static_cast<VkInstance>(vk_instance())));
    if (!raw)
        throw Error(ErrorCode::InvalidState, "SurfaceProvider failed to create a surface");
    surface->surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(reinterpret_cast<VkSurfaceKHR>(raw)),
        vk::detail::ObjectDestroy<vk::Instance, VULKAN_HPP_DEFAULT_DISPATCHER_TYPE>(vk_instance()));
    // Every frame is submitted and presented on the device's one queue.
    if (!physical_device_.getSurfaceSupportKHR(queue_state(Queue::Graphics).family, surface->surface.get()))
        throw Error(ErrorCode::UnsupportedFeature,
            "the device's queue cannot present to this surface");

    auto chain = std::make_shared<SwapchainImpl>();
    chain->device = self_.lock();
    chain->provider = &provider;
    chain->surface = std::move(surface);
    chain->public_format = desc.format == ImageFormat::Auto ? ImageFormat::Bgra8Unorm : desc.format;
    chain->requested_mode = desc.present_mode;
    chain->max_frames_in_flight = std::max(desc.max_frames_in_flight, 1u);
    chain->transparent = desc.transparent;
    std::lock_guard lock(mutex_);
    rebuild_swapchain(*chain);
    return chain;
}

bool DeviceImpl::wait_frame_slot(SwapchainImpl& chain, const bool block) {
    for (;;) {
        std::uint64_t oldest = 0;
        {
            std::lock_guard lock(mutex_);
            if (shut_down_)
                throw Error(ErrorCode::InvalidState, "noorrhi::Device has been shut down");
            const std::uint64_t completed = completed_value(Queue::Graphics);
            while (!chain.in_flight.empty() && chain.in_flight.front().value <= completed)
                chain.in_flight.pop_front();
            if (chain.in_flight.size() < chain.max_frames_in_flight)
                return true;
            if (!block)
                return false;
            oldest = chain.in_flight.front().value;
        }
        wait({oldest});
    }
}

std::shared_ptr<Frame::State> DeviceImpl::begin_frame(
    const std::shared_ptr<SwapchainImpl>& chain, const bool wait_for_slot) {
    if (!chain)
        throw Error(ErrorCode::InvalidArgument, "begin_frame requires a valid Swapchain");
    // Pace before acquiring. With one frame in flight the CPU never queues
    // work behind a busy GPU, which is where input latency would accumulate.
    if (!wait_frame_slot(*chain, wait_for_slot))
        return {};
    std::lock_guard lock(mutex_);
    if (shut_down_)
        throw Error(ErrorCode::InvalidState, "noorrhi::Device has been shut down");
    QueueState& graphics = queue_state(Queue::Graphics);
    if (graphics.recording)
        throw Error(ErrorCode::InvalidState, "a frame or Recording is already open on Queue::Graphics");
    reap_completed();

    const bool size_changed = chain->width != chain->provider->width()
        || chain->height != chain->provider->height();
    if (chain->stale || size_changed)
        rebuild_swapchain(*chain);
    if (chain->stale || chain->images.empty())
        return {};

    const std::uint32_t semaphore_index = chain->semaphore_cursor;
    chain->semaphore_cursor =
        (chain->semaphore_cursor + 1) % static_cast<std::uint32_t>(chain->acquire_semaphores.size());

    // A binary acquire semaphore may only be reused once the submission that
    // waited on it has started. Frame pacing normally guarantees that already.
    const auto& slot = chain->acquire_semaphores[semaphore_index];
    if (slot.token.value) {
        const vk::Semaphore semaphore = graphics.timeline.get();
        const vk::SemaphoreWaitInfo wait_info({}, semaphore, slot.token.value);
        if (vk_device().waitSemaphores(wait_info, UINT64_MAX) != vk::Result::eSuccess)
            throw Error(ErrorCode::DeviceLost, "swapchain acquire semaphore wait failed");
    }
    std::uint32_t image_index = 0;
    const vk::Result acquired = vk_device().acquireNextImageKHR(chain->swapchain.get(),
        wait_for_slot ? std::numeric_limits<std::uint64_t>::max() : 0,
        chain->acquire_semaphores[semaphore_index].semaphore.get(), {}, &image_index);
    if (!wait_for_slot && (acquired == vk::Result::eTimeout || acquired == vk::Result::eNotReady))
        return {};
    if (acquired == vk::Result::eErrorOutOfDateKHR) {
        chain->stale = true;
        return {};
    }
    if (acquired != vk::Result::eSuccess && acquired != vk::Result::eSuboptimalKHR)
        throw Error(ErrorCode::DeviceLost, "swapchain image acquisition failed");
    // A suboptimal chain still presents correctly, so this frame is rendered
    // and the rebuild happens on the next one.
    if (acquired == vk::Result::eSuboptimalKHR)
        chain->stale = true;

    CommandPool& pool = thread_command_pool(Queue::Graphics);
    auto state = std::make_shared<Frame::State>();
    state->device = self_.lock();
    state->swapchain = chain;
    state->command = allocate_command(pool);
    state->image_index = image_index;
    state->semaphore_index = semaphore_index;
    state->target = chain->images[image_index]->handle;
    state->open = true;

    state->command.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    transition(state->command, chain->images[image_index]->image,
        chain->presented[image_index] ? vk::ImageLayout::ePresentSrcKHR
                                      : vk::ImageLayout::eUndefined,
        vk::ImageLayout::eGeneral);
    // Work inside the frame is ordered explicitly, so order it once against
    // everything submitted before the frame.
    record_full_barrier(state->command);

    graphics.recording = state->command;
    graphics.recording_pool = &pool;
    graphics.recording_value = graphics.next_timeline.load();
    graphics.recording_thread = std::this_thread::get_id();
    graphics.recording_resources.clear();
    return state;
}

void DeviceImpl::end_frame(Frame::State& state) {
    std::lock_guard lock(mutex_);
    QueueState& graphics = queue_state(Queue::Graphics);
    if (!state.open || graphics.recording != state.command)
        throw Error(ErrorCode::InvalidState, "present called without a matching begin_frame");
    SwapchainImpl& chain = *state.swapchain;

    // Publishes the frame's explicitly ordered work to later submissions.
    record_full_barrier(state.command);
    transition(state.command, chain.images[state.image_index]->image,
        vk::ImageLayout::eGeneral, vk::ImageLayout::ePresentSrcKHR);
    state.command.end();

    const GpuToken token{graphics.recording_value, Queue::Graphics};
    const vk::Semaphore presented = chain.present_semaphores[state.image_index].get();
    // The binary acquire semaphore goes first, with an ignored value, ahead of
    // the timeline waits queue_wait() asked for.
    QueueWaits acquire;
    acquire.semaphores.push_back(chain.acquire_semaphores[state.semaphore_index].semaphore.get());
    acquire.values.push_back(0);
    acquire.stages.push_back(vk::PipelineStageFlagBits::eAllCommands);
    submit_command(graphics, token, state.command, graphics.recording_pool,
        std::move(graphics.recording_resources), acquire, std::span(&presented, 1));
    graphics.recording = nullptr;
    graphics.recording_pool = nullptr;
    graphics.recording_value = 0;
    graphics.recording_resources.clear();
    chain.acquire_semaphores[state.semaphore_index].token = token;
    chain.in_flight.push_back(token);
    state.open = false;
    state.command = nullptr;
    chain.presented[state.image_index] = true;

    // The present waits on the submission's semaphore inside the driver, so
    // the CPU returns immediately instead of waiting for the frame to finish.
    vk::PresentInfoKHR present_info{};
    const vk::SwapchainKHR raw_swapchain = chain.swapchain.get();
    present_info.setWaitSemaphores(presented)
        .setSwapchains(raw_swapchain)
        .setPImageIndices(&state.image_index);
    const vk::Result presented_result = graphics.queue.presentKHR(&present_info);
    if (presented_result == vk::Result::eErrorOutOfDateKHR
        || presented_result == vk::Result::eSuboptimalKHR)
        chain.stale = true;
    else if (presented_result != vk::Result::eSuccess)
        throw Error(ErrorCode::DeviceLost, "swapchain presentation failed");
}

} // namespace detail

// --- Public swapchain surface -------------------------------------------

Swapchain::Swapchain(Device& device, SurfaceProvider& surface, const SwapchainDesc& desc) {
    if (!device.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot create a swapchain on an empty noorrhi::Device");
    impl_ = device.impl_->create_swapchain(surface, desc);
}

void Swapchain::wait_until_ready() {
    if (!impl_)
        throw Error(ErrorCode::InvalidResource, "swapchain is empty");
    (void)impl_->device->wait_frame_slot(*impl_);
}

Frame Swapchain::begin_frame() {
    if (!impl_)
        throw Error(ErrorCode::InvalidResource, "swapchain is empty");
    return Frame(impl_->device->begin_frame(impl_));
}

Frame Swapchain::try_begin_frame() {
    if (!impl_)
        throw Error(ErrorCode::InvalidResource, "swapchain is empty");
    return Frame(impl_->device->begin_frame(impl_, false));
}

void Swapchain::present(Frame&& frame) {
    if (!impl_)
        throw Error(ErrorCode::InvalidResource, "swapchain is empty");
    if (!frame.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot present an unacquired frame");
    if (frame.impl_->swapchain != impl_)
        throw Error(ErrorCode::InvalidArgument, "frame was acquired from a different swapchain");
    impl_->device->end_frame(*frame.impl_);
}

void Swapchain::set_present_mode(const PresentMode mode) {
    if (!impl_ || impl_->requested_mode == mode)
        return;
    impl_->requested_mode = mode;
    impl_->stale = true;
}

PresentMode Swapchain::present_mode() const noexcept {
    return impl_ ? impl_->requested_mode : PresentMode::LowLatency;
}

ActivePresentMode Swapchain::active_present_mode() const noexcept {
    return impl_ ? detail::to_active(impl_->present_mode) : ActivePresentMode::Fifo;
}

ImageFormat Swapchain::format() const noexcept {
    return impl_ ? impl_->public_format : ImageFormat::Auto;
}
std::uint32_t Swapchain::width() const noexcept { return impl_ ? impl_->width : 0; }
std::uint32_t Swapchain::height() const noexcept { return impl_ ? impl_->height : 0; }
std::uint32_t Swapchain::image_count() const noexcept {
    return impl_ ? static_cast<std::uint32_t>(impl_->images.size()) : 0;
}
void Swapchain::invalidate() {
    if (impl_)
        impl_->stale = true;
}

Frame::Frame(Frame&&) noexcept = default;
Frame& Frame::operator=(Frame&& other) noexcept {
    if (this != &other) {
        Frame previous(std::move(*this));
        impl_ = std::move(other.impl_);
    }
    return *this;
}

Frame::~Frame() {
    // A frame dropped without present discards its work rather than
    // presenting a half-recorded image. The command buffer and any resources
    // it referenced are released with it.
    if (impl_ && impl_->open && impl_->device) {
        try {
            impl_->device->abandon_frame(*impl_);
        } catch (...) {
            // Destruction cannot report device loss.
        }
    }
}

ImageHandle Frame::target() const noexcept { return impl_ ? impl_->target : ImageHandle{}; }
std::uint32_t Frame::index() const noexcept { return impl_ ? impl_->image_index : 0; }
std::uint32_t Frame::width() const noexcept {
    return impl_ && impl_->swapchain ? impl_->swapchain->width : 0;
}
std::uint32_t Frame::height() const noexcept {
    return impl_ && impl_->swapchain ? impl_->swapchain->height : 0;
}
ImageFormat Frame::format() const noexcept {
    return impl_ && impl_->swapchain ? impl_->swapchain->public_format : ImageFormat::Auto;
}

} // namespace noorrhi
