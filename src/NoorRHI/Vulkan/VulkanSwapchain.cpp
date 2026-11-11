#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Present/Swapchain.h"
#include "NoorRHI/Vulkan/Vulkan.h"

namespace NoorRHI::Backend {

namespace {

std::optional<ImageFormat> getPublicFormat(const vk::Format format) {
    for (std::uint8_t index = 1; index <= static_cast<std::uint8_t>(ImageFormat::Bc7Srgb); ++index)
        if (getVulkanFormat(static_cast<ImageFormat>(index)) == format)
            return static_cast<ImageFormat>(index);
    return std::nullopt;
}

std::expected<vk::SurfaceFormatKHR, Error> chooseFormat(const std::vector<vk::SurfaceFormatKHR>& available, const ImageFormat preferred) {
    const auto isUsable = [](const vk::SurfaceFormatKHR& candidate, const vk::Format format) {
        return candidate.format == format && candidate.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
    };
    if (preferred != ImageFormat::Undefined) {
        for (const vk::SurfaceFormatKHR& candidate : available)
            if (isUsable(candidate, getVulkanFormat(preferred)))
                return candidate;
        return fail(ErrorCode::Unsupported, "the surface does not offer the requested swapchain format");
    }
    for (const vk::Format format : {vk::Format::eB8G8R8A8Unorm, vk::Format::eR8G8B8A8Unorm})
        for (const vk::SurfaceFormatKHR& candidate : available)
            if (isUsable(candidate, format))
                return candidate;
    return fail(ErrorCode::Unsupported, "the surface offers no 8-bit RGBA or BGRA format");
}

vk::PresentModeKHR choosePresentMode(const std::vector<vk::PresentModeKHR>& available, const PresentMode policy, const bool fifoLatestReady) {
    using Mode = vk::PresentModeKHR;
    static constexpr std::array lowLatency{Mode::eMailbox, Mode::eFifoLatestReadyEXT, Mode::eFifo};
    static constexpr std::array vsync{Mode::eFifo};
    static constexpr std::array immediate{Mode::eImmediate, Mode::eMailbox, Mode::eFifoLatestReadyEXT, Mode::eFifo};
    const std::span<const Mode> order = policy == PresentMode::Immediate ? std::span<const Mode>(immediate)
                                        : policy == PresentMode::Vsync   ? std::span<const Mode>(vsync)
                                                                         : std::span<const Mode>(lowLatency);
    for (const Mode mode : order) {
        if (mode == Mode::eFifoLatestReadyEXT && !fifoLatestReady)
            continue;
        if (std::ranges::contains(available, mode))
            return mode;
    }
    //! FIFO is the only present mode every surface supports.
    return Mode::eFifo;
}

ActivePresentMode getActiveMode(const vk::PresentModeKHR mode) {
    switch (mode) {
    case vk::PresentModeKHR::eImmediate:
        return ActivePresentMode::Immediate;
    case vk::PresentModeKHR::eMailbox:
        return ActivePresentMode::Mailbox;
    case vk::PresentModeKHR::eFifoLatestReadyEXT:
        return ActivePresentMode::FifoLatestReady;
    default:
        return ActivePresentMode::Fifo;
    }
}

std::uint32_t chooseImageCount(const vk::SurfaceCapabilitiesKHR& capabilities, const vk::PresentModeKHR mode) {
    std::uint32_t count = capabilities.minImageCount + 1;
    //! Mailbox and immediate replace a waiting image, so three images keep acquire from blocking.
    if (mode == vk::PresentModeKHR::eMailbox || mode == vk::PresentModeKHR::eImmediate)
        count = std::max(count, 3u);
    if (capabilities.maxImageCount > 0)
        count = std::min(count, capabilities.maxImageCount);
    return count;
}

std::expected<vk::CompositeAlphaFlagBitsKHR, Error> chooseCompositeAlpha(const vk::SurfaceCapabilitiesKHR& capabilities, const bool transparent) {
    using Alpha = vk::CompositeAlphaFlagBitsKHR;
    static constexpr std::array transparentOrder{Alpha::ePreMultiplied, Alpha::ePostMultiplied, Alpha::eInherit, Alpha::eOpaque};
    static constexpr std::array opaqueOrder{Alpha::eOpaque, Alpha::eInherit, Alpha::ePostMultiplied, Alpha::ePreMultiplied};
    for (const Alpha mode : transparent ? std::span<const Alpha>(transparentOrder) : std::span<const Alpha>(opaqueOrder))
        if (capabilities.supportedCompositeAlpha & mode)
            return mode;
    return fail(ErrorCode::Unsupported, "the surface reports no composite alpha mode");
}

}

VulkanSwapchain::~VulkanSwapchain() {
    if (!device)
        return;
    retireChain();
    device->retire([retiredSurface = std::make_shared<vk::UniqueSurfaceKHR>(std::move(surface))] {});
}

void VulkanSwapchain::retireChain() {
    if (!swapchain && images.empty())
        return;
    struct Chain {
        vk::UniqueSwapchainKHR swapchain;
        std::vector<std::shared_ptr<VulkanImage>> images;
        std::vector<AcquireSemaphore> acquireSemaphores;
        std::vector<vk::UniqueSemaphore> presentSemaphores;
    };
    auto chain = std::make_shared<Chain>(std::move(swapchain), std::move(images), std::move(acquireSemaphores), std::move(presentSemaphores));
    images.clear();
    acquireSemaphores.clear();
    presentSemaphores.clear();
    presented.clear();
    device->retire([chain] {
        //! The views belong to images the swapchain owns, so they go before the swapchain and not through the deferred image release.
        for (const std::shared_ptr<VulkanImage>& image : chain->images) {
            image->view.reset();
            image->image = vk::Image{};
        }
        chain->images.clear();
        chain->acquireSemaphores.clear();
        chain->presentSemaphores.clear();
        chain->swapchain.reset();
    });
}

std::expected<void, Error> VulkanSwapchain::rebuild() {
    const vk::PhysicalDevice physical = device->physicalDevice;
    std::expected<vk::SurfaceCapabilitiesKHR, Error> capabilities =
        checkVulkan(physical.getSurfaceCapabilitiesKHR(*surface), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    if (!capabilities)
        return std::unexpected(std::move(capabilities.error()));
    std::uint32_t extentWidth = provider->getWidth();
    std::uint32_t extentHeight = provider->getHeight();
    if (capabilities->currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
        extentWidth = capabilities->currentExtent.width;
        extentHeight = capabilities->currentExtent.height;
    }
    extentWidth = std::clamp(extentWidth, capabilities->minImageExtent.width, capabilities->maxImageExtent.width);
    extentHeight = std::clamp(extentHeight, capabilities->minImageExtent.height, capabilities->maxImageExtent.height);
    if (extentWidth == 0 || extentHeight == 0) {
        width = 0;
        height = 0;
        stale = true;
        return {};
    }

    std::expected<std::vector<vk::SurfaceFormatKHR>, Error> formats =
        checkVulkan(physical.getSurfaceFormatsKHR(*surface), "vkGetPhysicalDeviceSurfaceFormatsKHR");
    if (!formats)
        return std::unexpected(std::move(formats.error()));
    std::expected<vk::SurfaceFormatKHR, Error> surfaceFormat = chooseFormat(*formats, format);
    if (!surfaceFormat)
        return std::unexpected(std::move(surfaceFormat.error()));
    const std::optional<ImageFormat> publicFormat = getPublicFormat(surfaceFormat->format);
    if (!publicFormat)
        return fail(ErrorCode::Unsupported, "the swapchain format has no NoorRHI image format");
    std::expected<std::vector<vk::PresentModeKHR>, Error> modes =
        checkVulkan(physical.getSurfacePresentModesKHR(*surface), "vkGetPhysicalDeviceSurfacePresentModesKHR");
    if (!modes)
        return std::unexpected(std::move(modes.error()));
    std::expected<vk::CompositeAlphaFlagBitsKHR, Error> compositeAlpha = chooseCompositeAlpha(*capabilities, transparent);
    if (!compositeAlpha)
        return std::unexpected(std::move(compositeAlpha.error()));
    vulkanFormat = surfaceFormat->format;
    format = *publicFormat;
    presentMode = choosePresentMode(*modes, requestedMode, device->fifoLatestReady);
    activeMode = getActiveMode(presentMode);

    const vk::SwapchainCreateInfoKHR createInfo =
        vk::SwapchainCreateInfoKHR()
            .setSurface(*surface)
            .setMinImageCount(chooseImageCount(*capabilities, presentMode))
            .setImageFormat(vulkanFormat)
            .setImageColorSpace(surfaceFormat->colorSpace)
            .setImageExtent({extentWidth, extentHeight})
            .setImageArrayLayers(1)
            .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc)
            .setImageSharingMode(vk::SharingMode::eExclusive)
            .setPreTransform(capabilities->currentTransform)
            .setCompositeAlpha(*compositeAlpha)
            .setPresentMode(presentMode)
            .setClipped(true)
            .setOldSwapchain(*swapchain);
    std::expected<vk::UniqueSwapchainKHR, Error> replacement = checkVulkan(device->device->createSwapchainKHRUnique(createInfo), "vkCreateSwapchainKHR");
    if (!replacement)
        return std::unexpected(std::move(replacement.error()));
    retireChain();
    swapchain = std::move(*replacement);

    std::expected<std::vector<vk::Image>, Error> chainImages = checkVulkan(device->device->getSwapchainImagesKHR(*swapchain), "vkGetSwapchainImagesKHR");
    if (!chainImages)
        return std::unexpected(std::move(chainImages.error()));
    for (const vk::Image chainImage : *chainImages) {
        auto wrapped = std::make_shared<VulkanImage>();
        wrapped->device = device;
        wrapped->image = chainImage;
        wrapped->ownsImage = false;
        wrapped->format = vulkanFormat;
        wrapped->publicFormat = format;
        wrapped->width = extentWidth;
        wrapped->height = extentHeight;
        std::expected<vk::UniqueImageView, Error> view = checkVulkan(
            device->device->createImageViewUnique({{}, chainImage, vk::ImageViewType::e2D, vulkanFormat, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}),
            "vkCreateImageView");
        if (!view)
            return std::unexpected(std::move(view.error()));
        wrapped->view = std::move(*view);
        images.push_back(std::move(wrapped));

        std::expected<vk::UniqueSemaphore, Error> acquired = checkVulkan(device->device->createSemaphoreUnique({}), "vkCreateSemaphore");
        if (!acquired)
            return std::unexpected(std::move(acquired.error()));
        acquireSemaphores.push_back({std::move(*acquired), {}});
        std::expected<vk::UniqueSemaphore, Error> presentDone = checkVulkan(device->device->createSemaphoreUnique({}), "vkCreateSemaphore");
        if (!presentDone)
            return std::unexpected(std::move(presentDone.error()));
        presentSemaphores.push_back(std::move(*presentDone));
    }
    presented.assign(images.size(), false);
    semaphoreCursor = 0;
    width = extentWidth;
    height = extentHeight;
    imageCount = static_cast<std::uint32_t>(images.size());
    stale = false;
    return {};
}

std::expected<bool, Error> VulkanSwapchain::waitForSlot(const bool block) {
    while (!inFlight.empty()) {
        if (device->isFinished(inFlight.front())) {
            inFlight.pop_front();
            continue;
        }
        if (inFlight.size() < maxFramesInFlight)
            return true;
        if (!block)
            return false;
        if (std::expected<void, Error> waited = device->wait(inFlight.front()); !waited)
            return std::unexpected(std::move(waited.error()));
    }
    return true;
}

std::expected<void, Error> VulkanSwapchain::waitUntilReady() {
    std::lock_guard lock(mutex);
    std::expected<bool, Error> ready = waitForSlot(true);
    if (!ready)
        return std::unexpected(std::move(ready.error()));
    return {};
}

std::expected<std::optional<AcquiredImage>, Error> VulkanSwapchain::acquire(const bool block) {
    std::lock_guard lock(mutex);
    std::expected<bool, Error> ready = waitForSlot(block);
    if (!ready)
        return std::unexpected(std::move(ready.error()));
    if (!*ready)
        return std::nullopt;
    if (stale || width != provider->getWidth() || height != provider->getHeight())
        if (std::expected<void, Error> rebuilt = rebuild(); !rebuilt)
            return std::unexpected(std::move(rebuilt.error()));
    if (stale)
        return std::nullopt;

    const std::uint32_t semaphoreIndex = semaphoreCursor;
    AcquireSemaphore& slot = acquireSemaphores[semaphoreIndex];
    //! A binary semaphore can be signalled again only after the submission that waited on it has run.
    if (std::expected<void, Error> waited = device->wait(slot.token); !waited)
        return std::unexpected(std::move(waited.error()));

    std::uint32_t index = 0;
    const vk::Result acquired =
        device->device->acquireNextImageKHR(*swapchain, block ? std::numeric_limits<std::uint64_t>::max() : 0, *slot.semaphore, {}, &index);
    if (acquired == vk::Result::eTimeout || acquired == vk::Result::eNotReady)
        return std::nullopt;
    if (acquired == vk::Result::eErrorOutOfDateKHR) {
        stale = true;
        return std::nullopt;
    }
    if (acquired != vk::Result::eSuccess && acquired != vk::Result::eSuboptimalKHR)
        return failVulkan(acquired, "vkAcquireNextImageKHR");
    //! A suboptimal swapchain still presents correctly, so the rebuild waits for the next acquire.
    if (acquired == vk::Result::eSuboptimalKHR)
        stale = true;
    semaphoreCursor = (semaphoreCursor + 1) % static_cast<std::uint32_t>(acquireSemaphores.size());

    std::unique_ptr<VulkanCommandList> list = device->openList(Queue::Graphics);
    if (!list->error)
        recordImageBarrier(list->pool.commands, images[index]->image, vk::ImageAspectFlagBits::eColor,
                           presented[index] ? vk::ImageLayout::ePresentSrcKHR : vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    std::expected<Token, Error> token = device->submit(*list, {}, {*slot.semaphore, {}});
    if (!token)
        return std::unexpected(std::move(token.error()));
    slot.token = *token;
    return AcquiredImage{images[index], index, semaphoreIndex};
}

std::expected<void, Error> VulkanSwapchain::present(const std::uint32_t index, const std::uint32_t semaphoreIndex) {
    std::lock_guard lock(mutex);
    if (index >= images.size() || semaphoreIndex >= acquireSemaphores.size())
        return fail(ErrorCode::InvalidArgument, "the frame belongs to an older swapchain");
    std::unique_ptr<VulkanCommandList> list = device->openList(Queue::Graphics);
    if (!list->error)
        recordImageBarrier(list->pool.commands, images[index]->image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
                           vk::ImageLayout::ePresentSrcKHR);
    const vk::Semaphore presentReady = *presentSemaphores[index];
    std::expected<Token, Error> token = device->submit(*list, {}, {{}, presentReady});
    if (!token)
        return std::unexpected(std::move(token.error()));
    inFlight.push_back(*token);
    presented[index] = true;

    const vk::SwapchainKHR handle = *swapchain;
    const vk::PresentInfoKHR presentInfo(presentReady, handle, index);
    vk::Result result = vk::Result::eSuccess;
    {
        std::lock_guard queueLock(device->mutex);
        result = device->getQueue(Queue::Graphics).queue.presentKHR(&presentInfo);
    }
    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
        stale = true;
        return {};
    }
    return checkVulkan(result, "vkQueuePresentKHR");
}

void VulkanSwapchain::setPresentMode(const PresentMode mode) {
    std::lock_guard lock(mutex);
    if (requestedMode == mode)
        return;
    requestedMode = mode;
    stale = true;
}

void VulkanSwapchain::invalidate() {
    std::lock_guard lock(mutex);
    stale = true;
}

}

namespace NoorRHI {

std::expected<void, Error> Swapchain::open(Device& device, SurfaceProvider& provider, const SwapchainDesc& desc) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    auto& owner = static_cast<Backend::VulkanDevice&>(*device.getBackend());
    if (!owner.presentation)
        return fail(ErrorCode::InvalidState, "a swapchain needs a device opened with DeviceConfig::presentation");
    const std::uintptr_t raw = provider.createSurface(reinterpret_cast<std::uintptr_t>(static_cast<VkInstance>(*owner.instance)));
    if (!raw)
        return fail(ErrorCode::InvalidState, "the SurfaceProvider did not create a surface");
    auto result = std::make_shared<Backend::VulkanSwapchain>();
    result->device = owner.shared_from_this();
    result->provider = &provider;
    result->surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(reinterpret_cast<VkSurfaceKHR>(raw)), *owner.instance);
    std::expected<vk::Bool32, Error> supported = Backend::checkVulkan(
        owner.physicalDevice.getSurfaceSupportKHR(owner.getQueue(Queue::Graphics).family, *result->surface), "vkGetPhysicalDeviceSurfaceSupportKHR");
    if (!supported)
        return std::unexpected(std::move(supported.error()));
    if (!*supported)
        return fail(ErrorCode::Unsupported, "the graphics queue cannot present to this surface");
    result->format = desc.format;
    result->requestedMode = desc.presentMode;
    result->maxFramesInFlight = std::max(desc.maxFramesInFlight, 1u);
    result->transparent = desc.transparent;
    {
        std::lock_guard lock(result->mutex);
        if (std::expected<void, Error> rebuilt = result->rebuild(); !rebuilt)
            return rebuilt;
    }
    backend = std::move(result);
    return {};
}

}
