#include <algorithm>

#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Present/Swapchain.h"
#include "NoorRHI/WebGpu/WebGpu.h"

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

namespace NoorRHI::Backend {

namespace {

std::optional<ImageFormat> getPublicFormat(const wgpu::TextureFormat format) {
    for (std::uint8_t index = 1; index <= static_cast<std::uint8_t>(ImageFormat::Bc7Srgb); ++index)
        if (getWebGpuFormat(static_cast<ImageFormat>(index)) == format)
            return static_cast<ImageFormat>(index);
    return std::nullopt;
}

template <class T>
bool contains(const T* values, const std::size_t count, const T value) {
    return std::ranges::contains(std::span(values, count), value);
}

std::expected<wgpu::TextureFormat, Error> chooseFormat(const wgpu::SurfaceCapabilities& capabilities, const ImageFormat preferred) {
    if (preferred != ImageFormat::Undefined) {
        if (contains(capabilities.formats, capabilities.formatCount, getWebGpuFormat(preferred)))
            return getWebGpuFormat(preferred);
        return fail(ErrorCode::Unsupported, "the surface does not offer the requested swapchain format");
    }
    //! The surface lists its preferred format first, and a browser copies the canvas on every frame in any other format.
    for (const wgpu::TextureFormat format : std::span(capabilities.formats, capabilities.formatCount))
        if (format == wgpu::TextureFormat::BGRA8Unorm || format == wgpu::TextureFormat::RGBA8Unorm)
            return format;
    return fail(ErrorCode::Unsupported, "the surface offers no 8-bit RGBA or BGRA format");
}

wgpu::PresentMode choosePresentMode(const wgpu::SurfaceCapabilities& capabilities, const PresentMode policy) {
    using Mode = wgpu::PresentMode;
    static constexpr std::array lowLatency{Mode::Mailbox, Mode::Fifo};
    static constexpr std::array vsync{Mode::Fifo};
    static constexpr std::array immediate{Mode::Immediate, Mode::Mailbox, Mode::Fifo};
    const std::span<const Mode> order = policy == PresentMode::Immediate ? std::span<const Mode>(immediate)
                                        : policy == PresentMode::Vsync   ? std::span<const Mode>(vsync)
                                                                         : std::span<const Mode>(lowLatency);
    for (const Mode mode : order)
        if (contains(capabilities.presentModes, capabilities.presentModeCount, mode))
            return mode;
    //! FIFO is the only present mode every surface supports.
    return Mode::Fifo;
}

ActivePresentMode getActiveMode(const wgpu::PresentMode mode) {
    switch (mode) {
    case wgpu::PresentMode::Immediate:
        return ActivePresentMode::Immediate;
    case wgpu::PresentMode::Mailbox:
        return ActivePresentMode::Mailbox;
    default:
        return ActivePresentMode::Fifo;
    }
}

wgpu::CompositeAlphaMode chooseAlphaMode(const wgpu::SurfaceCapabilities& capabilities, const bool transparent) {
    const wgpu::CompositeAlphaMode wanted = transparent ? wgpu::CompositeAlphaMode::Premultiplied : wgpu::CompositeAlphaMode::Opaque;
    return contains(capabilities.alphaModes, capabilities.alphaModeCount, wanted) ? wanted : wgpu::CompositeAlphaMode::Auto;
}

std::expected<wgpu::Surface, Error> createSurface(const WebGpuDevice& device, const SurfaceSource& source) {
    wgpu::SurfaceDescriptor descriptor;
#if defined(__EMSCRIPTEN__)
    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvas;
    canvas.selector = wgpu::StringView(source.canvasSelector.data(), source.canvasSelector.size());
    if (source.kind != SurfaceSource::Kind::HtmlCanvas)
        return fail(ErrorCode::Unsupported, "a WebGPU swapchain on the web needs an HTML canvas surface source");
    descriptor.nextInChain = &canvas;
#else
    wgpu::SurfaceSourceXlibWindow xlib;
    xlib.display = reinterpret_cast<void*>(source.display);
    xlib.window = source.window;
    wgpu::SurfaceSourceWaylandSurface wayland;
    wayland.display = reinterpret_cast<void*>(source.display);
    wayland.surface = reinterpret_cast<void*>(source.window);
    wgpu::SurfaceSourceWindowsHWND windows;
    windows.hinstance = reinterpret_cast<void*>(source.display);
    windows.hwnd = reinterpret_cast<void*>(source.window);
    wgpu::SurfaceSourceMetalLayer metal;
    metal.layer = reinterpret_cast<void*>(source.window);
    switch (source.kind) {
    case SurfaceSource::Kind::Xlib:
        descriptor.nextInChain = &xlib;
        break;
    case SurfaceSource::Kind::Wayland:
        descriptor.nextInChain = &wayland;
        break;
    case SurfaceSource::Kind::Win32:
        descriptor.nextInChain = &windows;
        break;
    case SurfaceSource::Kind::Metal:
        descriptor.nextInChain = &metal;
        break;
    default:
        return fail(ErrorCode::Unsupported, "the SurfaceProvider gives no native window for a WebGPU swapchain");
    }
#endif
    wgpu::Surface surface = device.instance.CreateSurface(&descriptor);
    if (!surface)
        return fail(ErrorCode::InvalidState, "WebGPU could not create a surface for the window");
    return surface;
}

}

std::expected<void, Error> WebGpuSwapchain::configure() {
    const std::uint32_t extentWidth = provider->getWidth();
    const std::uint32_t extentHeight = provider->getHeight();
    if (extentWidth == 0 || extentHeight == 0) {
        width = 0;
        height = 0;
        stale = true;
        return {};
    }
    wgpu::SurfaceCapabilities capabilities;
    if (surface.GetCapabilities(device->adapter, &capabilities) != wgpu::Status::Success)
        return fail(ErrorCode::InvalidState, "the WebGPU surface reports no capabilities");
    std::expected<wgpu::TextureFormat, Error> chosen = chooseFormat(capabilities, format);
    if (!chosen)
        return std::unexpected(std::move(chosen.error()));
    const std::optional<ImageFormat> publicFormat = getPublicFormat(*chosen);
    if (!publicFormat)
        return fail(ErrorCode::Unsupported, "the swapchain format has no NoorRHI image format");
    const wgpu::PresentMode presentMode = choosePresentMode(capabilities, requestedMode);
    nativeUsage = wgpu::TextureUsage::RenderAttachment | (capabilities.usages & (wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst));
    const wgpu::SurfaceConfiguration configuration{.device = device->device,
                                                   .format = *chosen,
                                                   .usage = nativeUsage,
                                                   .width = extentWidth,
                                                   .height = extentHeight,
                                                   .alphaMode = chooseAlphaMode(capabilities, transparent),
                                                   .presentMode = presentMode};
#if defined(__EMSCRIPTEN__)
    //! Waiting on an error scope yields to the browser, which then shows the cleared canvas. Errors reach present through takeError.
    surface.Configure(&configuration);
#else
    {
        std::lock_guard lock(device->mutex);
        device->pushErrorScope();
        surface.Configure(&configuration);
        if (std::expected<void, Error> configured = device->popErrorScope("configuring the WebGPU surface"); !configured)
            return configured;
    }
#endif
    nativeFormat = *chosen;
    format = *publicFormat;
    activeMode = getActiveMode(presentMode);
    width = extentWidth;
    height = extentHeight;
    imageCount = presentMode == wgpu::PresentMode::Fifo ? 2 : 3;
    stale = false;
    return {};
}

std::expected<bool, Error> WebGpuSwapchain::waitForSlot(const bool block) {
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

std::expected<void, Error> WebGpuSwapchain::waitUntilReady() {
    std::lock_guard lock(mutex);
    std::expected<bool, Error> ready = waitForSlot(true);
    if (!ready)
        return std::unexpected(std::move(ready.error()));
    return {};
}

std::expected<std::optional<AcquiredImage>, Error> WebGpuSwapchain::acquire(const bool block) {
    std::lock_guard lock(mutex);
    std::expected<bool, Error> ready = waitForSlot(block);
    if (!ready)
        return std::unexpected(std::move(ready.error()));
    if (!*ready)
        return std::nullopt;
    if (stale || width != provider->getWidth() || height != provider->getHeight())
        if (std::expected<void, Error> configured = configure(); !configured)
            return std::unexpected(std::move(configured.error()));
    if (stale)
        return std::nullopt;

    wgpu::SurfaceTexture current;
    surface.GetCurrentTexture(&current);
    switch (current.status) {
    case wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal:
        break;
    //! A suboptimal surface still presents correctly, so the reconfigure waits for the next acquire.
    case wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal:
        stale = true;
        break;
    case wgpu::SurfaceGetCurrentTextureStatus::Timeout:
        return std::nullopt;
    case wgpu::SurfaceGetCurrentTextureStatus::Outdated:
    case wgpu::SurfaceGetCurrentTextureStatus::Lost:
        stale = true;
        return std::nullopt;
    default:
        return fail(ErrorCode::DeviceLost, "the WebGPU surface gives no texture");
    }
    auto wrapped = std::make_shared<WebGpuImage>();
    wrapped->device = device;
    wrapped->texture = current.texture;
    wrapped->nativeFormat = nativeFormat;
    wrapped->format = format;
    wrapped->width = width;
    wrapped->height = height;
    acquired = true;
    return AcquiredImage{std::move(wrapped), ++generation, 0};
}

std::expected<void, Error> WebGpuSwapchain::present(const std::uint32_t index, std::uint32_t) {
    std::lock_guard lock(mutex);
    if (!acquired || index != generation)
        return fail(ErrorCode::InvalidArgument, "the frame belongs to an older swapchain");
    acquired = false;
    std::expected<Token, Error> token;
    {
        std::lock_guard deviceLock(device->mutex);
        token = device->submitCommands(device->device.CreateCommandEncoder().Finish());
    }
    if (!token)
        return std::unexpected(std::move(token.error()));
    inFlight.push_back(*token);
#if defined(__EMSCRIPTEN__)
    //! The browser shows the canvas only after control returns to the event loop.
    emscripten_sleep(0);
#else
    if (surface.Present() != wgpu::Status::Success)
        stale = true;
#endif
    return device->takeError();
}

void WebGpuSwapchain::setPresentMode(const PresentMode mode) {
    std::lock_guard lock(mutex);
    if (requestedMode == mode)
        return;
    requestedMode = mode;
    stale = true;
}

void WebGpuSwapchain::invalidate() {
    std::lock_guard lock(mutex);
    stale = true;
}

}

namespace NoorRHI {

std::expected<void, Error> Swapchain::open(Device& device, SurfaceProvider& provider, const SwapchainDesc& desc) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    auto& owner = static_cast<Backend::WebGpuDevice&>(*device.getBackend());
    if (!owner.presentation)
        return fail(ErrorCode::InvalidState, "a swapchain needs a device opened with DeviceConfig::presentation");
    std::expected<wgpu::Surface, Error> surface = Backend::createSurface(owner, provider.getSurfaceSource());
    if (!surface)
        return std::unexpected(std::move(surface.error()));
    auto result = std::make_shared<Backend::WebGpuSwapchain>();
    result->device = owner.shared_from_this();
    result->provider = &provider;
    result->surface = std::move(*surface);
    result->format = desc.format;
    result->requestedMode = desc.presentMode;
    result->maxFramesInFlight = std::max(desc.maxFramesInFlight, 1u);
    result->transparent = desc.transparent;
    {
        std::lock_guard lock(result->mutex);
        if (std::expected<void, Error> configured = result->configure(); !configured)
            return configured;
    }
    backend = std::move(result);
    return {};
}

}
