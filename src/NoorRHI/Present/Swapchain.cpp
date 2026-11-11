#include "NoorRHI/Present/Swapchain.h"

#include "NoorRHI/Core/Backend.h"

namespace NoorRHI {

std::expected<void, Error> Swapchain::wait() {
    if (!backend)
        return fail(ErrorCode::InvalidState, "the swapchain is not open");
    return backend->waitUntilReady();
}

std::expected<Frame, Error> Swapchain::acquire() {
    return acquire(true);
}

std::expected<Frame, Error> Swapchain::tryAcquire() {
    return acquire(false);
}

std::expected<Frame, Error> Swapchain::acquire(const bool block) {
    if (!backend)
        return fail(ErrorCode::InvalidState, "the swapchain is not open");
    std::expected<std::optional<Backend::AcquiredImage>, Error> acquired = backend->acquire(block);
    if (!acquired)
        return std::unexpected(std::move(acquired.error()));
    Frame frame;
    if (!*acquired)
        return frame;
    frame.image.backend = std::move((*acquired)->image);
    frame.image.width = backend->width;
    frame.image.height = backend->height;
    frame.image.mipLevels = 1;
    frame.image.format = backend->format;
    frame.image.usage = ImageUsage::ColorAttachment;
    frame.index = (*acquired)->index;
    frame.semaphoreIndex = (*acquired)->semaphoreIndex;
    return frame;
}

std::expected<void, Error> Swapchain::present(Frame&& frame) {
    if (!backend || !frame)
        return fail(ErrorCode::InvalidArgument, "present needs an open swapchain and an acquired frame");
    const Frame consumed = std::move(frame);
    return backend->present(consumed.index, consumed.semaphoreIndex);
}

void Swapchain::setPresentMode(const PresentMode mode) {
    if (backend)
        backend->setPresentMode(mode);
}

PresentMode Swapchain::getPresentMode() const {
    return backend ? backend->requestedMode : PresentMode::LowLatency;
}

ActivePresentMode Swapchain::getActivePresentMode() const {
    return backend ? backend->activeMode : ActivePresentMode::Fifo;
}

ImageFormat Swapchain::getFormat() const {
    return backend ? backend->format : ImageFormat::Undefined;
}

std::uint32_t Swapchain::getWidth() const {
    return backend ? backend->width : 0;
}

std::uint32_t Swapchain::getHeight() const {
    return backend ? backend->height : 0;
}

std::uint32_t Swapchain::getImageCount() const {
    return backend ? backend->imageCount : 0;
}

void Swapchain::invalidate() {
    if (backend)
        backend->invalidate();
}

}
