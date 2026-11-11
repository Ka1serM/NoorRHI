#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <string_view>
#include <vector>

#include "NoorRHI/Core/Error.h"
#include "NoorRHI/Resources/Image.h"

namespace NoorRHI {

class Device;
namespace Backend {
struct Swapchain;
}

struct SurfaceSource {
    enum class Kind : std::uint8_t {
        None,
        Xlib,
        Wayland,
        Win32,
        Metal,
        HtmlCanvas,
    };
    Kind kind = Kind::None;
    std::uintptr_t display = 0;
    std::uintptr_t window = 0;
    std::string_view canvasSelector = "#canvas";
};

class SurfaceProvider {
public:
    virtual ~SurfaceProvider() = default;

    virtual std::uintptr_t getInstanceProcAddress() const { return 0; }
    virtual std::vector<const char*> getInstanceExtensions() const { return {}; }
    virtual std::uintptr_t createSurface(std::uintptr_t) const { return 0; }
    virtual SurfaceSource getSurfaceSource() const {
#if defined(__EMSCRIPTEN__)
        return {.kind = SurfaceSource::Kind::HtmlCanvas};
#else
        return {};
#endif
    }
    virtual std::uint32_t getWidth() const = 0;
    virtual std::uint32_t getHeight() const = 0;
};

enum class PresentMode : std::uint8_t {
    LowLatency,
    Vsync,
    Immediate,
};

enum class ActivePresentMode : std::uint8_t { Immediate, Mailbox, FifoLatestReady, Fifo };

struct SwapchainDesc {
    ImageFormat format = ImageFormat::Undefined;
    PresentMode presentMode = PresentMode::LowLatency;
    std::uint32_t maxFramesInFlight = 1;
    bool transparent = false;
};

class Frame {
public:
    const Image& getImage() const { return image; }
    std::uint32_t getIndex() const { return index; }
    explicit operator bool() const { return static_cast<bool>(image); }

private:
    friend class Swapchain;

    Image image;
    std::uint32_t index = 0;
    std::uint32_t semaphoreIndex = 0;
};

class Swapchain {
public:
    [[nodiscard]] std::expected<void, Error> open(Device& device, SurfaceProvider& provider, const SwapchainDesc& desc = {});

    [[nodiscard]] std::expected<void, Error> wait();
    [[nodiscard]] std::expected<Frame, Error> acquire();
    [[nodiscard]] std::expected<Frame, Error> tryAcquire();
    [[nodiscard]] std::expected<void, Error> present(Frame&& frame);

    void setPresentMode(PresentMode mode);
    PresentMode getPresentMode() const;
    ActivePresentMode getActivePresentMode() const;
    ImageFormat getFormat() const;
    std::uint32_t getWidth() const;
    std::uint32_t getHeight() const;
    std::uint32_t getImageCount() const;
    void invalidate();
    explicit operator bool() const { return backend != nullptr; }

private:
    std::expected<Frame, Error> acquire(bool block);

    std::shared_ptr<Backend::Swapchain> backend;
};

}
