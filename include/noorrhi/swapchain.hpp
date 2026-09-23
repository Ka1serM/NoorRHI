#pragma once

#include "image.hpp"
#include "types.hpp"

#include <cstdint>
#include <memory>

namespace noorrhi {

class Device;
class SurfaceProvider;
class Swapchain;
namespace detail { class SwapchainImpl; class DeviceImpl; }
namespace interop { std::uintptr_t command_buffer(const class Frame&); }

// How finished frames reach the display. Each policy names an intent; the
// swapchain picks the best mode the surface offers for it and reports the
// result through Swapchain::active_present_mode().
enum class PresentMode {
    // Tear-free with the newest finished frame shown at each refresh:
    // MAILBOX, then FIFO_LATEST_READY, then FIFO.
    LowLatency,
    // Every frame is shown, one per refresh. The CPU blocks when it gets ahead
    // of the display, which is the lowest-power choice: FIFO.
    Vsync,
    // Frames are shown as soon as they are finished and may tear: IMMEDIATE,
    // then the LowLatency order.
    Immediate,
};

// The mode the surface actually runs in.
enum class ActivePresentMode {
    Immediate,
    Mailbox,
    FifoLatestReady,
    Fifo,
};

struct SwapchainDesc {
    // Honoured when the surface supports it, otherwise quietly replaced; read
    // the result back with Swapchain::format().
    ImageFormat format = ImageFormat::Auto;
    PresentMode present_mode = PresentMode::LowLatency;
    // Frames the CPU may submit before the oldest has finished on the GPU.
    // 1 gives the lowest input latency: every frame starts from input sampled
    // after the previous frame's GPU work completed. Raise it only when CPU
    // frame preparation, not the GPU, is the bottleneck.
    std::uint32_t max_frames_in_flight = 1;
    // Request compositor-owned alpha for a transparent native window. If the
    // surface does not support a transparent composite mode, the swapchain
    // falls back to opaque presentation.
    bool transparent = false;
};

// One acquired presentation image, plus the command buffer that every
// operation recorded between Swapchain::begin_frame and Swapchain::present
// goes into. Dispatches, traces and render scopes issued while a frame is open
// are batched into that one submission instead of being submitted
// individually.
//
// A falsy Frame means no image was acquired (the window is minimised, or the
// chain was just replaced); skip the frame and try again. Frames are
// move-only, and dropping one without present discards its recorded work.
class Frame {
public:
    Frame() = default;
    Frame(Frame&&) noexcept;
    Frame& operator=(Frame&&) noexcept;
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;
    ~Frame();

    // The acquired image, usable as a RenderTarget colour attachment.
    ImageHandle target() const noexcept;
    std::uint32_t index() const noexcept;
    std::uint32_t width() const noexcept;
    std::uint32_t height() const noexcept;
    ImageFormat format() const noexcept;

    explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
    friend class Swapchain;
    friend class detail::DeviceImpl;
    friend std::uintptr_t interop::command_buffer(const Frame&);
    struct State;
    explicit Frame(std::shared_ptr<State> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<State> impl_;
};

// An optional presentation chain for one window. A Device renders into images
// and never needs one; create a Swapchain only to show results in a window.
// Several swapchains may share a Device, one per window.
//
// The device must have been created with DeviceConfig::presentation, and the
// SurfaceProvider must outlive the Swapchain. Swapchain images are ordinary
// ImageHandles: they go into a RenderTarget and are copy destinations, and the
// layout transitions presentation needs are the library's business.
//
// Resizing never stalls the GPU. The chain is rebuilt from the previous one
// when the provider's size changes or the surface reports it is out of date,
// and the old chain is released once the work that used it has finished.
class Swapchain {
public:
    Swapchain() = default;
    Swapchain(Device& device, SurfaceProvider& surface, const SwapchainDesc& desc = {});
    Swapchain(Swapchain&&) noexcept = default;
    Swapchain& operator=(Swapchain&&) noexcept = default;
    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    ~Swapchain() = default;

    // Blocks until a new frame may start without exceeding
    // SwapchainDesc::max_frames_in_flight. begin_frame() enforces the same
    // limit, but calling this first, before polling input and updating the
    // application, is what keeps input latency down: the wait happens before
    // input is sampled instead of after.
    void wait_until_ready();

    // Acquires the next image and opens the frame's recording scope.
    Frame begin_frame();
    // Closes the frame, submits everything recorded into it, and queues the
    // image for presentation. The CPU does not wait for the GPU here.
    void present(Frame&& frame);

    // Takes effect on the next begin_frame, by rebuilding the chain.
    void set_present_mode(PresentMode mode);
    PresentMode present_mode() const noexcept;
    ActivePresentMode active_present_mode() const noexcept;

    ImageFormat format() const noexcept;
    std::uint32_t width() const noexcept;
    std::uint32_t height() const noexcept;
    std::uint32_t image_count() const noexcept;

    // Force a rebuild before the next acquire. A size mismatch is detected on
    // its own, but a window manager can change a surface without changing the
    // reported size, and a resize event is a cheap way to stay ahead of it.
    void invalidate();

    explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
    std::shared_ptr<detail::SwapchainImpl> impl_;
};

} // namespace noorrhi
