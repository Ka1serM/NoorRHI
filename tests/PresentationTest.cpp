#include <array>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <utility>

#include "Test.h"
#include "TestWindow.h"

using namespace NoorRHI;

namespace {

bool hasDisplay() {
    return std::getenv("DISPLAY") || std::getenv("WAYLAND_DISPLAY");
}

std::expected<Frame, Error> acquireFrame(Swapchain& swapchain) {
    for (int attempt = 0; attempt < 4; ++attempt) {
        if (std::expected<void, Error> ready = swapchain.wait(); !ready)
            return std::unexpected(std::move(ready.error()));
        std::expected<Frame, Error> frame = swapchain.acquire();
        if (!frame || *frame)
            return frame;
    }
    return fail(ErrorCode::InvalidState, "the swapchain gives no image after a rebuild");
}

}

TEST_CASE("presentation survives resizes and present mode changes") {
    if (!hasDisplay())
        return;
    TestWindow window;
    REQUIRE_OK(window.open(320, 240));
    Device device;
    REQUIRE_OK(device.open({.enableValidation = true, .applicationName = "NoorRHI tests", .presentation = &window}));
    Swapchain swapchain;
    REQUIRE_OK(swapchain.open(device, window));
    CHECK(swapchain.getPresentMode() == PresentMode::LowLatency);

    constexpr std::array modes{PresentMode::Vsync, PresentMode::Immediate, PresentMode::LowLatency};
    for (std::uint32_t iteration = 0; iteration < 48; ++iteration) {
        if (iteration % 8 == 7)
            swapchain.setPresentMode(modes[(iteration / 8) % modes.size()]);
        REQUIRE(window.resize(iteration % 2 ? 320 : 480, iteration % 2 ? 240 : 360));

        std::expected<Frame, Error> frame = acquireFrame(swapchain);
        REQUIRE_OK(frame);
        CHECK(frame->getImage().getWidth() == window.getWidth());
        CHECK(frame->getImage().getHeight() == window.getHeight());

        CommandList list(device);
        list.beginRender({.color = &frame->getImage(), .clearColor = {0.0f, 0.0f, static_cast<float>(iteration % 2), 1.0f}});
        list.endRender();
        REQUIRE_OK(device.submit(std::move(list)));
        REQUIRE_OK(swapchain.present(std::move(*frame)));
    }
    REQUIRE_OK(device.waitIdle());
}

TEST_CASE("a swapchain needs a device opened for presentation") {
    if (!hasDisplay())
        return;
    TestWindow window;
    REQUIRE_OK(window.open(320, 240));
    Device* device = Test::getDevice();
    REQUIRE(device);
    Swapchain swapchain;
    CHECK(swapchain.open(*device, window).error().code == ErrorCode::InvalidState);
}
