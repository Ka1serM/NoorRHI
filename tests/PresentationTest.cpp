#include "TestWindow.hpp"

#include <noorrhi/noorrhi.hpp>
#include <SDL3/SDL.h>
#include <cstdlib>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("NoorRHI presentation survives resize, mode switches and abandoned frames",
    "[NoorRHI][window]") {
    if (!std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY"))
        SKIP("presentation test requires a desktop display");
    TestWindow window(320, 240);
    noorrhi::Device device({.enable_validation = true, .presentation = &window});
    noorrhi::Swapchain swapchain(device, window);
    REQUIRE(swapchain.present_mode() == noorrhi::PresentMode::LowLatency);
    noorrhi::Shared<unsigned> record(device);
    record.commit();
    device.synchronize();
    const auto allocation_bytes = device.memory_report().allocation_bytes;

    constexpr noorrhi::PresentMode modes[] = {noorrhi::PresentMode::Vsync,
        noorrhi::PresentMode::Immediate, noorrhi::PresentMode::LowLatency};
    for (unsigned iteration = 0; iteration < 48; ++iteration) {
        if (iteration % 8 == 7)
            swapchain.set_present_mode(modes[(iteration / 8) % 3]);
        const int width = iteration % 2 ? 320 : 480;
        const int height = iteration % 2 ? 240 : 360;
        REQUIRE(SDL_SetWindowSize(window.nativeHandle(), width, height));
        REQUIRE(SDL_SyncWindow(window.nativeHandle()));
        SDL_Event event{};
        while (window.pollEvent(event)) {}

        swapchain.wait_until_ready();
        auto frame = swapchain.begin_frame();
        if (!frame) frame = swapchain.begin_frame();
        REQUIRE(frame);
        CHECK(frame.width() == window.width());
        CHECK(frame.height() == window.height());
        record.setData(iteration + 1);
        // Uploads are rejected while a frame's command buffer is open.
        CHECK_THROWS_AS(record.commit(), noorrhi::Error);

        if (iteration % 3 == 0) {
            frame = {}; // Discard the acquire and command buffer through RAII.
            record.commit();
            frame = swapchain.begin_frame();
            REQUIRE(frame);
        }
        device.render({frame.target(), {}}, [] {});
        swapchain.present(std::move(frame));
        record.commit();
    }
    // Replaced chains are retired, not waited on; they are gone once the GPU
    // has caught up.
    device.synchronize();
    CHECK(device.memory_report().allocation_bytes == allocation_bytes);
}

TEST_CASE("NoorRHI swapchains are optional and need a presentation device", "[NoorRHI][window]") {
    if (!std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY"))
        SKIP("presentation test requires a desktop display");
    TestWindow window(320, 240);
    noorrhi::Device headless({.enable_validation = true});
    CHECK_THROWS_AS(noorrhi::Swapchain(headless, window), noorrhi::Error);
}
