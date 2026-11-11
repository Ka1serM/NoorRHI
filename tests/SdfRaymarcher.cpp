#include <cmath>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <utility>

#include <emscripten/emscripten.h>
#include <emscripten/html5.h>

#include "NoorRHI/NoorRHI.h"
#include "Sdf.h"
#include "Sdf.slang.h"

using namespace NoorRHI;

namespace {

EM_ASYNC_JS(double, waitForAnimationFrame, (), { return await new Promise(requestAnimationFrame); });

class Canvas final : public SurfaceProvider {
public:
    void measure() {
        double cssWidth = 0.0;
        double cssHeight = 0.0;
        emscripten_get_element_css_size(selector, &cssWidth, &cssHeight);
        const double ratio = emscripten_get_device_pixel_ratio();
        width = static_cast<std::uint32_t>(std::lround(cssWidth * ratio));
        height = static_cast<std::uint32_t>(std::lround(cssHeight * ratio));
    }

    std::uint32_t getWidth() const override { return width; }
    std::uint32_t getHeight() const override { return height; }

private:
    static constexpr const char* selector = "#canvas";

    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

int report(const Error& error) {
    std::fprintf(stderr, "%s\n", error.message.c_str());
    return 1;
}

}

int main() {
    Canvas canvas;
    canvas.measure();
    Device device;
    if (std::expected<void, Error> opened = device.open({.applicationName = "NoorRHI SDF raymarcher", .presentation = &canvas}); !opened)
        return report(opened.error());
    Swapchain swapchain;
    if (std::expected<void, Error> opened = swapchain.open(device, canvas, {.presentMode = PresentMode::Vsync}); !opened)
        return report(opened.error());
    GraphicsPipeline pipeline;
    const GraphicsPipelineDesc desc{
        .vertex = Sdf::vertMain, .fragment = Sdf::fragMain, .state = {.cull = CullMode::None}, .colorFormat = swapchain.getFormat()};
    if (std::expected<void, Error> opened = pipeline.open(device, desc); !opened)
        return report(opened.error());

    const double start = waitForAnimationFrame();
    while (true) {
        const double now = waitForAnimationFrame();
        canvas.measure();
        std::expected<Frame, Error> frame = swapchain.acquire();
        if (!frame)
            return report(frame.error());
        if (!*frame)
            continue;
        const Image& image = frame->getImage();
        const Sdf::Arguments arguments{.resolution = {static_cast<float>(image.getWidth()), static_cast<float>(image.getHeight())},
                                       .time = static_cast<float>((now - start) / 1000.0)};
        CommandList list(device);
        list.beginRender({.color = &image});
        list.draw(pipeline, 3, 1, arguments);
        list.endRender();
        if (std::expected<Token, Error> token = device.submit(std::move(list)); !token)
            return report(token.error());
        if (std::expected<void, Error> presented = swapchain.present(std::move(*frame)); !presented)
            return report(presented.error());
    }
}
