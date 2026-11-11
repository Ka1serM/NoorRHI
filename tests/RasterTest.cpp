#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "Raster.h"
#include "Raster.slang.h"
#include "Test.h"
#include "Triangle.slang.h"

using namespace NoorRHI;

namespace {

constexpr std::uint32_t size = 8;
constexpr std::uint32_t black = 0xff000000u;
constexpr std::uint32_t white = 0xffffffffu;

GraphicsPipelineDesc makeDesc(const Shader& fragment) {
    return {.vertex = Raster::vertMain, .fragment = fragment, .state = {.cull = CullMode::None}};
}

Raster::Arguments makeArguments(const float4 color) {
    return {.color = color, .uvScale = {1.0f, 1.0f}, .targetSize = {static_cast<float>(size), static_cast<float>(size)}};
}

std::expected<Image, Error> openTarget(Device& device, const std::uint32_t samples = 1) {
    Image image;
    const ImageUsage usage = samples > 1 ? ImageUsage::ColorAttachment : ImageUsage::ColorAttachment | ImageUsage::Sampled;
    if (std::expected<void, Error> opened = image.open(device, {.width = size, .height = size, .usage = usage, .samples = samples}); !opened)
        return std::unexpected(std::move(opened.error()));
    return image;
}

}

TEST_CASE("a triangle draws into a color target") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    GraphicsPipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, {.vertex = Triangle::vertMain, .fragment = Triangle::fragMain, .state = {.cull = CullMode::None}}));
    std::expected<Image, Error> target = openTarget(*device);
    REQUIRE_OK(target);

    CommandList list(*device);
    list.beginRender({.color = &*target});
    list.draw(pipeline, 3);
    list.endRender();
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::expected<std::vector<std::uint32_t>, Error> texels = Test::downloadTexels(*target);
    REQUIRE_OK(texels);
    CHECK((*texels)[(size / 2) * size + size / 2] != black);
    CHECK((*texels)[0] == black);
}

TEST_CASE("clip space has Y up and counterclockwise triangles face front") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    std::expected<Image, Error> target = openTarget(*device);
    REQUIRE_OK(target);
    for (const FrontFace front : {FrontFace::Clockwise, FrontFace::CounterClockwise}) {
        GraphicsPipeline pipeline;
        REQUIRE_OK(
            pipeline.open(*device, {.vertex = Raster::cornerMain, .fragment = Raster::colorMain, .state = {.cull = CullMode::Back, .frontFace = front}}));
        CommandList list(*device);
        list.beginRender({.color = &*target});
        list.draw(pipeline, 3, 1, makeArguments({1.0f, 1.0f, 1.0f, 1.0f}));
        list.endRender();
        REQUIRE_OK(Test::execute(*device, std::move(list)));
        std::expected<std::vector<std::uint32_t>, Error> texels = Test::downloadTexels(*target);
        REQUIRE_OK(texels);
        CHECK((*texels)[(size - 1) * size] == (front == FrontFace::CounterClockwise ? white : black));
        CHECK((*texels)[0] == black);
        CHECK((*texels)[size - 1] == black);
    }
}

TEST_CASE("the scissor limits draws and clamps to the target") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    GraphicsPipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, makeDesc(Raster::colorMain)));
    std::expected<Image, Error> target = openTarget(*device);
    REQUIRE_OK(target);

    CommandList list(*device);
    list.beginRender({.color = &*target});
    list.setScissor(Rect{-4, 2, 7, 100});
    list.draw(pipeline, 3, 1, makeArguments({1.0f, 1.0f, 1.0f, 1.0f}));
    list.endRender();
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::expected<std::vector<std::uint32_t>, Error> texels = Test::downloadTexels(*target);
    REQUIRE_OK(texels);
    for (std::uint32_t y = 0; y < size; ++y)
        for (std::uint32_t x = 0; x < size; ++x)
            REQUIRE((*texels)[y * size + x] == (x < 3 && y >= 2 ? white : black));
}

TEST_CASE("stencil masks restrict later draws") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    Image depth;
    const std::expected<void, Error> opened =
        depth.open(*device, {.width = size, .height = size, .format = ImageFormat::D32FloatS8Uint, .usage = ImageUsage::DepthAttachment});
    if (!opened && opened.error().code == ErrorCode::Unsupported)
        return static_cast<void>(std::printf("skip: %s\n", opened.error().message.c_str()));
    REQUIRE_OK(opened);
    GraphicsPipelineDesc maskDesc = makeDesc(Raster::colorMain);
    maskDesc.depthFormat = ImageFormat::D32FloatS8Uint;
    maskDesc.state.colorWrite = false;
    maskDesc.state.stencil = {.compare = CompareOp::Always, .pass = StencilOp::Replace};
    GraphicsPipelineDesc testDesc = makeDesc(Raster::colorMain);
    testDesc.depthFormat = ImageFormat::D32FloatS8Uint;
    testDesc.state.stencil = {.compare = CompareOp::Equal, .pass = StencilOp::Keep};
    GraphicsPipeline mask;
    GraphicsPipeline test;
    REQUIRE_OK(mask.open(*device, maskDesc));
    REQUIRE_OK(test.open(*device, testDesc));
    std::expected<Image, Error> target = openTarget(*device);
    REQUIRE_OK(target);

    CommandList list(*device);
    list.beginRender({.color = &*target, .depth = &depth});
    list.setStencilReference(1);
    list.setScissor(Rect{0, 0, size / 2, size});
    list.draw(mask, 3, 1, makeArguments({}));
    list.setScissor(std::nullopt);
    list.draw(test, 3, 1, makeArguments({1.0f, 1.0f, 1.0f, 1.0f}));
    list.endRender();
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::expected<std::vector<std::uint32_t>, Error> texels = Test::downloadTexels(*target);
    REQUIRE_OK(texels);
    for (std::uint32_t y = 0; y < size; ++y)
        for (std::uint32_t x = 0; x < size; ++x)
            REQUIRE((*texels)[y * size + x] == (x < size / 2 ? white : black));
}

TEST_CASE("multisampled targets resolve into single-sampled images") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    GraphicsPipelineDesc desc = makeDesc(Raster::colorMain);
    desc.samples = 4;
    GraphicsPipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, desc));
    std::expected<Image, Error> multisampled = openTarget(*device, 4);
    std::expected<Image, Error> resolved = openTarget(*device);
    REQUIRE_OK(multisampled);
    REQUIRE_OK(resolved);

    CommandList list(*device);
    list.beginRender({.color = &*multisampled, .resolve = &*resolved});
    list.draw(pipeline, 3, 1, makeArguments({1.0f, 1.0f, 1.0f, 1.0f}));
    list.endRender();
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::expected<std::vector<std::uint32_t>, Error> texels = Test::downloadTexels(*resolved);
    REQUIRE_OK(texels);
    CHECK(std::ranges::all_of(*texels, [](const std::uint32_t texel) { return texel == white; }));
    CHECK(multisampled->download(std::as_writable_bytes(std::span(*texels))).error().code == ErrorCode::InvalidArgument);
}

TEST_CASE("textures sample with repeat, mirror and clamp modes") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    GraphicsPipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, makeDesc(Raster::textureMain)));
    std::vector<Image> neighbours(5);
    for (std::uint32_t index = 0; index < neighbours.size(); ++index) {
        REQUIRE_OK(neighbours[index].open(*device, {.width = 2 + index, .height = 2}));
        REQUIRE_OK(Test::uploadTexels(neighbours[index], std::vector<std::uint32_t>(std::size_t{2 + index} * 2, 0xff00ffffu)));
    }
    constexpr std::array<std::uint32_t, 4> quad{0xff0000ffu, 0xff00ff00u, 0xffff0000u, 0xffffffffu};
    Image texture;
    REQUIRE_OK(texture.open(*device, {.width = 2, .height = 2}));
    REQUIRE_OK(Test::uploadTexels(texture, quad));
    std::expected<Image, Error> target = openTarget(*device);
    REQUIRE_OK(target);

    const auto expected = [&](const AddressMode mode, const std::uint32_t x, const std::uint32_t y) {
        const auto wrap = [mode](const std::uint32_t pixel) -> std::uint32_t {
            const std::uint32_t texel = pixel / 2;
            if (mode == AddressMode::Repeat)
                return texel % 2;
            if (mode == AddressMode::MirroredRepeat)
                return texel < 2 ? texel : 3 - texel;
            return std::min(texel, 1u);
        };
        return quad[wrap(y) * 2 + wrap(x)];
    };
    for (const AddressMode mode : {AddressMode::Repeat, AddressMode::MirroredRepeat, AddressMode::ClampToEdge}) {
        Sampler sampler;
        REQUIRE_OK(sampler.open(*device, {.filter = Filter::Nearest, .addressU = mode, .addressV = mode}));
        CommandList list(*device);
        list.beginRender({.color = &*target});
        list.draw(pipeline, 3, 1,
                  Raster::Arguments{.texture = texture.getSampledHandle(),
                                    .sampler = sampler.getHandle(),
                                    .uvScale = {2.0f, 2.0f},
                                    .targetSize = {static_cast<float>(size), static_cast<float>(size)}});
        list.endRender();
        REQUIRE_OK(Test::execute(*device, std::move(list)));
        std::expected<std::vector<std::uint32_t>, Error> texels = Test::downloadTexels(*target);
        REQUIRE_OK(texels);
        for (std::uint32_t y = 0; y < size; ++y)
            for (std::uint32_t x = 0; x < size; ++x)
                REQUIRE((*texels)[y * size + x] == expected(mode, x, y));
    }
}

TEST_CASE("draws need a pipeline that matches the render target") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    GraphicsPipelineDesc desc = makeDesc(Raster::colorMain);
    desc.colorFormat = ImageFormat::Rgba16Float;
    GraphicsPipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, desc));
    std::expected<Image, Error> target = openTarget(*device);
    REQUIRE_OK(target);
    CommandList list(*device);
    list.beginRender({.color = &*target});
    list.draw(pipeline, 3);
    list.endRender();
    REQUIRE(list.getError().has_value());
    CHECK(list.getError()->code == ErrorCode::InvalidArgument);

    GraphicsPipelineDesc stencil = makeDesc(Raster::colorMain);
    stencil.state.stencil.pass = StencilOp::Replace;
    CHECK(pipeline.open(*device, stencil).error().code == ErrorCode::InvalidArgument);
}
