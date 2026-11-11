#include <array>
#include <cstdint>
#include <vector>

#include "Scene.h"
#include "Scene.slang.h"
#include "Test.h"

using namespace NoorRHI;

namespace {

constexpr std::uint32_t grid = 32;
constexpr std::uint32_t instanceCount = grid * grid;
constexpr std::uint32_t size = grid * 2;
constexpr std::uint32_t materialCount = 64;
constexpr std::uint32_t black = 0xff000000u;

std::uint32_t getMaterialColor(const std::uint32_t material) {
    return 0xff000000u | (material * 37u & 0xffu) | (material * 91u & 0xffu) << 8 | (255u - material * 3u) << 16;
}

std::uint32_t getMaterial(const std::uint32_t instance) {
    return instance * 7u % materialCount;
}

bool isVisible(const std::uint32_t instance) {
    return instance % 3u != 0;
}

struct SceneResources {
    std::vector<Image> textures;
    Buffer<::Scene::Material> materials;
    Buffer<::Scene::Instance> instances;
    Buffer<float2> corners;
    Buffer<std::uint32_t> indices;
    Sampler sampler;
    GraphicsPipeline pipeline;
    Image target;
};

std::expected<void, Error> openScene(Device& device, SceneResources& scene) {
    std::vector<::Scene::Material> materials;
    materials.reserve(materialCount);
    for (std::uint32_t material = 0; material < materialCount; ++material) {
        Image& texture = scene.textures.emplace_back();
        const std::uint32_t width = 1 + material % 8;
        const std::uint32_t height = 1 + material / 8;
        if (std::expected<void, Error> opened = texture.open(device, {.width = width, .height = height, .usage = ImageUsage::Sampled}); !opened)
            return opened;
        const std::vector<std::uint32_t> texels(std::size_t{width} * height, getMaterialColor(material));
        if (std::expected<void, Error> uploaded = Test::uploadTexels(texture, texels); !uploaded)
            return uploaded;
        materials.push_back({texture.getSampledHandle(), 0});
    }
    std::vector<::Scene::Instance> instances;
    instances.reserve(instanceCount);
    for (std::uint32_t instance = 0; instance < instanceCount; ++instance)
        instances.push_back({{instance % grid, instance / grid}, getMaterial(instance), isVisible(instance)});
    const std::array<float2, 14> corners{float2{9, 9}, {9, 9}, {9, 9}, {9, 9}, {0, 0}, {1, 0}, {0, 1}, {1, 1}, {0, 0}, {1, 0}, {0, 1}, {0, 1}, {1, 0}, {1, 1}};
    const std::array<std::uint32_t, 12> indices{0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 3};
    if (std::expected<void, Error> opened = scene.materials.open(device, materials.size()); !opened)
        return opened;
    if (std::expected<void, Error> opened = scene.instances.open(device, instances.size()); !opened)
        return opened;
    if (std::expected<void, Error> opened = scene.corners.open(device, corners.size()); !opened)
        return opened;
    if (std::expected<void, Error> opened = scene.indices.open(device, indices.size()); !opened)
        return opened;
    if (std::expected<void, Error> uploaded = scene.materials.upload(materials); !uploaded)
        return uploaded;
    if (std::expected<void, Error> uploaded = scene.instances.upload(instances); !uploaded)
        return uploaded;
    if (std::expected<void, Error> uploaded = scene.corners.upload(corners); !uploaded)
        return uploaded;
    if (std::expected<void, Error> uploaded = scene.indices.upload(indices); !uploaded)
        return uploaded;
    if (std::expected<void, Error> opened = scene.sampler.open(device, {.filter = Filter::Nearest}); !opened)
        return opened;
    if (std::expected<void, Error> opened =
            scene.pipeline.openAsync(device, {.vertex = Scene::vertMain, .fragment = Scene::fragMain, .state = {.cull = CullMode::None}});
        !opened)
        return opened;
    if (std::expected<void, Error> compiled = scene.pipeline.wait(); !compiled)
        return compiled;
    return scene.target.open(device, {.width = size, .height = size, .usage = ImageUsage::ColorAttachment | ImageUsage::Sampled});
}

::Scene::DrawArguments makeArguments(const SceneResources& scene, const Ptr<float2> corners) {
    return {corners, scene.instances.getPtr(), scene.materials.getPtr(), scene.sampler.getHandle(), 2.0f / grid};
}

std::uint32_t countMismatches(const std::span<const std::uint32_t> texels, const bool culled) {
    std::uint32_t mismatches = 0;
    for (std::uint32_t y = 0; y < size; ++y)
        for (std::uint32_t x = 0; x < size; ++x) {
            const std::uint32_t instance = y / 2 * grid + x / 2;
            const std::uint32_t expected = culled && !isVisible(instance) ? black : getMaterialColor(getMaterial(instance));
            mismatches += texels[y * size + x] != expected;
        }
    return mismatches;
}

}

TEST_CASE("instanced indexed draws read many materials through handles") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    SceneResources scene;
    REQUIRE_OK(openScene(*device, scene));
    CHECK(scene.pipeline.isReady());

    CommandList list(*device);
    list.beginRender({.color = &scene.target});
    list.draw(scene.pipeline, scene.indices.getPtr() + 6, 6, instanceCount, makeArguments(scene, scene.corners.getPtr() + 4));
    list.endRender();
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::expected<std::vector<std::uint32_t>, Error> texels = Test::downloadTexels(scene.target);
    REQUIRE_OK(texels);
    CHECK(countMismatches(*texels, false) == 0);
}

TEST_CASE("indexed indirect draws take their first index, vertex offset and first instance from the commands") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    SceneResources scene;
    REQUIRE_OK(openScene(*device, scene));
    std::vector<DrawIndexedArgs> draws;
    draws.reserve(instanceCount);
    for (std::uint32_t instance = 0; instance < instanceCount; ++instance)
        draws.push_back({6, 1, 6, 4, instance});
    Buffer<DrawIndexedArgs> commands;
    REQUIRE_OK(commands.open(*device, draws.size()));
    REQUIRE_OK(commands.upload(draws));

    CommandList list(*device);
    list.beginRender({.color = &scene.target});
    list.drawIndirect(scene.pipeline, scene.indices.getPtr(), commands.getPtr(), instanceCount, makeArguments(scene, scene.corners.getPtr()));
    list.endRender();
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::expected<std::vector<std::uint32_t>, Error> texels = Test::downloadTexels(scene.target);
    REQUIRE_OK(texels);
    CHECK(countMismatches(*texels, false) == 0);
}

TEST_CASE("GPU culling on the async queue feeds draws with a count") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    SceneResources scene;
    REQUIRE_OK(openScene(*device, scene));
    ComputePipeline cull;
    REQUIRE_OK(cull.openAsync(*device, Scene::cullMain));
    REQUIRE_OK(cull.wait());
    Buffer<DrawArgs> draws;
    Buffer<DrawIndexedArgs> indexedDraws;
    Buffer<Atomic<std::uint32_t>> count;
    REQUIRE_OK(draws.open(*device, instanceCount));
    REQUIRE_OK(indexedDraws.open(*device, instanceCount));
    REQUIRE_OK(count.open(*device, 1));
    Readback<Atomic<std::uint32_t>> visible;
    REQUIRE_OK(visible.open(*device, 1));
    std::uint32_t expectedVisible = 0;
    for (std::uint32_t instance = 0; instance < instanceCount; ++instance)
        expectedVisible += isVisible(instance);

    for (const bool indexed : {true, false}) {
        REQUIRE_OK(count.upload(std::array{Atomic<std::uint32_t>{0}}));
        CommandList compute(*device, Queue::Async);
        compute.dispatch(cull, {instanceCount},
                         ::Scene::CullArguments{scene.instances.getPtr(), draws.getPtr(), indexedDraws.getPtr(), count.getPtr(), instanceCount, indexed});
        compute.download(count, visible);
        CHECK(!visible.isReady());
        std::expected<Token, Error> culled = device->submit(std::move(compute));
        REQUIRE_OK(culled);

        CommandList list(*device);
        list.beginRender({.color = &scene.target});
        if (indexed)
            list.drawIndirect(scene.pipeline, scene.indices.getPtr(), indexedDraws.getPtr(), count.getPtr(), instanceCount,
                              makeArguments(scene, scene.corners.getPtr()));
        else
            list.drawIndirect(scene.pipeline, draws.getPtr(), count.getPtr(), instanceCount, makeArguments(scene, scene.corners.getPtr()));
        list.endRender();
        std::expected<Token, Error> drawn = device->submit(std::move(list), std::array{*culled});
        REQUIRE_OK(drawn);
        REQUIRE_OK(device->wait(*drawn));
        std::expected<std::span<const Atomic<std::uint32_t>>, Error> read = visible.read();
        REQUIRE_OK(read);
        CHECK((*read)[0].value == expectedVisible);
        CHECK(visible.isReady());
        std::expected<std::vector<std::uint32_t>, Error> texels = Test::downloadTexels(scene.target);
        REQUIRE_OK(texels);
        CHECK(countMismatches(*texels, true) == 0);
    }
}

TEST_CASE("a draw with a count draws at most maxDraws commands") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    SceneResources scene;
    REQUIRE_OK(openScene(*device, scene));
    ComputePipeline cull;
    REQUIRE_OK(cull.open(*device, Scene::cullMain));
    Buffer<DrawIndexedArgs> indexedDraws;
    Buffer<Atomic<std::uint32_t>> count;
    REQUIRE_OK(indexedDraws.open(*device, instanceCount));
    REQUIRE_OK(count.open(*device, 1));
    REQUIRE_OK(count.upload(std::array{Atomic<std::uint32_t>{0}}));
    constexpr std::uint32_t maxDraws = 10;

    CommandList list(*device);
    list.dispatch(cull, {instanceCount}, ::Scene::CullArguments{scene.instances.getPtr(), {}, indexedDraws.getPtr(), count.getPtr(), instanceCount, 1});
    list.beginRender({.color = &scene.target});
    list.drawIndirect(scene.pipeline, scene.indices.getPtr(), indexedDraws.getPtr(), count.getPtr(), maxDraws, makeArguments(scene, scene.corners.getPtr()));
    list.endRender();
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::expected<std::vector<std::uint32_t>, Error> texels = Test::downloadTexels(scene.target);
    REQUIRE_OK(texels);
    std::uint32_t covered = 0;
    for (const std::uint32_t texel : *texels)
        covered += texel != black;
    CHECK(covered == maxDraws * 4);
}

TEST_CASE("a readback stays pending until its list is submitted") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    Buffer<std::uint32_t> values;
    REQUIRE_OK(values.open(*device, 5));
    REQUIRE_OK(values.upload(std::array<std::uint32_t, 5>{1, 2, 3, 4, 5}));
    Readback<std::uint32_t> readback;
    REQUIRE_OK(readback.open(*device, 3));
    CHECK(!readback.read());

    CommandList list(*device);
    list.download(values, readback, 1);
    std::expected<Token, Error> token = device->submit(std::move(list));
    REQUIRE_OK(token);
    REQUIRE_OK(device->wait(*token));
    std::expected<std::span<const std::uint32_t>, Error> read = readback.read();
    REQUIRE_OK(read);
    CHECK((*read)[0] == 2 && (*read)[1] == 3 && (*read)[2] == 4);
    CHECK(readback.isReady());
}

TEST_CASE("pipelines that compile in the background are not drawn before they are ready") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    std::array<GraphicsPipeline, 8> pipelines;
    for (std::size_t index = 0; index < pipelines.size(); ++index)
        REQUIRE_OK(pipelines[index].openAsync(*device, {.vertex = Scene::vertMain,
                                                        .fragment = Scene::fragMain,
                                                        .state = {.cull = CullMode::None, .blend = index % 2 ? BlendMode::Alpha : BlendMode::Opaque}}));
    for (const GraphicsPipeline& pipeline : pipelines) {
        REQUIRE_OK(pipeline.wait());
        CHECK(pipeline.isReady());
    }
    GraphicsPipeline invalid;
    REQUIRE_OK(invalid.openAsync(*device, {.vertex = Scene::vertMain, .fragment = Scene::fragMain, .samples = 4}));
    REQUIRE_OK(invalid.wait());
    Image target;
    REQUIRE_OK(target.open(*device, {.width = 4, .height = 4, .usage = ImageUsage::ColorAttachment}));
    CommandList list(*device);
    list.beginRender({.color = &target});
    list.draw(invalid, 3);
    list.endRender();
    CHECK(list.getError().has_value());
}
