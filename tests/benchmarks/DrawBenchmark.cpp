#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "Bench.h"
#include "BenchDraw.slang.h"
#include "Benchmark.h"
#include "Scene.h"
#include "Scene.slang.h"

using namespace NoorRHI;

namespace {

constexpr std::uint32_t drawCount = 10000;
constexpr std::uint32_t materialCount = 64;

std::string perDraw(const Benchmark::Timing& timing, const std::uint32_t draws) {
    char text[96];
    std::snprintf(text, sizeof(text), "record %.3f us/draw, gpu %.3f us/draw", timing.recordMs * 1000.0 / draws, timing.gpuMs * 1000.0 / draws);
    return text;
}

struct Materials {
    std::vector<Image> textures;
    Buffer<::Scene::Material> buffer;
    Sampler sampler;
};

Benchmark::Result openMaterials(Device& device, Materials& materials) {
    std::vector<::Scene::Material> entries;
    entries.reserve(materialCount);
    materials.textures.resize(materialCount);
    for (std::uint32_t material = 0; material < materialCount; ++material) {
        Image& texture = materials.textures[material];
        TRY(texture.open(device, {.width = 64, .height = 64, .usage = ImageUsage::Sampled}));
        const std::vector<std::uint32_t> texels(64 * 64, 0xff000000u | material * 0x030507u);
        TRY(texture.upload(std::as_bytes(std::span(texels))));
        entries.push_back({texture.getSampledHandle(), 0});
    }
    TRY(materials.buffer.open(device, entries.size()));
    TRY(materials.buffer.upload(entries));
    return materials.sampler.open(device, {.filter = Filter::Linear});
}

struct Quads {
    Buffer<float2> corners;
    Buffer<std::uint32_t> indices;
};

//! Scene::cullMain writes commands with first index 6 and vertex offset 4, so both buffers start with unused entries.
Benchmark::Result openQuads(Device& device, Quads& quads) {
    const std::array<float2, 8> corners{float2{9, 9}, {9, 9}, {9, 9}, {9, 9}, {0, 0}, {1, 0}, {0, 1}, {1, 1}};
    const std::array<std::uint32_t, 12> indices{0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 3};
    TRY(quads.corners.open(device, corners.size()));
    TRY(quads.indices.open(device, indices.size()));
    TRY(quads.corners.upload(corners));
    return quads.indices.upload(indices);
}

Benchmark::Result openInstances(Device& device, Buffer<::Scene::Instance>& buffer, const std::uint32_t grid) {
    std::vector<::Scene::Instance> instances;
    instances.reserve(std::size_t{grid} * grid);
    for (std::uint32_t index = 0; index < grid * grid; ++index)
        instances.push_back({{index % grid, index / grid}, index * 7 % materialCount, index % 3 != 0});
    TRY(buffer.open(device, instances.size()));
    return buffer.upload(instances);
}

}

BENCHMARK("draw: 10k draws with own arguments") {
    Image target;
    TRY(target.open(device, {.width = 1024, .height = 1024, .usage = ImageUsage::ColorAttachment}));
    GraphicsPipeline pipeline;
    TRY(pipeline.open(device, {.vertex = BenchDraw::tinyVertexMain, .fragment = BenchDraw::tinyFragmentMain, .state = {.cull = CullMode::None}}));
    std::expected<Benchmark::Timing, Error> timing = Benchmark::measure(device, 10, [&](CommandList& list) {
        list.beginRender({.color = &target});
        for (std::uint32_t index = 0; index < drawCount; ++index)
            list.draw(pipeline, 3, 1, Bench::DrawArguments{{1.0f, 0.5f, 0.25f, 1.0f}, index});
        list.endRender();
    });
    if (!timing)
        return std::unexpected(std::move(timing.error()));
    Benchmark::report("draw: 10k draws with own arguments", *timing, perDraw(*timing, drawCount));
    return {};
}

BENCHMARK("draw: 10k draws switching between 2 pipelines") {
    Image target;
    TRY(target.open(device, {.width = 1024, .height = 1024, .usage = ImageUsage::ColorAttachment}));
    std::array<GraphicsPipeline, 2> pipelines;
    TRY(pipelines[0].open(device, {.vertex = BenchDraw::tinyVertexMain, .fragment = BenchDraw::tinyFragmentMain, .state = {.cull = CullMode::None}}));
    TRY(pipelines[1].open(
        device,
        {.vertex = BenchDraw::tinyVertexMain, .fragment = BenchDraw::tinyFragmentMain, .state = {.cull = CullMode::None, .blend = BlendMode::Additive}}));
    std::expected<Benchmark::Timing, Error> timing = Benchmark::measure(device, 10, [&](CommandList& list) {
        list.beginRender({.color = &target});
        for (std::uint32_t index = 0; index < drawCount; ++index)
            list.draw(pipelines[index % 2], 3, 1, Bench::DrawArguments{{0.1f, 0.1f, 0.1f, 1.0f}, index});
        list.endRender();
    });
    if (!timing)
        return std::unexpected(std::move(timing.error()));
    Benchmark::report("draw: 10k draws switching between 2 pipelines", *timing, perDraw(*timing, drawCount));
    return {};
}

BENCHMARK("draw: 1M instances, 64 bindless materials, 2048x2048") {
    constexpr std::uint32_t grid = 1024;
    Materials materials;
    TRY(openMaterials(device, materials));
    Quads quads;
    TRY(openQuads(device, quads));
    Buffer<::Scene::Instance> instances;
    TRY(openInstances(device, instances, grid));
    Image target;
    TRY(target.open(device, {.width = 2048, .height = 2048, .usage = ImageUsage::ColorAttachment}));
    GraphicsPipeline pipeline;
    TRY(pipeline.open(device, {.vertex = Scene::vertMain, .fragment = Scene::fragMain, .state = {.cull = CullMode::None}}));
    const ::Scene::DrawArguments arguments{quads.corners.getPtr() + 4, instances.getPtr(), materials.buffer.getPtr(), materials.sampler.getHandle(),
                                           2.0f / grid};
    std::expected<Benchmark::Timing, Error> timing = Benchmark::measure(device, 10, [&](CommandList& list) {
        list.beginRender({.color = &target});
        list.draw(pipeline, quads.indices.getPtr() + 6, 6, grid * grid, arguments);
        list.endRender();
    });
    if (!timing)
        return std::unexpected(std::move(timing.error()));
    char extra[64];
    std::snprintf(extra, sizeof(extra), "%.1f M instances/s on the GPU", grid * grid / timing->gpuMs / 1000.0);
    Benchmark::report("draw: 1M instances, 64 bindless materials, 2048x2048", *timing, extra);
    return {};
}

namespace {

constexpr std::uint32_t cullGrid = 256;
constexpr std::uint32_t cullCount = cullGrid * cullGrid;

struct CullScene {
    Materials materials;
    Quads quads;
    Buffer<::Scene::Instance> instances;
    Image target;
    GraphicsPipeline pipeline;
};

Benchmark::Result openCullScene(Device& device, CullScene& scene) {
    TRY(openMaterials(device, scene.materials));
    TRY(openQuads(device, scene.quads));
    TRY(openInstances(device, scene.instances, cullGrid));
    TRY(scene.target.open(device, {.width = 1024, .height = 1024, .usage = ImageUsage::ColorAttachment}));
    return scene.pipeline.open(device, {.vertex = Scene::vertMain, .fragment = Scene::fragMain, .state = {.cull = CullMode::None}});
}

::Scene::DrawArguments getCullArguments(const CullScene& scene) {
    return {scene.quads.corners.getPtr(), scene.instances.getPtr(), scene.materials.buffer.getPtr(), scene.materials.sampler.getHandle(), 2.0f / cullGrid};
}

}

BENCHMARK("draw: 65k indexed indirect commands") {
    CullScene scene;
    TRY(openCullScene(device, scene));
    std::vector<DrawIndexedArgs> draws;
    draws.reserve(cullCount);
    for (std::uint32_t instance = 0; instance < cullCount; ++instance)
        draws.push_back({6, 1, 6, 4, instance});
    Buffer<DrawIndexedArgs> commands;
    TRY(commands.open(device, draws.size()));
    TRY(commands.upload(draws));
    std::expected<Benchmark::Timing, Error> timing = Benchmark::measure(device, 10, [&](CommandList& list) {
        list.beginRender({.color = &scene.target});
        list.drawIndirect(scene.pipeline, scene.quads.indices.getPtr(), commands.getPtr(), cullCount, getCullArguments(scene));
        list.endRender();
    });
    if (!timing)
        return std::unexpected(std::move(timing.error()));
    Benchmark::report("draw: 65k indexed indirect commands", *timing, perDraw(*timing, cullCount));
    return {};
}

BENCHMARK("draw: GPU cull of 65k instances, then indexed indirect count") {
    CullScene scene;
    TRY(openCullScene(device, scene));
    ComputePipeline cull;
    TRY(cull.open(device, Scene::cullMain));
    Buffer<DrawIndexedArgs> commands;
    Buffer<Atomic<std::uint32_t>> count;
    TRY(commands.open(device, cullCount));
    TRY(count.open(device, 1));
    const std::array zero{Atomic<std::uint32_t>{0}};
    std::expected<Benchmark::Timing, Error> timing = Benchmark::measure(device, 10, [&](CommandList& list) {
        list.upload(count, std::span<const Atomic<std::uint32_t>>(zero));
        list.dispatch(cull, {cullCount}, ::Scene::CullArguments{scene.instances.getPtr(), {}, commands.getPtr(), count.getPtr(), cullCount, 1});
        list.beginRender({.color = &scene.target});
        list.drawIndirect(scene.pipeline, scene.quads.indices.getPtr(), commands.getPtr(), count.getPtr(), cullCount, getCullArguments(scene));
        list.endRender();
    });
    if (!timing)
        return std::unexpected(std::move(timing.error()));
    Benchmark::report("draw: GPU cull of 65k instances, then indexed indirect count", *timing, perDraw(*timing, cullCount));
    return {};
}

BENCHMARK("pipeline: compile 12 graphics pipeline variants") {
    std::array<GraphicsPipeline, 12> pipelines;
    const double start = Benchmark::getNow();
    for (std::size_t index = 0; index < pipelines.size(); ++index)
        TRY(pipelines[index].open(device, {.vertex = Scene::vertMain,
                                           .fragment = Scene::fragMain,
                                           .state = {.cull = static_cast<CullMode>(index % 3), .blend = static_cast<BlendMode>(index / 3)}}));
    Benchmark::Timing timing;
    timing.wallMs = Benchmark::getNow() - start;
    Benchmark::report("pipeline: compile 12 graphics pipeline variants", timing, "first compile in the process");
    return {};
}
