#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "Bench.h"
#include "BenchQuery.slang.h"
#include "BenchTrace.slang.h"
#include "Benchmark.h"

using namespace NoorRHI;

namespace {

constexpr std::uint32_t resolution = 256;
constexpr std::uint32_t tiles = 8;
constexpr std::uint32_t width = 1920;
constexpr std::uint32_t height = 1080;

struct Terrain {
    Buffer<float3> positions;
    Buffer<std::uint32_t> indices;
    AccelerationStructure blas;
    AccelerationStructure tlas;
    std::vector<Instance> instances;
    TriangleGeometry geometry;
};

Benchmark::Result openTerrain(Device& device, Terrain& terrain) {
    std::vector<float3> vertices;
    std::vector<std::uint32_t> triangles;
    for (std::uint32_t z = 0; z <= resolution; ++z)
        for (std::uint32_t x = 0; x <= resolution; ++x) {
            const float u = static_cast<float>(x) / resolution;
            const float v = static_cast<float>(z) / resolution;
            vertices.push_back({u, 0.05f * (std::sin(u * 40.0f) + std::cos(v * 37.0f)) + 0.03f * std::sin((u + v) * 113.0f), v});
        }
    for (std::uint32_t z = 0; z < resolution; ++z)
        for (std::uint32_t x = 0; x < resolution; ++x) {
            const std::uint32_t corner = z * (resolution + 1) + x;
            triangles.insert(triangles.end(), {corner, corner + resolution + 1, corner + 1, corner + 1, corner + resolution + 1, corner + resolution + 2});
        }
    TRY(terrain.positions.open(device, vertices.size()));
    TRY(terrain.indices.open(device, triangles.size()));
    TRY(terrain.positions.upload(vertices));
    TRY(terrain.indices.upload(triangles));
    terrain.geometry = {
        .positions = terrain.positions.getPtr(), .indices = terrain.indices.getPtr(), .triangleCount = static_cast<std::uint32_t>(triangles.size() / 3)};
    TRY(terrain.blas.build(device, {&terrain.geometry, 1}, BuildMode::Static));
    for (std::uint32_t tile = 0; tile < tiles * tiles; ++tile) {
        Instance instance{.blas = &terrain.blas, .customIndex = tile};
        for (std::size_t index = 0; index < 4; ++index)
            instance.transform.values[index][index] = 1.0f;
        instance.transform.values[0][3] = static_cast<float>(tile % tiles);
        instance.transform.values[2][3] = static_cast<float>(tile / tiles);
        terrain.instances.push_back(instance);
    }
    return terrain.tlas.build(device, terrain.instances);
}

std::string describeRays(Device& device, const Buffer<float>& output, const Benchmark::Timing& timing) {
    std::vector<float> distances(std::size_t{width} * height);
    std::uint32_t hits = 0;
    double sum = 0.0;
    if (output.download(distances))
        for (const float distance : distances) {
            hits += distance > 0.0f;
            sum += distance;
        }
    static_cast<void>(device);
    char text[128];
    std::snprintf(text, sizeof(text), "%.0f Mrays/s, %.1f%% hits, distance sum %.0f", static_cast<double>(width) * height / timing.gpuMs / 1000.0,
                  100.0 * hits / (static_cast<double>(width) * height), sum);
    return text;
}

}

BENCHMARK("ray tracing: build 131k triangle BLAS and 64 instance TLAS") {
    Terrain terrain;
    TRY(openTerrain(device, terrain));
    std::expected<double, Error> blas = Benchmark::measureWall(5, [&] -> Benchmark::Result {
        AccelerationStructure built;
        TRY(built.build(device, {&terrain.geometry, 1}, BuildMode::Static));
        return device.waitIdle();
    });
    if (!blas)
        return std::unexpected(std::move(blas.error()));
    std::expected<double, Error> tlas = Benchmark::measureWall(5, [&] -> Benchmark::Result {
        AccelerationStructure built;
        TRY(built.build(device, terrain.instances));
        return device.waitIdle();
    });
    if (!tlas)
        return std::unexpected(std::move(tlas.error()));
    char extra[96];
    std::snprintf(extra, sizeof(extra), "BLAS %.2f ms, TLAS %.2f ms, with submit and wait", *blas, *tlas);
    Benchmark::Timing timing;
    timing.wallMs = *blas + *tlas;
    Benchmark::report("ray tracing: build 131k triangle BLAS and 64 instance TLAS", timing, extra);
    return {};
}

namespace {

Benchmark::Result runRays(Device& device, const bool incoherent, const bool pipeline) {
    Terrain terrain;
    TRY(openTerrain(device, terrain));
    Buffer<float> output;
    TRY(output.open(device, std::size_t{width} * height));
    const Bench::TraceArguments arguments{output.getPtr(), terrain.tlas.getHandle(), width, height, incoherent, static_cast<float>(tiles)};
    std::expected<Benchmark::Timing, Error> timing = std::unexpected(Error{});
    if (pipeline) {
        const RayTracingInterface interface{.maxPayloadSize = sizeof(float)};
        std::array<RayTracingLibrary, 1> libraries;
        TRY(libraries[0].open(device,
                              {.raygen = {BenchTrace::benchRayGenMain}, .miss = {BenchTrace::benchMissMain}, .closestHit = {BenchTrace::benchClosestHitMain}},
                              interface));
        const std::array<std::uint32_t, 1> hitGroups{0};
        RayTracingPipeline traced;
        TRY(traced.open(device, libraries, hitGroups));
        timing = Benchmark::measure(device, 10, [&](CommandList& list) { list.traceRays(traced, BenchTrace::benchRayGenMain, {width, height}, arguments); });
    } else {
        ComputePipeline query;
        TRY(query.open(device, BenchQuery::benchQueryMain));
        timing = Benchmark::measure(device, 10, [&](CommandList& list) { list.dispatch(query, {width, height}, arguments); });
    }
    if (!timing)
        return std::unexpected(std::move(timing.error()));
    char name[96];
    std::snprintf(name, sizeof(name), "ray tracing: %s rays 1920x1080, %s", incoherent ? "incoherent" : "primary", pipeline ? "pipeline" : "ray query");
    Benchmark::report(name, *timing, describeRays(device, output, *timing));
    return {};
}

}

BENCHMARK("ray tracing: primary rays 1920x1080, pipeline") {
    return runRays(device, false, true);
}

BENCHMARK("ray tracing: primary rays 1920x1080, ray query") {
    return runRays(device, false, false);
}

BENCHMARK("ray tracing: incoherent rays 1920x1080, pipeline") {
    return runRays(device, true, true);
}

BENCHMARK("ray tracing: incoherent rays 1920x1080, ray query") {
    return runRays(device, true, false);
}
