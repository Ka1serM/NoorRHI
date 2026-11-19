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
constexpr std::uint32_t clutterSpheres = 400;
constexpr std::uint32_t sphereSegments = 24;
constexpr std::uint32_t sphereRings = 12;
constexpr std::uint32_t clutterSticks = 3000;

struct View {
    float4 eye;
    float4 forward;
    float4 originLow;
    float4 originHigh;
};

struct BenchScene {
    Buffer<float3> positions;
    Buffer<std::uint32_t> indices;
    AccelerationStructure blas;
    AccelerationStructure tlas;
    std::vector<Instance> instances;
    TriangleGeometry geometry;
    View view;
};

Instance getInstance(BenchScene& scene, const std::uint32_t customIndex, const float x, const float z) {
    Instance instance{.blas = &scene.blas, .customIndex = customIndex};
    for (std::size_t index = 0; index < 4; ++index)
        instance.transform.values[index][index] = 1.0f;
    instance.transform.values[0][3] = x;
    instance.transform.values[2][3] = z;
    return instance;
}

Benchmark::Result openScene(Device& device, BenchScene& scene, const std::vector<float3>& vertices, const std::vector<std::uint32_t>& triangles) {
    TRY(scene.positions.open(device, vertices.size()));
    TRY(scene.indices.open(device, triangles.size()));
    TRY(scene.positions.upload(vertices));
    TRY(scene.indices.upload(triangles));
    scene.geometry = {
        .positions = scene.positions.getPtr(), .indices = scene.indices.getPtr(), .triangleCount = static_cast<std::uint32_t>(triangles.size() / 3)};
    TRY(scene.blas.build(device, {&scene.geometry, 1}, BuildMode::Static));
    return scene.tlas.build(device, scene.instances);
}

Benchmark::Result openTerrain(Device& device, BenchScene& scene) {
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
    for (std::uint32_t tile = 0; tile < tiles * tiles; ++tile)
        scene.instances.push_back(getInstance(scene, tile, static_cast<float>(tile % tiles), static_cast<float>(tile / tiles)));
    constexpr float extent = tiles;
    scene.view = {{extent * 0.5f, 1.5f, -1.0f, 0.0f}, {0.0f, -0.35f, 1.0f, 0.0f}, {0.0f, 0.3f, 0.0f, 0.0f}, {extent, 1.0f, extent, 0.0f}};
    return openScene(device, scene, vertices, triangles);
}

class Random {
public:
    float next() {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(state >> 8) * (1.0f / 16777216.0f);
    }

    float next(const float low, const float high) { return low + next() * (high - low); }

private:
    std::uint32_t state = 12345u;
};

void appendSphere(std::vector<float3>& vertices, std::vector<std::uint32_t>& triangles, const float3 center, const float radius) {
    const auto first = static_cast<std::uint32_t>(vertices.size());
    for (std::uint32_t ring = 0; ring <= sphereRings; ++ring) {
        const float polar = 3.14159265f * static_cast<float>(ring) / sphereRings;
        for (std::uint32_t segment = 0; segment <= sphereSegments; ++segment) {
            const float azimuth = 6.2831853f * static_cast<float>(segment) / sphereSegments;
            vertices.push_back({center.x + radius * std::sin(polar) * std::cos(azimuth), center.y + radius * std::cos(polar),
                                center.z + radius * std::sin(polar) * std::sin(azimuth)});
        }
    }
    for (std::uint32_t ring = 0; ring < sphereRings; ++ring)
        for (std::uint32_t segment = 0; segment < sphereSegments; ++segment) {
            const std::uint32_t corner = first + ring * (sphereSegments + 1) + segment;
            const std::uint32_t below = corner + sphereSegments + 1;
            triangles.insert(triangles.end(), {corner, below, corner + 1, corner + 1, below, below + 1});
        }
}

void appendStick(std::vector<float3>& vertices, std::vector<std::uint32_t>& triangles, Random& random) {
    const float3 base{random.next(-8.0f, 8.0f), random.next(0.0f, 4.0f), random.next(-8.0f, 8.0f)};
    const float z = random.next(-1.0f, 1.0f);
    const float angle = random.next(0.0f, 6.2831853f);
    const float radius = std::sqrt(1.0f - z * z);
    const float length = random.next(0.5f, 4.0f);
    const float3 tip{base.x + length * radius * std::cos(angle), base.y + length * z, base.z + length * radius * std::sin(angle)};
    const auto first = static_cast<std::uint32_t>(vertices.size());
    vertices.insert(vertices.end(), {base, tip, {base.x + 0.01f, base.y + 0.01f, base.z}});
    triangles.insert(triangles.end(), {first, first + 1, first + 2});
}

Benchmark::Result openClutter(Device& device, BenchScene& scene) {
    std::vector<float3> vertices{{-10.0f, 0.0f, -10.0f}, {10.0f, 0.0f, -10.0f}, {10.0f, 0.0f, 10.0f}, {-10.0f, 0.0f, 10.0f}};
    std::vector<std::uint32_t> triangles{0, 2, 1, 0, 3, 2};
    Random random;
    for (std::uint32_t sphere = 0; sphere < clutterSpheres; ++sphere) {
        const float radius = 0.03f * std::pow(30.0f, random.next());
        appendSphere(vertices, triangles, {random.next(-8.0f, 8.0f), radius + random.next(0.0f, 3.0f), random.next(-8.0f, 8.0f)}, radius);
    }
    for (std::uint32_t stick = 0; stick < clutterSticks; ++stick)
        appendStick(vertices, triangles, random);
    scene.instances.push_back(getInstance(scene, 0, 0.0f, 0.0f));
    scene.view = {{0.0f, 4.0f, -14.0f, 0.0f}, {0.0f, -3.0f, 14.0f, 0.0f}, {-8.0f, 0.1f, -8.0f, 0.0f}, {8.0f, 4.0f, 8.0f, 0.0f}};
    return openScene(device, scene, vertices, triangles);
}

std::string describeRays(const Buffer<float>& output, const Benchmark::Timing& timing) {
    std::vector<float> distances(std::size_t{width} * height);
    std::uint32_t hits = 0;
    double sum = 0.0;
    if (output.download(distances))
        for (const float distance : distances) {
            hits += distance > 0.0f;
            sum += distance;
        }
    char text[128];
    std::snprintf(text, sizeof(text), "%.0f Mrays/s, %.1f%% hits, distance sum %.0f", static_cast<double>(width) * height / timing.gpuMs / 1000.0,
                  100.0 * hits / (static_cast<double>(width) * height), sum);
    return text;
}

}

namespace {

Benchmark::Result runBuild(Device& device, Benchmark::Result (*open)(Device&, BenchScene&), const std::string_view name) {
    BenchScene scene;
    TRY(open(device, scene));
    std::expected<double, Error> blas = Benchmark::measureWall(5, [&] -> Benchmark::Result {
        AccelerationStructure built;
        TRY(built.build(device, {&scene.geometry, 1}, BuildMode::Static));
        return device.waitIdle();
    });
    if (!blas)
        return std::unexpected(std::move(blas.error()));
    std::expected<double, Error> tlas = Benchmark::measureWall(5, [&] -> Benchmark::Result {
        AccelerationStructure built;
        TRY(built.build(device, scene.instances));
        return device.waitIdle();
    });
    if (!tlas)
        return std::unexpected(std::move(tlas.error()));
    char extra[96];
    std::snprintf(extra, sizeof(extra), "BLAS %.2f ms, TLAS %.2f ms, with submit and wait", *blas, *tlas);
    Benchmark::Timing timing;
    timing.wallMs = *blas + *tlas;
    Benchmark::report(name, timing, extra);
    return {};
}

}

BENCHMARK("ray tracing: build 131k triangle BLAS and 64 instance TLAS") {
    return runBuild(device, openTerrain, "ray tracing: build 131k triangle BLAS and 64 instance TLAS");
}

BENCHMARK("ray tracing: build 233k triangle clutter BLAS") {
    return runBuild(device, openClutter, "ray tracing: build 233k triangle clutter BLAS");
}

namespace {

Benchmark::Result runRays(Device& device, Benchmark::Result (*open)(Device&, BenchScene&), const std::string_view label, const bool incoherent,
                          const bool pipeline) {
    BenchScene scene;
    TRY(open(device, scene));
    Buffer<float> output;
    TRY(output.open(device, std::size_t{width} * height));
    const Bench::TraceArguments arguments{
        output.getPtr(), scene.tlas.getHandle(), scene.view.eye, scene.view.forward, scene.view.originLow, scene.view.originHigh, width, height, incoherent, 0};
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
    std::snprintf(name, sizeof(name), "ray tracing: %.*s%s rays 1920x1080, %s", static_cast<int>(label.size()), label.data(),
                  incoherent ? "incoherent" : "primary", pipeline ? "pipeline" : "ray query");
    Benchmark::report(name, *timing, describeRays(output, *timing));
    return {};
}

}

BENCHMARK("ray tracing: primary rays 1920x1080, pipeline") {
    return runRays(device, openTerrain, "", false, true);
}

BENCHMARK("ray tracing: primary rays 1920x1080, ray query") {
    return runRays(device, openTerrain, "", false, false);
}

BENCHMARK("ray tracing: incoherent rays 1920x1080, pipeline") {
    return runRays(device, openTerrain, "", true, true);
}

BENCHMARK("ray tracing: incoherent rays 1920x1080, ray query") {
    return runRays(device, openTerrain, "", true, false);
}

BENCHMARK("ray tracing: clutter primary rays 1920x1080, ray query") {
    return runRays(device, openClutter, "clutter ", false, false);
}

BENCHMARK("ray tracing: clutter incoherent rays 1920x1080, ray query") {
    return runRays(device, openClutter, "clutter ", true, false);
}
