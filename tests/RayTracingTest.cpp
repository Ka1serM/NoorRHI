#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "RayGrid.slang.h"
#include "RayQuery.slang.h"
#include "RayTracing.h"
#include "RayTracing.slang.h"
#include "Test.h"

using namespace NoorRHI;

namespace {

constexpr std::uint32_t width = 16;
constexpr std::uint32_t height = 16;
constexpr std::uint32_t frontFaceHit = 2;

struct Scene {
    Buffer<float3> positions;
    Buffer<std::uint32_t> indices;
    AccelerationStructure blas;
    AccelerationStructure tlas;
    Instance instance;
};

std::expected<void, Error> openScene(Device& device, Scene& scene) {
    const std::vector<float3> vertices{{-0.6f, -0.6f, 0.0f}, {0.6f, -0.6f, 0.0f}, {0.0f, 0.6f, 0.0f}};
    const std::vector<std::uint32_t> triangle{0, 1, 2};
    for (std::expected<void, Error> step : {scene.positions.open(device, vertices.size()), scene.indices.open(device, triangle.size())})
        if (!step)
            return step;
    for (std::expected<void, Error> step : {scene.positions.upload(vertices), scene.indices.upload(triangle)})
        if (!step)
            return step;
    const TriangleGeometry geometry{.positions = scene.positions.getPtr(), .indices = scene.indices.getPtr(), .triangleCount = 1};
    if (std::expected<void, Error> built = scene.blas.build(device, {&geometry, 1}); !built)
        return built;
    scene.instance.blas = &scene.blas;
    for (std::size_t index = 0; index < 4; ++index)
        scene.instance.transform.values[index][index] = 1.0f;
    return scene.tlas.build(device, {&scene.instance, 1});
}

std::expected<std::vector<std::uint32_t>, Error> finish(Device& device, CommandList&& list, const Buffer<std::uint32_t>& output) {
    if (std::expected<void, Error> executed = Test::execute(device, std::move(list)); !executed)
        return std::unexpected(std::move(executed.error()));
    std::vector<std::uint32_t> hits(width * height);
    if (std::expected<void, Error> downloaded = output.download(hits); !downloaded)
        return std::unexpected(std::move(downloaded.error()));
    return hits;
}

std::expected<std::vector<std::uint32_t>, Error> trace(Device& device, const RayTracingPipeline& pipeline, const Buffer<std::uint32_t>& output,
                                                       const AccelerationStructure& scene) {
    CommandList list(device);
    list.traceRays(pipeline, RayTracing::rayGenMain, {width, height}, RayTracing::Arguments{output.getPtr(), scene.getHandle(), width});
    return finish(device, std::move(list), output);
}

std::ptrdiff_t countHits(const std::vector<std::uint32_t>& hits) {
    return std::count(hits.begin(), hits.end(), frontFaceHit);
}

void checkTriangle(const std::vector<std::uint32_t>& hits) {
    CHECK(hits[(height / 2) * width + width / 2] == frontFaceHit);
    CHECK(hits[0] == 0u);
    CHECK(hits[width - 1] == 0u);
    CHECK(hits[(height - 1) * width] == 0u);
    CHECK(countHits(hits) > 0);
    CHECK(countHits(hits) + std::count(hits.begin(), hits.end(), 0u) == static_cast<std::ptrdiff_t>(hits.size()));
}

}

TEST_CASE("ray tracing hits a triangle in an acceleration structure and follows instance updates") {
    static_assert(sizeof(RayTracing::Arguments) == 20 && offsetof(RayTracing::Arguments, scene) == 8);
    Device* device = Test::getDevice();
    REQUIRE(device);
    Scene scene;
    REQUIRE_OK(openScene(*device, scene));
    REQUIRE(scene.tlas.getAddress() != 0);

    const RayTracingInterface interface{.maxPayloadSize = sizeof(std::uint32_t)};
    std::array<RayTracingLibrary, 2> libraries;
    REQUIRE_OK(libraries[0].open(*device, {.raygen = {RayTracing::rayGenMain}, .miss = {RayTracing::missMain}}, interface));
    REQUIRE_OK(libraries[1].open(*device, {.closestHit = {RayTracing::closestHitMain}}, interface));
    const std::array<std::uint32_t, 1> hitGroups{0};
    RayTracingPipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, libraries, hitGroups));

    Buffer<std::uint32_t> output;
    REQUIRE_OK(output.open(*device, width * height));
    std::expected<std::vector<std::uint32_t>, Error> hits = trace(*device, pipeline, output, scene.tlas);
    REQUIRE_OK(hits);
    checkTriangle(*hits);

    scene.instance.transform.values[0][3] = 10.0f;
    REQUIRE_OK(scene.tlas.update({&scene.instance, 1}));
    hits = trace(*device, pipeline, output, scene.tlas);
    REQUIRE_OK(hits);
    CHECK(countHits(*hits) == 0);

    scene.instance.transform.values[0][3] = 0.0f;
    REQUIRE_OK(scene.tlas.update({&scene.instance, 1}));
    hits = trace(*device, pipeline, output, scene.tlas);
    REQUIRE_OK(hits);
    checkTriangle(*hits);

    RayTracingPipeline relinked;
    REQUIRE_OK(relinked.open(pipeline, hitGroups));
    hits = trace(*device, relinked, output, scene.tlas);
    REQUIRE_OK(hits);
    checkTriangle(*hits);
}

TEST_CASE("inline ray queries in a compute shader hit the same triangle") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    Scene scene;
    REQUIRE_OK(openScene(*device, scene));
    ComputePipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, RayQuery::rayQueryMain));
    Buffer<std::uint32_t> output;
    REQUIRE_OK(output.open(*device, width * height));
    CommandList list(*device);
    list.dispatch(pipeline, {width, height}, RayTracing::Arguments{output.getPtr(), scene.tlas.getHandle(), width});
    std::expected<std::vector<std::uint32_t>, Error> hits = finish(*device, std::move(list), output);
    REQUIRE_OK(hits);
    checkTriangle(*hits);
}

namespace {

constexpr std::uint32_t gridSize = 128;
constexpr std::uint32_t shiftedInstance = 7;

bool isCellPresent(const std::int64_t x, const std::int64_t y) {
    return x >= 0 && x < gridSize && (x + y) % 3 != 0;
}

struct Grid {
    Buffer<float3> positions;
    Buffer<std::uint32_t> indices;
    std::vector<std::uint32_t> cellPrimitives;
    AccelerationStructure blas;
    AccelerationStructure tlas;
};

std::expected<void, Error> openGrid(Device& device, Grid& grid) {
    std::vector<float3> vertices;
    std::vector<std::uint32_t> triangles;
    grid.cellPrimitives.assign(gridSize * gridSize, 0);
    const float cell = 2.0f / gridSize;
    for (std::uint32_t y = 0; y < gridSize; ++y)
        for (std::uint32_t x = 0; x < gridSize; ++x) {
            if (!isCellPresent(x, y))
                continue;
            const auto first = static_cast<std::uint32_t>(vertices.size());
            const float left = -1.0f + x * cell;
            const float bottom = -1.0f + y * cell;
            vertices.push_back({left, bottom, 0.0f});
            vertices.push_back({left + cell, bottom, 0.0f});
            vertices.push_back({left + cell, bottom + cell, 0.0f});
            vertices.push_back({left, bottom + cell, 0.0f});
            grid.cellPrimitives[y * gridSize + x] = static_cast<std::uint32_t>(triangles.size() / 3) + 1;
            triangles.insert(triangles.end(), {first, first + 1, first + 2, first, first + 2, first + 3});
        }
    if (std::expected<void, Error> opened = grid.positions.open(device, vertices.size()); !opened)
        return opened;
    if (std::expected<void, Error> opened = grid.indices.open(device, triangles.size()); !opened)
        return opened;
    if (std::expected<void, Error> uploaded = grid.positions.upload(vertices); !uploaded)
        return uploaded;
    if (std::expected<void, Error> uploaded = grid.indices.upload(triangles); !uploaded)
        return uploaded;
    const TriangleGeometry geometry{
        .positions = grid.positions.getPtr(), .indices = grid.indices.getPtr(), .triangleCount = static_cast<std::uint32_t>(triangles.size() / 3)};
    if (std::expected<void, Error> built = grid.blas.build(device, {&geometry, 1}); !built)
        return built;
    std::array<Instance, 2> instances{Instance{.blas = &grid.blas}, Instance{.blas = &grid.blas, .customIndex = shiftedInstance}};
    for (Instance& instance : instances)
        for (std::size_t index = 0; index < 4; ++index)
            instance.transform.values[index][index] = 1.0f;
    instances[1].transform.values[0][3] = cell;
    instances[1].transform.values[2][3] = -0.5f;
    return grid.tlas.build(device, instances);
}

std::uint32_t getExpectedGridHit(const Grid& grid, const std::uint32_t x, const std::uint32_t y) {
    if (isCellPresent(x, y))
        return grid.cellPrimitives[y * gridSize + x] + 1;
    if (isCellPresent(std::int64_t{x} - 1, y))
        return shiftedInstance << 24 | (grid.cellPrimitives[y * gridSize + x - 1] + 1);
    return 0;
}

std::expected<std::vector<std::uint32_t>, Error> downloadGrid(Device& device, CommandList&& list, const Buffer<std::uint32_t>& output) {
    if (std::expected<void, Error> executed = Test::execute(device, std::move(list)); !executed)
        return std::unexpected(std::move(executed.error()));
    std::vector<std::uint32_t> hits(gridSize * gridSize);
    if (std::expected<void, Error> downloaded = output.download(hits); !downloaded)
        return std::unexpected(std::move(downloaded.error()));
    return hits;
}

std::size_t countGridMismatches(const Grid& grid, const std::vector<std::uint32_t>& hits) {
    std::size_t mismatches = 0;
    for (std::uint32_t y = 0; y < gridSize; ++y)
        for (std::uint32_t x = 0; x < gridSize; ++x)
            mismatches += hits[y * gridSize + x] != getExpectedGridHit(grid, x, y);
    return mismatches;
}

}

TEST_CASE("ray tracing finds the closest primitive and instance in a large scene") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    Grid grid;
    REQUIRE_OK(openGrid(*device, grid));
    Buffer<std::uint32_t> output;
    REQUIRE_OK(output.open(*device, gridSize * gridSize));
    const RayTracing::Arguments arguments{output.getPtr(), grid.tlas.getHandle(), gridSize};

    const RayTracingInterface interface{.maxPayloadSize = sizeof(std::uint32_t)};
    std::array<RayTracingLibrary, 1> libraries;
    REQUIRE_OK(libraries[0].open(*device, {.raygen = {RayGrid::gridRayGenMain}, .miss = {RayGrid::gridMissMain}, .closestHit = {RayGrid::gridClosestHitMain}},
                                 interface));
    const std::array<std::uint32_t, 1> hitGroups{0};
    RayTracingPipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, libraries, hitGroups));
    CommandList traced(*device);
    traced.traceRays(pipeline, RayGrid::gridRayGenMain, {gridSize, gridSize}, arguments);
    std::expected<std::vector<std::uint32_t>, Error> hits = downloadGrid(*device, std::move(traced), output);
    REQUIRE_OK(hits);
    CHECK(countGridMismatches(grid, *hits) == 0);

    ComputePipeline query;
    REQUIRE_OK(query.open(*device, RayQuery::gridQueryMain));
    CommandList queried(*device);
    queried.dispatch(query, {gridSize, gridSize}, arguments);
    hits = downloadGrid(*device, std::move(queried), output);
    REQUIRE_OK(hits);
    CHECK(countGridMismatches(grid, *hits) == 0);
}

TEST_CASE("an acceleration structure without geometry is rejected") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    AccelerationStructure blas;
    CHECK(blas.build(*device, std::span<const TriangleGeometry>{}).error().code == ErrorCode::InvalidArgument);
}
