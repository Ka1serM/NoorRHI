#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <vector>

#include "NoorRHI/Core/Error.h"
#include "NoorRHI/Core/Types.h"
#include "NoorRHI/Pipelines/Pipeline.h"

namespace NoorRHI {

class Device;
namespace Backend {
struct AccelerationStructure;
struct RayTracingLibrary;
struct RayTracingPipeline;
}

struct TriangleGeometry {
    Ptr<float3> positions;
    Ptr<std::uint32_t> indices;
    std::uint32_t triangleCount = 0;
    std::uint32_t stride = 3 * sizeof(float);
    bool opaque = true;
};

enum class BuildMode : std::uint8_t { Static, Dynamic };

//! Bit-exact VkAccelerationStructureInstanceKHR.
struct InstanceRecord {
    float transform[12];
    std::uint32_t customIndexAndMask;
    std::uint32_t shaderBindingTableOffsetAndFlags;
    std::uint64_t blasAddress;
};
static_assert(sizeof(InstanceRecord) == 64);

//! Flags in the top byte of InstanceRecord::shaderBindingTableOffsetAndFlags.
inline constexpr std::uint32_t instanceNoCull = 0x1u;
inline constexpr std::uint32_t instanceCounterClockwise = 0x2u;

class AccelerationStructure;

struct Instance {
    const AccelerationStructure* blas = nullptr;
    float4x4 transform;
    std::uint32_t customIndex = 0;
    std::uint32_t shaderBindingTableOffset = 0;
    std::uint8_t mask = 0xff;
};

class AccelerationStructure {
public:
    [[nodiscard]] std::expected<void, Error> build(Device& device, std::span<const TriangleGeometry> geometry, BuildMode mode = BuildMode::Dynamic);
    [[nodiscard]] std::expected<void, Error> build(Device& device, std::span<const Instance> instances);
    [[nodiscard]] std::expected<void, Error> build(Device& device, Ptr<InstanceRecord> records, std::uint32_t count);
    //! Without instances, it refits the triangles of a dynamic BLAS after their positions change.
    [[nodiscard]] std::expected<void, Error> update();
    [[nodiscard]] std::expected<void, Error> update(std::span<const Instance> instances);
    [[nodiscard]] std::expected<void, Error> update(Ptr<InstanceRecord> records, std::uint32_t count);

    std::uint64_t getAddress() const { return address; }
    AccelerationStructureHandle getHandle() const { return AccelerationStructureHandle(address); }
    Backend::AccelerationStructure* getBackend() const { return backend.get(); }
    explicit operator bool() const { return backend != nullptr; }

private:
    std::expected<void, Error> adopt(std::expected<std::shared_ptr<Backend::AccelerationStructure>, Error> built);

    std::shared_ptr<Backend::AccelerationStructure> backend;
    std::uint64_t address = 0;
};

struct RayTracingPipelineDesc {
    std::vector<Shader> raygen;
    std::vector<Shader> miss;
    std::vector<Shader> closestHit;
    std::vector<Shader> anyHit;
    std::vector<Shader> intersection;
    std::vector<Shader> callable;
    bool usePipelineCache = true;
};

struct RayTracingInterface {
    std::uint32_t maxPayloadSize = 0;
    std::uint32_t maxHitAttributeSize = 8;
    bool operator==(const RayTracingInterface&) const = default;
};

class RayTracingLibrary {
public:
    [[nodiscard]] std::expected<void, Error> open(Device& device, const RayTracingPipelineDesc& desc, const RayTracingInterface& interface);

    Backend::RayTracingLibrary* getBackend() const { return backend.get(); }
    explicit operator bool() const { return backend != nullptr; }

private:
    friend class RayTracingPipeline;

    std::shared_ptr<Backend::RayTracingLibrary> backend;
};

class RayTracingPipeline {
public:
    [[nodiscard]] std::expected<void, Error> open(Device& device, std::span<const RayTracingLibrary> libraries, std::span<const std::uint32_t> hitGroups);
    [[nodiscard]] std::expected<void, Error> open(const RayTracingPipeline& linked, std::span<const std::uint32_t> hitGroups);

    const Backend::RayTracingPipeline* getBackend() const { return backend.get(); }
    explicit operator bool() const { return backend != nullptr; }

private:
    std::shared_ptr<const Backend::RayTracingPipeline> backend;
};

}
