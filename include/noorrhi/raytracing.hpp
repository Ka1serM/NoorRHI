#pragma once

#include "image.hpp"
#include "shader.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace noorrhi {

class RayTracingPipeline;

namespace detail {
class DeviceImpl;
class RayTracingPipelineImpl;
struct RayTracingLibraryImpl;
struct AccelerationStructureImpl;
}

struct TriangleGeometry {
    // Address of the first position. May point into a larger interleaved
    // vertex buffer, in which case `stride` is that vertex's size.
    GpuPtr<float3> positions;
    GpuPtr<std::uint32_t> indices;
    std::uint32_t triangle_count = 0;
    // Byte distance between consecutive positions; tightly packed by default.
    std::uint32_t stride = 3 * sizeof(float);
    // Opaque geometry skips any-hit invocation. Gaussian proxy triangles set
    // this to false so their stochastic acceptance shader can reject hits.
    // Non-opaque geometry invokes any-hit at most once per primitive and ray.
    bool opaque = true;
};

class AccelerationStructure {
public:
    AccelerationStructure() = default;
    AccelerationStructureHandle handle() const noexcept;
    // Device address of the structure itself, for writing into an
    // InstanceRecord from a shader.
    std::uint64_t device_address() const;
    explicit operator bool() const noexcept;

private:
    friend class Device;
    friend class detail::DeviceImpl;
    explicit AccelerationStructure(std::shared_ptr<detail::AccelerationStructureImpl> impl)
        : impl_(std::move(impl)) {}
    std::shared_ptr<detail::AccelerationStructureImpl> impl_;
};

// Bit-exact mirror of VkAccelerationStructureInstanceKHR. Exposing the packed
// record lets a compute shader produce TLAS input directly in device-local
// memory, which avoids staging one 64-byte record per instance through the
// host. Scenes with millions of instances cannot afford that staging buffer:
// it is host-visible, and on a system without resizable BAR the host-visible
// device-local heap is only a few hundred megabytes.
struct InstanceRecord {
    // Row-major 3x4 object-to-world affine transform.
    float transform[12];
    // Low 24 bits the custom index, high 8 bits the visibility mask.
    std::uint32_t custom_index_and_mask;
    // Low 24 bits the hit-group record offset, high 8 bits the instance flags.
    std::uint32_t shader_binding_table_offset_and_flags;
    std::uint64_t blas_address;
};
static_assert(sizeof(InstanceRecord) == 64);

// Matches VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR, which is
// what the host-span overloads below apply to every instance they build.
inline constexpr std::uint32_t InstanceFlagTriangleFacingCullDisable = 0x1u;
// Matches VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR.
inline constexpr std::uint32_t InstanceFlagTriangleFrontCounterClockwise = 0x2u;

struct Instance {
    AccelerationStructure blas{};
    float4x4 transform{};
    // Vulkan packs these fields into the TLAS instance record. Keeping them
    // in the clean API lets callers select hit groups without touching Vulkan.
    std::uint32_t custom_index = 0;
    std::uint32_t shader_binding_table_offset = 0;
    std::uint8_t mask = 0xff;
};
struct RayTracingPipelineDesc {
    // Every ray-generation shader the pipeline can launch. Linking several
    // into one pipeline shares a single driver compile of the other stages.
    std::vector<Shader> raygen;
    std::vector<Shader> miss;
    std::vector<Shader> closest_hit;
    std::vector<Shader> any_hit;
    std::vector<Shader> intersection;
    // Callable shaders, invoked by index with CallShader().
    std::vector<Shader> callable;
    // Keep libraries built from imported shaders out of the persisted pipeline cache.
    bool use_pipeline_cache = true;
};

// What every stage of a pipeline built from libraries agrees on. Libraries
// linked together must share one.
struct RayTracingInterface {
    // Largest ray payload any stage declares, in bytes.
    std::uint32_t max_payload_size = 0;
    // Largest hit attribute any stage declares; triangles use two floats.
    std::uint32_t max_hit_attribute_size = 8;
    bool operator==(const RayTracingInterface&) const = default;
};

// A separately compiled part of a ray-tracing pipeline. Linking libraries
// reuses their compiled stages, so a pipeline assembled from libraries that
// already exist costs a link rather than a compile of every stage.
class RayTracingLibrary {
public:
    RayTracingLibrary() = default;
    explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
    friend class Device;
    explicit RayTracingLibrary(std::shared_ptr<detail::RayTracingLibraryImpl> impl)
        : impl_(std::move(impl)) {}
    std::shared_ptr<detail::RayTracingLibraryImpl> impl_;
};

class RayTracingPipeline {
public:
    RayTracingPipeline() = default;
    explicit operator bool() const noexcept { return static_cast<bool>(impl_); }
    // Launches `raygen`, which must be one of the desc's raygen shaders.
    template<class Args>
    void trace(const Shader& raygen, DispatchSize size, const Args& args) const {
        static_assert(std::is_trivially_copyable_v<Args>, "GPU arguments must be trivially copyable");
        trace_bytes(raygen, size, &args, sizeof(Args));
    }

private:
    friend class Device;
    explicit RayTracingPipeline(std::shared_ptr<detail::RayTracingPipelineImpl> impl)
        : impl_(std::move(impl)) {}
    void trace_bytes(const Shader&, DispatchSize, const void*, std::size_t) const;
    std::shared_ptr<detail::RayTracingPipelineImpl> impl_;
};
} // namespace noorrhi
