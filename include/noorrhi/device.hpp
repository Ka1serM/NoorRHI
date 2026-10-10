#pragma once

#include "types.hpp"

#include <filesystem>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <span>
#include <string>
#include <string_view>

namespace noorrhi {

class Device;

class Shader;
class ComputePipeline;
class GraphicsPipeline;
class RayTracingPipeline;
class AccelerationStructure;
class Swapchain;
class Frame;
class SurfaceProvider;
struct GraphicsPipelineDesc;
struct RenderTarget;
struct RayTracingPipelineDesc;
struct RayTracingInterface;
class RayTracingLibrary;
struct TriangleGeometry;
enum class AccelerationStructureBuildMode : std::uint8_t {
    Static,
    Dynamic,
};
struct Instance;
struct InstanceRecord;
class Sampler;
template<class T> class Buffer;
template<class T> class Image;
enum class ImageUsage : std::uint32_t;
enum class ImageFormat;
struct SamplerDesc;

namespace detail { class DeviceImpl; }

namespace interop {
struct DeviceHandles;
struct ExternalImageMemory;
struct ExternalSemaphore;
DeviceHandles device_handles(Device&);
std::uintptr_t image_view(Device&, ImageHandle);
std::uintptr_t image(Device&, ImageHandle);
void record(Device&, const std::function<void(std::uintptr_t)>&);
ExternalImageMemory export_image_memory(Device&, ImageHandle);
ExternalSemaphore signal_external(Device&);
}

// A GPU-side timing scope. The query is written by the GPU, so its result is
// available once the submission that recorded it has completed - after
// synchronize(), or after the frame that contained it has been presented and
// its token waited on.
class TimestampQuery {
public:
    TimestampQuery() = default;

    // Elapsed GPU time for the last completed measurement, in milliseconds.
    // Returns 0 before the first measurement completes.
    double milliseconds() const;
    explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
    friend class Device;
    friend class detail::DeviceImpl;
    struct State;
    explicit TimestampQuery(std::shared_ptr<State> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<State> impl_;
};

// A record staged once in the device's argument arena and read by several
// launches through its address - for arguments too large to copy into every
// launch, which then pass only address(). The region is never reused while
// this object lives; once it is destroyed, the region is freed after the work
// recorded up to then has completed.
class StagedArguments {
public:
    StagedArguments() = default;
    StagedArguments(StagedArguments&& other) noexcept;
    StagedArguments& operator=(StagedArguments&& other) noexcept;
    StagedArguments(const StagedArguments&) = delete;
    StagedArguments& operator=(const StagedArguments&) = delete;
    ~StagedArguments();

    std::uint64_t address() const noexcept { return address_; }

private:
    friend class Device;
    StagedArguments(std::shared_ptr<detail::DeviceImpl> device, std::uint64_t id,
        std::uint64_t address)
        : device_(std::move(device)), id_(id), address_(address) {}
    void release() noexcept;
    std::shared_ptr<detail::DeviceImpl> device_;
    std::uint64_t id_ = 0;
    std::uint64_t address_ = 0;
};

// Everything but ray tracing is core Vulkan 1.3 and mandatory: buffer device
// addresses, timeline semaphores, synchronization2, dynamic rendering, the
// descriptor indexing behind the bindless set, scalarBlockLayout, shaderInt64
// and shaderDrawParameters. A device that lacks any of them is rejected at
// construction. Ray tracing is reported.
struct DeviceFeatures {
    bool ray_query = false;
    bool ray_tracing = false;
};

// Vulkan driver identity, for diagnostics and driver-specific workarounds.
struct DeviceInfo {
    std::string name;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::uint32_t driver_version = 0;
};

// VMA totals include allocations awaiting GPU completion and allocator slack.
// Dedicated external images are owned by the external-memory interop boundary.
struct MemoryReport {
    std::uint64_t allocation_bytes = 0;
    std::uint64_t reserved_bytes = 0;
    std::uint64_t allocation_count = 0;
    std::uint64_t argument_arena_bytes = 0;
};

struct DeviceConfig {
    bool enable_validation = false;
    std::string_view application_name = "NoorRHI";
    // Size of each chunk of the arena holding the per-launch root argument
    // records. Records are tens to a few hundred bytes. Chunks are added as
    // submissions need them and reused once those submissions complete, so this
    // sets the granularity, not a limit.
    std::size_t argument_arena_bytes = 1024u * 1024u;
    // Slots in the device's bindless descriptor set. Every sampled or storage
    // image view takes one texture slot and every sampler one sampler slot;
    // slot 0 of each is reserved. The set never grows, so exhausting either
    // throws ErrorCode::OutOfMemory.
    std::uint32_t texture_descriptor_capacity = 16384;
    std::uint32_t sampler_descriptor_capacity = 256;
    // Optional. Supplying a window's provider lets Swapchains be created on
    // this device: it adopts the window system's Vulkan loader and instance
    // extensions, enables the swapchain extensions, and only selects a GPU and
    // queue that can present to that window. The device keeps no surface of
    // its own; each Swapchain creates one. Leaving it null gives a headless
    // device, which is what offline rendering and the tests want.
    SurfaceProvider* presentation = nullptr;
    // Optional. A folder owned by the device, holding the driver's pipeline
    // cache (pipeline.cache), so later runs skip compiling what an earlier run
    // already did. Without it, pipelines are compiled every run.
    std::filesystem::path pipeline_directory;
};

class Device {
public:
    explicit Device(const DeviceConfig& config = {});
    ~Device();
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    DeviceFeatures features() const;
    DeviceInfo info() const;
    // Current allocation totals, broken down by category.
    MemoryReport memory_report() const;

    template<class T> Buffer<T> buffer(std::size_t count);
    template<class T> Image<T> image(std::uint32_t width, std::uint32_t height, ImageUsage usage);
    // A sampled-only image may have a mip chain; its upload holds every level,
    // largest first, tightly packed.
    template<class T> Image<T> image(std::uint32_t width, std::uint32_t height,
        ImageUsage usage, ImageFormat format, std::uint32_t mip_levels = 1);

    Shader create_shader(std::span<const std::byte> spirv);
    Shader create_shader(std::span<const std::byte> spirv, std::string_view entry_point);
    ComputePipeline compute(const Shader& shader);
    GraphicsPipeline graphics(const GraphicsPipelineDesc& desc);
    RayTracingLibrary ray_tracing_library(const RayTracingPipelineDesc& desc,
        const RayTracingInterface& interface);
    // Links the libraries into one pipeline. Each shader binding table region
    // lists its groups library by library, in the order given, so a callable's
    // index counts the callables of every library before its own. The hit
    // region is `hit_groups` instead: record i invokes the hit group with that
    // index, counted the same way, so one group may back many records - an
    // instance's hit offset and a geometry's index select among them.
    RayTracingPipeline ray_tracing(std::span<const RayTracingLibrary> libraries,
        std::span<const std::uint32_t> hit_groups);
    // The linked pipeline with another hit-group table. Only the shader
    // binding table is rebuilt; the pipeline itself is shared.
    RayTracingPipeline ray_tracing(const RayTracingPipeline& linked,
        std::span<const std::uint32_t> hit_groups);
    AccelerationStructure build_blas(std::span<const TriangleGeometry> geometry,
        AccelerationStructureBuildMode mode = AccelerationStructureBuildMode::Dynamic);
    // Refits a BLAS in place against the current contents of the vertex and
    // index buffers it was built from. Much cheaper than a rebuild, but only
    // valid while those buffers keep their allocation and triangle count --
    // deforming geometry, not changing topology.
    void refit_blas(AccelerationStructure& blas);
    AccelerationStructure build_tlas(std::span<const Instance> instances);
    void update_tlas(AccelerationStructure& tlas, std::span<const Instance> instances);
    // Builds from records already resident in device memory. The caller owns
    // the buffer and must keep every referenced BLAS alive through the span.
    AccelerationStructure build_tlas(GpuPtr<InstanceRecord> records, std::uint32_t count,
        std::span<const AccelerationStructure> referenced);
    void update_tlas(AccelerationStructure& tlas, GpuPtr<InstanceRecord> records,
        std::uint32_t count);
    Sampler sampler(const SamplerDesc& desc);

    // Blit one image onto another, scaling if the extents differ. Both images
    // may belong to this device or be swapchain images; layout handling is
    // internal, as everywhere else in this API.
    void copy(ImageHandle source, ImageHandle destination);

    // Inside an open frame or Recording, compute launches and ray-tracing
    // traces are not ordered with each other: a pass that reads what an
    // earlier launch or trace wrote needs a barrier between them. Every other
    // operation (copies, builds, render scopes, interop recording, timestamps)
    // is ordered against all work before and after it, and so is everything
    // submitted outside one.
    void barrier(Stage source, Stage destination);
    // A token for the work submitted so far on the calling thread's queue.
    GpuToken signal();
    void wait(GpuToken token);
    // Makes the calling thread's queue wait on the GPU for the token before it
    // runs the work submitted after this call. The CPU does not wait. A queue
    // waiting on the GPU also holds back everything queued behind the wait.
    void queue_wait(GpuToken token);
    // Whether the work up to the token has completed, without waiting for it.
    bool finished(GpuToken token) const;
    // Waits for the work submitted so far on every queue.
    void synchronize();

    // Draw into `target`. Draws issued by the callback are recorded into the
    // enclosing frame when one is open, and into a private submission when
    // none is.
    void render(const RenderTarget& target, const std::function<void()>& draw_commands);

    // Copies `args` into the argument arena once; see StagedArguments.
    template<class Args>
    StagedArguments stage(const Args& args) {
        static_assert(std::is_trivially_copyable_v<Args>, "GPU arguments must be trivially copyable");
        return stage_bytes(&args, sizeof(Args));
    }
    StagedArguments stage_bytes(const void* args, std::size_t size);

    TimestampQuery timestamp();
    // Bracket `commands` with GPU timestamps written into `query`.
    void measure(const TimestampQuery& query, const std::function<void()>& commands);
    // Names the GPU work `commands` records, for profilers and debuggers such
    // as Nsight. Adds no ordering; labels nest.
    void label(std::string_view name, const std::function<void()>& commands);

    // Presentation is not part of the device: see noorrhi::Swapchain.

private:
    friend class Swapchain;
    friend class QueueScope;
    friend class Recording;
    friend interop::DeviceHandles interop::device_handles(Device&);
    friend std::uintptr_t interop::image_view(Device&, ImageHandle);
    friend std::uintptr_t interop::image(Device&, ImageHandle);
    friend void interop::record(Device&, const std::function<void(std::uintptr_t)>&);
    friend interop::ExternalImageMemory interop::export_image_memory(Device&, ImageHandle);
    friend interop::ExternalSemaphore interop::signal_external(Device&);
    std::shared_ptr<detail::DeviceImpl> impl_;
};

// Threads may use one Device concurrently. Each records its commands into
// command buffers of its own without holding any device-wide lock; only
// submitting takes one briefly. A queue with an open frame or Recording belongs
// to the thread that opened it until it is submitted: other threads' work for
// that queue is rejected meanwhile.

// Routes the device work the calling thread issues to `queue` until the scope
// ends; a thread outside any scope submits to Queue::Graphics. Scopes nest and
// end in reverse order on the thread that began them.
//
// The queues run concurrently on the GPU, but nothing orders one queue's work
// against the other's: a thread that hands results across waits for their
// token first, with wait() or queue_wait().
class QueueScope {
public:
    QueueScope(Device& device, Queue queue);
    ~QueueScope();
    QueueScope(const QueueScope&) = delete;
    QueueScope& operator=(const QueueScope&) = delete;

private:
    const detail::DeviceImpl* device_;
};

// Batches the work the calling thread issues to its queue into one submission,
// as a swapchain Frame does on Graphics: open one per unit of work, such as a
// rendered image, rather than submitting every launch on its own. Uploads and
// readbacks belong before or after it. Destroying an unsubmitted Recording
// discards its work.
class Recording {
public:
    explicit Recording(Device& device);
    ~Recording();
    Recording(const Recording&) = delete;
    Recording& operator=(const Recording&) = delete;

    // Submits the recorded work; the token completes when it has run.
    GpuToken submit();

    struct State;

private:
    std::shared_ptr<State> impl_;
};

} // namespace noorrhi
