#pragma once

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>

#include "noorrhi/compute.hpp"
#include "noorrhi/graphics.hpp"
#include "noorrhi/image.hpp"
#include "noorrhi/interop.hpp"
#include "noorrhi/surface.hpp"
#include "noorrhi/swapchain.hpp"
#include "noorrhi/raytracing.hpp"
#include "noorrhi/sampler.hpp"
#include "pipeline_cache_file.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <functional>
#include <memory>
#include <span>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace noorrhi {

namespace detail { class DeviceImpl; class SwapchainImpl; struct ImageImpl; }

struct TimestampQuery::State {
    std::shared_ptr<detail::DeviceImpl> device;
    // Two consecutive query-pool slots: begin and end.
    std::uint32_t first_query = ~0u;
    // The queue the last measurement ran on, whose family sets the valid bits.
    Queue queue = Queue::Graphics;
    double milliseconds = 0.0;
};

// A frame is the Graphics recording of the thread that began it, plus the
// swapchain image it draws into and presents.
struct Frame::State {
    std::shared_ptr<detail::DeviceImpl> device;
    std::shared_ptr<detail::SwapchainImpl> swapchain;
    vk::CommandBuffer command;
    std::uint32_t image_index = 0;
    std::uint32_t semaphore_index = 0;
    ImageHandle target{};
    bool open = false;
};

struct Recording::State {
    std::shared_ptr<detail::DeviceImpl> device;
    Queue queue = Queue::Graphics;
    vk::CommandBuffer command;
};

} // namespace noorrhi

namespace noorrhi::detail {

class DeviceImpl;
struct SamplerImpl;

// One value per queue timeline, indexed by Queue. Work is covered once every
// queue has reached its value; a zero value covers nothing on that queue.
using Timelines = std::array<std::uint64_t, queue_count>;


// The library renders depth into a single format; RenderTarget depth images
// and depth-enabled pipelines are both built against it.
inline constexpr vk::Format depth_format = vk::Format::eD32Sfloat;

struct FormatLayout {
    vk::Format format;
    // Texel footprint of one block; 1 for uncompressed formats.
    std::uint32_t block_extent;
    std::uint32_t block_bytes;
};

inline FormatLayout format_layout(const ImageFormat format) {
    switch (format) {
    case ImageFormat::Bgra8Unorm: return {vk::Format::eB8G8R8A8Unorm, 1, 4};
    case ImageFormat::Rgba32Float: return {vk::Format::eR32G32B32A32Sfloat, 1, 16};
    case ImageFormat::Rgba16Float: return {vk::Format::eR16G16B16A16Sfloat, 1, 8};
    case ImageFormat::R32Uint: return {vk::Format::eR32Uint, 1, 4};
    case ImageFormat::R32Float: return {vk::Format::eR32Sfloat, 1, 4};
    case ImageFormat::D32Float: return {depth_format, 1, 4};
    case ImageFormat::R8Unorm: return {vk::Format::eR8Unorm, 1, 1};
    case ImageFormat::L8Unorm: return {vk::Format::eR8Unorm, 1, 1};
    case ImageFormat::L8Srgb: return {vk::Format::eR8Srgb, 1, 1};
    case ImageFormat::Rg8Unorm: return {vk::Format::eR8G8Unorm, 1, 2};
    case ImageFormat::Rgba8Srgb: return {vk::Format::eR8G8B8A8Srgb, 1, 4};
    case ImageFormat::Bgra8Srgb: return {vk::Format::eB8G8R8A8Srgb, 1, 4};
    case ImageFormat::R16Unorm: return {vk::Format::eR16Unorm, 1, 2};
    case ImageFormat::Rg16Unorm: return {vk::Format::eR16G16Unorm, 1, 4};
    case ImageFormat::Rgba16Unorm: return {vk::Format::eR16G16B16A16Unorm, 1, 8};
    case ImageFormat::R16Float: return {vk::Format::eR16Sfloat, 1, 2};
    case ImageFormat::Rg16Float: return {vk::Format::eR16G16Sfloat, 1, 4};
    case ImageFormat::Rg32Float: return {vk::Format::eR32G32Sfloat, 1, 8};
    case ImageFormat::A2b10g10r10Unorm: return {vk::Format::eA2B10G10R10UnormPack32, 1, 4};
    case ImageFormat::B10g11r11Float: return {vk::Format::eB10G11R11UfloatPack32, 1, 4};
    case ImageFormat::Bc1Unorm: return {vk::Format::eBc1RgbaUnormBlock, 4, 8};
    case ImageFormat::Bc1Srgb: return {vk::Format::eBc1RgbaSrgbBlock, 4, 8};
    case ImageFormat::Bc2Unorm: return {vk::Format::eBc2UnormBlock, 4, 16};
    case ImageFormat::Bc2Srgb: return {vk::Format::eBc2SrgbBlock, 4, 16};
    case ImageFormat::Bc3Unorm: return {vk::Format::eBc3UnormBlock, 4, 16};
    case ImageFormat::Bc3Srgb: return {vk::Format::eBc3SrgbBlock, 4, 16};
    case ImageFormat::Bc4Unorm: return {vk::Format::eBc4UnormBlock, 4, 8};
    case ImageFormat::Bc5Unorm: return {vk::Format::eBc5UnormBlock, 4, 16};
    case ImageFormat::Bc6hUfloat: return {vk::Format::eBc6HUfloatBlock, 4, 16};
    case ImageFormat::Bc6hSfloat: return {vk::Format::eBc6HSfloatBlock, 4, 16};
    case ImageFormat::Bc7Unorm: return {vk::Format::eBc7UnormBlock, 4, 16};
    case ImageFormat::Bc7Srgb: return {vk::Format::eBc7SrgbBlock, 4, 16};
    case ImageFormat::Auto:
    case ImageFormat::Rgba8Unorm:
    default: return {vk::Format::eR8G8B8A8Unorm, 1, 4};
    }
}

inline vk::Format to_vulkan_format(const ImageFormat format) {
    return format_layout(format).format;
}

inline bool is_block_compressed(const ImageFormat format) {
    return format_layout(format).block_extent > 1;
}

// Bytes of one full mip level, counting partial edge blocks as whole blocks.
inline std::size_t format_byte_size(const ImageFormat format, const std::uint32_t width,
    const std::uint32_t height) {
    const FormatLayout layout = format_layout(format);
    const std::size_t blocks_x = (width + layout.block_extent - 1) / layout.block_extent;
    const std::size_t blocks_y = (height + layout.block_extent - 1) / layout.block_extent;
    return blocks_x * blocks_y * layout.block_bytes;
}

struct BufferImpl {
    std::shared_ptr<DeviceImpl> device;
    vk::Buffer buffer;
    VmaAllocation allocation = VK_NULL_HANDLE;
    vk::DeviceAddress address = 0;
    std::size_t size = 0;
    void* mapped = nullptr;
    bool host_visible = false;
    bool mapped_by_api = false;

    ~BufferImpl();
};

struct ShaderImpl {
    std::shared_ptr<DeviceImpl> device;
    vk::UniqueShaderModule module;
    std::string entry_point = "main";

    ~ShaderImpl() = default;
};

struct ImageImpl {
    std::shared_ptr<DeviceImpl> device;
    vk::Image image;
    vk::UniqueImageView view;
    VmaAllocation allocation = VK_NULL_HANDLE;
    vk::DeviceMemory external_memory;
    ImageHandle handle{};
    TextureHandle sampled_handle{};
    TextureHandle storage_handle{};
    vk::Format format = vk::Format::eR8G8B8A8Unorm;
    ImageFormat public_format = ImageFormat::Rgba8Unorm;
    vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
    std::size_t byte_size = 0;
    bool block_compressed = false;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t mip_levels = 1;
    // False for swapchain images: the presentation engine owns the VkImage and
    // there is no VMA allocation, but the view and descriptors are still ours.
    bool owns_image = true;
    bool exportable = false;

    ~ImageImpl();
};

struct SamplerImpl {
    std::shared_ptr<DeviceImpl> device;
    SamplerHandle handle{};

    ~SamplerImpl();
};

// One device-owned descriptor heap: a mapped buffer of fixed-size slots
// followed by the implementation's reserved range. Slot 0 is never handed out,
// so a zero handle stays null.
struct DescriptorHeap {
    std::shared_ptr<BufferImpl> buffer;
    // First slot, at the aligned heap address inside `buffer`.
    std::byte* slots = nullptr;
    std::size_t buffer_offset = 0;
    vk::DeviceAddress address = 0;
    vk::DeviceSize descriptor_size = 0;
    vk::DeviceSize reserved_offset = 0;
    vk::DeviceSize reserved_size = 0;
    std::uint32_t capacity = 0;
    // Slots below `next` have been handed out at least once; freed ones wait
    // in `free` for reuse.
    std::uint32_t next = 1;
    std::vector<std::uint32_t> free;
    const char* capacity_name = "";
};

struct AccelerationStructureImpl {
    std::shared_ptr<DeviceImpl> device;
    std::shared_ptr<BufferImpl> storage;
    vk::UniqueAccelerationStructureKHR acceleration_structure;
    vk::DeviceAddress address = 0;
    vk::DeviceSize storage_size = 0;
    AccelerationStructureHandle handle{};
    std::uint32_t primitive_count = 0;
    bool updateable = false;
    vk::BuildAccelerationStructureFlagsKHR blas_build_flags{};
    std::vector<std::shared_ptr<AccelerationStructureImpl>> references;
    std::shared_ptr<BufferImpl> update_input;
    std::shared_ptr<BufferImpl> update_scratch;
    // Retained so a BLAS refit can re-issue the original build description.
    // The geometry holds device addresses, so a refit is only valid while the
    // source buffers keep their allocation -- i.e. same vertex/index count.
    std::vector<vk::AccelerationStructureGeometryKHR> blas_geometries;
    std::vector<std::uint32_t> blas_primitive_counts;
    std::vector<std::shared_ptr<BufferImpl>> blas_sources;

    ~AccelerationStructureImpl();
};

class ComputePipelineImpl : public std::enable_shared_from_this<ComputePipelineImpl> {
public:
    std::shared_ptr<DeviceImpl> device;
    vk::UniquePipeline pipeline;

    void launch(DispatchSize groups, const void* args, std::size_t size) const;
    void launch_indirect(GpuPtr<DispatchArgs> args, const void* argument_data,
        std::size_t argument_size) const;
};

class GraphicsPipelineImpl : public std::enable_shared_from_this<GraphicsPipelineImpl> {
public:
    std::shared_ptr<DeviceImpl> device;
    vk::UniquePipeline pipeline;
    vk::Format color_format = vk::Format::eR8G8B8A8Unorm;
    bool uses_depth = false;
    GraphicsState state{};

    void draw(std::uint32_t vertex_count, const void* args, std::size_t size) const;
    void draw_indirect(GpuPtr<DrawArgs> commands, std::uint32_t draw_count,
        const void* args, std::size_t size) const;
};

// The shader groups of a pipeline or library, in group-index order.
struct RayTracingGroups {
    enum class Kind { Raygen, Miss, Hit, Callable };
    struct Group {
        Kind kind;
        // The ray-generation shader, for Raygen groups.
        const ShaderImpl* raygen = nullptr;
    };
    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> create_infos;
    std::vector<Group> groups;
    std::vector<std::shared_ptr<ShaderImpl>> shaders;
};

struct RayTracingLibraryImpl {
    std::shared_ptr<DeviceImpl> device;
    vk::UniquePipeline pipeline;
    RayTracingInterface interface;
    RayTracingGroups groups;
    bool use_pipeline_cache = true;
};

class RayTracingPipelineImpl : public std::enable_shared_from_this<RayTracingPipelineImpl> {
public:
    std::shared_ptr<DeviceImpl> device;
    // Linked pipelines keep their libraries alive.
    std::vector<std::shared_ptr<RayTracingLibraryImpl>> libraries;
    // Shared by every pipeline that only differs in its hit-group table.
    std::shared_ptr<const vk::UniquePipeline> pipeline;
    // What the linked libraries define, in pipeline order.
    std::vector<RayTracingGroups::Group> groups;
    std::shared_ptr<BufferImpl> shader_binding_table;
    // One region per ray-generation shader, keyed by that shader.
    std::vector<std::pair<const ShaderImpl*, vk::StridedDeviceAddressRegionKHR>> raygen_regions;
    vk::StridedDeviceAddressRegionKHR miss_region{};
    vk::StridedDeviceAddressRegionKHR hit_region{};
    vk::StridedDeviceAddressRegionKHR callable_region{};
    void trace(const ShaderImpl& raygen, DispatchSize, const void*, std::size_t) const;
};

// A window's presentation surface. Shared by the live chain and every retired
// one, so the surface outlives each VkSwapchainKHR created against it.
struct SurfaceImpl {
    vk::UniqueSurfaceKHR surface;
};

// One presentation chain. Rebuilt in place whenever the surface size stops
// matching the provider's, so the public Swapchain handle stays valid across
// resizes and the caller never sees an out-of-date error. A replaced chain's
// Vulkan objects go through the device's retire queue instead of waiting for
// the GPU.
class SwapchainImpl {
public:
    std::shared_ptr<DeviceImpl> device;
    SurfaceProvider* provider = nullptr;
    std::shared_ptr<SurfaceImpl> surface;
    vk::UniqueSwapchainKHR swapchain;
    std::vector<std::shared_ptr<ImageImpl>> images;
    // Acquire semaphores cycle independently of image indices: the index is
    // only known after the acquire that the semaphore belongs to.
    struct Acquire {
        vk::UniqueSemaphore semaphore;
        GpuToken token{};
    };
    std::vector<Acquire> acquire_semaphores;
    std::vector<vk::UniqueSemaphore> present_semaphores;
    // A swapchain image must be transitioned from PRESENT_SRC rather than
    // UNDEFINED once it has been presented at least once, or its contents are
    // discarded. Tracked per image because acquisition order is arbitrary.
    std::vector<bool> presented;
    // Timeline values of presented frames whose GPU work may still be running,
    // oldest first. Bounded by max_frames_in_flight.
    std::deque<GpuToken> in_flight;
    vk::Format format = vk::Format::eUndefined;
    ImageFormat public_format = ImageFormat::Bgra8Unorm;
    vk::ColorSpaceKHR color_space = vk::ColorSpaceKHR::eSrgbNonlinear;
    vk::PresentModeKHR present_mode = vk::PresentModeKHR::eFifo;
    PresentMode requested_mode = PresentMode::LowLatency;
    std::uint32_t max_frames_in_flight = 1;
    bool transparent = false;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t semaphore_cursor = 0;
    bool stale = true;

    ~SwapchainImpl();
};

class DeviceImpl : public std::enable_shared_from_this<DeviceImpl> {
public:
    explicit DeviceImpl(const DeviceConfig& config);
    ~DeviceImpl();

    void attach_self(const std::shared_ptr<DeviceImpl>& self) { self_ = self; }
    void shutdown() noexcept;
    void initialize_resources();
    DeviceFeatures features() const noexcept { return features_; }
    DeviceInfo info() const {
        const auto properties = physical_device_.getProperties();
        return {properties.deviceName.data(), properties.vendorID, properties.deviceID,
            properties.driverVersion};
    }
    bool acceleration_structure_supported() const noexcept { return acceleration_structure_supported_; }

    std::shared_ptr<BufferImpl> create_buffer(std::size_t size, vk::BufferUsageFlags usage,
                                               VmaMemoryUsage memory_usage, bool mapped, std::size_t alignment = 1);
    std::shared_ptr<ImageImpl> create_image(std::uint32_t width, std::uint32_t height,
        ImageUsage usage, ImageFormat format, std::uint32_t mip_levels);
    std::shared_ptr<ShaderImpl> create_shader(std::span<const std::byte> spirv,
                                              std::string_view entry_point);
    std::shared_ptr<ComputePipelineImpl> create_compute(const Shader& shader);
    std::shared_ptr<GraphicsPipelineImpl> create_graphics(const GraphicsPipelineDesc& desc);
    std::shared_ptr<SamplerImpl> create_sampler(const SamplerDesc& desc);

    void upload(const std::shared_ptr<BufferImpl>& destination, const void* data,
                std::size_t bytes, std::size_t destination_offset = 0);
    void download(const std::shared_ptr<BufferImpl>& source, void* data, std::size_t bytes,
        std::size_t source_offset = 0);
    void upload_image(const std::shared_ptr<ImageImpl>&, const void* data, std::size_t bytes);
    void download_image(const std::shared_ptr<ImageImpl>&, void* data, std::size_t bytes);
    void download_image_region(const std::shared_ptr<ImageImpl>&, std::uint32_t x,
        std::uint32_t y, std::uint32_t width, std::uint32_t height, void* data, std::size_t bytes);
    void copy_image(ImageHandle source, ImageHandle destination);
    void record_copy_image(vk::CommandBuffer, ImageImpl& source, ImageImpl& destination);
    void transfer_barrier(vk::CommandBuffer, bool before) const;
    void record_barrier(vk::CommandBuffer, Stage source, Stage destination) const;
    static void record_full_barrier(vk::CommandBuffer);
    void barrier(Stage source, Stage destination);
    GpuToken signal();
    void wait(GpuToken token);
    void queue_wait(GpuToken token);
    bool finished(GpuToken token) const;
    void synchronize();

    // The queue the calling thread submits to; see noorrhi::QueueScope.
    Queue current_queue() const;
    // Whether the calling thread's work is batched into a recording it opened.
    bool recording_frame() const;
    // Opens the calling thread's recording on its queue; see noorrhi::Recording.
    std::shared_ptr<Recording::State> begin_recording();
    GpuToken end_recording(Recording::State&);
    void abandon_recording(Recording::State&);
    void bind_queue(Queue queue) const;
    void unbind_queue() const;

    vk::Device device() const noexcept { return vk_device(); }
    vk::PhysicalDevice physical_device() const noexcept { return physical_device_; }
    vk::DeviceAddress buffer_address(vk::Buffer buffer) const;
    std::shared_ptr<BufferImpl> find_buffer_resource(vk::DeviceAddress address) const;
    std::pair<vk::Buffer, vk::DeviceSize> find_buffer(vk::DeviceAddress address) const;
    void retain_active(std::shared_ptr<void> resource);
    // Stage `size` bytes of root arguments in the argument arena and return the
    // GPU address the shader's root pointer should carry.
    vk::DeviceAddress stage_arguments(const void* args, std::size_t size);
    // A region that stays reserved until release_staged(id); see
    // noorrhi::StagedArguments. Takes mutex_.
    vk::DeviceAddress stage_held(const void* args, std::size_t size, std::uint64_t& id);
    void release_staged(std::uint64_t id) noexcept;
    MemoryReport memory_report() const;
    // Root arguments are one 8-byte device address in a Vulkan push constant.
    void push_root(vk::CommandBuffer, vk::DeviceAddress root) const;
    // Bind the global bindless descriptor set used by every pipeline.
    void bind_heaps(vk::CommandBuffer) const;
    static vk::PipelineCreateFlags2CreateInfo pipeline_heap_flags(const void* next = nullptr) {
        return vk::PipelineCreateFlags2CreateInfo{
            vk::PipelineCreateFlagBits2::eDescriptorHeapEXT, next};
    }
    std::uint32_t write_image_descriptor(const ImageImpl&, vk::DescriptorType);
    std::shared_ptr<ImageImpl> find_image(ImageHandle handle) const;
    void render(const RenderTarget&, const std::function<void()>&);
    // Shared prologue for every in-render draw: validates that the pipeline
    // matches the active render target, binds it, and sets viewport/scissor.
    void begin_draw(const GraphicsPipelineImpl&);
    void record_draw(const GraphicsPipelineImpl&, std::uint32_t, std::uint32_t,
        const void*, std::size_t);
    void record_draw_indirect(const GraphicsPipelineImpl&, GpuPtr<DrawArgs>,
        std::uint32_t, const void*, std::size_t);
    void record_compute(const ComputePipelineImpl&, vk::CommandBuffer, DispatchSize,
        const void*, std::size_t);
    void record_ray_tracing(const RayTracingPipelineImpl&,
        const vk::StridedDeviceAddressRegionKHR& raygen, vk::CommandBuffer,
        DispatchSize, const void*, std::size_t);

    // --- Timestamps
    std::shared_ptr<TimestampQuery::State> create_timestamp();
    void measure(const std::shared_ptr<TimestampQuery::State>&,
        const std::function<void()>&);
    double timestamp_milliseconds(TimestampQuery::State&);

    // --- Debug labels
    void label(std::string_view name, const std::function<void()>&);
    void begin_label(vk::CommandBuffer, const char* name) const;
    void end_label(vk::CommandBuffer) const;

    // --- Presentation
    bool presenting() const noexcept { return presentation_enabled_; }
    std::shared_ptr<SwapchainImpl> create_swapchain(SurfaceProvider&, const SwapchainDesc&);
    // Register a presentation-engine image as an ordinary ImageImpl so it can
    // be used as a RenderTarget and a copy destination like any other image.
    std::shared_ptr<ImageImpl> wrap_presentation_image(vk::Image, vk::Format,
        ImageFormat, std::uint32_t width, std::uint32_t height);
    void rebuild_swapchain(SwapchainImpl&);
    // Hands the chain's current Vulkan objects to the retire queue. They are
    // destroyed once every submission that could use them has completed.
    // Never takes mutex_, so it is safe from destructors.
    void retire_chain(SwapchainImpl&);
    // Blocks until the chain has fewer than max_frames_in_flight frames still
    // running on the GPU. Must be called without mutex_ held.
    bool wait_frame_slot(SwapchainImpl&, bool block = true);
    std::shared_ptr<Frame::State> begin_frame(const std::shared_ptr<SwapchainImpl>&, bool wait_for_slot = true);
    void end_frame(Frame::State&);
    // Discard a frame that was never ended: release its command buffer and
    // referenced resources without submitting or presenting.
    void abandon_frame(Frame::State&);
    interop::DeviceHandles native_handles() const noexcept;
    interop::ExternalImageMemory export_image_memory(ImageHandle);
    interop::ExternalSemaphore signal_external();
    // interop::record: native commands batched like any other submission.
    void record_native(const std::function<void(std::uintptr_t)>& commands);
    vk::UniquePipeline create_ray_tracing_pipeline(const vk::RayTracingPipelineCreateInfoKHR&,
        vk::PipelineCache);
    std::shared_ptr<RayTracingLibraryImpl> create_ray_tracing_library(
        const RayTracingPipelineDesc&, const RayTracingInterface&);
    std::shared_ptr<RayTracingPipelineImpl> link_ray_tracing(
        std::span<const std::shared_ptr<RayTracingLibraryImpl>>,
        std::span<const std::uint32_t> hit_groups);
    std::shared_ptr<RayTracingPipelineImpl> rebind_hit_groups(
        const RayTracingPipelineImpl& linked, std::span<const std::uint32_t> hit_groups);
    RayTracingGroups ray_tracing_groups(const RayTracingPipelineDesc&) const;
    // Creates a pipeline with the persistent file-backed VkPipelineCache when
    // requested. Imported material libraries pass use_cache=false so their
    // runtime-generated pipelines never enter pipeline.cache.
    template <class CreateInfo>
    vk::UniquePipeline create_pipeline(const CreateInfo& info, bool use_cache,
        const std::function<vk::UniquePipeline(const CreateInfo&, vk::PipelineCache)>& create) {
        if (!use_cache)
            return create(info, {});
        if (!pipeline_cache_)
            return create(info, {});
        // Vulkan requires external synchronization for concurrent access to a
        // shared VkPipelineCache.
        std::lock_guard lock(pipeline_creation_mutex_);
        return create(info, pipeline_cache());
    }
    // The persisted file cache, or null when caching is disabled.
    vk::PipelineCache pipeline_cache() const noexcept {
        return pipeline_cache_ ? pipeline_cache_->handle() : vk::PipelineCache{};
    }
    void retain_linked_pipeline(std::shared_ptr<const vk::UniquePipeline> pipeline,
        std::vector<std::shared_ptr<RayTracingLibraryImpl>> libraries);
    // Publishes the shader binding table of a created pipeline whose groups,
    // in group-index order, are `groups`. The hit region holds one record per
    // entry of `hit_groups`, each naming a hit group by its index among them.
    void build_shader_binding_table(RayTracingPipelineImpl&,
        std::span<const RayTracingGroups::Group> groups,
        std::span<const std::uint32_t> hit_groups);
    AccelerationStructure build_blas(std::span<const TriangleGeometry>, AccelerationStructureBuildMode);
    AccelerationStructure build_tlas(std::span<const Instance>);
    void refit_blas(AccelerationStructure&);
    void update_tlas(AccelerationStructure&, std::span<const Instance>);
    AccelerationStructure build_tlas(GpuPtr<InstanceRecord>, std::uint32_t,
        std::span<const AccelerationStructure>);
    void update_tlas(AccelerationStructure&, GpuPtr<InstanceRecord>, std::uint32_t);
    // Shared tail of both build paths: everything after the instance records
    // exist at a device address. owned_input is retained for the host-span
    // overload, whose staging buffer the structure keeps for later refits.
    AccelerationStructure build_tlas_at(vk::DeviceAddress records, std::uint32_t count,
        std::vector<std::shared_ptr<AccelerationStructureImpl>> references,
        std::shared_ptr<BufferImpl> owned_input);
    void update_tlas_at(AccelerationStructure&, vk::DeviceAddress records,
        std::uint32_t count);


private:
    friend class ComputePipelineImpl;
    friend class GraphicsPipelineImpl;
    friend class RayTracingPipelineImpl;
    friend struct BufferImpl;
    friend struct ImageImpl;
    friend struct SamplerImpl;
    friend struct AccelerationStructureImpl;
    // Vulkan requires a command pool to be externally synchronized while any
    // of its buffers records, so every thread records from pools of its own.
    // Buffers whose submission completed return to `free` (under mutex_) and
    // are reused, and reset, only by the owning thread.
    struct CommandPool {
        vk::UniqueCommandPool pool;
        std::vector<vk::CommandBuffer> free;
    };
    struct Pending {
        GpuToken token;
        vk::CommandBuffer command;
        CommandPool* pool = nullptr;
        std::vector<std::shared_ptr<void>> resources;
    };
    // `size` bytes of launch records in a queue's argument ring.
    struct ArgumentRegion { std::size_t begin = 0, end = 0; Timelines timelines{}; std::uint64_t held = 0; };
    // One Vulkan queue and the submissions made to it. Both entries may name
    // the same VkQueue; each still keeps its own timeline.
    struct QueueState {
        vk::Queue queue;
        std::uint32_t family = 0;
        vk::UniqueSemaphore timeline;
        // Read without mutex_ by retire(), which destructors call.
        std::atomic<std::uint64_t> next_timeline = 1;
        std::deque<Pending> pending;
        std::uint32_t timestamp_valid_bits = 64;
        // Values of the other queues' timelines the next submission waits for.
        Timelines waits{};
        // The one open recording on this queue, if any. Only its thread
        // records into `recording` or appends to `recording_resources`; the
        // value the submission will signal is reserved when it opens.
        vk::CommandBuffer recording;
        CommandPool* recording_pool = nullptr;
        std::atomic<std::uint64_t> recording_value = 0;
        std::thread::id recording_thread;
        std::vector<std::shared_ptr<void>> recording_resources;
        // Launch records of this queue's submissions only, so a queue running
        // behind never makes another queue's recording wait for it.
        std::shared_ptr<BufferImpl> argument_arena;
        std::size_t argument_offset = 0;
        std::deque<ArgumentRegion> argument_pending;
        std::mutex argument_mutex;
    };
    // The semaphore waits a submission to `queue` takes on; clears them.
    struct QueueWaits {
        std::vector<vk::Semaphore> semaphores;
        std::vector<std::uint64_t> values;
        std::vector<vk::PipelineStageFlags> stages;
    };
    QueueWaits take_queue_waits(QueueState& queue);
    // The calling thread's pool for `queue`. Called with mutex_ held.
    CommandPool& thread_command_pool(Queue queue);
    vk::CommandBuffer allocate_command(CommandPool&);
    // Submits a finished command buffer to `queue`, signalling `token` and
    // any extra binary semaphores, after `extra_waits` and the queue_wait()
    // values. Called with mutex_ held.
    void submit_command(QueueState& queue, GpuToken token, vk::CommandBuffer command, CommandPool* pool,
        std::vector<std::shared_ptr<void>> resources, const QueueWaits& extra_waits = {},
        std::span<const vk::Semaphore> extra_signals = {});
    // A resource destroyed while the GPU may still be reading it is retired
    // here and released once every queue passes the value it was retired at.
    struct Retired {
        Timelines timelines{};
        std::function<void()> release;
    };
    QueueState& queue_state(Queue queue) { return queues_[static_cast<std::size_t>(queue)]; }
    const QueueState& queue_state(Queue queue) const { return queues_[static_cast<std::size_t>(queue)]; }
    std::uint64_t completed_value(Queue queue) const;
    bool completed(const Timelines& timelines) const;
    // For each queue, the submission that ends the work recorded so far: the
    // open frame on Graphics, the last submission otherwise.
    Timelines recorded_timelines() const;
    // The submission being recorded on the calling thread's queue.
    Timelines recording_timelines() const;

    struct Capabilities {
        bool mandatory = false;   // BDA, timeline, sync2, dynamic rendering,
                                  // descriptor heaps, untyped pointers,
                                  // maintenance5
        bool acceleration_structure = false;
        bool ray_query = false;
        bool ray_tracing = false;
    };
    static Capabilities probe(vk::PhysicalDevice);
    void adopt_capabilities(const Capabilities&);

    void create_instance(const DeviceConfig& config);
    void create_probe_surface(const DeviceConfig& config);
    void select_physical_device();
    void create_device(const DeviceConfig& config);
    void create_allocator();
    void create_command_state();
    void create_descriptor_heaps();
    void create_heap(DescriptorHeap&, std::uint32_t capacity, vk::DeviceSize descriptor_size,
        vk::DeviceSize descriptor_alignment, vk::DeviceSize heap_alignment,
        vk::DeviceSize reserved_size, vk::DeviceSize max_size, const char* capacity_name);
    static vk::BindHeapInfoEXT heap_bind_info(const DescriptorHeap&);
    // Slots are taken on resource creation and returned from the resource's
    // retire callback, so a slot is never reused while in-flight work may
    // still read it.
    std::uint32_t allocate_slot(DescriptorHeap&);
    void release_slot(DescriptorHeap&, std::uint32_t slot) noexcept;
    vk::HostAddressRangeEXT slot_range(const DescriptorHeap&, std::uint32_t slot) const;
    void flush_slot(const DescriptorHeap&, std::uint32_t slot) const;
    void reap_completed();
public:
    // Called from resource destructors: defers the Vulkan/VMA release until
    // every submission that could reference the resource has completed.
    void retire(std::function<void()> release);
private:
    // Serialized work is ordered against everything recorded before and after
    // it. Inside an open frame, Explicit work (dispatches, traces, barriers)
    // is ordered only by the caller's barrier() calls, so independent passes
    // can overlap on the GPU. A standalone submission is always serialized.
    enum class Ordering { Serialized, Explicit };
    GpuToken submit(const std::function<void(vk::CommandBuffer)>& record,
                    std::vector<std::shared_ptr<void>> resources = {},
                    Ordering ordering = Ordering::Serialized);
    static vk::PipelineStageFlags2 stage_mask(Stage stage);
    vk::Instance vk_instance() const noexcept { return instance_.get(); }
    vk::Device vk_device() const noexcept { return device_.get(); }

    std::weak_ptr<DeviceImpl> self_;
    DeviceFeatures features_{};
    vk::UniqueInstance instance_;
    vk::UniqueDebugUtilsMessengerEXT messenger_;
    vk::PhysicalDevice physical_device_;
    vk::UniqueDevice device_;
    std::array<QueueState, queue_count> queues_;
    bool presentation_enabled_ = false;
    bool fifo_latest_ready_enabled_ = false;
    // Exists only during construction, to select a GPU and queue that can
    // present to the host's window. Swapchains create their own surfaces.
    vk::UniqueSurfaceKHR probe_surface_;
    // Keyed by thread and queue; see CommandPool. Guarded by mutex_.
    std::map<std::pair<std::thread::id, Queue>, std::unique_ptr<CommandPool>> command_pools_;
    bool debug_labels_supported_ = false;
    bool acceleration_structure_supported_ = false;
    bool ray_query_supported_ = false;
    bool ray_tracing_supported_ = false;
    bool external_memory_fd_enabled_ = false;
    bool external_semaphore_fd_enabled_ = false;
    bool nvidia_610_driver_ = false;
    bool pipeline_binaries_enabled_ = false;
    VmaAllocator allocator_ = VK_NULL_HANDLE;

    // File-backed cache used by ordinary/static pipelines. Runtime material
    // libraries and links explicitly bypass it.
    std::optional<PipelineCacheFile> pipeline_cache_;
    std::mutex pipeline_creation_mutex_;
    // NVIDIA 610 may keep compiling a linked pipeline after the create call
    // returns. Retain it until device shutdown; an elapsed-time grace period
    // cannot establish when the driver's background work is finished.
    struct RetainedLinkedPipeline {
        std::shared_ptr<const vk::UniquePipeline> pipeline;
        std::vector<std::shared_ptr<RayTracingLibraryImpl>> libraries;
    };
    std::mutex linked_pipelines_mutex_;
    std::vector<RetainedLinkedPipeline> retained_linked_pipelines_;
    vk::UniqueQueryPool timestamp_query_pool_;
    std::uint32_t next_timestamp_query_ = 0;
    float timestamp_period_ns_ = 1.0f;
    std::size_t argument_arena_size_ = 0;
    std::size_t host_alignment_ = 16;
    // Places a record in the calling thread's queue's ring. `held` is the
    // StagedArguments id while its owner lives, and 0 once the region only
    // waits for its submissions to retire.
    vk::DeviceAddress place_arguments(const void* args, std::size_t size, std::uint64_t held);
    vk::PhysicalDeviceDescriptorHeapPropertiesEXT heap_properties_{};
    std::uint32_t texture_descriptor_capacity_ = 0;
    std::uint32_t sampler_descriptor_capacity_ = 0;
    DescriptorHeap texture_heap_;
    DescriptorHeap sampler_heap_;
    // Guarded by its own mutex for the same reason as retire_mutex_: slots are
    // released from retire callbacks.
    mutable std::mutex heap_mutex_;
    vk::PhysicalDeviceAccelerationStructurePropertiesKHR acceleration_structure_properties_{};
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties_{};
    std::atomic<std::uint64_t> next_staged_id_ = 1;
    std::map<vk::DeviceAddress, std::weak_ptr<BufferImpl>> buffers_;
    // Recording threads resolve addresses while others create buffers.
    mutable std::mutex buffers_mutex_;
    // Guarded by its own mutex: retiring happens inside resource destructors,
    // which can run while mutex_ is already held (a completed submission
    // dropping its last reference to a buffer, for instance).
    // In submission order, so every queue's timeline is non-decreasing along
    // it: the completed entries are always a prefix.
    std::deque<Retired> retired_;
    // Dead entries of buffers_ are swept once it doubles, which keeps the
    // sweep amortized constant per buffer instead of linear per submission.
    std::size_t buffer_sweep_size_ = 64;
    std::mutex retire_mutex_;
    // What the calling thread is recording. submit() runs its callback
    // without mutex_, so every recording thread keeps its own render scope
    // and the resources its commands reference.
    static thread_local vk::CommandBuffer active_command_;
    static thread_local vk::Format active_color_format_;
    static thread_local bool active_has_depth_;
    static thread_local bool active_flip_y_;
    static thread_local std::uint32_t active_width_;
    static thread_local std::uint32_t active_height_;
    static thread_local std::vector<std::shared_ptr<void>> active_resources_;
    bool shut_down_ = false;
    // Guards the queues' submission state, the command pools and presentation.
    // Never held while commands are recorded or while waiting for the GPU.
    mutable std::mutex mutex_;
};

std::shared_ptr<BufferImpl> make_buffer(const std::shared_ptr<DeviceImpl>&, std::size_t, std::size_t);
void upload_buffer(const std::shared_ptr<BufferImpl>&, const void*, std::size_t, std::size_t);
void download_buffer(const std::shared_ptr<BufferImpl>&, void*, std::size_t, std::size_t);
std::uint64_t buffer_address(const std::shared_ptr<BufferImpl>&);
std::shared_ptr<ImageImpl> make_image(const std::shared_ptr<DeviceImpl>&, std::uint32_t,
                                      std::uint32_t, ImageUsage, ImageFormat, std::uint32_t);
std::uint32_t image_sampled_handle(const std::shared_ptr<ImageImpl>&);
std::uint32_t image_storage_handle(const std::shared_ptr<ImageImpl>&);
AccelerationStructureHandle acceleration_structure_handle(
    const std::shared_ptr<AccelerationStructureImpl>&);

} // namespace noorrhi::detail
