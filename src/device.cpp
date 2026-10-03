#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

#include "noorrhi/noorrhi.hpp"
#include "noorrhi/interop.hpp"
#include "internal.hpp"
#include "noorrhi/shared.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <cstring>
#include <limits>
#include <set>
#include <thread>
#include <utility>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace noorrhi::detail {

namespace {
[[noreturn]] void throw_vk(const vk::SystemError& error, const char* operation) {
    throw Error(ErrorCode::InvalidState, std::string(operation) + ": " + error.what());
}

vk::Bool32 VKAPI_CALL debug_callback(
    const vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT,
    const vk::DebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (data && data->pMessage) {
        const char* label =
            severity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ? "error"
            : severity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning ? "warning"
            : "info";
        std::fprintf(stderr, "[NoorRHI validation %s] %s\n", label, data->pMessage);
    }
    return vk::False;
}

bool has_extension(const std::vector<vk::ExtensionProperties>& extensions, const char* name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const auto& extension) {
        return std::string_view(extension.extensionName) == name;
    });
}
}

DeviceImpl::DeviceImpl(const DeviceConfig& config) {
    presentation_enabled_ = config.presentation != nullptr;
    // Root arguments use a fixed mapped arena; large assets use temporary staging.
    argument_arena_size_ = config.argument_arena_bytes;
    texture_descriptor_capacity_ = config.texture_descriptor_capacity;
    sampler_descriptor_capacity_ = config.sampler_descriptor_capacity;
    // Slot 0 is reserved, so a usable heap needs at least two.
    if (texture_descriptor_capacity_ < 2 || sampler_descriptor_capacity_ < 2)
        throw Error(ErrorCode::InvalidArgument,
            "descriptor heap capacities must be at least 2 (slot 0 is reserved)");
    try {
        create_instance(config);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_instance());
        // A probe surface has to exist before the physical device is chosen:
        // a device that cannot present to the host's window is not a
        // candidate. It is released once the queue has been selected.
        create_probe_surface(config);
        select_physical_device();
        host_alignment_ = std::max<std::size_t>(16,
            physical_device_.getProperties().limits.nonCoherentAtomSize);
        if (argument_arena_size_ < host_alignment_ || argument_arena_size_ % host_alignment_)
            throw Error(ErrorCode::InvalidArgument,
                "argument chunk size must be a multiple of the device host alignment");
        create_device(config);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_device());
        // Only now does the dispatcher hold the device's own functions.
        // Keep ordinary/static pipelines in one Vulkan pipeline cache persisted
        // to pipeline.cache. Material ray-tracing libraries opt out explicitly,
        // so large imported material sets never grow this startup cache.
        if (!config.pipeline_directory.empty()
            && std::getenv("NOORRHI_DISABLE_PIPELINE_CACHE") == nullptr) {
            pipeline_cache_.emplace(vk_device(), config.pipeline_directory);
        } else if (!config.pipeline_directory.empty()) {
            std::fprintf(stderr, "[NoorRHI] pipeline cache disabled by NOORRHI_DISABLE_PIPELINE_CACHE\n");
        }
        create_allocator();
        create_command_state();
        probe_surface_.reset();
    } catch (const vk::SystemError& error) {
        throw_vk(error, "Vulkan device initialization failed");
    }
}

// The mandatory set is what the whole library is built on: buffer device
// addresses for every buffer, a timeline semaphore for all synchronization,
// synchronization2 and dynamic rendering for command recording, descriptor
// heaps for textures and samplers. Every image lives in GENERAL, which core
// Vulkan accepts for every usage this library records.
//
// VK_EXT_descriptor_heap is what keeps pipelines layout-free: images and
// samplers are encoded into device-owned heaps that shaders index directly
// (SPV_EXT_descriptor_heap), and root pointers travel as push data. Slang
// lowers heap access through untyped pointers, and the pipeline flag that
// opts into heaps lives in maintenance5's flags2 field, so both are required
// alongside it.
DeviceImpl::Capabilities DeviceImpl::probe(const vk::PhysicalDevice candidate) {
    Capabilities capabilities{};
    const auto properties = candidate.getProperties();
    if (properties.apiVersion < VK_API_VERSION_1_3)
        return capabilities;

    const auto extensions = candidate.enumerateDeviceExtensionProperties();
    const bool has_descriptor_heap = has_extension(extensions,
        VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
    const bool has_untyped_pointers = has_extension(extensions,
        VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
    const bool has_maintenance5 = properties.apiVersion >= VK_API_VERSION_1_4
        || has_extension(extensions, VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    const bool has_as = has_extension(extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
        && has_extension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    const bool has_query = has_as && has_extension(extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME);
    const bool has_pipeline = has_as
        && has_extension(extensions, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
        && has_extension(extensions, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);

    vk::PhysicalDeviceVulkan11Features supported11{};
    vk::PhysicalDeviceVulkan12Features supported12{};
    vk::PhysicalDeviceVulkan13Features supported13{};
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR supported_as{};
    vk::PhysicalDeviceRayQueryFeaturesKHR supported_query{};
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR supported_rt{};
    vk::PhysicalDeviceDescriptorHeapFeaturesEXT supported_heap{};
    vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR supported_untyped{};
    vk::PhysicalDeviceMaintenance5FeaturesKHR supported_maintenance5{};
    vk::PhysicalDeviceFeatures2 supported{};
    // Chain unconditionally: querying a feature struct whose extension is
    // absent simply reports it unsupported.
    supported.pNext = &supported11;
    supported11.pNext = &supported12;
    supported12.pNext = &supported13;
    supported13.pNext = &supported_as;
    supported_as.pNext = &supported_query;
    supported_query.pNext = &supported_rt;
    supported_rt.pNext = &supported_heap;
    supported_heap.pNext = &supported_untyped;
    supported_untyped.pNext = &supported_maintenance5;
    candidate.getFeatures2(&supported);

    bool has_compute = false;
    for (const auto& queue : candidate.getQueueFamilyProperties())
        has_compute |= static_cast<bool>(queue.queueFlags & vk::QueueFlagBits::eCompute);

    capabilities.mandatory = has_compute
        && supported12.bufferDeviceAddress && supported12.timelineSemaphore
        && supported13.synchronization2 && supported13.dynamicRendering;
    capabilities.descriptor_heap = supported.features.shaderInt64 && supported11.shaderDrawParameters
        && supported12.scalarBlockLayout
        && has_descriptor_heap && supported_heap.descriptorHeap
        && has_untyped_pointers && supported_untyped.shaderUntypedPointers
        && has_maintenance5 && supported_maintenance5.maintenance5;
    capabilities.acceleration_structure = has_as && supported_as.accelerationStructure;
    capabilities.ray_query = has_query && capabilities.acceleration_structure
        && supported_query.rayQuery;
    capabilities.ray_tracing = has_pipeline && capabilities.acceleration_structure
        && supported_rt.rayTracingPipeline;
    return capabilities;
}

void DeviceImpl::adopt_capabilities(const Capabilities& capabilities) {
    acceleration_structure_supported_ = capabilities.acceleration_structure;
    ray_query_supported_ = capabilities.ray_query;
    ray_tracing_supported_ = capabilities.ray_tracing;
    descriptor_heap_supported_ = capabilities.descriptor_heap;
    features_.descriptor_heap = descriptor_heap_supported_;
    features_.ray_query = ray_query_supported_;
    features_.ray_tracing = ray_tracing_supported_;
}

std::shared_ptr<BufferImpl> DeviceImpl::create_argument_arena(const std::size_t size) {
    return create_buffer(size,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_MEMORY_USAGE_CPU_TO_GPU, true);
}

void DeviceImpl::initialize_resources() {
    query_properties();
    if (descriptor_heap_supported_)
        create_descriptor_heaps();
    for (QueueState& queue : queues_)
        queue.argument_chunks.push_back({create_argument_arena(argument_arena_size_)});
}

MemoryReport DeviceImpl::memory_report() const {
    VmaTotalStatistics statistics{};
    vmaCalculateStatistics(allocator_, &statistics);
    std::uint64_t argument_bytes = 0;
    for (const QueueState& queue : queues_) {
        std::lock_guard argument_lock(queue.argument_mutex);
        for (const auto* chunks : {&queue.argument_chunks, &queue.argument_dedicated})
            for (const ArgumentChunk& chunk : *chunks)
                argument_bytes += chunk.arena->size;
    }
    return {statistics.total.statistics.allocationBytes, statistics.total.statistics.blockBytes,
        statistics.total.statistics.allocationCount, argument_bytes};
}

void DeviceImpl::create_instance(const DeviceConfig& config) {
    // A windowing toolkit loads its own Vulkan library, and a surface created
    // against a different loader is undefined. When one is present, its
    // loader is the one the whole device uses.
    auto loader = vkGetInstanceProcAddr;
    if (config.presentation) {
        if (const auto provided = config.presentation->instance_proc_address())
            loader = reinterpret_cast<PFN_vkGetInstanceProcAddr>(provided);
    }
    if (!loader)
        throw Error(ErrorCode::InvalidState, "Vulkan loader did not provide vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(loader);

    std::vector<const char*> layers;
    std::vector<const char*> extensions;
    // Platform extensions are supplied by the window provider below.
    if (config.presentation)
        for (const char* extension : config.presentation->instance_extensions())
            extensions.push_back(extension);
    if (config.enable_validation) {
        for (const auto& layer : vk::enumerateInstanceLayerProperties()) {
            if (std::string_view(layer.layerName) == "VK_LAYER_KHRONOS_validation") {
                layers.push_back("VK_LAYER_KHRONOS_validation");
                break;
            }
        }
        if (layers.empty())
            throw Error(ErrorCode::UnsupportedFeature, "Vulkan validation was requested but is unavailable");
    }
    // Debug labels name passes for profilers and debuggers; without the
    // extension label() records nothing.
    for (const auto& extension : vk::enumerateInstanceExtensionProperties()) {
        if (std::string_view(extension.extensionName) == VK_EXT_DEBUG_UTILS_EXTENSION_NAME) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            debug_labels_supported_ = true;
            break;
        }
    }
    if (config.enable_validation && !debug_labels_supported_)
        throw Error(ErrorCode::UnsupportedFeature, "Vulkan validation needs VK_EXT_debug_utils");

    const std::string application_name(config.application_name);
    // 1.4 so that a 1.4 device exposes maintenance5 as core; 1.3 devices
    // still qualify through the extension.
    const vk::ApplicationInfo appInfo(application_name.c_str(), 1, "NoorRHI", 1, VK_API_VERSION_1_4);
    vk::InstanceCreateInfo createInfo{};
    createInfo.setPApplicationInfo(&appInfo)
        .setPEnabledLayerNames(layers)
        .setPEnabledExtensionNames(extensions);
    instance_ = vk::createInstanceUnique(createInfo);

    if (!layers.empty()) {
        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_.get());
        messenger_= instance_->createDebugUtilsMessengerEXTUnique({{},
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
                | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning,
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
                | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
                | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            debug_callback});
    }
}

void DeviceImpl::create_probe_surface(const DeviceConfig& config) {
    if (!config.presentation)
        return;
    const std::uintptr_t raw = config.presentation->create_surface(
        reinterpret_cast<std::uintptr_t>(static_cast<VkInstance>(vk_instance())));
    if (!raw)
        throw Error(ErrorCode::InvalidState, "SurfaceProvider failed to create a surface");
    probe_surface_ = vk::UniqueSurfaceKHR(vk::SurfaceKHR(reinterpret_cast<VkSurfaceKHR>(raw)),
        vk::detail::ObjectDestroy<vk::Instance, VULKAN_HPP_DEFAULT_DISPATCHER_TYPE>(vk_instance()));
}

void DeviceImpl::select_physical_device() {
    const auto devices = vk_instance().enumeratePhysicalDevices();
    if (devices.empty())
        throw Error(ErrorCode::UnsupportedFeature, "No Vulkan physical device is available");

    vk::PhysicalDevice best;
    Capabilities best_capabilities{};
    vk::DeviceSize best_memory = 0;
    bool best_is_discrete = false;
    bool best_is_cpu = false;
    for (const auto candidate : devices) {
        const Capabilities capabilities = probe(candidate);
        if (!capabilities.mandatory)
            continue;
        if (probe_surface_ && !has_extension(candidate.enumerateDeviceExtensionProperties(),
                VK_KHR_SWAPCHAIN_EXTENSION_NAME))
            continue;

        vk::DeviceSize memory = 0;
        const auto memory_properties = candidate.getMemoryProperties();
        for (std::uint32_t i = 0; i < memory_properties.memoryHeapCount; ++i) {
            const auto& heap = memory_properties.memoryHeaps[i];
            if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal)
                memory += heap.size;
        }
        const auto properties = candidate.getProperties();
        const bool is_discrete = properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
        // CPU Vulkan implementations such as lavapipe/llvmpipe can report the
        // host's RAM as device-local memory. Never let that heap size make a
        // software device outrank an eligible GPU; keep it only as a fallback.
        const bool is_cpu = properties.deviceType == vk::PhysicalDeviceType::eCpu;
        if (!best || (best_is_cpu && !is_cpu)
            || (best_is_cpu == is_cpu && ((is_discrete && !best_is_discrete)
                || (is_discrete == best_is_discrete && memory > best_memory)))) {
            best = candidate;
            best_capabilities = capabilities;
            best_memory = memory;
            best_is_discrete = is_discrete;
            best_is_cpu = is_cpu;
        }
    }
    if (!best)
        throw Error(ErrorCode::UnsupportedFeature,
            "No Vulkan 1.3 device supports buffer device address, timeline semaphores, "
            "synchronization2 and dynamic rendering");
    physical_device_ = best;
    adopt_capabilities(best_capabilities);
}

void DeviceImpl::create_device(const DeviceConfig& config) {
    const auto queues = physical_device_.getQueueFamilyProperties();
    std::optional<std::uint32_t> selected;
    for (std::uint32_t i = 0; i < queues.size(); ++i) {
        if (!(queues[i].queueFlags & vk::QueueFlagBits::eCompute))
            continue;
        // A presenting device needs one queue that can do everything: the
        // frame's dispatches, its draws and its present all go through the
        // single queue this library owns.
        if (probe_surface_ && !physical_device_.getSurfaceSupportKHR(i, probe_surface_.get()))
            continue;
        // Prefer a unified graphics+compute queue. The public API exposes
        // both dispatch and raster operations, so selecting a compute-only
        // queue makes Device::render invalid even though compute works.
        if (queues[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            selected = i;
            break;
        }
        if (!selected)
            selected = i;
    }
    if (!selected)
        throw Error(ErrorCode::UnsupportedFeature, probe_surface_
            ? "Selected Vulkan device has no queue that supports both compute and presentation"
            : "Selected Vulkan device has no compute queue");
    QueueState& graphics = queue_state(Queue::Graphics);
    QueueState& async = queue_state(Queue::Async);
    graphics.family = *selected;
    async.family = graphics.family;
    // Async is the graphics family's second queue, not a compute-only family,
    // because the work it carries includes raster passes and blits.
    const std::uint32_t async_index = queues[graphics.family].queueCount > 1 ? 1 : 0;
    // Graphics carries the frames the user waits on, so it takes precedence
    // where the driver honours queue priorities.
    constexpr std::array priorities{1.0f, 0.5f};
    const std::vector<vk::DeviceQueueCreateInfo> queue_infos{
        {{}, graphics.family, async_index + 1, priorities.data()}};

    const auto extensions = physical_device_.enumerateDeviceExtensionProperties();
    std::vector<const char*> enabled_extensions;
    if (descriptor_heap_supported_) {
        enabled_extensions.push_back(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
        enabled_extensions.push_back(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
        if (physical_device_.getProperties().apiVersion < VK_API_VERSION_1_4)
            enabled_extensions.push_back(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    }
    if (presentation_enabled_)
        enabled_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    external_memory_fd_enabled_ = has_extension(extensions,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    if (external_memory_fd_enabled_)
        enabled_extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    external_semaphore_fd_enabled_ = has_extension(extensions,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
    if (external_semaphore_fd_enabled_)
        enabled_extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);

    const vk::PhysicalDeviceFeatures supported_base = physical_device_.getFeatures();
    vk::PhysicalDeviceFeatures base{};
    base.shaderInt64 = descriptor_heap_supported_;
    // RTXDI's path tracer declares Int16; without the feature the module is
    // invalid and the driver's results undefined.
    if (supported_base.shaderInt16)
        base.shaderInt16 = VK_TRUE;

    // shaderDrawParameters is declared by Slang-compiled vertex shaders; the
    // validation layers reject the module without it.
    vk::PhysicalDeviceVulkan11Features features11{};
    features11.shaderDrawParameters = descriptor_heap_supported_;
    vk::PhysicalDeviceVulkan12Features features12{};
    // Slang declares Float16 for shared records containing half-precision
    // Gaussian coefficients, including shaders that do not load them.
    const auto supported_numeric = physical_device_.getFeatures2<
        vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceVulkan12Features>();
    features11.storageBuffer16BitAccess = supported_numeric
        .get<vk::PhysicalDeviceVulkan11Features>().storageBuffer16BitAccess;
    features12.shaderFloat16 = supported_numeric
        .get<vk::PhysicalDeviceVulkan12Features>().shaderFloat16;
    // Atomic compare-exchange on 64-bit keys in storage buffers, used by
    // lock-free GPU hash tables.
    features12.shaderBufferInt64Atomics = supported_numeric
        .get<vk::PhysicalDeviceVulkan12Features>().shaderBufferInt64Atomics;
    features12.bufferDeviceAddress = VK_TRUE;
    features12.timelineSemaphore = VK_TRUE;
    // Slang's natural layout for records reached through GPU pointers matches
    // C++ struct packing (vec3 followed by a scalar, for instance), which is
    // only legal SPIR-V under scalar block layout.
    features12.scalarBlockLayout = descriptor_heap_supported_;
    vk::PhysicalDeviceVulkan13Features features13{};
    features13.synchronization2 = VK_TRUE;
    features13.dynamicRendering = VK_TRUE;
    features11.pNext = &features12;
    features12.pNext = &features13;
    void** tail = &features13.pNext;
    vk::PhysicalDeviceDescriptorHeapFeaturesEXT heap_features{};
    vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR untyped_pointers{};
    vk::PhysicalDeviceMaintenance5FeaturesKHR maintenance5{};
    if (descriptor_heap_supported_) {
        heap_features.descriptorHeap = VK_TRUE;
        untyped_pointers.shaderUntypedPointers = VK_TRUE;
        maintenance5.maintenance5 = VK_TRUE;
        *tail = &heap_features;
        heap_features.pNext = &untyped_pointers;
        untyped_pointers.pNext = &maintenance5;
        tail = &maintenance5.pNext;
    }

    vk::PhysicalDevicePipelineBinaryFeaturesKHR pipeline_binary_features{};
    pipeline_binaries_enabled_ =
        has_extension(extensions, VK_KHR_PIPELINE_BINARY_EXTENSION_NAME)
        && physical_device_.getFeatures2<vk::PhysicalDeviceFeatures2,
               vk::PhysicalDevicePipelineBinaryFeaturesKHR>()
               .get<vk::PhysicalDevicePipelineBinaryFeaturesKHR>().pipelineBinaries;
    if (pipeline_binaries_enabled_) {
        pipeline_binary_features.pipelineBinaries = VK_TRUE;
        *tail = &pipeline_binary_features;
        tail = &pipeline_binary_features.pNext;
        enabled_extensions.push_back(VK_KHR_PIPELINE_BINARY_EXTENSION_NAME);
    }

    vk::PhysicalDeviceAccelerationStructureFeaturesKHR as_features{};
    vk::PhysicalDeviceRayQueryFeaturesKHR query_features{};
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rt_features{};
    if (acceleration_structure_supported_) {
        as_features.accelerationStructure = VK_TRUE;
        *tail = &as_features;
        tail = &as_features.pNext;
        enabled_extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        enabled_extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        if (ray_query_supported_) {
            query_features.rayQuery = VK_TRUE;
            *tail = &query_features;
            tail = &query_features.pNext;
            enabled_extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        }
        if (ray_tracing_supported_) {
            rt_features.rayTracingPipeline = VK_TRUE;
            *tail = &rt_features;
            tail = &rt_features.pNext;
            enabled_extensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
            enabled_extensions.push_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
        }
    }

    // FIFO_LATEST_READY is the tear-free low-latency fallback where a surface
    // offers no MAILBOX. Optional: swapchains simply skip it without support.
    vk::PhysicalDevicePresentModeFifoLatestReadyFeaturesKHR latest_ready_features{};
    if (presentation_enabled_) {
        const char* latest_ready_extension =
            has_extension(extensions, VK_KHR_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME)
                ? VK_KHR_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME
            : has_extension(extensions, VK_EXT_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME)
                ? VK_EXT_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME
                : nullptr;
        if (latest_ready_extension) {
            const auto supported = physical_device_.getFeatures2<vk::PhysicalDeviceFeatures2,
                vk::PhysicalDevicePresentModeFifoLatestReadyFeaturesKHR>();
            if (supported.get<vk::PhysicalDevicePresentModeFifoLatestReadyFeaturesKHR>()
                    .presentModeFifoLatestReady) {
                latest_ready_features.presentModeFifoLatestReady = VK_TRUE;
                *tail = &latest_ready_features;
                tail = &latest_ready_features.pNext;
                enabled_extensions.push_back(latest_ready_extension);
                fifo_latest_ready_enabled_ = true;
            }
        }
    }

    vk::DeviceCreateInfo createInfo{};
    createInfo.setPEnabledFeatures(&base)
        .setQueueCreateInfos(queue_infos)
        .setPEnabledExtensionNames(enabled_extensions)
        .setPNext(&features11);
    device_ = physical_device_.createDeviceUnique(createInfo);
    graphics.queue = vk_device().getQueue(graphics.family, 0);
    async.queue = vk_device().getQueue(async.family, async_index);
}

void DeviceImpl::create_allocator() {
    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo info{};
    info.instance = vk_instance();
    info.physicalDevice = physical_device_;
    info.device = vk_device();
    info.vulkanApiVersion = VK_API_VERSION_1_3;
    info.pVulkanFunctions = &functions;
    info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    if (vmaCreateAllocator(&info, &allocator_) != VK_SUCCESS)
        throw Error(ErrorCode::OutOfMemory, "VMA allocator creation failed");
}

void DeviceImpl::create_command_state() {
    const auto queue_properties = physical_device_.getQueueFamilyProperties();
    for (QueueState& queue : queues_) {
        vk::SemaphoreTypeCreateInfo timelineInfo(vk::SemaphoreType::eTimeline, 0);
        queue.timeline = vk_device().createSemaphoreUnique({{}, &timelineInfo});
        queue.timestamp_valid_bits = queue_properties[queue.family].timestampValidBits;
    }
    timestamp_query_pool_ = vk_device().createQueryPoolUnique(
        {{}, vk::QueryType::eTimestamp, 256});
    timestamp_period_ns_ = physical_device_.getProperties().limits.timestampPeriod;
}

std::shared_ptr<TimestampQuery::State> DeviceImpl::create_timestamp() {
    std::lock_guard lock(mutex_);
    if (!timestamp_query_pool_ || next_timestamp_query_ + 2u > 256u)
        throw Error(ErrorCode::OutOfMemory, "GPU timestamp query capacity exhausted");
    auto state = std::make_shared<TimestampQuery::State>();
    state->device = self_.lock();
    state->first_query = next_timestamp_query_;
    next_timestamp_query_ += 2u;
    return state;
}

void DeviceImpl::measure(const std::shared_ptr<TimestampQuery::State>& state,
    const std::function<void()>& commands) {
    if (!state || !timestamp_query_pool_ || state->first_query + 1u >= 256u)
        throw Error(ErrorCode::InvalidArgument, "invalid GPU timestamp query");
    const std::uint32_t first = state->first_query;
    state->queue = current_queue();
    // The bracket is recorded into whatever command buffer the enclosed work
    // goes into, so a measured scope inside a frame times only that scope
    // rather than the whole frame.
    submit([first, this](const vk::CommandBuffer command) {
        command.resetQueryPool(*timestamp_query_pool_, first, 2u);
        command.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe,
            *timestamp_query_pool_, first);
    });
    commands();
    submit([first, this](const vk::CommandBuffer command) {
        command.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe,
            *timestamp_query_pool_, first + 1u);
    });
}

// Explicit ordering: a label must not add barriers, or it would change the
// overlap of the passes it names.
void DeviceImpl::label(const std::string_view name, const std::function<void()>& commands) {
    if (!debug_labels_supported_) {
        commands();
        return;
    }
    const std::string text(name);
    submit([this, &text](const vk::CommandBuffer command) {
        begin_label(command, text.c_str());
    }, {}, Ordering::Explicit);
    commands();
    submit([this](const vk::CommandBuffer command) {
        end_label(command);
    }, {}, Ordering::Explicit);
}

void DeviceImpl::begin_label(const vk::CommandBuffer command, const char* name) const {
    if (debug_labels_supported_)
        command.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{name});
}

void DeviceImpl::end_label(const vk::CommandBuffer command) const {
    if (debug_labels_supported_)
        command.endDebugUtilsLabelEXT();
}

double DeviceImpl::timestamp_milliseconds(TimestampQuery::State& query) {
    std::uint64_t values[2]{};
    const VkResult result = vkGetQueryPoolResults(
        static_cast<VkDevice>(vk_device()),
        static_cast<VkQueryPool>(*timestamp_query_pool_), query.first_query, 2u,
        sizeof(values), values, sizeof(values[0]),
        VK_QUERY_RESULT_64_BIT);
    if (result == VK_NOT_READY) return query.milliseconds;
    if (result != VK_SUCCESS)
        throw Error(ErrorCode::InvalidState, "failed to read GPU timestamps");
    std::uint64_t ticks = values[1] - values[0];
    const std::uint32_t valid_bits = queue_state(query.queue).timestamp_valid_bits;
    if (valid_bits != 0u && valid_bits < 64u)
        ticks &= (std::uint64_t{1} << valid_bits) - 1u;
    query.milliseconds = static_cast<double>(ticks) * timestamp_period_ns_ * 1.0e-6;
    return query.milliseconds;
}

namespace {
constexpr vk::DeviceSize align_up(const vk::DeviceSize value, const vk::DeviceSize alignment) {
    const vk::DeviceSize a = std::max<vk::DeviceSize>(alignment, 1);
    return (value + a - 1) / a * a;
}
}

// Every shader in this library takes the same thing: one 8-byte pointer to its
// root argument record as push data, plus whatever textures and samplers it
// indexes out of the two device-owned heaps. No pipeline layout exists.
void DeviceImpl::query_properties() {
    vk::PhysicalDeviceProperties2 properties{};
    void** tail = &properties.pNext;
    if (descriptor_heap_supported_) {
        *tail = &heap_properties_;
        tail = &heap_properties_.pNext;
    }
    if (acceleration_structure_supported_) {
        *tail = &acceleration_structure_properties_;
        acceleration_structure_properties_.pNext = &ray_tracing_properties_;
    }
    physical_device_.getProperties2(&properties);
    heap_properties_.pNext = nullptr;
    acceleration_structure_properties_.pNext = nullptr;
}

void DeviceImpl::require_descriptor_heaps() const {
    if (!descriptor_heap_supported_)
        throw Error(ErrorCode::UnsupportedFeature,
            "this device lacks what pipelines, samplers and shader-visible images need: "
            "VK_EXT_descriptor_heap, VK_KHR_shader_untyped_pointers, maintenance5, shaderInt64, "
            "shaderDrawParameters and scalarBlockLayout");
}

void DeviceImpl::create_descriptor_heaps() {
    if (heap_properties_.maxPushDataSize < sizeof(vk::DeviceAddress))
        throw Error(ErrorCode::UnsupportedFeature,
            "descriptor heap push data cannot hold an 8-byte root pointer");
    create_heap(texture_heap_, texture_descriptor_capacity_,
        heap_properties_.imageDescriptorSize,
        std::max(heap_properties_.imageDescriptorAlignment,
            heap_properties_.bufferDescriptorAlignment),
        heap_properties_.resourceHeapAlignment,
        heap_properties_.minResourceHeapReservedRange,
        heap_properties_.maxResourceHeapSize,
        "DeviceConfig::texture_descriptor_capacity");
    create_heap(sampler_heap_, sampler_descriptor_capacity_,
        heap_properties_.samplerDescriptorSize,
        heap_properties_.samplerDescriptorAlignment,
        heap_properties_.samplerHeapAlignment,
        heap_properties_.minSamplerHeapReservedRange,
        heap_properties_.maxSamplerHeapSize,
        "DeviceConfig::sampler_descriptor_capacity");
}

void DeviceImpl::create_heap(DescriptorHeap& heap, const std::uint32_t capacity,
    const vk::DeviceSize descriptor_size, const vk::DeviceSize descriptor_alignment,
    const vk::DeviceSize heap_alignment, const vk::DeviceSize reserved_size,
    const vk::DeviceSize max_size, const char* capacity_name) {
    if (descriptor_size == 0)
        throw Error(ErrorCode::UnsupportedFeature, "device reports a zero descriptor size");
    const vk::DeviceSize reserved_offset = align_up(capacity * descriptor_size,
        descriptor_alignment);
    const vk::DeviceSize bind_size = reserved_offset + reserved_size;
    if (max_size && bind_size > max_size)
        throw Error(ErrorCode::InvalidArgument,
            std::string(capacity_name) + " exceeds the device's maximum descriptor heap size");
    // VMA does not promise an address aligned to the heap alignment, so
    // over-allocate and bind the heap at an aligned address inside the buffer.
    const vk::DeviceSize alignment = std::max<vk::DeviceSize>(heap_alignment, 1);
    heap.buffer = create_buffer(static_cast<std::size_t>(bind_size + alignment - 1),
        vk::BufferUsageFlagBits::eDescriptorHeapEXT
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_MEMORY_USAGE_CPU_TO_GPU, true);
    heap.address = align_up(heap.buffer->address, alignment);
    heap.buffer_offset = static_cast<std::size_t>(heap.address - heap.buffer->address);
    heap.slots = static_cast<std::byte*>(heap.buffer->mapped) + heap.buffer_offset;
    heap.descriptor_size = descriptor_size;
    heap.reserved_offset = reserved_offset;
    heap.reserved_size = reserved_size;
    heap.capacity = capacity;
    heap.capacity_name = capacity_name;
    // Slot 0 stays zeroed: it is the null handle and is never read.
    std::memset(heap.buffer->mapped, 0, heap.buffer->size);
}

vk::BindHeapInfoEXT DeviceImpl::heap_bind_info(const DescriptorHeap& heap) {
    using Range = decltype(vk::BindHeapInfoEXT::heapRange);
    return vk::BindHeapInfoEXT{Range{heap.address, heap.reserved_offset + heap.reserved_size},
        heap.reserved_offset, heap.reserved_size};
}

void DeviceImpl::bind_heaps(const vk::CommandBuffer command) const {
    command.bindResourceHeapEXT(heap_bind_info(texture_heap_));
    command.bindSamplerHeapEXT(heap_bind_info(sampler_heap_));
}

std::uint32_t DeviceImpl::allocate_slot(DescriptorHeap& heap) {
    std::lock_guard lock(heap_mutex_);
    if (heap.capacity == 0)
        throw Error(ErrorCode::InvalidState, "noorrhi::Device has been shut down");
    if (!heap.free.empty()) {
        const std::uint32_t slot = heap.free.back();
        heap.free.pop_back();
        return slot;
    }
    if (heap.next < heap.capacity)
        return heap.next++;
    throw Error(ErrorCode::OutOfMemory,
        std::string("descriptor heap is full; raise ") + heap.capacity_name);
}

void DeviceImpl::release_slot(DescriptorHeap& heap, const std::uint32_t slot) noexcept {
    if (slot == 0)
        return;
    std::lock_guard lock(heap_mutex_);
    // After shutdown the allocator is inactive and there is nothing to return to.
    if (heap.capacity != 0)
        heap.free.push_back(slot);
}

vk::HostAddressRangeEXT DeviceImpl::slot_range(const DescriptorHeap& heap,
    const std::uint32_t slot) const {
    return vk::HostAddressRangeEXT{heap.slots + slot * heap.descriptor_size,
        static_cast<std::size_t>(heap.descriptor_size)};
}

void DeviceImpl::flush_slot(const DescriptorHeap& heap, const std::uint32_t slot) const {
    if (vmaFlushAllocation(allocator_, heap.buffer->allocation,
            heap.buffer_offset + slot * heap.descriptor_size, heap.descriptor_size) != VK_SUCCESS)
        throw Error(ErrorCode::DeviceLost, "flushing a descriptor heap write failed");
}

void DeviceImpl::retain_active(std::shared_ptr<void> resource) {
    if (resource && active_command_)
        active_resources_.push_back(std::move(resource));
}

void DeviceImpl::shutdown() noexcept {
    if (shut_down_)
        return;
    shut_down_ = true;
    try {
        synchronize();
    } catch (...) {
        // Destructors cannot report device-loss errors.
    }
    active_resources_.clear();
    for (QueueState& queue : queues_) {
        queue.recording = nullptr;
        queue.recording_resources.clear();
        queue.pending.clear();
    }
    // The queue is idle, so everything still deferred can be released.
    std::deque<Retired> retired;
    {
        std::lock_guard retire_lock(retire_mutex_);
        retired.swap(retired_);
    }
    for (auto& entry : retired) {
        if (entry.release)
            entry.release();
    }
    // These internal buffers own VMA allocations and are destroyed before the
    // allocator itself. External resource objects keep DeviceImpl alive, so
    // no user-visible resource should remain here.
    for (QueueState& queue : queues_) {
        queue.argument_chunks.clear();
        queue.argument_dedicated.clear();
    }
    {
        std::lock_guard lock(mutex_);
        command_pools_.clear();
    }
    std::lock_guard heap_lock(heap_mutex_);
    texture_heap_ = {};
    sampler_heap_ = {};
}

DeviceImpl::~DeviceImpl() {
    shutdown();
    if (allocator_)
        vmaDestroyAllocator(allocator_);
}

std::shared_ptr<BufferImpl> DeviceImpl::create_buffer(const std::size_t size,
    const vk::BufferUsageFlags usage, const VmaMemoryUsage memory_usage, const bool mapped, const std::size_t alignment) {
    if (size == 0)
        throw Error(ErrorCode::InvalidArgument, "GPU buffers cannot have zero bytes");
    auto result = std::make_shared<BufferImpl>();
    // Both queues belong to one family, so exclusive ownership covers them.
    vk::BufferCreateInfo bufferInfo({}, size, usage, vk::SharingMode::eExclusive);
    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = memory_usage;
    if (mapped)
        allocationInfo.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocationResult{};
    VkBuffer rawBuffer = VK_NULL_HANDLE;
    if (vmaCreateBufferWithAlignment(allocator_, reinterpret_cast<const VkBufferCreateInfo*>(&bufferInfo),
                        &allocationInfo, alignment, &rawBuffer, &allocation, &allocationResult) != VK_SUCCESS)
        throw Error(ErrorCode::OutOfMemory, "VMA buffer allocation failed");

    result->device = self_.lock();
    result->buffer = rawBuffer;
    result->allocation = allocation;
    result->size = size;
    result->mapped = allocationResult.pMappedData;
    if (mapped && !result->mapped) {
        if (vmaMapMemory(allocator_, allocation, &result->mapped) != VK_SUCCESS) {
            throw Error(ErrorCode::OutOfMemory, "VMA could not map a host-visible buffer");
        }
    }
    result->host_visible = result->mapped != nullptr;
    result->mapped_by_api = mapped && allocationResult.pMappedData == nullptr;
    if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
        result->address = buffer_address(result->buffer);
        // A freed buffer's device address can be handed straight back to the
        // next allocation, so the new owner must replace any dead entry rather
        // than losing to it.
        std::lock_guard lock(buffers_mutex_);
        buffers_.insert_or_assign(result->address, result);
    }
    return result;
}

vk::DeviceAddress DeviceImpl::buffer_address(const vk::Buffer buffer) const {
    if (!buffer)
        return 0;
    return vk_device().getBufferAddress({buffer});
}

std::pair<vk::Buffer, vk::DeviceSize> DeviceImpl::find_buffer(const vk::DeviceAddress address) const {
    const auto buffer = find_buffer_resource(address);
    return {buffer->buffer, address - buffer->address};
}

std::shared_ptr<BufferImpl> DeviceImpl::find_buffer_resource(const vk::DeviceAddress address) const {
    // buffers_ is keyed by base address, so the candidate is the last live
    // buffer that starts at or before `address`. Dead entries linger until
    // the next sweep, and freed memory can be reused by a buffer starting
    // below them, so they must be skipped rather than end the search.
    std::lock_guard lock(buffers_mutex_);
    for (auto entry = buffers_.upper_bound(address); entry != buffers_.begin();) {
        const auto buffer = (--entry)->second.lock();
        if (!buffer)
            continue;
        if (address < buffer->address + buffer->size)
            return buffer;
        break;
    }
    throw Error(ErrorCode::InvalidResource, "GPU address does not refer to a live buffer");
}

interop::DeviceHandles DeviceImpl::native_handles() const noexcept {
    return {
        reinterpret_cast<std::uintptr_t>(static_cast<VkInstance>(vk_instance())),
        reinterpret_cast<std::uintptr_t>(static_cast<VkPhysicalDevice>(physical_device_)),
        reinterpret_cast<std::uintptr_t>(static_cast<VkDevice>(vk_device())),
        reinterpret_cast<std::uintptr_t>(static_cast<VkQueue>(queue_state(Queue::Graphics).queue)),
        queue_state(Queue::Graphics).family,
        reinterpret_cast<std::uintptr_t>(static_cast<VkPipelineCache>(pipeline_cache())),
    };
}

std::shared_ptr<ImageImpl> DeviceImpl::find_image(const ImageHandle handle) const {
    auto image = handle.image_.lock();
    return image && image->device.get() == this ? image : nullptr;
}

std::shared_ptr<ShaderImpl> DeviceImpl::create_shader(const std::span<const std::byte> spirv,
    const std::string_view entry_point) {
    if (spirv.empty() || spirv.size_bytes() % sizeof(std::uint32_t) != 0)
        throw Error(ErrorCode::InvalidShader, "SPIR-V must be non-empty and 4-byte aligned");
    const auto* words = reinterpret_cast<const std::uint32_t*>(spirv.data());
    if (reinterpret_cast<std::uintptr_t>(spirv.data()) % alignof(std::uint32_t) != 0)
        throw Error(ErrorCode::InvalidShader, "SPIR-V data is not 4-byte aligned");
    if (words[0] != 0x07230203u)
        throw Error(ErrorCode::InvalidShader, "SPIR-V magic number is invalid");
    if (entry_point.empty())
        throw Error(ErrorCode::InvalidArgument, "shader entry point cannot be empty");

    vk::ShaderModuleCreateInfo info({}, spirv.size_bytes(), words);
    auto result = std::make_shared<ShaderImpl>();
    result->device = self_.lock();
    result->entry_point = entry_point;
    try {
        result->module = vk_device().createShaderModuleUnique(info);
    } catch (const vk::SystemError& error) {
        throw Error(ErrorCode::ShaderCreationFailed, error.what());
    }
    return result;
}

std::shared_ptr<ComputePipelineImpl> DeviceImpl::create_compute(const Shader& shader) {
    if (!shader.impl_)
        throw Error(ErrorCode::InvalidResource,
            "cannot create a compute pipeline from an empty shader");
    auto result = std::make_shared<ComputePipelineImpl>();
    result->device = self_.lock();
    const vk::PipelineShaderStageCreateInfo stage({}, vk::ShaderStageFlagBits::eCompute,
        *shader.impl_->module, shader.impl_->entry_point.c_str());
    const auto heap_flags = pipeline_heap_flags();
    vk::ComputePipelineCreateInfo pipelineInfo{{}, stage, {}};
    pipelineInfo.pNext = &heap_flags;
    try {
        result->pipeline = create_pipeline<vk::ComputePipelineCreateInfo>(pipelineInfo, true,
            [this, entry = shader.impl_->entry_point](const vk::ComputePipelineCreateInfo& info, vk::PipelineCache cache) {
                try {
                    return vk_device().createComputePipelineUnique(cache, info).value;
                } catch (const vk::SystemError& error) {
                    std::fprintf(stderr, "[NoorRHI] compute pipeline creation failed for entry '%s': %s\n",
                        entry.c_str(), error.what());
                    throw;
                }
            });
    } catch (const vk::SystemError& error) {
        throw Error(ErrorCode::ShaderCreationFailed, error.what());
    }
    return result;
}

std::shared_ptr<SamplerImpl> DeviceImpl::create_sampler(const SamplerDesc& desc) {
    require_descriptor_heaps();
    const auto filter = desc.filter == Filter::Linear ? vk::Filter::eLinear : vk::Filter::eNearest;
    const auto address = [](const AddressMode mode) {
        switch (mode) {
        case AddressMode::MirroredRepeat: return vk::SamplerAddressMode::eMirroredRepeat;
        case AddressMode::ClampToEdge: return vk::SamplerAddressMode::eClampToEdge;
        case AddressMode::ClampToBorder: return vk::SamplerAddressMode::eClampToBorder;
        default: return vk::SamplerAddressMode::eRepeat;
        }
    };
    vk::SamplerCreateInfo info({}, filter, filter, vk::SamplerMipmapMode::eLinear,
        address(desc.address_u), address(desc.address_v), address(desc.address_w));
    info.maxLod = vk::LodClampNone;
    auto result = std::make_shared<SamplerImpl>();
    result->device = self_.lock();
    const std::uint32_t slot = allocate_slot(sampler_heap_);
    const vk::HostAddressRangeEXT destination = slot_range(sampler_heap_, slot);
    if (vk_device().writeSamplerDescriptorsEXT(1, &info, &destination) != vk::Result::eSuccess) {
        release_slot(sampler_heap_, slot);
        throw Error(ErrorCode::InvalidState, "writing a sampler descriptor failed");
    }
    flush_slot(sampler_heap_, slot);
    result->handle = SamplerHandle{slot};
    return result;
}

void DeviceImpl::record_native(const std::function<void(std::uintptr_t)>& commands) {
    submit([&commands](const vk::CommandBuffer command) {
        commands(reinterpret_cast<std::uintptr_t>(static_cast<VkCommandBuffer>(command)));
    });
}

namespace {
// The queues each thread is bound to, innermost last; see noorrhi::QueueScope.
struct QueueBinding {
    const DeviceImpl* device;
    Queue queue;
};
thread_local std::vector<QueueBinding> queue_bindings;
}

thread_local vk::CommandBuffer DeviceImpl::active_command_;
thread_local vk::Format DeviceImpl::active_color_format_ = vk::Format::eUndefined;
thread_local bool DeviceImpl::active_has_depth_ = false;
thread_local bool DeviceImpl::active_flip_y_ = false;
thread_local std::uint32_t DeviceImpl::active_width_ = 0;
thread_local std::uint32_t DeviceImpl::active_height_ = 0;
thread_local std::vector<std::shared_ptr<void>> DeviceImpl::active_resources_;

Queue DeviceImpl::current_queue() const {
    for (auto binding = queue_bindings.rbegin(); binding != queue_bindings.rend(); ++binding)
        if (binding->device == this)
            return binding->queue;
    return Queue::Graphics;
}

void DeviceImpl::bind_queue(const Queue queue) const {
    queue_bindings.push_back({this, queue});
}

void DeviceImpl::unbind_queue() const {
    if (queue_bindings.empty() || queue_bindings.back().device != this)
        throw Error(ErrorCode::InvalidState, "QueueScopes must end in reverse order on their own thread");
    queue_bindings.pop_back();
}

bool DeviceImpl::recording_frame() const {
    const QueueState& queue = queue_state(current_queue());
    std::lock_guard lock(mutex_);
    return queue.recording && queue.recording_thread == std::this_thread::get_id();
}

DeviceImpl::CommandPool& DeviceImpl::thread_command_pool(const Queue queue) {
    auto& pool = command_pools_[{std::this_thread::get_id(), queue}];
    if (!pool) {
        pool = std::make_unique<CommandPool>();
        pool->pool = vk_device().createCommandPoolUnique(
            {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queue_state(queue).family});
    }
    return *pool;
}

vk::CommandBuffer DeviceImpl::allocate_command(CommandPool& pool) {
    // A recycled buffer is reset implicitly when it begins again.
    if (!pool.free.empty()) {
        const vk::CommandBuffer command = pool.free.back();
        pool.free.pop_back();
        return command;
    }
    const auto commands = vk_device().allocateCommandBuffers(
        {*pool.pool, vk::CommandBufferLevel::ePrimary, 1});
    if (commands.empty())
        throw Error(ErrorCode::OutOfMemory, "Vulkan command-buffer allocation failed");
    return commands.front();
}

void DeviceImpl::submit_command(QueueState& queue, const GpuToken token, const vk::CommandBuffer command,
    CommandPool* const pool, std::vector<std::shared_ptr<void>> resources, const QueueWaits& extra_waits,
    const std::span<const vk::Semaphore> extra_signals) {
    QueueWaits waits = take_queue_waits(queue);
    waits.semaphores.insert(waits.semaphores.begin(), extra_waits.semaphores.begin(), extra_waits.semaphores.end());
    waits.values.insert(waits.values.begin(), extra_waits.values.begin(), extra_waits.values.end());
    waits.stages.insert(waits.stages.begin(), extra_waits.stages.begin(), extra_waits.stages.end());
    // Only the timeline entry carries a value; binary semaphores' slots are
    // ignored but must be present for the arrays to line up.
    std::vector<vk::Semaphore> signals{queue.timeline.get()};
    signals.insert(signals.end(), extra_signals.begin(), extra_signals.end());
    std::vector<std::uint64_t> signal_values(signals.size(), 0);
    signal_values.front() = token.value;
    vk::TimelineSemaphoreSubmitInfo timeline_info{};
    timeline_info.setWaitSemaphoreValues(waits.values).setSignalSemaphoreValues(signal_values);
    vk::SubmitInfo submit_info{};
    submit_info.setCommandBuffers(command).setSignalSemaphores(signals)
        .setWaitSemaphores(waits.semaphores).setWaitDstStageMask(waits.stages);
    submit_info.pNext = &timeline_info;
    queue.queue.submit(submit_info);
    queue.pending.push_back({token, command, pool, std::move(resources)});
    queue.next_timeline = token.value + 1;
}

std::uint64_t DeviceImpl::completed_value(const Queue queue) const {
    return vk_device().getSemaphoreCounterValue(queue_state(queue).timeline.get());
}

bool DeviceImpl::completed(const Timelines& timelines) const {
    for (std::size_t queue = 0; queue < queue_count; ++queue)
        if (timelines[queue] > completed_value(static_cast<Queue>(queue)))
            return false;
    return true;
}

Timelines DeviceImpl::recorded_timelines() const {
    Timelines timelines{};
    for (std::size_t queue = 0; queue < queue_count; ++queue) {
        const std::uint64_t recording = queues_[queue].recording_value;
        timelines[queue] = recording != 0 ? recording : queues_[queue].next_timeline - 1;
    }
    return timelines;
}

Timelines DeviceImpl::recording_timelines() const {
    const Queue queue = current_queue();
    Timelines timelines{};
    timelines[static_cast<std::size_t>(queue)] = queue_state(queue).next_timeline;
    return timelines;
}

GpuToken DeviceImpl::submit(const std::function<void(vk::CommandBuffer)>& record,
    std::vector<std::shared_ptr<void>> resources, const Ordering ordering) {
    const Queue queue = current_queue();
    QueueState& target = queue_state(queue);
    vk::CommandBuffer command;
    CommandPool* pool = nullptr;
    bool batched = false;
    {
        std::lock_guard lock(mutex_);
        if (shut_down_)
            throw Error(ErrorCode::InvalidState, "noorrhi::Device has been shut down");
        if (target.recording) {
            // The recording reserved the queue's next timeline value, so no
            // other thread's submission may go in front of it.
            if (std::this_thread::get_id() != target.recording_thread)
                throw Error(ErrorCode::InvalidState,
                    "another thread has a frame or Recording open on this queue");
            command = target.recording;
            batched = true;
        } else {
            reap_completed();
            pool = &thread_command_pool(queue);
            command = allocate_command(*pool);
        }
    }
    // Recording needs no lock: the command buffer and its pool belong to
    // this thread, and the render scope state is thread-local.
    struct ActiveResources {
        ActiveResources() { active_resources_.clear(); }
        ~ActiveResources() { active_resources_.clear(); }
    } active_resources_scope;
    if (batched) {
        // Serialized work gets the same ordering barriers a standalone
        // submission gets. Explicit work gets none: consecutive dispatches and
        // traces overlap unless the caller orders them with barrier().
        const bool serialized = ordering == Ordering::Serialized;
        if (serialized)
            record_full_barrier(command);
        record(command);
        if (serialized)
            record_full_barrier(command);
        // The recording's submission is what will retire these, so they are
        // held until it is submitted.
        target.recording_resources.insert(target.recording_resources.end(),
            std::make_move_iterator(resources.begin()), std::make_move_iterator(resources.end()));
        target.recording_resources.insert(target.recording_resources.end(),
            active_resources_.begin(), active_resources_.end());
        return {target.recording_value, queue};
    }
    command.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    record_full_barrier(command);
    record(command);
    resources.insert(resources.end(), active_resources_.begin(), active_resources_.end());
    // Publish writes made by this submission to later queue submissions. This
    // is required for upload copies consumed by externally recorded NoorRay
    // commands, which are submitted directly to the same queue after this
    // API-owned command buffer.
    record_full_barrier(command);
    command.end();

    std::lock_guard lock(mutex_);
    const GpuToken token{target.next_timeline, queue};
    submit_command(target, token, command, pool, std::move(resources));
    return token;
}

std::shared_ptr<Recording::State> DeviceImpl::begin_recording() {
    const Queue queue = current_queue();
    QueueState& target = queue_state(queue);
    auto state = std::make_shared<Recording::State>();
    state->device = self_.lock();
    state->queue = queue;
    {
        std::lock_guard lock(mutex_);
        if (shut_down_)
            throw Error(ErrorCode::InvalidState, "noorrhi::Device has been shut down");
        if (target.recording)
            throw Error(ErrorCode::InvalidState, "a frame or Recording is already open on this queue");
        reap_completed();
        CommandPool& pool = thread_command_pool(queue);
        state->command = allocate_command(pool);
        target.recording = state->command;
        target.recording_pool = &pool;
        target.recording_value = target.next_timeline.load();
        target.recording_thread = std::this_thread::get_id();
        target.recording_resources.clear();
    }
    state->command.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    // Work inside the recording is ordered explicitly, so order it once
    // against everything submitted before it.
    record_full_barrier(state->command);
    return state;
}

GpuToken DeviceImpl::end_recording(Recording::State& state) {
    QueueState& target = queue_state(state.queue);
    if (!state.command || target.recording != state.command)
        throw Error(ErrorCode::InvalidState, "the Recording was already submitted");
    // Publishes the recording's explicitly ordered work to later submissions.
    record_full_barrier(state.command);
    state.command.end();
    std::lock_guard lock(mutex_);
    const GpuToken token{target.recording_value, state.queue};
    submit_command(target, token, state.command, target.recording_pool,
        std::move(target.recording_resources));
    target.recording = nullptr;
    target.recording_pool = nullptr;
    target.recording_value = 0;
    target.recording_resources.clear();
    state.command = nullptr;
    return token;
}

void DeviceImpl::abandon_recording(Recording::State& state) {
    QueueState& target = queue_state(state.queue);
    std::lock_guard lock(mutex_);
    if (!state.command || target.recording != state.command)
        return;
    const auto index = static_cast<std::size_t>(state.queue);
    const std::uint64_t value = target.recording_value;
    // Nothing was submitted, so the reserved value never signals: records
    // and resources protected for it are protected for the last one instead.
    {
        std::lock_guard argument_lock(target.argument_mutex);
        // The submission never signals, so records written for it are dead and
        // the chunks they share with older records wait for the last one instead.
        for (ArgumentChunk& chunk : target.argument_chunks)
            if (chunk.timelines[index] == value)
                chunk.timelines[index] = value - 1;
        std::erase_if(target.argument_dedicated, [index, value](const ArgumentChunk& chunk) {
            return chunk.held == 0 && chunk.timelines[index] == value;
        });
    }
    {
        std::lock_guard retire_lock(retire_mutex_);
        for (auto& entry : retired_)
            if (entry.timelines[index] == value)
                entry.timelines[index] = value - 1;
    }
    state.command.reset();
    target.recording_pool->free.push_back(state.command);
    target.recording = nullptr;
    target.recording_pool = nullptr;
    target.recording_value = 0;
    target.recording_resources.clear();
    state.command = nullptr;
}

void DeviceImpl::abandon_frame(Frame::State& state) {
    Recording::State recording{state.device, Queue::Graphics, state.command};
    abandon_recording(recording);
    state.swapchain->stale = true;
    state.open = false;
    state.command = nullptr;
}

void DeviceImpl::retire(std::function<void()> release) {
    if (!release)
        return;
    {
        std::lock_guard lock(retire_mutex_);
        if (!shut_down_) {
            // Include commands recorded into the current, not-yet-submitted frame.
            retired_.push_back({recorded_timelines(), std::move(release)});
            return;
        }
    }
    // After shutdown the queues are idle, so releasing immediately is safe.
    release();
}

void DeviceImpl::reap_completed() {
    if (!queues_.front().timeline)
        return;
    Timelines completed_values{};
    for (std::size_t index = 0; index < queue_count; ++index) {
        QueueState& queue = queues_[index];
        completed_values[index] = completed_value(static_cast<Queue>(index));
        while (!queue.pending.empty() && queue.pending.front().token.value <= completed_values[index]) {
            // The owning thread resets and reuses the buffer; see CommandPool.
            if (Pending& done = queue.pending.front(); done.pool)
                done.pool->free.push_back(done.command);
            queue.pending.pop_front();
        }
    }
    const auto is_completed = [&completed_values](const Timelines& timelines) {
        for (std::size_t index = 0; index < queue_count; ++index)
            if (timelines[index] > completed_values[index])
                return false;
        return true;
    };

    // Collect first, then release with no lock held: a release can drop the
    // last reference to another resource and re-enter retire().
    std::vector<std::function<void()>> releases;
    {
        std::lock_guard retire_lock(retire_mutex_);
        while (!retired_.empty() && is_completed(retired_.front().timelines)) {
            releases.push_back(std::move(retired_.front().release));
            retired_.pop_front();
        }
    }
    for (auto& release : releases)
        release();

    std::lock_guard buffers_lock(buffers_mutex_);
    if (buffers_.size() > buffer_sweep_size_) {
        std::erase_if(buffers_, [](const auto& entry) { return entry.second.expired(); });
        buffer_sweep_size_ = std::max<std::size_t>(64, buffers_.size() * 2);
    }
}

void DeviceImpl::wait(const GpuToken token) {
    if (token.value == 0)
        return;
    {
        const QueueState& queue = queue_state(token.queue);
        std::lock_guard lock(mutex_);
        if (queue.recording && queue.recording_thread == std::this_thread::get_id()
            && token.value >= queue.recording_value)
            throw Error(ErrorCode::InvalidState,
                "cannot wait for the calling thread's open frame or Recording; submit it first");
    }
    // Blocks without mutex_, so other threads keep submitting meanwhile.
    vk::SemaphoreWaitInfo info({}, queue_state(token.queue).timeline.get(), token.value);
    const auto result = vk_device().waitSemaphores(info, std::numeric_limits<std::uint64_t>::max());
    if (result != vk::Result::eSuccess)
        throw Error(ErrorCode::DeviceLost, "waiting for the GPU timeline failed");
    std::lock_guard lock(mutex_);
    reap_completed();
}

bool DeviceImpl::finished(const GpuToken token) const {
    return token.value <= completed_value(token.queue);
}

GpuToken DeviceImpl::signal() {
    std::lock_guard lock(mutex_);
    const Queue queue = current_queue();
    const QueueState& state = queue_state(queue);
    // Every submission signals its queue's timeline, so the last one's value
    // already covers everything submitted so far.
    if (state.recording && state.recording_thread == std::this_thread::get_id())
        return {state.recording_value, queue};
    return {state.next_timeline - 1, queue};
}

void DeviceImpl::queue_wait(const GpuToken token) {
    std::lock_guard lock(mutex_);
    const Queue queue = current_queue();
    // Submission order already covers the calling thread's own queue.
    if (token.queue == queue || token.value <= completed_value(token.queue))
        return;
    std::uint64_t& wait = queue_state(queue).waits[static_cast<std::size_t>(token.queue)];
    wait = std::max(wait, token.value);
}

DeviceImpl::QueueWaits DeviceImpl::take_queue_waits(QueueState& queue) {
    QueueWaits waits;
    for (std::size_t index = 0; index < queue_count; ++index) {
        if (queue.waits[index] == 0)
            continue;
        waits.semaphores.push_back(queues_[index].timeline.get());
        waits.values.push_back(std::exchange(queue.waits[index], 0));
        waits.stages.push_back(vk::PipelineStageFlagBits::eAllCommands);
    }
    return waits;
}

void DeviceImpl::synchronize() {
    Timelines submitted{};
    {
        std::lock_guard lock(mutex_);
        for (std::size_t queue = 0; queue < queue_count; ++queue)
            submitted[queue] = queues_[queue].next_timeline - 1;
    }
    for (std::size_t queue = 0; queue < queue_count; ++queue)
        wait({submitted[queue], static_cast<Queue>(queue)});
}

vk::DeviceAddress DeviceImpl::stage_arguments(const void* args, const std::size_t size) {
    return place_arguments(args, size, 0);
}

vk::DeviceAddress DeviceImpl::stage_held(const void* args, const std::size_t size,
    std::uint64_t& id) {
    id = next_staged_id_++;
    return place_arguments(args, size, id);
}

void DeviceImpl::release_staged(const std::uint64_t id) noexcept {
    for (QueueState& queue : queues_) {
        std::lock_guard argument_lock(queue.argument_mutex);
        for (ArgumentChunk& chunk : queue.argument_dedicated) {
            if (chunk.held != id)
                continue;
            // Every launch that read the record was recorded before this
            // call: into an open recording, or into submissions already made.
            chunk.held = 0;
            chunk.timelines = recorded_timelines();
            return;
        }
    }
}

// The chunk with room for `size` bytes: the current one, else the oldest
// chunk whose submissions have all completed, else a new one. Taking a chunk
// never waits, so the argument data one submission may record is limited only
// by device memory.
DeviceImpl::ArgumentChunk& DeviceImpl::pooled_argument_chunk(QueueState& queue, const std::size_t size) {
    ArgumentChunk& current = queue.argument_chunks.back();
    if (((current.offset + host_alignment_ - 1) & ~(host_alignment_ - 1)) + size <= current.arena->size)
        return current;
    auto reusable = std::find_if(queue.argument_chunks.begin(), std::prev(queue.argument_chunks.end()),
        [this](const ArgumentChunk& chunk) { return completed(chunk.timelines); });
    if (reusable == std::prev(queue.argument_chunks.end()))
        queue.argument_chunks.push_back({create_argument_arena(argument_arena_size_)});
    else {
        reusable->offset = 0;
        reusable->timelines = {};
        queue.argument_chunks.splice(queue.argument_chunks.end(), queue.argument_chunks, reusable);
    }
    return queue.argument_chunks.back();
}

DeviceImpl::ArgumentChunk& DeviceImpl::dedicated_argument_chunk(QueueState& queue, const std::size_t size,
    const std::uint64_t held) {
    std::erase_if(queue.argument_dedicated, [this](const ArgumentChunk& chunk) {
        return chunk.held == 0 && completed(chunk.timelines);
    });
    const std::size_t capacity = (size + host_alignment_ - 1) & ~(host_alignment_ - 1);
    queue.argument_dedicated.push_back({create_argument_arena(capacity), 0, {}, held});
    return queue.argument_dedicated.back();
}

// Places the record in the calling thread's queue's chunks, protected for the
// submission it is being recorded into. A staged record (`held` is its id) and
// a record larger than a chunk each get a chunk of their own, so a long-lived
// staged record never pins the space of other records.
vk::DeviceAddress DeviceImpl::place_arguments(const void* args, const std::size_t size,
    const std::uint64_t held) {
    if (!args || size == 0)
        return 0;
    QueueState& queue = queue_state(current_queue());
    std::lock_guard lock(queue.argument_mutex);
    if (queue.argument_chunks.empty())
        throw Error(ErrorCode::InvalidState, "noorrhi::Device has been shut down");
    const bool dedicated = held != 0 || size > argument_arena_size_;
    ArgumentChunk& chunk = dedicated ? dedicated_argument_chunk(queue, size, held)
                                     : pooled_argument_chunk(queue, size);
    const std::size_t offset = (chunk.offset + host_alignment_ - 1) & ~(host_alignment_ - 1);
    const Timelines recording = recording_timelines();
    for (std::size_t index = 0; index < queue_count; ++index)
        chunk.timelines[index] = std::max(chunk.timelines[index], recording[index]);
    std::memcpy(static_cast<std::byte*>(chunk.arena->mapped) + offset, args, size);
    vmaFlushAllocation(allocator_, chunk.arena->allocation, offset, size);
    chunk.offset = offset + size;
    return chunk.arena->address + offset;
}

void DeviceImpl::push_root(const vk::CommandBuffer command,
    const vk::DeviceAddress root) const {
    command.pushDataEXT(vk::PushDataInfoEXT{0,
        vk::HostAddressRangeConstEXT{&root, sizeof(root)}});
}

void DeviceImpl::record_compute(const ComputePipelineImpl& pipeline,
    const vk::CommandBuffer command, const DispatchSize groups,
    const void* args, const std::size_t size) {
    if (!pipeline.pipeline || !args || size == 0
        || groups.x == 0 || groups.y == 0 || groups.z == 0)
        throw Error(ErrorCode::InvalidArgument, "invalid compute dispatch");
    command.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline.pipeline);
    bind_heaps(command);
    push_root(command, stage_arguments(args, size));
    command.dispatch(groups.x, groups.y, groups.z);
}

vk::PipelineStageFlags2 DeviceImpl::stage_mask(const Stage stage) {
    switch (stage) {
    case Stage::Copy: return vk::PipelineStageFlagBits2::eCopy;
    case Stage::Compute: return vk::PipelineStageFlagBits2::eComputeShader;
    case Stage::Vertex: return vk::PipelineStageFlagBits2::eVertexShader;
    case Stage::Fragment: return vk::PipelineStageFlagBits2::eFragmentShader;
    case Stage::RayTracing: return vk::PipelineStageFlagBits2::eRayTracingShaderKHR;
    case Stage::AccelerationStructure: return vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR;
    case Stage::Present: return vk::PipelineStageFlagBits2::eAllCommands;
    }
    return vk::PipelineStageFlagBits2::eAllCommands;
}

void DeviceImpl::record_barrier(const vk::CommandBuffer command, const Stage source,
    const Stage destination) const {
    const vk::PipelineStageFlags2 sourceStages = stage_mask(source);
    const vk::PipelineStageFlags2 destinationStages = stage_mask(destination);
    vk::MemoryBarrier2 memory{};
    memory.setSrcStageMask(sourceStages)
        .setSrcAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
        .setDstStageMask(destinationStages)
        .setDstAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
    command.pipelineBarrier2({{}, memory, {}, {}});
}

void DeviceImpl::record_full_barrier(const vk::CommandBuffer command) {
    vk::MemoryBarrier2 memory{};
    memory.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
        .setSrcAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
        .setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
        .setDstAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
    command.pipelineBarrier2({{}, memory, {}, {}});
}

void DeviceImpl::barrier(const Stage source, const Stage destination) {
    submit([this, source, destination](const vk::CommandBuffer command) {
        record_barrier(command, source, destination);
    }, {}, Ordering::Explicit);
}

namespace {
// Joins threads to a deferred operation until it completes, as many as the
// driver can use, and returns the operation's result.
VkResult join_deferred_operation(const vk::Device device, const vk::DeferredOperationKHR operation) {
    const auto& dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
    const auto join = [&dispatch, device, operation] {
        while (dispatch.vkDeferredOperationJoinKHR(device, operation) == VK_THREAD_IDLE_KHR)
            std::this_thread::yield();
    };
    const std::uint32_t concurrency = std::min(
        dispatch.vkGetDeferredOperationMaxConcurrencyKHR(device, operation),
        std::max(std::thread::hardware_concurrency(), 1u));
    {
        std::vector<std::jthread> helpers;
        for (std::uint32_t i = 1; i < concurrency; ++i)
            helpers.emplace_back(join);
        join();
    }
    return dispatch.vkGetDeferredOperationResultKHR(device, operation);
}
} // namespace

// Ray-tracing pipelines are created one at a time. NoorRay keeps CPU-side
// material work parallel, but uncached runtime material libraries must not
// enter the driver's ray-tracing compiler concurrently: some drivers keep
// compiler work alive internally after vkCreateRayTracingPipelinesKHR returns.
// Cached calls are already serialized by create_pipeline(); uncached calls take
// the same mutex here. Each creation is deferred instead, so the one compile in
// flight runs on every core the driver can use.
vk::UniquePipeline DeviceImpl::create_ray_tracing_pipeline(
    const vk::RayTracingPipelineCreateInfoKHR& info, vk::PipelineCache cache) {
    std::unique_lock<std::mutex> creation_lock;
    if (!cache)
        creation_lock = std::unique_lock<std::mutex>(pipeline_creation_mutex_);
    const vk::Device device = vk_device();
    const vk::UniqueDeferredOperationKHR operation = device.createDeferredOperationKHRUnique();
    // Written by the driver once the deferred operation completes.
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult result = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateRayTracingPipelinesKHR(device, *operation,
        cache, 1, reinterpret_cast<const VkRayTracingPipelineCreateInfoKHR*>(&info), nullptr, &pipeline);
    if (result == VK_OPERATION_DEFERRED_KHR)
        result = join_deferred_operation(device, *operation);
    if (result != VK_SUCCESS && result != VK_OPERATION_NOT_DEFERRED_KHR)
        throw Error(ErrorCode::ShaderCreationFailed,
            "vkCreateRayTracingPipelinesKHR: " + vk::to_string(static_cast<vk::Result>(result)));
    return vk::UniquePipeline(pipeline, device);
}

RayTracingGroups DeviceImpl::ray_tracing_groups(const RayTracingPipelineDesc& desc) const {
    if (!ray_tracing_supported_)
        throw Error(ErrorCode::UnsupportedFeature,
            "acceleration structures are not enabled on this noorrhi::Device");

    RayTracingGroups result;
    // One stage per distinct shader: drivers compile every stage entry, so a
    // shader repeated across groups (the shared closest hit, for one) would
    // otherwise be compiled once per group.
    std::map<const ShaderImpl*, std::uint32_t> stage_indices;
    auto add_stage = [&](const Shader& shader, const vk::ShaderStageFlagBits stage) {
        if (!shader.impl_)
            throw Error(ErrorCode::InvalidResource, "ray-tracing shader list contains an empty shader");
        if (const auto existing = stage_indices.find(shader.impl_.get());
            existing != stage_indices.end())
            return existing->second;
        const auto index = static_cast<std::uint32_t>(result.stages.size());
        stage_indices.emplace(shader.impl_.get(), index);
        result.shaders.push_back(shader.impl_);
        // Nothing to map: the raygen stage reads its acceleration structure
        // from an address in its root record, not from a binding.
        result.stages.push_back({{}, stage, *shader.impl_->module,
            shader.impl_->entry_point.c_str()});
        return index;
    };
    auto add_general = [&](const Shader& shader, const vk::ShaderStageFlagBits stage,
                           const RayTracingGroups::Kind kind) {
        const auto index = add_stage(shader, stage);
        result.create_infos.emplace_back(vk::RayTracingShaderGroupTypeKHR::eGeneral, index,
            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
        result.groups.push_back({kind, kind == RayTracingGroups::Kind::Raygen
            ? shader.impl_.get() : nullptr});
    };
    for (const auto& shader : desc.raygen)
        add_general(shader, vk::ShaderStageFlagBits::eRaygenKHR, RayTracingGroups::Kind::Raygen);
    for (const auto& shader : desc.miss)
        add_general(shader, vk::ShaderStageFlagBits::eMissKHR, RayTracingGroups::Kind::Miss);
    const auto hit_count = std::max({desc.closest_hit.size(), desc.any_hit.size(), desc.intersection.size()});
    for (std::size_t i = 0; i < hit_count; ++i) {
        const bool has_intersection = i < desc.intersection.size();
        const auto type = has_intersection ? vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup
                                            : vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup;
        std::uint32_t closest = VK_SHADER_UNUSED_KHR;
        std::uint32_t any = VK_SHADER_UNUSED_KHR;
        std::uint32_t intersection = VK_SHADER_UNUSED_KHR;
        if (i < desc.closest_hit.size() && desc.closest_hit[i].impl_)
            closest = add_stage(desc.closest_hit[i], vk::ShaderStageFlagBits::eClosestHitKHR);
        if (i < desc.any_hit.size() && desc.any_hit[i].impl_)
            any = add_stage(desc.any_hit[i], vk::ShaderStageFlagBits::eAnyHitKHR);
        if (has_intersection)
            intersection = add_stage(desc.intersection[i], vk::ShaderStageFlagBits::eIntersectionKHR);
        result.create_infos.emplace_back(type, VK_SHADER_UNUSED_KHR, closest, any, intersection);
        result.groups.push_back({RayTracingGroups::Kind::Hit});
    }
    for (const auto& shader : desc.callable)
        add_general(shader, vk::ShaderStageFlagBits::eCallableKHR, RayTracingGroups::Kind::Callable);
    if (result.groups.empty())
        throw Error(ErrorCode::InvalidArgument, "ray-tracing pipeline contains no shader groups");
    return result;
}

std::shared_ptr<RayTracingLibraryImpl> DeviceImpl::create_ray_tracing_library(
    const RayTracingPipelineDesc& desc, const RayTracingInterface& interface) {
    require_descriptor_heaps();
    auto result = std::make_shared<RayTracingLibraryImpl>();
    result->device = self_.lock();
    result->interface = interface;
    result->groups = ray_tracing_groups(desc);

    const vk::RayTracingPipelineInterfaceCreateInfoKHR library_interface(
        interface.max_payload_size, interface.max_hit_attribute_size);
    vk::PipelineCreateFlags2CreateInfo flags{
        vk::PipelineCreateFlagBits2::eDescriptorHeapEXT | vk::PipelineCreateFlagBits2::eLibraryKHR};
    vk::RayTracingPipelineCreateInfoKHR info({}, result->groups.stages,
        result->groups.create_infos, 1, nullptr, &library_interface, nullptr, {});
    info.pNext = &flags;
    result->pipeline = create_pipeline<vk::RayTracingPipelineCreateInfoKHR>(info,
        desc.use_pipeline_cache,
        [this](const vk::RayTracingPipelineCreateInfoKHR& library, vk::PipelineCache cache) {
            return create_ray_tracing_pipeline(library, cache);
        });
    result->use_pipeline_cache = desc.use_pipeline_cache;
    // Libraries take seconds to compile; keep them even if the process dies.
    if (desc.use_pipeline_cache && pipeline_cache_) {
        std::lock_guard lock(pipeline_creation_mutex_);
        pipeline_cache_->save();
    }
    return result;
}

std::shared_ptr<RayTracingPipelineImpl> DeviceImpl::link_ray_tracing(
    const std::span<const std::shared_ptr<RayTracingLibraryImpl>> libraries,
    const std::span<const std::uint32_t> hit_groups) {
    if (libraries.empty())
        throw Error(ErrorCode::InvalidArgument, "ray-tracing pipelines require at least one library");
    auto result = std::make_shared<RayTracingPipelineImpl>();
    result->device = self_.lock();

    const RayTracingInterface interface = libraries.front()->interface;
    std::vector<vk::Pipeline> handles;
    // A linked pipeline numbers its groups library by library.
    std::vector<RayTracingGroups::Group> groups;
    for (const auto& library : libraries) {
        if (!library)
            throw Error(ErrorCode::InvalidResource, "ray-tracing library is empty");
        if (library->interface != interface)
            throw Error(ErrorCode::InvalidArgument, "linked ray-tracing libraries must share one interface");
        handles.push_back(*library->pipeline);
        groups.insert(groups.end(), library->groups.groups.begin(), library->groups.groups.end());
        result->libraries.push_back(library);
    }
    if (std::ranges::none_of(groups, [](const RayTracingGroups::Group& group) {
            return group.kind == RayTracingGroups::Kind::Raygen; }))
        throw Error(ErrorCode::InvalidArgument, "ray-tracing pipelines require a ray-generation shader");

    const vk::RayTracingPipelineInterfaceCreateInfoKHR library_interface(
        interface.max_payload_size, interface.max_hit_attribute_size);
    const vk::PipelineLibraryCreateInfoKHR library_info(handles);
    const auto heap_flags = pipeline_heap_flags();
    vk::RayTracingPipelineCreateInfoKHR info({}, {}, {}, 1, &library_info, &library_interface,
        nullptr, {});
    info.pNext = &heap_flags;
    try {
        std::lock_guard lock(pipeline_creation_mutex_);
        result->pipeline = std::make_shared<const vk::UniquePipeline>(
            vk_device().createRayTracingPipelineKHRUnique({},
                std::ranges::all_of(libraries, [](const auto& library) {
                    return library->use_pipeline_cache;
                }) ? pipeline_cache() : vk::PipelineCache{}, info).value);
    } catch (const vk::SystemError& error) {
        throw Error(ErrorCode::ShaderCreationFailed, error.what());
    }
    result->groups = std::move(groups);
    build_shader_binding_table(*result, result->groups, hit_groups);
    return result;
}

std::shared_ptr<RayTracingPipelineImpl> DeviceImpl::rebind_hit_groups(
    const RayTracingPipelineImpl& linked, const std::span<const std::uint32_t> hit_groups) {
    if (!linked.pipeline)
        throw Error(ErrorCode::InvalidResource, "ray-tracing pipeline is empty");
    auto result = std::make_shared<RayTracingPipelineImpl>();
    result->device = self_.lock();
    result->libraries = linked.libraries;
    result->pipeline = linked.pipeline;
    result->groups = linked.groups;
    build_shader_binding_table(*result, result->groups, hit_groups);
    return result;
}

void DeviceImpl::build_shader_binding_table(RayTracingPipelineImpl& result,
    const std::span<const RayTracingGroups::Group> groups,
    const std::span<const std::uint32_t> hit_groups) {
    const std::uint32_t handle_size = ray_tracing_properties_.shaderGroupHandleSize;
    const std::uint32_t alignment = std::max(ray_tracing_properties_.shaderGroupHandleAlignment, 1u);
    const std::uint32_t base_alignment = std::max(
        ray_tracing_properties_.shaderGroupBaseAlignment, 1u);
    const std::uint32_t handle_stride = (handle_size + alignment - 1) / alignment * alignment;
    // Each region begins at a group-base-aligned address. Since the regions
    // follow each other in one compact table, the record stride must also
    // preserve that stronger alignment (handle alignment alone is commonly
    // only 32 bytes on NVIDIA hardware).
    const std::uint32_t stride = (handle_stride + base_alignment - 1)
        / base_alignment * base_alignment;
    const std::size_t group_count = groups.size();
    std::vector<std::byte> handle_bytes(group_count * handle_size);
    if (vk_device().getRayTracingShaderGroupHandlesKHR(**result.pipeline, 0,
            static_cast<std::uint32_t>(group_count), handle_bytes.size(), handle_bytes.data()) != vk::Result::eSuccess)
        throw Error(ErrorCode::ShaderCreationFailed, "Vulkan shader binding table handle query failed");
    // The records of each region, as group indices. Groups keep their
    // relative order inside the raygen, miss and callable regions, which is
    // what callable indices address; the hit region is the caller's table.
    const auto groups_of = [&](const RayTracingGroups::Kind kind) {
        std::vector<std::size_t> result;
        for (std::size_t i = 0; i < group_count; ++i)
            if (groups[i].kind == kind)
                result.push_back(i);
        return result;
    };
    const std::vector<std::size_t> raygen_records = groups_of(RayTracingGroups::Kind::Raygen);
    const std::vector<std::size_t> miss_records = groups_of(RayTracingGroups::Kind::Miss);
    const std::vector<std::size_t> hit_group_indices = groups_of(RayTracingGroups::Kind::Hit);
    const std::vector<std::size_t> callable_records = groups_of(RayTracingGroups::Kind::Callable);
    std::vector<std::size_t> hit_records;
    hit_records.reserve(hit_groups.size());
    for (const std::uint32_t hit_group : hit_groups) {
        if (hit_group >= hit_group_indices.size())
            throw Error(ErrorCode::InvalidArgument,
                "shader binding table names hit group " + std::to_string(hit_group)
                    + " of a pipeline with " + std::to_string(hit_group_indices.size()));
        hit_records.push_back(hit_group_indices[hit_group]);
    }
    const std::size_t record_count = raygen_records.size() + miss_records.size()
        + hit_records.size() + callable_records.size();

    // VMA does not promise that a storage allocation's device address is
    // aligned to shaderGroupBaseAlignment. Reserve a prefix and publish an
    // aligned address inside the allocation; using the raw allocation base
    // makes vkCmdTraceRaysKHR reject the miss/hit regions (and some drivers
    // report the resulting device loss only at queue submission time).
    result.shader_binding_table = create_buffer(record_count * stride + base_alignment,
        vk::BufferUsageFlagBits::eShaderBindingTableKHR
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_MEMORY_USAGE_CPU_TO_GPU, true);
    std::memset(result.shader_binding_table->mapped, 0, result.shader_binding_table->size);
    const vk::DeviceAddress base = (result.shader_binding_table->address
        + base_alignment - 1) & ~(static_cast<vk::DeviceAddress>(base_alignment) - 1);
    const std::size_t table_offset = static_cast<std::size_t>(
        base - result.shader_binding_table->address);

    std::size_t record = 0;
    const auto write_region = [&](const std::span<const std::size_t> group_indices) {
        const std::size_t first = record;
        for (const std::size_t group : group_indices) {
            std::memcpy(static_cast<std::byte*>(result.shader_binding_table->mapped)
                    + table_offset + record * stride,
                handle_bytes.data() + group * handle_size, handle_size);
            // A trace names exactly one raygen record, so each raygen is its
            // own single-record region.
            if (groups[group].kind == RayTracingGroups::Kind::Raygen)
                result.raygen_regions.emplace_back(groups[group].raygen, vk::StridedDeviceAddressRegionKHR{
                    base + static_cast<vk::DeviceSize>(record) * stride, stride, stride});
            ++record;
        }
        const std::size_t count = record - first;
        return count ? vk::StridedDeviceAddressRegionKHR{
            base + static_cast<vk::DeviceSize>(first) * stride, stride,
            static_cast<vk::DeviceSize>(count) * stride} : vk::StridedDeviceAddressRegionKHR{};
    };
    write_region(raygen_records);
    result.miss_region = write_region(miss_records);
    result.hit_region = write_region(hit_records);
    result.callable_region = write_region(callable_records);
    vmaFlushAllocation(allocator_, result.shader_binding_table->allocation, 0,
        result.shader_binding_table->size);
}

AccelerationStructure DeviceImpl::build_blas(const std::span<const TriangleGeometry> geometry,
    const AccelerationStructureBuildMode mode) {
    if (!acceleration_structure_supported_)
        throw Error(ErrorCode::UnsupportedFeature,
            "acceleration structures are not enabled on this noorrhi::Device");
    if (geometry.empty())
        throw Error(ErrorCode::InvalidArgument, "BLAS requires at least one triangle geometry");

    std::vector<vk::AccelerationStructureGeometryKHR> geometries;
    std::vector<std::shared_ptr<BufferImpl>> source_buffers;
    std::vector<std::uint32_t> primitive_counts;
    geometries.reserve(geometry.size());
    primitive_counts.reserve(geometry.size());
    for (const auto& item : geometry) {
        if (!item.positions.address || !item.indices.address || item.triangle_count == 0)
            throw Error(ErrorCode::InvalidArgument, "BLAS geometry contains an empty GPU address or triangle count");
        const auto positions_buffer = find_buffer_resource(item.positions.address);
        // The positions are read as R32G32B32, which needs three floats, not
        // the padded size of noorrhi::float3.
        if (item.stride < 3 * sizeof(float) || positions_buffer->size < item.stride)
            throw Error(ErrorCode::InvalidArgument,
                "BLAS position buffer is smaller than one strided vertex");
        const std::size_t vertex_count = positions_buffer->size / item.stride;
        if (vertex_count > std::numeric_limits<std::uint32_t>::max())
            throw Error(ErrorCode::InvalidArgument, "BLAS position buffer has too many vertices");
        source_buffers.push_back(positions_buffer);
        source_buffers.push_back(find_buffer_resource(item.indices.address));
        const vk::AccelerationStructureGeometryTrianglesDataKHR triangles{
            vk::Format::eR32G32B32Sfloat,
            vk::DeviceOrHostAddressConstKHR{item.positions.address}, item.stride,
            static_cast<std::uint32_t>(vertex_count - 1),
            vk::IndexType::eUint32,
            vk::DeviceOrHostAddressConstKHR{item.indices.address}, {}};
        geometries.emplace_back(vk::GeometryTypeKHR::eTriangles, triangles,
            item.opaque ? vk::GeometryFlagBitsKHR::eOpaque
                        : vk::GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation);
        primitive_counts.push_back(item.triangle_count);
    }

    vk::BuildAccelerationStructureFlagsKHR build_flags{};
    const bool allow_update = mode == AccelerationStructureBuildMode::Dynamic;
    if (allow_update)
        build_flags |= vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
    build_flags |= allow_update
        ? vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild
        : vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
    const vk::AccelerationStructureBuildGeometryInfoKHR size_info{
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        build_flags,
        vk::BuildAccelerationStructureModeKHR::eBuild, {}, {}, geometries, {}, {}};
    const auto sizes = vk_device().getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, size_info, primitive_counts);
    auto result = std::make_shared<AccelerationStructureImpl>();
    result->device = self_.lock();
    result->blas_build_flags = build_flags;
    result->storage = create_buffer(sizes.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_MEMORY_USAGE_GPU_ONLY, false);
    const vk::AccelerationStructureCreateInfoKHR create_info({}, result->storage->buffer, 0,
        sizes.accelerationStructureSize, vk::AccelerationStructureTypeKHR::eBottomLevel);
    result->acceleration_structure = vk_device().createAccelerationStructureKHRUnique(create_info);
    auto scratch = create_buffer(sizes.buildScratchSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_MEMORY_USAGE_GPU_ONLY, false,
        acceleration_structure_properties_.minAccelerationStructureScratchOffsetAlignment);
    const auto acceleration_structure = *result->acceleration_structure;
    // Not waited for: queue order and the full barriers around every
    // submission put the build before any later build or trace, and the
    // submission holds the storage and scratch until it retires.
    submit([this, acceleration_structure, scratch, geometries, primitive_counts, build_flags]
        (const vk::CommandBuffer command) mutable {
        const vk::AccelerationStructureBuildGeometryInfoKHR build_info{
            vk::AccelerationStructureTypeKHR::eBottomLevel,
            build_flags,
            vk::BuildAccelerationStructureModeKHR::eBuild, {}, acceleration_structure,
            geometries, {}, vk::DeviceOrHostAddressKHR{scratch->address}};
        std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges;
        ranges.reserve(primitive_counts.size());
        for (const auto count : primitive_counts)
            ranges.emplace_back(count, 0, 0, 0);
        // One pointer per build, to that build's range for each of its geometries.
        const vk::AccelerationStructureBuildRangeInfoKHR* geometry_ranges = ranges.data();
        begin_label(command, "BLAS Build");
        command.buildAccelerationStructuresKHR(build_info, geometry_ranges);
        end_label(command);
        const vk::MemoryBarrier2 ready{
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            vk::AccessFlagBits2::eAccelerationStructureReadKHR
                | vk::AccessFlagBits2::eAccelerationStructureWriteKHR};
        command.pipelineBarrier2({{}, ready, {}, {}});
    }, {result->storage, scratch});
    result->updateable = allow_update;
    if (allow_update)
        result->update_scratch = create_buffer(sizes.updateScratchSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            VMA_MEMORY_USAGE_GPU_ONLY, false,
            acceleration_structure_properties_.minAccelerationStructureScratchOffsetAlignment);
    result->blas_geometries = geometries;
    result->blas_primitive_counts = primitive_counts;
    result->blas_sources = source_buffers;
    result->address = vk_device().getAccelerationStructureAddressKHR({acceleration_structure});
    result->storage_size = result->storage->size;
    // The shader-facing handle for an acceleration structure is simply its
    // device address; OpConvertUToAccelerationStructureKHR turns it back into
    // a traceable structure, so there is no descriptor here either.
    result->handle = AccelerationStructureHandle{result->address};
    return AccelerationStructure(std::move(result));
}

AccelerationStructure DeviceImpl::build_tlas(const std::span<const Instance> instances) {
    if (!acceleration_structure_supported_)
        throw Error(ErrorCode::UnsupportedFeature, "ray tracing is not enabled on this noorrhi::Device");
    if (instances.empty())
        throw Error(ErrorCode::InvalidArgument, "TLAS requires at least one instance");

    std::vector<vk::AccelerationStructureInstanceKHR> records;
    records.reserve(instances.size());
    std::vector<std::shared_ptr<AccelerationStructureImpl>> acceleration_structures;
    acceleration_structures.reserve(instances.size());
    for (const auto& instance : instances) {
        if (!instance.blas.impl_ || !instance.blas.impl_->acceleration_structure)
            throw Error(ErrorCode::InvalidResource, "TLAS instance references an empty BLAS");
        acceleration_structures.push_back(instance.blas.impl_);
        std::array<std::array<float, 4>, 3> matrix{};
        for (std::size_t row = 0; row < 3; ++row)
            for (std::size_t column = 0; column < 4; ++column)
                matrix[row][column] = instance.transform.values[row][column];
        if (instance.custom_index > 0x00ffffffu
            || instance.shader_binding_table_offset > 0x00ffffffu)
            throw Error(ErrorCode::InvalidArgument,
                "TLAS instance custom index or SBT offset exceeds 24 bits");
        records.emplace_back(vk::TransformMatrixKHR{matrix},
            instance.custom_index, instance.mask,
            instance.shader_binding_table_offset,
            vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable,
            instance.blas.impl_->address);
    }

    auto instance_buffer = create_buffer(records.size() * sizeof(records[0]),
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress
            | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
        VMA_MEMORY_USAGE_CPU_TO_GPU, true);
    std::memcpy(instance_buffer->mapped, records.data(), records.size() * sizeof(records[0]));
    vmaFlushAllocation(allocator_, instance_buffer->allocation, 0,
        records.size() * sizeof(records[0]));
    const vk::DeviceAddress address = instance_buffer->address;
    return build_tlas_at(address, static_cast<std::uint32_t>(records.size()),
        std::move(acceleration_structures), std::move(instance_buffer));
}

AccelerationStructure DeviceImpl::build_tlas(const GpuPtr<InstanceRecord> records,
    const std::uint32_t count, const std::span<const AccelerationStructure> referenced) {
    if (!acceleration_structure_supported_)
        throw Error(ErrorCode::UnsupportedFeature, "ray tracing is not enabled on this noorrhi::Device");
    if (count == 0)
        throw Error(ErrorCode::InvalidArgument, "TLAS requires at least one instance");
    if (records.address == 0)
        throw Error(ErrorCode::InvalidArgument, "TLAS instance records have no device address");

    // The records are opaque to the host here, so the BLASes they point at are
    // only kept alive by this span. Dropping one would leave the TLAS holding
    // dangling device addresses.
    std::vector<std::shared_ptr<AccelerationStructureImpl>> acceleration_structures;
    acceleration_structures.reserve(referenced.size());
    for (const auto& structure : referenced) {
        if (!structure.impl_ || !structure.impl_->acceleration_structure)
            throw Error(ErrorCode::InvalidResource, "TLAS references an empty BLAS");
        acceleration_structures.push_back(structure.impl_);
    }
    return build_tlas_at(records.address, count, std::move(acceleration_structures), nullptr);
}

AccelerationStructure DeviceImpl::build_tlas_at(const vk::DeviceAddress records,
    const std::uint32_t count,
    std::vector<std::shared_ptr<AccelerationStructureImpl>> references,
    std::shared_ptr<BufferImpl> owned_input) {
    const vk::AccelerationStructureGeometryInstancesDataKHR instances_data{
        VK_FALSE, vk::DeviceOrHostAddressConstKHR{records}};
    const vk::AccelerationStructureGeometryKHR geometry{
        vk::GeometryTypeKHR::eInstances, instances_data, {}};
    const std::vector<std::uint32_t> primitive_counts{count};
    const vk::AccelerationStructureBuildGeometryInfoKHR size_info{
        vk::AccelerationStructureTypeKHR::eTopLevel,
        vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace
            | vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
        vk::BuildAccelerationStructureModeKHR::eBuild, {}, {}, 1, &geometry};
    const auto sizes = vk_device().getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, size_info, primitive_counts);
    auto result = std::make_shared<AccelerationStructureImpl>();
    result->device = self_.lock();
    result->primitive_count = count;
    result->updateable = true;
    result->references = std::move(references);
    result->storage = create_buffer(sizes.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_MEMORY_USAGE_GPU_ONLY, false);
    result->acceleration_structure = vk_device().createAccelerationStructureKHRUnique({{},
        result->storage->buffer, 0, sizes.accelerationStructureSize,
        vk::AccelerationStructureTypeKHR::eTopLevel});
    auto scratch = create_buffer(std::max(sizes.buildScratchSize, sizes.updateScratchSize),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_MEMORY_USAGE_GPU_ONLY, false,
        acceleration_structure_properties_.minAccelerationStructureScratchOffsetAlignment);
    const auto acceleration_structure = *result->acceleration_structure;
    submit([this, acceleration_structure, scratch, geometry, count]
        (const vk::CommandBuffer command) mutable {
        const vk::AccelerationStructureBuildGeometryInfoKHR build_info{
            vk::AccelerationStructureTypeKHR::eTopLevel,
            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace
                | vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
            vk::BuildAccelerationStructureModeKHR::eBuild, {}, acceleration_structure,
            1, &geometry, {}, vk::DeviceOrHostAddressKHR{scratch->address}};
        const vk::AccelerationStructureBuildRangeInfoKHR range{count, 0, 0, 0};
        const vk::AccelerationStructureBuildRangeInfoKHR* range_ptr = &range;
        const std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*> ranges{range_ptr};
        begin_label(command, "TLAS Build");
        command.buildAccelerationStructuresKHR(build_info, ranges);
        end_label(command);
        const vk::MemoryBarrier2 ready{
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eAccelerationStructureReadKHR};
        command.pipelineBarrier2({{}, ready, {}, {}});
    }, owned_input ? std::vector<std::shared_ptr<void>>{result->storage, scratch, owned_input}
                   : std::vector<std::shared_ptr<void>>{result->storage, scratch});
    // Not waited for: like a BLAS build, queue order and the barrier above
    // order it before every trace that reads it.
    result->address = vk_device().getAccelerationStructureAddressKHR({acceleration_structure});
    result->storage_size = result->storage->size;
    // The shader-facing handle for an acceleration structure is simply its
    // device address; OpConvertUToAccelerationStructureKHR turns it back into
    // a traceable structure, so there is no descriptor here either.
    result->handle = AccelerationStructureHandle{result->address};
    result->update_input = std::move(owned_input);
    result->update_scratch = scratch;
    return AccelerationStructure(std::move(result));
}

void DeviceImpl::refit_blas(AccelerationStructure& blas) {
    auto target = blas.impl_;
    if (!target || !target->acceleration_structure || !target->updateable
        || target->blas_geometries.empty())
        throw Error(ErrorCode::InvalidResource, "BLAS was not built for updates");
    auto scratch = target->update_scratch;
    if (!scratch)
        throw Error(ErrorCode::InvalidResource, "BLAS update scratch storage is invalid");
    const auto acceleration_structure = *target->acceleration_structure;
    const auto geometries = target->blas_geometries;
    const auto primitive_counts = target->blas_primitive_counts;
    const auto build_flags = target->blas_build_flags;
    submit([this, acceleration_structure, scratch, geometries, primitive_counts, build_flags]
        (const vk::CommandBuffer command) mutable {
        const vk::AccelerationStructureBuildGeometryInfoKHR build_info{
            vk::AccelerationStructureTypeKHR::eBottomLevel,
            build_flags,
            vk::BuildAccelerationStructureModeKHR::eUpdate, acceleration_structure,
            acceleration_structure, geometries, {},
            vk::DeviceOrHostAddressKHR{scratch->address}};
        std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges;
        ranges.reserve(primitive_counts.size());
        for (const auto count : primitive_counts)
            ranges.emplace_back(count, 0, 0, 0);
        // One pointer per build, to that build's range for each of its geometries.
        const vk::AccelerationStructureBuildRangeInfoKHR* geometry_ranges = ranges.data();
        begin_label(command, "BLAS Refit");
        command.buildAccelerationStructuresKHR(build_info, geometry_ranges);
        end_label(command);
        const vk::MemoryBarrier2 ready{
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR
                | vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
            vk::AccessFlagBits2::eAccelerationStructureReadKHR};
        command.pipelineBarrier2({{}, ready, {}, {}});
    }, {target, scratch});
}

void DeviceImpl::update_tlas(AccelerationStructure& tlas,
    const std::span<const Instance> instances) {
    auto target = tlas.impl_;
    if (!target || !target->acceleration_structure || !target->updateable)
        throw Error(ErrorCode::InvalidResource, "TLAS was not built for updates");
    if (instances.size() != target->primitive_count)
        throw Error(ErrorCode::InvalidArgument, "TLAS update cannot change the instance count");

    std::vector<vk::AccelerationStructureInstanceKHR> records;
    std::vector<std::shared_ptr<AccelerationStructureImpl>> references;
    records.reserve(instances.size());
    references.reserve(instances.size());
    for (const auto& instance : instances) {
        if (!instance.blas.impl_ || !instance.blas.impl_->acceleration_structure)
            throw Error(ErrorCode::InvalidResource, "TLAS instance references an empty BLAS");
        if (instance.custom_index > 0x00ffffffu
            || instance.shader_binding_table_offset > 0x00ffffffu)
            throw Error(ErrorCode::InvalidArgument,
                "TLAS instance custom index or SBT offset exceeds 24 bits");
        references.push_back(instance.blas.impl_);
        std::array<std::array<float, 4>, 3> matrix{};
        for (std::size_t row = 0; row < 3; ++row)
            for (std::size_t column = 0; column < 4; ++column)
                matrix[row][column] = instance.transform.values[row][column];
        records.emplace_back(vk::TransformMatrixKHR{matrix}, instance.custom_index,
            instance.mask, instance.shader_binding_table_offset,
            vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable,
            instance.blas.impl_->address);
    }

    auto instance_buffer = target->update_input;
    if (!instance_buffer || instance_buffer->size
            < records.size() * sizeof(records[0]))
        throw Error(ErrorCode::InvalidResource, "TLAS update input storage is invalid");
    upload(instance_buffer, records.data(), records.size() * sizeof(records[0]), 0);
    update_tlas_at(tlas, instance_buffer->address,
        static_cast<std::uint32_t>(records.size()));
    target->references = std::move(references);
}

void DeviceImpl::update_tlas(AccelerationStructure& tlas,
    const GpuPtr<InstanceRecord> records, const std::uint32_t count) {
    if (records.address == 0)
        throw Error(ErrorCode::InvalidArgument, "TLAS instance records have no device address");
    update_tlas_at(tlas, records.address, count);
}

void DeviceImpl::update_tlas_at(AccelerationStructure& tlas,
    const vk::DeviceAddress records, const std::uint32_t count) {
    auto target = tlas.impl_;
    if (!target || !target->acceleration_structure || !target->updateable)
        throw Error(ErrorCode::InvalidResource, "TLAS was not built for updates");
    if (count != target->primitive_count)
        throw Error(ErrorCode::InvalidArgument, "TLAS update cannot change the instance count");
    const vk::AccelerationStructureGeometryInstancesDataKHR instances_data{
        VK_FALSE, vk::DeviceOrHostAddressConstKHR{records}};
    const vk::AccelerationStructureGeometryKHR geometry{
        vk::GeometryTypeKHR::eInstances, instances_data, {}};
    const auto flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace
        | vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
    auto scratch = target->update_scratch;
    if (!scratch)
        throw Error(ErrorCode::InvalidResource, "TLAS update scratch storage is invalid");
    const auto acceleration_structure = *target->acceleration_structure;
    const GpuToken token = submit([this, acceleration_structure, scratch,
        geometry, flags, count](const vk::CommandBuffer command) mutable {
        const vk::AccelerationStructureBuildGeometryInfoKHR build_info{
            vk::AccelerationStructureTypeKHR::eTopLevel, flags,
            vk::BuildAccelerationStructureModeKHR::eUpdate, acceleration_structure,
            acceleration_structure, 1, &geometry, {},
            vk::DeviceOrHostAddressKHR{scratch->address}};
        const vk::AccelerationStructureBuildRangeInfoKHR range{count, 0, 0, 0};
        const vk::AccelerationStructureBuildRangeInfoKHR* range_ptr = &range;
        begin_label(command, "TLAS Update");
        command.buildAccelerationStructuresKHR(build_info, range_ptr);
        end_label(command);
        const vk::MemoryBarrier2 ready{
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
            vk::PipelineStageFlagBits2::eComputeShader
                | vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
            vk::AccessFlagBits2::eAccelerationStructureReadKHR};
        command.pipelineBarrier2({{}, ready, {}, {}});
    }, {target, scratch});
    // Queue ordering protects input/scratch reuse and subsequent traversal.
}
void DeviceImpl::upload(const std::shared_ptr<BufferImpl>& destination, const void* data,
    const std::size_t bytes, const std::size_t destination_offset) {
    if (!destination || destination->device.get() != this || !data
        || destination_offset > destination->size || bytes > destination->size - destination_offset)
        throw Error(ErrorCode::InvalidArgument, "invalid GPU upload range");
    if (!bytes) return;
    if (recording_frame())
        throw Error(ErrorCode::InvalidState, "upload resources before beginning a frame");
    auto staging = create_buffer(bytes, vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_CPU_TO_GPU, true);
    std::memcpy(staging->mapped, data, bytes);
    if (vmaFlushAllocation(allocator_, staging->allocation, 0, bytes) != VK_SUCCESS)
        throw Error(ErrorCode::DeviceLost, "flushing GPU upload failed");
    submit([=](vk::CommandBuffer command) {
        command.copyBuffer(staging->buffer, destination->buffer,
            vk::BufferCopy(0, destination_offset, bytes));
    }, {staging, destination});
}

void DeviceImpl::download(const std::shared_ptr<BufferImpl>& source, void* data,
    const std::size_t bytes, const std::size_t source_offset) {
    if (!source || source->device.get() != this || !data
        || source_offset > source->size || bytes > source->size - source_offset)
        throw Error(ErrorCode::InvalidArgument, "invalid GPU download range");
    if (!bytes) return;
    if (recording_frame())
        throw Error(ErrorCode::InvalidState, "read back resources after ending a frame");
    auto staging = create_buffer(bytes, vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_GPU_TO_CPU, true);
    const auto token = submit([=](vk::CommandBuffer command) {
        command.copyBuffer(source->buffer, staging->buffer,
            vk::BufferCopy(source_offset, 0, bytes));
    }, {source, staging});
    wait(token);
    if (vmaInvalidateAllocation(allocator_, staging->allocation, 0, bytes) != VK_SUCCESS)
        throw Error(ErrorCode::DeviceLost, "invalidating GPU readback failed");
    std::memcpy(data, staging->mapped, bytes);
}

void DeviceImpl::record_copy_image(const vk::CommandBuffer command,
    ImageImpl& source, ImageImpl& destination) {
    // Both images stay in GENERAL for the copy, so only the memory dependency
    // has to be expressed.
    const auto memory_dependency = [&command](const vk::PipelineStageFlags2 source_stage,
        const vk::AccessFlags2 source_access, const vk::PipelineStageFlags2 destination_stage,
        const vk::AccessFlags2 destination_access) {
        vk::MemoryBarrier2 barrier{};
        barrier.setSrcStageMask(source_stage)
            .setSrcAccessMask(source_access)
            .setDstStageMask(destination_stage)
            .setDstAccessMask(destination_access);
        command.pipelineBarrier2({{}, barrier, {}, {}});
    };
    memory_dependency(vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite);
    if (source.width == destination.width && source.height == destination.height
        && source.format == destination.format) {
        command.copyImage(source.image, vk::ImageLayout::eGeneral,
            destination.image, vk::ImageLayout::eGeneral,
            vk::ImageCopy{{vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0},
                {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0},
                {source.width, source.height, 1}});
    } else {
        // Differing extents or formats need a blit, which also gives us the
        // filtered downscale a viewport composite wants.
        const vk::ImageBlit region{
            {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            {vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<std::int32_t>(source.width),
                static_cast<std::int32_t>(source.height), 1}},
            {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            {vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<std::int32_t>(destination.width),
                static_cast<std::int32_t>(destination.height), 1}}};
        command.blitImage(source.image, vk::ImageLayout::eGeneral,
            destination.image, vk::ImageLayout::eGeneral, region, vk::Filter::eLinear);
    }
    memory_dependency(vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
}

void DeviceImpl::copy_image(const ImageHandle source, const ImageHandle destination) {
    const auto from = find_image(source);
    const auto to = find_image(destination);
    if (!from || !to)
        throw Error(ErrorCode::InvalidResource, "image copy given a handle that is not live");
    submit([this, from, to](const vk::CommandBuffer command) {
        record_copy_image(command, *from, *to);
    }, {from, to});
}

void ComputePipelineImpl::launch(const DispatchSize groups, const void* args,
    const std::size_t size) const {
    if (!device || !pipeline || !args || size == 0
        || groups.x == 0 || groups.y == 0 || groups.z == 0)
        throw Error(ErrorCode::InvalidArgument, "invalid compute dispatch");
    auto self = const_cast<ComputePipelineImpl*>(this)->shared_from_this();
    device->submit([self, groups, args, size](const vk::CommandBuffer command) {
        self->device->record_compute(*self, command, groups, args, size);
    }, {self}, DeviceImpl::Ordering::Explicit);
}

void ComputePipelineImpl::launch_indirect(const GpuPtr<DispatchArgs> args,
    const void* argument_data, const std::size_t argument_size) const {
    if (!device || !pipeline || args.address == 0 || !argument_data || argument_size == 0)
        throw Error(ErrorCode::InvalidArgument, "invalid indirect compute dispatch");
    const auto resource = device->find_buffer_resource(args.address);
    const vk::Buffer buffer = resource->buffer;
    const vk::DeviceSize offset = args.address - resource->address;
    auto self = const_cast<ComputePipelineImpl*>(this)->shared_from_this();
    device->submit([self, buffer, offset, argument_data, argument_size]
        (const vk::CommandBuffer command) {
        DeviceImpl& device_impl = *self->device;
        // The dispatch dimensions come from the GPU, but the shader still
        // reads its root arguments through the same push-data path.
        const vk::DeviceAddress root = device_impl.stage_arguments(argument_data, argument_size);
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *self->pipeline);
        device_impl.bind_heaps(command);
        device_impl.push_root(command, root);
        command.dispatchIndirect(buffer, offset);
    }, std::vector<std::shared_ptr<void>>{self, resource});
}

BufferImpl::~BufferImpl() {
    if (!device || !allocation)
        return;
    device->retire([allocator = device->allocator_, buffer = this->buffer,
        allocation = this->allocation, unmap = mapped_by_api] {
        if (unmap)
            vmaUnmapMemory(allocator, allocation);
        if (buffer)
            vmaDestroyBuffer(allocator, buffer, allocation);
    });
}

SamplerImpl::~SamplerImpl() {
    if (!device || !handle)
        return;
    device->retire([owner = device.get(), slot = handle.value] {
        owner->release_slot(owner->sampler_heap_, slot);
    });
}

AccelerationStructureImpl::~AccelerationStructureImpl() {
    if (!device || !acceleration_structure)
        return;
    device->retire([acceleration = acceleration_structure.release(),
        storage = std::move(storage), vk_device = device->device()] {
        if (acceleration)
            vk_device.destroyAccelerationStructureKHR(acceleration);
    });
}

std::shared_ptr<BufferImpl> make_buffer(const std::shared_ptr<DeviceImpl>& device,
    const std::size_t size, const std::size_t alignment) {
    if (!device)
        throw Error(ErrorCode::InvalidResource, "empty GPU device");
    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
            | vk::BufferUsageFlagBits::eTransferSrc
            | vk::BufferUsageFlagBits::eTransferDst
            | vk::BufferUsageFlagBits::eIndirectBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress;
    if (device->acceleration_structure_supported())
        usage |= vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    return device->create_buffer(size, usage, VMA_MEMORY_USAGE_GPU_ONLY, false, alignment);
}

void upload_buffer(const std::shared_ptr<BufferImpl>& destination, const void* data,
    const std::size_t bytes, const std::size_t offset) {
    destination->device->upload(destination, data, bytes, offset);
}

void download_buffer(const std::shared_ptr<BufferImpl>& source, void* data, const std::size_t bytes,
    const std::size_t offset) {
    source->device->download(source, data, bytes, offset);
}

std::uint64_t buffer_address(const std::shared_ptr<BufferImpl>& buffer) {
    return buffer ? buffer->address : 0;
}

std::uint32_t sampler_handle(const std::shared_ptr<SamplerImpl>& sampler) {
    return sampler ? sampler->handle.value : 0;
}

AccelerationStructureHandle acceleration_structure_handle(
    const std::shared_ptr<AccelerationStructureImpl>& acceleration_structure) {
    return acceleration_structure ? acceleration_structure->handle : AccelerationStructureHandle{};
}

} // namespace noorrhi::detail

namespace noorrhi {

Device::Device(const DeviceConfig& config)
    : impl_(std::make_shared<detail::DeviceImpl>(config)) {
    impl_->attach_self(impl_);
    impl_->initialize_resources();
}
Device::~Device() {
    if (impl_)
        impl_->shutdown();
}
Device::Device(Device&&) noexcept = default;
Device& Device::operator=(Device&& other) noexcept {
    if (this != &other) {
        if (impl_)
            impl_->shutdown();
        impl_ = std::move(other.impl_);
    }
    return *this;
}

DeviceFeatures Device::features() const { return impl_->features(); }
DeviceInfo Device::info() const { return impl_->info(); }
MemoryReport Device::memory_report() const { return impl_->memory_report(); }

Shader Device::create_shader(const std::span<const std::byte> spirv) {
    return create_shader(spirv, "main");
}
Shader Device::create_shader(const std::span<const std::byte> spirv, const std::string_view entry_point) {
    auto shader = impl_->create_shader(spirv, entry_point);
    return Shader(shader, shader->entry_point);
}
ComputePipeline Device::compute(const Shader& shader) {
    return ComputePipeline(impl_->create_compute(shader));
}
GraphicsPipeline Device::graphics(const GraphicsPipelineDesc& desc) {
    return GraphicsPipeline(impl_->create_graphics(desc));
}
RayTracingLibrary Device::ray_tracing_library(const RayTracingPipelineDesc& desc,
    const RayTracingInterface& interface) {
    return RayTracingLibrary(impl_->create_ray_tracing_library(desc, interface));
}
RayTracingPipeline Device::ray_tracing(const std::span<const RayTracingLibrary> libraries,
    const std::span<const std::uint32_t> hit_groups) {
    std::vector<std::shared_ptr<detail::RayTracingLibraryImpl>> impls;
    impls.reserve(libraries.size());
    for (const RayTracingLibrary& library : libraries)
        impls.push_back(library.impl_);
    return RayTracingPipeline(impl_->link_ray_tracing(impls, hit_groups));
}

RayTracingPipeline Device::ray_tracing(const RayTracingPipeline& linked,
    const std::span<const std::uint32_t> hit_groups) {
    if (!linked.impl_)
        throw Error(ErrorCode::InvalidResource, "ray-tracing pipeline is empty");
    return RayTracingPipeline(impl_->rebind_hit_groups(*linked.impl_, hit_groups));
}
AccelerationStructure Device::build_blas(const std::span<const TriangleGeometry> geometry,
    const AccelerationStructureBuildMode mode) {
    return impl_->build_blas(geometry, mode);
}

void Device::refit_blas(AccelerationStructure& blas) {
    impl_->refit_blas(blas);
}
AccelerationStructure Device::build_tlas(const std::span<const Instance> instances) {
    return impl_->build_tlas(instances);
}
void Device::update_tlas(AccelerationStructure& tlas,
    const std::span<const Instance> instances) {
    impl_->update_tlas(tlas, instances);
}
AccelerationStructure Device::build_tlas(const GpuPtr<InstanceRecord> records,
    const std::uint32_t count, const std::span<const AccelerationStructure> referenced) {
    return impl_->build_tlas(records, count, referenced);
}
void Device::update_tlas(AccelerationStructure& tlas, const GpuPtr<InstanceRecord> records,
    const std::uint32_t count) {
    impl_->update_tlas(tlas, records, count);
}
Sampler Device::sampler(const SamplerDesc& desc) {
    return Sampler(impl_->create_sampler(desc));
}
void Device::barrier(const Stage source, const Stage destination) { impl_->barrier(source, destination); }
GpuToken Device::signal() { return impl_->signal(); }

StagedArguments Device::stage_bytes(const void* args, const std::size_t size) {
    std::uint64_t id = 0;
    const vk::DeviceAddress address = impl_->stage_held(args, size, id);
    return StagedArguments(impl_, id, address);
}

StagedArguments::StagedArguments(StagedArguments&& other) noexcept
    : device_(std::move(other.device_)), id_(std::exchange(other.id_, 0))
    , address_(std::exchange(other.address_, 0)) {}

StagedArguments& StagedArguments::operator=(StagedArguments&& other) noexcept {
    if (this != &other) {
        release();
        device_ = std::move(other.device_);
        id_ = std::exchange(other.id_, 0);
        address_ = std::exchange(other.address_, 0);
    }
    return *this;
}

StagedArguments::~StagedArguments() { release(); }

void StagedArguments::release() noexcept {
    if (device_ && id_ != 0)
        device_->release_staged(id_);
    device_.reset();
    id_ = 0;
}
void Device::wait(const GpuToken token) { impl_->wait(token); }
void Device::queue_wait(const GpuToken token) { impl_->queue_wait(token); }
bool Device::finished(const GpuToken token) const { return impl_->finished(token); }
void Device::synchronize() { impl_->synchronize(); }

QueueScope::QueueScope(Device& device, const Queue queue) : device_(device.impl_.get()) {
    device_->bind_queue(queue);
}

QueueScope::~QueueScope() {
    device_->unbind_queue();
}

Recording::Recording(Device& device) : impl_(device.impl_->begin_recording()) {}

Recording::~Recording() {
    if (impl_->command)
        impl_->device->abandon_recording(*impl_);
}

GpuToken Recording::submit() {
    return impl_->device->end_recording(*impl_);
}
void Device::render(const RenderTarget& target, const std::function<void()>& draw_commands) {
    impl_->render(target, draw_commands);
}
void Device::copy(const ImageHandle source, const ImageHandle destination) {
    impl_->copy_image(source, destination);
}
TimestampQuery Device::timestamp() {
    return TimestampQuery(impl_->create_timestamp());
}
void Device::measure(const TimestampQuery& query, const std::function<void()>& commands) {
    if (!query.impl_)
        throw Error(ErrorCode::InvalidResource, "timestamp query is empty");
    impl_->measure(query.impl_, commands);
}
void Device::label(const std::string_view name, const std::function<void()>& commands) {
    impl_->label(name, commands);
}
double TimestampQuery::milliseconds() const {
    if (!impl_ || !impl_->device)
        return 0.0;
    impl_->milliseconds = impl_->device->timestamp_milliseconds(*impl_);
    return impl_->milliseconds;
}

void ComputePipeline::launch_bytes(const DispatchSize groups, const void* args, const std::size_t size) const {
    if (!impl_)
        throw Error(ErrorCode::InvalidResource, "compute pipeline is empty");
    impl_->launch(groups, args, size);
}
void ComputePipeline::launch_indirect_bytes(const GpuPtr<DispatchArgs> groups,
    const void* args, const std::size_t size) const {
    if (!impl_)
        throw Error(ErrorCode::InvalidResource, "compute pipeline is empty");
    impl_->launch_indirect(groups, args, size);
}

void detail::DeviceImpl::record_ray_tracing(const detail::RayTracingPipelineImpl& pipeline,
    const vk::StridedDeviceAddressRegionKHR& raygen,
    const vk::CommandBuffer command, const DispatchSize groups,
    const void* args, const std::size_t size) {
    if (!pipeline.pipeline || !args || size == 0
        || groups.x == 0 || groups.y == 0 || groups.z == 0)
        throw Error(ErrorCode::InvalidArgument, "invalid ray-tracing dispatch");
    const vk::DeviceAddress root = stage_arguments(args, size);
    command.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, **pipeline.pipeline);
    bind_heaps(command);
    push_root(command, root);
    command.traceRaysKHR(raygen, pipeline.miss_region,
        pipeline.hit_region, pipeline.callable_region, groups.x, groups.y, groups.z);
}

void detail::RayTracingPipelineImpl::trace(const ShaderImpl& raygen, const DispatchSize groups,
    const void* args, const std::size_t size) const {
    if (!device || !pipeline || !args || size == 0 || groups.x == 0 || groups.y == 0 || groups.z == 0)
        throw Error(ErrorCode::InvalidArgument, "invalid ray-tracing dispatch");
    const auto found = std::ranges::find(raygen_regions, &raygen,
        &std::pair<const ShaderImpl*, vk::StridedDeviceAddressRegionKHR>::first);
    if (found == raygen_regions.end())
        throw Error(ErrorCode::InvalidArgument, "shader is not a ray-generation shader of this pipeline");
    auto self = const_cast<RayTracingPipelineImpl*>(this)->shared_from_this();
    device->submit([self, region = found->second, groups, args, size](const vk::CommandBuffer command) {
        self->device->record_ray_tracing(*self, region, command, groups, args, size);
    }, std::vector<std::shared_ptr<void>>{self, shader_binding_table},
        DeviceImpl::Ordering::Explicit);
}

void RayTracingPipeline::trace_bytes(const Shader& raygen, const DispatchSize size,
    const void* args, const std::size_t bytes) const {
    if (!impl_)
        throw Error(ErrorCode::InvalidResource, "ray-tracing pipeline is empty");
    if (!raygen.impl_)
        throw Error(ErrorCode::InvalidResource, "ray-generation shader is empty");
    impl_->trace(*raygen.impl_, size, args, bytes);
}

AccelerationStructureHandle AccelerationStructure::handle() const noexcept {
    return detail::acceleration_structure_handle(impl_);
}

AccelerationStructure::operator bool() const noexcept {
    return static_cast<bool>(handle());
}


} // namespace noorrhi

namespace noorrhi {

namespace interop {

DeviceHandles device_handles(Device& device) {
    if (!device.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot read handles from an empty noorrhi::Device");
    return device.impl_->native_handles();
}

void record(Device& device, const std::function<void(std::uintptr_t)>& commands) {
    if (!device.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot record through an empty noorrhi::Device");
    if (!commands)
        throw Error(ErrorCode::InvalidArgument, "native command recording requires a callback");
    device.impl_->record_native(commands);
}

std::uintptr_t image(Device& device, const ImageHandle handle) {
    if (!device.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot inspect through an empty noorrhi::Device");
    const auto found = device.impl_->find_image(handle);
    if (!found)
        throw Error(ErrorCode::InvalidResource, "GPU image handle is not live");
    return reinterpret_cast<std::uintptr_t>(static_cast<VkImage>(found->image));
}

std::uintptr_t image_view(Device& device, const ImageHandle handle) {
    if (!device.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot inspect through an empty noorrhi::Device");
    const auto image = device.impl_->find_image(handle);
    if (!image)
        throw Error(ErrorCode::InvalidResource, "GPU image handle is not live");
    return reinterpret_cast<std::uintptr_t>(static_cast<VkImageView>(*image->view));
}

ExternalImageMemory export_image_memory(Device& device, const ImageHandle handle) {
    if (!device.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot export from an empty noorrhi::Device");
    return device.impl_->export_image_memory(handle);
}

ExternalSemaphore signal_external(Device& device) {
    if (!device.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot export from an empty noorrhi::Device");
    return device.impl_->signal_external();
}

std::uintptr_t command_buffer(const Frame& frame) {
    if (!frame.impl_ || !frame.impl_->open)
        throw Error(ErrorCode::InvalidState,
            "interop::command_buffer requires a frame between begin_frame and end_frame");
    return reinterpret_cast<std::uintptr_t>(
        static_cast<VkCommandBuffer>(frame.impl_->command));
}

std::uint32_t native_format(const ImageFormat format) {
    return static_cast<std::uint32_t>(detail::to_vulkan_format(format));
}

} // namespace interop
} // namespace noorrhi
