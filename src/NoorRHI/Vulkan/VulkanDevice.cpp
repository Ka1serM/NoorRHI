#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>

#include "NoorRHI/Present/Swapchain.h"
#include "NoorRHI/Vulkan/Vulkan.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace NoorRHI::Backend {

namespace {

vk::Bool32 VKAPI_CALL printValidationMessage(const vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT,
                                             const vk::DebugUtilsMessengerCallbackDataEXT* data, void*) {
    const char* label = severity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ? "error" : "warning";
    std::fprintf(stderr, "[NoorRHI validation %s] %s\n", label, data->pMessage);
    return vk::False;
}

bool hasExtension(const std::span<const vk::ExtensionProperties> extensions, const std::string_view name) {
    return std::ranges::any_of(extensions, [name](const vk::ExtensionProperties& extension) { return std::string_view(extension.extensionName) == name; });
}

struct Capabilities {
    bool mandatory = false;
    bool rayTracing = false;
};

Capabilities probe(const vk::PhysicalDevice candidate) {
    if (candidate.getProperties().apiVersion < VK_API_VERSION_1_3)
        return {};
    std::vector<vk::ExtensionProperties> extensions = candidate.enumerateDeviceExtensionProperties().value;
    const bool hasAccelerationStructure =
        hasExtension(extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) && hasExtension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    const bool hasRayQuery = hasAccelerationStructure && hasExtension(extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME);
    const bool hasRayTracing = hasAccelerationStructure && hasExtension(extensions, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
                               && hasExtension(extensions, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);

    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
                       vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceRayQueryFeaturesKHR,
                       vk::PhysicalDeviceRayTracingPipelineFeaturesKHR, vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT>
        chain;
    candidate.getFeatures2(&chain.get<vk::PhysicalDeviceFeatures2>());
    const auto& base = chain.get<vk::PhysicalDeviceFeatures2>().features;
    const auto& features11 = chain.get<vk::PhysicalDeviceVulkan11Features>();
    const auto& features12 = chain.get<vk::PhysicalDeviceVulkan12Features>();
    const auto& features13 = chain.get<vk::PhysicalDeviceVulkan13Features>();

    Capabilities capabilities;
    capabilities.mandatory = features12.bufferDeviceAddress && features12.timelineSemaphore && features13.synchronization2 && features13.dynamicRendering
                             && features12.runtimeDescriptorArray && features12.descriptorBindingPartiallyBound
                             && features12.descriptorBindingUpdateUnusedWhilePending && features12.descriptorBindingSampledImageUpdateAfterBind
                             && features12.descriptorBindingStorageImageUpdateAfterBind && features12.shaderSampledImageArrayNonUniformIndexing
                             && features12.shaderStorageImageArrayNonUniformIndexing && features12.scalarBlockLayout && base.shaderInt64
                             && features11.shaderDrawParameters && base.fragmentStoresAndAtomics && base.multiDrawIndirect && base.drawIndirectFirstInstance
                             && features12.drawIndirectCount && hasExtension(extensions, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)
                             && chain.get<vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT>().shaderBufferFloat32AtomicAdd;
    capabilities.rayTracing = hasRayQuery && hasRayTracing && chain.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().accelerationStructure
                              && chain.get<vk::PhysicalDeviceRayQueryFeaturesKHR>().rayQuery
                              && chain.get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>().rayTracingPipeline;
    return capabilities;
}

std::optional<MatrixComponent> getMatrixComponent(const vk::ComponentTypeKHR type, const bool bytes) {
    switch (type) {
    case vk::ComponentTypeKHR::eFloat16:
        return MatrixComponent::Float16;
    case vk::ComponentTypeKHR::eFloat32:
        return MatrixComponent::Float32;
    case vk::ComponentTypeKHR::eSint8:
        return bytes ? std::optional(MatrixComponent::Int8) : std::nullopt;
    case vk::ComponentTypeKHR::eUint8:
        return bytes ? std::optional(MatrixComponent::Uint8) : std::nullopt;
    case vk::ComponentTypeKHR::eSint32:
        return MatrixComponent::Int32;
    case vk::ComponentTypeKHR::eUint32:
        return MatrixComponent::Uint32;
    default:
        return std::nullopt;
    }
}

std::vector<CooperativeMatrixShape> getCooperativeMatrixShapes(const vk::PhysicalDevice physicalDevice, const bool bytes) {
    std::vector<CooperativeMatrixShape> shapes;
    const auto [result, properties] = physicalDevice.getCooperativeMatrixPropertiesKHR();
    if (result != vk::Result::eSuccess)
        return shapes;
    for (const vk::CooperativeMatrixPropertiesKHR& property : properties) {
        const std::optional<MatrixComponent> a = getMatrixComponent(property.AType, bytes);
        const std::optional<MatrixComponent> b = getMatrixComponent(property.BType, bytes);
        const std::optional<MatrixComponent> c = getMatrixComponent(property.CType, bytes);
        const std::optional<MatrixComponent> output = getMatrixComponent(property.ResultType, bytes);
        if (property.scope != vk::ScopeKHR::eSubgroup || property.saturatingAccumulation || !a || !b || !c || !output)
            continue;
        const CooperativeMatrixShape shape{property.MSize, property.NSize, property.KSize, *a, *b, *c, *output};
        if (!std::ranges::contains(shapes, shape))
            shapes.push_back(shape);
    }
    return shapes;
}

vk::DeviceSize getDeviceLocalBytes(const vk::PhysicalDevice candidate) {
    const vk::PhysicalDeviceMemoryProperties memory = candidate.getMemoryProperties();
    vk::DeviceSize bytes = 0;
    for (std::uint32_t heap = 0; heap < memory.memoryHeapCount; ++heap)
        if (memory.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
            bytes += memory.memoryHeaps[heap].size;
    return bytes;
}

std::uint32_t rankDevice(const vk::PhysicalDevice candidate) {
    switch (candidate.getProperties().deviceType) {
    case vk::PhysicalDeviceType::eDiscreteGpu:
        return 3;
    case vk::PhysicalDeviceType::eIntegratedGpu:
        return 2;
    case vk::PhysicalDeviceType::eCpu:
        return 0;
    default:
        return 1;
    }
}

}

std::unexpected<Error> failVulkan(const vk::Result result, const std::string_view operation) {
    const std::string message = std::string(operation) + ": " + vk::to_string(result);
    switch (result) {
    case vk::Result::eErrorOutOfHostMemory:
    case vk::Result::eErrorOutOfDeviceMemory:
    case vk::Result::eErrorOutOfPoolMemory:
    case vk::Result::eErrorFragmentedPool:
        return fail(ErrorCode::OutOfMemory, message);
    case vk::Result::eErrorDeviceLost:
        return fail(ErrorCode::DeviceLost, message);
    case vk::Result::eErrorFeatureNotPresent:
    case vk::Result::eErrorExtensionNotPresent:
    case vk::Result::eErrorLayerNotPresent:
    case vk::Result::eErrorIncompatibleDriver:
    case vk::Result::eErrorFormatNotSupported:
        return fail(ErrorCode::Unsupported, message);
    case vk::Result::eErrorInvalidShaderNV:
        return fail(ErrorCode::InvalidShader, message);
    default:
        return fail(ErrorCode::InvalidState, message);
    }
}

std::expected<void, Error> checkVulkan(const vk::Result result, const std::string_view operation) {
    if (result != vk::Result::eSuccess)
        return failVulkan(result, operation);
    return {};
}

vk::Format getVulkanFormat(const ImageFormat format) {
    switch (format) {
    case ImageFormat::Undefined:
        return vk::Format::eUndefined;
    case ImageFormat::Rgba8Unorm:
        return vk::Format::eR8G8B8A8Unorm;
    case ImageFormat::Bgra8Unorm:
        return vk::Format::eB8G8R8A8Unorm;
    case ImageFormat::Rgba32Float:
        return vk::Format::eR32G32B32A32Sfloat;
    case ImageFormat::Rgba16Float:
        return vk::Format::eR16G16B16A16Sfloat;
    case ImageFormat::R32Uint:
        return vk::Format::eR32Uint;
    case ImageFormat::R32Float:
        return vk::Format::eR32Sfloat;
    case ImageFormat::D32Float:
        return vk::Format::eD32Sfloat;
    case ImageFormat::D32FloatS8Uint:
        return vk::Format::eD32SfloatS8Uint;
    case ImageFormat::R8Unorm:
        return vk::Format::eR8Unorm;
    case ImageFormat::L8Unorm:
        return vk::Format::eR8Unorm;
    case ImageFormat::L8Srgb:
        return vk::Format::eR8Srgb;
    case ImageFormat::Rg8Unorm:
        return vk::Format::eR8G8Unorm;
    case ImageFormat::Rgba8Srgb:
        return vk::Format::eR8G8B8A8Srgb;
    case ImageFormat::Bgra8Srgb:
        return vk::Format::eB8G8R8A8Srgb;
    case ImageFormat::R16Unorm:
        return vk::Format::eR16Unorm;
    case ImageFormat::Rg16Unorm:
        return vk::Format::eR16G16Unorm;
    case ImageFormat::Rgba16Unorm:
        return vk::Format::eR16G16B16A16Unorm;
    case ImageFormat::R16Float:
        return vk::Format::eR16Sfloat;
    case ImageFormat::Rg16Float:
        return vk::Format::eR16G16Sfloat;
    case ImageFormat::Rg32Float:
        return vk::Format::eR32G32Sfloat;
    case ImageFormat::A2b10g10r10Unorm:
        return vk::Format::eA2B10G10R10UnormPack32;
    case ImageFormat::B10g11r11Float:
        return vk::Format::eB10G11R11UfloatPack32;
    case ImageFormat::Bc1Unorm:
        return vk::Format::eBc1RgbaUnormBlock;
    case ImageFormat::Bc1Srgb:
        return vk::Format::eBc1RgbaSrgbBlock;
    case ImageFormat::Bc2Unorm:
        return vk::Format::eBc2UnormBlock;
    case ImageFormat::Bc2Srgb:
        return vk::Format::eBc2SrgbBlock;
    case ImageFormat::Bc3Unorm:
        return vk::Format::eBc3UnormBlock;
    case ImageFormat::Bc3Srgb:
        return vk::Format::eBc3SrgbBlock;
    case ImageFormat::Bc4Unorm:
        return vk::Format::eBc4UnormBlock;
    case ImageFormat::Bc5Unorm:
        return vk::Format::eBc5UnormBlock;
    case ImageFormat::Bc6hUfloat:
        return vk::Format::eBc6HUfloatBlock;
    case ImageFormat::Bc6hSfloat:
        return vk::Format::eBc6HSfloatBlock;
    case ImageFormat::Bc7Unorm:
        return vk::Format::eBc7UnormBlock;
    case ImageFormat::Bc7Srgb:
        return vk::Format::eBc7SrgbBlock;
    }
    return vk::Format::eUndefined;
}

void recordImageBarrier(const vk::CommandBuffer commands, const vk::Image image, const vk::ImageAspectFlags aspect, const vk::ImageLayout from,
                        const vk::ImageLayout to) {
    vk::ImageMemoryBarrier2 barrier;
    barrier.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
        .setSrcAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
        .setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
        .setDstAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
        .setOldLayout(from)
        .setNewLayout(to)
        .setSrcQueueFamilyIndex(vk::QueueFamilyIgnored)
        .setDstQueueFamilyIndex(vk::QueueFamilyIgnored)
        .setImage(image)
        .setSubresourceRange({aspect, 0, vk::RemainingMipLevels, 0, 1});
    commands.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(barrier));
}

std::expected<void, Error> PipelineCache::open(const vk::Device owner, const std::filesystem::path& directory) {
    device = owner;
    path = directory / "pipeline.cache";
    writerId = std::random_device{}();
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error)
        return fail(ErrorCode::InvalidArgument, "cannot create the pipeline cache folder " + directory.string() + ": " + error.message());
    std::ifstream file(path, std::ios::binary);
    const std::vector<char> data{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    std::expected<vk::UniquePipelineCache, Error> created =
        checkVulkan(device.createPipelineCacheUnique(vk::PipelineCacheCreateInfo({}, data.size(), data.data())), "vkCreatePipelineCache");
    if (!created)
        return std::unexpected(std::move(created.error()));
    cache = std::move(*created);
    return {};
}

std::expected<void, Error> PipelineCache::save() {
    std::lock_guard lock(saveMutex);
    std::expected<std::vector<std::uint8_t>, Error> data = checkVulkan(device.getPipelineCacheData(*cache), "vkGetPipelineCacheData");
    if (!data)
        return std::unexpected(std::move(data.error()));
    const std::filesystem::path partial = path.string() + "." + std::to_string(writerId) + ".partial";
    {
        std::ofstream file(partial, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(data->data()), static_cast<std::streamsize>(data->size()));
        if (!file)
            return fail(ErrorCode::InvalidState, "cannot write the pipeline cache file " + partial.string());
    }
    std::error_code error;
    std::filesystem::rename(partial, path, error);
    if (error)
        return fail(ErrorCode::InvalidState, "cannot replace the pipeline cache file " + path.string() + ": " + error.message());
    return {};
}

VulkanDevice::~VulkanDevice() {
    {
        std::lock_guard lock(compileMutex);
        stopCompiling = true;
    }
    compileSignal.notify_all();
    for (std::thread& thread : compileThreads)
        thread.join();
    if (!device)
        return;
    static_cast<void>(device->waitIdle());
    for (Retired& entry : retired)
        entry.release();
    retired.clear();
    for (QueueState& queue : queues) {
        for (InFlight& done : queue.inFlight)
            for (const ArgumentChunk& chunk : done.chunks)
                release(chunk.memory);
        queue.inFlight.clear();
        queue.freePools.clear();
    }
    for (const Allocation& chunk : freeChunks)
        release(chunk);
    freeChunks.clear();
    if (pipelineCache)
        static_cast<void>(pipelineCache->save());
    releaseMemoryHeaps();
}

std::expected<void, Error> VulkanDevice::open(const DeviceConfig& config) {
    presentation = config.presentation != nullptr;
    allowHardwareRayTracing = config.hardwareRayTracing;
    allowCooperativeMatrices = config.cooperativeMatrices;
    if (std::expected<void, Error> created = createInstance(config); !created)
        return created;
    std::expected<vk::UniqueSurfaceKHR, Error> probeSurface = createProbeSurface(config);
    if (!probeSurface)
        return std::unexpected(std::move(probeSurface.error()));
    if (std::expected<void, Error> selected = selectPhysicalDevice(probeSurface->get()); !selected)
        return selected;
    if (std::expected<void, Error> created = createDevice(probeSurface->get()); !created)
        return created;
    if (!config.pipelineDirectory.empty()) {
        pipelineCache.emplace();
        if (std::expected<void, Error> opened = pipelineCache->open(*device, config.pipelineDirectory); !opened)
            return opened;
    }
    if (std::expected<void, Error> created = createMemoryHeaps(); !created)
        return created;
    if (std::expected<void, Error> created = createQueues(); !created)
        return created;
    return createBindlessSet();
}

std::expected<void, Error> VulkanDevice::createInstance(const DeviceConfig& config) {
    if (!loader.success())
        return fail(ErrorCode::Unsupported, "the Vulkan loader library is not installed");
    auto getInstanceProcAddr = loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    if (config.presentation && config.presentation->getInstanceProcAddress())
        getInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(config.presentation->getInstanceProcAddress());
    VULKAN_HPP_DEFAULT_DISPATCHER.init(getInstanceProcAddr);

    std::vector<const char*> layers;
    std::vector<const char*> extensions;
    if (config.presentation)
        extensions = config.presentation->getInstanceExtensions();
    if (config.enableValidation) {
        std::expected<std::vector<vk::LayerProperties>, Error> available =
            checkVulkan(vk::enumerateInstanceLayerProperties(), "vkEnumerateInstanceLayerProperties");
        if (!available)
            return std::unexpected(std::move(available.error()));
        const bool hasValidation = std::ranges::any_of(
            *available, [](const vk::LayerProperties& layer) { return std::string_view(layer.layerName) == "VK_LAYER_KHRONOS_validation"; });
        if (!hasValidation)
            return fail(ErrorCode::Unsupported, "Vulkan validation was requested, but VK_LAYER_KHRONOS_validation is not installed");
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    std::expected<std::vector<vk::ExtensionProperties>, Error> available =
        checkVulkan(vk::enumerateInstanceExtensionProperties(), "vkEnumerateInstanceExtensionProperties");
    if (!available)
        return std::unexpected(std::move(available.error()));
    debugLabels = hasExtension(*available, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (debugLabels)
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    const std::string applicationName(config.applicationName);
    const vk::ApplicationInfo application(applicationName.c_str(), 1, "NoorRHI", 1, VK_API_VERSION_1_3);
    std::expected<vk::UniqueInstance, Error> created =
        checkVulkan(vk::createInstanceUnique(vk::InstanceCreateInfo({}, &application, layers, extensions)), "vkCreateInstance");
    if (!created)
        return std::unexpected(std::move(created.error()));
    instance = std::move(*created);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
    if (layers.empty() || !debugLabels)
        return {};
    const vk::DebugUtilsMessengerCreateInfoEXT messengerInfo(
        {}, vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning,
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        printValidationMessage);
    std::expected<vk::UniqueDebugUtilsMessengerEXT, Error> createdMessenger =
        checkVulkan(instance->createDebugUtilsMessengerEXTUnique(messengerInfo), "vkCreateDebugUtilsMessengerEXT");
    if (!createdMessenger)
        return std::unexpected(std::move(createdMessenger.error()));
    messenger = std::move(*createdMessenger);
    return {};
}

std::expected<vk::UniqueSurfaceKHR, Error> VulkanDevice::createProbeSurface(const DeviceConfig& config) {
    if (!config.presentation)
        return vk::UniqueSurfaceKHR{};
    const std::uintptr_t raw = config.presentation->createSurface(reinterpret_cast<std::uintptr_t>(static_cast<VkInstance>(*instance)));
    if (!raw)
        return fail(ErrorCode::InvalidState, "the SurfaceProvider did not create a surface");
    return vk::UniqueSurfaceKHR(vk::SurfaceKHR(reinterpret_cast<VkSurfaceKHR>(raw)), *instance);
}

std::expected<void, Error> VulkanDevice::selectPhysicalDevice(const vk::SurfaceKHR probeSurface) {
    std::expected<std::vector<vk::PhysicalDevice>, Error> candidates = checkVulkan(instance->enumeratePhysicalDevices(), "vkEnumeratePhysicalDevices");
    if (!candidates)
        return std::unexpected(std::move(candidates.error()));
    Capabilities chosen;
    std::uint32_t bestRank = 0;
    vk::DeviceSize bestMemory = 0;
    for (const vk::PhysicalDevice candidate : *candidates) {
        const Capabilities capabilities = probe(candidate);
        if (!capabilities.mandatory)
            continue;
        if (probeSurface && !hasExtension(candidate.enumerateDeviceExtensionProperties().value, VK_KHR_SWAPCHAIN_EXTENSION_NAME))
            continue;
        const std::uint32_t rank = rankDevice(candidate);
        const vk::DeviceSize memory = getDeviceLocalBytes(candidate);
        if (physicalDevice && (rank < bestRank || (rank == bestRank && memory <= bestMemory)))
            continue;
        physicalDevice = candidate;
        chosen = capabilities;
        bestRank = rank;
        bestMemory = memory;
    }
    if (!physicalDevice)
        return fail(ErrorCode::Unsupported, "no Vulkan 1.3 device supports buffer device addresses, timeline semaphores, synchronization2, dynamic "
                                            "rendering, bindless descriptor indexing, scalar block layout, shaderInt64, shaderDrawParameters, fragment stores, "
                                            "float atomic add and multi draw indirect with a count");
    features.hardwareRayTracing = chosen.rayTracing && allowHardwareRayTracing;
    const vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
    info.api = GraphicsApi::Vulkan;
    info.name = properties.deviceName.data();
    info.vendorId = properties.vendorID;
    info.deviceId = properties.deviceID;
    timestampPeriod = properties.limits.timestampPeriod;
    const auto subgroupChain =
        physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceSubgroupProperties, vk::PhysicalDeviceVulkan13Properties>();
    const auto& subgroup = subgroupChain.get<vk::PhysicalDeviceSubgroupProperties>();
    features.minSubgroupSize = subgroupChain.get<vk::PhysicalDeviceVulkan13Properties>().minSubgroupSize;
    constexpr vk::SubgroupFeatureFlags operations = vk::SubgroupFeatureFlagBits::eBasic | vk::SubgroupFeatureFlagBits::eVote
                                                    | vk::SubgroupFeatureFlagBits::eArithmetic | vk::SubgroupFeatureFlagBits::eBallot
                                                    | vk::SubgroupFeatureFlagBits::eShuffle | vk::SubgroupFeatureFlagBits::eShuffleRelative;
    constexpr vk::ShaderStageFlags stages = vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eFragment;
    features.subgroups = (subgroup.supportedOperations & operations) == operations && (subgroup.supportedStages & stages) == stages;
    if (!features.hardwareRayTracing)
        return {};
    vk::PhysicalDeviceProperties2 chain;
    chain.pNext = &accelerationProperties;
    accelerationProperties.pNext = &rayTracingProperties;
    physicalDevice.getProperties2(&chain);
    accelerationProperties.pNext = nullptr;
    return {};
}

std::expected<void, Error> VulkanDevice::createDevice(const vk::SurfaceKHR probeSurface) {
    const std::vector<vk::QueueFamilyProperties> families = physicalDevice.getQueueFamilyProperties();
    std::optional<std::uint32_t> family;
    for (std::uint32_t index = 0; index < families.size(); ++index) {
        constexpr vk::QueueFlags required = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute;
        if ((families[index].queueFlags & required) != required)
            continue;
        if (probeSurface && !physicalDevice.getSurfaceSupportKHR(index, probeSurface).value)
            continue;
        family = index;
        break;
    }
    if (!family)
        return fail(ErrorCode::Unsupported, "the Vulkan device has no graphics and compute queue that can present to the window");
    std::uint32_t asyncFamily = *family;
    for (std::uint32_t index = 0; index < families.size(); ++index)
        if ((families[index].queueFlags & vk::QueueFlagBits::eCompute) && !(families[index].queueFlags & vk::QueueFlagBits::eGraphics)) {
            asyncFamily = index;
            break;
        }
    const bool dedicated = asyncFamily != *family;
    const std::uint32_t asyncIndex = dedicated || families[*family].queueCount == 1 ? 0 : 1;
    constexpr std::array priorities{1.0f, 0.5f};
    std::vector<vk::DeviceQueueCreateInfo> queueInfos{vk::DeviceQueueCreateInfo({}, *family, dedicated ? 1 : asyncIndex + 1, priorities.data())};
    if (dedicated)
        queueInfos.emplace_back(vk::DeviceQueueCreateFlags{}, asyncFamily, 1, &priorities[1]);

    std::expected<std::vector<vk::ExtensionProperties>, Error> available =
        checkVulkan(physicalDevice.enumerateDeviceExtensionProperties(), "vkEnumerateDeviceExtensionProperties");
    if (!available)
        return std::unexpected(std::move(available.error()));
    std::vector<const char*> extensions;
    if (presentation)
        extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    externalMemoryFd = hasExtension(*available, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    if (externalMemoryFd)
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    externalSemaphoreFd = hasExtension(*available, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
    if (externalSemaphoreFd)
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);

    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features> supported;
    physicalDevice.getFeatures2(&supported.get<vk::PhysicalDeviceFeatures2>());

    vk::PhysicalDeviceFeatures2 base;
    base.features.shaderInt64 = vk::True;
    base.features.fragmentStoresAndAtomics = vk::True;
    base.features.multiDrawIndirect = vk::True;
    base.features.drawIndirectFirstInstance = vk::True;
    base.features.shaderInt16 = supported.get<vk::PhysicalDeviceFeatures2>().features.shaderInt16;
    base.features.samplerAnisotropy = supported.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy;
    base.features.fillModeNonSolid = supported.get<vk::PhysicalDeviceFeatures2>().features.fillModeNonSolid;
    vk::PhysicalDeviceVulkan11Features features11;
    features11.shaderDrawParameters = vk::True;
    features11.storageBuffer16BitAccess = supported.get<vk::PhysicalDeviceVulkan11Features>().storageBuffer16BitAccess;
    vk::PhysicalDeviceVulkan12Features features12;
    features12.shaderFloat16 = supported.get<vk::PhysicalDeviceVulkan12Features>().shaderFloat16;
    features.shaderFloat16 = features12.shaderFloat16;
    features12.shaderInt8 = supported.get<vk::PhysicalDeviceVulkan12Features>().shaderInt8;
    features12.shaderSubgroupExtendedTypes = supported.get<vk::PhysicalDeviceVulkan12Features>().shaderSubgroupExtendedTypes;
    subgroupExtendedTypes = features12.shaderSubgroupExtendedTypes;
    features12.storageBuffer8BitAccess = supported.get<vk::PhysicalDeviceVulkan12Features>().storageBuffer8BitAccess;
    features12.shaderBufferInt64Atomics = supported.get<vk::PhysicalDeviceVulkan12Features>().shaderBufferInt64Atomics;
    features12.vulkanMemoryModel = supported.get<vk::PhysicalDeviceVulkan12Features>().vulkanMemoryModel;
    features12.vulkanMemoryModelDeviceScope = supported.get<vk::PhysicalDeviceVulkan12Features>().vulkanMemoryModelDeviceScope;
    features12.bufferDeviceAddress = vk::True;
    features12.drawIndirectCount = vk::True;
    features12.timelineSemaphore = vk::True;
    features12.scalarBlockLayout = vk::True;
    features12.runtimeDescriptorArray = vk::True;
    features12.descriptorBindingPartiallyBound = vk::True;
    features12.descriptorBindingUpdateUnusedWhilePending = vk::True;
    features12.descriptorBindingSampledImageUpdateAfterBind = vk::True;
    features12.descriptorBindingStorageImageUpdateAfterBind = vk::True;
    features12.shaderSampledImageArrayNonUniformIndexing = vk::True;
    features12.shaderStorageImageArrayNonUniformIndexing = vk::True;
    vk::PhysicalDeviceVulkan13Features features13;
    features13.synchronization2 = vk::True;
    features13.dynamicRendering = vk::True;
    base.pNext = &features11;
    features11.pNext = &features12;
    features12.pNext = &features13;
    void** tail = &features13.pNext;
    const auto append = [&tail](auto& feature) {
        *tail = &feature;
        tail = &feature.pNext;
    };

    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures;
    atomicFloatFeatures.shaderBufferFloat32Atomics = vk::True;
    atomicFloatFeatures.shaderBufferFloat32AtomicAdd = vk::True;
    append(atomicFloatFeatures);
    extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);

    vk::PhysicalDeviceCooperativeMatrixFeaturesKHR cooperativeMatrixFeatures;
    if (allowCooperativeMatrices && hasExtension(*available, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) && features12.vulkanMemoryModel
        && features12.shaderFloat16 && features11.storageBuffer16BitAccess) {
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceCooperativeMatrixFeaturesKHR> cooperative;
        physicalDevice.getFeatures2(&cooperative.get<vk::PhysicalDeviceFeatures2>());
        const auto properties = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceCooperativeMatrixPropertiesKHR>();
        const bool compute = static_cast<bool>(properties.get<vk::PhysicalDeviceCooperativeMatrixPropertiesKHR>().cooperativeMatrixSupportedStages
                                               & vk::ShaderStageFlagBits::eCompute);
        if (cooperative.get<vk::PhysicalDeviceCooperativeMatrixFeaturesKHR>().cooperativeMatrix && compute)
            features.cooperativeMatrixShapes = getCooperativeMatrixShapes(physicalDevice, features12.shaderInt8 && features12.storageBuffer8BitAccess);
    }
    if (!features.cooperativeMatrixShapes.empty()) {
        cooperativeMatrixFeatures.cooperativeMatrix = vk::True;
        append(cooperativeMatrixFeatures);
        extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }

    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelerationFeatures;
    vk::PhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures;
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingFeatures;
    if (features.hardwareRayTracing) {
        accelerationFeatures.accelerationStructure = vk::True;
        rayQueryFeatures.rayQuery = vk::True;
        rayTracingFeatures.rayTracingPipeline = vk::True;
        append(accelerationFeatures);
        append(rayQueryFeatures);
        append(rayTracingFeatures);
        extensions.insert(extensions.end(),
                          {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME,
                           VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME});
    }
    vk::PhysicalDevicePresentModeFifoLatestReadyFeaturesKHR latestReadyFeatures;
    const char* latestReadyExtension =
        hasExtension(*available, VK_KHR_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME)   ? VK_KHR_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME
        : hasExtension(*available, VK_EXT_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME) ? VK_EXT_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME
                                                                                         : nullptr;
    if (presentation && latestReadyExtension) {
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDevicePresentModeFifoLatestReadyFeaturesKHR> latestReady;
        physicalDevice.getFeatures2(&latestReady.get<vk::PhysicalDeviceFeatures2>());
        fifoLatestReady = latestReady.get<vk::PhysicalDevicePresentModeFifoLatestReadyFeaturesKHR>().presentModeFifoLatestReady;
    }
    if (fifoLatestReady) {
        latestReadyFeatures.presentModeFifoLatestReady = vk::True;
        append(latestReadyFeatures);
        extensions.push_back(latestReadyExtension);
    }

    const vk::DeviceCreateInfo createInfo = vk::DeviceCreateInfo({}, queueInfos, {}, extensions).setPNext(&base);
    std::expected<vk::UniqueDevice, Error> created = checkVulkan(physicalDevice.createDeviceUnique(createInfo), "vkCreateDevice");
    if (!created)
        return std::unexpected(std::move(created.error()));
    device = std::move(*created);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
    QueueState& graphics = getQueue(Queue::Graphics);
    QueueState& async = getQueue(Queue::Async);
    graphics.family = *family;
    async.family = asyncFamily;
    graphics.queue = device->getQueue(*family, 0);
    async.queue = device->getQueue(asyncFamily, asyncIndex);
    graphics.timestampValidBits = families[*family].timestampValidBits;
    async.timestampValidBits = families[asyncFamily].timestampValidBits;
    sharedFamilies = dedicated ? std::vector{*family, asyncFamily} : std::vector{*family};
    return {};
}

std::expected<void, Error> VulkanDevice::createQueues() {
    for (QueueState& queue : queues) {
        vk::SemaphoreTypeCreateInfo timelineInfo(vk::SemaphoreType::eTimeline, 0);
        std::expected<vk::UniqueSemaphore, Error> timeline = checkVulkan(device->createSemaphoreUnique({{}, &timelineInfo}), "vkCreateSemaphore");
        if (!timeline)
            return std::unexpected(std::move(timeline.error()));
        queue.timeline = std::move(*timeline);
    }
    return {};
}

std::expected<void, Error> VulkanDevice::createBindlessSet() {
    const vk::PhysicalDeviceVulkan12Properties limits =
        physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceVulkan12Properties>().get<vk::PhysicalDeviceVulkan12Properties>();
    const std::uint32_t resourceLimit = limits.maxPerStageUpdateAfterBindResources;
    samplerSlots.capacity = std::min({limits.maxDescriptorSetUpdateAfterBindSamplers, limits.maxPerStageDescriptorUpdateAfterBindSamplers, 4096u});
    textureSlots.capacity = std::min({limits.maxDescriptorSetUpdateAfterBindSampledImages, limits.maxPerStageDescriptorUpdateAfterBindSampledImages,
                                      limits.maxDescriptorSetUpdateAfterBindStorageImages, limits.maxPerStageDescriptorUpdateAfterBindStorageImages,
                                      (resourceLimit - samplerSlots.capacity) / 2, 1u << 20});

    constexpr vk::ShaderStageFlags stages = vk::ShaderStageFlagBits::eAll;
    const std::array bindings{
        vk::DescriptorSetLayoutBinding(samplerBinding, vk::DescriptorType::eSampler, samplerSlots.capacity, stages),
        vk::DescriptorSetLayoutBinding(sampledImageBinding, vk::DescriptorType::eSampledImage, textureSlots.capacity, stages),
        vk::DescriptorSetLayoutBinding(storageImageBinding, vk::DescriptorType::eStorageImage, textureSlots.capacity, stages),
    };
    constexpr vk::DescriptorBindingFlags bindingFlags = vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind
                                                        | vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
    const std::array flags{bindingFlags, bindingFlags, bindingFlags};
    const vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo(flags);
    std::expected<vk::UniqueDescriptorSetLayout, Error> layout =
        checkVulkan(device->createDescriptorSetLayoutUnique({vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool, bindings, &flagsInfo}),
                    "vkCreateDescriptorSetLayout");
    if (!layout)
        return std::unexpected(std::move(layout.error()));
    descriptorLayout = std::move(*layout);

    const std::array sizes{
        vk::DescriptorPoolSize(vk::DescriptorType::eSampler, samplerSlots.capacity),
        vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, textureSlots.capacity),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage, textureSlots.capacity),
    };
    std::expected<vk::UniqueDescriptorPool, Error> pool =
        checkVulkan(device->createDescriptorPoolUnique({vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind, 1, sizes}), "vkCreateDescriptorPool");
    if (!pool)
        return std::unexpected(std::move(pool.error()));
    descriptorPool = std::move(*pool);
    const vk::DescriptorSetLayout setLayout = *descriptorLayout;
    std::expected<std::vector<vk::DescriptorSet>, Error> sets =
        checkVulkan(device->allocateDescriptorSets({*descriptorPool, setLayout}), "vkAllocateDescriptorSets");
    if (!sets)
        return std::unexpected(std::move(sets.error()));
    descriptorSet = sets->front();

    const vk::PushConstantRange rootRange(stages, 0, sizeof(vk::DeviceAddress));
    std::expected<vk::UniquePipelineLayout, Error> pipeline =
        checkVulkan(device->createPipelineLayoutUnique({{}, setLayout, rootRange}), "vkCreatePipelineLayout");
    if (!pipeline)
        return std::unexpected(std::move(pipeline.error()));
    pipelineLayout = std::move(*pipeline);
    return {};
}

void VulkanDevice::retire(std::function<void()> release) {
    std::lock_guard lock(mutex);
    retired.push_back({nextListId - 1, {}, false, std::move(release)});
}

std::unique_ptr<VulkanCommandList> VulkanDevice::openList(const Queue queue) {
    std::unique_ptr<VulkanCommandList> list = std::make_unique<VulkanCommandList>();
    list->device = shared_from_this();
    list->queue = queue;
    QueueState& state = getQueue(queue);
    {
        std::lock_guard lock(mutex);
        list->id = nextListId++;
        openLists.insert(list->id);
        if (!state.freePools.empty()) {
            list->pool = std::move(state.freePools.back());
            state.freePools.pop_back();
        }
    }
    if (!list->pool.pool) {
        std::expected<vk::UniqueCommandPool, Error> pool =
            checkVulkan(device->createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eTransient, state.family}), "vkCreateCommandPool");
        if (!pool) {
            list->error = std::move(pool.error());
            return list;
        }
        list->pool.pool = std::move(*pool);
        std::expected<std::vector<vk::CommandBuffer>, Error> commands =
            checkVulkan(device->allocateCommandBuffers({*list->pool.pool, vk::CommandBufferLevel::ePrimary, 1}), "vkAllocateCommandBuffers");
        if (!commands) {
            list->error = std::move(commands.error());
            return list;
        }
        list->pool.commands = commands->front();
    }
    if (std::expected<void, Error> begun = checkVulkan(list->pool.commands.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit}), "vkBeginCommandBuffer");
        !begun) {
        list->error = std::move(begun.error());
        return list;
    }
    return list;
}

void VulkanDevice::recycle(QueueState& queue, InFlight&& done) {
    if (done.pool.pool && device->resetCommandPool(*done.pool.pool) == vk::Result::eSuccess)
        queue.freePools.push_back(std::move(done.pool));
    for (const ArgumentChunk& chunk : done.chunks)
        if (chunk.memory.size == argumentChunkBytes)
            freeChunks.push_back(chunk.memory);
        else
            release(chunk.memory);
}

void VulkanDevice::discard(VulkanCommandList& list) {
    if (list.closed)
        return;
    list.closed = true;
    std::lock_guard lock(mutex);
    openLists.erase(list.id);
    recycle(getQueue(list.queue), {0, std::move(list.pool), std::move(list.chunks)});
}

std::expected<Token, Error> VulkanDevice::submit(CommandList& list, const std::span<const Token> waits) {
    auto& vulkanList = static_cast<VulkanCommandList&>(list);
    if (vulkanList.device.get() != this)
        return fail(ErrorCode::InvalidArgument, "the command list belongs to another device");
    return submit(vulkanList, waits, {});
}

std::expected<Token, Error> VulkanDevice::submit(VulkanCommandList& list, const std::span<const Token> waits, const BinarySemaphores binary) {
    if (list.closed)
        return fail(ErrorCode::InvalidState, "the command list was already submitted");
    if (list.error) {
        Error error = *list.error;
        discard(list);
        return std::unexpected(std::move(error));
    }
    if (list.rendering) {
        discard(list);
        return fail(ErrorCode::InvalidState, "the command list ends inside beginRender");
    }
    if (list.concurrent) {
        discard(list);
        return fail(ErrorCode::InvalidState, "the command list ends inside beginConcurrent");
    }
    list.finish();
    if (std::expected<void, Error> ended = checkVulkan(list.pool.commands.end(), "vkEndCommandBuffer"); !ended) {
        discard(list);
        return std::unexpected(std::move(ended.error()));
    }

    QueueState& queue = getQueue(list.queue);
    std::vector<vk::SemaphoreSubmitInfo> waitInfos;
    waitInfos.reserve(waits.size() + 1);
    for (const Token token : waits)
        if (token && token.queue != list.queue)
            waitInfos.emplace_back(*getQueue(token.queue).timeline, token.value, vk::PipelineStageFlagBits2::eAllCommands);
    if (binary.wait)
        waitInfos.emplace_back(binary.wait, 0, vk::PipelineStageFlagBits2::eAllCommands);
    const vk::CommandBufferSubmitInfo commandInfo(list.pool.commands);

    std::lock_guard lock(mutex);
    const std::uint64_t value = queue.submitted + 1;
    std::array<vk::SemaphoreSubmitInfo, 2> signalInfos{vk::SemaphoreSubmitInfo(*queue.timeline, value, vk::PipelineStageFlagBits2::eAllCommands),
                                                       vk::SemaphoreSubmitInfo(binary.signal, 0, vk::PipelineStageFlagBits2::eAllCommands)};
    const vk::SubmitInfo2 submitInfo = vk::SubmitInfo2()
                                           .setWaitSemaphoreInfos(waitInfos)
                                           .setCommandBufferInfos(commandInfo)
                                           .setSignalSemaphoreInfoCount(binary.signal ? 2 : 1)
                                           .setPSignalSemaphoreInfos(signalInfos.data());
    const vk::Result submitted = queue.queue.submit2(submitInfo);
    list.closed = true;
    openLists.erase(list.id);
    if (submitted != vk::Result::eSuccess) {
        recycle(queue, {0, std::move(list.pool), std::move(list.chunks)});
        return failVulkan(submitted, "vkQueueSubmit2");
    }
    queue.submitted = value;
    queue.inFlight.push_back({value, std::move(list.pool), std::move(list.chunks)});
    for (const std::shared_ptr<VulkanReadback>& readback : list.readbacks)
        readback->token = {value, list.queue};
    list.readbacks.clear();
    reap();
    return Token{value, list.queue};
}

std::expected<Token, Error> VulkanDevice::execute(const std::function<void(VulkanCommandList&)>& record) {
    std::unique_ptr<VulkanCommandList> list = openList(Queue::Graphics);
    if (!list->error)
        record(*list);
    return submit(*list, {}, {});
}

std::expected<void, Error> VulkanDevice::download(const std::span<std::byte> destination,
                                                  const std::function<void(vk::CommandBuffer, const Allocation&)>& record) {
    std::expected<Allocation, Error> staging = allocate(destination.size(), MemoryKind::Readback);
    if (!staging)
        return std::unexpected(std::move(staging.error()));
    std::expected<Token, Error> token = execute([&](VulkanCommandList& list) {
        list.synchronize(vk::PipelineStageFlagBits2::eAllTransfer);
        record(list.pool.commands, *staging);
    });
    if (!token) {
        release(*staging);
        return std::unexpected(std::move(token.error()));
    }
    if (std::expected<void, Error> waited = wait(*token); !waited) {
        retire([this, allocation = *staging] { release(allocation); });
        return waited;
    }
    std::memcpy(destination.data(), staging->mapped, destination.size());
    release(*staging);
    return {};
}

void VulkanDevice::reap() {
    std::array<std::uint64_t, queueCount> completed{};
    for (std::size_t index = 0; index < queueCount; ++index) {
        QueueState& queue = queues[index];
        completed[index] = device->getSemaphoreCounterValue(*queue.timeline).value;
        while (!queue.inFlight.empty() && queue.inFlight.front().value <= completed[index]) {
            recycle(queue, std::move(queue.inFlight.front()));
            queue.inFlight.pop_front();
        }
    }
    const std::uint64_t lowestOpen = openLists.empty() ? std::numeric_limits<std::uint64_t>::max() : *openLists.begin();
    for (Retired& entry : retired) {
        if (entry.snapshotted)
            continue;
        if (entry.listWatermark >= lowestOpen)
            break;
        for (std::size_t index = 0; index < queueCount; ++index)
            entry.timelines[index] = queues[index].submitted;
        entry.snapshotted = true;
    }
    std::vector<std::function<void()>> releases;
    while (!retired.empty() && retired.front().snapshotted) {
        const Retired& front = retired.front();
        bool done = true;
        for (std::size_t index = 0; index < queueCount; ++index)
            done = done && front.timelines[index] <= completed[index];
        if (!done)
            break;
        releases.push_back(std::move(retired.front().release));
        retired.pop_front();
    }
    if (releases.empty())
        return;
    mutex.unlock();
    for (std::function<void()>& release : releases)
        release();
    mutex.lock();
}

std::expected<void, Error> VulkanDevice::wait(const Token token) {
    if (!token)
        return {};
    const vk::Semaphore timeline = *getQueue(token.queue).timeline;
    const vk::Result waited = device->waitSemaphores(vk::SemaphoreWaitInfo({}, timeline, token.value), std::numeric_limits<std::uint64_t>::max());
    if (waited != vk::Result::eSuccess)
        return failVulkan(waited, "vkWaitSemaphores");
    std::lock_guard lock(mutex);
    reap();
    return {};
}

std::expected<void, Error> VulkanDevice::waitIdle() {
    std::array<Token, queueCount> last;
    {
        std::lock_guard lock(mutex);
        for (std::size_t index = 0; index < queueCount; ++index)
            last[index] = {queues[index].submitted, static_cast<Queue>(index)};
    }
    for (const Token token : last)
        if (std::expected<void, Error> waited = wait(token); !waited)
            return waited;
    std::lock_guard lock(mutex);
    reap();
    return {};
}

bool VulkanDevice::isFinished(const Token token) {
    return device->getSemaphoreCounterValue(*getQueue(token.queue).timeline).value >= token.value;
}

std::expected<std::uint32_t, Error> VulkanDevice::writeDescriptor(SlotAllocator& slots, const std::uint32_t binding, const vk::DescriptorType type,
                                                                  const vk::DescriptorImageInfo& image) {
    std::lock_guard lock(descriptorMutex);
    std::uint32_t slot = 0;
    if (!slots.free.empty()) {
        slot = slots.free.back();
        slots.free.pop_back();
    } else if (slots.next < slots.capacity) {
        slot = slots.next++;
    } else {
        return fail(ErrorCode::OutOfMemory, "every bindless descriptor slot of the device is in use");
    }
    device->updateDescriptorSets(vk::WriteDescriptorSet(descriptorSet, binding, slot, type, image), {});
    return slot;
}

void VulkanDevice::releaseSlot(SlotAllocator& slots, const std::uint32_t slot) {
    if (slot == 0)
        return;
    std::lock_guard lock(descriptorMutex);
    slots.free.push_back(slot);
}

}

namespace NoorRHI {

std::expected<void, Error> Device::open(const DeviceConfig& config) {
    auto opened = std::make_shared<Backend::VulkanDevice>();
    if (std::expected<void, Error> result = opened->open(config); !result)
        return result;
    backend = std::move(opened);
    return {};
}

}
