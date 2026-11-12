#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>

#if !defined(__EMSCRIPTEN__)
#include <dawn/native/DawnNative.h>
#include <dawn/platform/DawnPlatform.h>
#endif

#include "NoorRHI/WebGpu/WebGpu.h"

namespace NoorRHI::Backend {

#if defined(__EMSCRIPTEN__)
//! The browser keeps its own shader cache.
struct WebGpuPlatform {};
#else
//! Each entry is a file named by the key hash that holds the key size, the key and the data, so a hash collision reads as a miss.
class WebGpuBlobCache final : public dawn::platform::CachingInterface {
public:
    explicit WebGpuBlobCache(std::filesystem::path folder) : directory(std::move(folder)), writerId(std::random_device{}()) {}

    std::size_t FindKey(const std::span<const std::byte> key) override { return read(key, {}); }
    std::size_t LoadData(const std::span<const std::byte> key, const std::span<std::byte> destination) override { return read(key, destination); }

    void StoreData(const std::span<const std::byte> key, const std::span<const std::byte> data) override {
        const std::filesystem::path path = getPath(key);
        const std::filesystem::path partial = path.string() + "." + std::to_string(writerId) + "." + std::to_string(nextWrite++) + ".partial";
        {
            std::ofstream file(partial, std::ios::binary | std::ios::trunc);
            const std::uint64_t keySize = key.size();
            file.write(reinterpret_cast<const char*>(&keySize), sizeof(keySize));
            file.write(reinterpret_cast<const char*>(key.data()), static_cast<std::streamsize>(key.size()));
            file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
            if (!file)
                return;
        }
        std::error_code error;
        std::filesystem::rename(partial, path, error);
    }

private:
    std::filesystem::path getPath(const std::span<const std::byte> key) const {
        std::uint64_t hash = 14695981039346656037ull;
        for (const std::byte value : key)
            hash = (hash ^ static_cast<std::uint64_t>(value)) * 1099511628211ull;
        char name[17];
        std::snprintf(name, sizeof(name), "%016llx", static_cast<unsigned long long>(hash));
        return directory / name;
    }

    std::size_t read(const std::span<const std::byte> key, const std::span<std::byte> destination) const {
        std::ifstream file(getPath(key), std::ios::binary | std::ios::ate);
        if (!file)
            return 0;
        const std::uint64_t fileSize = static_cast<std::uint64_t>(file.tellg());
        std::uint64_t keySize = 0;
        file.seekg(0);
        file.read(reinterpret_cast<char*>(&keySize), sizeof(keySize));
        if (!file || keySize != key.size() || fileSize < sizeof(keySize) + keySize)
            return 0;
        std::vector<std::byte> stored(keySize);
        file.read(reinterpret_cast<char*>(stored.data()), static_cast<std::streamsize>(keySize));
        if (!file || !std::ranges::equal(stored, key))
            return 0;
        const std::size_t dataSize = fileSize - sizeof(keySize) - keySize;
        if (destination.empty())
            return dataSize;
        if (destination.size() < dataSize)
            return 0;
        file.read(reinterpret_cast<char*>(destination.data()), static_cast<std::streamsize>(dataSize));
        return file ? dataSize : 0;
    }

    std::filesystem::path directory;
    std::uint64_t writerId;
    std::atomic<std::uint64_t> nextWrite = 0;
};

struct WebGpuPlatform final : dawn::platform::Platform {
    explicit WebGpuPlatform(std::filesystem::path directory) : cache(std::move(directory)) {}
    dawn::platform::CachingInterface* GetCachingInterface() override { return &cache; }

    WebGpuBlobCache cache;
};
#endif

namespace {

constexpr wgpu::ShaderStage renderStages = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
constexpr wgpu::BufferUsage heapUsage =
    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Indirect | wgpu::BufferUsage::Index;
constexpr std::array<const char*, 2> heapLabels{"NoorRHI heap", "NoorRHI atomics"};
constexpr wgpu::BufferUsage argumentUsage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
constexpr std::uint64_t argumentBufferBytes = 64 * 1024;
constexpr std::array optionalFeatures{wgpu::FeatureName::TimestampQuery,
                                      wgpu::FeatureName::Float32Filterable,
                                      wgpu::FeatureName::TextureCompressionBC,
                                      wgpu::FeatureName::TextureComponentSwizzle,
                                      wgpu::FeatureName::TextureFormatsTier1,
                                      wgpu::FeatureName::TextureFormatsTier2,
                                      wgpu::FeatureName::IndirectFirstInstance,
                                      wgpu::FeatureName::Depth32FloatStencil8,
                                      wgpu::FeatureName::Subgroups,
                                      wgpu::FeatureName::ShaderF16,
                                      wgpu::FeatureName::MultiDrawIndirect};
constexpr std::array poolSampleTypes{wgpu::TextureSampleType::Float, wgpu::TextureSampleType::UnfilterableFloat, wgpu::TextureSampleType::Uint};
constexpr std::array emptySampledFormats{wgpu::TextureFormat::RGBA8Unorm, wgpu::TextureFormat::R32Float, wgpu::TextureFormat::R32Uint};
constexpr std::array storageFormats{wgpu::TextureFormat::RGBA8Unorm, wgpu::TextureFormat::RGBA16Float, wgpu::TextureFormat::R32Float,
                                    wgpu::TextureFormat::R32Uint,    wgpu::TextureFormat::RGBA32Float, wgpu::TextureFormat::R16Float,
                                    wgpu::TextureFormat::R8Unorm};
static_assert(storageFormats.size() == storagePoolBindings.count);

constexpr std::string_view blitSource = R"(
@group(0) @binding(0) var source : texture_2d<f32>;
@group(0) @binding(1) var linearSampler : sampler;
@group(0) @binding(2) var<uniform> transform : vec4f;

struct Varyings {
    @builtin(position) position : vec4f,
    @location(0) uv : vec2f,
};

@vertex fn vertexMain(@builtin(vertex_index) index : u32) -> Varyings {
    let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
    return Varyings(vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0), uv);
}

@fragment fn fragmentMain(input : Varyings) -> @location(0) vec4f {
    let half = 0.5 / vec2f(textureDimensions(source));
    let uv = clamp(transform.xy + input.uv * transform.zw, transform.xy + half, transform.xy + transform.zw - half);
    return textureSampleLevel(source, linearSampler, uv, 0.0);
}
)";

constexpr std::string_view patchSource = R"(
@group(0) @binding(0) var<storage, read_write> heap : array<u32>;
@group(1) @binding(0) var<storage, read> bytePatch : vec4u;

@compute @workgroup_size(1) fn main() {
    heap[bytePatch.x] = (heap[bytePatch.x] & ~bytePatch.y) | (bytePatch.z & bytePatch.y);
}
)";

constexpr std::string_view countSource = R"(
@group(0) @binding(0) var<storage, read> heap : array<u32>;
@group(0) @binding(1) var<storage, read> atomics : array<u32>;
@group(0) @binding(2) var<storage, read_write> scratch : array<u32>;
@group(1) @binding(0) var<storage, read> job : vec4u;

fn load(word : u32, inAtomics : bool) -> u32 {
    if (inAtomics) {
        return atomics[word];
    }
    return heap[word];
}

@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id : vec3u) {
    if (id.x >= (job.w & 0x1fffffffu)) {
        return;
    }
    let stride = select(4u, 5u, (job.w >> 31u) != 0u);
    let input = job.x + id.x * stride;
    let output = job.z + id.x * stride;
    for (var word = 0u; word < stride; word++) {
        scratch[output + word] = load(input + word, ((job.w >> 30u) & 1u) != 0u);
    }
    if (id.x >= load(job.y, ((job.w >> 29u) & 1u) != 0u)) {
        scratch[output + 1u] = 0u;
    }
}
)";

wgpu::ShaderModule createWgslModule(const wgpu::Device& device, const std::string_view source) {
    wgpu::ShaderSourceWGSL wgsl;
    wgsl.code = source;
    const wgpu::ShaderModuleDescriptor descriptor{.nextInChain = &wgsl};
    return device.CreateShaderModule(&descriptor);
}

wgpu::BindGroupLayoutEntry makeBufferEntry(const std::uint32_t binding, const wgpu::ShaderStage visibility, const wgpu::BufferBindingType type) {
    wgpu::BindGroupLayoutEntry entry{.binding = binding, .visibility = visibility};
    entry.buffer.type = type;
    return entry;
}

wgpu::BindGroupLayoutEntry makeTextureEntry(const std::uint32_t binding, const wgpu::ShaderStage visibility, const wgpu::TextureSampleType type) {
    wgpu::BindGroupLayoutEntry entry{.binding = binding, .visibility = visibility};
    entry.texture.sampleType = type;
    entry.texture.viewDimension = wgpu::TextureViewDimension::e2DArray;
    return entry;
}

wgpu::BindGroupLayoutEntry makeSamplerEntry(const std::uint32_t binding, const wgpu::ShaderStage visibility) {
    wgpu::BindGroupLayoutEntry entry{.binding = binding, .visibility = visibility};
    entry.sampler.type = wgpu::SamplerBindingType::Filtering;
    return entry;
}

wgpu::BindGroupLayoutEntry makeStorageEntry(const std::uint32_t binding, const wgpu::TextureFormat format) {
    wgpu::BindGroupLayoutEntry entry{.binding = binding, .visibility = wgpu::ShaderStage::Compute};
    entry.storageTexture.access = wgpu::StorageTextureAccess::ReadWrite;
    entry.storageTexture.format = format;
    entry.storageTexture.viewDimension = wgpu::TextureViewDimension::e2DArray;
    return entry;
}

void appendSharedEntries(std::vector<wgpu::BindGroupLayoutEntry>& entries, const std::span<const PoolBindings> ranges, const wgpu::ShaderStage visibility) {
    for (std::size_t kind = 0; kind < ranges.size(); ++kind)
        for (std::uint32_t index = 0; index < ranges[kind].count; ++index)
            entries.push_back(makeTextureEntry(ranges[kind].first + index, visibility, poolSampleTypes[kind]));
    entries.push_back(makeSamplerEntry(samplerBinding, visibility));
}

std::size_t getPoolKind(const std::uint32_t binding) {
    for (std::size_t kind = 0; kind < sampledPoolBindings.size(); ++kind)
        if (binding >= sampledPoolBindings[kind].first && binding < sampledPoolBindings[kind].first + sampledPoolBindings[kind].count)
            return kind;
    return 0;
}

}

wgpu::TextureFormat getWebGpuFormat(const ImageFormat format) {
    switch (format) {
    case ImageFormat::Undefined:
        return wgpu::TextureFormat::Undefined;
    case ImageFormat::Rgba8Unorm:
        return wgpu::TextureFormat::RGBA8Unorm;
    case ImageFormat::Bgra8Unorm:
        return wgpu::TextureFormat::BGRA8Unorm;
    case ImageFormat::Rgba32Float:
        return wgpu::TextureFormat::RGBA32Float;
    case ImageFormat::Rgba16Float:
        return wgpu::TextureFormat::RGBA16Float;
    case ImageFormat::R32Uint:
        return wgpu::TextureFormat::R32Uint;
    case ImageFormat::R32Float:
        return wgpu::TextureFormat::R32Float;
    case ImageFormat::D32Float:
        return wgpu::TextureFormat::Depth32Float;
    case ImageFormat::D32FloatS8Uint:
        return wgpu::TextureFormat::Depth32FloatStencil8;
    case ImageFormat::R8Unorm:
        return wgpu::TextureFormat::R8Unorm;
    case ImageFormat::L8Unorm:
        return wgpu::TextureFormat::R8Unorm;
    case ImageFormat::L8Srgb:
        return wgpu::TextureFormat::Undefined;
    case ImageFormat::Rg8Unorm:
        return wgpu::TextureFormat::RG8Unorm;
    case ImageFormat::Rgba8Srgb:
        return wgpu::TextureFormat::RGBA8UnormSrgb;
    case ImageFormat::Bgra8Srgb:
        return wgpu::TextureFormat::BGRA8UnormSrgb;
    case ImageFormat::R16Unorm:
        return wgpu::TextureFormat::R16Unorm;
    case ImageFormat::Rg16Unorm:
        return wgpu::TextureFormat::RG16Unorm;
    case ImageFormat::Rgba16Unorm:
        return wgpu::TextureFormat::RGBA16Unorm;
    case ImageFormat::R16Float:
        return wgpu::TextureFormat::R16Float;
    case ImageFormat::Rg16Float:
        return wgpu::TextureFormat::RG16Float;
    case ImageFormat::Rg32Float:
        return wgpu::TextureFormat::RG32Float;
    case ImageFormat::A2b10g10r10Unorm:
        return wgpu::TextureFormat::RGB10A2Unorm;
    case ImageFormat::B10g11r11Float:
        return wgpu::TextureFormat::RG11B10Ufloat;
    case ImageFormat::Bc1Unorm:
        return wgpu::TextureFormat::BC1RGBAUnorm;
    case ImageFormat::Bc1Srgb:
        return wgpu::TextureFormat::BC1RGBAUnormSrgb;
    case ImageFormat::Bc2Unorm:
        return wgpu::TextureFormat::BC2RGBAUnorm;
    case ImageFormat::Bc2Srgb:
        return wgpu::TextureFormat::BC2RGBAUnormSrgb;
    case ImageFormat::Bc3Unorm:
        return wgpu::TextureFormat::BC3RGBAUnorm;
    case ImageFormat::Bc3Srgb:
        return wgpu::TextureFormat::BC3RGBAUnormSrgb;
    case ImageFormat::Bc4Unorm:
        return wgpu::TextureFormat::BC4RUnorm;
    case ImageFormat::Bc5Unorm:
        return wgpu::TextureFormat::BC5RGUnorm;
    case ImageFormat::Bc6hUfloat:
        return wgpu::TextureFormat::BC6HRGBUfloat;
    case ImageFormat::Bc6hSfloat:
        return wgpu::TextureFormat::BC6HRGBFloat;
    case ImageFormat::Bc7Unorm:
        return wgpu::TextureFormat::BC7RGBAUnorm;
    case ImageFormat::Bc7Srgb:
        return wgpu::TextureFormat::BC7RGBAUnormSrgb;
    }
    return wgpu::TextureFormat::Undefined;
}

bool WebGpuDevice::isRenderable(const ImageFormat format) const {
    switch (format) {
    case ImageFormat::Rgba8Unorm:
    case ImageFormat::Bgra8Unorm:
    case ImageFormat::Rgba8Srgb:
    case ImageFormat::Bgra8Srgb:
    case ImageFormat::Rgba16Float:
    case ImageFormat::Rgba32Float:
    case ImageFormat::R32Uint:
    case ImageFormat::R32Float:
    case ImageFormat::R8Unorm:
    case ImageFormat::Rg8Unorm:
    case ImageFormat::R16Float:
    case ImageFormat::Rg16Float:
    case ImageFormat::Rg32Float:
    case ImageFormat::A2b10g10r10Unorm:
        return true;
    case ImageFormat::R16Unorm:
    case ImageFormat::Rg16Unorm:
    case ImageFormat::Rgba16Unorm:
    case ImageFormat::B10g11r11Float:
        return formatsTier1;
    default:
        return false;
    }
}

std::string toString(const wgpu::StringView text) {
    if (!text.data)
        return {};
    return text.length == wgpu::kStrlen ? std::string(text.data) : std::string(text.data, text.length);
}

std::uint64_t alignUp(const std::uint64_t value, const std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

WebGpuDevice::~WebGpuDevice() {
    if (!device)
        return;
    static_cast<void>(waitIdle());
}

std::expected<void, Error> WebGpuDevice::open(const DeviceConfig& config) {
    presentation = config.presentation != nullptr;
    if (std::expected<void, Error> created = createAdapter(config.pipelineDirectory); !created)
        return created;
    if (std::expected<void, Error> created = createDevice(config.enableValidation); !created)
        return created;
    if (std::expected<void, Error> created = createLayouts(); !created)
        return created;
    return createEmptyResources();
}

std::expected<void, Error> WebGpuDevice::createAdapter([[maybe_unused]] const std::filesystem::path& pipelineDirectory) {
    const wgpu::InstanceFeatureName timedWait = wgpu::InstanceFeatureName::TimedWaitAny;
    wgpu::InstanceDescriptor instanceDescriptor{.requiredFeatureCount = 1, .requiredFeatures = &timedWait};
#if !defined(__EMSCRIPTEN__)
    dawn::native::DawnInstanceDescriptor dawnDescriptor;
    if (!pipelineDirectory.empty()) {
        const std::filesystem::path directory = pipelineDirectory / "webgpu";
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        if (error)
            return fail(ErrorCode::InvalidArgument, "cannot create the pipeline cache folder " + directory.string() + ": " + error.message());
        platform = std::make_unique<WebGpuPlatform>(directory);
        dawnDescriptor.platform = platform.get();
        instanceDescriptor.nextInChain = &dawnDescriptor;
    }
#endif
#if !defined(__EMSCRIPTEN__)
    //! Dawn shows experimental features such as multi-draw-indirect only with this toggle; Chrome shows them only with --enable-unsafe-webgpu.
    const char* const unsafeApis = "allow_unsafe_apis";
    wgpu::DawnTogglesDescriptor instanceToggles;
    instanceToggles.enabledToggleCount = 1;
    instanceToggles.enabledToggles = &unsafeApis;
    instanceToggles.nextInChain = instanceDescriptor.nextInChain;
    instanceDescriptor.nextInChain = &instanceToggles;
#endif
    instance = wgpu::CreateInstance(&instanceDescriptor);
    if (!instance)
        return fail(ErrorCode::Unsupported, "WebGPU is not available");
    const wgpu::RequestAdapterOptions options{.powerPreference = wgpu::PowerPreference::HighPerformance};
    std::string message;
    const wgpu::Future future =
        instance.RequestAdapter(&options, wgpu::CallbackMode::WaitAnyOnly,
                                [this, &message](const wgpu::RequestAdapterStatus status, wgpu::Adapter result, const wgpu::StringView text) {
                                    if (status == wgpu::RequestAdapterStatus::Success)
                                        adapter = std::move(result);
                                    message = toString(text);
                                });
    if (std::expected<void, Error> waited = waitFuture(future); !waited)
        return waited;
    if (!adapter)
        return fail(ErrorCode::Unsupported, "no WebGPU adapter is available: " + message);
    wgpu::AdapterInfo adapterInfo;
    adapter.GetInfo(&adapterInfo);
    info.api = GraphicsApi::WebGpu;
    info.name = toString(adapterInfo.description).empty() ? toString(adapterInfo.device) : toString(adapterInfo.description);
    info.vendorId = adapterInfo.vendorID;
    info.deviceId = adapterInfo.deviceID;
    features.minSubgroupSize = adapterInfo.subgroupMinSize;
    return {};
}

std::expected<void, Error> WebGpuDevice::createDevice([[maybe_unused]] const bool validation) {
    std::vector<wgpu::FeatureName> requested;
    for (const wgpu::FeatureName feature : optionalFeatures)
        if (adapter.HasFeature(feature))
            requested.push_back(feature);
    wgpu::Limits required;
    adapter.GetLimits(&required);

    wgpu::DeviceDescriptor descriptor;
    descriptor.requiredFeatureCount = requested.size();
    descriptor.requiredFeatures = requested.data();
    descriptor.requiredLimits = &required;
#if !defined(__EMSCRIPTEN__)
    //! Like Vulkan without layers, native Dawn without validation trusts the commands, and skips the GPU checks of indirect draws and spirv-val of each shader.
    const char* const skipValidation = "skip_validation";
    const char* const spirvValidation = "enable_spirv_validation";
    wgpu::DawnTogglesDescriptor toggles;
    toggles.enabledToggleCount = 1;
    toggles.enabledToggles = &skipValidation;
    toggles.disabledToggleCount = 1;
    toggles.disabledToggles = &spirvValidation;
    if (!validation)
        descriptor.nextInChain = &toggles;
#endif
    descriptor.SetUncapturedErrorCallback(
        [](const wgpu::Device&, const wgpu::ErrorType type, const wgpu::StringView text, WebGpuDevice* owner) {
            owner->recordError(type == wgpu::ErrorType::OutOfMemory ? ErrorCode::OutOfMemory : ErrorCode::InvalidState, "WebGPU: " + toString(text));
        },
        this);
    descriptor.SetDeviceLostCallback(wgpu::CallbackMode::AllowSpontaneous,
                                     [this](const wgpu::Device&, const wgpu::DeviceLostReason reason, const wgpu::StringView text) {
                                         if (reason == wgpu::DeviceLostReason::Destroyed || reason == wgpu::DeviceLostReason::CallbackCancelled)
                                             return;
                                         recordError(ErrorCode::DeviceLost, "the WebGPU device was lost: " + toString(text));
                                     });
    std::string message;
    const wgpu::Future future =
        adapter.RequestDevice(&descriptor, wgpu::CallbackMode::WaitAnyOnly,
                              [this, &message](const wgpu::RequestDeviceStatus status, wgpu::Device result, const wgpu::StringView text) {
                                  if (status == wgpu::RequestDeviceStatus::Success)
                                      device = std::move(result);
                                  message = toString(text);
                              });
    if (std::expected<void, Error> waited = waitFuture(future); !waited)
        return waited;
    if (!device)
        return fail(ErrorCode::Unsupported, "the WebGPU device could not be created: " + message);
    queue = device.GetQueue();
    device.GetLimits(&limits);
    timestamps = device.HasFeature(wgpu::FeatureName::TimestampQuery);
    float32Filterable = device.HasFeature(wgpu::FeatureName::Float32Filterable);
    compressionBc = device.HasFeature(wgpu::FeatureName::TextureCompressionBC);
    componentSwizzle = device.HasFeature(wgpu::FeatureName::TextureComponentSwizzle);
    formatsTier1 = device.HasFeature(wgpu::FeatureName::TextureFormatsTier1);
    depthStencil = device.HasFeature(wgpu::FeatureName::Depth32FloatStencil8);
    multiDrawIndirect = device.HasFeature(wgpu::FeatureName::MultiDrawIndirect);
    features.subgroups = device.HasFeature(wgpu::FeatureName::Subgroups);
    features.shaderFloat16 = device.HasFeature(wgpu::FeatureName::ShaderF16);
    storageImages = device.HasFeature(wgpu::FeatureName::TextureFormatsTier2) && limits.maxStorageTexturesPerShaderStage >= storagePoolBindings.count;
    PoolBindings& floatPools = sampledRanges[static_cast<std::size_t>(PoolKind::Float)];
    const std::uint32_t otherPools = sampledTextureCount - floatPools.count;
    if (limits.maxSampledTexturesPerShaderStage <= otherPools)
        return fail(ErrorCode::Unsupported, "the WebGPU device needs more than " + std::to_string(otherPools) + " sampled textures per shader stage");
    floatPools.count = std::min(floatPools.count, limits.maxSampledTexturesPerShaderStage - otherPools);
    if (!device.HasFeature(wgpu::FeatureName::IndirectFirstInstance))
        return fail(ErrorCode::Unsupported, "the WebGPU device has no indirect-first-instance");
    heapLimit =
        std::min({limits.maxStorageBufferBindingSize, limits.maxBufferSize, std::uint64_t{std::numeric_limits<std::uint32_t>::max()}}) & ~std::uint64_t{3};
    return {};
}

std::expected<void, Error> WebGpuDevice::createLayouts() {
    std::vector<wgpu::BindGroupLayoutEntry> computeEntries{makeBufferEntry(heapBinding, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage),
                                                           makeBufferEntry(atomicBinding, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage)};
    appendSharedEntries(computeEntries, sampledRanges, wgpu::ShaderStage::Compute);
    for (std::uint32_t index = 0; storageImages && index < storagePoolBindings.count; ++index)
        computeEntries.push_back(makeStorageEntry(storagePoolBindings.first + index, storageFormats[index]));
    std::vector<wgpu::BindGroupLayoutEntry> renderEntries{makeBufferEntry(heapReadBinding, renderStages, wgpu::BufferBindingType::ReadOnlyStorage),
                                                          makeBufferEntry(atomicBinding, wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::Storage)};
    appendSharedEntries(renderEntries, sampledRanges, renderStages);
    wgpu::BindGroupLayoutEntry argumentEntry =
        makeBufferEntry(argumentBinding, wgpu::ShaderStage::Compute | renderStages, wgpu::BufferBindingType::ReadOnlyStorage);
    argumentEntry.buffer.hasDynamicOffset = true;

    pushErrorScope();
    const wgpu::BindGroupLayoutDescriptor argumentDescriptor{.entryCount = 1, .entries = &argumentEntry};
    argumentGroupLayout = device.CreateBindGroupLayout(&argumentDescriptor);
    const wgpu::BindGroupLayoutDescriptor computeDescriptor{.entryCount = computeEntries.size(), .entries = computeEntries.data()};
    computeGroupLayout = device.CreateBindGroupLayout(&computeDescriptor);
    const std::array computeGroups{computeGroupLayout, argumentGroupLayout};
    const wgpu::PipelineLayoutDescriptor computeLayoutDescriptor{.bindGroupLayoutCount = computeGroups.size(), .bindGroupLayouts = computeGroups.data()};
    computeLayout = device.CreatePipelineLayout(&computeLayoutDescriptor);
    const wgpu::BindGroupLayoutDescriptor renderDescriptor{.entryCount = renderEntries.size(), .entries = renderEntries.data()};
    renderGroupLayout = device.CreateBindGroupLayout(&renderDescriptor);
    const std::array renderGroups{renderGroupLayout, argumentGroupLayout};
    const wgpu::PipelineLayoutDescriptor renderLayoutDescriptor{.bindGroupLayoutCount = renderGroups.size(), .bindGroupLayouts = renderGroups.data()};
    renderLayout = device.CreatePipelineLayout(&renderLayoutDescriptor);

    std::array blitEntries{makeTextureEntry(0, wgpu::ShaderStage::Fragment, wgpu::TextureSampleType::Float), makeSamplerEntry(1, wgpu::ShaderStage::Fragment),
                           makeBufferEntry(2, wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::Uniform)};
    blitEntries[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;
    const wgpu::BindGroupLayoutDescriptor blitDescriptor{.entryCount = blitEntries.size(), .entries = blitEntries.data()};
    blitGroupLayout = device.CreateBindGroupLayout(&blitDescriptor);
    const wgpu::PipelineLayoutDescriptor blitLayoutDescriptor{.bindGroupLayoutCount = 1, .bindGroupLayouts = &blitGroupLayout};
    blitLayout = device.CreatePipelineLayout(&blitLayoutDescriptor);
    blitModule = createWgslModule(device, blitSource);
    wgpu::ComputePipelineDescriptor patchDescriptor{.layout = computeLayout};
    patchDescriptor.compute.module = createWgslModule(device, patchSource);
    patchDescriptor.compute.entryPoint = "main";
    patchPipeline = device.CreateComputePipeline(&patchDescriptor);
    const std::array countEntries{makeBufferEntry(0, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage),
                                  makeBufferEntry(1, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage),
                                  makeBufferEntry(2, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage)};
    const wgpu::BindGroupLayoutDescriptor countDescriptor{.entryCount = countEntries.size(), .entries = countEntries.data()};
    countGroupLayout = device.CreateBindGroupLayout(&countDescriptor);
    const std::array countGroups{countGroupLayout, argumentGroupLayout};
    const wgpu::PipelineLayoutDescriptor countLayoutDescriptor{.bindGroupLayoutCount = countGroups.size(), .bindGroupLayouts = countGroups.data()};
    wgpu::ComputePipelineDescriptor countPipelineDescriptor{.layout = device.CreatePipelineLayout(&countLayoutDescriptor)};
    countPipelineDescriptor.compute.module = createWgslModule(device, countSource);
    countPipelineDescriptor.compute.entryPoint = "main";
    countPipeline = device.CreateComputePipeline(&countPipelineDescriptor);
    const wgpu::SamplerDescriptor samplerDescriptor{.magFilter = wgpu::FilterMode::Linear, .minFilter = wgpu::FilterMode::Linear};
    blitSampler = device.CreateSampler(&samplerDescriptor);
    return popErrorScope("the WebGPU bind group layouts");
}

std::expected<void, Error> WebGpuDevice::createEmptyResources() {
    pushErrorScope();
    const wgpu::TextureViewDescriptor arrayView{.dimension = wgpu::TextureViewDimension::e2DArray};
    for (std::size_t kind = 0; kind < emptySampled.size(); ++kind) {
        const wgpu::TextureDescriptor descriptor{.usage = wgpu::TextureUsage::TextureBinding, .size = {1, 1, 1}, .format = emptySampledFormats[kind]};
        emptySampled[kind] = device.CreateTexture(&descriptor).CreateView(&arrayView);
    }
    for (std::size_t index = 0; storageImages && index < emptyStorage.size(); ++index) {
        const wgpu::TextureDescriptor descriptor{.usage = wgpu::TextureUsage::StorageBinding, .size = {1, 1, 1}, .format = storageFormats[index]};
        emptyStorage[index] = device.CreateTexture(&descriptor).CreateView(&arrayView);
    }
    const wgpu::SamplerDescriptor samplerDescriptor{.magFilter = wgpu::FilterMode::Linear,
                                                    .minFilter = wgpu::FilterMode::Linear,
                                                    .mipmapFilter = wgpu::MipmapFilterMode::Linear,
                                                    .maxAnisotropy = 16};
    sampler = device.CreateSampler(&samplerDescriptor);
    reserve(argumentBuffer, argumentBufferBytes, argumentUsage);
    for (std::size_t index = 0; index < storagePools.size(); ++index)
        storagePools[index].format = storageFormats[index];

    for (std::size_t memory = 0; memory < heaps.size(); ++memory) {
        const std::uint64_t heapBytes = std::min(initialHeapBytes[memory], heapLimit);
        const wgpu::BufferDescriptor heapDescriptor{.label = heapLabels[memory], .usage = heapUsage, .size = heapBytes};
        heaps[memory].buffer = device.CreateBuffer(&heapDescriptor);
        heaps[memory].ranges = RangeAllocator(heapBytes);
    }
    if (std::expected<void, Error> created = popErrorScope("the WebGPU heap"); !created)
        return created;
    if (!heaps[0].ranges.allocate(heapReservedBytes, 1))
        return fail(ErrorCode::OutOfMemory, "the WebGPU heap has no room for its reserved range");
    return {};
}

void WebGpuDevice::reserve(wgpu::Buffer& buffer, const std::uint64_t size, const wgpu::BufferUsage usage) {
    if (size == 0 || (buffer && buffer.GetSize() >= size))
        return;
    const wgpu::BufferDescriptor descriptor{.usage = usage, .size = std::max<std::uint64_t>(alignUp(size, 256), buffer ? buffer.GetSize() * 2 : 0)};
    buffer = device.CreateBuffer(&descriptor);
}

std::expected<void, Error> WebGpuDevice::growHeap(const BufferMemory memory, const std::uint64_t required) {
    GpuHeap& heap = heaps[static_cast<std::size_t>(memory)];
    std::uint64_t target = heap.ranges.getSize();
    while (target < required && target < heapLimit)
        target = std::min(target * 2, heapLimit);
    if (target < required)
        return fail(ErrorCode::OutOfMemory, "the WebGPU heap is full; one storage binding of this device holds " + std::to_string(heapLimit) + " bytes");
    pushErrorScope();
    const wgpu::BufferDescriptor descriptor{.label = heapLabels[static_cast<std::size_t>(memory)], .usage = heapUsage, .size = target};
    wgpu::Buffer grown = device.CreateBuffer(&descriptor);
    const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(heap.buffer, 0, grown, 0, heap.ranges.getSize());
    const wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);
    if (std::expected<void, Error> grew = popErrorScope("the WebGPU heap"); !grew)
        return grew;
    heap.buffer = std::move(grown);
    heap.ranges.grow(target);
    ++version;
    return {};
}

std::expected<std::uint64_t, Error> WebGpuDevice::allocate(const BufferMemory memory, const std::uint64_t size, const std::uint64_t alignment) {
    std::lock_guard lock(mutex);
    RangeAllocator& ranges = heaps[static_cast<std::size_t>(memory)].ranges;
    const std::uint64_t base = std::uint64_t{static_cast<std::uint8_t>(memory)} << 32;
    if (std::optional<std::uint64_t> offset = ranges.allocate(size, alignment))
        return base | *offset;
    if (std::expected<void, Error> grown = growHeap(memory, ranges.getSize() + size + alignment); !grown)
        return std::unexpected(std::move(grown.error()));
    if (std::optional<std::uint64_t> offset = ranges.allocate(size, alignment))
        return base | *offset;
    return fail(ErrorCode::OutOfMemory, "the WebGPU heap has no free range for the allocation");
}

void WebGpuDevice::release(const std::uint64_t address, const std::uint64_t size) {
    retire([this, address, size] { heaps[address >> 32].ranges.release(address & addressOffsetMask, size); });
}

void WebGpuDevice::retire(std::function<void()> release) {
    std::lock_guard lock(mutex);
    retired.emplace_back(nextListId - 1, std::move(release));
    releaseRetired();
}

std::uint64_t WebGpuDevice::openList() {
    std::lock_guard lock(mutex);
    openLists.insert(nextListId);
    return nextListId++;
}

void WebGpuDevice::closeList(const std::uint64_t id) {
    std::lock_guard lock(mutex);
    if (openLists.erase(id))
        releaseRetired();
}

void WebGpuDevice::releaseRetired() {
    const std::uint64_t lowestOpen = openLists.empty() ? std::numeric_limits<std::uint64_t>::max() : *openLists.begin();
    while (!retired.empty() && retired.front().first < lowestOpen) {
        const std::function<void()> release = std::move(retired.front().second);
        retired.pop_front();
        release();
    }
}

void WebGpuDevice::updateBindGroups() {
    if (groupVersion == version)
        return;
    std::vector<wgpu::BindGroupEntry> shared;
    for (std::size_t kind = 0; kind < sampledRanges.size(); ++kind)
        for (std::uint32_t index = 0; index < sampledRanges[kind].count; ++index) {
            const std::uint32_t binding = sampledRanges[kind].first + index;
            const wgpu::TextureView& view = sampledPools[binding].view ? sampledPools[binding].view : emptySampled[getPoolKind(binding)];
            shared.push_back({.binding = binding, .textureView = view});
        }
    shared.push_back({.binding = samplerBinding, .sampler = sampler});

    std::vector<wgpu::BindGroupEntry> computeEntries{{.binding = heapBinding, .buffer = heaps[0].buffer},
                                                     {.binding = atomicBinding, .buffer = heaps[1].buffer}};
    computeEntries.insert(computeEntries.end(), shared.begin(), shared.end());
    for (std::uint32_t index = 0; storageImages && index < storagePoolBindings.count; ++index)
        computeEntries.push_back(
            {.binding = storagePoolBindings.first + index, .textureView = storagePools[index].view ? storagePools[index].view : emptyStorage[index]});
    std::vector<wgpu::BindGroupEntry> renderEntries{{.binding = heapReadBinding, .buffer = heaps[0].buffer},
                                                    {.binding = atomicBinding, .buffer = heaps[1].buffer}};
    renderEntries.insert(renderEntries.end(), shared.begin(), shared.end());

    const wgpu::BindGroupDescriptor computeDescriptor{.layout = computeGroupLayout, .entryCount = computeEntries.size(), .entries = computeEntries.data()};
    computeGroup = device.CreateBindGroup(&computeDescriptor);
    const wgpu::BindGroupDescriptor renderDescriptor{.layout = renderGroupLayout, .entryCount = renderEntries.size(), .entries = renderEntries.data()};
    renderGroup = device.CreateBindGroup(&renderDescriptor);
    const wgpu::BindGroupEntry argumentEntry{.binding = argumentBinding, .buffer = argumentBuffer, .size = argumentWindowBytes};
    const wgpu::BindGroupDescriptor argumentDescriptor{.layout = argumentGroupLayout, .entryCount = 1, .entries = &argumentEntry};
    argumentGroup = device.CreateBindGroup(&argumentDescriptor);
    groupVersion = version;
}

std::expected<wgpu::RenderPipeline, Error> WebGpuDevice::getBlitPipeline(const wgpu::TextureFormat format) {
    if (const auto found = blitPipelines.find(format); found != blitPipelines.end())
        return found->second;
    const wgpu::ColorTargetState target{.format = format};
    const wgpu::FragmentState fragment{.module = blitModule, .entryPoint = "fragmentMain", .targetCount = 1, .targets = &target};
    wgpu::RenderPipelineDescriptor descriptor{.layout = blitLayout, .fragment = &fragment};
    descriptor.vertex.module = blitModule;
    descriptor.vertex.entryPoint = "vertexMain";
    pushErrorScope();
    wgpu::RenderPipeline pipeline = device.CreateRenderPipeline(&descriptor);
    if (std::expected<void, Error> created = popErrorScope("the blit pipeline"); !created)
        return std::unexpected(std::move(created.error()));
    blitPipelines.emplace(format, pipeline);
    return pipeline;
}

void WebGpuDevice::recordError(const ErrorCode code, std::string message) {
    std::lock_guard lock(errorMutex);
    if (!deviceError)
        deviceError = Error{code, std::move(message)};
}

std::expected<void, Error> WebGpuDevice::takeError() {
    std::lock_guard lock(errorMutex);
    if (!deviceError)
        return {};
    Error error = std::move(*deviceError);
    deviceError.reset();
    return std::unexpected(std::move(error));
}

void WebGpuDevice::pushErrorScope() {
    device.PushErrorScope(wgpu::ErrorFilter::Validation);
    device.PushErrorScope(wgpu::ErrorFilter::OutOfMemory);
}

std::expected<void, Error> WebGpuDevice::popErrorScope(const std::string_view operation) {
#if defined(__EMSCRIPTEN__)
    //! Waiting on an error scope yields to the browser and expires the canvas texture of the frame, so the errors reach the next takeError.
    for (const ErrorCode code : {ErrorCode::OutOfMemory, ErrorCode::InvalidArgument})
        device.PopErrorScope(wgpu::CallbackMode::AllowSpontaneous,
                             [owner = weak_from_this(), code, label = std::string(operation)](const wgpu::PopErrorScopeStatus, const wgpu::ErrorType type,
                                                                                              const wgpu::StringView text) {
                                 if (type == wgpu::ErrorType::NoError)
                                     return;
                                 if (const std::shared_ptr<WebGpuDevice> locked = owner.lock())
                                     locked->recordError(code, label + " failed: " + toString(text));
                             });
    return {};
#else
    std::optional<Error> error;
    for (const ErrorCode code : {ErrorCode::OutOfMemory, ErrorCode::InvalidArgument}) {
        const wgpu::Future future =
            device.PopErrorScope(wgpu::CallbackMode::WaitAnyOnly,
                                 [&error, code, operation](const wgpu::PopErrorScopeStatus, const wgpu::ErrorType type, const wgpu::StringView text) {
                                     if (type != wgpu::ErrorType::NoError && !error)
                                         error = Error{code, std::string(operation) + " failed: " + toString(text)};
                                 });
        if (std::expected<void, Error> waited = waitFuture(future); !waited)
            return waited;
    }
    if (error)
        return std::unexpected(std::move(*error));
    return {};
#endif
}

std::expected<void, Error> WebGpuDevice::waitFuture(const wgpu::Future future) {
    const wgpu::WaitStatus status = instance.WaitAny(future, std::numeric_limits<std::uint64_t>::max());
    if (status != wgpu::WaitStatus::Success)
        return fail(ErrorCode::InvalidState, "waiting on a WebGPU future failed");
    return {};
}

std::expected<Token, Error> WebGpuDevice::submitCommands(const wgpu::CommandBuffer& commands) {
    queue.Submit(1, &commands);
    const std::uint64_t value = ++submitted;
    const wgpu::Future future =
        queue.OnSubmittedWorkDone(wgpu::CallbackMode::AllowProcessEvents, [this, value](const wgpu::QueueWorkDoneStatus, const wgpu::StringView) {
            std::uint64_t done = completed.load();
            while (done < value && !completed.compare_exchange_weak(done, value)) {
            }
        });
    pending.emplace_back(value, future);
    return Token{value, Queue::Graphics};
}

void WebGpuDevice::reap() {
    while (!pending.empty() && pending.front().first <= completed.load())
        pending.pop_front();
}

std::expected<Token, Error> WebGpuDevice::submit(CommandList& list, const std::span<const Token>) {
    auto& webList = static_cast<WebGpuCommandList&>(list);
    if (webList.device.get() != this)
        return fail(ErrorCode::InvalidArgument, "the command list belongs to another device");
    if (webList.closed)
        return fail(ErrorCode::InvalidState, "the command list was already submitted");
    webList.closed = true;
    if (webList.error)
        return std::unexpected(*webList.error);
    if (webList.rendering)
        return fail(ErrorCode::InvalidState, "the command list ends inside beginRender");
    if (webList.concurrent)
        return fail(ErrorCode::InvalidState, "the command list ends inside beginConcurrent");

    std::lock_guard lock(mutex);
    if (!webList.arguments.empty()) {
        const wgpu::Buffer previous = argumentBuffer;
        reserve(argumentBuffer, webList.arguments.size() + argumentWindowBytes, argumentUsage);
        version += argumentBuffer.Get() != previous.Get();
        queue.WriteBuffer(argumentBuffer, 0, webList.arguments.data(), webList.arguments.size());
    }
    reserve(staging, webList.uploads.size(), wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst);
    if (!webList.uploads.empty())
        queue.WriteBuffer(staging, 0, webList.uploads.data(), webList.uploads.size());
    reserve(indirect, std::uint64_t{webList.indirectDispatches} * indirectSlotBytes, wgpu::BufferUsage::Indirect | wgpu::BufferUsage::CopyDst);
    reserve(drawScratch, webList.drawScratchBytes, wgpu::BufferUsage::Storage | wgpu::BufferUsage::Indirect);
    for (const std::shared_ptr<WebGpuReadback>& readback : webList.readbacks)
        readback->buffer.Unmap();

    const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    std::vector<std::shared_ptr<WebGpuTimestampQuery>> ended;
    std::expected<void, Error> replayed = replay(webList, encoder, ended);
    if (!replayed)
        return std::unexpected(std::move(replayed.error()));
    std::expected<Token, Error> token = submitCommands(encoder.Finish());
    closeList(webList.id);
    for (const std::shared_ptr<WebGpuReadback>& readback : webList.readbacks) {
        if (!readback->recorded)
            continue;
        readback->recorded = false;
        readback->map = std::make_shared<MapState>();
        readback->future = readback->buffer.MapAsync(wgpu::MapMode::Read, 0, readback->buffer.GetSize(), wgpu::CallbackMode::AllowProcessEvents,
                                                     [map = readback->map](const wgpu::MapAsyncStatus status, const wgpu::StringView text) {
                                                         map->status = status;
                                                         map->message = toString(text);
                                                         map->finished = true;
                                                     });
    }
    for (const std::shared_ptr<WebGpuTimestampQuery>& query : ended) {
        query->mapping = true;
        query->readback.MapAsync(wgpu::MapMode::Read, 0, 16, wgpu::CallbackMode::AllowProcessEvents,
                                 [query](const wgpu::MapAsyncStatus status, const wgpu::StringView) {
                                     if (status == wgpu::MapAsyncStatus::Success) {
                                         std::array<std::uint64_t, 2> values{};
                                         std::memcpy(values.data(), query->readback.GetConstMappedRange(0, 16), sizeof(values));
                                         query->readback.Unmap();
                                         query->milliseconds = static_cast<double>(values[1] - values[0]) * 1.0e-6;
                                     }
                                     query->mapping = false;
                                 });
    }
    instance.ProcessEvents();
    reap();
    if (std::expected<void, Error> checked = takeError(); !checked)
        return std::unexpected(std::move(checked.error()));
    return Token{token->value, list.queue};
}

std::expected<void, Error> WebGpuDevice::download(const std::uint64_t size, const std::function<void(const wgpu::CommandEncoder&, const wgpu::Buffer&)>& record,
                                                  const std::function<void(std::span<const std::byte>)>& read) {
    wgpu::Buffer readback;
    {
        std::lock_guard lock(mutex);
        const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst, .size = alignUp(size, 4)};
        readback = device.CreateBuffer(&descriptor);
        const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
        record(encoder, readback);
        static_cast<void>(submitCommands(encoder.Finish()));
    }
    wgpu::MapAsyncStatus status = wgpu::MapAsyncStatus::Error;
    std::string message;
    const wgpu::Future future = readback.MapAsync(wgpu::MapMode::Read, 0, alignUp(size, 4), wgpu::CallbackMode::WaitAnyOnly,
                                                  [&status, &message](const wgpu::MapAsyncStatus result, const wgpu::StringView text) {
                                                      status = result;
                                                      message = toString(text);
                                                  });
    if (std::expected<void, Error> waited = waitFuture(future); !waited)
        return waited;
    if (std::expected<void, Error> checked = takeError(); !checked)
        return checked;
    if (status != wgpu::MapAsyncStatus::Success)
        return fail(ErrorCode::DeviceLost, "mapping the WebGPU readback buffer failed: " + message);
    read(std::span(static_cast<const std::byte*>(readback.GetConstMappedRange(0, alignUp(size, 4))), size));
    readback.Unmap();
    return {};
}

std::expected<void, Error> WebGpuDevice::wait(const Token token) {
    if (!token || completed.load() >= token.value)
        return takeError();
    wgpu::Future future;
    {
        std::lock_guard lock(mutex);
        const auto found = std::ranges::find_if(pending, [&token](const auto& entry) { return entry.first >= token.value; });
        if (found == pending.end())
            return fail(ErrorCode::InvalidArgument, "the token belongs to no submission of this device");
        future = found->second;
    }
    if (std::expected<void, Error> waited = waitFuture(future); !waited)
        return waited;
    {
        std::lock_guard lock(mutex);
        reap();
    }
    return takeError();
}

bool WebGpuDevice::isFinished(const Token token) {
    if (completed.load() >= token.value)
        return true;
    instance.ProcessEvents();
    return completed.load() >= token.value;
}

std::expected<void, Error> WebGpuDevice::waitIdle() {
    std::uint64_t last = 0;
    {
        std::lock_guard lock(mutex);
        last = submitted;
    }
    return wait({last, Queue::Graphics});
}

std::expected<std::shared_ptr<Readback>, Error> openReadback(NoorRHI::Device& device, const std::size_t bytes) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    auto& owner = static_cast<WebGpuDevice&>(*device.getBackend());
    auto result = std::make_shared<WebGpuReadback>();
    result->device = owner.shared_from_this();
    result->size = bytes;
    std::lock_guard lock(owner.mutex);
    owner.pushErrorScope();
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst, .size = alignUp(bytes + 3, 4)};
    result->buffer = owner.device.CreateBuffer(&descriptor);
    if (std::expected<void, Error> created = owner.popErrorScope("creating a WebGPU readback buffer"); !created)
        return std::unexpected(std::move(created.error()));
    return result;
}

bool WebGpuReadback::isReady() {
    if (!map || recorded)
        return false;
    if (!map->finished)
        device->instance.ProcessEvents();
    return map->finished;
}

std::expected<std::span<const std::byte>, Error> WebGpuReadback::read() {
    if (!map || recorded)
        return fail(ErrorCode::InvalidState, "no submitted command list downloads into the readback");
    if (!map->finished)
        if (std::expected<void, Error> waited = device->waitFuture(future); !waited)
            return std::unexpected(std::move(waited.error()));
    if (std::expected<void, Error> checked = device->takeError(); !checked)
        return std::unexpected(std::move(checked.error()));
    if (map->status != wgpu::MapAsyncStatus::Success)
        return fail(ErrorCode::DeviceLost, "mapping the WebGPU readback failed: " + map->message);
    const auto* mapped = static_cast<const std::byte*>(buffer.GetConstMappedRange(0, buffer.GetSize()));
    return std::span(mapped + head, size);
}

}

namespace NoorRHI {

std::expected<void, Error> Device::open(const DeviceConfig& config) {
    auto opened = std::make_shared<Backend::WebGpuDevice>();
    if (std::expected<void, Error> result = opened->open(config); !result)
        return result;
    backend = std::move(opened);
    return {};
}

}
