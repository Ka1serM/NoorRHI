#include <cstring>
#include <vector>

#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Pipelines/Pipeline.h"
#include "NoorRHI/Shaders/SpirvModule.h"
#include "NoorRHI/Shaders/WgslTranslator.h"
#include "NoorRHI/WebGpu/WebGpu.h"

namespace NoorRHI {

namespace {

//! The device declares only the float pools that its limits allow, so the other float pools move onto the first one.
std::expected<void, Error> moveFloatPools(SpirvModule& module, const Backend::PoolBindings used) {
    const Backend::PoolBindings declared = Backend::sampledPoolBindings[static_cast<std::size_t>(Backend::PoolKind::Float)];
    for (std::uint32_t binding = used.first + used.count; binding < declared.first + declared.count; ++binding)
        if (std::expected<void, Error> moved = module.moveBinding(0, binding, declared.first); !moved)
            return moved;
    return {};
}

wgpu::CullMode getCullMode(const CullMode mode) {
    switch (mode) {
    case CullMode::None:
        return wgpu::CullMode::None;
    case CullMode::Front:
        return wgpu::CullMode::Front;
    case CullMode::Back:
        return wgpu::CullMode::Back;
    }
    return wgpu::CullMode::Back;
}

wgpu::CompareFunction getCompare(const CompareOp op) {
    switch (op) {
    case CompareOp::Never:
        return wgpu::CompareFunction::Never;
    case CompareOp::Less:
        return wgpu::CompareFunction::Less;
    case CompareOp::Equal:
        return wgpu::CompareFunction::Equal;
    case CompareOp::LessOrEqual:
        return wgpu::CompareFunction::LessEqual;
    case CompareOp::Greater:
        return wgpu::CompareFunction::Greater;
    case CompareOp::NotEqual:
        return wgpu::CompareFunction::NotEqual;
    case CompareOp::GreaterOrEqual:
        return wgpu::CompareFunction::GreaterEqual;
    case CompareOp::Always:
        return wgpu::CompareFunction::Always;
    }
    return wgpu::CompareFunction::Always;
}

wgpu::StencilOperation getStencilOp(const StencilOp op) {
    switch (op) {
    case StencilOp::Keep:
        return wgpu::StencilOperation::Keep;
    case StencilOp::Zero:
        return wgpu::StencilOperation::Zero;
    case StencilOp::Replace:
        return wgpu::StencilOperation::Replace;
    case StencilOp::Increment:
        return wgpu::StencilOperation::IncrementClamp;
    case StencilOp::Decrement:
        return wgpu::StencilOperation::DecrementClamp;
    case StencilOp::Invert:
        return wgpu::StencilOperation::Invert;
    }
    return wgpu::StencilOperation::Keep;
}

wgpu::BlendState getBlendState(const BlendMode mode) {
    if (mode == BlendMode::Multiply)
        return {.color = {.operation = wgpu::BlendOperation::Add, .srcFactor = wgpu::BlendFactor::Dst, .dstFactor = wgpu::BlendFactor::Zero},
                .alpha = {.operation = wgpu::BlendOperation::Add, .srcFactor = wgpu::BlendFactor::Zero, .dstFactor = wgpu::BlendFactor::One}};
    const wgpu::BlendFactor colorSource = mode == BlendMode::Alpha ? wgpu::BlendFactor::SrcAlpha : wgpu::BlendFactor::One;
    const wgpu::BlendFactor colorDestination = mode == BlendMode::Additive ? wgpu::BlendFactor::One : wgpu::BlendFactor::OneMinusSrcAlpha;
    const wgpu::BlendFactor alphaDestination = mode == BlendMode::Additive ? wgpu::BlendFactor::One : wgpu::BlendFactor::OneMinusSrcAlpha;
    return {.color = {.operation = wgpu::BlendOperation::Add, .srcFactor = colorSource, .dstFactor = colorDestination},
            .alpha = {.operation = wgpu::BlendOperation::Add, .srcFactor = wgpu::BlendFactor::One, .dstFactor = alphaDestination}};
}

std::expected<void, Error> asShaderError(std::expected<void, Error> result) {
    if (!result && result.error().code == ErrorCode::InvalidArgument)
        result.error().code = ErrorCode::InvalidShader;
    return result;
}

std::string_view contentOf(const std::span<const std::byte> code) {
    return {reinterpret_cast<const char*>(code.data()), code.size()};
}

std::expected<std::string, Error> translateModule(const Backend::WebGpuDevice& owner, const std::span<const std::byte> code) {
    if (code.size() % sizeof(std::uint32_t) != 0)
        return fail(ErrorCode::InvalidShader, "a WebGPU shader must be a SPIR-V module");
    std::vector<std::uint32_t> words(code.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), code.data(), code.size());
    SpirvModule spirv;
    if (std::expected<void, Error> read = spirv.read(words); !read)
        return std::unexpected(std::move(read.error()));
    //! SPIR-V keeps no type names, and the cooperative matrix fallback shuffles across groups of eight lanes, so each shader with subgroup shuffles needs at least 8 lanes.
    if (spirv.hasCapability(spv::Capability::GroupNonUniformShuffle) && (!owner.features.subgroups || owner.features.minSubgroupSize < 8))
        return fail(ErrorCode::Unsupported, "subgroup shuffles on WebGPU need subgroups of at least 8 lanes");
    if (!owner.storageImages && spirv.hasStorageImages())
        return fail(ErrorCode::Unsupported, "a WebGPU shader with storage images needs the texture-formats-tier2 feature");
    if (std::expected<void, Error> moved = moveFloatPools(spirv, owner.sampledRanges[static_cast<std::size_t>(Backend::PoolKind::Float)]); !moved)
        return std::unexpected(std::move(moved.error()));
    return translateToWgsl(spirv);
}

std::expected<std::string, Error> takeTranslation(Backend::WebGpuDevice& owner, const std::span<const std::byte> code) {
    {
        std::lock_guard lock(owner.mutex);
        if (auto prepared = owner.translatedShaders.find(contentOf(code)); prepared != owner.translatedShaders.end())
            return std::move(owner.translatedShaders.extract(prepared).mapped());
    }
    return translateModule(owner, code);
}

std::expected<wgpu::ShaderModule, Error> createModule(Backend::WebGpuDevice& owner, const Shader& shader) {
    const std::span<const std::byte> code = shader.getCode(false, false);
    {
        std::lock_guard lock(owner.mutex);
        if (const auto cached = owner.shaderModules.find(code.data()); cached != owner.shaderModules.end())
            return cached->second;
    }
    std::expected<std::string, Error> text = takeTranslation(owner, code);
    if (!text)
        return std::unexpected(std::move(text.error()));
    wgpu::ShaderSourceWGSL wgsl;
    wgsl.code = wgpu::StringView(text->data(), text->size());
    const wgpu::ShaderModuleDescriptor descriptor{.nextInChain = &wgsl};
    std::lock_guard lock(owner.mutex);
    owner.pushErrorScope();
    wgpu::ShaderModule module = owner.device.CreateShaderModule(&descriptor);
    if (std::expected<void, Error> created = asShaderError(owner.popErrorScope("compiling WGSL")); !created)
        return std::unexpected(std::move(created.error()));
    owner.shaderModules.emplace(code.data(), module);
    return module;
}

wgpu::StringView getEntryPoint(const Shader& shader) {
    return {shader.entryPoint.data(), shader.entryPoint.size()};
}

template <class Pipeline, class Native>
void finishCompilation(const std::weak_ptr<Pipeline>& weak, const wgpu::CreatePipelineAsyncStatus status, Native native, const wgpu::StringView message) {
    const std::shared_ptr<Pipeline> pipeline = weak.lock();
    if (!pipeline)
        return;
    std::lock_guard lock(pipeline->compilation.mutex);
    pipeline->compilation.finished = true;
    if (status == wgpu::CreatePipelineAsyncStatus::Success) {
        pipeline->pipeline = std::move(native);
        pipeline->ready = true;
        return;
    }
    const ErrorCode code = status == wgpu::CreatePipelineAsyncStatus::ValidationError ? ErrorCode::InvalidShader : ErrorCode::InvalidState;
    pipeline->compilation.error = Error{code, "creating a WebGPU pipeline failed: " + Backend::toString(message)};
}

std::expected<void, Error> waitCompilation(Backend::WebGpuCompilation& compilation) {
    bool finished = false;
    {
        std::lock_guard lock(compilation.mutex);
        finished = compilation.finished;
    }
    if (!finished)
        if (std::expected<void, Error> waited = compilation.device->waitFuture(compilation.future); !waited)
            return waited;
    std::lock_guard lock(compilation.mutex);
    if (compilation.error)
        return std::unexpected(*compilation.error);
    return {};
}

}

namespace Backend {

std::expected<void, Error> prepareShader(NoorRHI::Device& device, const Shader& shader) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    auto& owner = static_cast<WebGpuDevice&>(*device.getBackend());
    const std::span<const std::byte> code = shader.getCode(false, false);
    {
        std::lock_guard lock(owner.mutex);
        if (owner.shaderModules.contains(code.data()) || owner.translatedShaders.contains(contentOf(code)))
            return {};
    }
    std::expected<std::string, Error> text = translateModule(owner, code);
    if (!text)
        return std::unexpected(std::move(text.error()));
    std::lock_guard lock(owner.mutex);
    owner.translatedShaders.emplace(contentOf(code), std::move(*text));
    return {};
}

void WebGpuComputePipeline::poll() {
    if (!ready)
        compilation.device->instance.ProcessEvents();
}

std::expected<void, Error> WebGpuComputePipeline::wait() {
    return waitCompilation(compilation);
}

void WebGpuGraphicsPipeline::poll() {
    if (!ready)
        compilation.device->instance.ProcessEvents();
}

std::expected<void, Error> WebGpuGraphicsPipeline::wait() {
    return waitCompilation(compilation);
}

//! Dawn compiles in the background either way, so open() and openAsync() differ only in whether Pipeline waits.
std::expected<std::shared_ptr<ComputePipeline>, Error> openComputePipeline(NoorRHI::Device& device, const Shader& shader, bool) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    if (std::expected<void, Error> checked = checkShader(shader, ShaderStage::Compute); !checked)
        return std::unexpected(std::move(checked.error()));
    auto& owner = static_cast<WebGpuDevice&>(*device.getBackend());
    std::expected<wgpu::ShaderModule, Error> module = createModule(owner, shader);
    if (!module)
        return std::unexpected(std::move(module.error()));
    wgpu::ComputePipelineDescriptor descriptor{.layout = owner.computeLayout};
    descriptor.compute.module = *module;
    descriptor.compute.entryPoint = getEntryPoint(shader);
    auto result = std::make_shared<WebGpuComputePipeline>();
    result->threadGroupSize = shader.threadGroupSize;
    result->compilation.device = owner.shared_from_this();
    std::lock_guard lock(owner.mutex);
    result->compilation.future = owner.device.CreateComputePipelineAsync(
        &descriptor, wgpu::CallbackMode::AllowProcessEvents,
        [weak = result->weak_from_this()](const wgpu::CreatePipelineAsyncStatus status, wgpu::ComputePipeline pipeline, const wgpu::StringView message) {
            finishCompilation(weak, status, std::move(pipeline), message);
        });
    return result;
}

std::expected<std::shared_ptr<GraphicsPipeline>, Error> openGraphicsPipeline(NoorRHI::Device& device, const GraphicsPipelineDesc& desc, bool) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    if (std::expected<void, Error> checked = checkGraphicsPipelineDesc(desc); !checked)
        return std::unexpected(std::move(checked.error()));
    if (desc.state.polygon != PolygonMode::Fill)
        return fail(ErrorCode::Unsupported, "WebGPU draws only filled polygons");
    auto& owner = static_cast<WebGpuDevice&>(*device.getBackend());
    if (!owner.isRenderable(desc.colorFormat))
        return fail(ErrorCode::Unsupported, "the WebGPU device cannot render to the color format of the pipeline");
    std::expected<wgpu::ShaderModule, Error> vertex = createModule(owner, desc.vertex);
    if (!vertex)
        return std::unexpected(std::move(vertex.error()));
    std::expected<wgpu::ShaderModule, Error> fragment = createModule(owner, desc.fragment);
    if (!fragment)
        return std::unexpected(std::move(fragment.error()));
    const bool usesDepth = desc.depthFormat != ImageFormat::Undefined;
    const wgpu::TextureFormat colorFormat = getWebGpuFormat(desc.colorFormat);

    const wgpu::BlendState blend = getBlendState(desc.state.blend);
    const wgpu::ColorTargetState target{.format = colorFormat,
                                        .blend = desc.state.blend == BlendMode::Opaque ? nullptr : &blend,
                                        .writeMask = desc.state.colorWrite ? wgpu::ColorWriteMask::All : wgpu::ColorWriteMask::None};
    const wgpu::FragmentState fragmentState{.module = *fragment, .entryPoint = getEntryPoint(desc.fragment), .targetCount = 1, .targets = &target};
    const StencilState& stencil = desc.state.stencil;
    const wgpu::StencilFaceState stencilFace{.compare = getCompare(stencil.compare), .passOp = getStencilOp(stencil.pass)};
    const wgpu::DepthStencilState depthState{
        .format = getWebGpuFormat(desc.depthFormat),
        .depthWriteEnabled = desc.state.depthWrite ? wgpu::OptionalBool::True : wgpu::OptionalBool::False,
        .depthCompare = desc.state.depthTest ? getCompare(desc.state.depthCompare) : wgpu::CompareFunction::Always,
        .stencilFront = stencilFace,
        .stencilBack = stencilFace,
        .stencilReadMask = 0xff,
        .stencilWriteMask = stencil.pass == StencilOp::Keep ? 0u : 0xffu,
    };
    wgpu::RenderPipelineDescriptor descriptor{
        .layout = owner.renderLayout,
        .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList,
                      .frontFace = desc.state.frontFace == FrontFace::Clockwise ? wgpu::FrontFace::CW : wgpu::FrontFace::CCW,
                      .cullMode = getCullMode(desc.state.cull)},
        .depthStencil = usesDepth ? &depthState : nullptr,
        .multisample = {.count = desc.samples},
        .fragment = &fragmentState};
    descriptor.vertex.module = *vertex;
    descriptor.vertex.entryPoint = getEntryPoint(desc.vertex);
    auto result = std::make_shared<WebGpuGraphicsPipeline>();
    setTarget(*result, desc);
    result->compilation.device = owner.shared_from_this();
    std::lock_guard lock(owner.mutex);
    result->compilation.future = owner.device.CreateRenderPipelineAsync(
        &descriptor, wgpu::CallbackMode::AllowProcessEvents,
        [weak = result->weak_from_this()](const wgpu::CreatePipelineAsyncStatus status, wgpu::RenderPipeline pipeline, const wgpu::StringView message) {
            finishCompilation(weak, status, std::move(pipeline), message);
        });
    return result;
}

}

}
