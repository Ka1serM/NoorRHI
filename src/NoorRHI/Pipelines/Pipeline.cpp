#include "NoorRHI/Pipelines/Pipeline.h"

#include "NoorRHI/Core/Backend.h"

namespace NoorRHI::Backend {

std::expected<void, Error> checkShader(const Shader& shader, const ShaderStage stage) {
    if (!shader || shader.entryPoint.empty())
        return fail(ErrorCode::InvalidArgument, "a pipeline needs shader code and an entry point");
    if (shader.stage != stage)
        return fail(ErrorCode::InvalidArgument, "the shader " + std::string(shader.entryPoint) + " is not of the stage that the pipeline needs");
    if (stage == ShaderStage::Compute && (shader.threadGroupSize.x == 0 || shader.threadGroupSize.y == 0 || shader.threadGroupSize.z == 0))
        return fail(ErrorCode::InvalidArgument, "a compute shader needs a thread group size");
    return {};
}

std::expected<void, Error> checkGraphicsPipelineDesc(const GraphicsPipelineDesc& desc) {
    if (std::expected<void, Error> checked = checkShader(desc.vertex, ShaderStage::Vertex); !checked)
        return checked;
    if (std::expected<void, Error> checked = checkShader(desc.fragment, ShaderStage::Fragment); !checked)
        return checked;
    const FormatInfo color = getFormatInfo(desc.colorFormat);
    const FormatInfo depth = getFormatInfo(desc.depthFormat);
    const bool hasDepth = desc.depthFormat != ImageFormat::Undefined;
    if (desc.colorFormat == ImageFormat::Undefined || color.depth || color.blockExtent > 1)
        return fail(ErrorCode::InvalidArgument, "the color format of a graphics pipeline must be a renderable color format");
    if (hasDepth && !depth.depth)
        return fail(ErrorCode::InvalidArgument, "the depth format must be D32Float, D32FloatS8Uint or Undefined");
    if (desc.samples != 1 && desc.samples != 4)
        return fail(ErrorCode::InvalidArgument, "a graphics pipeline has 1 or 4 samples");
    if ((desc.state.depthTest || desc.state.depthWrite) && (!hasDepth || !desc.state.depthTest))
        return fail(ErrorCode::InvalidArgument, "depth writes need a depth test and a depth test needs a depth format");
    if (desc.state.stencil.isEnabled() && !depth.stencil)
        return fail(ErrorCode::InvalidArgument, "stencil operations need the D32FloatS8Uint depth format");
    if (color.integer && desc.state.blend != BlendMode::Opaque)
        return fail(ErrorCode::InvalidArgument, "integer color formats cannot blend");
    return {};
}

void setTarget(GraphicsPipeline& pipeline, const GraphicsPipelineDesc& desc) {
    pipeline.colorFormat = desc.colorFormat;
    pipeline.depthFormat = desc.depthFormat;
    pipeline.samples = desc.samples;
}

}

namespace NoorRHI {

namespace {

template <class Pipeline>
std::expected<void, Error> waitForPipeline(Pipeline& pipeline) {
    if (!pipeline)
        return fail(ErrorCode::InvalidState, "the pipeline is not open");
    return pipeline->wait();
}

}

std::expected<void, Error> ComputePipeline::open(Device& device, const Shader& shader) {
    std::expected<std::shared_ptr<Backend::ComputePipeline>, Error> opened = Backend::openComputePipeline(device, shader, false);
    if (!opened)
        return std::unexpected(std::move(opened.error()));
    if (std::expected<void, Error> compiled = (*opened)->wait(); !compiled)
        return compiled;
    backend = std::move(*opened);
    return {};
}

std::expected<void, Error> ComputePipeline::openAsync(Device& device, const Shader& shader) {
    std::expected<std::shared_ptr<Backend::ComputePipeline>, Error> opened = Backend::openComputePipeline(device, shader, true);
    if (!opened)
        return std::unexpected(std::move(opened.error()));
    backend = std::move(*opened);
    return {};
}

bool ComputePipeline::isReady() const {
    if (!backend)
        return false;
    backend->poll();
    return backend->ready;
}

std::expected<void, Error> ComputePipeline::wait() const {
    return waitForPipeline(backend);
}

std::expected<void, Error> GraphicsPipeline::open(Device& device, const GraphicsPipelineDesc& desc) {
    std::expected<std::shared_ptr<Backend::GraphicsPipeline>, Error> opened = Backend::openGraphicsPipeline(device, desc, false);
    if (!opened)
        return std::unexpected(std::move(opened.error()));
    if (std::expected<void, Error> compiled = (*opened)->wait(); !compiled)
        return compiled;
    backend = std::move(*opened);
    return {};
}

std::expected<void, Error> GraphicsPipeline::openAsync(Device& device, const GraphicsPipelineDesc& desc) {
    std::expected<std::shared_ptr<Backend::GraphicsPipeline>, Error> opened = Backend::openGraphicsPipeline(device, desc, true);
    if (!opened)
        return std::unexpected(std::move(opened.error()));
    backend = std::move(*opened);
    return {};
}

bool GraphicsPipeline::isReady() const {
    if (!backend)
        return false;
    backend->poll();
    return backend->ready;
}

std::expected<void, Error> GraphicsPipeline::wait() const {
    return waitForPipeline(backend);
}

std::expected<void, Error> prepareShader(Device& device, const Shader& shader) {
    return Backend::prepareShader(device, shader);
}

}
