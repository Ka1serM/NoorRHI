#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <vector>

#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Pipelines/Pipeline.h"
#include "NoorRHI/Vulkan/Vulkan.h"

namespace NoorRHI {

namespace {

struct MatrixType {
    MatrixComponent component = MatrixComponent::Float16;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint32_t use = 0;
};

bool isSupported(const std::span<const CooperativeMatrixShape> shapes, const MatrixType& type) {
    return std::ranges::any_of(shapes, [&type](const CooperativeMatrixShape& shape) {
        switch (type.use) {
        case 0:
            return shape.a == type.component && shape.m == type.rows && shape.k == type.columns;
        case 1:
            return shape.b == type.component && shape.k == type.rows && shape.n == type.columns;
        default:
            return (shape.c == type.component || shape.result == type.component) && shape.m == type.rows && shape.n == type.columns;
        }
    });
}

//! The device lists the cooperative matrix shapes that it runs, so a module goes native only when the device lists every matrix type that the module declares.
bool supportsMatrices(const std::span<const CooperativeMatrixShape> shapes, const std::span<const std::byte> code) {
    constexpr std::uint32_t opTypeFloat = 22;
    constexpr std::uint32_t opConstant = 43;
    constexpr std::uint32_t opTypeCooperativeMatrix = 4456;
    if (shapes.empty() || code.size() % sizeof(std::uint32_t) != 0)
        return false;
    std::vector<std::uint32_t> words(code.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), code.data(), code.size());
    std::unordered_map<std::uint32_t, MatrixComponent> components;
    std::unordered_map<std::uint32_t, std::uint32_t> constants;
    std::vector<std::array<std::uint32_t, 5>> matrices;
    for (std::size_t offset = 5; offset < words.size();) {
        const std::uint32_t opcode = words[offset] & 0xffffu;
        const std::uint32_t count = words[offset] >> 16;
        if (count == 0 || offset + count > words.size())
            return false;
        const std::span<const std::uint32_t> operands(words.data() + offset + 1, count - 1);
        if (opcode == opTypeFloat && operands.size() >= 2 && (operands[1] == 16 || operands[1] == 32))
            components.emplace(operands[0], operands[1] == 16 ? MatrixComponent::Float16 : MatrixComponent::Float32);
        if (opcode == opConstant && operands.size() == 3)
            constants.emplace(operands[1], operands[2]);
        if (opcode == opTypeCooperativeMatrix && operands.size() == 6)
            matrices.push_back({operands[1], operands[3], operands[4], operands[5], 0});
        offset += count;
    }
    for (const auto& [component, rows, columns, use, unused] : matrices) {
        const auto found = components.find(component);
        if (found == components.end() || !constants.contains(rows) || !constants.contains(columns) || !constants.contains(use))
            return false;
        if (!isSupported(shapes, {found->second, constants[rows], constants[columns], constants[use]}))
            return false;
    }
    return true;
}

vk::CullModeFlags getCullMode(const CullMode mode) {
    switch (mode) {
    case CullMode::None:
        return vk::CullModeFlagBits::eNone;
    case CullMode::Front:
        return vk::CullModeFlagBits::eFront;
    case CullMode::Back:
        return vk::CullModeFlagBits::eBack;
    }
    return vk::CullModeFlagBits::eBack;
}

vk::PolygonMode getPolygonMode(const PolygonMode mode) {
    switch (mode) {
    case PolygonMode::Fill:
        return vk::PolygonMode::eFill;
    case PolygonMode::Line:
        return vk::PolygonMode::eLine;
    case PolygonMode::Point:
        return vk::PolygonMode::ePoint;
    }
    return vk::PolygonMode::eFill;
}

vk::CompareOp getCompareOp(const CompareOp op) {
    switch (op) {
    case CompareOp::Never:
        return vk::CompareOp::eNever;
    case CompareOp::Less:
        return vk::CompareOp::eLess;
    case CompareOp::Equal:
        return vk::CompareOp::eEqual;
    case CompareOp::LessOrEqual:
        return vk::CompareOp::eLessOrEqual;
    case CompareOp::Greater:
        return vk::CompareOp::eGreater;
    case CompareOp::NotEqual:
        return vk::CompareOp::eNotEqual;
    case CompareOp::GreaterOrEqual:
        return vk::CompareOp::eGreaterOrEqual;
    case CompareOp::Always:
        return vk::CompareOp::eAlways;
    }
    return vk::CompareOp::eAlways;
}

vk::StencilOp getStencilOp(const StencilOp op) {
    switch (op) {
    case StencilOp::Keep:
        return vk::StencilOp::eKeep;
    case StencilOp::Zero:
        return vk::StencilOp::eZero;
    case StencilOp::Replace:
        return vk::StencilOp::eReplace;
    case StencilOp::Increment:
        return vk::StencilOp::eIncrementAndClamp;
    case StencilOp::Decrement:
        return vk::StencilOp::eDecrementAndClamp;
    case StencilOp::Invert:
        return vk::StencilOp::eInvert;
    }
    return vk::StencilOp::eKeep;
}

vk::PipelineColorBlendAttachmentState getBlendState(const BlendMode mode, const bool colorWrite) {
    vk::PipelineColorBlendAttachmentState blend;
    if (colorWrite)
        blend.colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    if (mode == BlendMode::Opaque)
        return blend;
    blend.blendEnable = vk::True;
    blend.colorBlendOp = vk::BlendOp::eAdd;
    blend.alphaBlendOp = vk::BlendOp::eAdd;
    blend.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    blend.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    blend.srcColorBlendFactor = mode == BlendMode::Alpha ? vk::BlendFactor::eSrcAlpha : vk::BlendFactor::eOne;
    blend.dstColorBlendFactor = mode == BlendMode::Additive ? vk::BlendFactor::eOne : vk::BlendFactor::eOneMinusSrcAlpha;
    if (mode == BlendMode::Additive)
        blend.dstAlphaBlendFactor = vk::BlendFactor::eOne;
    return blend;
}

}

namespace Backend {

std::expected<vk::UniqueShaderModule, Error> createShaderModule(const VulkanDevice& device, const Shader& shader) {
    const bool rayTracing = device.features.hardwareRayTracing;
    std::span<const std::byte> code = shader.getCode(rayTracing, false);
    if (const std::span<const std::byte> matrices = shader.getCode(rayTracing, true); matrices.data() != code.data()) {
        if (supportsMatrices(device.features.cooperativeMatrixShapes, matrices))
            code = matrices;
        else if (!device.features.subgroups || device.features.minSubgroupSize < 8 || !device.subgroupExtendedTypes)
            return fail(ErrorCode::Unsupported, "cooperative matrices without the extension need subgroups of at least 8 lanes with 16-bit operations");
    }
    if (code.empty() || code.size() % sizeof(std::uint32_t) != 0 || reinterpret_cast<std::uintptr_t>(code.data()) % alignof(std::uint32_t) != 0)
        return fail(ErrorCode::InvalidShader, "SPIR-V code must be a 4-byte aligned sequence of 32-bit words");
    return checkVulkan(device.device->createShaderModuleUnique({{}, code.size(), reinterpret_cast<const std::uint32_t*>(code.data())}), "vkCreateShaderModule");
}

std::expected<void, Error> CompileState::wait() {
    std::unique_lock lock(mutex);
    done.wait(lock, [this] { return finished; });
    if (error)
        return std::unexpected(*error);
    return {};
}

void CompileState::finish(Compilation& pipeline, std::expected<void, Error> result) {
    std::lock_guard lock(mutex);
    finished = true;
    if (!result)
        error = std::move(result.error());
    pipeline.ready = !error;
    done.notify_all();
}

void VulkanDevice::compile(std::function<void()> job) {
    std::lock_guard lock(compileMutex);
    if (compileThreads.empty()) {
        const std::uint32_t threads = std::clamp(std::thread::hardware_concurrency() / 2, 1u, 8u);
        for (std::uint32_t index = 0; index < threads; ++index)
            compileThreads.emplace_back([this] {
                std::unique_lock lock(compileMutex);
                while (true) {
                    compileSignal.wait(lock, [this] { return stopCompiling || !compileJobs.empty(); });
                    if (compileJobs.empty())
                        return;
                    std::function<void()> next = std::move(compileJobs.front());
                    compileJobs.pop_front();
                    lock.unlock();
                    next();
                    lock.lock();
                }
            });
    }
    compileJobs.push_back(std::move(job));
    compileSignal.notify_one();
}

VulkanComputePipeline::~VulkanComputePipeline() {
    if (!device)
        return;
    static_cast<void>(state.wait());
    if (pipeline)
        device->retire([owner = device.get(), pipeline = this->pipeline] { owner->device->destroyPipeline(pipeline); });
}

VulkanGraphicsPipeline::~VulkanGraphicsPipeline() {
    if (!device)
        return;
    static_cast<void>(state.wait());
    if (pipeline)
        device->retire([owner = device.get(), pipeline = this->pipeline] { owner->device->destroyPipeline(pipeline); });
}

namespace {

std::expected<vk::Pipeline, Error> createComputePipeline(const VulkanDevice& owner, const Shader& shader) {
    std::expected<vk::UniqueShaderModule, Error> module = createShaderModule(owner, shader);
    if (!module)
        return std::unexpected(std::move(module.error()));
    const std::string entryPoint(shader.entryPoint);
    const vk::PipelineShaderStageCreateInfo stage({}, vk::ShaderStageFlagBits::eCompute, **module, entryPoint.c_str());
    return checkVulkan(owner.device->createComputePipeline(owner.getPipelineCache(true), {{}, stage, *owner.pipelineLayout}), "vkCreateComputePipelines");
}

std::expected<vk::Pipeline, Error> createGraphicsPipeline(const VulkanDevice& owner, const GraphicsPipelineDesc& desc) {
    std::expected<vk::UniqueShaderModule, Error> vertex = createShaderModule(owner, desc.vertex);
    if (!vertex)
        return std::unexpected(std::move(vertex.error()));
    std::expected<vk::UniqueShaderModule, Error> fragment = createShaderModule(owner, desc.fragment);
    if (!fragment)
        return std::unexpected(std::move(fragment.error()));
    const std::string vertexEntry(desc.vertex.entryPoint);
    const std::string fragmentEntry(desc.fragment.entryPoint);
    const std::array stages{
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, **vertex, vertexEntry.c_str()),
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, **fragment, fragmentEntry.c_str()),
    };
    const vk::PipelineVertexInputStateCreateInfo vertexInput;
    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList);
    const vk::PipelineViewportStateCreateInfo viewport({}, 1, nullptr, 1, nullptr);
    const vk::PipelineRasterizationStateCreateInfo raster(
        {}, vk::False, vk::False, getPolygonMode(desc.state.polygon), getCullMode(desc.state.cull),
        desc.state.frontFace == FrontFace::Clockwise ? vk::FrontFace::eClockwise : vk::FrontFace::eCounterClockwise, vk::False, 0, 0, 0, 1.0f);
    const vk::PipelineMultisampleStateCreateInfo multisample({}, static_cast<vk::SampleCountFlagBits>(desc.samples));
    const StencilState& stencil = desc.state.stencil;
    const vk::StencilOpState stencilOps(vk::StencilOp::eKeep, getStencilOp(stencil.pass), vk::StencilOp::eKeep, getCompareOp(stencil.compare), 0xff,
                                        stencil.pass == StencilOp::Keep ? 0u : 0xffu, 0);
    vk::PipelineDepthStencilStateCreateInfo depth;
    depth.depthTestEnable = desc.state.depthTest;
    depth.depthWriteEnable = desc.state.depthWrite;
    depth.depthCompareOp = getCompareOp(desc.state.depthCompare);
    depth.stencilTestEnable = stencil.isEnabled();
    depth.front = stencilOps;
    depth.back = stencilOps;
    const vk::PipelineColorBlendAttachmentState blend = getBlendState(desc.state.blend, desc.state.colorWrite);
    const vk::PipelineColorBlendStateCreateInfo colorBlend({}, vk::False, vk::LogicOp::eCopy, blend);
    constexpr std::array dynamicStates{vk::DynamicState::eViewport, vk::DynamicState::eScissor, vk::DynamicState::eStencilReference};
    const vk::PipelineDynamicStateCreateInfo dynamic({}, dynamicStates);
    const vk::Format colorFormat = getVulkanFormat(desc.colorFormat);
    vk::PipelineRenderingCreateInfo rendering({}, colorFormat, getVulkanFormat(desc.depthFormat));
    if (getFormatInfo(desc.depthFormat).stencil)
        rendering.stencilAttachmentFormat = rendering.depthAttachmentFormat;

    vk::GraphicsPipelineCreateInfo info;
    info.setStages(stages)
        .setPVertexInputState(&vertexInput)
        .setPInputAssemblyState(&inputAssembly)
        .setPViewportState(&viewport)
        .setPRasterizationState(&raster)
        .setPMultisampleState(&multisample)
        .setPDepthStencilState(&depth)
        .setPColorBlendState(&colorBlend)
        .setPDynamicState(&dynamic)
        .setLayout(*owner.pipelineLayout)
        .setPNext(&rendering);
    return checkVulkan(owner.device->createGraphicsPipeline(owner.getPipelineCache(true), info), "vkCreateGraphicsPipelines");
}

template <class Pipeline, class Create>
void start(VulkanDevice& owner, Pipeline& result, const bool background, Create create) {
    auto job = [&result, create = std::move(create)] {
        std::expected<vk::Pipeline, Error> created = create(*result.device);
        if (created)
            result.pipeline = *created;
        result.state.finish(result, created.transform([](vk::Pipeline) {}));
    };
    if (!background)
        return job();
    owner.compile(std::move(job));
}

}

std::expected<std::shared_ptr<ComputePipeline>, Error> openComputePipeline(NoorRHI::Device& device, const Shader& shader, const bool background) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    if (std::expected<void, Error> checked = checkShader(shader, ShaderStage::Compute); !checked)
        return std::unexpected(std::move(checked.error()));
    auto& owner = static_cast<VulkanDevice&>(*device.getBackend());
    auto result = std::make_shared<VulkanComputePipeline>();
    result->device = owner.shared_from_this();
    result->threadGroupSize = shader.threadGroupSize;
    start(owner, *result, background, [shader](const VulkanDevice& compiler) { return createComputePipeline(compiler, shader); });
    return result;
}

std::expected<std::shared_ptr<GraphicsPipeline>, Error> openGraphicsPipeline(NoorRHI::Device& device, const GraphicsPipelineDesc& desc, const bool background) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    if (std::expected<void, Error> checked = checkGraphicsPipelineDesc(desc); !checked)
        return std::unexpected(std::move(checked.error()));
    auto& owner = static_cast<VulkanDevice&>(*device.getBackend());
    auto result = std::make_shared<VulkanGraphicsPipeline>();
    result->device = owner.shared_from_this();
    setTarget(*result, desc);
    start(owner, *result, background, [desc](const VulkanDevice& compiler) { return createGraphicsPipeline(compiler, desc); });
    return result;
}

}

}
