#include <bit>
#include <cstring>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Vulkan/Vulkan.h"

namespace NoorRHI {

CommandList::CommandList(Device& device, const Queue queue) {
    if (device)
        backend = static_cast<Backend::VulkanDevice&>(*device.getBackend()).openList(queue);
}

namespace Backend {

namespace {

//! Draws only read in these stages, so a later command does not wait for them.
constexpr vk::PipelineStageFlags2 renderReadStages =
    vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eIndexInput | vk::PipelineStageFlagBits2::eVertexAttributeInput;
constexpr vk::PipelineStageFlags2 renderStages = vk::PipelineStageFlagBits2::eVertexShader | vk::PipelineStageFlagBits2::eFragmentShader
                                                 | vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests
                                                 | vk::PipelineStageFlagBits2::eColorAttachmentOutput;
constexpr vk::PipelineStageFlags2 copyStages = vk::PipelineStageFlagBits2::eAllTransfer;

void recordBarrier(const vk::CommandBuffer commands, const vk::PipelineStageFlags2 source, const vk::PipelineStageFlags2 destination,
                   const vk::AccessFlags2 destinationAccess) {
    const vk::MemoryBarrier2 memory(source, vk::AccessFlagBits2::eMemoryWrite, destination, destinationAccess);
    commands.pipelineBarrier2(vk::DependencyInfo().setMemoryBarriers(memory));
}

bool overlaps(const std::span<const CopyRange> left, const std::span<const CopyRange> right) {
    return std::ranges::any_of(left, [right](const CopyRange& first) {
        return std::ranges::any_of(right, [&first](const CopyRange& second) {
            return first.resource == second.resource && (first.resource != 0 || (first.begin < second.end && second.begin < first.end));
        });
    });
}

CopyRange getImageRange(const vk::Image image) {
    return {std::bit_cast<std::uint64_t>(static_cast<VkImage>(image))};
}

}

VulkanCommandList::~VulkanCommandList() {
    device->discard(*this);
}

std::expected<Placement, Error> VulkanCommandList::place(const std::span<const std::byte> data) {
    const std::size_t bytes = (data.size() + argumentAlignment - 1) & ~(argumentAlignment - 1);
    ArgumentChunk* chunk = chunks.empty() ? nullptr : &chunks.back();
    std::size_t offset = chunk ? chunk->used : 0;
    if (!chunk || offset + bytes > chunk->memory.size) {
        Allocation memory;
        if (bytes <= argumentChunkBytes) {
            std::lock_guard lock(device->mutex);
            if (!device->freeChunks.empty()) {
                memory = device->freeChunks.back();
                device->freeChunks.pop_back();
            }
        }
        if (!memory.buffer) {
            std::expected<Allocation, Error> allocated = device->allocate(std::max(bytes, argumentChunkBytes), MemoryKind::Upload);
            if (!allocated)
                return std::unexpected(std::move(allocated.error()));
            memory = *allocated;
        }
        chunks.push_back({memory, 0});
        chunk = &chunks.back();
        offset = 0;
    }
    std::memcpy(chunk->memory.mapped + offset, data.data(), data.size());
    chunk->used = offset + bytes;
    return Placement{chunk->memory.buffer, chunk->memory.offset + offset, chunk->memory.address + offset};
}

void VulkanCommandList::pushRoot(const std::span<const std::byte> arguments) {
    vk::DeviceAddress address = 0;
    if (!arguments.empty()) {
        std::expected<Placement, Error> placed = place(arguments);
        if (!placed)
            return static_cast<void>(check(false, placed.error().code, placed.error().message));
        address = placed->address;
    }
    if (address == boundRoot)
        return;
    boundRoot = address;
    pool.commands.pushConstants(*device->pipelineLayout, vk::ShaderStageFlagBits::eAll, 0, sizeof(address), &address);
}

void VulkanCommandList::bind(const vk::PipelineBindPoint point, const vk::Pipeline pipeline) {
    vk::Pipeline& bound = boundPipelines[point == vk::PipelineBindPoint::eCompute ? 0 : point == vk::PipelineBindPoint::eGraphics ? 1 : 2];
    if (bound == pipeline)
        return;
    if (!bound)
        pool.commands.bindDescriptorSets(point, *device->pipelineLayout, 0, device->descriptorSet, {});
    bound = pipeline;
    pool.commands.bindPipeline(point, pipeline);
}

bool VulkanCommandList::bindIndices(const std::uint64_t address) {
    if (address == 0 || address == boundIndices)
        return true;
    const std::optional<BufferRange> buffer = device->findBuffer(address);
    if (!check(buffer.has_value(), ErrorCode::InvalidArgument, "the indices are not in a buffer of this device"))
        return false;
    boundIndices = address;
    pool.commands.bindIndexBuffer(buffer->buffer, buffer->offset, vk::IndexType::eUint32);
    return true;
}

//! A barrier waits only for the stages that ran since the last one and blocks only the next stages; barriers chain, so older writes stay visible.
void VulkanCommandList::synchronize(const vk::PipelineStageFlags2 stages, const vk::PipelineStageFlags2 readStages) {
    const vk::PipelineStageFlags2 waiting = stages | readStages;
    if (concurrent) {
        const vk::PipelineStageFlags2 missing = waiting & ~concurrentWaiting;
        if (concurrentSource && missing)
            recordBarrier(pool.commands, concurrentSource, missing, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
        concurrentWaiting |= missing;
        unsynchronized |= stages;
        return;
    }
    if (unsynchronized)
        recordBarrier(pool.commands, unsynchronized, waiting, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
    unsynchronized = stages;
    copyReads.clear();
    copyWrites.clear();
}

//! Copies that touch disjoint memory need no barrier between them, so many uploads in one list run together.
void VulkanCommandList::synchronizeCopy(const std::span<const CopyRange> reads, const std::span<const CopyRange> writes) {
    const bool independent =
        !concurrent && unsynchronized == copyStages && !overlaps(writes, copyReads) && !overlaps(writes, copyWrites) && !overlaps(reads, copyWrites);
    if (!independent)
        synchronize(copyStages);
    copyReads.insert(copyReads.end(), reads.begin(), reads.end());
    copyWrites.insert(copyWrites.end(), writes.begin(), writes.end());
}

//! The host reads downloads and host-visible memory after the list.
void VulkanCommandList::finish() {
    if (unsynchronized)
        recordBarrier(pool.commands, unsynchronized, vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
    unsynchronized = {};
}

void VulkanCommandList::beginConcurrent() {
    concurrentSource = unsynchronized;
    concurrentWaiting = {};
}

void VulkanCommandList::endConcurrent() {
    concurrentSource = {};
}

void VulkanCommandList::dispatch(const ComputePipeline& pipeline, const DispatchSize groups, const std::span<const std::byte> arguments) {
    synchronize(vk::PipelineStageFlagBits2::eComputeShader);
    bind(vk::PipelineBindPoint::eCompute, static_cast<const VulkanComputePipeline&>(pipeline).pipeline);
    pushRoot(arguments);
    pool.commands.dispatch(groups.x, groups.y, groups.z);
}

void VulkanCommandList::dispatchIndirect(const ComputePipeline& pipeline, const std::uint64_t groups, const std::span<const std::byte> arguments) {
    const std::optional<BufferRange> buffer = device->findBuffer(groups);
    if (!check(buffer.has_value(), ErrorCode::InvalidArgument, "the indirect arguments are not in a buffer of this device"))
        return;
    synchronize(vk::PipelineStageFlagBits2::eComputeShader, vk::PipelineStageFlagBits2::eDrawIndirect);
    bind(vk::PipelineBindPoint::eCompute, static_cast<const VulkanComputePipeline&>(pipeline).pipeline);
    pushRoot(arguments);
    pool.commands.dispatchIndirect(buffer->buffer, buffer->offset);
}

void VulkanCommandList::beginRender(const RenderTarget& target) {
    synchronize(renderStages, renderReadStages);
    const auto& color = static_cast<const VulkanImage&>(*target.color->getBackend());
    const vk::AttachmentLoadOp load = target.clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
    vk::RenderingAttachmentInfo colorAttachment =
        vk::RenderingAttachmentInfo()
            .setImageView(*color.view)
            .setImageLayout(vk::ImageLayout::eGeneral)
            .setLoadOp(load)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setClearValue(vk::ClearColorValue(std::array{target.clearColor.x, target.clearColor.y, target.clearColor.z, target.clearColor.w}));
    if (target.resolve)
        colorAttachment.setResolveMode(vk::ResolveModeFlagBits::eAverage)
            .setResolveImageView(*static_cast<const VulkanImage&>(*target.resolve->getBackend()).view)
            .setResolveImageLayout(vk::ImageLayout::eGeneral);
    vk::RenderingInfo rendering =
        vk::RenderingInfo().setRenderArea({{0, 0}, {color.width, color.height}}).setLayerCount(1).setColorAttachments(colorAttachment);
    vk::RenderingAttachmentInfo depthAttachment;
    if (target.depth) {
        depthAttachment.setImageView(*static_cast<const VulkanImage&>(*target.depth->getBackend()).view)
            .setImageLayout(vk::ImageLayout::eGeneral)
            .setLoadOp(load)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setClearValue(vk::ClearDepthStencilValue(target.clearDepth, target.clearStencil));
        rendering.setPDepthAttachment(&depthAttachment);
        if (getFormatInfo(target.depth->getFormat()).stencil)
            rendering.setPStencilAttachment(&depthAttachment);
    }
    pool.commands.beginRendering(rendering);
    //! The viewport has a negative height, so clip space Y points up as in WebGPU, D3D and Metal, and one shader draws the same image on each backend.
    pool.commands.setViewport(0, vk::Viewport(0, static_cast<float>(color.height), static_cast<float>(color.width), -static_cast<float>(color.height), 0, 1));
    pool.commands.setScissor(0, vk::Rect2D({0, 0}, {color.width, color.height}));
    pool.commands.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, 0);
}

void VulkanCommandList::endRender() {
    pool.commands.endRendering();
}

void VulkanCommandList::setScissor(const Rect& rect) {
    pool.commands.setScissor(0, vk::Rect2D({rect.x, rect.y}, {rect.width, rect.height}));
}

void VulkanCommandList::setStencilReference(const std::uint32_t reference) {
    pool.commands.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, reference);
}

void VulkanCommandList::draw(const GraphicsPipeline& pipeline, const std::uint32_t vertexCount, const std::uint32_t instanceCount,
                             const std::span<const std::byte> arguments) {
    bind(vk::PipelineBindPoint::eGraphics, static_cast<const VulkanGraphicsPipeline&>(pipeline).pipeline);
    pushRoot(arguments);
    pool.commands.draw(vertexCount, instanceCount, 0, 0);
}

void VulkanCommandList::drawIndexed(const GraphicsPipeline& pipeline, const std::uint64_t indices, const std::uint32_t indexCount,
                                    const std::uint32_t instanceCount, const std::span<const std::byte> arguments) {
    if (!bindIndices(indices))
        return;
    bind(vk::PipelineBindPoint::eGraphics, static_cast<const VulkanGraphicsPipeline&>(pipeline).pipeline);
    pushRoot(arguments);
    pool.commands.drawIndexed(indexCount, instanceCount, 0, 0, 0);
}

void VulkanCommandList::drawIndirect(const GraphicsPipeline& pipeline, const IndirectDraw& draw, const std::span<const std::byte> arguments) {
    const std::optional<BufferRange> commands = device->findBuffer(draw.commands);
    const std::optional<BufferRange> count = draw.count ? device->findBuffer(draw.count) : std::nullopt;
    if (!check(commands.has_value() && (count.has_value() || draw.count == 0), ErrorCode::InvalidArgument,
               "the indirect draws or their count are not in a buffer of this device")
        || !bindIndices(draw.indices))
        return;
    bind(vk::PipelineBindPoint::eGraphics, static_cast<const VulkanGraphicsPipeline&>(pipeline).pipeline);
    pushRoot(arguments);
    if (draw.indices && count)
        return pool.commands.drawIndexedIndirectCount(commands->buffer, commands->offset, count->buffer, count->offset, draw.maxDraws, sizeof(DrawIndexedArgs));
    if (draw.indices)
        return pool.commands.drawIndexedIndirect(commands->buffer, commands->offset, draw.maxDraws, sizeof(DrawIndexedArgs));
    if (count)
        return pool.commands.drawIndirectCount(commands->buffer, commands->offset, count->buffer, count->offset, draw.maxDraws, sizeof(DrawArgs));
    pool.commands.drawIndirect(commands->buffer, commands->offset, draw.maxDraws, sizeof(DrawArgs));
}

void VulkanCommandList::traceRays(const VulkanRayTracingPipeline& tracing, const Shader& raygen, const DispatchSize size,
                                  const std::span<const std::byte> arguments) {
    const auto found = std::ranges::find_if(tracing.raygenRegions, [&raygen](const auto& region) {
        return region.first.getSource() == raygen.getSource() && region.first.entryPoint == raygen.entryPoint;
    });
    if (!check(found != tracing.raygenRegions.end(), ErrorCode::InvalidArgument, "the shader is not a ray generation shader of the pipeline"))
        return;
    synchronize(vk::PipelineStageFlagBits2::eRayTracingShaderKHR);
    bind(vk::PipelineBindPoint::eRayTracingKHR, tracing.linked->pipeline);
    pushRoot(arguments);
    pool.commands.traceRaysKHR(found->second, tracing.missRegion, tracing.hitRegion, tracing.callableRegion, size.x, size.y, size.z);
}

void VulkanCommandList::copy(const Image& source, const Image& destination, const Rect& sourceRegion) {
    const auto& from = static_cast<const VulkanImage&>(source);
    const auto& to = static_cast<const VulkanImage&>(destination);
    const vk::ImageSubresourceLayers layers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
    const vk::Offset3D origin{sourceRegion.x, sourceRegion.y, 0};
    synchronizeCopy(std::array{getImageRange(from.image)}, std::array{getImageRange(to.image)});
    if (sourceRegion.width == to.width && sourceRegion.height == to.height && from.format == to.format) {
        pool.commands.copyImage(from.image, vk::ImageLayout::eGeneral, to.image, vk::ImageLayout::eGeneral,
                                vk::ImageCopy(layers, origin, layers, {}, {to.width, to.height, 1}));
        return;
    }
    const vk::Offset3D end{sourceRegion.x + static_cast<std::int32_t>(sourceRegion.width), sourceRegion.y + static_cast<std::int32_t>(sourceRegion.height), 1};
    const vk::ImageBlit region(layers, {origin, end}, layers,
                               {vk::Offset3D{}, vk::Offset3D{static_cast<std::int32_t>(to.width), static_cast<std::int32_t>(to.height), 1}});
    pool.commands.blitImage(from.image, vk::ImageLayout::eGeneral, to.image, vk::ImageLayout::eGeneral, region, vk::Filter::eLinear);
}

void VulkanCommandList::copy(const std::uint64_t address, const Image& destination, const std::uint32_t level) {
    const auto& to = static_cast<const VulkanImage&>(destination);
    const FormatInfo info = getFormatInfo(to.publicFormat);
    const std::uint32_t width = std::max(to.width >> level, 1u);
    const std::uint32_t height = std::max(to.height >> level, 1u);
    const std::size_t bytes = getCopyLevelByteSize(to.publicFormat, width, height);
    const std::optional<BufferRange> buffer = device->findBuffer(address);
    if (!check(buffer.has_value() && bytes <= buffer->size, ErrorCode::InvalidArgument, "the copy range is not inside a buffer of this device"))
        return;
    synchronizeCopy(std::array{CopyRange{0, address, address + bytes}}, std::array{getImageRange(to.image)});
    const std::uint32_t rowLength = getCopyRowPitch(to.publicFormat, width) / info.blockBytes * info.blockExtent;
    pool.commands.copyBufferToImage(
        buffer->buffer, to.image, vk::ImageLayout::eGeneral,
        vk::BufferImageCopy(buffer->offset, rowLength, 0, vk::ImageSubresourceLayers(to.aspect, level, 0, 1), vk::Offset3D{}, vk::Extent3D{width, height, 1}));
}

void VulkanCommandList::upload(const std::uint64_t address, const std::span<const std::byte> data) {
    const std::optional<BufferRange> buffer = device->findBuffer(address);
    if (!check(buffer.has_value() && data.size() <= buffer->size, ErrorCode::InvalidArgument, "the upload range is not inside a buffer of this device"))
        return;
    std::expected<Placement, Error> staged = place(data);
    if (!staged)
        return static_cast<void>(check(false, staged.error().code, staged.error().message));
    synchronizeCopy({}, std::array{CopyRange{0, address, address + data.size()}});
    pool.commands.copyBuffer(staged->buffer, buffer->buffer, vk::BufferCopy(staged->offset, buffer->offset, data.size()));
}

void VulkanCommandList::download(const std::uint64_t address, Readback& readback, const std::size_t bytes) {
    auto& target = static_cast<VulkanReadback&>(readback);
    const std::optional<BufferRange> buffer = device->findBuffer(address);
    if (!check(target.device == device, ErrorCode::InvalidArgument, "the readback belongs to another device")
        || !check(buffer.has_value() && bytes <= buffer->size, ErrorCode::InvalidArgument, "the download range is not inside a buffer of this device"))
        return;
    synchronizeCopy(std::array{CopyRange{0, address, address + bytes}}, std::array{CopyRange{0, target.memory.address, target.memory.address + bytes}});
    pool.commands.copyBuffer(buffer->buffer, target.memory.buffer, vk::BufferCopy(buffer->offset, target.memory.offset, bytes));
    target.token = {};
    readbacks.push_back(target.shared_from_this());
}

void VulkanCommandList::beginTimestamp(TimestampQuery& query) {
    auto& timestamp = static_cast<VulkanTimestampQuery&>(query);
    if (!check(device->getQueue(queue).timestampValidBits != 0, ErrorCode::Unsupported, "the queue of the command list has no timestamps"))
        return;
    timestamp.queue = queue;
    pool.commands.resetQueryPool(timestamp.pool, 0, 2);
    pool.commands.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, timestamp.pool, 0);
}

void VulkanCommandList::endTimestamp(TimestampQuery& query) {
    pool.commands.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, static_cast<VulkanTimestampQuery&>(query).pool, 1);
}

void VulkanCommandList::pushLabel(const std::string_view name) {
    if (!device->debugLabels)
        return;
    const std::string text(name);
    pool.commands.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT(text.c_str()));
}

void VulkanCommandList::popLabel() {
    if (device->debugLabels)
        pool.commands.endDebugUtilsLabelEXT();
}

}

}
