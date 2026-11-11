#include <algorithm>
#include <cstring>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Device/Device.h"
#include "NoorRHI/WebGpu/WebGpu.h"

namespace NoorRHI {

CommandList::CommandList(Device& device, const Queue queue) {
    if (!device)
        return;
    auto list = std::make_unique<Backend::WebGpuCommandList>();
    list->device = static_cast<Backend::WebGpuDevice&>(*device.getBackend()).shared_from_this();
    list->id = list->device->openList();
    list->queue = queue;
    backend = std::move(list);
}

namespace Backend {

WebGpuCommandList::~WebGpuCommandList() {
    if (device)
        device->closeList(id);
}

namespace {

bool isFilterable(const WebGpuDevice& device, const ImageFormat format) {
    const FormatInfo info = getFormatInfo(format);
    if (info.integer || info.depth)
        return false;
    const bool float32 = format == ImageFormat::R32Float || format == ImageFormat::Rg32Float || format == ImageFormat::Rgba32Float;
    return !float32 || device.float32Filterable;
}

std::shared_ptr<WebGpuImage> getImage(const Image& image) {
    return const_cast<WebGpuImage&>(static_cast<const WebGpuImage&>(image)).shared_from_this();
}

bool isInHeap(const WebGpuDevice& device, const std::uint64_t address) {
    return address % 4 == 0 && (address >> 32) < device.heaps.size() && (address >= heapReservedBytes || (address >> 32) != 0);
}

bool isPlain(const std::uint64_t address) {
    return address % 4 == 0 && address >= heapReservedBytes && address <= addressOffsetMask;
}

std::uint32_t getWord(const std::uint64_t address) {
    return static_cast<std::uint32_t>((address & addressOffsetMask) / 4);
}

}

std::uint32_t WebGpuCommandList::place(const std::span<const std::byte> data) {
    if (data.empty() || !check(data.size() <= argumentWindowBytes, ErrorCode::InvalidArgument, "WebGPU arguments are at most 4096 bytes"))
        return 0;
    arguments.resize(alignUp(arguments.size(), device->limits.minStorageBufferOffsetAlignment));
    const auto offset = static_cast<std::uint32_t>(arguments.size());
    arguments.insert(arguments.end(), data.begin(), data.end());
    return offset;
}

void WebGpuCommandList::dispatch(const ComputePipeline& pipeline, const DispatchSize groups, const std::span<const std::byte> data) {
    commands.emplace_back(DispatchCommand{static_cast<const WebGpuComputePipeline&>(pipeline).shared_from_this(), groups, 0, place(data)});
}

void WebGpuCommandList::dispatchIndirect(const ComputePipeline& pipeline, const std::uint64_t groups, const std::span<const std::byte> data) {
    if (!check(groups % 4 == 0 && groups >= heapReservedBytes && groups <= addressOffsetMask, ErrorCode::InvalidArgument,
               "the indirect arguments are not in a buffer of this device"))
        return;
    commands.emplace_back(DispatchCommand{static_cast<const WebGpuComputePipeline&>(pipeline).shared_from_this(), {}, groups, place(data)});
    ++indirectDispatches;
}

void WebGpuCommandList::beginConcurrent() {
    commands.emplace_back(ConcurrentCommand{true});
}

void WebGpuCommandList::endConcurrent() {
    commands.emplace_back(ConcurrentCommand{false});
}

void WebGpuCommandList::beginRender(const RenderTarget& target) {
    BeginRenderCommand begin;
    begin.color = getImage(*target.color->getBackend());
    if (target.depth)
        begin.depth = getImage(*target.depth->getBackend());
    if (target.resolve)
        begin.resolve = getImage(*target.resolve->getBackend());
    begin.clear = target.clear;
    begin.clearColor = target.clearColor;
    begin.clearDepth = target.clearDepth;
    begin.clearStencil = target.clearStencil;
    renderBegin = commands.size();
    commands.emplace_back(std::move(begin));
}

void WebGpuCommandList::endRender() {
    commands.emplace_back(EndRenderCommand{});
}

void WebGpuCommandList::setScissor(const Rect& rect) {
    commands.emplace_back(ScissorCommand{rect});
}

void WebGpuCommandList::setStencilReference(const std::uint32_t reference) {
    commands.emplace_back(StencilReferenceCommand{reference});
}

void WebGpuCommandList::draw(const GraphicsPipeline& pipeline, const std::uint32_t vertexCount, const std::uint32_t instanceCount,
                             const std::span<const std::byte> data) {
    commands.emplace_back(DrawCommand{.pipeline = static_cast<const WebGpuGraphicsPipeline&>(pipeline).shared_from_this(),
                                      .elementCount = vertexCount,
                                      .instanceCount = instanceCount,
                                      .root = place(data)});
}

void WebGpuCommandList::drawIndexed(const GraphicsPipeline& pipeline, const std::uint64_t indices, const std::uint32_t indexCount,
                                    const std::uint32_t instanceCount, const std::span<const std::byte> data) {
    if (!check(isPlain(indices), ErrorCode::InvalidArgument, "on WebGPU the indices of a draw are in a plain buffer of this device"))
        return;
    commands.emplace_back(DrawCommand{.pipeline = static_cast<const WebGpuGraphicsPipeline&>(pipeline).shared_from_this(),
                                      .elementCount = indexCount,
                                      .instanceCount = instanceCount,
                                      .indices = indices,
                                      .root = place(data)});
}

//! A render pass reads draws only from the plain heap, so a draw with a count copies its draws there before the pass and zeroes the instances past the count.
void WebGpuCommandList::drawIndirect(const GraphicsPipeline& pipeline, const IndirectDraw& draw, const std::span<const std::byte> data) {
    constexpr std::uint32_t maxCountedDraws = 65535u * 64u;
    if (!check(draw.indices == 0 || isPlain(draw.indices), ErrorCode::InvalidArgument, "on WebGPU the indices of a draw are in a plain buffer of this device"))
        return;
    DrawCommand command{.pipeline = static_cast<const WebGpuGraphicsPipeline&>(pipeline).shared_from_this(),
                        .indices = draw.indices,
                        .indirect = draw.commands,
                        .drawCount = draw.maxDraws};
    if (draw.count == 0) {
        if (!check(isPlain(draw.commands), ErrorCode::InvalidArgument, "on WebGPU indirect draws without a count are in a plain buffer of this device"))
            return;
        command.root = place(data);
        commands.emplace_back(std::move(command));
        return;
    }
    if (!check(isInHeap(*device, draw.commands) && isInHeap(*device, draw.count), ErrorCode::InvalidArgument,
               "the indirect draws or their count are not in a buffer of this device")
        || !check(draw.maxDraws <= maxCountedDraws, ErrorCode::InvalidArgument, "a WebGPU draw with a count has at most 4194240 draws"))
        return;
    const bool indexed = draw.indices != 0;
    const std::uint32_t flags =
        static_cast<std::uint32_t>(indexed) << 31 | static_cast<std::uint32_t>(draw.commands >> 32) << 30 | static_cast<std::uint32_t>(draw.count >> 32) << 29;
    const std::array job{getWord(draw.commands), getWord(draw.count), getWord(drawScratchBytes), draw.maxDraws | flags};
    std::get<BeginRenderCommand>(commands[renderBegin]).counts.push_back({place(std::as_bytes(std::span(job))), draw.maxDraws});
    command.indirect = drawScratchBytes;
    command.scratch = true;
    command.root = place(data);
    commands.emplace_back(std::move(command));
    drawScratchBytes += std::uint64_t{draw.maxDraws} * (indexed ? sizeof(DrawIndexedArgs) : sizeof(DrawArgs));
}

void WebGpuCommandList::copy(const Image& source, const Image& destination, const Rect& sourceRegion) {
    std::shared_ptr<WebGpuImage> from = getImage(source);
    std::shared_ptr<WebGpuImage> to = getImage(destination);
    const bool direct = sourceRegion.width == to->width && sourceRegion.height == to->height && from->format == to->format;
    if (!direct
        && !check(isFilterable(*device, from->format) && device->isRenderable(to->format), ErrorCode::Unsupported,
                  "a scaled WebGPU copy needs a filterable source format and a renderable destination format"))
        return;
    commands.emplace_back(CopyCommand{std::move(from), std::move(to), sourceRegion});
}

void WebGpuCommandList::patch(const std::uint64_t address, const std::span<const std::byte> data) {
    for (std::size_t index = 0; index < data.size();) {
        const std::uint64_t byte = address + index;
        BytePatch bytes{.word = static_cast<std::uint32_t>(byte / 4)};
        for (std::uint64_t lane = byte % 4; lane < 4 && index < data.size(); ++lane, ++index) {
            bytes.mask |= 0xffu << (lane * 8);
            bytes.value |= std::to_integer<std::uint32_t>(data[index]) << (lane * 8);
        }
        commands.emplace_back(PatchCommand{place(std::as_bytes(std::span(&bytes, 1)))});
    }
}

void WebGpuCommandList::upload(const std::uint64_t address, const std::span<const std::byte> data) {
    if (!check(address >= heapReservedBytes && (address >> 32) < device->heaps.size(), ErrorCode::InvalidArgument,
               "the upload range is not inside a buffer of this device"))
        return;
    const std::uint64_t head = std::min<std::uint64_t>((4 - address % 4) % 4, data.size());
    const std::uint64_t body = (data.size() - head) & ~std::uint64_t{3};
    patch(address, data.first(head));
    if (body != 0) {
        commands.emplace_back(UploadCommand{address + head, uploads.size(), body});
        const std::span<const std::byte> aligned = data.subspan(head, body);
        uploads.insert(uploads.end(), aligned.begin(), aligned.end());
    }
    patch(address + head + body, data.subspan(head + body));
}

void WebGpuCommandList::download(const std::uint64_t address, Readback& readback, const std::size_t bytes) {
    auto& target = static_cast<WebGpuReadback&>(readback);
    const std::uint64_t head = address % 4;
    if (!check(target.device == device, ErrorCode::InvalidArgument, "the readback belongs to another device")
        || !check(isInHeap(*device, address - head), ErrorCode::InvalidArgument, "the download range is not inside a buffer of this device"))
        return;
    commands.emplace_back(DownloadCommand{target.shared_from_this(), address - head, alignUp(head + bytes, 4)});
    target.head = head;
    target.recorded = true;
    readbacks.push_back(target.shared_from_this());
}

void WebGpuCommandList::beginTimestamp(TimestampQuery& query) {
    auto& timestamp = static_cast<WebGpuTimestampQuery&>(query);
    if (!timestamp.querySet)
        return;
    if (!rendering)
        return static_cast<void>(commands.emplace_back(TimestampCommand{timestamp.shared_from_this(), false}));
    auto& begin = std::get<BeginRenderCommand>(commands[renderBegin]);
    begin.beginQuery = timestamp.shared_from_this();
}

void WebGpuCommandList::endTimestamp(TimestampQuery& query) {
    auto& timestamp = static_cast<WebGpuTimestampQuery&>(query);
    if (!timestamp.querySet)
        return;
    if (!rendering)
        return static_cast<void>(commands.emplace_back(TimestampCommand{timestamp.shared_from_this(), true}));
    auto& begin = std::get<BeginRenderCommand>(commands[renderBegin]);
    if (!check(!begin.beginQuery || begin.beginQuery.get() == &timestamp, ErrorCode::Unsupported,
               "a WebGPU render pass writes the timestamps of one query only"))
        return;
    begin.endQuery = timestamp.shared_from_this();
}

void WebGpuCommandList::pushLabel(const std::string_view name) {
    commands.emplace_back(LabelCommand{std::string(name), false});
}

void WebGpuCommandList::popLabel() {
    commands.emplace_back(LabelCommand{{}, true});
}

namespace {

struct Replayer {
    Replayer(WebGpuDevice& owner, const wgpu::CommandEncoder& commands, std::vector<std::shared_ptr<WebGpuTimestampQuery>>& queries)
        : device(owner), encoder(commands), ended(queries) {}

    WebGpuDevice& device;
    const wgpu::CommandEncoder& encoder;
    std::vector<std::shared_ptr<WebGpuTimestampQuery>>& ended;
    wgpu::ComputePassEncoder compute;
    wgpu::RenderPassEncoder render;
    const BeginRenderCommand* pass = nullptr;
    std::uint64_t indirectSlot = 0;
    std::uint32_t encoderLabels = 0;
    std::uint32_t passLabels = 0;
    bool dispatched = false;
    bool written = false;
    bool concurrent = false;
    const void* boundPipeline = nullptr;
    std::uint32_t boundRoot = ~0u;
    std::uint64_t boundIndices = 0;
    wgpu::BindGroup countGroup;

    template <class Pass, class Pipeline>
    void bind(Pass& pass, const Pipeline& pipeline, const std::uint32_t offset) {
        if (boundPipeline != pipeline.Get()) {
            boundPipeline = pipeline.Get();
            pass.SetPipeline(pipeline);
        }
        bindArguments(pass, offset);
    }

    template <class Pass>
    void bindArguments(Pass& pass, const std::uint32_t offset) {
        if (boundRoot == offset)
            return;
        boundRoot = offset;
        pass.SetBindGroup(argumentGroup, device.argumentGroup, 1, &offset);
    }

    void forgetBindings() {
        boundPipeline = nullptr;
        boundRoot = ~0u;
    }

    void beginCompute() {
        if (compute)
            return;
        compute = encoder.BeginComputePass();
        compute.SetBindGroup(0, device.computeGroup);
        forgetBindings();
    }

    void endCompute() {
        if (!compute)
            return;
        compute.End();
        compute = nullptr;
    }

    void resolve(const std::shared_ptr<WebGpuTimestampQuery>& query) {
        if (query->mapping || std::ranges::contains(ended, query))
            return;
        encoder.ResolveQuerySet(query->querySet, 0, 2, query->resolve, 0);
        encoder.CopyBufferToBuffer(query->resolve, 0, query->readback, 0, 16);
        ended.push_back(query);
    }

    //! WebGPU orders passes and dispatches itself; only the sampled copies of written images wait for a command that could sample them.
    void settle() {
        if (concurrent || (!dispatched && !written))
            return;
        if (device.hasStaleShadows(dispatched)) {
            endCompute();
            device.refreshShadows(encoder, dispatched);
        }
        dispatched = false;
        written = false;
    }

    std::expected<void, Error> operator()(const DispatchCommand& command) {
        settle();
        if (command.indirect != 0) {
            endCompute();
            encoder.CopyBufferToBuffer(device.heaps[0].buffer, command.indirect, device.indirect, indirectSlot, 12);
        }
        beginCompute();
        bind(compute, command.pipeline->pipeline, command.root);
        if (command.indirect != 0) {
            compute.DispatchWorkgroupsIndirect(device.indirect, indirectSlot);
            indirectSlot += indirectSlotBytes;
        } else {
            compute.DispatchWorkgroups(command.groups.x, command.groups.y, command.groups.z);
        }
        dispatched = true;
        return {};
    }

    std::expected<void, Error> operator()(const ConcurrentCommand& command) {
        if (command.begin)
            settle();
        concurrent = command.begin;
        return {};
    }

    void countDraws(const std::span<const CountJob> jobs) {
        if (!countGroup) {
            const std::array entries{wgpu::BindGroupEntry{.binding = 0, .buffer = device.heaps[0].buffer},
                                     wgpu::BindGroupEntry{.binding = 1, .buffer = device.heaps[1].buffer},
                                     wgpu::BindGroupEntry{.binding = 2, .buffer = device.drawScratch}};
            const wgpu::BindGroupDescriptor descriptor{.layout = device.countGroupLayout, .entryCount = entries.size(), .entries = entries.data()};
            countGroup = device.device.CreateBindGroup(&descriptor);
        }
        const wgpu::ComputePassEncoder counter = encoder.BeginComputePass();
        counter.SetPipeline(device.countPipeline);
        counter.SetBindGroup(0, countGroup);
        for (const CountJob& job : jobs) {
            counter.SetBindGroup(argumentGroup, device.argumentGroup, 1, &job.root);
            counter.DispatchWorkgroups((job.maxDraws + 63) / 64);
        }
        counter.End();
    }

    std::expected<void, Error> operator()(const BeginRenderCommand& command) {
        settle();
        endCompute();
        if (!command.counts.empty())
            countDraws(command.counts);
        const TextureRegion color = device.getPrimary(*command.color);
        if (command.depth && (color.texture.GetWidth() != command.depth->width || color.texture.GetHeight() != command.depth->height))
            return fail(ErrorCode::Unsupported, "on WebGPU a storage color image cannot share a render pass with a depth image");
        const wgpu::LoadOp load = command.clear ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load;
        wgpu::RenderPassColorAttachment colorAttachment{
            .view = command.color->createView(color),
            .loadOp = load,
            .storeOp = wgpu::StoreOp::Store,
            .clearValue = {command.clearColor.x, command.clearColor.y, command.clearColor.z, command.clearColor.w},
        };
        if (command.resolve)
            colorAttachment.resolveTarget = command.resolve->createView(device.getPrimary(*command.resolve));
        wgpu::RenderPassDepthStencilAttachment depthAttachment;
        if (command.depth) {
            depthAttachment.view = command.depth->createView(device.getPrimary(*command.depth));
            depthAttachment.depthLoadOp = load;
            depthAttachment.depthStoreOp = wgpu::StoreOp::Store;
            depthAttachment.depthClearValue = command.clearDepth;
        }
        if (command.depth && getFormatInfo(command.depth->format).stencil) {
            depthAttachment.stencilLoadOp = load;
            depthAttachment.stencilStoreOp = wgpu::StoreOp::Store;
            depthAttachment.stencilClearValue = command.clearStencil;
        }
        const std::shared_ptr<WebGpuTimestampQuery>& query = command.beginQuery ? command.beginQuery : command.endQuery;
        wgpu::PassTimestampWrites timestamps;
        if (query) {
            timestamps.querySet = query->querySet;
            timestamps.beginningOfPassWriteIndex = command.beginQuery ? 0 : wgpu::kQuerySetIndexUndefined;
            timestamps.endOfPassWriteIndex = command.endQuery ? 1 : wgpu::kQuerySetIndexUndefined;
        }
        const wgpu::RenderPassDescriptor descriptor{.colorAttachmentCount = 1,
                                                    .colorAttachments = &colorAttachment,
                                                    .depthStencilAttachment = command.depth ? &depthAttachment : nullptr,
                                                    .timestampWrites = query ? &timestamps : nullptr};
        render = encoder.BeginRenderPass(&descriptor);
        render.SetBindGroup(0, device.renderGroup);
        forgetBindings();
        boundIndices = 0;
        render.SetViewport(0, 0, static_cast<float>(command.color->width), static_cast<float>(command.color->height), 0, 1);
        render.SetScissorRect(0, 0, command.color->width, command.color->height);
        pass = &command;
        return {};
    }

    std::expected<void, Error> operator()(const EndRenderCommand&) {
        for (; passLabels != 0; --passLabels)
            render.PopDebugGroup();
        render.End();
        render = nullptr;
        pass->color->shadowStale = true;
        if (pass->depth)
            pass->depth->shadowStale = true;
        if (pass->resolve)
            pass->resolve->shadowStale = true;
        written = true;
        if (pass->endQuery)
            resolve(pass->endQuery);
        pass = nullptr;
        return {};
    }

    std::expected<void, Error> operator()(const ScissorCommand& command) {
        render.SetScissorRect(static_cast<std::uint32_t>(command.rect.x), static_cast<std::uint32_t>(command.rect.y), command.rect.width, command.rect.height);
        return {};
    }

    std::expected<void, Error> operator()(const StencilReferenceCommand& command) {
        render.SetStencilReference(command.reference);
        return {};
    }

    std::expected<void, Error> operator()(const DrawCommand& command) {
        bind(render, command.pipeline->pipeline, command.root);
        if (command.indices != 0 && command.indices != boundIndices) {
            render.SetIndexBuffer(device.heaps[0].buffer, wgpu::IndexFormat::Uint32, command.indices);
            boundIndices = command.indices;
        }
        if (command.drawCount == 0 && command.indices != 0)
            render.DrawIndexed(command.elementCount, command.instanceCount);
        else if (command.drawCount == 0)
            render.Draw(command.elementCount, command.instanceCount);
        const wgpu::Buffer& buffer = command.scratch ? device.drawScratch : device.heaps[0].buffer;
        if (command.drawCount != 0 && device.multiDrawIndirect && command.indices != 0)
            render.MultiDrawIndexedIndirect(buffer, command.indirect, command.drawCount);
        else if (command.drawCount != 0 && device.multiDrawIndirect)
            render.MultiDrawIndirect(buffer, command.indirect, command.drawCount);
        if (device.multiDrawIndirect)
            return {};
        const std::uint64_t stride = command.indices != 0 ? sizeof(DrawIndexedArgs) : sizeof(DrawArgs);
        for (std::uint32_t index = 0; index < command.drawCount; ++index)
            if (command.indices != 0)
                render.DrawIndexedIndirect(buffer, command.indirect + index * stride);
            else
                render.DrawIndirect(buffer, command.indirect + index * stride);
        return {};
    }

    void copyTexels(const TextureRegion& from, const TextureRegion& to, const std::uint32_t width, const std::uint32_t height) const {
        const wgpu::TexelCopyTextureInfo source{.texture = from.texture, .origin = {from.x, from.y, from.layer}};
        const wgpu::TexelCopyTextureInfo destination{.texture = to.texture, .origin = {to.x, to.y, to.layer}};
        const wgpu::Extent3D size{width, height, 1};
        encoder.CopyTextureToTexture(&source, &destination, &size);
    }

    std::expected<void, Error> operator()(const CopyCommand& command) {
        endCompute();
        const WebGpuImage& from = *command.source;
        WebGpuImage& to = *command.destination;
        const Rect& region = command.region;
        TextureRegion source = device.getPrimary(from);
        source.x += static_cast<std::uint32_t>(region.x);
        source.y += static_cast<std::uint32_t>(region.y);
        const TextureRegion destination = device.getPrimary(to);
        to.shadowStale = true;
        written = true;
        if (source.texture.Get() == destination.texture.Get() && source.layer == destination.layer) {
            const wgpu::TextureDescriptor descriptor{.usage = wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding,
                                                     .size = {region.width, region.height, 1},
                                                     .format = from.nativeFormat};
            const TextureRegion staged{device.device.CreateTexture(&descriptor)};
            copyTexels(source, staged, region.width, region.height);
            source = staged;
        }
        if (region.width == to.width && region.height == to.height && from.format == to.format) {
            copyTexels(source, destination, to.width, to.height);
            return {};
        }
        std::expected<wgpu::RenderPipeline, Error> pipeline = device.getBlitPipeline(to.nativeFormat);
        if (!pipeline)
            return std::unexpected(std::move(pipeline.error()));
        const auto sourceWidth = static_cast<float>(source.texture.GetWidth());
        const auto sourceHeight = static_cast<float>(source.texture.GetHeight());
        const std::array transform{static_cast<float>(source.x) / sourceWidth, static_cast<float>(source.y) / sourceHeight,
                                   static_cast<float>(region.width) / sourceWidth, static_cast<float>(region.height) / sourceHeight};
        const wgpu::BufferDescriptor transformDescriptor{.usage = wgpu::BufferUsage::Uniform, .size = sizeof(transform), .mappedAtCreation = true};
        const wgpu::Buffer transformBuffer = device.device.CreateBuffer(&transformDescriptor);
        std::memcpy(transformBuffer.GetMappedRange(), transform.data(), sizeof(transform));
        transformBuffer.Unmap();
        const std::array entries{wgpu::BindGroupEntry{.binding = 0, .textureView = from.createView(source)},
                                 wgpu::BindGroupEntry{.binding = 1, .sampler = device.blitSampler},
                                 wgpu::BindGroupEntry{.binding = 2, .buffer = transformBuffer}};
        const wgpu::BindGroupDescriptor groupDescriptor{.layout = device.blitGroupLayout, .entryCount = entries.size(), .entries = entries.data()};
        const wgpu::BindGroup group = device.device.CreateBindGroup(&groupDescriptor);
        const wgpu::RenderPassColorAttachment attachment{.view = to.createView(destination), .loadOp = wgpu::LoadOp::Load, .storeOp = wgpu::StoreOp::Store};
        const wgpu::RenderPassDescriptor descriptor{.colorAttachmentCount = 1, .colorAttachments = &attachment};
        const wgpu::RenderPassEncoder blit = encoder.BeginRenderPass(&descriptor);
        blit.SetPipeline(*pipeline);
        blit.SetBindGroup(0, group);
        blit.SetViewport(static_cast<float>(destination.x), static_cast<float>(destination.y), static_cast<float>(to.width), static_cast<float>(to.height), 0,
                         1);
        blit.SetScissorRect(destination.x, destination.y, to.width, to.height);
        blit.Draw(3);
        blit.End();
        return {};
    }

    std::expected<void, Error> operator()(const UploadCommand& command) {
        endCompute();
        encoder.CopyBufferToBuffer(device.staging, command.stagingOffset, device.getBuffer(command.address), command.address & addressOffsetMask, command.size);
        return {};
    }

    std::expected<void, Error> operator()(const DownloadCommand& command) {
        endCompute();
        encoder.CopyBufferToBuffer(device.getBuffer(command.address), command.address & addressOffsetMask, command.readback->buffer, 0, command.size);
        return {};
    }

    std::expected<void, Error> operator()(const PatchCommand& command) {
        beginCompute();
        if (boundPipeline != device.patchPipeline.Get()) {
            boundPipeline = device.patchPipeline.Get();
            compute.SetPipeline(device.patchPipeline);
        }
        bindArguments(compute, command.root);
        compute.DispatchWorkgroups(1);
        return {};
    }

    std::expected<void, Error> operator()(const TimestampCommand& command) {
        endCompute();
        const wgpu::PassTimestampWrites timestamps{.querySet = command.query->querySet,
                                                   .beginningOfPassWriteIndex = command.end ? wgpu::kQuerySetIndexUndefined : 0,
                                                   .endOfPassWriteIndex = command.end ? 1 : wgpu::kQuerySetIndexUndefined};
        const wgpu::ComputePassDescriptor descriptor{.timestampWrites = &timestamps};
        encoder.BeginComputePass(&descriptor).End();
        if (command.end)
            resolve(command.query);
        return {};
    }

    std::expected<void, Error> operator()(const LabelCommand& command) {
        const wgpu::StringView name(command.name.data(), command.name.size());
        if (render && !command.pop) {
            render.PushDebugGroup(name);
            ++passLabels;
        } else if (render) {
            if (passLabels != 0) {
                render.PopDebugGroup();
                --passLabels;
            }
        } else if (!command.pop) {
            endCompute();
            encoder.PushDebugGroup(name);
            ++encoderLabels;
        } else if (encoderLabels != 0) {
            endCompute();
            encoder.PopDebugGroup();
            --encoderLabels;
        }
        return {};
    }

    void finish() {
        endCompute();
        for (; encoderLabels != 0; --encoderLabels)
            encoder.PopDebugGroup();
        device.refreshShadows(encoder, dispatched);
    }
};

}

std::expected<void, Error> WebGpuDevice::replay(const WebGpuCommandList& list, const wgpu::CommandEncoder& encoder,
                                                std::vector<std::shared_ptr<WebGpuTimestampQuery>>& ended) {
    updateBindGroups();
    Replayer replayer(*this, encoder, ended);
    for (const RecordedCommand& command : list.commands)
        if (std::expected<void, Error> replayed = std::visit(replayer, command); !replayed)
            return replayed;
    replayer.finish();
    return {};
}

}

}
