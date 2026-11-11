#include "NoorRHI/Commands/CommandList.h"

#include <algorithm>

#include "NoorRHI/Core/Backend.h"

namespace NoorRHI {

CommandList::CommandList(CommandList&& other) noexcept = default;
CommandList& CommandList::operator=(CommandList&& other) noexcept = default;
CommandList::~CommandList() = default;

namespace {

Backend::CommandList* getRecording(const std::unique_ptr<Backend::CommandList>& backend) {
    return backend && !backend->error ? backend.get() : nullptr;
}

bool matches(const Image& image, const Image& color) {
    return image.getWidth() == color.getWidth() && image.getHeight() == color.getHeight();
}

bool isAligned(const std::uint64_t address) {
    return address % 4 == 0;
}

bool checkDraw(Backend::CommandList& list, const GraphicsPipeline& pipeline) {
    if (!list.check(list.rendering, ErrorCode::InvalidState, "draws must be inside beginRender and endRender"))
        return false;
    const Backend::GraphicsPipeline& graphics = *pipeline.getBackend();
    if (!list.check(graphics.ready, ErrorCode::InvalidState, "the graphics pipeline is still compiling or failed to compile"))
        return false;
    return list.check(graphics.colorFormat == list.render.colorFormat && graphics.depthFormat == list.render.depthFormat
                          && graphics.samples == list.render.samples,
                      ErrorCode::InvalidArgument, "the color format, depth format and sample count of the pipeline are not those of the render target");
}

}

void CommandList::setError(const ErrorCode code, const std::string_view message) {
    if (backend)
        backend->check(false, code, message);
}

Queue CommandList::getQueue() const {
    return backend ? backend->queue : Queue::Graphics;
}

const std::optional<Error>& CommandList::getError() const {
    static const std::optional<Error> closed = Error{ErrorCode::InvalidState, "the command list has no open device"};
    return backend ? backend->error : closed;
}

void CommandList::dispatchBytes(const ComputePipeline& pipeline, const DispatchSize threads, const std::span<const std::byte> arguments) {
    Backend::CommandList* list = getRecording(backend);
    if (!list || !list->check(static_cast<bool>(pipeline), ErrorCode::InvalidArgument, "dispatch needs an open compute pipeline")
        || !list->check(!list->rendering, ErrorCode::InvalidState, "dispatch is not allowed inside beginRender")
        || !list->check(pipeline.getBackend()->ready, ErrorCode::InvalidState, "the compute pipeline is still compiling or failed to compile"))
        return;
    const DispatchSize group = pipeline.getBackend()->threadGroupSize;
    const auto groups = [](const std::uint32_t count, const std::uint32_t size) { return count / size + (count % size != 0); };
    list->dispatch(*pipeline.getBackend(), {groups(threads.x, group.x), groups(threads.y, group.y), groups(threads.z, group.z)}, arguments);
}

void CommandList::dispatchIndirectBytes(const ComputePipeline& pipeline, const Ptr<DispatchSize> groups, const std::span<const std::byte> arguments) {
    Backend::CommandList* list = getRecording(backend);
    if (!list || !list->check(pipeline && groups, ErrorCode::InvalidArgument, "dispatchIndirect needs an open compute pipeline and arguments")
        || !list->check(!list->rendering, ErrorCode::InvalidState, "dispatchIndirect is not allowed inside beginRender")
        || !list->check(pipeline.getBackend()->ready, ErrorCode::InvalidState, "the compute pipeline is still compiling or failed to compile"))
        return;
    list->dispatchIndirect(*pipeline.getBackend(), groups.getAddress(), arguments);
}

void CommandList::beginConcurrent() {
    Backend::CommandList* list = getRecording(backend);
    if (!list || !list->check(!list->concurrent && !list->rendering, ErrorCode::InvalidState, "beginConcurrent needs no active beginConcurrent or beginRender"))
        return;
    list->concurrent = true;
    list->beginConcurrent();
}

void CommandList::endConcurrent() {
    Backend::CommandList* list = getRecording(backend);
    if (!list
        || !list->check(list->concurrent && !list->rendering, ErrorCode::InvalidState, "endConcurrent needs an active beginConcurrent outside beginRender"))
        return;
    list->concurrent = false;
    list->endConcurrent();
}

void CommandList::beginRender(const RenderTarget& target) {
    Backend::CommandList* list = getRecording(backend);
    if (!list || !list->check(!list->rendering, ErrorCode::InvalidState, "beginRender is already active")
        || !list->check(list->queue == Queue::Graphics, ErrorCode::InvalidState, "beginRender needs a command list of the graphics queue")
        || !list->check(target.color && *target.color && hasUsage(target.color->getUsage(), ImageUsage::ColorAttachment), ErrorCode::InvalidArgument,
                        "the render target needs a color image with ColorAttachment usage"))
        return;
    const Image& color = *target.color;
    if (target.depth
        && !list->check(*target.depth && hasUsage(target.depth->getUsage(), ImageUsage::DepthAttachment) && matches(*target.depth, color)
                            && target.depth->getSamples() == color.getSamples(),
                        ErrorCode::InvalidArgument, "the depth image needs DepthAttachment usage and the size and sample count of the color image"))
        return;
    if (target.resolve
        && !list->check(*target.resolve && color.getSamples() > 1 && target.resolve->getSamples() == 1 && matches(*target.resolve, color)
                            && target.resolve->getFormat() == color.getFormat() && hasUsage(target.resolve->getUsage(), ImageUsage::ColorAttachment),
                        ErrorCode::InvalidArgument,
                        "the resolve image needs a multisampled color image and a single-sampled ColorAttachment image of the same size and format"))
        return;
    list->render = {color.getFormat(), target.depth ? target.depth->getFormat() : ImageFormat::Undefined, color.getSamples(), color.getWidth(),
                    color.getHeight()};
    list->beginRender(target);
    list->rendering = true;
}

void CommandList::endRender() {
    Backend::CommandList* list = getRecording(backend);
    if (!list || !list->check(list->rendering, ErrorCode::InvalidState, "endRender needs an active beginRender"))
        return;
    list->endRender();
    list->rendering = false;
    list->render = {};
}

void CommandList::setScissor(const std::optional<Rect> rect) {
    Backend::CommandList* list = getRecording(backend);
    if (!list || !list->check(list->rendering, ErrorCode::InvalidState, "setScissor needs an active beginRender"))
        return;
    const auto width = static_cast<std::int64_t>(list->render.width);
    const auto height = static_cast<std::int64_t>(list->render.height);
    const Rect area = rect.value_or(Rect{0, 0, list->render.width, list->render.height});
    const std::int64_t left = std::clamp<std::int64_t>(area.x, 0, width);
    const std::int64_t top = std::clamp<std::int64_t>(area.y, 0, height);
    const std::int64_t right = std::clamp<std::int64_t>(std::int64_t{area.x} + area.width, left, width);
    const std::int64_t bottom = std::clamp<std::int64_t>(std::int64_t{area.y} + area.height, top, height);
    list->setScissor(
        {static_cast<std::int32_t>(left), static_cast<std::int32_t>(top), static_cast<std::uint32_t>(right - left), static_cast<std::uint32_t>(bottom - top)});
}

void CommandList::setStencilReference(const std::uint32_t reference) {
    Backend::CommandList* list = getRecording(backend);
    if (!list || !list->check(list->rendering, ErrorCode::InvalidState, "setStencilReference needs an active beginRender")
        || !list->check(reference <= 0xff, ErrorCode::InvalidArgument, "the stencil reference is an 8-bit value"))
        return;
    list->setStencilReference(reference);
}

void CommandList::drawBytes(const GraphicsPipeline& pipeline, const std::uint32_t vertexCount, const std::uint32_t instanceCount,
                            const std::span<const std::byte> arguments) {
    Backend::CommandList* list = getRecording(backend);
    if (!list || !list->check(static_cast<bool>(pipeline), ErrorCode::InvalidArgument, "draw needs an open graphics pipeline") || !checkDraw(*list, pipeline))
        return;
    list->draw(*pipeline.getBackend(), vertexCount, instanceCount, arguments);
}

void CommandList::drawIndexedBytes(const GraphicsPipeline& pipeline, const Ptr<std::uint32_t> indices, const std::uint32_t indexCount,
                                   const std::uint32_t instanceCount, const std::span<const std::byte> arguments) {
    Backend::CommandList* list = getRecording(backend);
    if (!list
        || !list->check(pipeline && indices && isAligned(indices.getAddress()), ErrorCode::InvalidArgument,
                        "an indexed draw needs an open graphics pipeline and 4-byte aligned indices")
        || !checkDraw(*list, pipeline))
        return;
    list->drawIndexed(*pipeline.getBackend(), indices.getAddress(), indexCount, instanceCount, arguments);
}

void CommandList::drawIndirectBytes(const GraphicsPipeline& pipeline, const Ptr<std::uint32_t> indices, const std::uint64_t commands,
                                    const AtomicPtr<std::uint32_t> count, const std::uint32_t maxDraws, const std::span<const std::byte> arguments) {
    Backend::CommandList* list = getRecording(backend);
    if (!list || !list->check(pipeline && commands != 0, ErrorCode::InvalidArgument, "an indirect draw needs an open graphics pipeline and draws")
        || !list->check(isAligned(commands) && isAligned(indices.getAddress()) && isAligned(count.getAddress()), ErrorCode::InvalidArgument,
                        "the draws, indices and count of an indirect draw need 4-byte alignment")
        || !checkDraw(*list, pipeline) || maxDraws == 0)
        return;
    list->drawIndirect(*pipeline.getBackend(), {indices.getAddress(), commands, count.getAddress(), maxDraws}, arguments);
}

void CommandList::traceRaysBytes(const RayTracingPipeline& pipeline, const Shader& raygen, const DispatchSize size,
                                 const std::span<const std::byte> arguments) {
    Backend::CommandList* list = getRecording(backend);
    if (!list || !list->check(pipeline && raygen, ErrorCode::InvalidArgument, "traceRays needs an open pipeline and shader")
        || !list->check(!list->rendering, ErrorCode::InvalidState, "traceRays is not allowed inside beginRender"))
        return;
    pipeline.getBackend()->traceRays(*list, raygen, size, arguments);
}

void CommandList::copy(const Image& source, const Image& destination, const std::optional<Rect> sourceRegion) {
    Backend::CommandList* list = getRecording(backend);
    if (!list || !list->check(source && destination, ErrorCode::InvalidArgument, "copy needs two open images")
        || !list->check(!list->rendering, ErrorCode::InvalidState, "copy is not allowed inside beginRender"))
        return;
    const Rect region = sourceRegion.value_or(Rect{0, 0, source.getWidth(), source.getHeight()});
    const FormatInfo from = getFormatInfo(source.getFormat());
    const FormatInfo to = getFormatInfo(destination.getFormat());
    const bool inside = region.x >= 0 && region.y >= 0 && region.width != 0 && region.height != 0
                        && std::uint64_t{static_cast<std::uint32_t>(region.x)} + region.width <= source.getWidth()
                        && std::uint64_t{static_cast<std::uint32_t>(region.y)} + region.height <= source.getHeight();
    const bool direct = region.width == destination.getWidth() && region.height == destination.getHeight() && source.getFormat() == destination.getFormat();
    const bool aligned = region.x % static_cast<std::int32_t>(from.blockExtent) == 0 && region.y % static_cast<std::int32_t>(from.blockExtent) == 0;
    if (!list->check(inside, ErrorCode::InvalidArgument, "the copy region is empty or exceeds the source image")
        || !list->check(!from.depth && !to.depth && source.getSamples() == 1 && destination.getSamples() == 1, ErrorCode::InvalidArgument,
                        "copy works on single-sampled color images")
        || !list->check(direct ? aligned : from.blockExtent == 1 && to.blockExtent == 1 && !from.integer && !to.integer, ErrorCode::InvalidArgument,
                        "a scaled copy needs filterable formats and a block-compressed copy needs a block-aligned region of the destination size")
        || !list->check(direct || list->queue == Queue::Graphics, ErrorCode::InvalidArgument, "a scaled or converting copy needs the graphics queue"))
        return;
    list->copy(*source.getBackend(), *destination.getBackend(), region);
}

void CommandList::uploadBytes(const std::uint64_t address, const std::span<const std::byte> data) {
    Backend::CommandList* list = getRecording(backend);
    if (!list || data.empty() || !list->check(!list->rendering, ErrorCode::InvalidState, "upload is not allowed inside beginRender"))
        return;
    list->upload(address, data);
}

void CommandList::downloadBytes(const std::uint64_t address, Backend::Readback& readback, const std::size_t bytes) {
    Backend::CommandList* list = getRecording(backend);
    if (!list || !list->check(!list->rendering, ErrorCode::InvalidState, "download is not allowed inside beginRender"))
        return;
    list->download(address, readback, bytes);
}

void CommandList::beginTimestamp(const TimestampQuery& query) {
    Backend::CommandList* list = getRecording(backend);
    if (!list || !list->check(static_cast<bool>(query), ErrorCode::InvalidArgument, "beginTimestamp needs an open query"))
        return;
    list->beginTimestamp(*query.getBackend());
}

void CommandList::endTimestamp(const TimestampQuery& query) {
    Backend::CommandList* list = getRecording(backend);
    if (!list || !list->check(static_cast<bool>(query), ErrorCode::InvalidArgument, "endTimestamp needs an open query"))
        return;
    list->endTimestamp(*query.getBackend());
}

void CommandList::pushLabel(const std::string_view name) {
    if (Backend::CommandList* list = getRecording(backend))
        list->pushLabel(name);
}

void CommandList::popLabel() {
    if (Backend::CommandList* list = getRecording(backend))
        list->popLabel();
}

}
