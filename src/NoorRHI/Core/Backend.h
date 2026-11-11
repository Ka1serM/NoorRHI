#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Core/Error.h"
#include "NoorRHI/Core/Format.h"
#include "NoorRHI/Core/Types.h"
#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Pipelines/RayTracing.h"
#include "NoorRHI/Present/Swapchain.h"

namespace NoorRHI::Backend {

struct SoftwareBuildKernels;

struct Device {
    virtual ~Device() = default;

    virtual std::expected<Token, Error> submit(CommandList& list, std::span<const Token> waits) = 0;
    virtual std::expected<void, Error> wait(Token token) = 0;
    virtual bool isFinished(Token token) = 0;
    virtual std::expected<void, Error> waitIdle() = 0;

    DeviceFeatures features;
    DeviceInfo info;
    std::mutex softwareBuildMutex;
    std::weak_ptr<const SoftwareBuildKernels> softwareBuildKernels;
};

struct Buffer {
    virtual ~Buffer() = default;

    virtual std::expected<void, Error> upload(std::span<const std::byte> data, std::size_t offset) = 0;
    virtual std::expected<void, Error> download(std::span<std::byte> destination, std::size_t offset) = 0;

    std::uint64_t address = 0;
};

struct Image {
    virtual ~Image() = default;

    virtual std::expected<void, Error> upload(std::span<const std::byte> mipChain) = 0;
    virtual std::expected<void, Error> download(std::uint32_t x, std::uint32_t y, std::uint32_t width, std::uint32_t height,
                                                std::span<std::byte> destination) = 0;
};

struct Sampler {
    virtual ~Sampler() = default;
};

struct Readback {
    virtual ~Readback() = default;

    virtual bool isReady() = 0;
    virtual std::expected<std::span<const std::byte>, Error> read() = 0;

    std::size_t size = 0;
};

struct Compilation {
    virtual ~Compilation() = default;

    virtual void poll() {}
    virtual std::expected<void, Error> wait() = 0;

    std::atomic<bool> ready = false;
};

struct ComputePipeline : Compilation {
    DispatchSize threadGroupSize;
};

struct GraphicsPipeline : Compilation {
    ImageFormat colorFormat = ImageFormat::Undefined;
    ImageFormat depthFormat = ImageFormat::Undefined;
    std::uint32_t samples = 1;
};

struct RenderInfo {
    ImageFormat colorFormat = ImageFormat::Undefined;
    ImageFormat depthFormat = ImageFormat::Undefined;
    std::uint32_t samples = 1;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

std::expected<void, Error> checkImageDesc(const ImageDesc& desc);
std::expected<void, Error> checkShader(const Shader& shader, ShaderStage stage);
std::expected<void, Error> checkGraphicsPipelineDesc(const GraphicsPipelineDesc& desc);
void setTarget(GraphicsPipeline& pipeline, const GraphicsPipelineDesc& desc);
std::expected<std::shared_ptr<ComputePipeline>, Error> openComputePipeline(NoorRHI::Device& device, const Shader& shader, bool background);
std::expected<std::shared_ptr<GraphicsPipeline>, Error> openGraphicsPipeline(NoorRHI::Device& device, const GraphicsPipelineDesc& desc, bool background);

struct TimestampQuery {
    virtual ~TimestampQuery() = default;

    virtual double getMilliseconds() = 0;
};

struct AccelerationStructure {
    virtual ~AccelerationStructure() = default;

    virtual std::expected<void, Error> refit() = 0;
    virtual std::expected<void, Error> update(std::span<const InstanceRecord> records) = 0;
    virtual std::expected<void, Error> update(Ptr<InstanceRecord> records, std::uint32_t count) = 0;

    std::uint64_t address = 0;
};

struct RayTracingLibrary {
    virtual ~RayTracingLibrary() = default;

    RayTracingInterface interface;
};

struct CommandList;

struct RayTracingPipeline {
    virtual ~RayTracingPipeline() = default;

    virtual void traceRays(CommandList& list, const Shader& raygen, DispatchSize size, std::span<const std::byte> arguments) const = 0;
    virtual std::expected<std::shared_ptr<const RayTracingPipeline>, Error> relink(std::span<const std::uint32_t> hitGroups) const = 0;
};

using AccelerationStructureResult = std::expected<std::shared_ptr<AccelerationStructure>, Error>;
using RayTracingLibraryResult = std::expected<std::shared_ptr<RayTracingLibrary>, Error>;
using RayTracingPipelineResult = std::expected<std::shared_ptr<const RayTracingPipeline>, Error>;

std::expected<InstanceRecord, Error> makeInstanceRecord(const Instance& instance);

AccelerationStructureResult buildHardwareBlas(NoorRHI::Device& device, std::span<const TriangleGeometry> geometry, BuildMode mode);
AccelerationStructureResult buildHardwareTlas(NoorRHI::Device& device, std::span<const InstanceRecord> records);
AccelerationStructureResult buildHardwareTlas(NoorRHI::Device& device, Ptr<InstanceRecord> records, std::uint32_t count);
RayTracingLibraryResult openHardwareLibrary(NoorRHI::Device& device, const RayTracingPipelineDesc& desc, const RayTracingInterface& interface);
RayTracingPipelineResult openHardwarePipeline(NoorRHI::Device& device, std::span<const std::shared_ptr<RayTracingLibrary>> libraries,
                                              std::span<const std::uint32_t> hitGroups);

AccelerationStructureResult buildSoftwareBlas(NoorRHI::Device& device, std::span<const TriangleGeometry> geometry, BuildMode mode);
AccelerationStructureResult buildSoftwareTlas(NoorRHI::Device& device, std::span<const InstanceRecord> records);
AccelerationStructureResult buildSoftwareTlas(NoorRHI::Device& device, Ptr<InstanceRecord> records, std::uint32_t count);
RayTracingLibraryResult openSoftwareLibrary(NoorRHI::Device& device, const RayTracingPipelineDesc& desc, const RayTracingInterface& interface);
RayTracingPipelineResult openSoftwarePipeline(NoorRHI::Device& device, std::span<const std::shared_ptr<RayTracingLibrary>> libraries,
                                              std::span<const std::uint32_t> hitGroups);

struct IndirectDraw {
    std::uint64_t indices = 0;
    std::uint64_t commands = 0;
    std::uint64_t count = 0;
    std::uint32_t maxDraws = 0;
};

struct CommandList {
    virtual ~CommandList() = default;

    virtual void dispatch(const ComputePipeline& pipeline, DispatchSize groups, std::span<const std::byte> arguments) = 0;
    virtual void dispatchIndirect(const ComputePipeline& pipeline, std::uint64_t groups, std::span<const std::byte> arguments) = 0;
    virtual void beginConcurrent() = 0;
    virtual void endConcurrent() = 0;
    virtual void beginRender(const RenderTarget& target) = 0;
    virtual void endRender() = 0;
    virtual void setScissor(const Rect& rect) = 0;
    virtual void setStencilReference(std::uint32_t reference) = 0;
    virtual void draw(const GraphicsPipeline& pipeline, std::uint32_t vertexCount, std::uint32_t instanceCount, std::span<const std::byte> arguments) = 0;
    virtual void drawIndexed(const GraphicsPipeline& pipeline, std::uint64_t indices, std::uint32_t indexCount, std::uint32_t instanceCount,
                             std::span<const std::byte> arguments) = 0;
    virtual void drawIndirect(const GraphicsPipeline& pipeline, const IndirectDraw& draw, std::span<const std::byte> arguments) = 0;
    virtual void copy(const Image& source, const Image& destination, const Rect& sourceRegion) = 0;
    virtual void upload(std::uint64_t address, std::span<const std::byte> data) = 0;
    virtual void download(std::uint64_t address, Readback& readback, std::size_t bytes) = 0;
    virtual void beginTimestamp(TimestampQuery& query) = 0;
    virtual void endTimestamp(TimestampQuery& query) = 0;
    virtual void pushLabel(std::string_view name) = 0;
    virtual void popLabel() = 0;

    bool check(const bool condition, const ErrorCode code, const std::string_view message) {
        if (condition)
            return true;
        if (!error)
            error = Error{code, std::string(message)};
        return false;
    }

    Queue queue = Queue::Graphics;
    std::optional<Error> error;
    bool rendering = false;
    bool concurrent = false;
    RenderInfo render;
};

struct AcquiredImage {
    std::shared_ptr<Image> image;
    std::uint32_t index = 0;
    std::uint32_t semaphoreIndex = 0;
};

struct Swapchain {
    virtual ~Swapchain() = default;

    virtual std::expected<void, Error> waitUntilReady() = 0;
    virtual std::expected<std::optional<AcquiredImage>, Error> acquire(bool block) = 0;
    virtual std::expected<void, Error> present(std::uint32_t index, std::uint32_t semaphoreIndex) = 0;
    virtual void setPresentMode(PresentMode mode) = 0;
    virtual void invalidate() = 0;

    PresentMode requestedMode = PresentMode::LowLatency;
    ActivePresentMode activeMode = ActivePresentMode::Fifo;
    ImageFormat format = ImageFormat::Undefined;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t imageCount = 0;
};

}
