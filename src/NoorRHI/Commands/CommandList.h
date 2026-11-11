#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>

#include "NoorRHI/Commands/TimestampQuery.h"
#include "NoorRHI/Core/Error.h"
#include "NoorRHI/Core/Types.h"
#include "NoorRHI/Pipelines/Pipeline.h"
#include "NoorRHI/Pipelines/RayTracing.h"
#include "NoorRHI/Resources/Buffer.h"
#include "NoorRHI/Resources/Image.h"
#include "NoorRHI/Resources/Readback.h"

namespace NoorRHI {

class Device;
namespace Backend {
struct CommandList;
}

struct RenderTarget {
    const Image* color = nullptr;
    const Image* depth = nullptr;
    const Image* resolve = nullptr;
    bool clear = true;
    float4 clearColor{0.0f, 0.0f, 0.0f, 1.0f};
    float clearDepth = 1.0f;
    std::uint32_t clearStencil = 0;
};

template <class T>
concept GpuArguments = std::is_trivially_copyable_v<T>;

struct NoArguments {};

class CommandList {
public:
    explicit CommandList(Device& device, Queue queue = Queue::Graphics);
    CommandList(CommandList&& other) noexcept;
    CommandList& operator=(CommandList&& other) noexcept;
    ~CommandList();

    template <GpuArguments Arguments = NoArguments>
    void dispatch(const ComputePipeline& pipeline, const DispatchSize threads, const Arguments& arguments = {}) {
        dispatchBytes(pipeline, threads, toBytes(arguments));
    }
    template <GpuArguments Arguments = NoArguments>
    void dispatchIndirect(const ComputePipeline& pipeline, const Ptr<DispatchSize> groups, const Arguments& arguments = {}) {
        dispatchIndirectBytes(pipeline, groups, toBytes(arguments));
    }

    //! Each command sees the writes of the commands before it, except between beginConcurrent and endConcurrent, where no command may read what another writes.
    void beginConcurrent();
    void endConcurrent();

    void beginRender(const RenderTarget& target);
    void endRender();
    void setScissor(std::optional<Rect> rect);
    void setStencilReference(std::uint32_t reference);
    template <GpuArguments Arguments = NoArguments>
    void draw(const GraphicsPipeline& pipeline, const std::uint32_t vertexCount, const std::uint32_t instanceCount = 1, const Arguments& arguments = {}) {
        drawBytes(pipeline, vertexCount, instanceCount, toBytes(arguments));
    }
    template <GpuArguments Arguments = NoArguments>
    void draw(const GraphicsPipeline& pipeline, const Ptr<std::uint32_t> indices, const std::uint32_t indexCount, const std::uint32_t instanceCount = 1,
              const Arguments& arguments = {}) {
        drawIndexedBytes(pipeline, indices, indexCount, instanceCount, toBytes(arguments));
    }
    template <GpuArguments Arguments = NoArguments>
    void drawIndirect(const GraphicsPipeline& pipeline, const Ptr<DrawArgs> commands, const std::uint32_t drawCount = 1, const Arguments& arguments = {}) {
        drawIndirectBytes(pipeline, {}, commands.getAddress(), {}, drawCount, toBytes(arguments));
    }
    template <GpuArguments Arguments = NoArguments>
    void drawIndirect(const GraphicsPipeline& pipeline, const Ptr<std::uint32_t> indices, const Ptr<DrawIndexedArgs> commands,
                      const std::uint32_t drawCount = 1, const Arguments& arguments = {}) {
        drawIndirectBytes(pipeline, indices, commands.getAddress(), {}, drawCount, toBytes(arguments));
    }
    //! The GPU reads the draw count from count and draws at most maxDraws commands.
    template <GpuArguments Arguments = NoArguments>
    void drawIndirect(const GraphicsPipeline& pipeline, const Ptr<DrawArgs> commands, const AtomicPtr<std::uint32_t> count, const std::uint32_t maxDraws,
                      const Arguments& arguments = {}) {
        drawIndirectBytes(pipeline, {}, commands.getAddress(), count, maxDraws, toBytes(arguments));
    }
    template <GpuArguments Arguments = NoArguments>
    void drawIndirect(const GraphicsPipeline& pipeline, const Ptr<std::uint32_t> indices, const Ptr<DrawIndexedArgs> commands,
                      const AtomicPtr<std::uint32_t> count, const std::uint32_t maxDraws, const Arguments& arguments = {}) {
        drawIndirectBytes(pipeline, indices, commands.getAddress(), count, maxDraws, toBytes(arguments));
    }

    template <GpuArguments Arguments = NoArguments>
    void traceRays(const RayTracingPipeline& pipeline, const Shader& raygen, const DispatchSize size, const Arguments& arguments = {}) {
        traceRaysBytes(pipeline, raygen, size, toBytes(arguments));
    }

    void copy(const Image& source, const Image& destination, std::optional<Rect> sourceRegion = std::nullopt);
    template <class T>
    void upload(const Buffer<T>& buffer, const std::span<const T> data, const std::size_t offset = 0) {
        if (offset > buffer.getSize() || data.size() > buffer.getSize() - offset)
            return setError(ErrorCode::InvalidArgument, "the upload range exceeds the buffer");
        uploadBytes(buffer.getPtr().getAddress() + offset * sizeof(T), std::as_bytes(data));
    }
    //! Copies readback.getSize() elements from offset; Readback::isReady tells when the CPU can read them.
    template <class T>
    void download(const Buffer<T>& buffer, Readback<T>& readback, const std::size_t offset = 0) {
        if (!buffer || !readback || offset > buffer.getSize() || readback.getSize() > buffer.getSize() - offset)
            return setError(ErrorCode::InvalidArgument, "the download needs an open buffer and readback, and the range must be inside the buffer");
        downloadBytes(buffer.getPtr().getAddress() + offset * sizeof(T), *readback.getBackend(), readback.getSize() * sizeof(T));
    }

    void beginTimestamp(const TimestampQuery& query);
    void endTimestamp(const TimestampQuery& query);
    void pushLabel(std::string_view name);
    void popLabel();

    Queue getQueue() const;
    const std::optional<Error>& getError() const;
    Backend::CommandList* getBackend() const { return backend.get(); }

private:
    template <GpuArguments Arguments>
    static std::span<const std::byte> toBytes(const Arguments& arguments) {
        if constexpr (std::is_empty_v<Arguments>)
            return {};
        else
            return std::as_bytes(std::span(&arguments, 1));
    }
    void dispatchBytes(const ComputePipeline& pipeline, DispatchSize threads, std::span<const std::byte> arguments);
    void dispatchIndirectBytes(const ComputePipeline& pipeline, Ptr<DispatchSize> groups, std::span<const std::byte> arguments);
    void drawBytes(const GraphicsPipeline& pipeline, std::uint32_t vertexCount, std::uint32_t instanceCount, std::span<const std::byte> arguments);
    void drawIndexedBytes(const GraphicsPipeline& pipeline, Ptr<std::uint32_t> indices, std::uint32_t indexCount, std::uint32_t instanceCount,
                          std::span<const std::byte> arguments);
    void drawIndirectBytes(const GraphicsPipeline& pipeline, Ptr<std::uint32_t> indices, std::uint64_t commands, AtomicPtr<std::uint32_t> count,
                           std::uint32_t maxDraws, std::span<const std::byte> arguments);
    void traceRaysBytes(const RayTracingPipeline& pipeline, const Shader& raygen, DispatchSize size, std::span<const std::byte> arguments);
    void uploadBytes(std::uint64_t address, std::span<const std::byte> data);
    void downloadBytes(std::uint64_t address, Backend::Readback& readback, std::size_t bytes);
    void setError(ErrorCode code, std::string_view message);

    std::unique_ptr<Backend::CommandList> backend;
};

}
