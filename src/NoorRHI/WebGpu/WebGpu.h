#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <webgpu/webgpu_cpp.h>

#include "NoorRHI/Core/Backend.h"
#include "NoorRHI/Core/RangeAllocator.h"
#include "NoorRHI/Resources/Sampler.h"
#include "NoorRHI/WebGpu/WebGpuBindings.h"

namespace NoorRHI::Backend {

inline constexpr std::size_t argumentAlignment = 16;
inline constexpr std::uint64_t argumentWindowBytes = 4096;
inline constexpr std::uint64_t heapReservedBytes = textureTableAddress + std::uint64_t{textureTableCapacity} * 16;
inline constexpr std::array<std::uint64_t, 2> initialHeapBytes{16ull << 20, 64ull << 10};
inline constexpr std::uint64_t addressOffsetMask = 0xffffffffull;
inline constexpr std::uint64_t indirectSlotBytes = 16;

wgpu::TextureFormat getWebGpuFormat(ImageFormat format);
std::string toString(wgpu::StringView text);
std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment);

struct SampledPool {
    ImageFormat format = ImageFormat::Undefined;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t mipLevels = 0;
    bool atlas = false;
    wgpu::Texture texture;
    wgpu::TextureView view;
    std::uint32_t capacity = 0;
    std::uint32_t used = 0;
    std::vector<std::set<std::uint64_t>> freeNodes;
};

struct SampledSlot {
    std::uint32_t index = 0;
    std::uint32_t binding = 0;
    std::uint32_t layer = 0;
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t depth = 0;
    std::uint32_t levels = 1;
};

struct StoragePool {
    wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
    wgpu::Texture texture;
    wgpu::TextureView view;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t capacity = 0;
    std::uint32_t nextLayer = 0;
    std::vector<std::uint32_t> freeLayers;
};

struct TextureRegion {
    wgpu::Texture texture;
    std::uint32_t layer = 0;
    std::uint32_t x = 0;
    std::uint32_t y = 0;
};

struct BytePatch {
    std::uint32_t word = 0;
    std::uint32_t mask = 0;
    std::uint32_t value = 0;
    std::uint32_t padding = 0;
};

struct WebGpuImage;
struct WebGpuCommandList;
struct WebGpuPlatform;
struct WebGpuTimestampQuery;

struct GpuHeap {
    wgpu::Buffer buffer;
    RangeAllocator ranges;
};

struct WebGpuDevice final : Device, std::enable_shared_from_this<WebGpuDevice> {
    ~WebGpuDevice() override;

    std::expected<Token, Error> submit(CommandList& list, std::span<const Token> waits) override;
    std::expected<void, Error> wait(Token token) override;
    bool isFinished(Token token) override;
    std::expected<void, Error> waitIdle() override;

    std::expected<void, Error> open(const DeviceConfig& config);
    std::expected<std::uint64_t, Error> allocate(BufferMemory memory, std::uint64_t size, std::uint64_t alignment);
    void release(std::uint64_t address, std::uint64_t size);
    //! The queue runs uploads before work that is submitted later, so released memory waits until each command list that was open at the release is closed.
    void retire(std::function<void()> release);
    std::uint64_t openList();
    void closeList(std::uint64_t id);
    const wgpu::Buffer& getBuffer(std::uint64_t address) const { return heaps[address >> 32].buffer; }
    std::expected<void, Error> download(std::uint64_t size, const std::function<void(const wgpu::CommandEncoder&, const wgpu::Buffer&)>& record,
                                        const std::function<void(std::span<const std::byte>)>& read);
    std::expected<Token, Error> submitCommands(const wgpu::CommandBuffer& commands);
    std::expected<void, Error> waitFuture(wgpu::Future future);
    std::expected<void, Error> takeError();
    void recordError(ErrorCode code, std::string message);
    void pushErrorScope();
    std::expected<void, Error> popErrorScope(std::string_view operation);

    std::expected<SampledSlot, Error> allocateSampled(ImageFormat format, std::uint32_t width, std::uint32_t height, std::uint32_t mipLevels);
    void releaseSampled(const SampledSlot& slot);
    std::expected<std::uint32_t, Error> allocateStorageLayer(std::uint32_t binding, std::uint32_t width, std::uint32_t height);
    void releaseStorageLayer(std::uint32_t binding, std::uint32_t layer);
    TextureRegion getPrimary(const WebGpuImage& image) const;
    TextureRegion getSampled(const WebGpuImage& image) const;
    bool hasStaleShadows(bool dispatched) const;
    void refreshShadows(const wgpu::CommandEncoder& encoder, bool dispatched);
    void updateBindGroups();
    std::expected<wgpu::RenderPipeline, Error> getBlitPipeline(wgpu::TextureFormat format);
    bool isRenderable(ImageFormat format) const;

    std::unique_ptr<WebGpuPlatform> platform;
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;
    wgpu::Limits limits;
    bool presentation = false;
    bool timestamps = false;
    bool float32Filterable = false;
    bool compressionBc = false;
    bool componentSwizzle = false;
    bool formatsTier1 = false;
    bool depthStencil = false;
    bool multiDrawIndirect = false;
    bool storageImages = false;
    std::array<PoolBindings, sampledPoolBindings.size()> sampledRanges = sampledPoolBindings;

    std::recursive_mutex mutex;
    std::array<GpuHeap, 2> heaps;
    std::uint64_t heapLimit = 0;
    wgpu::Buffer staging;
    wgpu::Buffer argumentBuffer;
    wgpu::Buffer indirect;
    wgpu::Buffer drawScratch;

    wgpu::BindGroupLayout computeGroupLayout;
    wgpu::BindGroupLayout renderGroupLayout;
    wgpu::BindGroupLayout argumentGroupLayout;
    wgpu::PipelineLayout computeLayout;
    wgpu::PipelineLayout renderLayout;
    wgpu::BindGroup computeGroup;
    wgpu::BindGroup renderGroup;
    wgpu::BindGroup argumentGroup;
    std::uint64_t version = 1;
    std::uint64_t groupVersion = 0;

    std::array<SampledPool, bindingCount> sampledPools;
    std::array<StoragePool, storagePoolBindings.count> storagePools;
    std::array<wgpu::TextureView, sampledPoolBindings.size()> emptySampled;
    std::array<wgpu::TextureView, storagePoolBindings.count> emptyStorage;
    wgpu::Sampler sampler;
    std::set<WebGpuImage*> shadowedImages;
    std::vector<std::uint32_t> freeTextureIndices;
    std::uint32_t nextTextureIndex = 1;

    wgpu::BindGroupLayout blitGroupLayout;
    wgpu::PipelineLayout blitLayout;
    wgpu::ShaderModule blitModule;
    wgpu::Sampler blitSampler;
    std::map<wgpu::TextureFormat, wgpu::RenderPipeline> blitPipelines;
    //! Each embedded Slang source holds all of its entry points, so one translated WGSL module serves every pipeline that it opens.
    std::map<const std::byte*, wgpu::ShaderModule> shaderModules;
    std::map<std::string, std::string, std::less<>> translatedShaders;
    wgpu::ComputePipeline patchPipeline;
    wgpu::BindGroupLayout countGroupLayout;
    wgpu::ComputePipeline countPipeline;

    std::uint64_t nextListId = 1;
    std::set<std::uint64_t> openLists;
    std::deque<std::pair<std::uint64_t, std::function<void()>>> retired;
    std::uint64_t submitted = 0;
    std::atomic<std::uint64_t> completed = 0;
    std::deque<std::pair<std::uint64_t, wgpu::Future>> pending;
    std::mutex errorMutex;
    std::optional<Error> deviceError;

private:
    std::expected<void, Error> createAdapter(const std::filesystem::path& pipelineDirectory);
    std::expected<void, Error> createDevice(bool validation);
    std::expected<void, Error> createLayouts();
    std::expected<void, Error> createEmptyResources();
    std::expected<void, Error> growHeap(BufferMemory memory, std::uint64_t required);
    std::expected<void, Error> growSampledPool(SampledPool& pool);
    std::expected<std::uint64_t, Error> allocateNode(SampledPool& pool, std::uint32_t depth);
    std::expected<void, Error> growStoragePool(StoragePool& pool, std::uint32_t width, std::uint32_t height, std::uint32_t capacity);
    std::expected<void, Error> replay(const WebGpuCommandList& list, const wgpu::CommandEncoder& encoder,
                                      std::vector<std::shared_ptr<WebGpuTimestampQuery>>& ended);
    void reserve(wgpu::Buffer& buffer, std::uint64_t size, wgpu::BufferUsage usage);
    void reap();
    void releaseRetired();
};

struct WebGpuBuffer final : Buffer {
    ~WebGpuBuffer() override;

    std::expected<void, Error> upload(std::span<const std::byte> data, std::size_t offset) override;
    std::expected<void, Error> download(std::span<std::byte> destination, std::size_t offset) override;

    std::shared_ptr<WebGpuDevice> device;
    std::uint64_t size = 0;
};

struct WebGpuImage final : Image, std::enable_shared_from_this<WebGpuImage> {
    ~WebGpuImage() override;

    std::expected<void, Error> upload(std::span<const std::byte> mipChain) override;
    std::expected<void, Error> download(std::uint32_t x, std::uint32_t y, std::uint32_t regionWidth, std::uint32_t regionHeight,
                                        std::span<std::byte> destination) override;

    bool isShadowed() const { return sampled.binding != 0 && (texture || storageBinding != 0); }
    bool needsShadowRefresh(const bool dispatched) const { return shadowStale || (dispatched && storageBinding != 0); }
    wgpu::TextureView createView(const TextureRegion& region) const;

    std::shared_ptr<WebGpuDevice> device;
    wgpu::Texture texture;
    wgpu::TextureFormat nativeFormat = wgpu::TextureFormat::Undefined;
    ImageFormat format = ImageFormat::Undefined;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t mipLevels = 1;
    std::uint32_t samples = 1;
    SampledSlot sampled;
    std::uint32_t storageBinding = 0;
    std::uint32_t storageLayer = 0;
    bool shadowStale = false;
};

struct WebGpuSampler final : Sampler {};

struct WebGpuCompilation {
    std::shared_ptr<WebGpuDevice> device;
    wgpu::Future future;
    std::mutex mutex;
    bool finished = false;
    std::optional<Error> error;
};

struct WebGpuComputePipeline final : ComputePipeline, std::enable_shared_from_this<WebGpuComputePipeline> {
    void poll() override;
    std::expected<void, Error> wait() override;

    wgpu::ComputePipeline pipeline;
    WebGpuCompilation compilation;
};

struct WebGpuGraphicsPipeline final : GraphicsPipeline, std::enable_shared_from_this<WebGpuGraphicsPipeline> {
    void poll() override;
    std::expected<void, Error> wait() override;

    wgpu::RenderPipeline pipeline;
    WebGpuCompilation compilation;
};

struct MapState {
    std::atomic<bool> finished = false;
    wgpu::MapAsyncStatus status = wgpu::MapAsyncStatus::Error;
    std::string message;
};

struct WebGpuReadback final : Readback, std::enable_shared_from_this<WebGpuReadback> {
    bool isReady() override;
    std::expected<std::span<const std::byte>, Error> read() override;

    std::shared_ptr<WebGpuDevice> device;
    wgpu::Buffer buffer;
    std::uint64_t head = 0;
    bool recorded = false;
    wgpu::Future future;
    std::shared_ptr<MapState> map;
};

constexpr std::uint32_t timestampBeginSlot = 0;
constexpr std::uint32_t timestampEndSlot = 1;
constexpr std::uint32_t timestampScratchBeginSlot = 2;
constexpr std::uint32_t timestampScratchEndSlot = 3;
constexpr std::uint32_t timestampSlotCount = 4;

struct WebGpuTimestampQuery final : TimestampQuery, std::enable_shared_from_this<WebGpuTimestampQuery> {
    double getMilliseconds() override;

    std::shared_ptr<WebGpuDevice> device;
    wgpu::QuerySet querySet;
    wgpu::Buffer resolve;
    wgpu::Buffer readback;
    std::atomic<bool> mapping = false;
    std::atomic<double> milliseconds = 0.0;
};

struct DispatchCommand {
    std::shared_ptr<const WebGpuComputePipeline> pipeline;
    DispatchSize groups;
    std::uint64_t indirect = 0;
    std::uint32_t root = 0;
};

struct ConcurrentCommand {
    bool begin = false;
};

struct CountJob {
    std::uint32_t root = 0;
    std::uint32_t maxDraws = 0;
};

struct BeginRenderCommand {
    std::shared_ptr<WebGpuImage> color;
    std::shared_ptr<WebGpuImage> depth;
    std::shared_ptr<WebGpuImage> resolve;
    bool clear = true;
    float4 clearColor;
    float clearDepth = 1.0f;
    std::uint32_t clearStencil = 0;
    std::shared_ptr<WebGpuTimestampQuery> beginQuery;
    std::shared_ptr<WebGpuTimestampQuery> endQuery;
    std::vector<CountJob> counts;
};

struct EndRenderCommand {};

struct ScissorCommand {
    Rect rect;
};

struct StencilReferenceCommand {
    std::uint32_t reference = 0;
};

struct DrawCommand {
    std::shared_ptr<const WebGpuGraphicsPipeline> pipeline;
    std::uint32_t elementCount = 0;
    std::uint32_t instanceCount = 0;
    std::uint64_t indices = 0;
    std::uint64_t indirect = 0;
    std::uint32_t drawCount = 0;
    bool scratch = false;
    std::uint32_t root = 0;
};

struct CopyCommand {
    std::shared_ptr<const WebGpuImage> source;
    std::shared_ptr<WebGpuImage> destination;
    Rect region;
};

struct BufferCopyCommand {
    std::uint64_t address = 0;
    std::shared_ptr<WebGpuImage> destination;
    std::uint32_t level = 0;
};

struct UploadCommand {
    std::uint64_t address = 0;
    std::uint64_t stagingOffset = 0;
    std::uint64_t size = 0;
};

struct DownloadCommand {
    std::shared_ptr<WebGpuReadback> readback;
    std::uint64_t address = 0;
    std::uint64_t size = 0;
};

struct PatchCommand {
    std::uint32_t root = 0;
};

struct TimestampCommand {
    std::shared_ptr<WebGpuTimestampQuery> query;
    bool end = false;
};

struct LabelCommand {
    std::string name;
    bool pop = false;
};

struct RecordedCommand : std::variant<DispatchCommand, ConcurrentCommand, BeginRenderCommand, EndRenderCommand, ScissorCommand, StencilReferenceCommand,
                                      DrawCommand, CopyCommand, BufferCopyCommand, UploadCommand, DownloadCommand, PatchCommand, TimestampCommand, LabelCommand> {
    using variant::variant;
};

struct WebGpuCommandList final : CommandList {
    ~WebGpuCommandList() override;

    void dispatch(const ComputePipeline& pipeline, DispatchSize groups, std::span<const std::byte> arguments) override;
    void dispatchIndirect(const ComputePipeline& pipeline, std::uint64_t groups, std::span<const std::byte> arguments) override;
    void beginConcurrent() override;
    void endConcurrent() override;
    void beginRender(const RenderTarget& target) override;
    void endRender() override;
    void setScissor(const Rect& rect) override;
    void setStencilReference(std::uint32_t reference) override;
    void draw(const GraphicsPipeline& pipeline, std::uint32_t vertexCount, std::uint32_t instanceCount, std::span<const std::byte> arguments) override;
    void drawIndexed(const GraphicsPipeline& pipeline, std::uint64_t indices, std::uint32_t indexCount, std::uint32_t instanceCount,
                     std::span<const std::byte> arguments) override;
    void drawIndirect(const GraphicsPipeline& pipeline, const IndirectDraw& draw, std::span<const std::byte> arguments) override;
    void copy(const Image& source, const Image& destination, const Rect& sourceRegion) override;
    void copy(std::uint64_t address, const Image& destination, std::uint32_t level) override;
    void upload(std::uint64_t address, std::span<const std::byte> data) override;
    void download(std::uint64_t address, Readback& readback, std::size_t bytes) override;
    void beginTimestamp(TimestampQuery& query) override;
    void endTimestamp(TimestampQuery& query) override;
    void pushLabel(std::string_view name) override;
    void popLabel() override;

    std::uint32_t place(std::span<const std::byte> data);
    void patch(std::uint64_t address, std::span<const std::byte> data);

    std::shared_ptr<WebGpuDevice> device;
    std::uint64_t id = 0;
    bool closed = false;
    std::vector<RecordedCommand> commands;
    std::vector<std::byte> arguments;
    std::vector<std::byte> uploads;
    std::uint32_t indirectDispatches = 0;
    std::uint64_t drawScratchBytes = 0;
    std::size_t renderBegin = 0;
    std::vector<std::shared_ptr<WebGpuReadback>> readbacks;
};

struct WebGpuSwapchain final : Swapchain {
    std::expected<void, Error> waitUntilReady() override;
    std::expected<std::optional<AcquiredImage>, Error> acquire(bool block) override;
    std::expected<void, Error> present(std::uint32_t index, std::uint32_t semaphoreIndex) override;
    void setPresentMode(PresentMode mode) override;
    void invalidate() override;

    std::expected<void, Error> configure();
    std::expected<bool, Error> waitForSlot(bool block);

    std::shared_ptr<WebGpuDevice> device;
    SurfaceProvider* provider = nullptr;
    wgpu::Surface surface;
    wgpu::TextureFormat nativeFormat = wgpu::TextureFormat::Undefined;
    wgpu::TextureUsage nativeUsage = wgpu::TextureUsage::RenderAttachment;
    std::uint32_t maxFramesInFlight = 1;
    bool transparent = false;
    bool stale = true;
    bool acquired = false;
    std::uint32_t generation = 0;
    std::deque<Token> inFlight;
    std::mutex mutex;
};

}
