#pragma once

#include <array>
#include <condition_variable>
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
#include <thread>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "NoorRHI/Core/Backend.h"
#include "NoorRHI/Core/RangeAllocator.h"

namespace NoorRHI::Backend {

inline constexpr std::size_t argumentAlignment = 16;
inline constexpr std::size_t argumentChunkBytes = 256u * 1024u;
inline constexpr std::uint32_t samplerBinding = 0;
inline constexpr std::uint32_t sampledImageBinding = 2;
inline constexpr std::uint32_t storageImageBinding = 3;

vk::Format getVulkanFormat(ImageFormat format);
std::unexpected<Error> failVulkan(vk::Result result, std::string_view operation);
std::expected<void, Error> checkVulkan(vk::Result result, std::string_view operation);

template <class T>
std::expected<T, Error> checkVulkan(vk::ResultValue<T>&& result, const std::string_view operation) {
    if (result.result != vk::Result::eSuccess)
        return failVulkan(result.result, operation);
    return std::move(result.value);
}

struct VulkanDevice;
std::expected<vk::UniqueShaderModule, Error> createShaderModule(const VulkanDevice& device, const Shader& shader);

enum class MemoryKind : std::uint8_t { Device, Upload, Readback, Image };

struct Allocation {
    vk::Buffer buffer;
    vk::DeviceMemory memory;
    vk::DeviceSize offset = 0;
    vk::DeviceAddress address = 0;
    std::size_t size = 0;
    std::byte* mapped = nullptr;
    std::uint32_t heap = 0;
    std::uint32_t block = 0;
};

struct MemoryBlock {
    vk::DeviceMemory memory;
    vk::Buffer buffer;
    vk::DeviceAddress address = 0;
    std::byte* mapped = nullptr;
    RangeAllocator ranges;
};

struct MemoryHeap {
    MemoryKind kind = MemoryKind::Device;
    std::uint32_t typeIndex = 0;
    std::vector<MemoryBlock> blocks;
};

struct CommandPool {
    vk::UniqueCommandPool pool;
    vk::CommandBuffer commands;
};

struct ArgumentChunk {
    Allocation memory;
    std::size_t used = 0;
};

struct InFlight {
    std::uint64_t value = 0;
    CommandPool pool;
    std::vector<ArgumentChunk> chunks;
};

struct QueueState {
    std::uint32_t family = 0;
    vk::Queue queue;
    vk::UniqueSemaphore timeline;
    std::uint64_t submitted = 0;
    std::uint32_t timestampValidBits = 0;
    std::vector<CommandPool> freePools;
    std::deque<InFlight> inFlight;
};

struct Retired {
    std::uint64_t listWatermark = 0;
    std::array<std::uint64_t, queueCount> timelines{};
    bool snapshotted = false;
    std::function<void()> release;
};

struct SlotAllocator {
    std::uint32_t capacity = 0;
    std::uint32_t next = 1;
    std::vector<std::uint32_t> free;
};

struct Placement {
    vk::Buffer buffer;
    vk::DeviceSize offset = 0;
    vk::DeviceAddress address = 0;
};

struct BufferRange {
    vk::Buffer buffer;
    vk::DeviceSize offset = 0;
    vk::DeviceSize size = 0;
};

struct BinarySemaphores {
    vk::Semaphore wait;
    vk::Semaphore signal;
};

struct CompileState {
    std::expected<void, Error> wait();
    void finish(Compilation& pipeline, std::expected<void, Error> result);

    std::mutex mutex;
    std::condition_variable done;
    bool finished = false;
    std::optional<Error> error;
};

class PipelineCache {
public:
    std::expected<void, Error> open(vk::Device device, const std::filesystem::path& directory);
    std::expected<void, Error> save();
    vk::PipelineCache getHandle() const { return *cache; }

private:
    vk::Device device;
    std::filesystem::path path;
    std::uint64_t writerId = 0;
    std::mutex saveMutex;
    vk::UniquePipelineCache cache;
};

struct VulkanCommandList;

//! A resource of zero means device addresses; otherwise it is an image handle and the range is unused.
struct CopyRange {
    std::uint64_t resource = 0;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
};

struct VulkanDevice final : Device, std::enable_shared_from_this<VulkanDevice> {
    ~VulkanDevice() override;

    std::expected<Token, Error> submit(CommandList& list, std::span<const Token> waits) override;
    std::expected<void, Error> wait(Token token) override;
    bool isFinished(Token token) override;
    std::expected<void, Error> waitIdle() override;

    std::expected<void, Error> open(const DeviceConfig& config);
    std::expected<Allocation, Error> allocate(std::size_t size, MemoryKind kind, std::size_t alignment = argumentAlignment);
    std::expected<Allocation, Error> allocate(const vk::MemoryRequirements& requirements);
    void release(const Allocation& allocation);
    void retire(std::function<void()> release);
    std::unique_ptr<VulkanCommandList> openList(Queue queue);
    std::expected<Token, Error> submit(VulkanCommandList& list, std::span<const Token> waits, BinarySemaphores binary);
    std::expected<Token, Error> execute(const std::function<void(VulkanCommandList&)>& record);
    std::expected<void, Error> download(std::span<std::byte> destination, const std::function<void(vk::CommandBuffer, const Allocation&)>& record);
    void discard(VulkanCommandList& list);
    std::expected<std::uint32_t, Error> writeDescriptor(SlotAllocator& slots, std::uint32_t binding, vk::DescriptorType type,
                                                        const vk::DescriptorImageInfo& image);
    void releaseSlot(SlotAllocator& slots, std::uint32_t slot);
    std::optional<BufferRange> findBuffer(vk::DeviceAddress address);
    vk::PipelineCache getPipelineCache(bool use) const { return use && pipelineCache ? pipelineCache->getHandle() : vk::PipelineCache{}; }
    QueueState& getQueue(const Queue queue) { return queues[static_cast<std::size_t>(queue)]; }
    void compile(std::function<void()> job);
    vk::SharingMode getSharing() const { return sharedFamilies.size() > 1 ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive; }

    vk::detail::DynamicLoader loader;
    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT messenger;
    vk::PhysicalDevice physicalDevice;
    vk::UniqueDevice device;
    std::array<QueueState, queueCount> queues;
    std::vector<std::uint32_t> sharedFamilies;
    bool presentation = false;
    bool debugLabels = false;
    bool externalMemoryFd = false;
    bool externalSemaphoreFd = false;
    bool fifoLatestReady = false;
    bool allowHardwareRayTracing = true;
    bool allowCooperativeMatrices = true;
    bool subgroupExtendedTypes = false;
    vk::PhysicalDeviceAccelerationStructurePropertiesKHR accelerationProperties;
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties;
    float timestampPeriod = 1.0f;
    vk::UniqueDescriptorSetLayout descriptorLayout;
    vk::UniqueDescriptorPool descriptorPool;
    vk::DescriptorSet descriptorSet;
    vk::UniquePipelineLayout pipelineLayout;
    std::mutex descriptorMutex;
    SlotAllocator textureSlots;
    SlotAllocator samplerSlots;
    std::optional<PipelineCache> pipelineCache;
    std::mutex rayTracingCompileMutex;

    std::mutex mutex;
    std::vector<Allocation> freeChunks;
    std::uint64_t nextListId = 1;
    std::set<std::uint64_t> openLists;
    std::deque<Retired> retired;

    std::mutex memoryMutex;
    std::vector<MemoryHeap> heaps;
    vk::PhysicalDeviceMemoryProperties memoryProperties;

    std::mutex compileMutex;
    std::condition_variable compileSignal;
    std::deque<std::function<void()>> compileJobs;
    std::vector<std::thread> compileThreads;
    bool stopCompiling = false;

private:
    std::expected<void, Error> createInstance(const DeviceConfig& config);
    std::expected<vk::UniqueSurfaceKHR, Error> createProbeSurface(const DeviceConfig& config);
    std::expected<void, Error> selectPhysicalDevice(vk::SurfaceKHR probeSurface);
    std::expected<void, Error> createDevice(vk::SurfaceKHR probeSurface);
    std::expected<void, Error> createMemoryHeaps();
    std::expected<std::uint32_t, Error> addBlock(std::uint32_t heapIndex, vk::DeviceSize size);
    std::expected<Allocation, Error> allocate(std::uint32_t heapIndex, vk::DeviceSize size, vk::DeviceSize alignment);
    vk::BufferUsageFlags getBlockUsage() const;
    void releaseMemoryHeaps();
    std::expected<void, Error> createQueues();
    std::expected<void, Error> createBindlessSet();
    void reap();
    void recycle(QueueState& queue, InFlight&& done);
};

struct VulkanBuffer final : Buffer {
    ~VulkanBuffer() override;

    std::expected<void, Error> upload(std::span<const std::byte> data, std::size_t offset) override;
    std::expected<void, Error> download(std::span<std::byte> destination, std::size_t offset) override;

    std::shared_ptr<VulkanDevice> device;
    Allocation memory;
};

struct VulkanImage final : Image {
    ~VulkanImage() override;

    std::expected<void, Error> upload(std::span<const std::byte> mipChain) override;
    std::expected<void, Error> download(std::uint32_t x, std::uint32_t y, std::uint32_t regionWidth, std::uint32_t regionHeight,
                                        std::span<std::byte> destination) override;

    std::shared_ptr<VulkanDevice> device;
    vk::Image image;
    Allocation memory;
    vk::DeviceMemory externalMemory;
    bool ownsImage = true;
    vk::UniqueImageView view;
    vk::UniqueImageView sampledView;
    vk::UniqueImageView storageView;
    std::uint32_t sampledSlot = 0;
    std::uint32_t storageSlot = 0;
    vk::Format format = vk::Format::eUndefined;
    ImageFormat publicFormat = ImageFormat::Undefined;
    vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t mipLevels = 1;
};

struct VulkanSampler final : Sampler {
    ~VulkanSampler() override;

    std::shared_ptr<VulkanDevice> device;
    vk::Sampler sampler;
    std::uint32_t slot = 0;
};

struct VulkanReadback final : Readback, std::enable_shared_from_this<VulkanReadback> {
    ~VulkanReadback() override;

    bool isReady() override;
    std::expected<std::span<const std::byte>, Error> read() override;

    std::shared_ptr<VulkanDevice> device;
    Allocation memory;
    Token token;
};

struct VulkanComputePipeline final : ComputePipeline {
    ~VulkanComputePipeline() override;

    std::expected<void, Error> wait() override { return state.wait(); }

    std::shared_ptr<VulkanDevice> device;
    vk::Pipeline pipeline;
    CompileState state;
};

struct VulkanGraphicsPipeline final : GraphicsPipeline {
    ~VulkanGraphicsPipeline() override;

    std::expected<void, Error> wait() override { return state.wait(); }

    std::shared_ptr<VulkanDevice> device;
    vk::Pipeline pipeline;
    CompileState state;
};

struct VulkanTimestampQuery final : TimestampQuery {
    ~VulkanTimestampQuery() override;

    double getMilliseconds() override;

    std::shared_ptr<VulkanDevice> device;
    vk::QueryPool pool;
    Queue queue = Queue::Graphics;
    double milliseconds = 0.0;
};

struct VulkanRayTracingPipeline;

struct VulkanCommandList final : CommandList {
    ~VulkanCommandList() override;

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

    void traceRays(const VulkanRayTracingPipeline& pipeline, const Shader& raygen, DispatchSize size, std::span<const std::byte> arguments);
    std::expected<Placement, Error> place(std::span<const std::byte> data);
    void pushRoot(std::span<const std::byte> arguments);
    void bind(vk::PipelineBindPoint point, vk::Pipeline pipeline);
    bool bindIndices(std::uint64_t address);
    void synchronize(vk::PipelineStageFlags2 stages, vk::PipelineStageFlags2 readStages = {});
    void synchronizeCopy(std::span<const CopyRange> reads, std::span<const CopyRange> writes);
    void finish();

    std::shared_ptr<VulkanDevice> device;
    std::uint64_t id = 0;
    bool closed = false;
    std::array<vk::Pipeline, 3> boundPipelines{};
    vk::DeviceAddress boundRoot = ~vk::DeviceAddress{0};
    vk::DeviceAddress boundIndices = 0;
    CommandPool pool;
    std::vector<ArgumentChunk> chunks;
    std::vector<std::shared_ptr<VulkanReadback>> readbacks;
    //! Earlier lists of the queue may have written anything, so the first command waits for all of them.
    vk::PipelineStageFlags2 unsynchronized = vk::PipelineStageFlagBits2::eAllCommands;
    vk::PipelineStageFlags2 concurrentSource;
    vk::PipelineStageFlags2 concurrentWaiting;
    std::vector<CopyRange> copyReads;
    std::vector<CopyRange> copyWrites;
};

struct VulkanAccelerationStructure final : AccelerationStructure {
    ~VulkanAccelerationStructure() override;

    std::expected<void, Error> refit() override;
    std::expected<void, Error> update(std::span<const InstanceRecord> records) override;
    std::expected<void, Error> update(Ptr<InstanceRecord> records, std::uint32_t count) override;

    std::expected<void, Error> build(vk::BuildAccelerationStructureModeKHR mode, std::span<const std::byte> records);

    std::shared_ptr<VulkanDevice> device;
    vk::AccelerationStructureKHR structure;
    Allocation storage;
    Allocation scratch;
    bool topLevel = false;
    bool updatable = false;
    vk::BuildAccelerationStructureFlagsKHR flags;
    std::vector<vk::AccelerationStructureGeometryKHR> geometries;
    std::vector<std::uint32_t> primitiveCounts;
};

enum class GroupKind : std::uint8_t { Raygen, Miss, Hit, Callable };

struct ShaderGroup {
    GroupKind kind = GroupKind::Raygen;
    Shader raygen;
};

struct VulkanRayTracingLibrary final : RayTracingLibrary {
    ~VulkanRayTracingLibrary() override;

    std::shared_ptr<VulkanDevice> device;
    vk::Pipeline pipeline;
    std::vector<ShaderGroup> groups;
    bool usePipelineCache = true;
};

struct LinkedPipeline {
    ~LinkedPipeline();

    std::shared_ptr<VulkanDevice> device;
    vk::Pipeline pipeline;
    std::vector<std::shared_ptr<VulkanRayTracingLibrary>> libraries;
    std::vector<ShaderGroup> groups;
};

struct VulkanRayTracingPipeline final : RayTracingPipeline {
    ~VulkanRayTracingPipeline() override;

    void traceRays(CommandList& list, const Shader& raygen, DispatchSize size, std::span<const std::byte> arguments) const override;
    RayTracingPipelineResult relink(std::span<const std::uint32_t> hitGroups) const override;

    std::shared_ptr<VulkanDevice> device;
    std::shared_ptr<LinkedPipeline> linked;
    Allocation table;
    std::vector<std::pair<Shader, vk::StridedDeviceAddressRegionKHR>> raygenRegions;
    vk::StridedDeviceAddressRegionKHR missRegion;
    vk::StridedDeviceAddressRegionKHR hitRegion;
    vk::StridedDeviceAddressRegionKHR callableRegion;
};

struct AcquireSemaphore {
    vk::UniqueSemaphore semaphore;
    Token token;
};

struct VulkanSwapchain final : Swapchain {
    ~VulkanSwapchain() override;

    std::expected<void, Error> waitUntilReady() override;
    std::expected<std::optional<AcquiredImage>, Error> acquire(bool block) override;
    std::expected<void, Error> present(std::uint32_t index, std::uint32_t semaphoreIndex) override;
    void setPresentMode(PresentMode mode) override;
    void invalidate() override;

    std::expected<void, Error> rebuild();
    void retireChain();
    std::expected<bool, Error> waitForSlot(bool block);

    std::shared_ptr<VulkanDevice> device;
    SurfaceProvider* provider = nullptr;
    vk::UniqueSurfaceKHR surface;
    vk::UniqueSwapchainKHR swapchain;
    std::vector<std::shared_ptr<VulkanImage>> images;
    std::vector<bool> presented;
    std::vector<AcquireSemaphore> acquireSemaphores;
    std::vector<vk::UniqueSemaphore> presentSemaphores;
    std::uint32_t semaphoreCursor = 0;
    vk::Format vulkanFormat = vk::Format::eUndefined;
    vk::PresentModeKHR presentMode = vk::PresentModeKHR::eFifo;
    std::uint32_t maxFramesInFlight = 1;
    bool transparent = false;
    bool stale = true;
    std::deque<Token> inFlight;
    std::mutex mutex;
};

void recordImageBarrier(vk::CommandBuffer commands, vk::Image image, vk::ImageAspectFlags aspect, vk::ImageLayout from, vk::ImageLayout to);

}
