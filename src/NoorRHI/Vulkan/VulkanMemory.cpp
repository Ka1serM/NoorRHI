#include "NoorRHI/Vulkan/Vulkan.h"

namespace NoorRHI::Backend {

namespace {

constexpr std::array bufferKinds{MemoryKind::Device, MemoryKind::Upload, MemoryKind::Readback};

vk::DeviceSize getBlockBytes(const MemoryKind kind) {
    switch (kind) {
    case MemoryKind::Device:
    case MemoryKind::Image:
        return 256ull << 20;
    case MemoryKind::Upload:
        return 64ull << 20;
    case MemoryKind::Readback:
        return 16ull << 20;
    }
    return 0;
}

vk::DeviceSize alignUp(const vk::DeviceSize value, const vk::DeviceSize alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

struct MemoryPreference {
    vk::MemoryPropertyFlags required;
    vk::MemoryPropertyFlags preferred;
};

MemoryPreference getPreference(const MemoryKind kind) {
    constexpr vk::MemoryPropertyFlags hostCoherent = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    switch (kind) {
    case MemoryKind::Device:
    case MemoryKind::Image:
        return {vk::MemoryPropertyFlagBits::eDeviceLocal, {}};
    case MemoryKind::Upload:
        return {hostCoherent, vk::MemoryPropertyFlagBits::eDeviceLocal};
    case MemoryKind::Readback:
        return {hostCoherent, vk::MemoryPropertyFlagBits::eHostCached};
    }
    return {};
}

std::optional<std::uint32_t> findMemoryType(const vk::PhysicalDeviceMemoryProperties& properties, const std::uint32_t typeBits, const MemoryKind kind) {
    const auto [required, preferred] = getPreference(kind);
    std::optional<std::uint32_t> fallback;
    for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
        const vk::MemoryPropertyFlags flags = properties.memoryTypes[index].propertyFlags;
        if (!(typeBits & (1u << index)) || (flags & required) != required)
            continue;
        if ((flags & preferred) == preferred)
            return index;
        if (!fallback)
            fallback = index;
    }
    return fallback;
}

}

vk::BufferUsageFlags VulkanDevice::getBlockUsage() const {
    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst
                                 | vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eIndexBuffer
                                 | vk::BufferUsageFlagBits::eShaderDeviceAddress;
    if (features.hardwareRayTracing)
        usage |= vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
                 | vk::BufferUsageFlagBits::eShaderBindingTableKHR;
    return usage;
}

std::expected<void, Error> VulkanDevice::createMemoryHeaps() {
    memoryProperties = physicalDevice.getMemoryProperties();
    std::expected<vk::UniqueBuffer, Error> probe =
        checkVulkan(device->createBufferUnique({{}, getBlockBytes(MemoryKind::Readback), getBlockUsage(), getSharing(), sharedFamilies}), "vkCreateBuffer");
    if (!probe)
        return std::unexpected(std::move(probe.error()));
    const std::uint32_t typeBits = device->getBufferMemoryRequirements(probe->get()).memoryTypeBits;
    for (const MemoryKind kind : bufferKinds) {
        const std::optional<std::uint32_t> typeIndex = findMemoryType(memoryProperties, typeBits, kind);
        if (!typeIndex)
            return fail(ErrorCode::Unsupported, "the device has no memory type for buffers of this kind");
        heaps.push_back({kind, *typeIndex, {}});
    }
    return {};
}

std::expected<std::uint32_t, Error> VulkanDevice::addBlock(const std::uint32_t heapIndex, const vk::DeviceSize size) {
    MemoryHeap& heap = heaps[heapIndex];
    const bool hasBuffer = heap.kind != MemoryKind::Image;
    const bool mappable = heap.kind == MemoryKind::Upload || heap.kind == MemoryKind::Readback;

    vk::UniqueBuffer buffer;
    if (hasBuffer) {
        std::expected<vk::UniqueBuffer, Error> created =
            checkVulkan(device->createBufferUnique({{}, size, getBlockUsage(), getSharing(), sharedFamilies}), "vkCreateBuffer");
        if (!created)
            return std::unexpected(std::move(created.error()));
        buffer = std::move(*created);
    }
    const vk::MemoryAllocateFlagsInfo flagsInfo(hasBuffer ? vk::MemoryAllocateFlagBits::eDeviceAddress : vk::MemoryAllocateFlags{});
    std::expected<vk::UniqueDeviceMemory, Error> memory = checkVulkan(device->allocateMemoryUnique({size, heap.typeIndex, &flagsInfo}), "vkAllocateMemory");
    if (!memory)
        return std::unexpected(std::move(memory.error()));

    MemoryBlock block;
    block.ranges = RangeAllocator(size);
    if (hasBuffer) {
        if (std::expected<void, Error> bound = checkVulkan(device->bindBufferMemory(*buffer, memory->get(), 0), "vkBindBufferMemory"); !bound)
            return std::unexpected(std::move(bound.error()));
        block.address = device->getBufferAddress({*buffer});
    }
    if (mappable) {
        std::expected<void*, Error> mapped = checkVulkan(device->mapMemory(memory->get(), 0, vk::WholeSize), "vkMapMemory");
        if (!mapped)
            return std::unexpected(std::move(mapped.error()));
        block.mapped = static_cast<std::byte*>(*mapped);
    }
    block.buffer = buffer.release();
    block.memory = memory->release();

    for (std::uint32_t index = 0; index < heap.blocks.size(); ++index)
        if (!heap.blocks[index].memory) {
            heap.blocks[index] = std::move(block);
            return index;
        }
    heap.blocks.push_back(std::move(block));
    return static_cast<std::uint32_t>(heap.blocks.size() - 1);
}

std::expected<Allocation, Error> VulkanDevice::allocate(std::uint32_t heapIndex, const vk::DeviceSize size, const vk::DeviceSize alignment) {
    std::lock_guard lock(memoryMutex);
    std::optional<std::uint64_t> offset;
    std::uint32_t blockIndex = 0;
    for (; blockIndex < heaps[heapIndex].blocks.size() && !offset; ++blockIndex)
        if (heaps[heapIndex].blocks[blockIndex].memory)
            offset = heaps[heapIndex].blocks[blockIndex].ranges.allocate(size, alignment);
    if (offset)
        --blockIndex;
    if (!offset) {
        const vk::DeviceSize blockBytes = std::max(getBlockBytes(heaps[heapIndex].kind), alignUp(size, 1ull << 20));
        std::expected<std::uint32_t, Error> added = addBlock(heapIndex, blockBytes);
        if (!added)
            return std::unexpected(std::move(added.error()));
        blockIndex = *added;
        offset = heaps[heapIndex].blocks[blockIndex].ranges.allocate(size, alignment);
    }
    const MemoryBlock& block = heaps[heapIndex].blocks[blockIndex];
    return Allocation{
        .buffer = block.buffer,
        .memory = block.memory,
        .offset = *offset,
        .address = block.address ? block.address + *offset : 0,
        .size = size,
        .mapped = block.mapped ? block.mapped + *offset : nullptr,
        .heap = heapIndex,
        .block = blockIndex,
    };
}

std::expected<Allocation, Error> VulkanDevice::allocate(const std::size_t size, const MemoryKind kind, const std::size_t alignment) {
    if (size == 0)
        return fail(ErrorCode::InvalidArgument, "an allocation needs a size");
    return allocate(static_cast<std::uint32_t>(kind), size, alignment);
}

std::expected<Allocation, Error> VulkanDevice::allocate(const vk::MemoryRequirements& requirements) {
    const std::optional<std::uint32_t> typeIndex = findMemoryType(memoryProperties, requirements.memoryTypeBits, MemoryKind::Image);
    if (!typeIndex)
        return fail(ErrorCode::Unsupported, "the device has no memory type for this image");
    std::uint32_t heapIndex = 0;
    {
        std::lock_guard lock(memoryMutex);
        heapIndex = static_cast<std::uint32_t>(
            std::ranges::find_if(heaps, [&](const MemoryHeap& heap) { return heap.kind == MemoryKind::Image && heap.typeIndex == *typeIndex; })
            - heaps.begin());
        if (heapIndex == heaps.size())
            heaps.push_back({MemoryKind::Image, *typeIndex, {}});
    }
    return allocate(heapIndex, requirements.size, requirements.alignment);
}

void VulkanDevice::release(const Allocation& allocation) {
    if (!allocation.memory)
        return;
    std::lock_guard lock(memoryMutex);
    MemoryBlock& block = heaps[allocation.heap].blocks[allocation.block];
    block.ranges.release(allocation.offset, allocation.size);
    if (allocation.block == 0 || !block.ranges.isEmpty())
        return;
    device->destroyBuffer(block.buffer);
    device->freeMemory(block.memory);
    block = {};
}

void VulkanDevice::releaseMemoryHeaps() {
    for (MemoryHeap& heap : heaps)
        for (MemoryBlock& block : heap.blocks) {
            device->destroyBuffer(block.buffer);
            device->freeMemory(block.memory);
        }
    heaps.clear();
}

std::optional<BufferRange> VulkanDevice::findBuffer(const vk::DeviceAddress address) {
    std::lock_guard lock(memoryMutex);
    for (const MemoryKind kind : bufferKinds)
        for (const MemoryBlock& block : heaps[static_cast<std::size_t>(kind)].blocks)
            if (block.buffer && address >= block.address && address < block.address + block.ranges.getSize())
                return BufferRange{block.buffer, address - block.address, block.ranges.getSize() - (address - block.address)};
    return std::nullopt;
}

}
