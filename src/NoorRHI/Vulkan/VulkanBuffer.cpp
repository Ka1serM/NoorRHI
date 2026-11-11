#include "NoorRHI/Resources/Buffer.h"
#include "NoorRHI/Vulkan/Vulkan.h"

namespace NoorRHI::Backend {

std::expected<std::shared_ptr<Buffer>, Error> openBuffer(NoorRHI::Device& device, const std::size_t bytes, const std::size_t alignment, BufferMemory) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    auto& owner = static_cast<VulkanDevice&>(*device.getBackend());
    std::expected<Allocation, Error> memory = owner.allocate(bytes, MemoryKind::Device, std::max(alignment, argumentAlignment));
    if (!memory)
        return std::unexpected(std::move(memory.error()));
    auto result = std::make_shared<VulkanBuffer>();
    result->device = owner.shared_from_this();
    result->memory = *memory;
    result->address = memory->address;
    return result;
}

VulkanBuffer::~VulkanBuffer() {
    device->retire([owner = device.get(), allocation = memory] { owner->release(allocation); });
}

std::expected<void, Error> VulkanBuffer::upload(const std::span<const std::byte> data, const std::size_t offset) {
    std::expected<Token, Error> token = device->execute([&](VulkanCommandList& list) {
        std::expected<Placement, Error> staged = list.place(data);
        if (!staged)
            return static_cast<void>(list.check(false, staged.error().code, staged.error().message));
        list.synchronizeCopy({}, std::array{CopyRange{0, memory.address + offset, memory.address + offset + data.size()}});
        list.pool.commands.copyBuffer(staged->buffer, memory.buffer, vk::BufferCopy(staged->offset, memory.offset + offset, data.size()));
    });
    if (!token)
        return std::unexpected(std::move(token.error()));
    return {};
}

std::expected<void, Error> VulkanBuffer::download(const std::span<std::byte> destination, const std::size_t offset) {
    return device->download(destination, [&](const vk::CommandBuffer commands, const Allocation& staging) {
        commands.copyBuffer(memory.buffer, staging.buffer, vk::BufferCopy(memory.offset + offset, staging.offset, destination.size()));
    });
}

std::expected<std::shared_ptr<Readback>, Error> openReadback(NoorRHI::Device& device, const std::size_t bytes) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    auto& owner = static_cast<VulkanDevice&>(*device.getBackend());
    std::expected<Allocation, Error> memory = owner.allocate(bytes, MemoryKind::Readback);
    if (!memory)
        return std::unexpected(std::move(memory.error()));
    auto result = std::make_shared<VulkanReadback>();
    result->device = owner.shared_from_this();
    result->memory = *memory;
    result->size = bytes;
    return result;
}

VulkanReadback::~VulkanReadback() {
    device->retire([owner = device.get(), allocation = memory] { owner->release(allocation); });
}

bool VulkanReadback::isReady() {
    return token && device->isFinished(token);
}

std::expected<std::span<const std::byte>, Error> VulkanReadback::read() {
    if (!token)
        return fail(ErrorCode::InvalidState, "no submitted command list downloads into the readback");
    if (std::expected<void, Error> waited = device->wait(token); !waited)
        return std::unexpected(std::move(waited.error()));
    return std::span<const std::byte>(memory.mapped, size);
}

}
