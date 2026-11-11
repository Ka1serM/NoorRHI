#include <cstring>

#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Resources/Buffer.h"
#include "NoorRHI/WebGpu/WebGpu.h"

namespace NoorRHI::Backend {

std::expected<std::shared_ptr<Buffer>, Error> openBuffer(NoorRHI::Device& device, const std::size_t bytes, const std::size_t alignment,
                                                         const BufferMemory memory) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    auto& owner = static_cast<WebGpuDevice&>(*device.getBackend());
    std::expected<std::uint64_t, Error> offset = owner.allocate(memory, alignUp(bytes, 4), std::max(alignment, argumentAlignment));
    if (!offset)
        return std::unexpected(std::move(offset.error()));
    auto result = std::make_shared<WebGpuBuffer>();
    result->device = owner.shared_from_this();
    result->address = *offset;
    result->size = bytes;
    return result;
}

WebGpuBuffer::~WebGpuBuffer() {
    if (device)
        device->release(address, alignUp(size, 4));
}

std::expected<void, Error> WebGpuBuffer::upload(const std::span<const std::byte> data, const std::size_t offset) {
    const std::uint64_t start = address + offset;
    if (start % 4 == 0 && data.size() % 4 == 0) {
        std::lock_guard lock(device->mutex);
        device->queue.WriteBuffer(device->getBuffer(start), start & addressOffsetMask, data.data(), data.size());
        return {};
    }
    WebGpuCommandList list;
    list.device = device;
    list.upload(address + offset, data);
    std::expected<Token, Error> token = device->submit(list, {});
    if (!token)
        return std::unexpected(std::move(token.error()));
    return {};
}

std::expected<void, Error> WebGpuBuffer::download(const std::span<std::byte> destination, const std::size_t offset) {
    const std::uint64_t start = (address & addressOffsetMask) + offset;
    const std::uint64_t alignedStart = start & ~std::uint64_t{3};
    const std::uint64_t bytes = alignUp(start + destination.size(), 4) - alignedStart;
    return device->download(
        bytes,
        [&](const wgpu::CommandEncoder& encoder, const wgpu::Buffer& readback) {
            encoder.CopyBufferToBuffer(device->getBuffer(address), alignedStart, readback, 0, bytes);
        },
        [&](const std::span<const std::byte> mapped) { std::memcpy(destination.data(), mapped.data() + (start - alignedStart), destination.size()); });
}

}
