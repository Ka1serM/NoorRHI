#include "NoorRHI/Resources/Buffer.h"

#include "NoorRHI/Core/Backend.h"

namespace NoorRHI::Backend {

std::uint64_t getAddress(const Buffer& buffer) {
    return buffer.address;
}

std::expected<void, Error> uploadBuffer(Buffer& buffer, const void* data, const std::size_t bytes, const std::size_t offset) {
    return buffer.upload(std::span(static_cast<const std::byte*>(data), bytes), offset);
}

std::expected<void, Error> downloadBuffer(const Buffer& buffer, void* data, const std::size_t bytes, const std::size_t offset) {
    return const_cast<Buffer&>(buffer).download(std::span(static_cast<std::byte*>(data), bytes), offset);
}

bool isReadbackReady(Readback& readback) {
    return readback.isReady();
}

std::expected<std::span<const std::byte>, Error> readReadback(Readback& readback) {
    return readback.read();
}

}
