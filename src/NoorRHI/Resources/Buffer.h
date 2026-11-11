#pragma once

#include <cstddef>
#include <expected>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>

#include "NoorRHI/Core/Error.h"
#include "NoorRHI/Core/Types.h"

namespace NoorRHI {

class Device;
namespace Backend {
struct Buffer;
std::expected<std::shared_ptr<Buffer>, Error> openBuffer(NoorRHI::Device& device, std::size_t bytes, std::size_t alignment, BufferMemory memory);
std::uint64_t getAddress(const Buffer& buffer);
std::expected<void, Error> uploadBuffer(Buffer& buffer, const void* data, std::size_t bytes, std::size_t offset);
std::expected<void, Error> downloadBuffer(const Buffer& buffer, void* data, std::size_t bytes, std::size_t offset);
}

template <class T>
struct BufferElement {
    using Pointer = Ptr<T>;
    static constexpr BufferMemory memory = BufferMemory::Plain;
};

template <class T>
struct BufferElement<Atomic<T>> {
    using Pointer = AtomicPtr<T>;
    static constexpr BufferMemory memory = BufferMemory::Atomic;
};

template <class T>
class Buffer {
    static_assert(std::is_trivially_copyable_v<T>);

public:
    [[nodiscard]] std::expected<void, Error> open(Device& device, const std::size_t elementCount) {
        if (elementCount == 0 || elementCount > std::numeric_limits<std::size_t>::max() / sizeof(T))
            return fail(ErrorCode::InvalidArgument, "a buffer needs between one element and the size of the address space");
        std::expected<std::shared_ptr<Backend::Buffer>, Error> opened =
            Backend::openBuffer(device, elementCount * sizeof(T), alignof(T), BufferElement<T>::memory);
        if (!opened)
            return std::unexpected(std::move(opened.error()));
        backend = std::move(*opened);
        address = Backend::getAddress(*backend);
        count = elementCount;
        return {};
    }

    BufferElement<T>::Pointer getPtr() const { return typename BufferElement<T>::Pointer(address); }
    std::size_t getSize() const { return count; }
    std::size_t getByteSize() const { return count * sizeof(T); }
    Backend::Buffer* getBackend() const { return backend.get(); }
    explicit operator bool() const { return backend != nullptr; }

    [[nodiscard]] std::expected<void, Error> upload(const std::span<const T> data, const std::size_t offset = 0) {
        if (!backend || offset > count || data.size() > count - offset)
            return fail(ErrorCode::InvalidArgument, "the upload range exceeds the buffer");
        if (data.empty())
            return {};
        return Backend::uploadBuffer(*backend, data.data(), data.size_bytes(), offset * sizeof(T));
    }

    [[nodiscard]] std::expected<void, Error> download(const std::span<T> destination, const std::size_t offset = 0) const {
        if (!backend || offset > count || destination.size() > count - offset)
            return fail(ErrorCode::InvalidArgument, "the download range exceeds the buffer");
        if (destination.empty())
            return {};
        return Backend::downloadBuffer(*backend, destination.data(), destination.size_bytes(), offset * sizeof(T));
    }

private:
    std::shared_ptr<Backend::Buffer> backend;
    std::uint64_t address = 0;
    std::size_t count = 0;
};

}
