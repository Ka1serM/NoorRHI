#pragma once

#include <cstddef>
#include <expected>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>

#include "NoorRHI/Core/Error.h"

namespace NoorRHI {

class Device;
namespace Backend {
struct Readback;
std::expected<std::shared_ptr<Readback>, Error> openReadback(NoorRHI::Device& device, std::size_t bytes);
bool isReadbackReady(Readback& readback);
std::expected<std::span<const std::byte>, Error> readReadback(Readback& readback);
}

//! CommandList::download copies into it, and read() is free of waits after isReady().
template <class T>
class Readback {
    static_assert(std::is_trivially_copyable_v<T>);

public:
    [[nodiscard]] std::expected<void, Error> open(Device& device, const std::size_t elementCount) {
        if (elementCount == 0 || elementCount > std::numeric_limits<std::size_t>::max() / sizeof(T))
            return fail(ErrorCode::InvalidArgument, "a readback needs between one element and the size of the address space");
        std::expected<std::shared_ptr<Backend::Readback>, Error> opened = Backend::openReadback(device, elementCount * sizeof(T));
        if (!opened)
            return std::unexpected(std::move(opened.error()));
        backend = std::move(*opened);
        count = elementCount;
        return {};
    }

    bool isReady() const { return backend && Backend::isReadbackReady(*backend); }

    [[nodiscard]] std::expected<std::span<const T>, Error> read() const {
        if (!backend)
            return fail(ErrorCode::InvalidState, "the readback is not open");
        std::expected<std::span<const std::byte>, Error> bytes = Backend::readReadback(*backend);
        if (!bytes)
            return std::unexpected(std::move(bytes.error()));
        return std::span(reinterpret_cast<const T*>(bytes->data()), count);
    }

    std::size_t getSize() const { return count; }
    Backend::Readback* getBackend() const { return backend.get(); }
    explicit operator bool() const { return backend != nullptr; }

private:
    std::shared_ptr<Backend::Readback> backend;
    std::size_t count = 0;
};

}
