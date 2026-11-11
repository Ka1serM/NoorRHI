#pragma once

#include <expected>
#include <span>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Core/Error.h"
#include "NoorRHI/Resources/Buffer.h"

namespace NoorRHI {

template <class T>
class Shared {
public:
    [[nodiscard]] std::expected<void, Error> open(Device& device) { return buffer.open(device, 1); }

    const T& getData() const { return data; }
    T& getData() { return data; }
    Ptr<T> getPtr() const { return buffer.getPtr(); }
    explicit operator bool() const { return static_cast<bool>(buffer); }

    [[nodiscard]] std::expected<void, Error> commit() { return buffer.upload(std::span<const T>(&data, 1)); }
    void commit(CommandList& list) const { list.upload(buffer, std::span<const T>(&data, 1)); }

private:
    T data{};
    Buffer<T> buffer;
};

}
