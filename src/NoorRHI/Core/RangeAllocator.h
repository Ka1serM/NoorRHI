#pragma once

#include <cstdint>
#include <map>
#include <optional>

namespace NoorRHI {

class RangeAllocator {
public:
    RangeAllocator() = default;
    explicit RangeAllocator(std::uint64_t size);

    std::optional<std::uint64_t> allocate(std::uint64_t size, std::uint64_t alignment);
    void release(std::uint64_t offset, std::uint64_t size);
    void grow(std::uint64_t newSize);
    std::uint64_t getSize() const { return size; }
    bool isEmpty() const { return byOffset.size() == 1 && byOffset.begin()->second == size; }

private:
    void insert(std::uint64_t offset, std::uint64_t length);
    void erase(std::map<std::uint64_t, std::uint64_t>::iterator range);

    std::uint64_t size = 0;
    std::map<std::uint64_t, std::uint64_t> byOffset;
    std::multimap<std::uint64_t, std::uint64_t> bySize;
};

}
