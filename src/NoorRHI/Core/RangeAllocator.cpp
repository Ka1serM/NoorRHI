#include "NoorRHI/Core/RangeAllocator.h"

namespace NoorRHI {

RangeAllocator::RangeAllocator(const std::uint64_t size) : size(size) {
    if (size > 0)
        insert(0, size);
}

void RangeAllocator::insert(const std::uint64_t offset, const std::uint64_t length) {
    byOffset.emplace(offset, length);
    bySize.emplace(length, offset);
}

void RangeAllocator::erase(const std::map<std::uint64_t, std::uint64_t>::iterator range) {
    auto [first, last] = bySize.equal_range(range->second);
    for (; first != last; ++first)
        if (first->second == range->first) {
            bySize.erase(first);
            break;
        }
    byOffset.erase(range);
}

std::optional<std::uint64_t> RangeAllocator::allocate(const std::uint64_t length, const std::uint64_t alignment) {
    for (auto candidate = bySize.lower_bound(length); candidate != bySize.end(); ++candidate) {
        const auto [freeLength, freeOffset] = *candidate;
        const std::uint64_t aligned = (freeOffset + alignment - 1) / alignment * alignment;
        if (aligned + length > freeOffset + freeLength)
            continue;
        erase(byOffset.find(freeOffset));
        if (aligned > freeOffset)
            insert(freeOffset, aligned - freeOffset);
        if (aligned + length < freeOffset + freeLength)
            insert(aligned + length, freeOffset + freeLength - aligned - length);
        return aligned;
    }
    return std::nullopt;
}

void RangeAllocator::release(std::uint64_t offset, std::uint64_t length) {
    auto next = byOffset.lower_bound(offset);
    if (next != byOffset.begin()) {
        const auto previous = std::prev(next);
        if (previous->first + previous->second == offset) {
            offset = previous->first;
            length += previous->second;
            erase(previous);
        }
    }
    if (next != byOffset.end() && offset + length == next->first) {
        length += next->second;
        erase(next);
    }
    insert(offset, length);
}

void RangeAllocator::grow(const std::uint64_t newSize) {
    const std::uint64_t oldSize = size;
    size = newSize;
    release(oldSize, newSize - oldSize);
}

}
