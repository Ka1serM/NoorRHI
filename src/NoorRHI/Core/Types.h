#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace NoorRHI {

enum class GraphicsApi : std::uint8_t { Vulkan, WebGpu };

using uint = std::uint32_t;

//! The GPU reads this as two 32-bit words, so it has the same layout on the CPU, on Vulkan and on WebGPU.
template <class T>
struct Ptr {
    std::uint32_t low = 0;
    std::uint32_t high = 0;

    constexpr Ptr() noexcept = default;
    constexpr explicit Ptr(const std::uint64_t address) noexcept : low(static_cast<std::uint32_t>(address)), high(static_cast<std::uint32_t>(address >> 32)) {}

    constexpr std::uint64_t getAddress() const noexcept { return std::uint64_t{high} << 32 | low; }
    constexpr explicit operator bool() const noexcept { return getAddress() != 0; }
    constexpr Ptr operator+(const std::size_t count) const noexcept { return Ptr(getAddress() + count * sizeof(T)); }
    template <class U>
    constexpr Ptr<U> as() const noexcept {
        return Ptr<U>(getAddress());
    }
};

static_assert(sizeof(Ptr<float>) == 8 && alignof(Ptr<float>) == 4);

enum class BufferMemory : std::uint8_t { Plain, Atomic };

template <class T>
concept AtomicValue = std::is_same_v<T, std::uint32_t> || std::is_same_v<T, float>;

template <AtomicValue T>
struct Atomic {
    T value{};
};

template <AtomicValue T>
struct AtomicPtr {

    std::uint32_t low = 0;
    std::uint32_t high = 0;

    constexpr AtomicPtr() noexcept = default;
    constexpr explicit AtomicPtr(const std::uint64_t address) noexcept
        : low(static_cast<std::uint32_t>(address)), high(static_cast<std::uint32_t>(address >> 32)) {}

    constexpr std::uint64_t getAddress() const noexcept { return std::uint64_t{high} << 32 | low; }
    constexpr explicit operator bool() const noexcept { return getAddress() != 0; }
    constexpr AtomicPtr operator+(const std::size_t count) const noexcept { return AtomicPtr(getAddress() + count * sizeof(T)); }
};

static_assert(sizeof(AtomicPtr<uint>) == 8 && alignof(AtomicPtr<uint>) == 4);

struct half {
    std::uint16_t bits = 0;
};

struct float2 {
    float x{}, y{};
};
struct float3 {
    float x{}, y{}, z{};
};
struct float4 {
    float x{}, y{}, z{}, w{};
};
struct int2 {
    std::int32_t x{}, y{};
};
struct int3 {
    std::int32_t x{}, y{}, z{};
};
struct int4 {
    std::int32_t x{}, y{}, z{}, w{};
};
struct uint2 {
    std::uint32_t x{}, y{};
};
struct uint3 {
    std::uint32_t x{}, y{}, z{};
};
struct uint4 {
    std::uint32_t x{}, y{}, z{}, w{};
};
struct float2x2 {
    float values[2][2]{};
};
struct float3x3 {
    float values[3][3]{};
};
struct float4x4 {
    float values[4][4]{};
};

enum class Queue : std::uint8_t { Graphics, Async };
inline constexpr std::size_t queueCount = 2;

struct Token {
    std::uint64_t value = 0;
    Queue queue = Queue::Graphics;

    explicit operator bool() const noexcept { return value != 0; }
};

struct Rect {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

struct DispatchSize {
    std::uint32_t x = 1;
    std::uint32_t y = 1;
    std::uint32_t z = 1;
};
struct DrawArgs {
    std::uint32_t vertexCount = 0;
    std::uint32_t instanceCount = 1;
    std::uint32_t firstVertex = 0;
    std::uint32_t firstInstance = 0;
};

struct DrawIndexedArgs {
    std::uint32_t indexCount = 0;
    std::uint32_t instanceCount = 1;
    std::uint32_t firstIndex = 0;
    std::int32_t vertexOffset = 0;
    std::uint32_t firstInstance = 0;
};

struct TextureHandle {
    std::uint32_t value = 0;
    explicit operator bool() const noexcept { return value != 0; }
};

struct StorageTextureHandle {
    std::uint32_t value = 0;
    explicit operator bool() const noexcept { return value != 0; }
};

using UintTextureHandle = TextureHandle;
using UintStorageTextureHandle = StorageTextureHandle;

struct SamplerHandle {
    std::uint32_t value = 0;
    explicit operator bool() const noexcept { return value != 0; }
};

struct AccelerationStructureHandle {
    std::uint32_t low = 0;
    std::uint32_t high = 0;

    constexpr AccelerationStructureHandle() noexcept = default;
    constexpr explicit AccelerationStructureHandle(const std::uint64_t address) noexcept
        : low(static_cast<std::uint32_t>(address)), high(static_cast<std::uint32_t>(address >> 32)) {}

    constexpr std::uint64_t getAddress() const noexcept { return std::uint64_t{high} << 32 | low; }
    constexpr explicit operator bool() const noexcept { return getAddress() != 0; }
};

}
