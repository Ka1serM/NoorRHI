#pragma once

#include <cstddef>
#include <cstdint>

namespace NoorRHI {

enum class ImageFormat : std::uint8_t {
    Undefined,
    Rgba8Unorm,
    Bgra8Unorm,
    Rgba32Float,
    Rgba16Float,
    R32Uint,
    R32Float,
    D32Float,
    D32FloatS8Uint,
    R8Unorm,
    L8Unorm,
    L8Srgb,
    Rg8Unorm,
    Rgba8Srgb,
    Bgra8Srgb,
    R16Unorm,
    Rg16Unorm,
    Rgba16Unorm,
    R16Float,
    Rg16Float,
    Rg32Float,
    A2b10g10r10Unorm,
    B10g11r11Float,
    Bc1Unorm,
    Bc1Srgb,
    Bc2Unorm,
    Bc2Srgb,
    Bc3Unorm,
    Bc3Srgb,
    Bc4Unorm,
    Bc5Unorm,
    Bc6hUfloat,
    Bc6hSfloat,
    Bc7Unorm,
    Bc7Srgb,
};

enum class ImageUsage : std::uint32_t {
    None = 0,
    Sampled = 1u << 0,
    Storage = 1u << 1,
    ColorAttachment = 1u << 2,
    DepthAttachment = 1u << 3,
    ExternalMemory = 1u << 4,
};

constexpr ImageUsage operator|(const ImageUsage a, const ImageUsage b) {
    return static_cast<ImageUsage>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}
constexpr bool hasUsage(const ImageUsage usage, const ImageUsage flag) {
    return (static_cast<std::uint32_t>(usage) & static_cast<std::uint32_t>(flag)) != 0;
}

struct FormatInfo {
    std::uint32_t blockExtent = 1;
    std::uint32_t blockBytes = 4;
    bool depth = false;
    bool integer = false;
    bool stencil = false;
};

FormatInfo getFormatInfo(ImageFormat format);
std::size_t getLevelByteSize(ImageFormat format, std::uint32_t width, std::uint32_t height);
std::size_t getImageByteSize(ImageFormat format, std::uint32_t width, std::uint32_t height, std::uint32_t mipLevels);
//! WebGPU copies from a buffer to an image only with rows of a multiple of 256 bytes.
std::uint32_t getCopyRowPitch(ImageFormat format, std::uint32_t width);
std::size_t getCopyLevelByteSize(ImageFormat format, std::uint32_t width, std::uint32_t height);

}
