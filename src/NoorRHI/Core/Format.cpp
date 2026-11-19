#include "NoorRHI/Core/Format.h"

#include <algorithm>

namespace NoorRHI {

FormatInfo getFormatInfo(const ImageFormat format) {
    switch (format) {
    case ImageFormat::Rgba32Float:
        return {1, 16};
    case ImageFormat::Rgba16Float:
    case ImageFormat::Rgba16Unorm:
    case ImageFormat::Rg32Float:
        return {1, 8};
    case ImageFormat::R32Uint:
        return {1, 4, false, true};
    case ImageFormat::D32Float:
        return {1, 4, true};
    case ImageFormat::D32FloatS8Uint:
        return {1, 4, true, false, true};
    case ImageFormat::R8Unorm:
    case ImageFormat::L8Unorm:
    case ImageFormat::L8Srgb:
        return {1, 1};
    case ImageFormat::Rg8Unorm:
    case ImageFormat::R16Unorm:
    case ImageFormat::R16Float:
        return {1, 2};
    case ImageFormat::Bc1Unorm:
    case ImageFormat::Bc1Srgb:
    case ImageFormat::Bc4Unorm:
        return {4, 8};
    case ImageFormat::Bc2Unorm:
    case ImageFormat::Bc2Srgb:
    case ImageFormat::Bc3Unorm:
    case ImageFormat::Bc3Srgb:
    case ImageFormat::Bc5Unorm:
    case ImageFormat::Bc6hUfloat:
    case ImageFormat::Bc6hSfloat:
    case ImageFormat::Bc7Unorm:
    case ImageFormat::Bc7Srgb:
        return {4, 16};
    default:
        return {1, 4};
    }
}

std::size_t getLevelByteSize(const ImageFormat format, const std::uint32_t width, const std::uint32_t height) {
    const FormatInfo info = getFormatInfo(format);
    const std::size_t blocksX = (std::size_t{width} + info.blockExtent - 1) / info.blockExtent;
    const std::size_t blocksY = (std::size_t{height} + info.blockExtent - 1) / info.blockExtent;
    return blocksX * blocksY * info.blockBytes;
}

std::uint32_t getCopyRowPitch(const ImageFormat format, const std::uint32_t width) {
    const FormatInfo info = getFormatInfo(format);
    const std::uint32_t rowBytes = (width + info.blockExtent - 1) / info.blockExtent * info.blockBytes;
    return (rowBytes + 255) / 256 * 256;
}

std::size_t getCopyLevelByteSize(const ImageFormat format, const std::uint32_t width, const std::uint32_t height) {
    const FormatInfo info = getFormatInfo(format);
    return std::size_t{getCopyRowPitch(format, width)} * ((height + info.blockExtent - 1) / info.blockExtent);
}

std::size_t getImageByteSize(const ImageFormat format, const std::uint32_t width, const std::uint32_t height, const std::uint32_t mipLevels) {
    std::size_t total = 0;
    for (std::uint32_t level = 0; level < mipLevels; level++)
        total += getLevelByteSize(format, std::max(1u, width >> level), std::max(1u, height >> level));
    return total;
}

}
