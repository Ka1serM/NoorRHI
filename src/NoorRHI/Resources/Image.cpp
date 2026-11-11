#include "NoorRHI/Resources/Image.h"

#include <bit>

#include "NoorRHI/Core/Backend.h"

namespace NoorRHI {

std::expected<void, Error> Backend::checkImageDesc(const ImageDesc& desc) {
    const FormatInfo info = getFormatInfo(desc.format);
    const bool attachmentOnly =
        !hasUsage(desc.usage, ImageUsage::Sampled) && !hasUsage(desc.usage, ImageUsage::Storage) && !hasUsage(desc.usage, ImageUsage::ExternalMemory);
    if (desc.width == 0 || desc.height == 0)
        return fail(ErrorCode::InvalidArgument, "an image needs a width and a height");
    if (desc.format == ImageFormat::Undefined)
        return fail(ErrorCode::InvalidArgument, "an image needs a format");
    if (hasUsage(desc.usage, ImageUsage::DepthAttachment) != info.depth)
        return fail(ErrorCode::InvalidArgument, "depth formats need DepthAttachment usage and DepthAttachment usage needs a depth format");
    if (info.stencil && !attachmentOnly)
        return fail(ErrorCode::InvalidArgument, "D32FloatS8Uint images are attachments only");
    if (desc.mipLevels == 0 || desc.mipLevels > static_cast<std::uint32_t>(std::bit_width(std::max(desc.width, desc.height))))
        return fail(ErrorCode::InvalidArgument, "the mip level count must be between one and the full chain of the extent");
    if (desc.mipLevels > 1 && desc.usage != ImageUsage::Sampled)
        return fail(ErrorCode::InvalidArgument, "only sampled-only images have mip chains");
    if (info.blockExtent > 1 && desc.usage != ImageUsage::Sampled)
        return fail(ErrorCode::InvalidArgument, "block-compressed images are sampled-only");
    if (desc.samples != 1 && desc.samples != 4)
        return fail(ErrorCode::InvalidArgument, "an image has 1 or 4 samples");
    if (desc.samples > 1 && (!attachmentOnly || desc.mipLevels > 1 || info.integer))
        return fail(ErrorCode::InvalidArgument, "multisampled images are single-level attachments of a non-integer format");
    return {};
}

std::expected<void, Error> Image::upload(const std::span<const std::byte> mipChain) {
    if (!backend || mipChain.size() != getByteSize())
        return fail(ErrorCode::InvalidArgument, "the upload must hold the whole mip chain of the image");
    if (samples > 1 || getFormatInfo(format).depth)
        return fail(ErrorCode::InvalidArgument, "multisampled and depth images take no uploads");
    return backend->upload(mipChain);
}

std::expected<void, Error> Image::download(const std::span<std::byte> destination) const {
    return download(destination, {.width = width, .height = height});
}

std::expected<void, Error> Image::download(const std::span<std::byte> destination, const Rect region) const {
    if (!backend || region.x < 0 || region.y < 0)
        return fail(ErrorCode::InvalidArgument, "the download region exceeds the image");
    const auto x = static_cast<std::uint32_t>(region.x);
    const auto y = static_cast<std::uint32_t>(region.y);
    const std::uint32_t regionWidth = region.width;
    const std::uint32_t regionHeight = region.height;
    if (regionWidth == 0 || regionHeight == 0 || x >= width || y >= height || regionWidth > width - x || regionHeight > height - y)
        return fail(ErrorCode::InvalidArgument, "the download region exceeds the image");
    if (samples > 1)
        return fail(ErrorCode::InvalidArgument, "multisampled images cannot be downloaded");
    if (getFormatInfo(format).blockExtent > 1 && (x != 0 || y != 0 || regionWidth != width || regionHeight != height))
        return fail(ErrorCode::InvalidArgument, "block-compressed images download only as a whole level");
    if (destination.size() != getLevelByteSize(format, regionWidth, regionHeight))
        return fail(ErrorCode::InvalidArgument, "the destination must match the size of the region");
    return backend->download(x, y, regionWidth, regionHeight, destination);
}

}
