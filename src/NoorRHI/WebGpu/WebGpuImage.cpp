#include <bit>
#include <cstring>

#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Resources/Image.h"
#include "NoorRHI/WebGpu/WebGpu.h"

namespace NoorRHI {

namespace {

std::uint32_t storagePoolBindingFor(const Backend::StorageKind kind) {
    return Backend::storagePoolBindings.first + static_cast<std::uint32_t>(kind);
}

std::optional<std::uint32_t> getStorageBinding(const ImageFormat format) {
    switch (format) {
    case ImageFormat::Rgba8Unorm:
        return storagePoolBindingFor(Backend::StorageKind::Rgba8Unorm);
    case ImageFormat::Rgba16Float:
        return storagePoolBindingFor(Backend::StorageKind::Rgba16Float);
    case ImageFormat::R32Float:
        return storagePoolBindingFor(Backend::StorageKind::R32Float);
    case ImageFormat::R32Uint:
        return storagePoolBindingFor(Backend::StorageKind::R32Uint);
    case ImageFormat::Rgba32Float:
        return storagePoolBindingFor(Backend::StorageKind::Rgba32Float);
    case ImageFormat::R16Float:
        return storagePoolBindingFor(Backend::StorageKind::R16Float);
    case ImageFormat::R8Unorm:
        return storagePoolBindingFor(Backend::StorageKind::R8Unorm);
    default:
        return std::nullopt;
    }
}

bool isCompressed(const ImageFormat format) {
    return getFormatInfo(format).blockExtent > 1;
}

std::expected<void, Error> checkSupport(const Backend::WebGpuDevice& device, const ImageDesc& desc) {
    const ImageFormat format = desc.format;
    if (hasUsage(desc.usage, ImageUsage::ExternalMemory))
        return fail(ErrorCode::Unsupported, "WebGPU images cannot export memory");
    if (format == ImageFormat::L8Srgb)
        return fail(ErrorCode::Unsupported, "WebGPU has no single-channel sRGB format for L8Srgb");
    if (format == ImageFormat::L8Unorm && hasUsage(desc.usage, ImageUsage::Sampled) && !device.componentSwizzle)
        return fail(ErrorCode::Unsupported, "a sampled L8Unorm image needs the WebGPU texture-component-swizzle feature");
    if (getFormatInfo(format).stencil && !device.depthStencil)
        return fail(ErrorCode::Unsupported, "D32FloatS8Uint images need the WebGPU depth32float-stencil8 feature");
    if (isCompressed(format) && !device.compressionBc)
        return fail(ErrorCode::Unsupported, "the WebGPU device has no texture-compression-bc");
    if (isCompressed(format) && (desc.width % 4 != 0 || desc.height % 4 != 0))
        return fail(ErrorCode::Unsupported, "WebGPU block-compressed images need a width and a height that are multiples of 4");
    const bool tier1 = format == ImageFormat::R16Unorm || format == ImageFormat::Rg16Unorm || format == ImageFormat::Rgba16Unorm;
    if (tier1 && !device.formatsTier1)
        return fail(ErrorCode::Unsupported, "16-bit unorm images need the WebGPU texture-formats-tier1 feature");
    if (hasUsage(desc.usage, ImageUsage::ColorAttachment) && !device.isRenderable(format))
        return fail(ErrorCode::Unsupported, "the WebGPU device cannot render to this image format");
    if (hasUsage(desc.usage, ImageUsage::Storage) && !device.storageImages)
        return fail(ErrorCode::Unsupported, "WebGPU storage images need the texture-formats-tier2 feature and 7 storage textures per shader stage");
    if (hasUsage(desc.usage, ImageUsage::Storage) && !getStorageBinding(format))
        return fail(ErrorCode::Unsupported, "WebGPU storage images use Rgba8Unorm, Rgba16Float, R32Float, R32Uint, Rgba32Float, R16Float or R8Unorm");
    return {};
}

}

namespace Backend {

WebGpuImage::~WebGpuImage() {
    if (!device)
        return;
    std::lock_guard lock(device->mutex);
    device->shadowedImages.erase(this);
    device->retire([owner = device.get(), sampled = this->sampled, binding = storageBinding, layer = storageLayer] {
        if (sampled.binding != 0)
            owner->releaseSampled(sampled);
        if (binding != 0)
            owner->releaseStorageLayer(binding, layer);
    });
}

}

std::expected<void, Error> Image::open(Device& device, const ImageDesc& desc) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    const bool depth = hasUsage(desc.usage, ImageUsage::DepthAttachment);
    if (std::expected<void, Error> checked = Backend::checkImageDesc(desc); !checked)
        return checked;
    auto& owner = static_cast<Backend::WebGpuDevice&>(*device.getBackend());
    if (std::expected<void, Error> supported = checkSupport(owner, desc); !supported)
        return supported;
    const std::uint32_t limit = owner.limits.maxTextureDimension2D;
    if (desc.width > limit || desc.height > limit)
        return fail(ErrorCode::Unsupported, "the image is larger than the WebGPU device allows");

    auto result = std::make_shared<Backend::WebGpuImage>();
    result->device = owner.shared_from_this();
    result->format = desc.format;
    result->nativeFormat = Backend::getWebGpuFormat(desc.format);
    result->width = desc.width;
    result->height = desc.height;
    result->mipLevels = desc.mipLevels;
    result->samples = desc.samples;

    const bool storage = hasUsage(desc.usage, ImageUsage::Storage);
    const bool sampled = hasUsage(desc.usage, ImageUsage::Sampled);
    const bool attachment = hasUsage(desc.usage, ImageUsage::ColorAttachment) || depth;
    if (storage) {
        std::expected<std::uint32_t, Error> layer = owner.allocateStorageLayer(*getStorageBinding(desc.format), desc.width, desc.height);
        if (!layer)
            return std::unexpected(std::move(layer.error()));
        result->storageBinding = *getStorageBinding(desc.format);
        result->storageLayer = *layer;
    }
    if (!storage && (attachment || !sampled)) {
        wgpu::TextureUsage usage =
            desc.samples > 1 ? wgpu::TextureUsage::None : wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst;
        if (attachment)
            usage |= wgpu::TextureUsage::RenderAttachment;
        const wgpu::TextureDescriptor descriptor{.label = wgpu::StringView(desc.name.data(), desc.name.size()),
                                                 .usage = usage,
                                                 .size = {desc.width, desc.height, 1},
                                                 .format = result->nativeFormat,
                                                 .mipLevelCount = desc.mipLevels,
                                                 .sampleCount = desc.samples};
        std::lock_guard lock(owner.mutex);
        owner.pushErrorScope();
        result->texture = owner.device.CreateTexture(&descriptor);
        if (std::expected<void, Error> created = owner.popErrorScope("creating a WebGPU image"); !created)
            return created;
    }
    if (sampled) {
        std::expected<Backend::SampledSlot, Error> slot = owner.allocateSampled(desc.format, desc.width, desc.height, desc.mipLevels);
        if (!slot)
            return std::unexpected(std::move(slot.error()));
        result->sampled = *slot;
    }
    if (result->isShadowed()) {
        std::lock_guard lock(owner.mutex);
        owner.shadowedImages.insert(result.get());
    }

    sampledHandle = {result->sampled.index};
    storageHandle = {result->storageBinding != 0 ? result->storageBinding << Backend::layerBits | result->storageLayer : 0};
    backend = std::move(result);
    width = desc.width;
    height = desc.height;
    mipLevels = desc.mipLevels;
    samples = desc.samples;
    format = desc.format;
    usage = desc.usage;
    return {};
}

namespace Backend {

std::expected<void, Error> WebGpuImage::upload(const std::span<const std::byte> mipChain) {
    const FormatInfo info = getFormatInfo(format);
    std::lock_guard lock(device->mutex);
    std::vector<TextureRegion> regions{device->getPrimary(*this)};
    if (isShadowed())
        regions.push_back(device->getSampled(*this));
    const std::uint32_t levels = sampled.binding != 0 ? sampled.levels : mipLevels;
    for (const TextureRegion& region : regions) {
        std::size_t offset = 0;
        for (std::uint32_t level = 0; level < levels; ++level) {
            const std::uint32_t levelWidth = std::max(width >> level, 1u);
            const std::uint32_t levelHeight = std::max(height >> level, 1u);
            const std::uint32_t columns = (levelWidth + info.blockExtent - 1) / info.blockExtent;
            const std::uint32_t rows = (levelHeight + info.blockExtent - 1) / info.blockExtent;
            const std::size_t bytes = getLevelByteSize(format, levelWidth, levelHeight);
            const wgpu::TexelCopyTextureInfo destination{
                .texture = region.texture, .mipLevel = level, .origin = {region.x >> level, region.y >> level, region.layer}};
            const wgpu::TexelCopyBufferLayout layout{.offset = 0, .bytesPerRow = columns * info.blockBytes, .rowsPerImage = rows};
            const wgpu::Extent3D extent{columns * info.blockExtent, rows * info.blockExtent, 1};
            device->queue.WriteTexture(&destination, mipChain.data() + offset, bytes, &layout, &extent);
            offset += bytes;
        }
    }
    return device->takeError();
}

std::expected<void, Error> WebGpuImage::download(const std::uint32_t x, const std::uint32_t y, const std::uint32_t regionWidth,
                                                 const std::uint32_t regionHeight, const std::span<std::byte> destination) {
    const FormatInfo info = getFormatInfo(format);
    const std::uint32_t columns = (regionWidth + info.blockExtent - 1) / info.blockExtent;
    const std::uint32_t rows = (regionHeight + info.blockExtent - 1) / info.blockExtent;
    const std::uint32_t rowBytes = columns * info.blockBytes;
    const std::uint32_t pitch = static_cast<std::uint32_t>(alignUp(rowBytes, 256));
    return device->download(
        std::uint64_t{pitch} * rows,
        [&](const wgpu::CommandEncoder& encoder, const wgpu::Buffer& readback) {
            const TextureRegion region = device->getPrimary(*this);
            const wgpu::TexelCopyTextureInfo source{.texture = region.texture,
                                                    .origin = {region.x + x, region.y + y, region.layer},
                                                    .aspect = info.depth ? wgpu::TextureAspect::DepthOnly : wgpu::TextureAspect::All};
            const wgpu::TexelCopyBufferInfo target{.layout = {.offset = 0, .bytesPerRow = pitch, .rowsPerImage = rows}, .buffer = readback};
            const wgpu::Extent3D extent{columns * info.blockExtent, rows * info.blockExtent, 1};
            encoder.CopyTextureToBuffer(&source, &target, &extent);
        },
        [&](const std::span<const std::byte> bytes) {
            for (std::uint32_t row = 0; row < rows; ++row)
                std::memcpy(destination.data() + std::size_t{row} * rowBytes, bytes.data() + std::size_t{row} * pitch, rowBytes);
        });
}

}

}
