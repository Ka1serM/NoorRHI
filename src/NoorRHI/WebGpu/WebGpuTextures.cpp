#include <algorithm>
#include <bit>
#include <ranges>

#include "NoorRHI/WebGpu/WebGpu.h"

namespace NoorRHI::Backend {

namespace {

constexpr std::uint32_t initialLayers = 4;
constexpr std::uint64_t atlasLayerBytes = 4ull << 20;
constexpr std::uint32_t minimumAtlasSize = 256;
constexpr std::uint32_t minimumNode = 4;

PoolKind getPoolKind(const ImageFormat format, const bool float32Filterable) {
    if (getFormatInfo(format).integer)
        return PoolKind::Uint;
    if (getFormatInfo(format).depth)
        return PoolKind::Unfilterable;
    const bool float32 = format == ImageFormat::R32Float || format == ImageFormat::Rg32Float || format == ImageFormat::Rgba32Float;
    return float32 && !float32Filterable ? PoolKind::Unfilterable : PoolKind::Float;
}

std::uint32_t getPhysicalExtent(const ImageFormat format, const std::uint32_t extent, const std::uint32_t level) {
    return static_cast<std::uint32_t>(alignUp(std::max(extent >> level, 1u), getFormatInfo(format).blockExtent));
}

std::uint32_t getAtlasSize(const ImageFormat format, const std::uint32_t limit) {
    const FormatInfo info = getFormatInfo(format);
    std::uint32_t size = minimumAtlasSize;
    while (size * 2 <= limit && std::uint64_t{size} * 2 * size * 2 * info.blockBytes / (info.blockExtent * info.blockExtent) <= atlasLayerBytes)
        size *= 2;
    return std::min(size, limit);
}

std::pair<std::uint32_t, std::uint32_t> getNodeSize(const SampledPool& pool, const std::uint32_t depth) {
    if (!pool.atlas)
        return {pool.width, pool.height};
    const std::uint32_t side = pool.width >> (depth / 2);
    return {depth % 2 == 0 ? side : side / 2, side};
}

std::uint64_t makeNode(const std::uint32_t layer, const std::uint32_t x, const std::uint32_t y) {
    return std::uint64_t{layer} << 32 | std::uint64_t{y} << 16 | x;
}

std::uint32_t getNodeX(const std::uint64_t node) {
    return static_cast<std::uint32_t>(node & 0xffff);
}

std::uint32_t getNodeY(const std::uint64_t node) {
    return static_cast<std::uint32_t>((node >> 16) & 0xffff);
}

std::uint32_t getNodeLayer(const std::uint64_t node) {
    return static_cast<std::uint32_t>(node >> 32);
}

bool isSameSampler(const SamplerDesc& a, const SamplerDesc& b) {
    return a.filter == b.filter && a.maxAnisotropy == b.maxAnisotropy;
}

void copyLayers(const wgpu::CommandEncoder& encoder, const wgpu::Texture& source, const wgpu::Texture& destination, const std::uint32_t mipLevel,
                const wgpu::Extent3D& size) {
    const wgpu::TexelCopyTextureInfo from{.texture = source, .mipLevel = mipLevel};
    const wgpu::TexelCopyTextureInfo to{.texture = destination, .mipLevel = mipLevel};
    encoder.CopyTextureToTexture(&from, &to, &size);
}

}

std::expected<void, Error> WebGpuDevice::growSampledPool(SampledPool& pool) {
    const std::uint32_t limit = std::min(limits.maxTextureArrayLayers, (1u << 24) - 1);
    if (pool.capacity >= limit)
        return fail(ErrorCode::OutOfMemory, "a WebGPU texture pool holds at most " + std::to_string(limit) + " layers");
    const std::uint32_t capacity = std::min(pool.capacity == 0 ? (pool.atlas ? 1 : initialLayers) : pool.capacity * 2, limit);
    wgpu::TextureUsage usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::CopySrc;
    if (isRenderable(pool.format))
        usage |= wgpu::TextureUsage::RenderAttachment;
    pushErrorScope();
    const wgpu::TextureDescriptor descriptor{.label = "NoorRHI sampled pool",
                                             .usage = usage,
                                             .size = {pool.width, pool.height, capacity},
                                             .format = getWebGpuFormat(pool.format),
                                             .mipLevelCount = pool.mipLevels};
    wgpu::Texture grown = device.CreateTexture(&descriptor);
    if (pool.texture) {
        const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
        for (std::uint32_t level = 0; level < pool.mipLevels; ++level)
            copyLayers(encoder, pool.texture, grown, level,
                       {getPhysicalExtent(pool.format, pool.width, level), getPhysicalExtent(pool.format, pool.height, level), pool.capacity});
        const wgpu::CommandBuffer commands = encoder.Finish();
        queue.Submit(1, &commands);
    }
    wgpu::TextureComponentSwizzleDescriptor swizzle;
    swizzle.swizzle = {.r = wgpu::ComponentSwizzle::R, .g = wgpu::ComponentSwizzle::R, .b = wgpu::ComponentSwizzle::R, .a = wgpu::ComponentSwizzle::One};
    const wgpu::TextureViewDescriptor viewDescriptor{.nextInChain = pool.format == ImageFormat::L8Unorm ? &swizzle : nullptr,
                                                     .dimension = wgpu::TextureViewDimension::e2DArray};
    wgpu::TextureView view = grown.CreateView(&viewDescriptor);
    if (std::expected<void, Error> created = popErrorScope("growing a WebGPU texture pool"); !created)
        return created;
    for (std::uint32_t layer = pool.capacity; layer < capacity; ++layer)
        pool.freeNodes[0].insert(makeNode(layer, 0, 0));
    pool.texture = std::move(grown);
    pool.view = std::move(view);
    pool.capacity = capacity;
    ++version;
    return {};
}

std::expected<std::uint64_t, Error> WebGpuDevice::allocateNode(SampledPool& pool, const std::uint32_t depth) {
    std::set<std::uint64_t>& free = pool.freeNodes[depth];
    if (!free.empty())
        return free.extract(free.begin()).value();
    if (depth == 0) {
        if (std::expected<void, Error> grown = growSampledPool(pool); !grown)
            return std::unexpected(std::move(grown.error()));
        return allocateNode(pool, 0);
    }
    const std::expected<std::uint64_t, Error> parent = allocateNode(pool, depth - 1);
    if (!parent)
        return parent;
    const auto [width, height] = getNodeSize(pool, depth);
    const bool vertical = depth % 2 == 0;
    free.insert(makeNode(getNodeLayer(*parent), getNodeX(*parent) + (vertical ? 0 : width), getNodeY(*parent) + (vertical ? height : 0)));
    return parent;
}

std::expected<SampledSlot, Error> WebGpuDevice::allocateSampled(const ImageFormat format, const std::uint32_t width, const std::uint32_t height,
                                                                const std::uint32_t mipLevels) {
    std::lock_guard lock(mutex);
    const FormatInfo info = getFormatInfo(format);
    const std::uint32_t limit = std::bit_floor(limits.maxTextureDimension2D);
    const bool atlas = !info.depth;
    const bool mipped = mipLevels > 1;
    std::uint32_t nodeWidth = std::max(std::bit_ceil(width), minimumNode);
    std::uint32_t nodeHeight = std::max(std::bit_ceil(height), minimumNode);
    if (mipped)
        nodeWidth = nodeHeight = std::max(nodeWidth, nodeHeight);
    const std::uint32_t side = std::max({getAtlasSize(format, limit), nodeWidth, nodeHeight});
    if (atlas && side > limit)
        return fail(ErrorCode::Unsupported, "the image needs a WebGPU texture larger than the device allows");
    const std::uint32_t poolWidth = atlas ? side : width;
    const std::uint32_t poolHeight = atlas ? side : height;
    const std::uint32_t poolLevels = !mipped ? 1 : static_cast<std::uint32_t>(std::bit_width(side)) - (info.blockExtent > 1 ? 2 : 0);

    const PoolBindings range = sampledRanges[static_cast<std::size_t>(getPoolKind(format, float32Filterable))];
    const auto bindings = std::views::iota(range.first, range.first + range.count);
    const auto matches = [&](const std::uint32_t binding) {
        const SampledPool& pool = sampledPools[binding];
        return pool.format == format && pool.width == poolWidth && pool.height == poolHeight && pool.mipLevels == poolLevels;
    };
    auto found = std::ranges::find_if(bindings, matches);
    if (found == bindings.end())
        found = std::ranges::find_if(bindings, [this](const std::uint32_t binding) { return sampledPools[binding].format == ImageFormat::Undefined; });
    if (found == bindings.end())
        return fail(ErrorCode::Unsupported, "all " + std::to_string(range.count) + " WebGPU texture pools of this kind hold other formats or sizes");
    if (freeTextureIndices.empty() && nextTextureIndex == textureTableCapacity)
        return fail(ErrorCode::OutOfMemory, "the WebGPU texture table holds at most " + std::to_string(textureTableCapacity - 1) + " sampled images");

    SampledPool& pool = sampledPools[*found];
    if (pool.format == ImageFormat::Undefined) {
        pool.format = format;
        pool.width = poolWidth;
        pool.height = poolHeight;
        pool.mipLevels = poolLevels;
        pool.atlas = atlas;
        pool.freeNodes.resize(atlas ? 2 * (std::bit_width(side / minimumNode) - 1) + 1 : 1);
    }
    std::uint32_t depth = 0;
    for (std::uint32_t next = 1; next < pool.freeNodes.size(); ++next) {
        const auto [fitWidth, fitHeight] = getNodeSize(pool, next);
        if (fitWidth < nodeWidth || fitHeight < nodeHeight)
            break;
        if (!mipped || next % 2 == 0)
            depth = next;
    }
    std::expected<std::uint64_t, Error> node = allocateNode(pool, depth);
    if (!node) {
        if (pool.used == 0)
            pool = {};
        return std::unexpected(std::move(node.error()));
    }
    ++pool.used;
    const std::uint32_t nodeSide = std::min(getNodeSize(pool, depth).first, getNodeSize(pool, depth).second);
    const std::uint32_t storedLevels = info.blockExtent > 1 ? static_cast<std::uint32_t>(std::bit_width(nodeSide)) - 2 : mipLevels;
    SampledSlot slot{.binding = *found,
                     .layer = getNodeLayer(*node),
                     .x = getNodeX(*node),
                     .y = getNodeY(*node),
                     .depth = depth,
                     .levels = std::min({mipLevels, poolLevels, storedLevels})};
    if (!freeTextureIndices.empty()) {
        slot.index = freeTextureIndices.back();
        freeTextureIndices.pop_back();
    } else {
        slot.index = nextTextureIndex++;
    }
    const std::array<std::uint32_t, 4> entry{slot.binding | slot.layer << 8, slot.x | slot.y << 16, width | height << 16, slot.levels};
    queue.WriteBuffer(heaps[0].buffer, textureTableAddress + std::uint64_t{slot.index} * sizeof(entry), entry.data(), sizeof(entry));
    return slot;
}

void WebGpuDevice::releaseSampled(const SampledSlot& slot) {
    std::lock_guard lock(mutex);
    SampledPool& pool = sampledPools[slot.binding];
    freeTextureIndices.push_back(slot.index);
    std::uint64_t node = makeNode(slot.layer, slot.x, slot.y);
    std::uint32_t depth = slot.depth;
    for (; depth > 0; --depth) {
        const auto [width, height] = getNodeSize(pool, depth);
        const bool vertical = depth % 2 == 0;
        const std::uint32_t x = getNodeX(node);
        const std::uint32_t y = getNodeY(node);
        const std::uint64_t buddy = makeNode(slot.layer, vertical ? x : x ^ width, vertical ? y ^ height : y);
        if (pool.freeNodes[depth].erase(buddy) == 0)
            break;
        node = makeNode(slot.layer, vertical ? x : x & ~width, vertical ? y & ~height : y);
    }
    pool.freeNodes[depth].insert(node);
    if (--pool.used != 0)
        return;
    pool = {};
    ++version;
}

std::expected<void, Error> WebGpuDevice::growStoragePool(StoragePool& pool, const std::uint32_t width, const std::uint32_t height,
                                                         const std::uint32_t capacity) {
    pushErrorScope();
    const wgpu::TextureDescriptor descriptor{.label = "NoorRHI storage pool",
                                             .usage = wgpu::TextureUsage::StorageBinding | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc
                                                      | wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::RenderAttachment,
                                             .size = {width, height, capacity},
                                             .format = pool.format};
    wgpu::Texture grown = device.CreateTexture(&descriptor);
    if (pool.texture) {
        const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
        copyLayers(encoder, pool.texture, grown, 0, {pool.width, pool.height, pool.capacity});
        const wgpu::CommandBuffer commands = encoder.Finish();
        queue.Submit(1, &commands);
    }
    const wgpu::TextureViewDescriptor viewDescriptor{.dimension = wgpu::TextureViewDimension::e2DArray};
    wgpu::TextureView view = grown.CreateView(&viewDescriptor);
    if (std::expected<void, Error> created = popErrorScope("growing a WebGPU storage pool"); !created)
        return created;
    pool.texture = std::move(grown);
    pool.view = std::move(view);
    pool.width = width;
    pool.height = height;
    pool.capacity = capacity;
    ++version;
    return {};
}

std::expected<std::uint32_t, Error> WebGpuDevice::allocateStorageLayer(const std::uint32_t binding, const std::uint32_t width, const std::uint32_t height) {
    std::lock_guard lock(mutex);
    const std::uint32_t index = binding - storagePoolBindings.first;
    StoragePool& pool = storagePools[index];
    const bool reuse = !pool.freeLayers.empty();
    const std::uint32_t layer = reuse ? pool.freeLayers.back() : pool.nextLayer;
    if (layer >= storageLayerCapacity)
        return fail(ErrorCode::Unsupported, "a WebGPU storage format holds at most " + std::to_string(storageLayerCapacity) + " storage images");
    const std::uint32_t capacity = layer < pool.capacity ? pool.capacity : std::min(std::max(initialLayers, pool.capacity * 2), storageLayerCapacity);
    if (capacity != pool.capacity || width > pool.width || height > pool.height)
        if (std::expected<void, Error> grown = growStoragePool(pool, std::max(width, pool.width), std::max(height, pool.height), capacity); !grown)
            return std::unexpected(std::move(grown.error()));
    if (reuse)
        pool.freeLayers.pop_back();
    else
        ++pool.nextLayer;
    const std::array<std::uint32_t, 2> extent{width, height};
    queue.WriteBuffer(heaps[0].buffer, storageExtentAddress + (std::uint64_t{index} * storageLayerCapacity + layer) * sizeof(extent), extent.data(),
                      sizeof(extent));
    return layer;
}

void WebGpuDevice::releaseStorageLayer(const std::uint32_t binding, const std::uint32_t layer) {
    std::lock_guard lock(mutex);
    storagePools[binding - storagePoolBindings.first].freeLayers.push_back(layer);
}

std::expected<std::uint32_t, Error> WebGpuDevice::acquireSampler(const SamplerDesc& desc, const wgpu::SamplerDescriptor& native) {
    std::lock_guard lock(mutex);
    auto found = std::ranges::find_if(samplers, [&desc](const SamplerSlot& slot) { return slot.references != 0 && isSameSampler(slot.desc, desc); });
    if (found == samplers.end()) {
        found = std::ranges::find_if(samplers, [](const SamplerSlot& slot) { return slot.references == 0; });
        if (found == samplers.end())
            return fail(ErrorCode::Unsupported, "all " + std::to_string(samplerBindings.count) + " WebGPU sampler slots hold other sampler descriptions");
        pushErrorScope();
        wgpu::Sampler sampler = device.CreateSampler(&native);
        if (std::expected<void, Error> created = popErrorScope("creating a WebGPU sampler"); !created)
            return std::unexpected(std::move(created.error()));
        found->sampler = std::move(sampler);
        found->desc = desc;
        ++version;
    }
    ++found->references;
    return samplerBindings.first + static_cast<std::uint32_t>(found - samplers.begin());
}

void WebGpuDevice::releaseSampler(const std::uint32_t binding) {
    std::lock_guard lock(mutex);
    --samplers[binding - samplerBindings.first].references;
}

TextureRegion WebGpuDevice::getPrimary(const WebGpuImage& image) const {
    if (image.storageBinding != 0)
        return {storagePools[image.storageBinding - storagePoolBindings.first].texture, image.storageLayer};
    if (image.texture)
        return {image.texture, 0};
    return getSampled(image);
}

TextureRegion WebGpuDevice::getSampled(const WebGpuImage& image) const {
    return {sampledPools[image.sampled.binding].texture, image.sampled.layer, image.sampled.x, image.sampled.y};
}

bool WebGpuDevice::hasStaleShadows(const bool dispatched) const {
    return std::ranges::any_of(shadowedImages, [dispatched](const WebGpuImage* image) { return image->needsShadowRefresh(dispatched); });
}

void WebGpuDevice::refreshShadows(const wgpu::CommandEncoder& encoder, const bool dispatched) {
    for (WebGpuImage* image : shadowedImages) {
        if (!image->needsShadowRefresh(dispatched))
            continue;
        const TextureRegion primary = getPrimary(*image);
        const TextureRegion sampled = getSampled(*image);
        const wgpu::TexelCopyTextureInfo from{.texture = primary.texture, .origin = {primary.x, primary.y, primary.layer}};
        const wgpu::TexelCopyTextureInfo to{.texture = sampled.texture, .origin = {sampled.x, sampled.y, sampled.layer}};
        const wgpu::Extent3D size{image->width, image->height, 1};
        encoder.CopyTextureToTexture(&from, &to, &size);
        image->shadowStale = false;
    }
}

wgpu::TextureView WebGpuImage::createView(const TextureRegion& region) const {
    const wgpu::TextureViewDescriptor descriptor{.format = nativeFormat,
                                                 .dimension = wgpu::TextureViewDimension::e2D,
                                                 .baseMipLevel = 0,
                                                 .mipLevelCount = 1,
                                                 .baseArrayLayer = region.layer,
                                                 .arrayLayerCount = 1};
    return region.texture.CreateView(&descriptor);
}

}
