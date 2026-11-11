#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Resources/Image.h"
#include "NoorRHI/Vulkan/Vulkan.h"

namespace NoorRHI {

namespace {

std::expected<std::uint32_t, Error> writeImageDescriptor(Backend::VulkanImage& image, const bool storage) {
    const bool luminance = !storage && (image.publicFormat == ImageFormat::L8Unorm || image.publicFormat == ImageFormat::L8Srgb);
    const vk::ComponentMapping components =
        luminance ? vk::ComponentMapping{vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eOne}
                  : vk::ComponentMapping{};
    const vk::ImageViewUsageCreateInfo usage(storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled);
    const vk::ImageViewCreateInfo info({}, image.image, vk::ImageViewType::e2D, image.format, components, {image.aspect, 0, image.mipLevels, 0, 1}, &usage);
    Backend::VulkanDevice& device = *image.device;
    std::expected<vk::UniqueImageView, Error> view = Backend::checkVulkan(device.device->createImageViewUnique(info), "vkCreateImageView");
    if (!view)
        return std::unexpected(std::move(view.error()));
    vk::UniqueImageView& target = storage ? image.storageView : image.sampledView;
    target = std::move(*view);
    return device.writeDescriptor(device.textureSlots, storage ? Backend::storageImageBinding : Backend::sampledImageBinding,
                                  storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage,
                                  vk::DescriptorImageInfo({}, *target, vk::ImageLayout::eGeneral));
}

std::expected<void, Error> allocateExternal(Backend::VulkanDevice& device, Backend::VulkanImage& image, const vk::ImageCreateInfo& createInfo) {
    std::expected<vk::Image, Error> created = Backend::checkVulkan(device.device->createImage(createInfo), "vkCreateImage");
    if (!created)
        return std::unexpected(std::move(created.error()));
    image.image = *created;
    const vk::MemoryRequirements requirements = device.device->getImageMemoryRequirements(image.image);
    const vk::PhysicalDeviceMemoryProperties properties = device.physicalDevice.getMemoryProperties();
    std::uint32_t memoryType = properties.memoryTypeCount;
    for (std::uint32_t index = 0; index < properties.memoryTypeCount && memoryType == properties.memoryTypeCount; ++index)
        if ((requirements.memoryTypeBits & (1u << index)) && (properties.memoryTypes[index].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal))
            memoryType = index;
    if (memoryType == properties.memoryTypeCount)
        return fail(ErrorCode::OutOfMemory, "the device has no device-local memory type for an exportable image");
    vk::MemoryDedicatedAllocateInfo dedicated({image.image});
    const vk::ExportMemoryAllocateInfo exportInfo(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd, &dedicated);
    std::expected<vk::DeviceMemory, Error> memory =
        Backend::checkVulkan(device.device->allocateMemory(vk::MemoryAllocateInfo(requirements.size, memoryType, &exportInfo)), "vkAllocateMemory");
    if (!memory)
        return std::unexpected(std::move(memory.error()));
    image.externalMemory = *memory;
    return Backend::checkVulkan(device.device->bindImageMemory(image.image, image.externalMemory, 0), "vkBindImageMemory");
}

}

namespace Backend {

VulkanImage::~VulkanImage() {
    if (!device || !image)
        return;
    device->retire([owner = device.get(), image = this->image, memory = this->memory, externalMemory = this->externalMemory, owns = ownsImage,
                    views = std::array{view.release(), sampledView.release(), storageView.release()}, sampled = sampledSlot, storage = storageSlot] {
        owner->releaseSlot(owner->textureSlots, sampled);
        owner->releaseSlot(owner->textureSlots, storage);
        for (const vk::ImageView imageView : views)
            if (imageView)
                owner->device->destroyImageView(imageView);
        if (!owns)
            return;
        owner->device->destroyImage(image);
        owner->release(memory);
        owner->device->freeMemory(externalMemory);
    });
}

}

std::expected<void, Error> Image::open(Device& device, const ImageDesc& desc) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    const bool depth = hasUsage(desc.usage, ImageUsage::DepthAttachment);
    const bool external = hasUsage(desc.usage, ImageUsage::ExternalMemory);
    if (std::expected<void, Error> checked = Backend::checkImageDesc(desc); !checked)
        return checked;
    auto& owner = static_cast<Backend::VulkanDevice&>(*device.getBackend());
    if (external && !owner.externalMemoryFd)
        return fail(ErrorCode::Unsupported, "the Vulkan device has no VK_KHR_external_memory_fd");
    const vk::FormatFeatureFlags features = owner.physicalDevice.getFormatProperties(Backend::getVulkanFormat(desc.format)).optimalTilingFeatures;
    if (depth && !(features & vk::FormatFeatureFlagBits::eDepthStencilAttachment))
        return fail(ErrorCode::Unsupported, "the Vulkan device cannot use this depth format as an attachment");

    vk::ImageUsageFlags usageFlags = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
    if (hasUsage(desc.usage, ImageUsage::Sampled))
        usageFlags |= vk::ImageUsageFlagBits::eSampled;
    if (hasUsage(desc.usage, ImageUsage::Storage))
        usageFlags |= vk::ImageUsageFlagBits::eStorage;
    if (hasUsage(desc.usage, ImageUsage::ColorAttachment))
        usageFlags |= vk::ImageUsageFlagBits::eColorAttachment;
    if (depth)
        usageFlags |= vk::ImageUsageFlagBits::eDepthStencilAttachment;

    auto result = std::make_shared<Backend::VulkanImage>();
    result->device = owner.shared_from_this();
    result->format = Backend::getVulkanFormat(desc.format);
    result->publicFormat = desc.format;
    result->aspect = !depth                               ? vk::ImageAspectFlags(vk::ImageAspectFlagBits::eColor)
                     : getFormatInfo(desc.format).stencil ? vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil
                                                          : vk::ImageAspectFlags(vk::ImageAspectFlagBits::eDepth);
    result->width = desc.width;
    result->height = desc.height;
    result->mipLevels = desc.mipLevels;

    const vk::ExternalMemoryImageCreateInfo externalInfo(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
    const vk::ImageCreateInfo createInfo({}, vk::ImageType::e2D, result->format, {desc.width, desc.height, 1}, desc.mipLevels, 1,
                                         static_cast<vk::SampleCountFlagBits>(desc.samples), vk::ImageTiling::eOptimal, usageFlags, owner.getSharing(),
                                         owner.sharedFamilies, vk::ImageLayout::eUndefined, external ? &externalInfo : nullptr);
    if (external) {
        if (std::expected<void, Error> allocated = allocateExternal(owner, *result, createInfo); !allocated)
            return allocated;
    } else {
        std::expected<vk::Image, Error> image = Backend::checkVulkan(owner.device->createImage(createInfo), "vkCreateImage");
        if (!image)
            return std::unexpected(std::move(image.error()));
        result->image = *image;
        std::expected<Backend::Allocation, Error> memory = owner.allocate(owner.device->getImageMemoryRequirements(result->image));
        if (!memory)
            return std::unexpected(std::move(memory.error()));
        result->memory = *memory;
        if (std::expected<void, Error> bound =
                Backend::checkVulkan(owner.device->bindImageMemory(result->image, memory->memory, memory->offset), "vkBindImageMemory");
            !bound)
            return bound;
    }

    std::expected<vk::UniqueImageView, Error> view =
        Backend::checkVulkan(owner.device->createImageViewUnique({{}, result->image, vk::ImageViewType::e2D, result->format, {}, {result->aspect, 0, 1, 0, 1}}),
                             "vkCreateImageView");
    if (!view)
        return std::unexpected(std::move(view.error()));
    result->view = std::move(*view);
    if (hasUsage(desc.usage, ImageUsage::Storage)) {
        std::expected<std::uint32_t, Error> slot = writeImageDescriptor(*result, true);
        if (!slot)
            return std::unexpected(std::move(slot.error()));
        result->storageSlot = *slot;
    }
    if (hasUsage(desc.usage, ImageUsage::Sampled)) {
        std::expected<std::uint32_t, Error> slot = writeImageDescriptor(*result, false);
        if (!slot)
            return std::unexpected(std::move(slot.error()));
        result->sampledSlot = *slot;
    }
    std::expected<Token, Error> initialized = owner.execute([&result](Backend::VulkanCommandList& list) {
        Backend::recordImageBarrier(list.pool.commands, result->image, result->aspect, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    });
    if (!initialized)
        return std::unexpected(std::move(initialized.error()));

    sampledHandle = {result->sampledSlot};
    storageHandle = {result->storageSlot};
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

std::expected<void, Error> VulkanImage::upload(const std::span<const std::byte> mipChain) {
    std::expected<Token, Error> token = device->execute([&](VulkanCommandList& list) {
        std::expected<Placement, Error> staged = list.place(mipChain);
        if (!staged)
            return static_cast<void>(list.check(false, staged.error().code, staged.error().message));
        std::vector<vk::BufferImageCopy> copies;
        vk::DeviceSize offset = staged->offset;
        for (std::uint32_t level = 0; level < mipLevels; ++level) {
            const std::uint32_t levelWidth = std::max(width >> level, 1u);
            const std::uint32_t levelHeight = std::max(height >> level, 1u);
            copies.emplace_back(offset, 0, 0, vk::ImageSubresourceLayers(aspect, level, 0, 1), vk::Offset3D{}, vk::Extent3D{levelWidth, levelHeight, 1});
            offset += getLevelByteSize(publicFormat, levelWidth, levelHeight);
        }
        list.synchronize(vk::PipelineStageFlagBits2::eAllTransfer);
        list.pool.commands.copyBufferToImage(staged->buffer, image, vk::ImageLayout::eGeneral, copies);
    });
    if (!token)
        return std::unexpected(std::move(token.error()));
    return {};
}

std::expected<void, Error> VulkanImage::download(const std::uint32_t x, const std::uint32_t y, const std::uint32_t regionWidth,
                                                 const std::uint32_t regionHeight, const std::span<std::byte> destination) {
    return device->download(destination, [&](const vk::CommandBuffer commands, const Allocation& staging) {
        commands.copyImageToBuffer(image, vk::ImageLayout::eGeneral, staging.buffer,
                                   vk::BufferImageCopy(staging.offset, 0, 0, {aspect & ~vk::ImageAspectFlagBits::eStencil, 0, 0, 1},
                                                       {static_cast<std::int32_t>(x), static_cast<std::int32_t>(y), 0}, {regionWidth, regionHeight, 1}));
    });
}

}

}
