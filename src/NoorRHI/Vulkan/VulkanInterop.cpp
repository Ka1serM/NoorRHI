#include "NoorRHI/Device/Interop.h"
#include "NoorRHI/Vulkan/Vulkan.h"

namespace NoorRHI::Interop {

namespace {

template <class Handle>
std::uintptr_t toAddress(const Handle handle) {
    return reinterpret_cast<std::uintptr_t>(static_cast<typename Handle::CType>(handle));
}

const Backend::VulkanImage* getVulkanImage(const Image& image) {
    return image ? static_cast<const Backend::VulkanImage*>(image.getBackend()) : nullptr;
}

}

std::expected<DeviceHandles, Error> getDeviceHandles(const Device& device) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    const auto& owner = static_cast<const Backend::VulkanDevice&>(*device.getBackend());
    const Backend::QueueState& graphics = owner.queues[static_cast<std::size_t>(Queue::Graphics)];
    return DeviceHandles{
        .instance = toAddress(*owner.instance),
        .physicalDevice = toAddress(owner.physicalDevice),
        .device = toAddress(*owner.device),
        .queue = toAddress(graphics.queue),
        .queueFamily = graphics.family,
        .pipelineCache = toAddress(owner.getPipelineCache(true)),
    };
}

std::expected<std::uintptr_t, Error> getCommandBuffer(const CommandList& list) {
    auto* vulkanList = static_cast<Backend::VulkanCommandList*>(list.getBackend());
    if (!vulkanList || vulkanList->closed)
        return fail(ErrorCode::InvalidState, "the command list is not open");
    //! The caller may record anything, so the list waits for all earlier commands and later commands wait for the caller's.
    vulkanList->synchronize(vk::PipelineStageFlagBits2::eAllCommands);
    return toAddress(vulkanList->pool.commands);
}

std::expected<std::uintptr_t, Error> getImage(const Image& image) {
    const Backend::VulkanImage* vulkanImage = getVulkanImage(image);
    if (!vulkanImage)
        return fail(ErrorCode::InvalidState, "the image is not open");
    return toAddress(vulkanImage->image);
}

std::expected<std::uintptr_t, Error> getImageView(const Image& image) {
    const Backend::VulkanImage* vulkanImage = getVulkanImage(image);
    if (!vulkanImage)
        return fail(ErrorCode::InvalidState, "the image is not open");
    return toAddress(*vulkanImage->view);
}

std::expected<ExternalImageMemory, Error> exportImageMemory(const Image& image) {
    const Backend::VulkanImage* vulkanImage = getVulkanImage(image);
    if (!vulkanImage || !vulkanImage->externalMemory)
        return fail(ErrorCode::InvalidArgument, "the image was not opened with ImageUsage::ExternalMemory");
    const vk::Device device = *vulkanImage->device->device;
    std::expected<int, Error> fd =
        Backend::checkVulkan(device.getMemoryFdKHR({vulkanImage->externalMemory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd}), "vkGetMemoryFdKHR");
    if (!fd)
        return std::unexpected(std::move(fd.error()));
    return ExternalImageMemory{
        .fd = *fd,
        .allocationSize = device.getImageMemoryRequirements(vulkanImage->image).size,
        .width = vulkanImage->width,
        .height = vulkanImage->height,
        .format = static_cast<std::uint32_t>(vulkanImage->format),
    };
}

std::expected<ExternalSemaphore, Error> signalExternal(Device& device) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    auto& owner = static_cast<Backend::VulkanDevice&>(*device.getBackend());
    if (!owner.externalSemaphoreFd)
        return fail(ErrorCode::Unsupported, "the Vulkan device has no VK_KHR_external_semaphore_fd");
    const vk::ExportSemaphoreCreateInfo exportInfo(vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd);
    std::expected<vk::Semaphore, Error> semaphore = Backend::checkVulkan(owner.device->createSemaphore({{}, &exportInfo}), "vkCreateSemaphore");
    if (!semaphore)
        return std::unexpected(std::move(semaphore.error()));
    std::unique_ptr<Backend::VulkanCommandList> list = owner.openList(Queue::Graphics);
    std::expected<Token, Error> token = owner.submit(*list, {}, {{}, *semaphore});
    owner.retire([&owner, signalled = *semaphore] { owner.device->destroySemaphore(signalled); });
    if (!token)
        return std::unexpected(std::move(token.error()));
    std::expected<int, Error> fd =
        Backend::checkVulkan(owner.device->getSemaphoreFdKHR({*semaphore, vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd}), "vkGetSemaphoreFdKHR");
    if (!fd)
        return std::unexpected(std::move(fd.error()));
    return ExternalSemaphore{*fd};
}

std::uint32_t getNativeFormat(const ImageFormat format) {
    return static_cast<std::uint32_t>(Backend::getVulkanFormat(format));
}

}
