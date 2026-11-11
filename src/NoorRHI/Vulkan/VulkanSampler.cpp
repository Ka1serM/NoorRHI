#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Resources/Sampler.h"
#include "NoorRHI/Vulkan/Vulkan.h"

namespace NoorRHI {

namespace {

vk::SamplerAddressMode getAddressMode(const AddressMode mode) {
    switch (mode) {
    case AddressMode::Repeat:
        return vk::SamplerAddressMode::eRepeat;
    case AddressMode::MirroredRepeat:
        return vk::SamplerAddressMode::eMirroredRepeat;
    case AddressMode::ClampToEdge:
        return vk::SamplerAddressMode::eClampToEdge;
    case AddressMode::ClampToBorder:
        return vk::SamplerAddressMode::eClampToBorder;
    }
    return vk::SamplerAddressMode::eRepeat;
}

}

namespace Backend {

VulkanSampler::~VulkanSampler() {
    if (!device)
        return;
    device->retire([owner = device.get(), sampler = this->sampler, slot = this->slot] {
        owner->releaseSlot(owner->samplerSlots, slot);
        owner->device->destroySampler(sampler);
    });
}

}

std::expected<void, Error> Sampler::open(Device& device, const SamplerDesc& desc) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    auto& owner = static_cast<Backend::VulkanDevice&>(*device.getBackend());
    const vk::Filter filter = desc.filter == Filter::Linear ? vk::Filter::eLinear : vk::Filter::eNearest;
    const vk::SamplerMipmapMode mipmap = desc.filter == Filter::Linear ? vk::SamplerMipmapMode::eLinear : vk::SamplerMipmapMode::eNearest;
    vk::SamplerCreateInfo info({}, filter, filter, mipmap, getAddressMode(desc.addressU), getAddressMode(desc.addressV), getAddressMode(desc.addressW));
    info.maxLod = vk::LodClampNone;
    info.anisotropyEnable = desc.maxAnisotropy > 1.0f;
    info.maxAnisotropy = desc.maxAnisotropy;
    std::expected<vk::Sampler, Error> created = Backend::checkVulkan(owner.device->createSampler(info), "vkCreateSampler");
    if (!created)
        return std::unexpected(std::move(created.error()));
    auto result = std::make_shared<Backend::VulkanSampler>();
    result->device = owner.shared_from_this();
    result->sampler = *created;
    std::expected<std::uint32_t, Error> slot =
        owner.writeDescriptor(owner.samplerSlots, Backend::samplerBinding, vk::DescriptorType::eSampler, vk::DescriptorImageInfo(result->sampler));
    if (!slot)
        return std::unexpected(std::move(slot.error()));
    result->slot = *slot;
    handle = {*slot};
    backend = std::move(result);
    return {};
}

}
