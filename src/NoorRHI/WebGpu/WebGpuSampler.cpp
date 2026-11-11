#include <algorithm>
#include <bit>

#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Resources/Sampler.h"
#include "NoorRHI/WebGpu/WebGpu.h"

namespace NoorRHI {

namespace Backend {

WebGpuSampler::~WebGpuSampler() {
    if (device && binding != 0)
        device->releaseSampler(binding);
}

}

std::expected<void, Error> Sampler::open(Device& device, const SamplerDesc& desc) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    if (desc.addressU == AddressMode::ClampToBorder || desc.addressV == AddressMode::ClampToBorder || desc.addressW == AddressMode::ClampToBorder)
        return fail(ErrorCode::Unsupported, "WebGPU has no ClampToBorder address mode");
    auto& owner = static_cast<Backend::WebGpuDevice&>(*device.getBackend());
    const bool linear = desc.filter == Filter::Linear;
    const wgpu::FilterMode filter = linear ? wgpu::FilterMode::Linear : wgpu::FilterMode::Nearest;
    //! The shader applies the address modes, so a slot holds a filter and a power-of-two anisotropy, and six slots cover every sampler.
    SamplerDesc slot{.filter = desc.filter};
    slot.maxAnisotropy = linear ? static_cast<float>(std::bit_floor(static_cast<std::uint32_t>(std::clamp(desc.maxAnisotropy, 1.0f, 16.0f)))) : 1.0f;
    const wgpu::SamplerDescriptor native{
        .magFilter = filter,
        .minFilter = filter,
        .mipmapFilter = linear ? wgpu::MipmapFilterMode::Linear : wgpu::MipmapFilterMode::Nearest,
        .maxAnisotropy = static_cast<std::uint16_t>(slot.maxAnisotropy),
    };
    std::expected<std::uint32_t, Error> binding = owner.acquireSampler(slot, native);
    if (!binding)
        return std::unexpected(std::move(binding.error()));
    auto result = std::make_shared<Backend::WebGpuSampler>();
    result->device = owner.shared_from_this();
    result->binding = *binding;
    const std::uint32_t modes = static_cast<std::uint32_t>(desc.addressU) | static_cast<std::uint32_t>(desc.addressV) << 2;
    handle = {*binding | modes << Backend::samplerModeShift};
    backend = std::move(result);
    return {};
}

}
