#include <algorithm>
#include <bit>

#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Resources/Sampler.h"
#include "NoorRHI/WebGpu/WebGpu.h"

namespace NoorRHI {

std::expected<void, Error> Sampler::open(Device& device, const SamplerDesc& desc) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    if (desc.addressU == AddressMode::ClampToBorder || desc.addressV == AddressMode::ClampToBorder || desc.addressW == AddressMode::ClampToBorder)
        return fail(ErrorCode::Unsupported, "WebGPU has no ClampToBorder address mode");
    const bool nearest = desc.filter == Filter::Nearest;
    const std::uint32_t anisotropy = nearest ? 1u : std::bit_floor(static_cast<std::uint32_t>(std::clamp(desc.maxAnisotropy, 1.0f, 16.0f)));
    const std::uint32_t modes = static_cast<std::uint32_t>(desc.addressU) | static_cast<std::uint32_t>(desc.addressV) << 2;
    handle = {modes | (nearest ? Backend::samplerNearestBit : 0u)
              | static_cast<std::uint32_t>(std::countr_zero(anisotropy)) << Backend::samplerAnisotropyShift};
    backend = std::make_shared<Backend::WebGpuSampler>();
    return {};
}

}
