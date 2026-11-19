#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Resources/Sampler.h"
#include "NoorRHI/WebGpu/WebGpu.h"

namespace NoorRHI {

std::expected<void, Error> Sampler::open(Device& device, const SamplerDesc& desc) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    if (desc.addressU == AddressMode::ClampToBorder || desc.addressV == AddressMode::ClampToBorder || desc.addressW == AddressMode::ClampToBorder)
        return fail(ErrorCode::Unsupported, "WebGPU has no ClampToBorder address mode");
    auto& owner = static_cast<Backend::WebGpuDevice&>(*device.getBackend());
    //! Every sample uses the device's one linear sampler, so the handle holds only the nearest flag and the address modes for the shader.
    auto result = std::make_shared<Backend::WebGpuSampler>();
    result->device = owner.shared_from_this();
    const std::uint32_t modes = static_cast<std::uint32_t>(desc.addressU) | static_cast<std::uint32_t>(desc.addressV) << 2;
    const std::uint32_t nearest = desc.filter == Filter::Nearest ? Backend::samplerNearest : 0u;
    handle = {nearest | modes << Backend::samplerModeShift};
    backend = std::move(result);
    return {};
}

}
