#pragma once

#include <cstdint>
#include <expected>
#include <memory>

#include "NoorRHI/Core/Error.h"
#include "NoorRHI/Core/Types.h"

namespace NoorRHI {

class Device;
namespace Backend {
struct Sampler;
}

enum class Filter : std::uint8_t { Nearest, Linear };
enum class AddressMode : std::uint8_t { Repeat, MirroredRepeat, ClampToEdge, ClampToBorder };

struct SamplerDesc {
    Filter filter = Filter::Linear;
    AddressMode addressU = AddressMode::Repeat;
    AddressMode addressV = AddressMode::Repeat;
    AddressMode addressW = AddressMode::Repeat;
    float maxAnisotropy = 1.0f;
};

class Sampler {
public:
    [[nodiscard]] std::expected<void, Error> open(Device& device, const SamplerDesc& desc = {});

    SamplerHandle getHandle() const { return handle; }
    explicit operator bool() const { return backend != nullptr; }

private:
    std::shared_ptr<Backend::Sampler> backend;
    SamplerHandle handle;
};

}
