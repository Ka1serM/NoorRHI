#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>

#include "NoorRHI/Core/Error.h"

namespace NoorRHI {

class SpirvModule;

//! WebGPU has no base vertex, base instance or texel buffers: draws start at 0, and a texel buffer becomes a storage buffer of 16-byte vectors.
[[nodiscard]] std::expected<std::string, Error> translateToWgsl(std::span<const std::uint32_t> spirv);
[[nodiscard]] std::expected<std::string, Error> translateToWgsl(SpirvModule& module);

}
