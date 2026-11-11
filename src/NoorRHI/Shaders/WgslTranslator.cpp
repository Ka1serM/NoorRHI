#include "NoorRHI/Shaders/WgslTranslator.h"

#include <utility>

#include <src/tint/lang/core/ir/module.h>
#include <src/tint/lang/spirv/reader/reader.h>
#include <src/tint/lang/wgsl/allowed_features.h>
#include <src/tint/lang/wgsl/writer/writer.h>

#include "NoorRHI/Shaders/SpirvModule.h"

namespace NoorRHI {

std::expected<std::string, Error> translateToWgsl(const std::span<const std::uint32_t> spirv) {
    SpirvModule module;
    if (auto read = module.read(spirv); !read)
        return std::unexpected(std::move(read.error()));
    return translateToWgsl(module);
}

std::expected<std::string, Error> translateToWgsl(SpirvModule& module) {
    if (auto lowered = module.lowerForWebGpu(); !lowered)
        return std::unexpected(std::move(lowered.error()));
    auto ir = tint::spirv::reader::ReadIR(module.write());
    if (ir != tint::Success)
        return fail(ErrorCode::InvalidShader, ir.Failure().reason);
    auto wgsl = tint::wgsl::writer::WgslFromIR(ir.Get(), {.allowed_features = tint::wgsl::AllowedFeatures::Everything()});
    if (wgsl != tint::Success)
        return fail(ErrorCode::InvalidShader, wgsl.Failure().reason);
    return std::move(wgsl->wgsl);
}

}
