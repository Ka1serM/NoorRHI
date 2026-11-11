#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string_view>

#include "NoorRHI/Core/Error.h"
#include "NoorRHI/Core/Format.h"
#include "NoorRHI/Core/Types.h"

namespace NoorRHI {

class Device;
namespace Backend {
struct ComputePipeline;
struct GraphicsPipeline;
}

enum class ShaderStage : std::uint8_t { Compute, Vertex, Fragment, RayGeneration, Miss, ClosestHit, AnyHit, Intersection, Callable };

//! noorrhi_add_shaders generates one per entry point, with SPIR-V or WGSL code indexed by hardware ray tracing (bit 0) and cooperative matrices (bit 1).
struct Shader {
    std::array<std::span<const std::byte>, 4> code;
    std::string_view entryPoint;
    ShaderStage stage = ShaderStage::Compute;
    DispatchSize threadGroupSize;
    std::uint32_t fallbackIndex = 0;

    std::span<const std::byte> getCode(const bool hardwareRayTracing, const bool cooperativeMatrices) const {
        return code[static_cast<std::size_t>(hardwareRayTracing) | static_cast<std::size_t>(cooperativeMatrices) << 1];
    }
    const std::byte* getSource() const { return code[0].data(); }
    explicit operator bool() const { return !code[0].empty(); }
};

enum class CullMode : std::uint8_t { None, Front, Back };
enum class FrontFace : std::uint8_t { CounterClockwise, Clockwise };
enum class PolygonMode : std::uint8_t { Fill, Line, Point };
enum class CompareOp : std::uint8_t { Never, Less, Equal, LessOrEqual, Greater, NotEqual, GreaterOrEqual, Always };
enum class BlendMode : std::uint8_t { Opaque, Alpha, PremultipliedAlpha, Additive };
enum class StencilOp : std::uint8_t { Keep, Zero, Replace, Increment, Decrement, Invert };

struct StencilState {
    CompareOp compare = CompareOp::Always;
    StencilOp pass = StencilOp::Keep;

    bool isEnabled() const { return compare != CompareOp::Always || pass != StencilOp::Keep; }
};

struct GraphicsState {
    CullMode cull = CullMode::Back;
    FrontFace frontFace = FrontFace::CounterClockwise;
    PolygonMode polygon = PolygonMode::Fill;
    CompareOp depthCompare = CompareOp::LessOrEqual;
    bool depthTest = false;
    bool depthWrite = false;
    BlendMode blend = BlendMode::Opaque;
    bool colorWrite = true;
    StencilState stencil;
};

struct GraphicsPipelineDesc {
    Shader vertex;
    Shader fragment;
    GraphicsState state;
    ImageFormat colorFormat = ImageFormat::Rgba8Unorm;
    ImageFormat depthFormat = ImageFormat::Undefined;
    std::uint32_t samples = 1;
};

class ComputePipeline {
public:
    [[nodiscard]] std::expected<void, Error> open(Device& device, const Shader& shader);
    //! Returns once the compilation starts; the shader code must stay valid until the pipeline is ready.
    [[nodiscard]] std::expected<void, Error> openAsync(Device& device, const Shader& shader);
    bool isReady() const;
    [[nodiscard]] std::expected<void, Error> wait() const;

    Backend::ComputePipeline* getBackend() const { return backend.get(); }
    explicit operator bool() const { return backend != nullptr; }

private:
    std::shared_ptr<Backend::ComputePipeline> backend;
};

class GraphicsPipeline {
public:
    [[nodiscard]] std::expected<void, Error> open(Device& device, const GraphicsPipelineDesc& desc);
    //! Returns once the compilation starts; the shader code must stay valid until the pipeline is ready.
    [[nodiscard]] std::expected<void, Error> openAsync(Device& device, const GraphicsPipelineDesc& desc);
    bool isReady() const;
    [[nodiscard]] std::expected<void, Error> wait() const;

    Backend::GraphicsPipeline* getBackend() const { return backend.get(); }
    explicit operator bool() const { return backend != nullptr; }

private:
    std::shared_ptr<Backend::GraphicsPipeline> backend;
};

}
