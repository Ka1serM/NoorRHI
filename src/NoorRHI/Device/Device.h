#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Core/Error.h"
#include "NoorRHI/Core/Types.h"

namespace NoorRHI {

class SurfaceProvider;
namespace Backend {
struct Device;
}

enum class MatrixComponent : std::uint8_t { Float16, Float32, Int8, Uint8, Int32, Uint32 };

struct CooperativeMatrixShape {
    std::uint32_t m = 0;
    std::uint32_t n = 0;
    std::uint32_t k = 0;
    MatrixComponent a = MatrixComponent::Float16;
    MatrixComponent b = MatrixComponent::Float16;
    MatrixComponent c = MatrixComponent::Float32;
    MatrixComponent result = MatrixComponent::Float32;

    bool operator==(const CooperativeMatrixShape&) const = default;
};

struct DeviceFeatures {
    bool hardwareRayTracing = false;
    bool subgroups = false;
    bool shaderFloat16 = false;
    std::uint32_t minSubgroupSize = 0;
    std::vector<CooperativeMatrixShape> cooperativeMatrixShapes;
};

struct DeviceInfo {
    GraphicsApi api = GraphicsApi::Vulkan;
    std::string name;
    std::uint32_t vendorId = 0;
    std::uint32_t deviceId = 0;
};

struct DeviceConfig {
    bool enableValidation = false;
    std::string_view applicationName = "NoorRHI";
    SurfaceProvider* presentation = nullptr;
    std::filesystem::path pipelineDirectory;
    bool hardwareRayTracing = true;
    bool cooperativeMatrices = true;
};

class Device {
public:
    [[nodiscard]] std::expected<void, Error> open(const DeviceConfig& config = {});

    const DeviceFeatures& getFeatures() const;
    const DeviceInfo& getInfo() const;

    [[nodiscard]] std::expected<Token, Error> submit(CommandList&& list, std::span<const Token> waits = {});
    [[nodiscard]] std::expected<void, Error> wait(Token token);
    bool isFinished(Token token) const;
    [[nodiscard]] std::expected<void, Error> waitIdle();

    Backend::Device* getBackend() const { return backend.get(); }
    explicit operator bool() const { return backend != nullptr; }

private:
    std::shared_ptr<Backend::Device> backend;
};

}
