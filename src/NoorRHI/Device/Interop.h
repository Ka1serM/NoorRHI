#pragma once

#include <cstdint>
#include <expected>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Core/Error.h"
#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Resources/Image.h"

namespace NoorRHI::Interop {

struct DeviceHandles {
    std::uintptr_t instance = 0;
    std::uintptr_t physicalDevice = 0;
    std::uintptr_t device = 0;
    std::uintptr_t queue = 0;
    std::uint32_t queueFamily = 0;
    std::uintptr_t pipelineCache = 0;
};

struct ExternalImageMemory {
    int fd = -1;
    std::uint64_t allocationSize = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t format = 0;
};

struct ExternalSemaphore {
    int fd = -1;
};

[[nodiscard]] std::expected<DeviceHandles, Error> getDeviceHandles(const Device& device);
[[nodiscard]] std::expected<std::uintptr_t, Error> getCommandBuffer(const CommandList& list);
[[nodiscard]] std::expected<std::uintptr_t, Error> getImage(const Image& image);
[[nodiscard]] std::expected<std::uintptr_t, Error> getImageView(const Image& image);
[[nodiscard]] std::expected<ExternalImageMemory, Error> exportImageMemory(const Image& image);
[[nodiscard]] std::expected<ExternalSemaphore, Error> signalExternal(Device& device);
std::uint32_t getNativeFormat(ImageFormat format);

}
