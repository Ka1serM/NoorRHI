#pragma once

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
struct Image;
}

struct ImageDesc {
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    ImageFormat format = ImageFormat::Rgba8Unorm;
    ImageUsage usage = ImageUsage::Sampled;
    std::uint32_t mipLevels = 1;
    std::uint32_t samples = 1;
    std::string_view name;
};

class Image {
public:
    [[nodiscard]] std::expected<void, Error> open(Device& device, const ImageDesc& desc);

    std::uint32_t getWidth() const { return width; }
    std::uint32_t getHeight() const { return height; }
    std::uint32_t getMipLevels() const { return mipLevels; }
    std::uint32_t getSamples() const { return samples; }
    ImageFormat getFormat() const { return format; }
    ImageUsage getUsage() const { return usage; }
    TextureHandle getSampledHandle() const { return sampledHandle; }
    StorageTextureHandle getStorageHandle() const { return storageHandle; }
    std::size_t getByteSize() const { return getImageByteSize(format, width, height, mipLevels); }
    Backend::Image* getBackend() const { return backend.get(); }
    explicit operator bool() const { return backend != nullptr; }

    [[nodiscard]] std::expected<void, Error> upload(std::span<const std::byte> mipChain);
    [[nodiscard]] std::expected<void, Error> download(std::span<std::byte> destination) const;
    [[nodiscard]] std::expected<void, Error> download(std::span<std::byte> destination, Rect region) const;

private:
    friend class Swapchain;

    std::shared_ptr<Backend::Image> backend;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t mipLevels = 0;
    std::uint32_t samples = 1;
    ImageFormat format = ImageFormat::Undefined;
    ImageUsage usage = ImageUsage::None;
    TextureHandle sampledHandle;
    StorageTextureHandle storageHandle;
};

}
