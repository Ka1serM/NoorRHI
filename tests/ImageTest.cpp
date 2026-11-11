#include <array>
#include <cstdint>
#include <vector>

#include "Test.h"

using namespace NoorRHI;

namespace {

std::vector<std::uint32_t> makePattern(const std::uint32_t width, const std::uint32_t height, const std::uint32_t seed) {
    std::vector<std::uint32_t> texels(std::size_t{width} * height);
    for (std::uint32_t index = 0; index < texels.size(); ++index)
        texels[index] = (index * 2654435761u) ^ seed;
    return texels;
}

}

TEST_CASE("images round trip and download regions") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    constexpr std::uint32_t width = 37;
    constexpr std::uint32_t height = 21;
    Image image;
    REQUIRE_OK(image.open(*device, {.width = width, .height = height}));
    const std::vector<std::uint32_t> texels = makePattern(width, height, 7);
    REQUIRE_OK(Test::uploadTexels(image, texels));
    std::expected<std::vector<std::uint32_t>, Error> downloaded = Test::downloadTexels(image);
    REQUIRE_OK(downloaded);
    CHECK(*downloaded == texels);

    std::vector<std::uint32_t> region(5 * 3);
    REQUIRE_OK(image.download(std::as_writable_bytes(std::span(region)), {4, 6, 5, 3}));
    for (std::uint32_t y = 0; y < 3; ++y)
        for (std::uint32_t x = 0; x < 5; ++x)
            REQUIRE(region[y * 5 + x] == texels[(y + 6) * width + x + 4]);
}

TEST_CASE("many sampled images of distinct sizes live side by side") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    std::vector<Image> images(48);
    for (std::uint32_t index = 0; index < images.size(); ++index) {
        const std::uint32_t width = 3 + index * 7;
        const std::uint32_t height = 5 + index * 3;
        REQUIRE_OK(images[index].open(*device, {.width = width, .height = height}));
        REQUIRE_OK(Test::uploadTexels(images[index], makePattern(width, height, index)));
    }
    for (std::uint32_t index = 0; index < images.size(); index += 5)
        images[index] = {};
    for (std::uint32_t index = 0; index < images.size(); ++index) {
        if (!images[index])
            continue;
        std::expected<std::vector<std::uint32_t>, Error> texels = Test::downloadTexels(images[index]);
        REQUIRE_OK(texels);
        REQUIRE(*texels == makePattern(images[index].getWidth(), images[index].getHeight(), index));
    }
}

TEST_CASE("mip chains upload as one block") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    Image image;
    REQUIRE_OK(image.open(*device, {.width = 64, .height = 32, .mipLevels = 7}));
    std::vector<std::uint32_t> chain(image.getByteSize() / 4);
    for (std::uint32_t index = 0; index < chain.size(); ++index)
        chain[index] = index;
    REQUIRE_OK(Test::uploadTexels(image, chain));
    std::expected<std::vector<std::uint32_t>, Error> level = Test::downloadTexels(image);
    REQUIRE_OK(level);
    CHECK(std::equal(level->begin(), level->end(), chain.begin()));
}

TEST_CASE("copies take a source region and scale when the sizes differ") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    constexpr std::uint32_t red = 0xff0000ffu;
    constexpr std::uint32_t green = 0xff00ff00u;
    std::vector<std::uint32_t> halves(8 * 8);
    for (std::uint32_t index = 0; index < halves.size(); ++index)
        halves[index] = index % 8 < 4 ? red : green;
    Image source;
    Image direct;
    Image scaled;
    REQUIRE_OK(source.open(*device, {.width = 8, .height = 8}));
    REQUIRE_OK(direct.open(*device, {.width = 4, .height = 2}));
    REQUIRE_OK(scaled.open(*device, {.width = 2, .height = 2}));
    REQUIRE_OK(Test::uploadTexels(source, halves));

    CommandList list(*device);
    list.copy(source, direct, Rect{2, 3, 4, 2});
    list.copy(source, scaled);
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::expected<std::vector<std::uint32_t>, Error> region = Test::downloadTexels(direct);
    REQUIRE_OK(region);
    CHECK(*region == (std::vector<std::uint32_t>{red, red, green, green, red, red, green, green}));
    std::expected<std::vector<std::uint32_t>, Error> small = Test::downloadTexels(scaled);
    REQUIRE_OK(small);
    CHECK(*small == (std::vector<std::uint32_t>{red, green, red, green}));
}

TEST_CASE("invalid image descriptions fail") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    Image image;
    CHECK(image.open(*device, {.width = 4, .height = 4, .samples = 2}).error().code == ErrorCode::InvalidArgument);
    CHECK(image.open(*device, {.width = 4, .height = 4, .samples = 4}).error().code == ErrorCode::InvalidArgument);
    CHECK(image.open(*device, {.width = 4, .height = 4, .format = ImageFormat::D32FloatS8Uint, .usage = ImageUsage::DepthAttachment | ImageUsage::Sampled})
              .error()
              .code
          == ErrorCode::InvalidArgument);
    CHECK(image.open(*device, {.width = 4, .height = 4, .usage = ImageUsage::ColorAttachment, .mipLevels = 2}).error().code == ErrorCode::InvalidArgument);
}
