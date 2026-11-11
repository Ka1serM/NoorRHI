#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>

#include "Matrix.h"
#include "Matrix.slang.h"
#include "MatrixFloat.slang.h"
#include "Test.h"

using namespace NoorRHI;

namespace {

constexpr std::uint32_t size = 16;

//! The test values are small multiples of 0.5, so dropping the low mantissa bits converts them to half exactly.
half toHalf(const float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto exponent = static_cast<std::int32_t>(bits >> 23 & 0xffu);
    const auto sign = static_cast<std::uint16_t>(bits >> 16 & 0x8000u);
    if (exponent == 0)
        return {sign};
    return {static_cast<std::uint16_t>(sign | (exponent - 112) << 10 | (bits >> 13 & 0x3ffu))};
}

float getA(const std::uint32_t row, const std::uint32_t column) {
    return static_cast<float>((row * 3 + column) % 7) - 3.0f;
}

float getB(const std::uint32_t row, const std::uint32_t column) {
    return static_cast<float>((row + column * 5) % 5) * 0.5f;
}

float getBias(const std::uint32_t row, const std::uint32_t column) {
    return static_cast<float>((row * 7 + column * 3) % 11) - 8.0f;
}

}

TEST_CASE("matrices multiply with the same shader on every device") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    if (!device->getFeatures().shaderFloat16)
        return;
    std::array<half, size * size> a{};
    std::array<half, size * size> b{};
    std::array<float, size * size> bias{};
    for (std::uint32_t row = 0; row < size; ++row)
        for (std::uint32_t column = 0; column < size; ++column) {
            a[row * size + column] = toHalf(getA(row, column));
            b[row * size + column] = toHalf(getB(row, column));
            bias[row * size + column] = getBias(row, column);
        }
    Buffer<half> left;
    Buffer<half> right;
    Buffer<float> initial;
    Buffer<float> result;
    REQUIRE_OK(left.open(*device, a.size()));
    REQUIRE_OK(right.open(*device, b.size()));
    REQUIRE_OK(initial.open(*device, bias.size()));
    REQUIRE_OK(result.open(*device, size * size));
    REQUIRE_OK(left.upload(a));
    REQUIRE_OK(right.upload(b));
    REQUIRE_OK(initial.upload(bias));
    ComputePipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, Matrix::main));

    CommandList list(*device);
    list.dispatch(pipeline, {32}, Matrix::Arguments{left.getPtr(), right.getPtr(), initial.getPtr(), result.getPtr()});
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::array<float, size * size> product{};
    REQUIRE_OK(result.download(product));
    for (std::uint32_t row = 0; row < size; ++row)
        for (std::uint32_t column = 0; column < size; ++column) {
            float expected = getBias(row, column);
            for (std::uint32_t index = 0; index < size; ++index)
                expected += getA(row, index) * getB(index, column);
            REQUIRE(product[row * size + column] == std::max(expected * 2.0f, 0.0f));
        }
}

TEST_CASE("float matrices of a shape without tensor cores multiply") {
    constexpr std::uint32_t depth = 8;
    Device* device = Test::getDevice();
    REQUIRE(device);
    std::array<float, size * depth> a{};
    std::array<float, depth * size> b{};
    std::array<float, size * size> bias{};
    for (std::uint32_t row = 0; row < size; ++row)
        for (std::uint32_t column = 0; column < depth; ++column) {
            a[row * depth + column] = getA(row, column);
            b[column * size + row] = getB(column, row);
        }
    for (std::uint32_t row = 0; row < size; ++row)
        for (std::uint32_t column = 0; column < size; ++column)
            bias[row * size + column] = getBias(row, column);
    Buffer<float> left;
    Buffer<float> right;
    Buffer<float> initial;
    Buffer<float> result;
    REQUIRE_OK(left.open(*device, a.size()));
    REQUIRE_OK(right.open(*device, b.size()));
    REQUIRE_OK(initial.open(*device, bias.size()));
    REQUIRE_OK(result.open(*device, size * size));
    REQUIRE_OK(left.upload(a));
    REQUIRE_OK(right.upload(b));
    REQUIRE_OK(initial.upload(bias));
    ComputePipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, MatrixFloat::main));

    CommandList list(*device);
    list.dispatch(pipeline, {32}, Matrix::FloatArguments{left.getPtr(), right.getPtr(), initial.getPtr(), result.getPtr()});
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::array<float, size * size> product{};
    REQUIRE_OK(result.download(product));
    for (std::uint32_t row = 0; row < size; ++row)
        for (std::uint32_t column = 0; column < size; ++column) {
            float expected = getBias(row, column);
            for (std::uint32_t index = 0; index < depth; ++index)
                expected += getA(row, index) * getB(index, column);
            REQUIRE(product[row * size + column] == expected);
        }
}

TEST_CASE("matrices multiply in a loop with a bound from the arguments") {
    constexpr std::uint32_t tiled = 32;
    Device* device = Test::getDevice();
    REQUIRE(device);
    if (!device->getFeatures().shaderFloat16)
        return;
    std::array<half, tiled * tiled> a{};
    std::array<half, tiled * tiled> b{};
    for (std::uint32_t row = 0; row < tiled; ++row)
        for (std::uint32_t column = 0; column < tiled; ++column) {
            a[row * tiled + column] = toHalf(getA(row, column));
            b[row * tiled + column] = toHalf(getB(row, column));
        }
    Buffer<half> left;
    Buffer<half> right;
    Buffer<float> result;
    REQUIRE_OK(left.open(*device, a.size()));
    REQUIRE_OK(right.open(*device, b.size()));
    REQUIRE_OK(result.open(*device, tiled * tiled));
    REQUIRE_OK(left.upload(a));
    REQUIRE_OK(right.upload(b));
    REQUIRE_OK(result.upload(std::array<float, tiled * tiled>{}));
    ComputePipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, Matrix::tiledMain));

    CommandList list(*device);
    list.dispatch(pipeline, {32 * (tiled / 16), tiled / 16}, Matrix::TiledArguments{left.getPtr(), right.getPtr(), result.getPtr(), tiled});
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::array<float, tiled * tiled> product{};
    REQUIRE_OK(result.download(product));
    std::uint32_t wrong = 0;
    for (std::uint32_t row = 0; row < tiled; ++row)
        for (std::uint32_t column = 0; column < tiled; ++column) {
            float expected = 0.0f;
            for (std::uint32_t index = 0; index < tiled; ++index)
                expected += getA(row, index) * getB(index, column);
            wrong += product[row * tiled + column] != expected;
        }
    CHECK(wrong == 0);
}
