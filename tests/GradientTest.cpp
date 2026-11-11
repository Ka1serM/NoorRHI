#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <vector>

#include "Gradient.h"
#include "Gradient.slang.h"
#include "Network.h"
#include "Network.slang.h"
#include "Test.h"

using namespace NoorRHI;

namespace {

constexpr std::uint32_t size = 16;
constexpr std::array gaussians{float4{5.0f, 6.0f, 2.5f, 0.8f}, float4{11.0f, 9.0f, 3.0f, 0.6f}};
constexpr std::size_t gradientCount = gaussians.size() * 4;

struct Scene {
    Buffer<float4> gaussians;
    Buffer<float> target;
    Buffer<Atomic<float>> gradients;
};

float getTarget(const std::uint32_t x, const std::uint32_t y) {
    return 0.25f + 0.2f * std::sin(0.4f * static_cast<float>(x)) * std::cos(0.3f * static_cast<float>(y));
}

std::array<double, gradientCount> getReference() {
    std::array<double, gradientCount> gradients{};
    for (std::uint32_t y = 0; y < size; ++y)
        for (std::uint32_t x = 0; x < size; ++x) {
            const double px = x + 0.5;
            const double py = y + 0.5;
            std::array<double, gaussians.size()> falloffs{};
            double value = 0.0;
            for (std::size_t index = 0; index < gaussians.size(); ++index) {
                const float4& gaussian = gaussians[index];
                const double distance = (px - gaussian.x) * (px - gaussian.x) + (py - gaussian.y) * (py - gaussian.y);
                falloffs[index] = std::exp(-distance / (2.0 * gaussian.z * gaussian.z));
                value += gaussian.w * falloffs[index];
            }
            const double residual = 2.0 * (value - getTarget(x, y));
            for (std::size_t index = 0; index < gaussians.size(); ++index) {
                const float4& gaussian = gaussians[index];
                const double splat = gaussian.w * falloffs[index] * residual;
                const double variance = double{gaussian.z} * gaussian.z;
                const double distance = (px - gaussian.x) * (px - gaussian.x) + (py - gaussian.y) * (py - gaussian.y);
                gradients[index * 4] += splat * (px - gaussian.x) / variance;
                gradients[index * 4 + 1] += splat * (py - gaussian.y) / variance;
                gradients[index * 4 + 2] += splat * distance / (variance * gaussian.z);
                gradients[index * 4 + 3] += falloffs[index] * residual;
            }
        }
    return gradients;
}

std::expected<void, Error> openScene(Device& device, Scene& scene) {
    std::vector<float> target(size * size);
    for (std::uint32_t y = 0; y < size; ++y)
        for (std::uint32_t x = 0; x < size; ++x)
            target[y * size + x] = getTarget(x, y);
    for (std::expected<void, Error> step :
         {scene.gaussians.open(device, gaussians.size()), scene.target.open(device, target.size()), scene.gradients.open(device, gradientCount)})
        if (!step)
            return step;
    for (std::expected<void, Error> step :
         {scene.gaussians.upload(gaussians), scene.target.upload(target), scene.gradients.upload(std::array<Atomic<float>, gradientCount>{})})
        if (!step)
            return step;
    return {};
}

Gradient::Arguments getArguments(const Scene& scene) {
    return {scene.gaussians.getPtr(), scene.target.getPtr(), scene.gradients.getPtr(), size, size, static_cast<uint>(gaussians.size())};
}

void checkGradients(const Scene& scene) {
    std::array<Atomic<float>, gradientCount> gradients{};
    REQUIRE_OK(scene.gradients.download(gradients));
    const std::array<double, gradientCount> reference = getReference();
    for (std::size_t index = 0; index < gradientCount; ++index)
        CHECK(std::abs(gradients[index].value - reference[index]) <= 1e-3 * std::max(1.0, std::abs(reference[index])));
}

}

TEST_CASE("autodiff gradients accumulate with float atomics in a compute shader") {
    static_assert(sizeof(Gradient::Arguments) == 36 && offsetof(Gradient::Arguments, width) == 24);
    Device* device = Test::getDevice();
    REQUIRE(device);
    Scene scene;
    REQUIRE_OK(openScene(*device, scene));
    ComputePipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, Gradient::computeMain));

    CommandList list(*device);
    list.dispatch(pipeline, {size, size}, getArguments(scene));
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    checkGradients(scene);
}

TEST_CASE("autodiff gradients accumulate with float atomics in a fragment shader") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    Scene scene;
    REQUIRE_OK(openScene(*device, scene));
    GraphicsPipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, {.vertex = Gradient::vertMain, .fragment = Gradient::fragMain, .state = {.cull = CullMode::None}}));
    Image target;
    REQUIRE_OK(target.open(*device, {.width = size, .height = size, .usage = ImageUsage::ColorAttachment}));

    CommandList list(*device);
    list.beginRender({.color = &target});
    list.draw(pipeline, 3, 1, getArguments(scene));
    list.endRender();
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    checkGradients(scene);
}

TEST_CASE("autodiff scatters the gradients of buffer weights through a custom backward load") {
    constexpr std::uint32_t sampleCount = 256;
    constexpr std::uint32_t weightOffset = Network::hidden * Network::inputs;
    constexpr std::uint32_t outputOffset = weightOffset + Network::hidden;
    Device* device = Test::getDevice();
    REQUIRE(device);
    std::vector<float> parameters(Network::parameterCount);
    for (std::uint32_t index = 0; index < parameters.size(); ++index)
        parameters[index] = static_cast<float>(index * 7 % 13) * 0.1f - 0.6f;
    std::vector<float4> samples(sampleCount);
    std::vector<float> targets(sampleCount);
    for (std::uint32_t index = 0; index < sampleCount; ++index) {
        const auto value = static_cast<float>(index);
        samples[index] = {std::sin(value), std::cos(value * 0.7f), value / sampleCount, 1.0f - value / sampleCount};
        targets[index] = std::sin(value * 0.3f);
    }
    std::vector<double> reference(Network::parameterCount);
    for (std::uint32_t sample = 0; sample < sampleCount; ++sample) {
        const std::array<double, Network::inputs> input{samples[sample].x, samples[sample].y, samples[sample].z, samples[sample].w};
        std::array<double, Network::hidden> sums{};
        double output = parameters[Network::parameterCount - 1];
        for (std::uint32_t row = 0; row < Network::hidden; ++row) {
            sums[row] = parameters[weightOffset + row];
            for (std::uint32_t column = 0; column < Network::inputs; ++column)
                sums[row] += double{parameters[row * Network::inputs + column]} * input[column];
            output += parameters[outputOffset + row] * std::max(sums[row], 0.0);
        }
        const double error = 2.0 * (output - targets[sample]);
        reference[Network::parameterCount - 1] += error;
        for (std::uint32_t row = 0; row < Network::hidden; ++row) {
            reference[outputOffset + row] += error * std::max(sums[row], 0.0);
            if (sums[row] <= 0.0)
                continue;
            const double hidden = error * parameters[outputOffset + row];
            reference[weightOffset + row] += hidden;
            for (std::uint32_t column = 0; column < Network::inputs; ++column)
                reference[row * Network::inputs + column] += hidden * input[column];
        }
    }
    Buffer<float4> sampleBuffer;
    Buffer<float> targetBuffer;
    Buffer<float> parameterBuffer;
    Buffer<Atomic<float>> gradientBuffer;
    REQUIRE_OK(sampleBuffer.open(*device, sampleCount));
    REQUIRE_OK(targetBuffer.open(*device, sampleCount));
    REQUIRE_OK(parameterBuffer.open(*device, Network::parameterCount));
    REQUIRE_OK(gradientBuffer.open(*device, Network::parameterCount));
    REQUIRE_OK(sampleBuffer.upload(samples));
    REQUIRE_OK(targetBuffer.upload(targets));
    REQUIRE_OK(parameterBuffer.upload(parameters));
    REQUIRE_OK(gradientBuffer.upload(std::vector<Atomic<float>>(Network::parameterCount)));
    ComputePipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, Network::main));

    CommandList list(*device);
    list.dispatch(pipeline, {sampleCount},
                  Network::Arguments{sampleBuffer.getPtr(), targetBuffer.getPtr(), parameterBuffer.getPtr(), gradientBuffer.getPtr(), sampleCount});
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::vector<Atomic<float>> gradients(Network::parameterCount);
    REQUIRE_OK(gradientBuffer.download(gradients));
    for (std::uint32_t index = 0; index < Network::parameterCount; ++index)
        CHECK(std::abs(gradients[index].value - reference[index]) <= 1e-3 * std::max(1.0, std::abs(reference[index])));
}
