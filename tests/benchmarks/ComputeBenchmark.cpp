#include <bit>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

#include "Bench.h"
#include "BenchCompute.slang.h"
#include "Benchmark.h"

using namespace NoorRHI;

namespace {

constexpr std::uint32_t dispatchCount = 10000;
constexpr std::uint32_t streamCount = 8u << 20;
constexpr std::uint32_t transferCount = 4u << 20;
constexpr std::uint32_t gemmSize = 2048;

//! The inputs are multiples of 0.5 in [-1, 1], so dropping the low mantissa bits converts them to half exactly.
half toHalf(const float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto exponent = static_cast<std::int32_t>(bits >> 23 & 0xffu);
    const auto sign = static_cast<std::uint16_t>(bits >> 16 & 0x8000u);
    if (exponent == 0)
        return {sign};
    return {static_cast<std::uint16_t>(sign | (exponent - 112) << 10 | (bits >> 13 & 0x3ffu))};
}

float getInput(const std::uint32_t index) {
    return static_cast<float>(static_cast<int>(index * 2654435761u >> 29) - 3) * 0.5f;
}

}

namespace {

Benchmark::Result runDispatches(Device& device, const std::string_view name, const bool concurrent) {
    Buffer<float4> source;
    Buffer<float4> destination;
    TRY(source.open(device, 256));
    TRY(destination.open(device, 256));
    ComputePipeline pipeline;
    TRY(pipeline.open(device, BenchCompute::streamMain));
    std::expected<Benchmark::Timing, Error> timing = Benchmark::measure(device, 10, [&](CommandList& list) {
        if (concurrent)
            list.beginConcurrent();
        for (std::uint32_t index = 0; index < dispatchCount; ++index)
            list.dispatch(pipeline, {256}, Bench::StreamArguments{source.getPtr(), destination.getPtr(), 256 - index % 2});
        if (concurrent)
            list.endConcurrent();
    });
    if (!timing)
        return std::unexpected(std::move(timing.error()));
    char extra[96];
    std::snprintf(extra, sizeof(extra), "record %.3f us/dispatch, gpu %.3f us/dispatch", timing->recordMs * 1000.0 / dispatchCount,
                  timing->gpuMs * 1000.0 / dispatchCount);
    Benchmark::report(name, *timing, extra);
    return {};
}

}

BENCHMARK("compute: 10k dependent small dispatches") {
    return runDispatches(device, "compute: 10k dependent small dispatches", false);
}

BENCHMARK("compute: 10k concurrent small dispatches") {
    return runDispatches(device, "compute: 10k concurrent small dispatches", true);
}

BENCHMARK("compute: stream 128 MB through pointers") {
    Buffer<float4> source;
    Buffer<float4> destination;
    TRY(source.open(device, streamCount));
    TRY(destination.open(device, streamCount));
    ComputePipeline pipeline;
    TRY(pipeline.open(device, BenchCompute::streamMain));
    std::expected<Benchmark::Timing, Error> timing = Benchmark::measure(device, 10, [&](CommandList& list) {
        list.dispatch(pipeline, {streamCount}, Bench::StreamArguments{source.getPtr(), destination.getPtr(), streamCount});
    });
    if (!timing)
        return std::unexpected(std::move(timing.error()));
    char extra[64];
    std::snprintf(extra, sizeof(extra), "%.1f GB/s", 2.0 * streamCount * sizeof(float4) / timing->gpuMs / 1.0e6);
    Benchmark::report("compute: stream 128 MB through pointers", *timing, extra);
    return {};
}

BENCHMARK("transfer: upload and download 64 MB") {
    Buffer<float4> buffer;
    TRY(buffer.open(device, transferCount));
    std::vector<float4> data(transferCount, float4{1.0f, 2.0f, 3.0f, 4.0f});
    std::expected<double, Error> upload = Benchmark::measureWall(5, [&] { return buffer.upload(data); });
    if (!upload)
        return std::unexpected(std::move(upload.error()));
    std::expected<double, Error> download = Benchmark::measureWall(5, [&] { return buffer.download(data); });
    if (!download)
        return std::unexpected(std::move(download.error()));
    char extra[96];
    const double bytes = static_cast<double>(transferCount) * sizeof(float4);
    std::snprintf(extra, sizeof(extra), "upload %.2f ms (%.1f GB/s), download %.2f ms (%.1f GB/s)", *upload, bytes / *upload / 1.0e6, *download,
                  bytes / *download / 1.0e6);
    Benchmark::Timing timing;
    timing.wallMs = *upload + *download;
    Benchmark::report("transfer: upload and download 64 MB", timing, extra);
    return {};
}

namespace {

struct Gemm {
    Buffer<half> a;
    Buffer<half> b;
    Buffer<float> c;
};

Benchmark::Result openGemm(Device& device, Gemm& gemm) {
    std::vector<half> a(std::size_t{gemmSize} * gemmSize);
    std::vector<half> b(a.size());
    for (std::uint32_t index = 0; index < a.size(); ++index) {
        a[index] = toHalf(getInput(index));
        b[index] = toHalf(getInput(index + 12345));
    }
    TRY(gemm.a.open(device, a.size()));
    TRY(gemm.b.open(device, b.size()));
    TRY(gemm.c.open(device, a.size()));
    TRY(gemm.a.upload(a));
    TRY(gemm.b.upload(b));
    return gemm.c.upload(std::vector<float>(a.size(), 0.0f));
}

std::uint32_t countWrongProducts(const Gemm& gemm) {
    std::vector<float> c(std::size_t{gemmSize} * gemmSize);
    if (!gemm.c.download(c))
        return gemmSize;
    std::uint32_t wrong = 0;
    for (std::uint32_t sample = 0; sample < 64; ++sample) {
        const std::uint32_t row = sample * 977 % gemmSize;
        const std::uint32_t column = sample * 1931 % gemmSize;
        float expected = 0.0f;
        for (std::uint32_t k = 0; k < gemmSize; ++k)
            expected += getInput(row * gemmSize + k) * getInput(k * gemmSize + column + 12345);
        wrong += c[row * gemmSize + column] != expected;
    }
    return wrong;
}

Benchmark::Result runGemm(Device& device, const std::string_view name, const Shader& shader, const DispatchSize threads) {
    if (!device.getFeatures().shaderFloat16)
        return {};
    Gemm gemm;
    TRY(openGemm(device, gemm));
    ComputePipeline pipeline;
    TRY(pipeline.open(device, shader));
    const Bench::GemmArguments arguments{gemm.a.getPtr(), gemm.b.getPtr(), gemm.c.getPtr(), gemmSize};
    CommandList first(device);
    first.dispatch(pipeline, threads, arguments);
    std::expected<Token, Error> token = device.submit(std::move(first));
    if (!token)
        return std::unexpected(std::move(token.error()));
    TRY(device.wait(*token));
    const std::uint32_t wrong = countWrongProducts(gemm);
    std::expected<Benchmark::Timing, Error> timing = Benchmark::measure(device, 5, [&](CommandList& list) { list.dispatch(pipeline, threads, arguments); });
    if (!timing)
        return std::unexpected(std::move(timing.error()));
    char extra[96];
    std::snprintf(extra, sizeof(extra), "%.2f TFLOPS, %u of 64 sampled results wrong", 2.0 * gemmSize * gemmSize * gemmSize / timing->gpuMs / 1.0e9, wrong);
    Benchmark::report(name, *timing, extra);
    return {};
}

}

BENCHMARK("matrix: 2048^3 half GEMM with the matrix API") {
    return runGemm(device, "matrix: 2048^3 half GEMM with the matrix API", BenchCompute::gemmMain, {32 * (gemmSize / 16), gemmSize / 16});
}

BENCHMARK("matrix: 2048^3 half GEMM with a scalar loop") {
    return runGemm(device, "matrix: 2048^3 half GEMM with a scalar loop", BenchCompute::scalarGemmMain, {gemmSize, gemmSize});
}
