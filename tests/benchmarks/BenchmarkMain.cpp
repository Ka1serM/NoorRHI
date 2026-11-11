#include <cstdio>
#include <cstdlib>
#include <string_view>

#include "Benchmark.h"

int main(const int argumentCount, char** arguments) {
    const std::string_view filter = argumentCount > 1 ? arguments[1] : "";
    const bool hardware = std::getenv("NOORRHI_TEST_FALLBACKS") == nullptr;
    NoorRHI::Device device;
    if (std::expected<void, NoorRHI::Error> opened = device.open({.applicationName = "NoorRHI benchmarks",
                                                                  .pipelineDirectory = "NoorRHIBenchmarkCache",
                                                                  .hardwareRayTracing = hardware,
                                                                  .cooperativeMatrices = hardware});
        !opened) {
        std::printf("the device does not open: %s\n", opened.error().message.c_str());
        return 1;
    }
    const NoorRHI::DeviceFeatures& features = device.getFeatures();
    std::printf("DEVICE|%s|%s|hardwareRayTracing=%d|cooperativeMatrices=%zu|subgroups=%d|float16=%d\n", device.getInfo().name.c_str(),
                device.getInfo().api == NoorRHI::GraphicsApi::Vulkan ? "Vulkan" : "WebGPU", features.hardwareRayTracing,
                features.cooperativeMatrixShapes.size(), features.subgroups, features.shaderFloat16);
    int failures = 0;
    for (const Benchmark::Case& benchmark : Benchmark::getCases()) {
        if (!benchmark.name.contains(filter))
            continue;
        if (Benchmark::Result result = benchmark.run(device); !result) {
            std::printf("ERROR|%.*s|%s\n", static_cast<int>(benchmark.name.size()), benchmark.name.data(), result.error().message.c_str());
            ++failures;
        }
    }
    std::printf("EXIT %d\n", failures);
    return failures == 0 ? 0 : 1;
}
