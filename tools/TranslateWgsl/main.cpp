#include <cstdint>
#include <filesystem>
#include <fstream>
#include <print>
#include <vector>

#include "NoorRHI/Shaders/WgslTranslator.h"

int main(const int argc, char** argv) {
    if (argc < 3) {
        std::println(stderr, "Usage: NoorRHITranslateWgsl <output directory> <shader.spv>...");
        return 2;
    }
    const std::filesystem::path output = argv[1];
    std::filesystem::create_directories(output);
    int failures = 0;
    for (int i = 2; i < argc; ++i) {
        const std::filesystem::path input = argv[i];
        std::ifstream file(input, std::ios::binary);
        std::vector<std::uint32_t> words(std::filesystem::file_size(input) / sizeof(std::uint32_t));
        file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(words.size() * sizeof(std::uint32_t)));
        const auto wgsl = NoorRHI::translateToWgsl(words);
        if (!wgsl) {
            std::println("FAIL\t{}\t{}", input.filename().string(), wgsl.error().message);
            ++failures;
            continue;
        }
        std::ofstream(output / input.filename().replace_extension(".wgsl")) << *wgsl;
        std::println("OK\t{}", input.filename().string());
    }
    return failures == 0 ? 0 : 1;
}
