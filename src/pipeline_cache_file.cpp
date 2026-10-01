#include "pipeline_cache_file.hpp"

#include "noorrhi/types.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <vector>

namespace noorrhi::detail {

namespace {
std::vector<char> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
} // namespace

PipelineCacheFile::PipelineCacheFile(const vk::Device device, const std::filesystem::path& directory)
    : device_(device), path_(directory / "pipeline.cache"), writer_id_(std::random_device{}()) {
    std::filesystem::create_directories(directory);
    const std::vector<char> data = read_file(path_);
    cache_ = device_.createPipelineCacheUnique(
        vk::PipelineCacheCreateInfo({}, data.size(), data.data()));
}

PipelineCacheFile::~PipelineCacheFile() {
    try {
        save();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[NoorRHI] %s\n", error.what());
    }
}

void PipelineCacheFile::save() {
    std::lock_guard lock(save_mutex_);
    const std::vector<std::uint8_t> data = device_.getPipelineCacheData(*cache_);
    // Written beside the file and renamed over it, so an instance loading it
    // never reads a partial file.
    const std::filesystem::path partial = path_.string() + "." + std::to_string(writer_id_) + ".partial";
    {
        std::ofstream file(partial, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!file)
            throw Error(ErrorCode::InvalidArgument, "cannot write pipeline cache file " + partial.string());
    }
    std::filesystem::rename(partial, path_);
}

} // namespace noorrhi::detail
