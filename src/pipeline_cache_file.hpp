#pragma once

#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <filesystem>
#include <mutex>

namespace noorrhi::detail {

// One VkPipelineCache persisted to a file, so pipelines are not compiled anew
// on every run.
// The driver checks the file's header itself and starts empty when the data
// belongs to another driver or GPU. Several instances may share the file.
class PipelineCacheFile {
public:
    PipelineCacheFile(vk::Device device, const std::filesystem::path& directory);
    // Saves, reporting a failure on stderr since a destructor cannot throw.
    ~PipelineCacheFile();
    PipelineCacheFile(const PipelineCacheFile&) = delete;
    PipelineCacheFile& operator=(const PipelineCacheFile&) = delete;

    vk::PipelineCache handle() const { return *cache_; }
    // Writes everything compiled so far, so it survives a crash.
    void save();

private:
    vk::Device device_;
    std::filesystem::path path_;
    // Tells this instance's partial files from those of others.
    std::uint64_t writer_id_;
    std::mutex save_mutex_;
    vk::UniquePipelineCache cache_;
};

} // namespace noorrhi::detail
