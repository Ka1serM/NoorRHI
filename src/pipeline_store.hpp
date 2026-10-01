#pragma once

#include <vulkan/vulkan.hpp>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace noorrhi::detail {

// Persists every pipeline it creates as VK_KHR_pipeline_binary data, one file
// per pipeline named by its pipeline key, so a pipeline is stored once and
// recreated without compiling on later runs.
//
// The files live in a folder named by the driver's global key, so a driver or
// GPU that cannot load them never sees them. Loading a pipeline refreshes its
// file's time, and construction deletes the least recently used files of all
// folders above StoreBudget. Several instances may share the folder.
class PipelineStore {
public:
    PipelineStore(vk::Device device, const std::filesystem::path& directory);

    // Creates the pipeline `info` describes with `create`, from stored binaries
    // when there are some and otherwise compiled and then stored. `flags` must
    // be chained into `info`. Pipelines are created without a VkPipelineCache,
    // which pipeline binaries do not allow.
    template <class CreateInfo>
    vk::UniquePipeline create(CreateInfo info, vk::PipelineCreateFlags2CreateInfo& flags,
        const std::function<vk::UniquePipeline(const CreateInfo&)>& create) {
        const std::string key = pipeline_key(&info);
        if (const std::optional<Binaries> stored = load(key)) {
            const vk::PipelineBinaryInfoKHR binary_info(
                static_cast<std::uint32_t>(stored->handles.size()), stored->handles.data(), info.pNext);
            info.pNext = &binary_info;
            return create(info);
        }
        flags.flags |= vk::PipelineCreateFlagBits2::eCaptureDataKHR;
        vk::UniquePipeline pipeline = create(info);
        save(key, *pipeline);
        return pipeline;
    }

private:
    // Budget for the stored files, which grow with every distinct material
    // compiled. Evicting the least recently used ones costs a recompile only
    // if they are needed again.
    static constexpr std::uintmax_t StoreBudget = 512ull << 20;

    struct Binaries {
        vk::Device device;
        std::vector<vk::PipelineBinaryKHR> handles;
        explicit Binaries(vk::Device device) : device(device) {}
        Binaries(Binaries&& other) noexcept;
        Binaries(const Binaries&) = delete;
        Binaries& operator=(const Binaries&) = delete;
        Binaries& operator=(Binaries&&) = delete;
        ~Binaries();
    };

    static Binaries create_binaries(vk::Device device, const VkPipelineBinaryCreateInfoKHR& info);
    std::string pipeline_key(const void* create_info) const;
    std::optional<Binaries> load(const std::string& key) const;
    void save(const std::string& key, vk::Pipeline pipeline);
    void evict_least_recently_used() const;

    vk::Device device_;
    std::filesystem::path root_;
    std::filesystem::path directory_;
    // Tells this instance's partial files from those of others.
    std::uint64_t writer_id_;
    std::atomic<std::uint64_t> next_partial_ = 0;
};

} // namespace noorrhi::detail
