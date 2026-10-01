#include "pipeline_store.hpp"

#include "noorrhi/types.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <random>

namespace noorrhi::detail {

namespace {
void check(const VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw Error(ErrorCode::ShaderCreationFailed,
            std::string(operation) + " failed: " + vk::to_string(static_cast<vk::Result>(result)));
}

std::string key_name(const VkPipelineBinaryKeyKHR& key) {
    constexpr char digits[] = "0123456789abcdef";
    std::string name;
    for (std::uint32_t i = 0; i < key.keySize; ++i) {
        name += digits[key.key[i] >> 4];
        name += digits[key.key[i] & 15];
    }
    return name;
}

template <class T>
void write_value(std::ofstream& file, const T& value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <class T>
T read_value(std::ifstream& file) {
    T value{};
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

} // namespace

PipelineStore::Binaries PipelineStore::create_binaries(const vk::Device device,
    const VkPipelineBinaryCreateInfoKHR& info) {
    VkPipelineBinaryHandlesInfoKHR handles{VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
    check(VULKAN_HPP_DEFAULT_DISPATCHER.vkCreatePipelineBinariesKHR(device, &info, nullptr, &handles),
        "vkCreatePipelineBinariesKHR");
    PipelineStore::Binaries binaries(device);
    binaries.handles.resize(handles.pipelineBinaryCount);
    handles.pPipelineBinaries = reinterpret_cast<VkPipelineBinaryKHR*>(binaries.handles.data());
    const VkResult result = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreatePipelineBinariesKHR(
        device, &info, nullptr, &handles);
    if (result != VK_SUCCESS)
        binaries.handles.clear();
    check(result, "vkCreatePipelineBinariesKHR");
    return binaries;
}

PipelineStore::Binaries::Binaries(Binaries&& other) noexcept
    : device(other.device), handles(std::move(other.handles)) {}

PipelineStore::Binaries::~Binaries() {
    for (const vk::PipelineBinaryKHR handle : handles)
        VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyPipelineBinaryKHR(device, handle, nullptr);
}

PipelineStore::PipelineStore(const vk::Device device, const std::filesystem::path& directory)
    : device_(device), root_(directory), writer_id_(std::random_device{}()) {
    VkPipelineBinaryKeyKHR global{VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
    check(VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPipelineKeyKHR(device_, nullptr, &global), "vkGetPipelineKeyKHR");
    directory_ = root_ / key_name(global);
    std::filesystem::create_directories(directory_);
    evict_least_recently_used();
}

std::string PipelineStore::pipeline_key(const void* create_info) const {
    // The header declares pNext mutable, but vkGetPipelineKeyKHR only reads it.
    const VkPipelineCreateInfoKHR info{VK_STRUCTURE_TYPE_PIPELINE_CREATE_INFO_KHR,
        const_cast<void*>(create_info)};
    VkPipelineBinaryKeyKHR key{VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
    check(VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPipelineKeyKHR(device_, &info, &key), "vkGetPipelineKeyKHR");
    return key_name(key);
}

std::optional<PipelineStore::Binaries> PipelineStore::load(const std::string& key) const {
    const std::filesystem::path path = directory_ / key;
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return std::nullopt;
    const auto count = read_value<std::uint32_t>(file);
    std::vector<VkPipelineBinaryKeyKHR> keys(count, {VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR});
    std::vector<std::vector<std::uint8_t>> data(count);
    for (std::uint32_t i = 0; i < count && file; ++i) {
        keys[i].keySize = read_value<std::uint32_t>(file);
        if (keys[i].keySize > VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR)
            throw Error(ErrorCode::InvalidState, "pipeline store file is malformed: " + path.string());
        file.read(reinterpret_cast<char*>(keys[i].key), keys[i].keySize);
        data[i].resize(read_value<std::uint64_t>(file));
        file.read(reinterpret_cast<char*>(data[i].data()), static_cast<std::streamsize>(data[i].size()));
    }
    if (!file)
        throw Error(ErrorCode::InvalidState, "pipeline store file is truncated: " + path.string());
    // Fails only when another instance evicted the file after it was
    // opened; the data read is still valid.
    std::error_code touched;
    std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), touched);

    std::vector<VkPipelineBinaryDataKHR> blobs;
    for (std::vector<std::uint8_t>& blob : data)
        blobs.push_back({blob.size(), blob.data()});
    const VkPipelineBinaryKeysAndDataKHR keys_and_data{count, keys.data(), blobs.data()};
    VkPipelineBinaryCreateInfoKHR info{VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR};
    info.pKeysAndDataInfo = &keys_and_data;
    return create_binaries(device_, info);
}

void PipelineStore::save(const std::string& key, const vk::Pipeline pipeline) {
    VkPipelineBinaryCreateInfoKHR info{VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR};
    info.pipeline = pipeline;
    const Binaries binaries = create_binaries(device_, info);
    const VkReleaseCapturedPipelineDataInfoKHR release{
        VK_STRUCTURE_TYPE_RELEASE_CAPTURED_PIPELINE_DATA_INFO_KHR, nullptr, pipeline};
    check(VULKAN_HPP_DEFAULT_DISPATCHER.vkReleaseCapturedPipelineDataKHR(device_, &release, nullptr),
        "vkReleaseCapturedPipelineDataKHR");

    // Written beside the file and renamed over it, so a pipeline another
    // thread or instance loads is never partially written. Each writer, in
    // any running instance, has partial files of its own.
    const std::filesystem::path partial = directory_ / (key + "." + std::to_string(writer_id_) + "."
        + std::to_string(next_partial_++) + ".partial");
    {
        std::ofstream file(partial, std::ios::binary | std::ios::trunc);
        write_value(file, static_cast<std::uint32_t>(binaries.handles.size()));
        for (const vk::PipelineBinaryKHR handle : binaries.handles) {
            const VkPipelineBinaryDataInfoKHR data_info{
                VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR, nullptr, handle};
            VkPipelineBinaryKeyKHR binary_key{VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
            std::size_t size = 0;
            check(VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPipelineBinaryDataKHR(
                device_, &data_info, &binary_key, &size, nullptr), "vkGetPipelineBinaryDataKHR");
            std::vector<std::uint8_t> data(size);
            check(VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPipelineBinaryDataKHR(
                device_, &data_info, &binary_key, &size, data.data()), "vkGetPipelineBinaryDataKHR");
            write_value(file, binary_key.keySize);
            file.write(reinterpret_cast<const char*>(binary_key.key), binary_key.keySize);
            write_value(file, static_cast<std::uint64_t>(size));
            file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(size));
        }
        if (!file)
            throw Error(ErrorCode::InvalidArgument, "cannot write pipeline store file " + partial.string());
    }
    std::error_code renamed;
    std::filesystem::rename(partial, directory_ / key, renamed);
    if (!renamed)
        return;
    // Windows refuses to replace a file another instance has open. That
    // instance stored the same pipeline, so this copy is not needed.
    std::filesystem::remove(partial);
    if (!std::filesystem::exists(directory_ / key))
        throw std::filesystem::filesystem_error("cannot store pipeline", partial, directory_ / key, renamed);
}

void PipelineStore::evict_least_recently_used() const {
    struct Stored {
        std::filesystem::path path;
        std::uintmax_t size;
        std::filesystem::file_time_type used;
    };
    // Any older than this was left by a process that ended while writing;
    // a younger one may still be written by a running instance.
    constexpr auto AbandonedPartial = std::chrono::hours(1);
    const auto now = std::filesystem::file_time_type::clock::now();
    // Covers the folders of every driver and GPU. A folder another GPU uses
    // stays while it is used, and the one a replaced driver left ages out.
    std::vector<Stored> stored;
    std::uintmax_t total = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root_)) {
        std::error_code error;
        if (!entry.is_regular_file(error))
            continue;
        const auto used = entry.last_write_time(error);
        const auto size = entry.file_size(error);
        // Another instance removed or replaced it while this one listed it.
        if (error)
            continue;
        if (entry.path().extension() == ".partial") {
            if (now - used > AbandonedPartial)
                std::filesystem::remove(entry.path(), error);
            continue;
        }
        stored.push_back({entry.path(), size, used});
        total += size;
    }
    std::ranges::sort(stored, {}, &Stored::used);
    for (const Stored& file : stored) {
        if (total <= StoreBudget)
            return;
        // Another instance may hold the file open, which Windows does not
        // let us delete; it goes on a later start.
        std::error_code error;
        if (std::filesystem::remove(file.path, error))
            total -= file.size;
    }
}

} // namespace noorrhi::detail
