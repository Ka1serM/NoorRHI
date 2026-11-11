#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "NoorRHI/NoorRHI.h"

namespace Test {

struct Case {
    std::string_view name;
    void (*run)();
};

inline std::vector<Case>& getCases() {
    static std::vector<Case> cases;
    return cases;
}

inline int& getFailures() {
    static int failures = 0;
    return failures;
}

inline bool check(const bool condition, const std::string_view expression, const char* file, const int line) {
    if (condition)
        return true;
    std::fprintf(stderr, "%s:%d: check failed: %.*s\n", file, line, static_cast<int>(expression.size()), expression.data());
    ++getFailures();
    return false;
}

template <class T>
bool checkResult(const std::expected<T, NoorRHI::Error>& result, const std::string_view expression, const char* file, const int line) {
    if (result)
        return true;
    std::fprintf(stderr, "%s:%d: %.*s failed: %s\n", file, line, static_cast<int>(expression.size()), expression.data(), result.error().message.c_str());
    ++getFailures();
    return false;
}

struct Registrar {
    Registrar(const std::string_view name, void (*run)()) { getCases().push_back({name, run}); }
};

inline constexpr bool webGpu = NOORRHI_TEST_WEBGPU;

struct DeviceSlot {
    std::unique_ptr<NoorRHI::Device> device;
    bool opened = false;
};

inline DeviceSlot& getDeviceSlot() {
    static DeviceSlot slot;
    return slot;
}

inline NoorRHI::Device* getDevice() {
    DeviceSlot& slot = getDeviceSlot();
    if (slot.opened)
        return slot.device.get();
    slot.opened = true;
    slot.device = std::make_unique<NoorRHI::Device>();
    const bool hardware = std::getenv("NOORRHI_TEST_FALLBACKS") == nullptr;
    const std::expected<void, NoorRHI::Error> result =
        slot.device->open({.enableValidation = true, .applicationName = "NoorRHI tests", .hardwareRayTracing = hardware, .cooperativeMatrices = hardware});
    if (result)
        return slot.device.get();
    std::fprintf(stderr, "the test device does not open: %s\n", result.error().message.c_str());
    slot.device.reset();
    return nullptr;
}

//! The validation layer frees its state before static destructors run, so the device must close before main returns.
inline void closeDevice() {
    getDeviceSlot().device.reset();
}

inline std::expected<void, NoorRHI::Error> execute(NoorRHI::Device& device, NoorRHI::CommandList&& list) {
    std::expected<NoorRHI::Token, NoorRHI::Error> token = device.submit(std::move(list));
    if (!token)
        return std::unexpected(std::move(token.error()));
    return device.wait(*token);
}

inline std::expected<std::vector<std::uint32_t>, NoorRHI::Error> downloadTexels(const NoorRHI::Image& image) {
    std::vector<std::uint32_t> texels(std::size_t{image.getWidth()} * image.getHeight());
    if (std::expected<void, NoorRHI::Error> downloaded = image.download(std::as_writable_bytes(std::span(texels))); !downloaded)
        return std::unexpected(std::move(downloaded.error()));
    return texels;
}

inline std::expected<void, NoorRHI::Error> uploadTexels(NoorRHI::Image& image, const std::span<const std::uint32_t> texels) {
    return image.upload(std::as_bytes(texels));
}

}

#define NOORRHI_TEST_JOIN2(a, b) a##b
#define NOORRHI_TEST_JOIN(a, b) NOORRHI_TEST_JOIN2(a, b)
#define TEST_CASE(name)                                                                                                                                        \
    static void NOORRHI_TEST_JOIN(testCase, __LINE__)();                                                                                                       \
    static const Test::Registrar NOORRHI_TEST_JOIN(testRegistrar, __LINE__)(name, &NOORRHI_TEST_JOIN(testCase, __LINE__));                                     \
    static void NOORRHI_TEST_JOIN(testCase, __LINE__)()
#define CHECK(condition) Test::check(static_cast<bool>(condition), #condition, __FILE__, __LINE__)
#define REQUIRE(condition)                                                                                                                                     \
    if (!CHECK(condition))                                                                                                                                     \
    return
#define REQUIRE_OK(result)                                                                                                                                     \
    if (!Test::checkResult((result), #result, __FILE__, __LINE__))                                                                                             \
    return
