#pragma once

#include <algorithm>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include "Test.h"

class TestWindow final : public NoorRHI::SurfaceProvider {
public:
    TestWindow() = default;
    TestWindow(const TestWindow&) = delete;
    TestWindow& operator=(const TestWindow&) = delete;

    ~TestWindow() override {
        if (window)
            SDL_DestroyWindow(window);
        if (vulkanLoaded)
            SDL_Vulkan_UnloadLibrary();
        if (initialized)
            SDL_Quit();
    }

    std::expected<void, NoorRHI::Error> open(const int width, const int height) {
        if (!SDL_Init(SDL_INIT_VIDEO))
            return failSdl("SDL_Init");
        initialized = true;
        SDL_WindowFlags flags = SDL_WINDOW_RESIZABLE;
        if (!Test::webGpu) {
            if (!SDL_Vulkan_LoadLibrary(nullptr))
                return failSdl("SDL_Vulkan_LoadLibrary");
            vulkanLoaded = true;
            flags |= SDL_WINDOW_VULKAN;
        }
        window = SDL_CreateWindow("NoorRHI tests", width, height, flags);
        if (!window)
            return failSdl("SDL_CreateWindow");
        return {};
    }

    bool resize(const int width, const int height) const {
        if (!SDL_SetWindowSize(window, width, height) || !SDL_SyncWindow(window))
            return false;
        SDL_Event event{};
        while (SDL_PollEvent(&event))
            continue;
        return true;
    }

    std::uintptr_t getInstanceProcAddress() const override { return reinterpret_cast<std::uintptr_t>(SDL_Vulkan_GetVkGetInstanceProcAddr()); }

    std::vector<const char*> getInstanceExtensions() const override {
        Uint32 count = 0;
        const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&count);
        if (!extensions)
            return {};
        return {extensions, extensions + count};
    }

    std::uintptr_t createSurface(const std::uintptr_t instance) const override {
        VkSurfaceKHR surface{};
        if (!SDL_Vulkan_CreateSurface(window, reinterpret_cast<VkInstance>(instance), nullptr, &surface))
            return 0;
        return reinterpret_cast<std::uintptr_t>(surface);
    }

    NoorRHI::SurfaceSource getSurfaceSource() const override {
        using Kind = NoorRHI::SurfaceSource::Kind;
        const SDL_PropertiesID properties = SDL_GetWindowProperties(window);
        if (void* display = SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, nullptr))
            return {.kind = Kind::Wayland,
                    .display = reinterpret_cast<std::uintptr_t>(display),
                    .window = reinterpret_cast<std::uintptr_t>(SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr))};
        if (void* display = SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr))
            return {.kind = Kind::Xlib,
                    .display = reinterpret_cast<std::uintptr_t>(display),
                    .window = static_cast<std::uintptr_t>(SDL_GetNumberProperty(properties, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0))};
        if (void* handle = SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr))
            return {.kind = Kind::Win32,
                    .display = reinterpret_cast<std::uintptr_t>(SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER, nullptr)),
                    .window = reinterpret_cast<std::uintptr_t>(handle)};
        return {};
    }

    std::uint32_t getWidth() const override { return getPixelSize().first; }
    std::uint32_t getHeight() const override { return getPixelSize().second; }

private:
    static std::unexpected<NoorRHI::Error> failSdl(const std::string_view call) {
        return NoorRHI::fail(NoorRHI::ErrorCode::Unsupported, std::string(call) + ": " + SDL_GetError());
    }

    std::pair<std::uint32_t, std::uint32_t> getPixelSize() const {
        int width = 0;
        int height = 0;
        SDL_GetWindowSizeInPixels(window, &width, &height);
        return {static_cast<std::uint32_t>(std::max(width, 0)), static_cast<std::uint32_t>(std::max(height, 0))};
    }

    SDL_Window* window = nullptr;
    bool initialized = false;
    bool vulkanLoaded = false;
};
