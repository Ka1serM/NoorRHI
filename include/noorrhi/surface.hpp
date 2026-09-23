#pragma once

#include <cstdint>
#include <vector>

namespace noorrhi {

// The window system's side of presentation. The library never links a
// windowing toolkit, so the application implements this for its window and
// hands it to DeviceConfig::presentation and to each Swapchain. Every value
// crossing the boundary is either a plain integer or an opaque handle widened
// to uintptr_t, which keeps Vulkan types out of the public headers exactly as
// the rest of the API does.
//
// A Device constructed without a provider is headless and cannot present.
class SurfaceProvider {
public:
    virtual ~SurfaceProvider() = default;

    // PFN_vkGetInstanceProcAddr for the loader the window system already
    // initialised. SDL, GLFW and Qt each own their loader, and creating a
    // surface against a different one is undefined.
    virtual std::uintptr_t instance_proc_address() const = 0;

    // Instance extensions the platform needs for presentation - typically
    // VK_KHR_surface plus one platform-specific extension. The library adds
    // its own instance extensions to these.
    virtual std::vector<const char*> instance_extensions() const = 0;

    // Create a presentation surface for `instance` and return the VkSurfaceKHR
    // widened to uintptr_t. The library takes ownership and destroys it.
    virtual std::uintptr_t create_surface(std::uintptr_t instance) const = 0;

    // Current drawable size in pixels. Consulted whenever the swapchain is
    // (re)created, so a provider should report the live framebuffer size
    // rather than a cached one.
    virtual std::uint32_t width() const = 0;
    virtual std::uint32_t height() const = 0;
};

} // namespace noorrhi
