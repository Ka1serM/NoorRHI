#include "NoorRHI/Core/Backend.h"
#include "NoorRHI/Device/Interop.h"
#include "NoorRHI/WebGpu/WebGpu.h"

namespace NoorRHI {

namespace {

std::unexpected<Error> failHardwareRayTracing() {
    return fail(ErrorCode::Unsupported, "WebGPU has no hardware ray tracing");
}

std::unexpected<Error> failInterop() {
    return fail(ErrorCode::Unsupported, "WebGPU has no Vulkan interop");
}

}

namespace Backend {

AccelerationStructureResult buildHardwareBlas(NoorRHI::Device&, std::span<const TriangleGeometry>, BuildMode) {
    return failHardwareRayTracing();
}

AccelerationStructureResult buildHardwareTlas(NoorRHI::Device&, std::span<const InstanceRecord>) {
    return failHardwareRayTracing();
}

AccelerationStructureResult buildHardwareTlas(NoorRHI::Device&, Ptr<InstanceRecord>, std::uint32_t) {
    return failHardwareRayTracing();
}

RayTracingLibraryResult openHardwareLibrary(NoorRHI::Device&, const RayTracingPipelineDesc&, const RayTracingInterface&) {
    return failHardwareRayTracing();
}

RayTracingPipelineResult openHardwarePipeline(NoorRHI::Device&, std::span<const std::shared_ptr<RayTracingLibrary>>, std::span<const std::uint32_t>) {
    return failHardwareRayTracing();
}

}

namespace Interop {

std::expected<DeviceHandles, Error> getDeviceHandles(const Device&) {
    return failInterop();
}

std::expected<std::uintptr_t, Error> getCommandBuffer(const CommandList&) {
    return failInterop();
}

std::expected<std::uintptr_t, Error> getImage(const Image&) {
    return failInterop();
}

std::expected<std::uintptr_t, Error> getImageView(const Image&) {
    return failInterop();
}

std::expected<ExternalImageMemory, Error> exportImageMemory(const Image&) {
    return failInterop();
}

std::expected<ExternalSemaphore, Error> signalExternal(Device&) {
    return failInterop();
}

std::uint32_t getNativeFormat(const ImageFormat format) {
    return static_cast<std::uint32_t>(Backend::getWebGpuFormat(format));
}

}

}
