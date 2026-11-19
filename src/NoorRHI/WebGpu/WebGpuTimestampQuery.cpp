#include "NoorRHI/Commands/TimestampQuery.h"
#include "NoorRHI/Device/Device.h"
#include "NoorRHI/WebGpu/WebGpu.h"

namespace NoorRHI {

std::expected<void, Error> TimestampQuery::open(Device& device) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    auto& owner = static_cast<Backend::WebGpuDevice&>(*device.getBackend());
    auto result = std::make_shared<Backend::WebGpuTimestampQuery>();
    result->device = owner.shared_from_this();
    if (owner.timestamps) {
        std::lock_guard lock(owner.mutex);
        owner.pushErrorScope();
        const wgpu::QuerySetDescriptor querySet{.type = wgpu::QueryType::Timestamp, .count = Backend::timestampSlotCount};
        result->querySet = owner.device.CreateQuerySet(&querySet);
        const wgpu::BufferDescriptor resolve{.usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc, .size = 16};
        result->resolve = owner.device.CreateBuffer(&resolve);
        const wgpu::BufferDescriptor readback{.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst, .size = 16};
        result->readback = owner.device.CreateBuffer(&readback);
        if (std::expected<void, Error> created = owner.popErrorScope("creating a WebGPU timestamp query"); !created)
            return created;
    }
    backend = std::move(result);
    return {};
}

namespace Backend {

double WebGpuTimestampQuery::getMilliseconds() {
    if (querySet)
        device->instance.ProcessEvents();
    return milliseconds;
}

}

}
