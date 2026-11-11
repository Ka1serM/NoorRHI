#include "NoorRHI/Pipelines/RayTracing.h"

#include "NoorRHI/Core/Backend.h"

namespace NoorRHI {

namespace {

bool usesHardware(const Device& device) {
    return device.getFeatures().hardwareRayTracing;
}

std::expected<std::vector<InstanceRecord>, Error> makeInstanceRecords(const std::span<const Instance> instances) {
    std::vector<InstanceRecord> records;
    records.reserve(instances.size());
    for (const Instance& instance : instances) {
        std::expected<InstanceRecord, Error> record = Backend::makeInstanceRecord(instance);
        if (!record)
            return std::unexpected(std::move(record.error()));
        records.push_back(*record);
    }
    return records;
}

std::expected<void, Error> checkGeometry(const std::span<const TriangleGeometry> geometry) {
    if (geometry.empty())
        return fail(ErrorCode::InvalidArgument, "a BLAS needs at least one triangle geometry");
    for (const TriangleGeometry& item : geometry) {
        if (!item.positions || !item.indices || item.triangleCount == 0)
            return fail(ErrorCode::InvalidArgument, "a BLAS geometry needs positions, indices and a triangle count");
        if (item.stride < 3 * sizeof(float) || item.stride % sizeof(float) != 0)
            return fail(ErrorCode::InvalidArgument, "the BLAS vertex stride must be a multiple of 4 bytes and hold three floats");
    }
    return {};
}

}

namespace Backend {

std::expected<InstanceRecord, Error> makeInstanceRecord(const Instance& instance) {
    if (!instance.blas || !*instance.blas)
        return fail(ErrorCode::InvalidArgument, "a TLAS instance needs a built BLAS");
    if (instance.customIndex > 0x00ffffffu || instance.shaderBindingTableOffset > 0x00ffffffu)
        return fail(ErrorCode::InvalidArgument, "the custom index and the shader binding table offset of an instance must fit in 24 bits");
    InstanceRecord record{};
    for (std::size_t row = 0; row < 3; ++row)
        for (std::size_t column = 0; column < 4; ++column)
            record.transform[row * 4 + column] = instance.transform.values[row][column];
    record.customIndexAndMask = instance.customIndex | std::uint32_t{instance.mask} << 24;
    record.shaderBindingTableOffsetAndFlags = instance.shaderBindingTableOffset | instanceNoCull << 24;
    record.blasAddress = instance.blas->getAddress();
    return record;
}

}

std::expected<void, Error> AccelerationStructure::adopt(std::expected<std::shared_ptr<Backend::AccelerationStructure>, Error> built) {
    if (!built)
        return std::unexpected(std::move(built.error()));
    address = (*built)->address;
    backend = std::move(*built);
    return {};
}

std::expected<void, Error> AccelerationStructure::build(Device& device, const std::span<const TriangleGeometry> geometry, const BuildMode mode) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    if (std::expected<void, Error> checked = checkGeometry(geometry); !checked)
        return checked;
    return adopt(usesHardware(device) ? Backend::buildHardwareBlas(device, geometry, mode) : Backend::buildSoftwareBlas(device, geometry, mode));
}

std::expected<void, Error> AccelerationStructure::build(Device& device, const std::span<const Instance> instances) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    if (instances.empty())
        return fail(ErrorCode::InvalidArgument, "a TLAS needs at least one instance");
    std::expected<std::vector<InstanceRecord>, Error> records = makeInstanceRecords(instances);
    if (!records)
        return std::unexpected(std::move(records.error()));
    return adopt(usesHardware(device) ? Backend::buildHardwareTlas(device, *records) : Backend::buildSoftwareTlas(device, *records));
}

std::expected<void, Error> AccelerationStructure::build(Device& device, const Ptr<InstanceRecord> records, const std::uint32_t count) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    if (!records || count == 0)
        return fail(ErrorCode::InvalidArgument, "a TLAS needs instance records and at least one instance");
    return adopt(usesHardware(device) ? Backend::buildHardwareTlas(device, records, count) : Backend::buildSoftwareTlas(device, records, count));
}

std::expected<void, Error> AccelerationStructure::update() {
    if (!backend)
        return fail(ErrorCode::InvalidState, "update needs a built acceleration structure");
    return backend->refit();
}

std::expected<void, Error> AccelerationStructure::update(const std::span<const Instance> instances) {
    if (!backend)
        return fail(ErrorCode::InvalidState, "update needs a built acceleration structure");
    std::expected<std::vector<InstanceRecord>, Error> records = makeInstanceRecords(instances);
    if (!records)
        return std::unexpected(std::move(records.error()));
    return backend->update(*records);
}

std::expected<void, Error> AccelerationStructure::update(const Ptr<InstanceRecord> records, const std::uint32_t count) {
    if (!backend)
        return fail(ErrorCode::InvalidState, "update needs a built acceleration structure");
    if (!records)
        return fail(ErrorCode::InvalidArgument, "the TLAS instance records have no address");
    return backend->update(records, count);
}

std::expected<void, Error> RayTracingLibrary::open(Device& device, const RayTracingPipelineDesc& desc, const RayTracingInterface& interface) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    Backend::RayTracingLibraryResult opened =
        usesHardware(device) ? Backend::openHardwareLibrary(device, desc, interface) : Backend::openSoftwareLibrary(device, desc, interface);
    if (!opened)
        return std::unexpected(std::move(opened.error()));
    (*opened)->interface = interface;
    backend = std::move(*opened);
    return {};
}

std::expected<void, Error> RayTracingPipeline::open(Device& device, const std::span<const RayTracingLibrary> libraries,
                                                    const std::span<const std::uint32_t> hitGroups) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    if (libraries.empty())
        return fail(ErrorCode::InvalidArgument, "a ray tracing pipeline needs at least one library");
    std::vector<std::shared_ptr<Backend::RayTracingLibrary>> backends;
    backends.reserve(libraries.size());
    for (const RayTracingLibrary& library : libraries) {
        if (!library)
            return fail(ErrorCode::InvalidArgument, "a ray tracing library is not open");
        if (library.backend->interface != libraries.front().backend->interface)
            return fail(ErrorCode::InvalidArgument, "linked ray tracing libraries must share one interface");
        backends.push_back(library.backend);
    }
    Backend::RayTracingPipelineResult opened =
        usesHardware(device) ? Backend::openHardwarePipeline(device, backends, hitGroups) : Backend::openSoftwarePipeline(device, backends, hitGroups);
    if (!opened)
        return std::unexpected(std::move(opened.error()));
    backend = std::move(*opened);
    return {};
}

std::expected<void, Error> RayTracingPipeline::open(const RayTracingPipeline& linked, const std::span<const std::uint32_t> hitGroups) {
    if (!linked)
        return fail(ErrorCode::InvalidArgument, "the linked ray tracing pipeline is not open");
    Backend::RayTracingPipelineResult opened = linked.backend->relink(hitGroups);
    if (!opened)
        return std::unexpected(std::move(opened.error()));
    backend = std::move(*opened);
    return {};
}

}
