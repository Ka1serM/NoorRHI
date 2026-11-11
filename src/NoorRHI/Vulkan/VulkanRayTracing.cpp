#include <algorithm>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <string>
#include <tuple>

#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Pipelines/RayTracing.h"
#include "NoorRHI/Vulkan/Vulkan.h"

namespace NoorRHI {

namespace {

constexpr vk::BuildAccelerationStructureFlagsKHR topLevelFlags =
    vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace | vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;

std::uint64_t alignUp(const std::uint64_t value, const std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

std::expected<vk::Pipeline, Error> createPipeline(Backend::VulkanDevice& owner, const vk::RayTracingPipelineCreateInfoKHR& info,
                                                  const vk::PipelineCache cache) {
    //! Some drivers keep ray tracing compiler work alive after creation returns, so creations must not overlap.
    std::lock_guard lock(owner.rayTracingCompileMutex);
    //! NVIDIA 595 crashes in vkCmdTraceRaysKHR when the pipeline was created with a deferred operation.
    VkPipeline pipeline = VK_NULL_HANDLE;
    const auto result = static_cast<vk::Result>(VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateRayTracingPipelinesKHR(
        *owner.device, VK_NULL_HANDLE, cache, 1, reinterpret_cast<const VkRayTracingPipelineCreateInfoKHR*>(&info), nullptr, &pipeline));
    if (result != vk::Result::eSuccess)
        return Backend::failVulkan(result, "vkCreateRayTracingPipelinesKHR");
    return vk::Pipeline(pipeline);
}

struct ShaderGroups {
    explicit ShaderGroups(const Backend::VulkanDevice& device) : device(device) {}

    const Backend::VulkanDevice& device;
    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> infos;
    std::vector<Backend::ShaderGroup> groups;
    std::vector<vk::UniqueShaderModule> modules;
    std::deque<std::string> entryPoints;
    std::map<std::tuple<const std::byte*, std::string_view, ShaderStage>, std::uint32_t> stageIndices;
    std::optional<Error> error;

    std::uint32_t addStage(const Shader& shader, const ShaderStage stage, const vk::ShaderStageFlagBits flag) {
        if (error)
            return vk::ShaderUnusedKHR;
        if (std::expected<void, Error> checked = Backend::checkShader(shader, stage); !checked) {
            error = std::move(checked.error());
            return vk::ShaderUnusedKHR;
        }
        const auto key = std::tuple(shader.getSource(), shader.entryPoint, stage);
        if (const auto existing = stageIndices.find(key); existing != stageIndices.end())
            return existing->second;
        std::expected<vk::UniqueShaderModule, Error> module = Backend::createShaderModule(device, shader);
        if (!module) {
            error = std::move(module.error());
            return vk::ShaderUnusedKHR;
        }
        const auto index = static_cast<std::uint32_t>(stages.size());
        stageIndices.emplace(key, index);
        stages.emplace_back(vk::PipelineShaderStageCreateFlags{}, flag, **module, entryPoints.emplace_back(shader.entryPoint).c_str());
        modules.push_back(std::move(*module));
        return index;
    }

    void addGeneral(const Shader& shader, const ShaderStage stage, const vk::ShaderStageFlagBits flag, const Backend::GroupKind kind) {
        const std::uint32_t index = addStage(shader, stage, flag);
        infos.emplace_back(vk::RayTracingShaderGroupTypeKHR::eGeneral, index, vk::ShaderUnusedKHR, vk::ShaderUnusedKHR, vk::ShaderUnusedKHR);
        groups.push_back({kind, kind == Backend::GroupKind::Raygen ? shader : Shader{}});
    }
};

ShaderGroups collectGroups(const Backend::VulkanDevice& device, const RayTracingPipelineDesc& desc) {
    ShaderGroups result{device};
    for (const Shader& shader : desc.raygen)
        result.addGeneral(shader, ShaderStage::RayGeneration, vk::ShaderStageFlagBits::eRaygenKHR, Backend::GroupKind::Raygen);
    for (const Shader& shader : desc.miss)
        result.addGeneral(shader, ShaderStage::Miss, vk::ShaderStageFlagBits::eMissKHR, Backend::GroupKind::Miss);
    const std::size_t hitCount = std::max({desc.closestHit.size(), desc.anyHit.size(), desc.intersection.size()});
    for (std::size_t index = 0; index < hitCount; ++index) {
        const bool procedural = index < desc.intersection.size();
        std::uint32_t closest = vk::ShaderUnusedKHR;
        std::uint32_t any = vk::ShaderUnusedKHR;
        std::uint32_t intersection = vk::ShaderUnusedKHR;
        if (index < desc.closestHit.size() && desc.closestHit[index])
            closest = result.addStage(desc.closestHit[index], ShaderStage::ClosestHit, vk::ShaderStageFlagBits::eClosestHitKHR);
        if (index < desc.anyHit.size() && desc.anyHit[index])
            any = result.addStage(desc.anyHit[index], ShaderStage::AnyHit, vk::ShaderStageFlagBits::eAnyHitKHR);
        if (procedural)
            intersection = result.addStage(desc.intersection[index], ShaderStage::Intersection, vk::ShaderStageFlagBits::eIntersectionKHR);
        result.infos.emplace_back(procedural ? vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup : vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                                  vk::ShaderUnusedKHR, closest, any, intersection);
        result.groups.push_back({Backend::GroupKind::Hit, Shader{}});
    }
    for (const Shader& shader : desc.callable)
        result.addGeneral(shader, ShaderStage::Callable, vk::ShaderStageFlagBits::eCallableKHR, Backend::GroupKind::Callable);
    return result;
}

std::expected<void, Error> buildShaderBindingTable(Backend::VulkanRayTracingPipeline& result, const std::span<const std::uint32_t> hitGroups) {
    Backend::VulkanDevice& owner = *result.device;
    const std::vector<Backend::ShaderGroup>& groups = result.linked->groups;
    const std::uint32_t handleSize = owner.rayTracingProperties.shaderGroupHandleSize;
    const std::uint64_t baseAlignment = std::max(owner.rayTracingProperties.shaderGroupBaseAlignment, 1u);
    const std::uint64_t stride = alignUp(alignUp(handleSize, std::max(owner.rayTracingProperties.shaderGroupHandleAlignment, 1u)), baseAlignment);

    const auto collect = [&groups](const Backend::GroupKind kind) {
        std::vector<std::size_t> indices;
        for (std::size_t index = 0; index < groups.size(); ++index)
            if (groups[index].kind == kind)
                indices.push_back(index);
        return indices;
    };
    const std::vector<std::size_t> raygenRecords = collect(Backend::GroupKind::Raygen);
    const std::vector<std::size_t> missRecords = collect(Backend::GroupKind::Miss);
    const std::vector<std::size_t> hitGroupIndices = collect(Backend::GroupKind::Hit);
    const std::vector<std::size_t> callableRecords = collect(Backend::GroupKind::Callable);
    std::vector<std::size_t> hitRecords;
    hitRecords.reserve(hitGroups.size());
    for (const std::uint32_t hitGroup : hitGroups) {
        if (hitGroup >= hitGroupIndices.size())
            return fail(ErrorCode::InvalidArgument, "the shader binding table names hit group " + std::to_string(hitGroup) + " of a pipeline with "
                                                        + std::to_string(hitGroupIndices.size()));
        hitRecords.push_back(hitGroupIndices[hitGroup]);
    }

    std::vector<std::byte> handles(groups.size() * handleSize);
    if (std::expected<void, Error> queried =
            Backend::checkVulkan(owner.device->getRayTracingShaderGroupHandlesKHR(result.linked->pipeline, 0, static_cast<std::uint32_t>(groups.size()),
                                                                                  handles.size(), handles.data()),
                                 "vkGetRayTracingShaderGroupHandlesKHR");
        !queried)
        return queried;

    const std::size_t recordCount = raygenRecords.size() + missRecords.size() + hitRecords.size() + callableRecords.size();
    std::expected<Backend::Allocation, Error> table = owner.allocate(recordCount * stride, Backend::MemoryKind::Upload, baseAlignment);
    if (!table)
        return std::unexpected(std::move(table.error()));
    result.table = *table;
    std::memset(table->mapped, 0, table->size);
    const vk::DeviceAddress base = table->address;
    std::byte* const records = table->mapped;

    std::size_t record = 0;
    const auto writeRegion = [&](const std::span<const std::size_t> indices) {
        const std::size_t first = record;
        for (const std::size_t group : indices) {
            std::memcpy(records + record * stride, handles.data() + group * handleSize, handleSize);
            if (groups[group].kind == Backend::GroupKind::Raygen)
                result.raygenRegions.emplace_back(groups[group].raygen, vk::StridedDeviceAddressRegionKHR(base + record * stride, stride, stride));
            ++record;
        }
        if (record == first)
            return vk::StridedDeviceAddressRegionKHR{};
        return vk::StridedDeviceAddressRegionKHR(base + first * stride, stride, (record - first) * stride);
    };
    writeRegion(raygenRecords);
    result.missRegion = writeRegion(missRecords);
    result.hitRegion = writeRegion(hitRecords);
    result.callableRegion = writeRegion(callableRecords);
    return {};
}

std::expected<std::shared_ptr<Backend::VulkanAccelerationStructure>, Error>
createStructure(Backend::VulkanDevice& owner, const vk::AccelerationStructureTypeKHR type, const vk::BuildAccelerationStructureFlagsKHR flags,
                std::vector<vk::AccelerationStructureGeometryKHR> geometries, std::vector<std::uint32_t> primitiveCounts) {
    vk::AccelerationStructureBuildGeometryInfoKHR sizeInfo(type, flags, vk::BuildAccelerationStructureModeKHR::eBuild);
    sizeInfo.setGeometries(geometries);
    const vk::AccelerationStructureBuildSizesInfoKHR sizes =
        owner.device->getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, sizeInfo, primitiveCounts);
    const bool updatable = static_cast<bool>(flags & vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate);

    auto result = std::make_shared<Backend::VulkanAccelerationStructure>();
    result->device = owner.shared_from_this();
    result->topLevel = type == vk::AccelerationStructureTypeKHR::eTopLevel;
    result->updatable = updatable;
    result->flags = flags;
    result->geometries = std::move(geometries);
    result->primitiveCounts = std::move(primitiveCounts);
    std::expected<Backend::Allocation, Error> storage =
        //! Acceleration structure storage offsets must be multiples of 256 bytes.
        owner.allocate(sizes.accelerationStructureSize, Backend::MemoryKind::Device, 256);
    if (!storage)
        return std::unexpected(std::move(storage.error()));
    result->storage = *storage;
    std::expected<vk::AccelerationStructureKHR, Error> structure =
        Backend::checkVulkan(owner.device->createAccelerationStructureKHR({{}, storage->buffer, storage->offset, sizes.accelerationStructureSize, type}),
                             "vkCreateAccelerationStructureKHR");
    if (!structure)
        return std::unexpected(std::move(structure.error()));
    result->structure = *structure;
    std::expected<Backend::Allocation, Error> scratch =
        owner.allocate(updatable ? std::max(sizes.buildScratchSize, sizes.updateScratchSize) : sizes.buildScratchSize, Backend::MemoryKind::Device,
                       owner.accelerationProperties.minAccelerationStructureScratchOffsetAlignment);
    if (!scratch)
        return std::unexpected(std::move(scratch.error()));
    result->scratch = *scratch;
    result->address = owner.device->getAccelerationStructureAddressKHR({result->structure});
    return result;
}

vk::AccelerationStructureGeometryKHR getInstanceGeometry(const vk::DeviceAddress records) {
    return {vk::GeometryTypeKHR::eInstances, vk::AccelerationStructureGeometryInstancesDataKHR(vk::False, records)};
}

Backend::VulkanDevice& getVulkanDevice(Device& device) {
    return static_cast<Backend::VulkanDevice&>(*device.getBackend());
}

std::span<const std::byte> getRecordBytes(const std::span<const InstanceRecord> records) {
    static_assert(sizeof(InstanceRecord) == sizeof(vk::AccelerationStructureInstanceKHR));
    return std::as_bytes(records);
}

}

namespace Backend {

VulkanAccelerationStructure::~VulkanAccelerationStructure() {
    if (!device)
        return;
    device->retire([owner = device.get(), structure = this->structure, storage = this->storage, scratch = this->scratch] {
        owner->device->destroyAccelerationStructureKHR(structure);
        owner->release(storage);
        owner->release(scratch);
    });
}

std::expected<void, Error> Backend::VulkanAccelerationStructure::build(const vk::BuildAccelerationStructureModeKHR mode,
                                                                       const std::span<const std::byte> records) {
    std::expected<Token, Error> token = device->execute([&](VulkanCommandList& list) {
        if (!records.empty()) {
            std::expected<Placement, Error> placed = list.place(records);
            if (!placed)
                return static_cast<void>(list.check(false, placed.error().code, placed.error().message));
            geometries.front().geometry.instances.data.deviceAddress = placed->address;
        }
        vk::AccelerationStructureBuildGeometryInfoKHR info(
            topLevel ? vk::AccelerationStructureTypeKHR::eTopLevel : vk::AccelerationStructureTypeKHR::eBottomLevel, flags, mode);
        info.setSrcAccelerationStructure(mode == vk::BuildAccelerationStructureModeKHR::eUpdate ? structure : vk::AccelerationStructureKHR{})
            .setDstAccelerationStructure(structure)
            .setGeometries(geometries)
            .setScratchData(scratch.address);
        std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges;
        ranges.reserve(primitiveCounts.size());
        for (const std::uint32_t count : primitiveCounts)
            ranges.emplace_back(count, 0, 0, 0);
        const vk::AccelerationStructureBuildRangeInfoKHR* const rangePointer = ranges.data();
        list.synchronize(vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR);
        list.pool.commands.buildAccelerationStructuresKHR(info, rangePointer);
    });
    if (!token)
        return std::unexpected(std::move(token.error()));
    return {};
}

std::expected<void, Error> Backend::VulkanAccelerationStructure::refit() {
    if (topLevel || !updatable)
        return fail(ErrorCode::InvalidState, "refit needs a BLAS built with BuildMode::Dynamic");
    return build(vk::BuildAccelerationStructureModeKHR::eUpdate, {});
}

std::expected<void, Error> Backend::VulkanAccelerationStructure::update(const std::span<const InstanceRecord> records) {
    if (!topLevel)
        return fail(ErrorCode::InvalidState, "update needs a TLAS");
    if (records.size() != primitiveCounts.front())
        return fail(ErrorCode::InvalidArgument, "a TLAS update cannot change the instance count");
    return build(vk::BuildAccelerationStructureModeKHR::eUpdate, getRecordBytes(records));
}

std::expected<void, Error> Backend::VulkanAccelerationStructure::update(const Ptr<InstanceRecord> records, const std::uint32_t count) {
    if (!topLevel)
        return fail(ErrorCode::InvalidState, "update needs a TLAS");
    if (count != primitiveCounts.front())
        return fail(ErrorCode::InvalidArgument, "a TLAS update cannot change the instance count");
    geometries.front().geometry.instances.data.deviceAddress = records.getAddress();
    return build(vk::BuildAccelerationStructureModeKHR::eUpdate, {});
}

VulkanRayTracingLibrary::~VulkanRayTracingLibrary() {
    if (!device)
        return;
    device->retire([owner = device.get(), pipeline = this->pipeline] { owner->device->destroyPipeline(pipeline); });
}

LinkedPipeline::~LinkedPipeline() {
    if (!device)
        return;
    device->retire([owner = device.get(), pipeline = this->pipeline] { owner->device->destroyPipeline(pipeline); });
}

VulkanRayTracingPipeline::~VulkanRayTracingPipeline() {
    if (!device)
        return;
    device->retire([owner = device.get(), table = this->table] { owner->release(table); });
}

void VulkanRayTracingPipeline::traceRays(CommandList& list, const Shader& raygen, const DispatchSize size, const std::span<const std::byte> arguments) const {
    static_cast<VulkanCommandList&>(list).traceRays(*this, raygen, size, arguments);
}

RayTracingPipelineResult VulkanRayTracingPipeline::relink(const std::span<const std::uint32_t> hitGroups) const {
    auto result = std::make_shared<VulkanRayTracingPipeline>();
    result->device = device;
    result->linked = linked;
    if (std::expected<void, Error> built = buildShaderBindingTable(*result, hitGroups); !built)
        return std::unexpected(std::move(built.error()));
    return result;
}

AccelerationStructureResult buildHardwareBlas(NoorRHI::Device& device, const std::span<const TriangleGeometry> geometry, const BuildMode mode) {
    VulkanDevice& owner = getVulkanDevice(device);
    std::vector<vk::AccelerationStructureGeometryKHR> geometries;
    std::vector<std::uint32_t> primitiveCounts;
    geometries.reserve(geometry.size());
    primitiveCounts.reserve(geometry.size());
    for (const TriangleGeometry& item : geometry) {
        const std::optional<BufferRange> positions = owner.findBuffer(item.positions.getAddress());
        if (!positions || positions->size < item.stride)
            return fail(ErrorCode::InvalidArgument, "the BLAS positions must be inside a buffer of this device and hold at least one vertex");
        const std::uint64_t vertexCount = std::min<std::uint64_t>(positions->size / item.stride, std::numeric_limits<std::uint32_t>::max());
        const vk::AccelerationStructureGeometryTrianglesDataKHR triangles(vk::Format::eR32G32B32Sfloat, item.positions.getAddress(), item.stride,
                                                                          static_cast<std::uint32_t>(vertexCount - 1), vk::IndexType::eUint32,
                                                                          item.indices.getAddress());
        geometries.emplace_back(vk::GeometryTypeKHR::eTriangles, triangles,
                                item.opaque ? vk::GeometryFlagBitsKHR::eOpaque : vk::GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation);
        primitiveCounts.push_back(item.triangleCount);
    }

    const vk::BuildAccelerationStructureFlagsKHR flags =
        mode == BuildMode::Dynamic ? vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate | vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild
                                   : vk::BuildAccelerationStructureFlagsKHR(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
    std::expected<std::shared_ptr<VulkanAccelerationStructure>, Error> result =
        createStructure(owner, vk::AccelerationStructureTypeKHR::eBottomLevel, flags, std::move(geometries), std::move(primitiveCounts));
    if (!result)
        return std::unexpected(std::move(result.error()));
    if (std::expected<void, Error> built = (*result)->build(vk::BuildAccelerationStructureModeKHR::eBuild, {}); !built)
        return std::unexpected(std::move(built.error()));
    if (!(*result)->updatable) {
        owner.retire([device = &owner, scratch = (*result)->scratch] { device->release(scratch); });
        (*result)->scratch = {};
    }
    return std::move(*result);
}

AccelerationStructureResult buildHardwareTlas(NoorRHI::Device& device, const std::span<const InstanceRecord> records) {
    std::expected<std::shared_ptr<VulkanAccelerationStructure>, Error> result =
        createStructure(getVulkanDevice(device), vk::AccelerationStructureTypeKHR::eTopLevel, topLevelFlags, {getInstanceGeometry(0)},
                        {static_cast<std::uint32_t>(records.size())});
    if (!result)
        return std::unexpected(std::move(result.error()));
    if (std::expected<void, Error> built = (*result)->build(vk::BuildAccelerationStructureModeKHR::eBuild, getRecordBytes(records)); !built)
        return std::unexpected(std::move(built.error()));
    return std::move(*result);
}

AccelerationStructureResult buildHardwareTlas(NoorRHI::Device& device, const Ptr<InstanceRecord> records, const std::uint32_t count) {
    std::expected<std::shared_ptr<VulkanAccelerationStructure>, Error> result = createStructure(
        getVulkanDevice(device), vk::AccelerationStructureTypeKHR::eTopLevel, topLevelFlags, {getInstanceGeometry(records.getAddress())}, {count});
    if (!result)
        return std::unexpected(std::move(result.error()));
    if (std::expected<void, Error> built = (*result)->build(vk::BuildAccelerationStructureModeKHR::eBuild, {}); !built)
        return std::unexpected(std::move(built.error()));
    return std::move(*result);
}

RayTracingLibraryResult openHardwareLibrary(NoorRHI::Device& device, const RayTracingPipelineDesc& desc, const RayTracingInterface& interface) {
    VulkanDevice& owner = getVulkanDevice(device);
    ShaderGroups groups = collectGroups(owner, desc);
    if (groups.error)
        return std::unexpected(std::move(*groups.error));
    if (groups.groups.empty())
        return fail(ErrorCode::InvalidArgument, "a ray tracing library needs at least one shader group");

    const vk::RayTracingPipelineInterfaceCreateInfoKHR libraryInterface(interface.maxPayloadSize, interface.maxHitAttributeSize);
    vk::RayTracingPipelineCreateInfoKHR info;
    info.setFlags(vk::PipelineCreateFlagBits::eLibraryKHR)
        .setStages(groups.stages)
        .setGroups(groups.infos)
        .setMaxPipelineRayRecursionDepth(1)
        .setPLibraryInterface(&libraryInterface)
        .setLayout(*owner.pipelineLayout);
    std::expected<vk::Pipeline, Error> pipeline = createPipeline(owner, info, owner.getPipelineCache(desc.usePipelineCache));
    if (!pipeline)
        return std::unexpected(std::move(pipeline.error()));
    auto result = std::make_shared<VulkanRayTracingLibrary>();
    result->device = owner.shared_from_this();
    result->pipeline = *pipeline;
    result->interface = interface;
    result->groups = std::move(groups.groups);
    result->usePipelineCache = desc.usePipelineCache;
    if (desc.usePipelineCache && owner.pipelineCache)
        if (std::expected<void, Error> saved = owner.pipelineCache->save(); !saved)
            return std::unexpected(std::move(saved.error()));
    return result;
}

RayTracingPipelineResult openHardwarePipeline(NoorRHI::Device& device, const std::span<const std::shared_ptr<RayTracingLibrary>> libraries,
                                              const std::span<const std::uint32_t> hitGroups) {
    VulkanDevice& owner = getVulkanDevice(device);
    auto linked = std::make_shared<LinkedPipeline>();
    std::vector<vk::Pipeline> handles;
    bool usePipelineCache = true;
    for (const std::shared_ptr<RayTracingLibrary>& library : libraries) {
        auto vulkan = std::static_pointer_cast<VulkanRayTracingLibrary>(library);
        handles.push_back(vulkan->pipeline);
        linked->groups.insert(linked->groups.end(), vulkan->groups.begin(), vulkan->groups.end());
        usePipelineCache = usePipelineCache && vulkan->usePipelineCache;
        linked->libraries.push_back(std::move(vulkan));
    }
    if (std::ranges::none_of(linked->groups, [](const ShaderGroup& group) { return group.kind == GroupKind::Raygen; }))
        return fail(ErrorCode::InvalidArgument, "a ray tracing pipeline needs a ray generation shader");

    const RayTracingInterface& interface = linked->libraries.front()->interface;
    const vk::RayTracingPipelineInterfaceCreateInfoKHR libraryInterface(interface.maxPayloadSize, interface.maxHitAttributeSize);
    const vk::PipelineLibraryCreateInfoKHR libraryInfo(handles);
    vk::RayTracingPipelineCreateInfoKHR info;
    info.setMaxPipelineRayRecursionDepth(1).setPLibraryInfo(&libraryInfo).setPLibraryInterface(&libraryInterface).setLayout(*owner.pipelineLayout);
    std::expected<vk::Pipeline, Error> pipeline = createPipeline(owner, info, owner.getPipelineCache(usePipelineCache));
    if (!pipeline)
        return std::unexpected(std::move(pipeline.error()));
    linked->device = owner.shared_from_this();
    linked->pipeline = *pipeline;

    auto result = std::make_shared<VulkanRayTracingPipeline>();
    result->device = linked->device;
    result->linked = std::move(linked);
    if (std::expected<void, Error> built = buildShaderBindingTable(*result, hitGroups); !built)
        return std::unexpected(std::move(built.error()));
    return result;
}

}

}
