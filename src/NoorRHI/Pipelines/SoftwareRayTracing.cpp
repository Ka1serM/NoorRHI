#include <array>
#include <bit>
#include <cstring>
#include <deque>
#include <string>
#include <tuple>
#include <utility>

#include "NoorRHI/Core/Backend.h"
#include "NoorRHI/Shaders/RayTracingBuild.h"
#include "RayTracingBuild.slang.h"

namespace NoorRHI {

namespace {

struct alignas(16) Word {
    std::array<std::uint32_t, 4> values;
};

//! NoorRHIRayTracing.slang reads this launch record in front of the arguments of a software ray generation kernel.
struct RayLaunch {
    NoorRHI::uint3 size;
    std::uint32_t missCount = 0;
    std::uint32_t hitGroupCount = 0;
    std::array<std::uint32_t, 3> padding{};
    Ptr<std::uint32_t> misses;
    Ptr<NoorRHI::uint2> hitGroups;
};
static_assert(sizeof(RayLaunch) == 48);

constexpr std::uint32_t headerWords = 4;
constexpr std::uint32_t nodeWords = 4;
constexpr std::uint32_t triangleWords = 3;
constexpr std::uint32_t instanceWords = 7;
constexpr std::uint32_t recordWords = 4;
constexpr std::uint32_t buildGroupSize = 128;
constexpr std::uint32_t radixPasses = 8;
constexpr std::uint32_t radixBits = 4;
constexpr std::uint32_t maxPrimitives = 1u << 22;
constexpr std::uint32_t noShader = 0xffffffffu;
constexpr DispatchSize raygenGroupSize{8, 8, 1};

std::uint32_t getWords(const std::uint32_t values) {
    return (values + 3) / 4;
}

struct BuildLayout {
    BuildLayout() = default;
    BuildLayout(const std::uint32_t count, const bool topLevel)
        : count(count), topLevel(topLevel), leafBase(std::bit_ceil(count)), blockCount((count + buildGroupSize - 1) / buildGroupSize),
          primitiveOffset(headerWords + (std::max(count, 2u) - 1) * nodeWords) {
        leafBoxes = count * (topLevel ? recordWords : triangleWords);
        tree = leafBoxes + count * 2;
        keys = tree + leafBase * 4;
        values = keys + getWords(count);
        sortedKeys = values + getWords(count);
        sortedValues = sortedKeys + getWords(count);
        histogram = sortedValues + getWords(count);
        scratchWords = histogram + getWords(blockCount * 16);
    }

    std::uint32_t getStorageWords() const { return primitiveOffset + count * (topLevel ? instanceWords : triangleWords); }

    std::uint32_t count = 0;
    bool topLevel = false;
    std::uint32_t leafBase = 0;
    std::uint32_t blockCount = 0;
    std::uint32_t primitiveOffset = 0;
    std::uint32_t leafBoxes = 0;
    std::uint32_t tree = 0;
    std::uint32_t keys = 0;
    std::uint32_t values = 0;
    std::uint32_t sortedKeys = 0;
    std::uint32_t sortedValues = 0;
    std::uint32_t histogram = 0;
    std::uint32_t scratchWords = 0;
};

}

struct Backend::SoftwareBuildKernels {
    NoorRHI::ComputePipeline resetBounds;
    NoorRHI::ComputePipeline gatherTriangles;
    NoorRHI::ComputePipeline prepareInstances;
    NoorRHI::ComputePipeline computeMortonCodes;
    NoorRHI::ComputePipeline countDigits;
    NoorRHI::ComputePipeline scanDigitCounts;
    NoorRHI::ComputePipeline scatterDigits;
    NoorRHI::ComputePipeline placeLeaves;
    NoorRHI::ComputePipeline reduceTreeLevel;
    NoorRHI::ComputePipeline writeNodes;
};

namespace {

std::expected<std::shared_ptr<const Backend::SoftwareBuildKernels>, Error> getBuildKernels(Device& device) {
    Backend::Device& owner = *device.getBackend();
    std::lock_guard lock(owner.softwareBuildMutex);
    if (std::shared_ptr<const Backend::SoftwareBuildKernels> existing = owner.softwareBuildKernels.lock())
        return existing;
    auto kernels = std::make_shared<Backend::SoftwareBuildKernels>();
    const std::array<std::pair<ComputePipeline Backend::SoftwareBuildKernels::*, const Shader*>, 10> shaders{{
        {&Backend::SoftwareBuildKernels::resetBounds, &RayTracingBuild::resetBounds},
        {&Backend::SoftwareBuildKernels::gatherTriangles, &RayTracingBuild::gatherTriangles},
        {&Backend::SoftwareBuildKernels::prepareInstances, &RayTracingBuild::prepareInstances},
        {&Backend::SoftwareBuildKernels::computeMortonCodes, &RayTracingBuild::computeMortonCodes},
        {&Backend::SoftwareBuildKernels::countDigits, &RayTracingBuild::countDigits},
        {&Backend::SoftwareBuildKernels::scanDigitCounts, &RayTracingBuild::scanDigitCounts},
        {&Backend::SoftwareBuildKernels::scatterDigits, &RayTracingBuild::scatterDigits},
        {&Backend::SoftwareBuildKernels::placeLeaves, &RayTracingBuild::placeLeaves},
        {&Backend::SoftwareBuildKernels::reduceTreeLevel, &RayTracingBuild::reduceTreeLevel},
        {&Backend::SoftwareBuildKernels::writeNodes, &RayTracingBuild::writeNodes},
    }};
    for (const auto& [member, shader] : shaders)
        if (std::expected<void, Error> opened = ((*kernels).*member).open(device, *shader); !opened)
            return std::unexpected(std::move(opened.error()));
    owner.softwareBuildKernels = kernels;
    return kernels;
}

class BuildRecorder {
public:
    explicit BuildRecorder(Device& device) : list(device) {}

    template <class Arguments>
    void dispatch(const ComputePipeline& pipeline, const std::uint32_t threads, const Arguments& arguments) {
        list.dispatch(pipeline, {threads}, arguments);
    }

    CommandList& getList() { return list; }

    std::expected<void, Error> submit(Device& device) {
        std::expected<Token, Error> token = device.submit(std::move(list));
        if (!token)
            return std::unexpected(std::move(token.error()));
        return {};
    }

private:
    CommandList list;
};

struct SoftwareAccelerationStructure final : Backend::AccelerationStructure {
    std::expected<void, Error> refit() override {
        if (layout.topLevel || mode != BuildMode::Dynamic)
            return fail(ErrorCode::InvalidState, "refit needs a BLAS built with BuildMode::Dynamic");
        BuildRecorder recorder(device);
        recordGather(recorder);
        recordHierarchy(recorder, getArguments());
        return recorder.submit(device);
    }

    std::expected<void, Error> update(const std::span<const InstanceRecord> records) override {
        if (!layout.topLevel)
            return fail(ErrorCode::InvalidState, "update needs a TLAS");
        if (records.size() != layout.count)
            return fail(ErrorCode::InvalidArgument, "a TLAS update cannot change the instance count");
        return buildTop(records);
    }

    std::expected<void, Error> update(const Ptr<InstanceRecord> records, const std::uint32_t count) override {
        if (!layout.topLevel)
            return fail(ErrorCode::InvalidState, "update needs a TLAS");
        if (count != layout.count)
            return fail(ErrorCode::InvalidArgument, "a TLAS update cannot change the instance count");
        return buildTop(records);
    }

    std::expected<void, Error> open(Device& owner, const BuildLayout& buildLayout) {
        device = owner;
        layout = buildLayout;
        if (layout.count == 0 || layout.count > maxPrimitives)
            return fail(ErrorCode::InvalidArgument, "a software acceleration structure holds between 1 and 2^22 primitives");
        std::expected<std::shared_ptr<const Backend::SoftwareBuildKernels>, Error> built = getBuildKernels(device);
        if (!built)
            return std::unexpected(std::move(built.error()));
        kernels = std::move(*built);
        if (std::expected<void, Error> opened = storage.open(device, layout.getStorageWords()); !opened)
            return opened;
        if (std::expected<void, Error> opened = scratch.open(device, layout.scratchWords); !opened)
            return opened;
        if (std::expected<void, Error> opened = bounds.open(device, 6); !opened)
            return opened;
        address = storage.getPtr().getAddress();
        return {};
    }

    void releaseScratch() {
        scratch = {};
        bounds = {};
        kernels.reset();
    }

    RayTracingBuild::BuildArguments getArguments() const {
        const Ptr<Word> base = scratch.getPtr();
        return {storage.getPtr().as<NoorRHI::uint4>(),
                base.as<NoorRHI::uint4>(),
                (base + layout.leafBoxes).as<NoorRHI::float4>(),
                (base + layout.tree).as<NoorRHI::float4>(),
                (base + layout.keys).as<std::uint32_t>(),
                (base + layout.values).as<std::uint32_t>(),
                (base + layout.sortedKeys).as<std::uint32_t>(),
                (base + layout.sortedValues).as<std::uint32_t>(),
                (base + layout.histogram).as<std::uint32_t>(),
                bounds.getPtr(),
                layout.count,
                layout.leafBase,
                layout.primitiveOffset,
                layout.topLevel ? 1u : 0u,
                0,
                layout.blockCount,
                0,
                0};
    }

    void recordGather(BuildRecorder& recorder) const {
        const RayTracingBuild::BuildArguments arguments = getArguments();
        std::uint32_t first = 0;
        for (std::uint32_t index = 0; index < geometry.size(); ++index) {
            const TriangleGeometry& item = geometry[index];
            recorder.dispatch(kernels->gatherTriangles, item.triangleCount,
                              RayTracingBuild::GatherArguments{item.positions.as<float>(), item.indices, arguments.source, arguments.leafBoxes,
                                                               arguments.bounds, item.stride / static_cast<std::uint32_t>(sizeof(float)), item.triangleCount,
                                                               first, index, item.opaque ? 1u : 0u});
            first += item.triangleCount;
        }
    }

    void recordSort(BuildRecorder& recorder, RayTracingBuild::BuildArguments arguments) const {
        recorder.dispatch(kernels->computeMortonCodes, layout.count, arguments);
        for (std::uint32_t pass = 0; pass < radixPasses; ++pass) {
            arguments.shift = pass * radixBits;
            recorder.dispatch(kernels->countDigits, layout.blockCount * buildGroupSize, arguments);
            recorder.dispatch(kernels->scanDigitCounts, buildGroupSize, arguments);
            recorder.dispatch(kernels->scatterDigits, layout.blockCount * buildGroupSize, arguments);
            std::swap(arguments.keys, arguments.sortedKeys);
            std::swap(arguments.values, arguments.sortedValues);
        }
    }

    void recordHierarchy(BuildRecorder& recorder, RayTracingBuild::BuildArguments arguments) const {
        recorder.dispatch(kernels->placeLeaves, layout.leafBase, arguments);
        for (std::uint32_t level = layout.leafBase / 2; level > 0; level /= 2) {
            arguments.level = level;
            recorder.dispatch(kernels->reduceTreeLevel, level, arguments);
        }
        recorder.dispatch(kernels->writeNodes, std::max(layout.count, 2u) - 1, arguments);
    }

    std::expected<void, Error> buildBottom() {
        BuildRecorder recorder(device);
        const RayTracingBuild::BuildArguments arguments = getArguments();
        recorder.dispatch(kernels->resetBounds, 1, arguments);
        recordGather(recorder);
        recordSort(recorder, arguments);
        recordHierarchy(recorder, arguments);
        return recorder.submit(device);
    }

    std::expected<void, Error> buildTop(const std::span<const InstanceRecord> records) {
        std::vector<Word> words(records.size() * recordWords);
        std::memcpy(words.data(), records.data(), records.size_bytes());
        BuildRecorder recorder(device);
        recorder.getList().upload(scratch, std::span<const Word>(words));
        recordTop(recorder, getArguments());
        return recorder.submit(device);
    }

    std::expected<void, Error> buildTop(const Ptr<InstanceRecord> records) {
        BuildRecorder recorder(device);
        RayTracingBuild::BuildArguments arguments = getArguments();
        arguments.source = records.as<NoorRHI::uint4>();
        recordTop(recorder, arguments);
        return recorder.submit(device);
    }

    void recordTop(BuildRecorder& recorder, const RayTracingBuild::BuildArguments& arguments) const {
        recorder.dispatch(kernels->resetBounds, 1, arguments);
        recorder.dispatch(kernels->prepareInstances, layout.count, arguments);
        recordSort(recorder, arguments);
        recordHierarchy(recorder, arguments);
    }

    Device device;
    std::shared_ptr<const Backend::SoftwareBuildKernels> kernels;
    Buffer<Word> storage;
    Buffer<Word> scratch;
    Buffer<Atomic<std::uint32_t>> bounds;
    BuildLayout layout;
    std::vector<TriangleGeometry> geometry;
    BuildMode mode = BuildMode::Static;
};

std::expected<std::shared_ptr<SoftwareAccelerationStructure>, Error> openTopLevel(Device& device, const std::uint32_t count) {
    auto result = std::make_shared<SoftwareAccelerationStructure>();
    if (std::expected<void, Error> opened = result->open(device, BuildLayout(count, true)); !opened)
        return std::unexpected(std::move(opened.error()));
    return result;
}

struct SoftwareRayTracingLibrary final : Backend::RayTracingLibrary {
    RayTracingPipelineDesc desc;
};

struct SoftwareRayTracingProgram {
    Device device;
    Shader source;
    std::deque<std::string> kernelNames;
    std::vector<std::string_view> raygenNames;
    std::vector<ComputePipeline> raygenKernels;
    std::vector<std::uint32_t> misses;
    std::vector<std::array<std::uint32_t, 2>> hitGroups;
};

struct SoftwareRayTracingPipeline final : Backend::RayTracingPipeline {
    void traceRays(Backend::CommandList& list, const Shader& raygen, const DispatchSize size, const std::span<const std::byte> arguments) const override {
        const auto found = std::ranges::find(program->raygenNames, raygen.entryPoint);
        if (!list.check(found != program->raygenNames.end() && raygen.getSource() == program->source.getSource(), ErrorCode::InvalidArgument,
                        "the shader is not a ray generation shader of the pipeline"))
            return;
        const RayLaunch launch{{size.x, size.y, size.z}, static_cast<std::uint32_t>(program->misses.size()),    hitGroupCount, {},
                               table.getPtr(),           (table.getPtr() + hitGroupOffset).as<NoorRHI::uint2>()};
        std::vector<std::byte> bytes(sizeof(RayLaunch) + arguments.size());
        std::memcpy(bytes.data(), &launch, sizeof(RayLaunch));
        if (!arguments.empty())
            std::memcpy(bytes.data() + sizeof(RayLaunch), arguments.data(), arguments.size());
        const auto groups = [](const std::uint32_t count, const std::uint32_t groupSize) { return count / groupSize + (count % groupSize != 0); };
        const ComputePipeline& kernel = program->raygenKernels[static_cast<std::size_t>(found - program->raygenNames.begin())];
        list.dispatch(*kernel.getBackend(), {groups(size.x, raygenGroupSize.x), groups(size.y, raygenGroupSize.y), groups(size.z, raygenGroupSize.z)}, bytes);
    }

    Backend::RayTracingPipelineResult relink(const std::span<const std::uint32_t> selected) const override;

    std::shared_ptr<const SoftwareRayTracingProgram> program;
    Buffer<std::uint32_t> table;
    std::uint32_t hitGroupOffset = 0;
    std::uint32_t hitGroupCount = 0;
};

Backend::RayTracingPipelineResult openPipeline(std::shared_ptr<const SoftwareRayTracingProgram> program, const std::span<const std::uint32_t> selected) {
    auto result = std::make_shared<SoftwareRayTracingPipeline>();
    result->hitGroupOffset = static_cast<std::uint32_t>(program->misses.size() + program->misses.size() % 2);
    result->hitGroupCount = static_cast<std::uint32_t>(selected.size());
    std::vector<std::uint32_t> entries(std::max<std::size_t>(result->hitGroupOffset + selected.size() * 2, 1));
    std::ranges::copy(program->misses, entries.begin());
    for (std::size_t index = 0; index < selected.size(); ++index) {
        if (selected[index] >= program->hitGroups.size())
            return fail(ErrorCode::InvalidArgument, "the shader binding table names hit group " + std::to_string(selected[index]) + " of a pipeline with "
                                                        + std::to_string(program->hitGroups.size()));
        entries[result->hitGroupOffset + index * 2] = program->hitGroups[selected[index]][0];
        entries[result->hitGroupOffset + index * 2 + 1] = program->hitGroups[selected[index]][1];
    }
    Device device = program->device;
    if (std::expected<void, Error> opened = result->table.open(device, entries.size()); !opened)
        return std::unexpected(std::move(opened.error()));
    if (std::expected<void, Error> uploaded = result->table.upload(entries); !uploaded)
        return std::unexpected(std::move(uploaded.error()));
    result->program = std::move(program);
    return result;
}

Backend::RayTracingPipelineResult SoftwareRayTracingPipeline::relink(const std::span<const std::uint32_t> selected) const {
    return openPipeline(program, selected);
}

std::uint32_t getShaderIndex(const std::span<const Shader> shaders, const std::size_t index) {
    return index < shaders.size() && shaders[index] ? shaders[index].fallbackIndex : noShader;
}

}

namespace Backend {

AccelerationStructureResult buildSoftwareBlas(NoorRHI::Device& device, const std::span<const TriangleGeometry> geometry, const BuildMode mode) {
    std::uint64_t count = 0;
    for (const TriangleGeometry& item : geometry)
        count += item.triangleCount;
    if (count > maxPrimitives)
        return fail(ErrorCode::InvalidArgument, "a software BLAS holds at most 2^22 triangles");
    auto result = std::make_shared<SoftwareAccelerationStructure>();
    result->mode = mode;
    result->geometry.assign(geometry.begin(), geometry.end());
    if (std::expected<void, Error> opened = result->open(device, BuildLayout(static_cast<std::uint32_t>(count), false)); !opened)
        return std::unexpected(std::move(opened.error()));
    if (std::expected<void, Error> built = result->buildBottom(); !built)
        return std::unexpected(std::move(built.error()));
    if (mode == BuildMode::Static)
        result->releaseScratch();
    return result;
}

AccelerationStructureResult buildSoftwareTlas(NoorRHI::Device& device, const std::span<const InstanceRecord> records) {
    std::expected<std::shared_ptr<SoftwareAccelerationStructure>, Error> result = openTopLevel(device, static_cast<std::uint32_t>(records.size()));
    if (!result)
        return std::unexpected(std::move(result.error()));
    if (std::expected<void, Error> built = (*result)->buildTop(records); !built)
        return std::unexpected(std::move(built.error()));
    return std::move(*result);
}

AccelerationStructureResult buildSoftwareTlas(NoorRHI::Device& device, const Ptr<InstanceRecord> records, const std::uint32_t count) {
    std::expected<std::shared_ptr<SoftwareAccelerationStructure>, Error> result = openTopLevel(device, count);
    if (!result)
        return std::unexpected(std::move(result.error()));
    if (std::expected<void, Error> built = (*result)->buildTop(records); !built)
        return std::unexpected(std::move(built.error()));
    return std::move(*result);
}

RayTracingLibraryResult openSoftwareLibrary(NoorRHI::Device&, const RayTracingPipelineDesc& desc, const RayTracingInterface&) {
    if (!desc.intersection.empty() || !desc.callable.empty())
        return fail(ErrorCode::Unsupported, "software ray tracing has no intersection or callable shaders");
    if (desc.raygen.empty() && desc.miss.empty() && desc.closestHit.empty() && desc.anyHit.empty())
        return fail(ErrorCode::InvalidArgument, "a ray tracing library needs at least one shader group");
    const auto check = [](const std::span<const Shader> shaders, const ShaderStage stage, const bool optional) -> std::expected<void, Error> {
        for (const Shader& shader : shaders) {
            if (optional && !shader)
                continue;
            if (std::expected<void, Error> checked = checkShader(shader, stage); !checked)
                return checked;
        }
        return {};
    };
    for (const auto& [shaders, stage, optional] : {std::tuple(std::span<const Shader>(desc.raygen), ShaderStage::RayGeneration, false),
                                                   std::tuple(std::span<const Shader>(desc.miss), ShaderStage::Miss, false),
                                                   std::tuple(std::span<const Shader>(desc.closestHit), ShaderStage::ClosestHit, true),
                                                   std::tuple(std::span<const Shader>(desc.anyHit), ShaderStage::AnyHit, true)})
        if (std::expected<void, Error> checked = check(shaders, stage, optional); !checked)
            return std::unexpected(std::move(checked.error()));
    auto result = std::make_shared<SoftwareRayTracingLibrary>();
    result->desc = desc;
    return result;
}

RayTracingPipelineResult openSoftwarePipeline(NoorRHI::Device& device, const std::span<const std::shared_ptr<RayTracingLibrary>> libraries,
                                              const std::span<const std::uint32_t> hitGroups) {
    auto program = std::make_shared<SoftwareRayTracingProgram>();
    program->device = device;
    for (const std::shared_ptr<RayTracingLibrary>& library : libraries) {
        const RayTracingPipelineDesc& desc = static_cast<const SoftwareRayTracingLibrary&>(*library).desc;
        for (const std::vector<Shader>* shaders : {&desc.raygen, &desc.miss, &desc.closestHit, &desc.anyHit})
            for (const Shader& shader : *shaders) {
                if (!shader)
                    continue;
                if (!program->source)
                    program->source = shader;
                if (shader.getSource() != program->source.getSource())
                    return fail(ErrorCode::InvalidArgument, "the shaders of a software ray tracing pipeline must come from one Slang source");
            }
        for (const Shader& shader : desc.raygen)
            program->raygenNames.push_back(shader.entryPoint);
        for (const Shader& shader : desc.miss)
            program->misses.push_back(shader.fallbackIndex);
        const std::size_t hitCount = std::max(desc.closestHit.size(), desc.anyHit.size());
        for (std::size_t index = 0; index < hitCount; ++index)
            program->hitGroups.push_back({getShaderIndex(desc.closestHit, index), getShaderIndex(desc.anyHit, index)});
    }
    if (program->raygenNames.empty())
        return fail(ErrorCode::InvalidArgument, "a ray tracing pipeline needs a ray generation shader");
    for (const std::string_view name : program->raygenNames) {
        const std::string& kernelName = program->kernelNames.emplace_back("noorrhi_raygen_" + std::string(name));
        const std::array<std::span<const std::byte>, 4>& code = program->source.code;
        const Shader kernelShader{
            .code = {code[0], code[0], code[2], code[2]}, .entryPoint = kernelName, .stage = ShaderStage::Compute, .threadGroupSize = raygenGroupSize};
        if (std::expected<void, Error> opened = program->raygenKernels.emplace_back().open(device, kernelShader); !opened)
            return std::unexpected(std::move(opened.error()));
    }
    return openPipeline(std::move(program), hitGroups);
}

}

}
