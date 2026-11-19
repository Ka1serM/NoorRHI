#include "NoorRHI/Shaders/SpirvModule.h"

#include <algorithm>
#include <array>
#include <format>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <utility>

#include <spirv/unified1/GLSL.std.450.h>

namespace NoorRHI {

namespace {

constexpr std::uint32_t versionWithStorageBuffers = 0x10300;

bool isAnnotation(const spv::Op op) {
    return op == spv::Op::OpDecorate || op == spv::Op::OpMemberDecorate || op == spv::Op::OpDecorateId || op == spv::Op::OpDecorateString ||
           op == spv::Op::OpMemberDecorateString;
}

bool isTerminator(const spv::Op op) {
    return op == spv::Op::OpBranch || op == spv::Op::OpBranchConditional || op == spv::Op::OpSwitch || op == spv::Op::OpReturn ||
           op == spv::Op::OpReturnValue || op == spv::Op::OpKill || op == spv::Op::OpUnreachable || op == spv::Op::OpTerminateInvocation;
}

std::vector<std::uint32_t> literalString(const std::string_view text) {
    std::vector<std::uint32_t> words(text.size() / 4 + 1);
    std::copy(text.begin(), text.end(), reinterpret_cast<char*>(words.data()));
    return words;
}

std::size_t literalStringWords(const std::span<const std::uint32_t> words) {
    for (std::size_t i = 0; i < words.size(); ++i)
        if ((words[i] & 0xff000000u) == 0 || (words[i] & 0xff0000u) == 0 || (words[i] & 0xff00u) == 0 || (words[i] & 0xffu) == 0)
            return i + 1;
    return words.size();
}

bool isTypeOrConstant(const spv::Op op) {
    switch (op) {
    case spv::Op::OpTypeVoid:
    case spv::Op::OpTypeBool:
    case spv::Op::OpTypeInt:
    case spv::Op::OpTypeFloat:
    case spv::Op::OpTypeVector:
    case spv::Op::OpTypeMatrix:
    case spv::Op::OpTypeImage:
    case spv::Op::OpTypeSampler:
    case spv::Op::OpTypeSampledImage:
    case spv::Op::OpTypeArray:
    case spv::Op::OpTypeRuntimeArray:
    case spv::Op::OpTypeStruct:
    case spv::Op::OpTypePointer:
    case spv::Op::OpTypeFunction:
    case spv::Op::OpTypeAccelerationStructureKHR:
    case spv::Op::OpTypeRayQueryKHR:
    case spv::Op::OpTypeCooperativeMatrixKHR:
    case spv::Op::OpConstantTrue:
    case spv::Op::OpConstantFalse:
    case spv::Op::OpConstant:
    case spv::Op::OpConstantComposite:
    case spv::Op::OpConstantNull:
    case spv::Op::OpSpecConstantTrue:
    case spv::Op::OpSpecConstantFalse:
    case spv::Op::OpSpecConstant:
    case spv::Op::OpSpecConstantComposite:
    case spv::Op::OpSpecConstantOp:
    case spv::Op::OpUndef:
        return true;
    default:
        return false;
    }
}

std::string_view withoutSuffix(const std::string_view name) {
    const std::size_t underscore = name.rfind('_');
    if (underscore == std::string_view::npos || underscore + 1 == name.size())
        return name;
    const std::string_view suffix = name.substr(underscore + 1);
    return std::ranges::all_of(suffix, [](const char character) { return character >= '0' && character <= '9'; }) ? name.substr(0, underscore) : name;
}

}

std::expected<void, Error> SpirvModule::read(const std::span<const std::uint32_t> words) {
    if (words.size() < 5 || words[0] != spv::MagicNumber)
        return fail(ErrorCode::InvalidShader, "The shader is not a SPIR-V module.");
    version = words[1];
    generator = words[2];
    bound = words[3];
    instructions.clear();
    for (std::size_t offset = 5; offset < words.size();) {
        const std::uint32_t count = words[offset] >> 16;
        if (count == 0 || offset + count > words.size())
            return fail(ErrorCode::InvalidShader, std::format("The SPIR-V instruction at word {} has a bad length.", offset));
        instructions.push_back({static_cast<spv::Op>(words[offset] & 0xffffu), {words.begin() + offset + 1, words.begin() + offset + count}});
        offset += count;
    }
    return {};
}

std::vector<std::uint32_t> SpirvModule::write() const {
    std::vector<std::uint32_t> words{spv::MagicNumber, version, generator, bound, 0};
    for (const Instruction& instruction : instructions) {
        words.push_back(static_cast<std::uint32_t>(instruction.operands.size() + 1) << 16 | static_cast<std::uint32_t>(instruction.op));
        words.insert(words.end(), instruction.operands.begin(), instruction.operands.end());
    }
    return words;
}

std::expected<void, Error> SpirvModule::lowerForWebGpu() {
    lowerDrawParameters();
    lowerHalfConversions();
    return lowerSwitchBreaks()
        .and_then([&] { return lowerPackedComponents(); })
        .and_then([&] { return lowerTexelBuffers(); })
        .and_then([&] { return mergeAliasedBuffers(); })
        .and_then([&] { return checkBindings(); })
        .and_then([&] { return checkFiniteConstants(); });
}

void SpirvModule::lowerDrawParameters() {
    for (const std::uint32_t variable : variables()) {
        const std::uint32_t builtIn = decoration(variable, spv::Decoration::BuiltIn);
        if (builtIn != static_cast<std::uint32_t>(spv::BuiltIn::BaseVertex) && builtIn != static_cast<std::uint32_t>(spv::BuiltIn::BaseInstance))
            continue;
        const std::uint32_t zero = constant(pointee(variable), 0);
        removeVariable(variable);
        rewriteFunctions([&](const Instruction& instruction, std::vector<Instruction>& output) {
            if (instruction.op != spv::Op::OpLoad || instruction.operands[2] != variable)
                return false;
            output.push_back({spv::Op::OpCopyObject, {instruction.operands[0], instruction.operands[1], zero}});
            return true;
        });
    }
    removeCapability(spv::Capability::DrawParameters);
}

bool SpirvModule::hasCapability(const spv::Capability capability) const {
    return std::ranges::any_of(instructions, [&](const Instruction& instruction) {
        return instruction.op == spv::Op::OpCapability && instruction.operands[0] == static_cast<std::uint32_t>(capability);
    });
}

bool SpirvModule::hasStorageImages() {
    return std::ranges::any_of(variables(), [&](const std::uint32_t variable) {
        const Instruction& image = *definition(pointee(variable));
        return image.op == spv::Op::OpTypeImage && image.operands[6] == 2;
    });
}

std::expected<void, Error> SpirvModule::moveBinding(const std::uint32_t set, const std::uint32_t binding, const std::uint32_t target) {
    const std::uint32_t moved = variableAt(set, binding);
    if (moved == 0)
        return {};
    const std::uint32_t kept = variableAt(set, target);
    if (kept == 0) {
        for (Instruction& instruction : instructions)
            if (instruction.op == spv::Op::OpDecorate && instruction.operands[0] == moved && instruction.operands[1] == static_cast<std::uint32_t>(spv::Decoration::Binding))
                instruction.operands[2] = target;
        return {};
    }
    if (definition(kept)->operands[0] != definition(moved)->operands[0])
        return fail(ErrorCode::InvalidShader, std::format("Set {} bindings {} and {} have different types.", set, binding, target));
    rewriteFunctions([&](const Instruction& instruction, std::vector<Instruction>& output) {
        const bool access = instruction.op == spv::Op::OpLoad || instruction.op == spv::Op::OpAccessChain || instruction.op == spv::Op::OpInBoundsAccessChain ||
                            instruction.op == spv::Op::OpImageTexelPointer;
        const std::size_t first = access ? 2 : instruction.op == spv::Op::OpFunctionCall ? 3 : instruction.operands.size();
        const std::size_t last = access ? 3 : instruction.operands.size();
        if (!std::ranges::contains(std::span(instruction.operands).subspan(first, last - first), moved))
            return false;
        output.push_back(instruction);
        std::ranges::replace(std::span(output.back().operands).subspan(first, last - first), moved, kept);
        return true;
    });
    removeVariable(moved);
    return {};
}

void SpirvModule::lowerHalfConversions() {
    std::map<std::uint32_t, std::uint32_t> narrowed;
    std::set<std::uint32_t> halves;
    std::set<std::uint32_t> shorts;
    for (const Instruction& instruction : instructions) {
        if (instruction.op == spv::Op::OpTypeFloat && instruction.operands[1] == 16)
            halves.insert(instruction.operands[0]);
        if (instruction.op == spv::Op::OpTypeInt && instruction.operands[1] == 16)
            shorts.insert(instruction.operands[0]);
        if ((instruction.op == spv::Op::OpUConvert || instruction.op == spv::Op::OpSConvert) && shorts.contains(instruction.operands[0]))
            narrowed[instruction.operands[1]] = instruction.operands[2];
    }
    if (narrowed.empty())
        return;
    const std::uint32_t single = type(spv::Op::OpTypeFloat, {32});
    const std::uint32_t pair = type(spv::Op::OpTypeVector, {single, 2});
    const std::uint32_t glsl = extInstImport("GLSL.std.450");
    rewriteFunctions([&](const Instruction& instruction, std::vector<Instruction>& output) {
        if (instruction.op != spv::Op::OpBitcast || !halves.contains(instruction.operands[0]))
            return false;
        const auto source = narrowed.find(instruction.operands[2]);
        if (source == narrowed.end())
            return false;
        const std::uint32_t unpacked = newId();
        const std::uint32_t low = newId();
        output.push_back({spv::Op::OpExtInst, {pair, unpacked, glsl, GLSLstd450UnpackHalf2x16, source->second}});
        output.push_back({spv::Op::OpCompositeExtract, {single, low, unpacked, 0}});
        output.push_back({spv::Op::OpFConvert, {instruction.operands[0], instruction.operands[1], low}});
        return true;
    });
    for (const auto& [id, source] : narrowed)
        if (!referenced(id))
            removeDefinition(id);
    for (const std::uint32_t id : shorts)
        if (!referenced(id))
            removeDefinition(id);
    if (std::ranges::none_of(instructions, [](const Instruction& instruction) { return instruction.op == spv::Op::OpTypeInt && instruction.operands[1] == 16; }))
        removeCapability(spv::Capability::Int16);
}

std::expected<void, Error> SpirvModule::lowerSwitchBreaks() {
    bool lowered = false;
    for (std::optional<SwitchBreak> found = findSwitchBreak(); found; found = findSwitchBreak(), lowered = true)
        if (auto result = lowerSwitchBreak(*found); !result)
            return result;
    if (lowered)
        sortBlocks();
    return {};
}

std::optional<SpirvModule::SwitchBreak> SpirvModule::findSwitchBreak() {
    const std::vector<Block> all = blocks();
    std::map<std::uint32_t, std::vector<std::uint32_t>> next;
    for (const Block& block : all)
        next[block.label] = successors(instructions[block.end]);
    const auto reach = [&](std::vector<std::uint32_t> pending, const std::vector<std::uint32_t>& stops) {
        std::set<std::uint32_t> seen;
        while (!pending.empty()) {
            const std::uint32_t label = pending.back();
            pending.pop_back();
            if (std::ranges::contains(stops, label) || !seen.insert(label).second)
                continue;
            pending.insert(pending.end(), next[label].begin(), next[label].end());
        }
        return seen;
    };
    struct Loop {
        std::uint32_t merge = 0;
        std::uint32_t next = 0;
        std::set<std::uint32_t> body;
    };
    std::vector<Loop> loops;
    for (const Block& block : all) {
        const Instruction& merge = instructions[block.end - 1];
        if (merge.op == spv::Op::OpLoopMerge)
            loops.push_back({merge.operands[0], merge.operands[1], reach({block.label}, {merge.operands[0]})});
    }
    std::optional<SwitchBreak> best;
    std::size_t bestSize = std::numeric_limits<std::size_t>::max();
    for (const Block& block : all) {
        if (instructions[block.end].op != spv::Op::OpSwitch || instructions[block.end - 1].op != spv::Op::OpSelectionMerge)
            continue;
        const Loop* loop = nullptr;
        for (const Loop& candidate : loops)
            if (candidate.body.contains(block.label) && (!loop || candidate.body.size() < loop->body.size()))
                loop = &candidate;
        if (!loop)
            continue;
        SwitchBreak found{block.label, instructions[block.end - 1].operands[0], loop->merge, loop->next, {}};
        const std::set<std::uint32_t> body = reach(next[block.label], {found.merge, found.loopMerge, found.loopContinue});
        for (const std::uint32_t label : body)
            if (std::ranges::contains(next[label], found.loopMerge))
                found.breaks.push_back(label);
        if (found.breaks.empty() || body.size() >= bestSize)
            continue;
        bestSize = body.size();
        best = std::move(found);
    }
    return best;
}

std::expected<void, Error> SpirvModule::lowerSwitchBreak(const SwitchBreak& found) {
    if (found.merge == found.loopContinue)
        return fail(ErrorCode::Unsupported, "WGSL cannot leave a loop from a switch that ends at the continue target of the loop.");
    const std::uint32_t boolean = type(spv::Op::OpTypeBool, {});
    const std::uint32_t pointer = type(spv::Op::OpTypePointer, {static_cast<std::uint32_t>(spv::StorageClass::Function), boolean});
    const std::uint32_t yes = boolConstant(true);
    const std::uint32_t no = boolConstant(false);
    const std::uint32_t flag = newId();
    const std::vector<Block> all = blocks();
    const auto blockOf = [&](const std::uint32_t label) -> const Block& { return *std::ranges::find(all, label, &Block::label); };
    for (const std::uint32_t label : {found.merge, found.loopMerge})
        for (std::size_t index = blockOf(label).begin; index < blockOf(label).end; ++index)
            if (instructions[index].op == spv::Op::OpPhi || instructions[index].op == spv::Op::OpLoopMerge)
                return fail(ErrorCode::Unsupported, "WGSL cannot leave a loop from a switch whose merge block has a phi or starts a loop.");
    std::vector<std::pair<std::size_t, std::vector<Instruction>>> insertions;
    for (const std::uint32_t label : found.breaks) {
        const Block& block = blockOf(label);
        Instruction& terminator = instructions[block.end];
        if (terminator.op == spv::Op::OpBranch) {
            terminator.operands[0] = found.merge;
            insertions.push_back({terminatorPosition(block), {{spv::Op::OpStore, {flag, yes}}}});
            continue;
        }
        if (terminator.op != spv::Op::OpBranchConditional)
            return fail(ErrorCode::Unsupported, "WGSL cannot leave a loop from a case of a switch.");
        const bool onTrue = terminator.operands[1] == found.loopMerge;
        const bool onFalse = terminator.operands[2] == found.loopMerge;
        std::ranges::replace(std::span(terminator.operands).subspan(1, 2), found.loopMerge, found.merge);
        if (onTrue && onFalse) {
            insertions.push_back({terminatorPosition(block), {{spv::Op::OpStore, {flag, yes}}}});
            continue;
        }
        if (onTrue) {
            insertions.push_back({terminatorPosition(block), {{spv::Op::OpStore, {flag, terminator.operands[0]}}}});
            continue;
        }
        const std::uint32_t negated = newId();
        insertions.push_back({terminatorPosition(block), {{spv::Op::OpLogicalNot, {boolean, negated, terminator.operands[0]}}, {spv::Op::OpStore, {flag, negated}}}});
    }
    insertions.push_back({terminatorPosition(blockOf(found.header)), {{spv::Op::OpStore, {flag, no}}}});
    const std::uint32_t loaded = newId();
    const std::uint32_t leave = newId();
    const std::uint32_t rest = newId();
    insertions.push_back({blockOf(found.merge).begin + 1,
                          {{spv::Op::OpLoad, {boolean, loaded, flag}},
                           {spv::Op::OpSelectionMerge, {rest, 0}},
                           {spv::Op::OpBranchConditional, {loaded, leave, rest}},
                           {spv::Op::OpLabel, {leave}},
                           {spv::Op::OpBranch, {found.loopMerge}},
                           {spv::Op::OpLabel, {rest}}}});
    std::size_t first = blockOf(found.header).begin;
    while (instructions[first].op != spv::Op::OpFunction)
        --first;
    while (instructions[first].op != spv::Op::OpLabel)
        ++first;
    insertions.push_back({first + 1, {{spv::Op::OpVariable, {pointer, flag, static_cast<std::uint32_t>(spv::StorageClass::Function)}}}});
    std::ranges::stable_sort(insertions, std::greater{}, [](const auto& insertion) { return insertion.first; });
    for (const auto& [position, added] : insertions)
        instructions.insert(instructions.begin() + static_cast<std::ptrdiff_t>(position), added.begin(), added.end());
    return {};
}

std::vector<SpirvModule::Block> SpirvModule::blocks() const {
    std::vector<Block> result;
    for (std::size_t index = 0; index < instructions.size(); ++index) {
        if (instructions[index].op == spv::Op::OpLabel)
            result.push_back({instructions[index].operands[0], index, 0});
        else if (isTerminator(instructions[index].op) && !result.empty())
            result.back().end = index;
    }
    return result;
}

std::vector<std::uint32_t> SpirvModule::successors(const Instruction& terminator) {
    if (terminator.op == spv::Op::OpBranch)
        return {terminator.operands[0]};
    if (terminator.op == spv::Op::OpBranchConditional)
        return {terminator.operands[1], terminator.operands[2]};
    if (terminator.op != spv::Op::OpSwitch)
        return {};
    const std::vector<std::uint32_t> operands = terminator.operands;
    const std::size_t literal = definition(definition(operands[0])->operands[0])->operands[1] > 32 ? 2 : 1;
    std::vector<std::uint32_t> result{operands[1]};
    for (std::size_t index = 2; index + literal < operands.size(); index += literal + 1)
        result.push_back(operands[index + literal]);
    return result;
}

//! The order is the structured order of spirv-opt: a reverse post-order that visits the merge and continue targets of a header first, so each block comes after its dominators.
void SpirvModule::sortBlocks() {
    std::map<std::uint32_t, std::vector<std::uint32_t>> next;
    for (const Block& block : blocks()) {
        const Instruction& merge = instructions[block.end - 1];
        if (merge.op == spv::Op::OpSelectionMerge || merge.op == spv::Op::OpLoopMerge)
            next[block.label].push_back(merge.operands[0]);
        if (merge.op == spv::Op::OpLoopMerge)
            next[block.label].push_back(merge.operands[1]);
        std::ranges::copy(successors(instructions[block.end]), std::back_inserter(next[block.label]));
    }
    std::vector<Instruction> output;
    output.reserve(instructions.size());
    std::size_t index = 0;
    while (index < instructions.size()) {
        if (instructions[index].op != spv::Op::OpLabel) {
            output.push_back(std::move(instructions[index++]));
            continue;
        }
        std::vector<std::uint32_t> order;
        std::map<std::uint32_t, std::vector<Instruction>> bodies;
        while (instructions[index].op == spv::Op::OpLabel) {
            const std::uint32_t label = instructions[index].operands[0];
            order.push_back(label);
            do
                bodies[label].push_back(std::move(instructions[index]));
            while (!isTerminator(instructions[index++].op));
        }
        std::vector<std::uint32_t> postOrder;
        std::set<std::uint32_t> seen{order.front()};
        std::vector<std::pair<std::uint32_t, std::size_t>> stack{{order.front(), 0}};
        while (!stack.empty()) {
            auto& [label, child] = stack.back();
            if (child == next[label].size()) {
                postOrder.push_back(label);
                stack.pop_back();
                continue;
            }
            const std::uint32_t target = next[label][child++];
            if (seen.insert(target).second)
                stack.push_back({target, 0});
        }
        std::vector<std::uint32_t> sorted(postOrder.rbegin(), postOrder.rend());
        for (const std::uint32_t label : order)
            if (!seen.contains(label))
                sorted.push_back(label);
        for (const std::uint32_t label : sorted)
            std::ranges::move(bodies[label], std::back_inserter(output));
    }
    instructions = std::move(output);
}

std::size_t SpirvModule::terminatorPosition(const Block& block) const {
    const spv::Op merge = instructions[block.end - 1].op;
    return merge == spv::Op::OpSelectionMerge || merge == spv::Op::OpLoopMerge ? block.end - 1 : block.end;
}

std::expected<void, Error> SpirvModule::lowerPackedComponents() {
    std::map<std::pair<spv::StorageClass, std::uint32_t>, std::vector<std::uint32_t>> groups;
    for (const std::uint32_t variable : variables())
        if (decorated(variable, spv::Decoration::Component))
            groups[{storageClass(variable), decoration(variable, spv::Decoration::Location)}];
    for (const std::uint32_t variable : variables()) {
        if (!decorated(variable, spv::Decoration::Location))
            continue;
        const auto group = groups.find({storageClass(variable), decoration(variable, spv::Decoration::Location)});
        if (group != groups.end())
            group->second.push_back(variable);
    }
    for (const auto& [slot, members] : groups)
        if (auto merged = mergeComponents(slot.first, slot.second, members); !merged)
            return merged;
    return {};
}

std::expected<void, Error> SpirvModule::mergeComponents(const spv::StorageClass storage, const std::uint32_t location,
                                                        const std::vector<std::uint32_t>& members) {
    const auto shapeOf = [&](const std::uint32_t variable) -> std::pair<std::uint32_t, std::uint32_t> {
        const Instruction& value = *definition(pointee(variable));
        if (value.op == spv::Op::OpTypeVector)
            return {value.operands[1], value.operands[2]};
        if (value.op == spv::Op::OpTypeFloat || value.op == spv::Op::OpTypeInt)
            return {value.operands[0], 1};
        return {0, 0};
    };
    constexpr std::array interpolations{spv::Decoration::Flat, spv::Decoration::NoPerspective, spv::Decoration::Centroid, spv::Decoration::Sample};
    const auto interpolationOf = [&](const std::uint32_t variable) {
        std::vector<spv::Decoration> result;
        for (const spv::Decoration kind : interpolations)
            if (decorated(variable, kind))
                result.push_back(kind);
        return result;
    };
    const std::uint32_t scalar = shapeOf(members.front()).first;
    std::uint32_t used = 0;
    for (const std::uint32_t member : members) {
        const auto [memberScalar, width] = shapeOf(member);
        const std::uint32_t component = decoration(member, spv::Decoration::Component);
        const std::uint32_t mask = ((1u << width) - 1u) << component;
        if (memberScalar == 0 || memberScalar != scalar || component + width > 4 || (used & mask) != 0 || interpolationOf(member) != interpolationOf(members.front()))
            return fail(ErrorCode::InvalidShader, std::format("WGSL cannot merge the variables at location {}.", location));
        used |= mask;
    }
    const auto storageWord = static_cast<std::uint32_t>(storage);
    const std::uint32_t wide = type(spv::Op::OpTypeVector, {scalar, 4});
    const std::uint32_t scalarPointer = type(spv::Op::OpTypePointer, {storageWord, scalar});
    const std::uint32_t uint = type(spv::Op::OpTypeInt, {32, 0});
    const std::uint32_t merged = newId();
    addGlobal({spv::Op::OpVariable, {type(spv::Op::OpTypePointer, {storageWord, wide}), merged, storageWord}});
    addAnnotation({spv::Op::OpDecorate, {merged, static_cast<std::uint32_t>(spv::Decoration::Location), location}});
    for (const spv::Decoration kind : interpolationOf(members.front()))
        addAnnotation({spv::Op::OpDecorate, {merged, static_cast<std::uint32_t>(kind)}});
    for (Instruction& instruction : instructions)
        if (instruction.op == spv::Op::OpEntryPoint && std::ranges::find_first_of(instruction.operands, members) != instruction.operands.end())
            instruction.operands.push_back(merged);
    const std::array componentIndices{constant(uint, 0), constant(uint, 1), constant(uint, 2), constant(uint, 3)};
    std::map<std::uint32_t, std::uint32_t> constants;
    for (const Instruction& instruction : instructions)
        if (instruction.op == spv::Op::OpConstant && instruction.operands.size() == 3)
            constants[instruction.operands[1]] = instruction.operands[2];
    bool dynamicIndex = false;
    for (const std::uint32_t member : members) {
        const auto [memberScalar, width] = shapeOf(member);
        const std::uint32_t component = decoration(member, spv::Decoration::Component);
        const auto componentIndex = [&](const std::uint32_t offset) { return componentIndices[component + offset]; };
        rewriteFunctions([&](const Instruction& instruction, std::vector<Instruction>& output) {
            if (instruction.op == spv::Op::OpLoad && instruction.operands[2] == member) {
                const std::uint32_t loaded = newId();
                output.push_back({spv::Op::OpLoad, {wide, loaded, merged}});
                if (width == 1) {
                    output.push_back({spv::Op::OpCompositeExtract, {instruction.operands[0], instruction.operands[1], loaded, component}});
                    return true;
                }
                std::vector<std::uint32_t> shuffle{instruction.operands[0], instruction.operands[1], loaded, loaded};
                for (std::uint32_t i = 0; i < width; ++i)
                    shuffle.push_back(component + i);
                output.push_back({spv::Op::OpVectorShuffle, std::move(shuffle)});
                return true;
            }
            if (instruction.op == spv::Op::OpStore && instruction.operands[0] == member) {
                for (std::uint32_t i = 0; i < width; ++i) {
                    const std::uint32_t pointer = newId();
                    output.push_back({spv::Op::OpAccessChain, {scalarPointer, pointer, merged, componentIndex(i)}});
                    std::uint32_t stored = instruction.operands[1];
                    if (width > 1) {
                        stored = newId();
                        output.push_back({spv::Op::OpCompositeExtract, {scalar, stored, instruction.operands[1], i}});
                    }
                    output.push_back({spv::Op::OpStore, {pointer, stored}});
                }
                return true;
            }
            const bool chain = instruction.op == spv::Op::OpAccessChain || instruction.op == spv::Op::OpInBoundsAccessChain;
            if (!chain || instruction.operands[2] != member)
                return false;
            const auto index = instruction.operands.size() == 4 ? constants.find(instruction.operands[3]) : constants.end();
            if (index == constants.end() || component + index->second > 3) {
                dynamicIndex = true;
                return false;
            }
            output.push_back({instruction.op, {instruction.operands[0], instruction.operands[1], merged, componentIndex(index->second)}});
            return true;
        });
        removeVariable(member);
    }
    if (dynamicIndex)
        return fail(ErrorCode::InvalidShader, std::format("A variable at location {} has a dynamic index.", location));
    return {};
}

std::expected<void, Error> SpirvModule::lowerTexelBuffers() {
    std::vector<std::uint32_t> images;
    for (const Instruction& instruction : instructions)
        if (instruction.op == spv::Op::OpTypeImage && instruction.operands[2] == static_cast<std::uint32_t>(spv::Dim::Buffer))
            images.push_back(instruction.operands[0]);
    for (const std::uint32_t image : images) {
        const Instruction& imageType = *definition(image);
        if (imageType.operands[6] != 1)
            return fail(ErrorCode::Unsupported, "WebGPU has no storage texel buffers.");
        const std::uint32_t element = type(spv::Op::OpTypeVector, {imageType.operands[1], 4});
        const std::uint32_t array = newType(spv::Op::OpTypeRuntimeArray, {element});
        const std::uint32_t block = newType(spv::Op::OpTypeStruct, {array});
        const auto storage = static_cast<std::uint32_t>(spv::StorageClass::StorageBuffer);
        const std::uint32_t blockPointer = newType(spv::Op::OpTypePointer, {storage, block});
        const std::uint32_t elementPointer = type(spv::Op::OpTypePointer, {storage, element});
        const std::uint32_t uint = type(spv::Op::OpTypeInt, {32, 0});
        const std::uint32_t zero = constant(uint, 0);
        addAnnotation({spv::Op::OpDecorate, {array, static_cast<std::uint32_t>(spv::Decoration::ArrayStride), 16}});
        addAnnotation({spv::Op::OpDecorate, {block, static_cast<std::uint32_t>(spv::Decoration::Block)}});
        addAnnotation({spv::Op::OpMemberDecorate, {block, 0, static_cast<std::uint32_t>(spv::Decoration::Offset), 0}});
        addAnnotation({spv::Op::OpMemberDecorate, {block, 0, static_cast<std::uint32_t>(spv::Decoration::NonWritable)}});
        std::map<std::uint32_t, std::uint32_t> loads;
        std::vector<std::uint32_t> imagePointers;
        for (const std::uint32_t variable : variables()) {
            if (pointee(variable) != image)
                continue;
            if (storageClass(variable) != spv::StorageClass::UniformConstant)
                return fail(ErrorCode::Unsupported, "A texel buffer must be a plain variable.");
            imagePointers.push_back(definition(variable)->operands[0]);
            std::erase_if(instructions, [&](const Instruction& instruction) { return instruction.op == spv::Op::OpVariable && instruction.operands[1] == variable; });
            addGlobal({spv::Op::OpVariable, {blockPointer, variable, storage}});
        }
        for (const std::uint32_t pointer : imagePointers)
            removeDefinition(pointer);
        rewriteFunctions([&](const Instruction& instruction, std::vector<Instruction>& output) {
            if (instruction.op == spv::Op::OpLoad && instruction.operands[0] == image) {
                loads[instruction.operands[1]] = instruction.operands[2];
                return true;
            }
            if (instruction.op == spv::Op::OpImageFetch && loads.contains(instruction.operands[2])) {
                const std::uint32_t pointer = newId();
                output.push_back({spv::Op::OpAccessChain, {elementPointer, pointer, loads[instruction.operands[2]], zero, instruction.operands[3]}});
                output.push_back({spv::Op::OpLoad, {instruction.operands[0], instruction.operands[1], pointer}});
                return true;
            }
            if (instruction.op != spv::Op::OpImageQuerySize || !loads.contains(instruction.operands[2]))
                return false;
            const std::uint32_t length = instruction.operands[0] == uint ? instruction.operands[1] : newId();
            output.push_back({spv::Op::OpArrayLength, {uint, length, loads[instruction.operands[2]], 0}});
            if (length != instruction.operands[1])
                output.push_back({spv::Op::OpBitcast, {instruction.operands[0], instruction.operands[1], length}});
            return true;
        });
        removeDefinition(image);
    }
    if (images.empty())
        return {};
    removeCapability(spv::Capability::SampledBuffer);
    if (version >= versionWithStorageBuffers)
        return {};
    Instruction extension{spv::Op::OpExtension, literalString("SPV_KHR_storage_buffer_storage_class")};
    const auto capabilities = std::ranges::find_if(instructions, [](const Instruction& instruction) { return instruction.op != spv::Op::OpCapability; });
    instructions.insert(capabilities, std::move(extension));
    return {};
}

std::expected<void, Error> SpirvModule::mergeAliasedBuffers() {
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::vector<std::uint32_t>> slots;
    for (const std::uint32_t variable : variables())
        if (decorated(variable, spv::Decoration::Binding))
            slots[{decoration(variable, spv::Decoration::DescriptorSet), decoration(variable, spv::Decoration::Binding)}].push_back(variable);
    const auto elementOf = [&](const std::uint32_t variable) -> std::uint32_t {
        const Instruction& block = *definition(pointee(variable));
        if (block.op != spv::Op::OpTypeStruct || block.operands.size() != 2)
            return 0;
        const Instruction& array = *definition(block.operands[1]);
        if (array.op != spv::Op::OpTypeRuntimeArray)
            return 0;
        const Instruction& element = *definition(array.operands[1]);
        if ((element.op != spv::Op::OpTypeInt && element.op != spv::Op::OpTypeFloat) || element.operands[1] != 32)
            return 0;
        return array.operands[1];
    };
    const std::uint32_t uint = type(spv::Op::OpTypeInt, {32, 0});
    for (const auto& [slot, aliases] : slots) {
        if (aliases.size() < 2)
            continue;
        const auto canonical = std::ranges::find_if(aliases, [&](const std::uint32_t variable) { return elementOf(variable) == uint; });
        if (canonical == aliases.end())
            return fail(ErrorCode::InvalidShader, std::format("Set {} binding {} has several variables and none of them reads uint words.", slot.first, slot.second));
        const std::uint32_t target = *canonical;
        const std::uint32_t wordPointer = type(spv::Op::OpTypePointer, {static_cast<std::uint32_t>(storageClass(target)), uint});
        for (const std::uint32_t alias : aliases) {
            if (alias == target)
                continue;
            const std::uint32_t element = elementOf(alias);
            if (element == 0)
                return fail(ErrorCode::InvalidShader, std::format("Set {} binding {} has several variables of different kinds.", slot.first, slot.second));
            std::vector<std::uint32_t> pointers;
            rewriteFunctions([&](const Instruction& instruction, std::vector<Instruction>& output) {
                const bool chain = instruction.op == spv::Op::OpAccessChain || instruction.op == spv::Op::OpInBoundsAccessChain;
                if (chain && instruction.operands[2] == alias) {
                    std::vector<std::uint32_t> operands = instruction.operands;
                    operands[0] = wordPointer;
                    operands[2] = target;
                    output.push_back({instruction.op, std::move(operands)});
                    pointers.push_back(instruction.operands[1]);
                    return true;
                }
                if (instruction.op == spv::Op::OpLoad && std::ranges::contains(pointers, instruction.operands[2])) {
                    const std::uint32_t word = newId();
                    output.push_back({spv::Op::OpLoad, {uint, word, instruction.operands[2]}});
                    output.push_back({spv::Op::OpBitcast, {instruction.operands[0], instruction.operands[1], word}});
                    return true;
                }
                if (instruction.op != spv::Op::OpStore || !std::ranges::contains(pointers, instruction.operands[0]))
                    return false;
                const std::uint32_t word = newId();
                output.push_back({spv::Op::OpBitcast, {uint, word, instruction.operands[1]}});
                output.push_back({spv::Op::OpStore, {instruction.operands[0], word}});
                return true;
            });
            removeVariable(alias);
        }
    }
    return {};
}

std::expected<void, Error> SpirvModule::checkBindings() const {
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> slots;
    for (const std::uint32_t variable : variables()) {
        if (!decorated(variable, spv::Decoration::Binding))
            continue;
        const std::pair slot{decoration(variable, spv::Decoration::DescriptorSet), decoration(variable, spv::Decoration::Binding)};
        if (!slots.emplace(slot, variable).second)
            return fail(ErrorCode::InvalidShader, std::format("Set {} binding {} has two different resources.", slot.first, slot.second));
    }
    return {};
}

std::expected<void, Error> SpirvModule::checkFiniteConstants() {
    for (const Instruction& instruction : instructions) {
        if (instruction.op != spv::Op::OpConstant || definition(instruction.operands[0])->op != spv::Op::OpTypeFloat)
            continue;
        const std::uint32_t width = definition(instruction.operands[0])->operands[1];
        const std::uint32_t high = instruction.operands.back();
        const bool finite = width == 16 ? (high & 0x7c00u) != 0x7c00u : width == 32 ? (high & 0x7f800000u) != 0x7f800000u : (high & 0x7ff00000u) != 0x7ff00000u;
        if (!finite)
            return fail(ErrorCode::InvalidShader, std::format("WGSL has no infinity or NaN, and constant %{} is one.", instruction.operands[1]));
    }
    return {};
}

std::uint32_t SpirvModule::resultId(const Instruction& instruction) {
    bool hasResult = false;
    bool hasType = false;
    spv::HasResultAndType(instruction.op, &hasResult, &hasType);
    if (!hasResult)
        return 0;
    return instruction.operands[hasType ? 1 : 0];
}

SpirvModule::Instruction* SpirvModule::definition(const std::uint32_t id) {
    const auto found = std::ranges::find_if(instructions, [&](const Instruction& instruction) { return resultId(instruction) == id; });
    return found == instructions.end() ? nullptr : &*found;
}

std::uint32_t SpirvModule::decoration(const std::uint32_t id, const spv::Decoration kind) const {
    for (const Instruction& instruction : instructions)
        if (instruction.op == spv::Op::OpDecorate && instruction.operands[0] == id && instruction.operands[1] == static_cast<std::uint32_t>(kind))
            return instruction.operands.size() > 2 ? instruction.operands[2] : 0;
    return 0;
}

bool SpirvModule::decorated(const std::uint32_t id, const spv::Decoration kind) const {
    return std::ranges::any_of(instructions, [&](const Instruction& instruction) {
        return instruction.op == spv::Op::OpDecorate && instruction.operands[0] == id && instruction.operands[1] == static_cast<std::uint32_t>(kind);
    });
}

std::vector<std::uint32_t> SpirvModule::variables() const {
    std::vector<std::uint32_t> result;
    for (const Instruction& instruction : instructions) {
        if (instruction.op == spv::Op::OpFunction)
            break;
        if (instruction.op == spv::Op::OpVariable)
            result.push_back(instruction.operands[1]);
    }
    return result;
}

std::uint32_t SpirvModule::pointee(const std::uint32_t variable) {
    return definition(definition(variable)->operands[0])->operands[2];
}

spv::StorageClass SpirvModule::storageClass(const std::uint32_t variable) {
    return static_cast<spv::StorageClass>(definition(variable)->operands[2]);
}

std::uint32_t SpirvModule::type(const spv::Op op, std::vector<std::uint32_t> operands) {
    for (const Instruction& instruction : instructions)
        if (instruction.op == op && std::ranges::equal(std::span(instruction.operands).subspan(1), operands))
            return instruction.operands[0];
    return newType(op, std::move(operands));
}

std::uint32_t SpirvModule::newType(const spv::Op op, std::vector<std::uint32_t> operands) {
    const std::uint32_t id = newId();
    operands.insert(operands.begin(), id);
    addGlobal({op, std::move(operands)});
    return id;
}

std::uint32_t SpirvModule::constant(const std::uint32_t type, const std::uint32_t value) {
    for (const Instruction& instruction : instructions)
        if (instruction.op == spv::Op::OpConstant && instruction.operands[0] == type && instruction.operands.size() == 3 && instruction.operands[2] == value)
            return instruction.operands[1];
    const std::uint32_t id = newId();
    addGlobal({spv::Op::OpConstant, {type, id, value}});
    return id;
}

std::uint32_t SpirvModule::boolConstant(const bool value) {
    const std::uint32_t boolean = type(spv::Op::OpTypeBool, {});
    const spv::Op op = value ? spv::Op::OpConstantTrue : spv::Op::OpConstantFalse;
    for (const Instruction& instruction : instructions)
        if (instruction.op == op && instruction.operands[0] == boolean)
            return instruction.operands[1];
    const std::uint32_t id = newId();
    addGlobal({op, {boolean, id}});
    return id;
}

std::uint32_t SpirvModule::extInstImport(const std::string_view name) {
    for (const Instruction& instruction : instructions)
        if (instruction.op == spv::Op::OpExtInstImport && reinterpret_cast<const char*>(instruction.operands.data() + 1) == name)
            return instruction.operands[0];
    const std::uint32_t id = newId();
    std::vector<std::uint32_t> operands = literalString(name);
    operands.insert(operands.begin(), id);
    const auto memoryModel = std::ranges::find_if(instructions, [](const Instruction& instruction) { return instruction.op == spv::Op::OpMemoryModel; });
    instructions.insert(memoryModel, {spv::Op::OpExtInstImport, std::move(operands)});
    return id;
}

std::uint32_t SpirvModule::variableAt(const std::uint32_t set, const std::uint32_t binding) const {
    for (const std::uint32_t variable : variables())
        if (decorated(variable, spv::Decoration::Binding) && decoration(variable, spv::Decoration::DescriptorSet) == set &&
            decoration(variable, spv::Decoration::Binding) == binding)
            return variable;
    return 0;
}

bool SpirvModule::referenced(const std::uint32_t id) const {
    return std::ranges::any_of(instructions, [&](const Instruction& instruction) {
        if (resultId(instruction) == id)
            return false;
        const bool named = instruction.op == spv::Op::OpName || instruction.op == spv::Op::OpMemberName;
        if ((named || isAnnotation(instruction.op)) && instruction.operands[0] == id)
            return false;
        return std::ranges::contains(instruction.operands, id);
    });
}

void SpirvModule::addAnnotation(Instruction instruction) {
    const auto last = std::ranges::find_if(instructions.rbegin(), instructions.rend(), [](const Instruction& item) { return isAnnotation(item.op); });
    const auto position = last == instructions.rend()
                              ? std::ranges::find_if(instructions, [](const Instruction& item) { return resultId(item) != 0 && item.op != spv::Op::OpExtInstImport; })
                              : last.base();
    instructions.insert(position, std::move(instruction));
}

void SpirvModule::addGlobal(Instruction instruction) {
    const auto function = std::ranges::find_if(instructions, [](const Instruction& item) { return item.op == spv::Op::OpFunction; });
    instructions.insert(function, std::move(instruction));
}

void SpirvModule::removeVariable(const std::uint32_t id) {
    for (Instruction& instruction : instructions) {
        if (instruction.op != spv::Op::OpEntryPoint)
            continue;
        const std::size_t interface = 2 + literalStringWords(std::span(instruction.operands).subspan(2));
        std::erase_if(instruction.operands, [&, index = std::size_t{0}](const std::uint32_t operand) mutable { return index++ >= interface && operand == id; });
    }
    removeDefinition(id);
}

void SpirvModule::removeDefinition(const std::uint32_t id) {
    std::erase_if(instructions, [&](const Instruction& instruction) {
        if (resultId(instruction) == id)
            return true;
        const bool named = instruction.op == spv::Op::OpName || instruction.op == spv::Op::OpMemberName;
        return (named || isAnnotation(instruction.op)) && instruction.operands[0] == id;
    });
}

void SpirvModule::removeCapability(const spv::Capability capability) {
    std::erase_if(instructions, [&](const Instruction& instruction) {
        return instruction.op == spv::Op::OpCapability && instruction.operands[0] == static_cast<std::uint32_t>(capability);
    });
}

void SpirvModule::rewriteFunctions(const Rewriter& rewriter) {
    std::vector<Instruction> output;
    output.reserve(instructions.size());
    bool inFunction = false;
    for (Instruction& instruction : instructions) {
        if (instruction.op == spv::Op::OpFunction)
            inFunction = true;
        if (!inFunction || !rewriter(instruction, output))
            output.push_back(std::move(instruction));
    }
    instructions = std::move(output);
}


std::expected<void, Error> SpirvModule::replaceFunction(const std::string_view functionName, const std::span<const std::uint32_t> words) {
    SpirvModule donor;
    if (std::expected<void, Error> read = donor.read(words); !read)
        return read;
    const std::uint32_t host = function(functionName);
    if (host == 0)
        return fail(ErrorCode::InvalidShader, std::format("The shader has no function {} to replace.", functionName));
    const std::uint32_t source = donor.function(functionName);
    if (source == 0)
        return fail(ErrorCode::InvalidShader, std::format("The donor module has no function {}.", functionName));
    Merge merge{donor, {}, {}, {}, {}};
    merge.ids[source] = host;
    mergeCapabilities(donor);
    const std::expected<std::uint32_t, Error> type = mergeGlobal(merge, donor.definition(source)->operands[3]);
    if (!type)
        return std::unexpected(std::move(type.error()));
    if (*type != definition(host)->operands[3])
        return fail(ErrorCode::InvalidShader, std::format("The donor function {} has another signature than the host function.", functionName));

    std::vector<std::uint32_t> parameters;
    const auto [hostBegin, hostEnd] = functionRange(host);
    for (std::size_t index = hostBegin + 1; index < hostEnd && instructions[index].op == spv::Op::OpFunctionParameter; ++index)
        parameters.push_back(instructions[index].operands[1]);
    std::expected<std::vector<Instruction>, Error> body = mergeBody(merge, source, true, parameters);
    if (!body)
        return std::unexpected(std::move(body.error()));
    std::vector<Instruction> copies;
    while (!merge.copies.empty()) {
        const std::uint32_t copy = merge.copies.back();
        merge.copies.pop_back();
        std::expected<std::vector<Instruction>, Error> copied = mergeBody(merge, copy, false, {});
        if (!copied)
            return std::unexpected(std::move(copied.error()));
        std::ranges::move(*copied, std::back_inserter(copies));
    }

    const auto [begin, end] = functionRange(host);
    const std::size_t first = begin + 1 + parameters.size();
    instructions.erase(instructions.begin() + static_cast<std::ptrdiff_t>(first), instructions.begin() + static_cast<std::ptrdiff_t>(end));
    instructions.insert(instructions.begin() + static_cast<std::ptrdiff_t>(first), std::make_move_iterator(body->begin()),
                        std::make_move_iterator(body->end()));
    std::ranges::move(copies, std::back_inserter(instructions));
    for (Instruction& annotation : merge.annotations)
        addAnnotation(std::move(annotation));
    return {};
}

void SpirvModule::mergeCapabilities(const SpirvModule& donor) {
    for (const Instruction& instruction : donor.instructions) {
        if (instruction.op != spv::Op::OpCapability && instruction.op != spv::Op::OpExtension)
            continue;
        if (std::ranges::any_of(instructions, [&](const Instruction& item) { return item.op == instruction.op && item.operands == instruction.operands; }))
            continue;
        const auto last = std::ranges::find_if(instructions.rbegin(), instructions.rend(), [](const Instruction& item) { return item.op == spv::Op::OpCapability; });
        instructions.insert(instruction.op == spv::Op::OpCapability ? instructions.begin() : last.base(), instruction);
    }
}

std::expected<std::uint32_t, Error> SpirvModule::mergeId(Merge& merge, const std::uint32_t id) {
    if (const auto found = merge.ids.find(id); found != merge.ids.end())
        return found->second;
    const Instruction* source = merge.donor.definition(id);
    if (source && source->op == spv::Op::OpFunction)
        return mergeFunction(merge, id);
    return mergeGlobal(merge, id);
}

std::expected<std::uint32_t, Error> SpirvModule::mergeGlobal(Merge& merge, const std::uint32_t id) {
    if (const auto found = merge.ids.find(id); found != merge.ids.end())
        return found->second;
    const Instruction* source = merge.donor.definition(id);
    if (!source)
        return fail(ErrorCode::InvalidShader, std::format("The donor module uses %{} without defining it.", id));
    if (source->op == spv::Op::OpExtInstImport) {
        const std::uint32_t imported = extInstImport(reinterpret_cast<const char*>(source->operands.data() + 1));
        merge.ids[id] = imported;
        return imported;
    }
    if (!isTypeOrConstant(source->op))
        return fail(ErrorCode::InvalidShader, std::format("The donor function uses %{}, which is neither a type nor a constant.", id));
    if (!merge.active.insert(id).second)
        return fail(ErrorCode::InvalidShader, std::format("The donor type %{} is recursive.", id));
    Instruction copy = *source;
    bool hasResult = false;
    bool hasType = false;
    spv::HasResultAndType(copy.op, &hasResult, &hasType);
    const std::size_t result = hasType ? 1 : 0;
    for (const std::size_t position : merge.donor.idPositions(copy)) {
        if (position == result)
            continue;
        const std::expected<std::uint32_t, Error> mapped = mergeId(merge, copy.operands[position]);
        if (!mapped)
            return mapped;
        copy.operands[position] = *mapped;
    }
    std::vector<Instruction> annotations = merge.donor.decorations(id);
    for (Instruction& annotation : annotations)
        for (const std::size_t position : merge.donor.idPositions(annotation))
            if (position != 0) {
                const std::expected<std::uint32_t, Error> mapped = mergeId(merge, annotation.operands[position]);
                if (!mapped)
                    return mapped;
                annotation.operands[position] = *mapped;
            }
    merge.active.erase(id);
    const auto withoutTarget = [](std::vector<Instruction> items) {
        std::vector<std::pair<spv::Op, std::vector<std::uint32_t>>> keys;
        for (Instruction& item : items)
            keys.emplace_back(item.op, std::vector<std::uint32_t>(item.operands.begin() + 1, item.operands.end()));
        std::ranges::sort(keys);
        return keys;
    };
    const auto wanted = withoutTarget(annotations);
    for (const Instruction& candidate : instructions) {
        if (candidate.op == spv::Op::OpFunction)
            break;
        if (candidate.op != copy.op || candidate.operands.size() != copy.operands.size())
            continue;
        bool same = true;
        for (std::size_t position = 0; position < copy.operands.size() && same; ++position)
            same = position == result || candidate.operands[position] == copy.operands[position];
        if (!same || withoutTarget(decorations(candidate.operands[result])) != wanted)
            continue;
        merge.ids[id] = candidate.operands[result];
        return candidate.operands[result];
    }
    const std::uint32_t fresh = newId();
    copy.operands[result] = fresh;
    addGlobal(std::move(copy));
    for (Instruction& annotation : annotations) {
        annotation.operands[0] = fresh;
        addAnnotation(std::move(annotation));
    }
    merge.ids[id] = fresh;
    return fresh;
}

std::expected<std::uint32_t, Error> SpirvModule::mergeFunction(Merge& merge, const std::uint32_t id) {
    if (const auto found = merge.ids.find(id); found != merge.ids.end())
        return found->second;
    const Instruction source = *merge.donor.definition(id);
    const std::expected<std::uint32_t, Error> type = mergeGlobal(merge, source.operands[3]);
    if (!type)
        return type;
    const std::string_view calleeName = merge.donor.name(id);
    if (const std::uint32_t linked = calleeName.empty() ? 0 : function(calleeName, *type)) {
        merge.ids[id] = linked;
        return linked;
    }
    const auto [begin, end] = merge.donor.functionRange(id);
    const bool defined = std::ranges::any_of(std::span(merge.donor.instructions).subspan(begin, end - begin),
                                             [](const Instruction& instruction) { return instruction.op == spv::Op::OpLabel; });
    if (!defined)
        return fail(ErrorCode::InvalidShader, std::format("The donor declares {}, and the host has no function of that name and type.", calleeName));
    const std::uint32_t fresh = newId();
    merge.ids[id] = fresh;
    merge.copies.push_back(id);
    return fresh;
}

//! A replacing body takes the host parameters and leaves out the function header, else the result is the whole function.
std::expected<std::vector<SpirvModule::Instruction>, Error> SpirvModule::mergeBody(Merge& merge, const std::uint32_t function, const bool replacing,
                                                                                    const std::span<const std::uint32_t> parameters) {
    SpirvModule& donor = merge.donor;
    const auto [begin, end] = donor.functionRange(function);
    std::set<std::uint32_t> debugSets;
    for (const Instruction& instruction : donor.instructions)
        if (instruction.op == spv::Op::OpExtInstImport && std::string_view(reinterpret_cast<const char*>(instruction.operands.data() + 1)).starts_with("NonSemantic."))
            debugSets.insert(instruction.operands[0]);
    const auto skipped = [&](const Instruction& instruction) {
        return instruction.op == spv::Op::OpLine || instruction.op == spv::Op::OpNoLine ||
               (instruction.op == spv::Op::OpExtInst && debugSets.contains(instruction.operands[2]));
    };
    std::size_t parameter = 0;
    std::set<std::uint32_t> locals;
    for (std::size_t index = begin; index <= end; ++index) {
        const std::uint32_t id = resultId(donor.instructions[index]);
        if (id == 0 || index == begin || skipped(donor.instructions[index]))
            continue;
        locals.insert(id);
        const bool hostParameter = replacing && donor.instructions[index].op == spv::Op::OpFunctionParameter;
        if (hostParameter && parameter >= parameters.size())
            return fail(ErrorCode::InvalidShader, "The donor function has more parameters than the host function.");
        merge.ids[id] = hostParameter ? parameters[parameter++] : newId();
    }
    for (const Instruction& instruction : donor.instructions) {
        if (!isAnnotation(instruction.op) || !locals.contains(instruction.operands[0]))
            continue;
        Instruction annotation = instruction;
        for (const std::size_t position : donor.idPositions(annotation)) {
            const std::expected<std::uint32_t, Error> mapped = mergeId(merge, annotation.operands[position]);
            if (!mapped)
                return std::unexpected(std::move(mapped.error()));
            annotation.operands[position] = *mapped;
        }
        merge.annotations.push_back(std::move(annotation));
    }
    std::vector<Instruction> output;
    for (std::size_t index = begin; index <= end; ++index) {
        Instruction instruction = donor.instructions[index];
        const bool header = instruction.op == spv::Op::OpFunction || instruction.op == spv::Op::OpFunctionParameter || instruction.op == spv::Op::OpFunctionEnd;
        if (skipped(instruction) || (replacing && header))
            continue;
        for (const std::size_t position : donor.idPositions(instruction)) {
            const std::expected<std::uint32_t, Error> mapped = mergeId(merge, instruction.operands[position]);
            if (!mapped)
                return std::unexpected(std::move(mapped.error()));
            instruction.operands[position] = *mapped;
        }
        output.push_back(std::move(instruction));
    }
    return output;
}

std::vector<std::size_t> SpirvModule::idPositions(const Instruction& instruction) {
    bool hasResult = false;
    bool hasType = false;
    spv::HasResultAndType(instruction.op, &hasResult, &hasType);
    const std::vector<std::uint32_t>& operands = instruction.operands;
    const std::size_t size = operands.size();
    const std::size_t start = (hasType ? 1 : 0) + (hasResult ? 1 : 0);
    std::vector<std::size_t> positions;
    for (std::size_t position = 0; position < start; ++position)
        positions.push_back(position);
    const auto range = [&](const std::size_t first, const std::size_t last) {
        for (std::size_t position = first; position < std::min(last, size); ++position)
            positions.push_back(position);
    };
    const auto memoryOperands = [&](std::size_t position) {
        if (position >= size)
            return position;
        const std::uint32_t mask = operands[position++];
        if ((mask & static_cast<std::uint32_t>(spv::MemoryAccessMask::Aligned)) != 0)
            ++position;
        if ((mask & static_cast<std::uint32_t>(spv::MemoryAccessMask::MakePointerAvailable)) != 0)
            positions.push_back(position++);
        if ((mask & static_cast<std::uint32_t>(spv::MemoryAccessMask::MakePointerVisible)) != 0)
            positions.push_back(position++);
        return position;
    };
    const auto imageOperands = [&](const std::size_t mask) {
        range(start, start + mask);
        range(start + mask + 1, size);
    };
    switch (instruction.op) {
    case spv::Op::OpExtInst:
        range(start, start + 1);
        range(start + 2, size);
        break;
    case spv::Op::OpCompositeExtract:
    case spv::Op::OpArrayLength:
        range(start, start + 1);
        break;
    case spv::Op::OpCompositeInsert:
    case spv::Op::OpVectorShuffle:
        range(start, start + 2);
        break;
    case spv::Op::OpLoad:
        range(start, start + 1);
        memoryOperands(start + 1);
        break;
    case spv::Op::OpStore:
        range(0, 2);
        memoryOperands(2);
        break;
    case spv::Op::OpCopyMemory:
        range(0, 2);
        memoryOperands(memoryOperands(2));
        break;
    case spv::Op::OpCopyMemorySized:
        range(0, 3);
        memoryOperands(memoryOperands(3));
        break;
    case spv::Op::OpCooperativeMatrixLoadKHR:
        range(start, start + 3);
        memoryOperands(start + 3);
        break;
    case spv::Op::OpCooperativeMatrixStoreKHR:
        range(0, 4);
        memoryOperands(4);
        break;
    case spv::Op::OpCooperativeMatrixMulAddKHR:
        range(start, start + 3);
        break;
    case spv::Op::OpSelectionMerge:
        range(0, 1);
        break;
    case spv::Op::OpLoopMerge:
        range(0, 2);
        break;
    case spv::Op::OpBranchConditional:
        range(0, 3);
        break;
    case spv::Op::OpSwitch: {
        range(0, 2);
        const Instruction* selector = definition(operands[0]);
        const Instruction* selectorType = selector ? definition(selector->operands[0]) : nullptr;
        const std::size_t literal = selectorType && selectorType->op == spv::Op::OpTypeInt && selectorType->operands[1] == 64 ? 2 : 1;
        for (std::size_t position = 2 + literal; position < size; position += literal + 1)
            positions.push_back(position);
        break;
    }
    case spv::Op::OpVariable:
        range(3, 4);
        break;
    case spv::Op::OpFunction:
        range(3, 4);
        break;
    case spv::Op::OpTypeVector:
    case spv::Op::OpTypeMatrix:
    case spv::Op::OpTypeImage:
        range(1, 2);
        break;
    case spv::Op::OpTypePointer:
        range(2, 3);
        break;
    case spv::Op::OpTypeForwardPointer:
    case spv::Op::OpLine:
    case spv::Op::OpDecorate:
    case spv::Op::OpMemberDecorate:
    case spv::Op::OpDecorateString:
    case spv::Op::OpMemberDecorateString:
    case spv::Op::OpName:
    case spv::Op::OpMemberName:
        range(0, 1);
        break;
    case spv::Op::OpDecorateId:
        range(0, 1);
        range(2, size);
        break;
    case spv::Op::OpSpecConstantOp:
        range(start + 1, size);
        break;
    case spv::Op::OpTypeInt:
    case spv::Op::OpTypeFloat:
    case spv::Op::OpConstant:
    case spv::Op::OpSpecConstant:
    case spv::Op::OpString:
    case spv::Op::OpExtInstImport:
        break;
    case spv::Op::OpImageSampleImplicitLod:
    case spv::Op::OpImageSampleExplicitLod:
    case spv::Op::OpImageSampleProjImplicitLod:
    case spv::Op::OpImageSampleProjExplicitLod:
    case spv::Op::OpImageFetch:
    case spv::Op::OpImageRead:
    case spv::Op::OpImageSparseSampleImplicitLod:
    case spv::Op::OpImageSparseSampleExplicitLod:
    case spv::Op::OpImageSparseFetch:
    case spv::Op::OpImageSparseRead:
        imageOperands(2);
        break;
    case spv::Op::OpImageSampleDrefImplicitLod:
    case spv::Op::OpImageSampleDrefExplicitLod:
    case spv::Op::OpImageSampleProjDrefImplicitLod:
    case spv::Op::OpImageSampleProjDrefExplicitLod:
    case spv::Op::OpImageGather:
    case spv::Op::OpImageDrefGather:
    case spv::Op::OpImageSparseSampleDrefImplicitLod:
    case spv::Op::OpImageSparseSampleDrefExplicitLod:
    case spv::Op::OpImageSparseGather:
    case spv::Op::OpImageSparseDrefGather:
        imageOperands(3);
        break;
    case spv::Op::OpImageWrite:
        range(0, 3);
        range(4, size);
        break;
    case spv::Op::OpGroupIAdd:
    case spv::Op::OpGroupFAdd:
    case spv::Op::OpGroupFMin:
    case spv::Op::OpGroupUMin:
    case spv::Op::OpGroupSMin:
    case spv::Op::OpGroupFMax:
    case spv::Op::OpGroupUMax:
    case spv::Op::OpGroupSMax:
    case spv::Op::OpGroupNonUniformBallotBitCount:
    case spv::Op::OpGroupNonUniformIAdd:
    case spv::Op::OpGroupNonUniformFAdd:
    case spv::Op::OpGroupNonUniformIMul:
    case spv::Op::OpGroupNonUniformFMul:
    case spv::Op::OpGroupNonUniformSMin:
    case spv::Op::OpGroupNonUniformUMin:
    case spv::Op::OpGroupNonUniformFMin:
    case spv::Op::OpGroupNonUniformSMax:
    case spv::Op::OpGroupNonUniformUMax:
    case spv::Op::OpGroupNonUniformFMax:
    case spv::Op::OpGroupNonUniformBitwiseAnd:
    case spv::Op::OpGroupNonUniformBitwiseOr:
    case spv::Op::OpGroupNonUniformBitwiseXor:
    case spv::Op::OpGroupNonUniformLogicalAnd:
    case spv::Op::OpGroupNonUniformLogicalOr:
    case spv::Op::OpGroupNonUniformLogicalXor:
        range(start, start + 1);
        range(start + 2, size);
        break;
    default:
        range(start, size);
        break;
    }
    return positions;
}

std::vector<SpirvModule::Instruction> SpirvModule::decorations(const std::uint32_t id) const {
    std::vector<Instruction> result;
    for (const Instruction& instruction : instructions)
        if (isAnnotation(instruction.op) && instruction.operands[0] == id)
            result.push_back(instruction);
    return result;
}

std::string_view SpirvModule::name(const std::uint32_t id) const {
    for (const Instruction& instruction : instructions)
        if (instruction.op == spv::Op::OpName && instruction.operands[0] == id)
            return reinterpret_cast<const char*>(instruction.operands.data() + 1);
    return {};
}

std::uint32_t SpirvModule::function(const std::string_view wanted, const std::uint32_t type) {
    std::map<std::uint32_t, std::uint32_t> types;
    for (const Instruction& instruction : instructions)
        if (instruction.op == spv::Op::OpFunction)
            types[instruction.operands[1]] = instruction.operands[3];
    const std::string_view base = withoutSuffix(wanted);
    for (const Instruction& instruction : instructions) {
        if (instruction.op != spv::Op::OpName || withoutSuffix(reinterpret_cast<const char*>(instruction.operands.data() + 1)) != base)
            continue;
        const auto found = types.find(instruction.operands[0]);
        if (found != types.end() && (type == 0 || found->second == type))
            return instruction.operands[0];
    }
    return 0;
}

std::pair<std::size_t, std::size_t> SpirvModule::functionRange(const std::uint32_t id) const {
    std::size_t begin = 0;
    while (begin < instructions.size() && !(instructions[begin].op == spv::Op::OpFunction && instructions[begin].operands[1] == id))
        ++begin;
    std::size_t end = begin;
    while (end < instructions.size() && instructions[end].op != spv::Op::OpFunctionEnd)
        ++end;
    return {begin, end};
}

}
