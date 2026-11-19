#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <vector>

#include <spirv/unified1/spirv.hpp11>

#include "NoorRHI/Core/Error.h"

namespace NoorRHI {

class SpirvModule {
public:
    [[nodiscard]] std::expected<void, Error> read(std::span<const std::uint32_t> words);
    [[nodiscard]] std::expected<void, Error> lowerForWebGpu();
    std::vector<std::uint32_t> write() const;

    bool hasCapability(spv::Capability capability) const;
    bool hasStorageImages();
    bool hasFunction(std::string_view name) { return function(name) != 0; }
    //! A variable at binding moves to target, and when target already has a variable of the same type, the two become one.
    [[nodiscard]] std::expected<void, Error> moveBinding(std::uint32_t set, std::uint32_t binding, std::uint32_t target);
    //! The named function takes the donor body, donor types merge by structure, and donor callees link to host functions of the same name and type.
    [[nodiscard]] std::expected<void, Error> replaceFunction(std::string_view name, std::span<const std::uint32_t> donor);

private:
    struct Instruction {
        spv::Op op;
        std::vector<std::uint32_t> operands;
    };
    using Rewriter = std::function<bool(const Instruction&, std::vector<Instruction>&)>;
    struct Block {
        std::uint32_t label = 0;
        std::size_t begin = 0;
        std::size_t end = 0;
    };
    struct Merge {
        SpirvModule& donor;
        std::map<std::uint32_t, std::uint32_t> ids;
        std::set<std::uint32_t> active;
        std::vector<std::uint32_t> copies;
        std::vector<Instruction> annotations;
    };
    struct SwitchBreak {
        std::uint32_t header = 0;
        std::uint32_t merge = 0;
        std::uint32_t loopMerge = 0;
        std::uint32_t loopContinue = 0;
        std::vector<std::uint32_t> breaks;
    };

    void lowerDrawParameters();
    void lowerHalfConversions();
    [[nodiscard]] std::expected<void, Error> lowerSwitchBreaks();
    [[nodiscard]] std::expected<void, Error> lowerSwitchBreak(const SwitchBreak& found);
    std::optional<SwitchBreak> findSwitchBreak();
    std::vector<Block> blocks() const;
    std::vector<std::uint32_t> successors(const Instruction& terminator);
    void sortBlocks();
    [[nodiscard]] std::expected<void, Error> lowerPackedComponents();
    [[nodiscard]] std::expected<void, Error> mergeComponents(spv::StorageClass storage, std::uint32_t location, const std::vector<std::uint32_t>& members);
    [[nodiscard]] std::expected<void, Error> lowerTexelBuffers();
    [[nodiscard]] std::expected<void, Error> mergeAliasedBuffers();
    [[nodiscard]] std::expected<void, Error> checkBindings() const;
    [[nodiscard]] std::expected<void, Error> checkFiniteConstants();

    [[nodiscard]] std::expected<std::uint32_t, Error> mergeId(Merge& merge, std::uint32_t id);
    [[nodiscard]] std::expected<std::uint32_t, Error> mergeGlobal(Merge& merge, std::uint32_t id);
    [[nodiscard]] std::expected<std::uint32_t, Error> mergeFunction(Merge& merge, std::uint32_t id);
    [[nodiscard]] std::expected<std::vector<Instruction>, Error> mergeBody(Merge& merge, std::uint32_t function, bool replacing,
                                                                       std::span<const std::uint32_t> parameters);
    void mergeCapabilities(const SpirvModule& donor);
    std::vector<std::size_t> idPositions(const Instruction& instruction);
    std::vector<Instruction> decorations(std::uint32_t id) const;
    std::string_view name(std::uint32_t id) const;
    std::uint32_t function(std::string_view name, std::uint32_t type = 0);
    std::pair<std::size_t, std::size_t> functionRange(std::uint32_t id) const;

    static std::uint32_t resultId(const Instruction& instruction);
    Instruction* definition(std::uint32_t id);
    std::uint32_t decoration(std::uint32_t id, spv::Decoration kind) const;
    bool decorated(std::uint32_t id, spv::Decoration kind) const;
    std::vector<std::uint32_t> variables() const;
    std::uint32_t pointee(std::uint32_t variable);
    spv::StorageClass storageClass(std::uint32_t variable);

    std::uint32_t newId() { return bound++; }
    std::uint32_t type(spv::Op op, std::vector<std::uint32_t> operands);
    std::uint32_t newType(spv::Op op, std::vector<std::uint32_t> operands);
    std::uint32_t constant(std::uint32_t type, std::uint32_t value);
    std::uint32_t boolConstant(bool value);
    std::uint32_t extInstImport(std::string_view name);
    std::uint32_t variableAt(std::uint32_t set, std::uint32_t binding) const;
    std::size_t terminatorPosition(const Block& block) const;
    bool referenced(std::uint32_t id) const;
    void addAnnotation(Instruction instruction);
    void addGlobal(Instruction instruction);
    void removeVariable(std::uint32_t id);
    void removeDefinition(std::uint32_t id);
    void removeCapability(spv::Capability capability);
    void rewriteFunctions(const Rewriter& rewriter);

    std::uint32_t version = 0;
    std::uint32_t generator = 0;
    std::uint32_t bound = 0;
    std::vector<Instruction> instructions;
};

}
