#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
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
    //! A variable at binding moves to target, and when target already has a variable of the same type, the two become one.
    [[nodiscard]] std::expected<void, Error> moveBinding(std::uint32_t set, std::uint32_t binding, std::uint32_t target);

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
