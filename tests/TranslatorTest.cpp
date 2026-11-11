#include <cstdint>
#include <cstring>
#include <vector>

#include "Add.slang.h"
#include "NoorRHI/Shaders/WgslTranslator.h"
#include "Test.h"

using namespace NoorRHI;

TEST_CASE("the SPIR-V of a NoorRHI shader translates to WGSL") {
    const std::span<const std::byte> code = Add::main.getCode(false, false);
    std::vector<std::uint32_t> words(code.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), code.data(), code.size());
    const std::expected<std::string, Error> wgsl = translateToWgsl(words);
    REQUIRE_OK(wgsl);
    REQUIRE(!wgsl->empty());
}

TEST_CASE("a module that is not SPIR-V is rejected") {
    const std::vector<std::uint32_t> words{1, 2, 3, 4, 5};
    const std::expected<std::string, Error> wgsl = translateToWgsl(words);
    REQUIRE(!wgsl);
    REQUIRE(wgsl.error().code == ErrorCode::InvalidShader);
}
