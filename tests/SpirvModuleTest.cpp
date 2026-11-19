#include <cstdint>
#include <cstring>
#include <vector>

#include "NoorRHI/Shaders/SpirvModule.h"
#include "Replace.h"
#include "ReplaceDonor.slang.h"
#include "ReplaceHost.slang.h"
#include "Test.h"

using namespace NoorRHI;

namespace {

std::vector<std::uint32_t> getWords(const Shader& shader) {
    const std::span<const std::byte> code = shader.getCode(false, false);
    std::vector<std::uint32_t> words(code.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), code.data(), code.size());
    return words;
}

}

TEST_CASE("a replaced function runs the donor body and links donor callees to host functions") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    SpirvModule module;
    REQUIRE_OK(module.read(getWords(ReplaceHost::hostMain)));
    const std::vector<std::uint32_t> donor = getWords(ReplaceDonor::donorMain);
    REQUIRE_OK(module.replaceFunction("shade", donor));
    const std::vector<std::uint32_t> words = module.write();
    const std::span<const std::byte> bytes = std::as_bytes(std::span(words));
    Shader shader = ReplaceHost::hostMain;
    shader.code = {bytes, bytes, bytes, bytes};

    constexpr std::uint32_t count = 256;
    Buffer<std::uint32_t> result;
    REQUIRE_OK(result.open(*device, count));
    ComputePipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, shader));
    CommandList list(*device);
    list.dispatch(pipeline, {count}, Replace::Arguments{result.getPtr(), count});
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::vector<std::uint32_t> values(count);
    REQUIRE_OK(result.download(values));
    std::uint32_t mismatches = 0;
    for (std::uint32_t index = 0; index < count; ++index)
        mismatches += values[index] != index * 3 + 1000;
    CHECK(mismatches == 0u);
}

TEST_CASE("replacing a function that a module lacks is rejected") {
    SpirvModule module;
    REQUIRE_OK(module.read(getWords(ReplaceHost::hostMain)));
    const std::expected<void, Error> replaced = module.replaceFunction("missing", getWords(ReplaceDonor::donorMain));
    REQUIRE(!replaced);
    CHECK(replaced.error().code == ErrorCode::InvalidShader);
}
