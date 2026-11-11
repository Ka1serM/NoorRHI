#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "Add.h"
#include "Add.slang.h"
#include "Count.h"
#include "Count.slang.h"
#include "Raster.h"
#include "Raster.slang.h"
#include "StorageImage.h"
#include "StorageImage.slang.h"
#include "Test.h"

using namespace NoorRHI;

TEST_CASE("compute adds buffers through pointers") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    constexpr std::uint32_t count = 1000;
    std::vector<float> a(count);
    std::vector<float> b(count);
    std::iota(a.begin(), a.end(), 0.0f);
    std::iota(b.begin(), b.end(), 1000.0f);
    Buffer<float> first;
    Buffer<float> second;
    Buffer<float> result;
    REQUIRE_OK(first.open(*device, count));
    REQUIRE_OK(second.open(*device, count));
    REQUIRE_OK(result.open(*device, count));
    REQUIRE_OK(first.upload(a));
    REQUIRE_OK(second.upload(b));
    ComputePipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, Add::main));

    CommandList list(*device);
    list.dispatch(pipeline, {count}, Add::Arguments{first.getPtr(), second.getPtr(), result.getPtr(), count});
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::vector<float> sums(count);
    REQUIRE_OK(result.download(sums));
    for (std::uint32_t index = 0; index < count; ++index)
        REQUIRE(sums[index] == a[index] + b[index]);
}

TEST_CASE("a shader from SPIR-V in memory runs like a built shader") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    const std::span<const std::byte> built = Add::main.getCode(false, false);
    const std::vector<std::byte> code(built.begin(), built.end());
    const Shader shader{.code = {code, code, code, code}, .entryPoint = Add::main.entryPoint, .stage = ShaderStage::Compute, .threadGroupSize = Add::main.threadGroupSize};
    constexpr std::uint32_t count = 100;
    std::vector<float> values(count);
    std::iota(values.begin(), values.end(), 1.0f);
    Buffer<float> input;
    Buffer<float> result;
    REQUIRE_OK(input.open(*device, count));
    REQUIRE_OK(result.open(*device, count));
    REQUIRE_OK(input.upload(values));
    ComputePipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, shader));

    CommandList list(*device);
    list.dispatch(pipeline, {count}, Add::Arguments{input.getPtr(), input.getPtr(), result.getPtr(), count});
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::vector<float> sums(count);
    REQUIRE_OK(result.download(sums));
    for (std::uint32_t index = 0; index < count; ++index)
        REQUIRE(sums[index] == 2.0f * values[index]);
}

TEST_CASE("a buffer released before its list is submitted keeps its contents for that list") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    constexpr std::uint32_t count = 256;
    const std::vector<float> ones(count, 1.0f);
    const std::vector<float> twos(count, 2.0f);
    Buffer<float> result;
    REQUIRE_OK(result.open(*device, count));
    ComputePipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, Add::main));

    CommandList list(*device);
    {
        Buffer<float> released;
        REQUIRE_OK(released.open(*device, count));
        REQUIRE_OK(released.upload(ones));
        list.dispatch(pipeline, {count}, Add::Arguments{released.getPtr(), released.getPtr(), result.getPtr(), count});
    }
    Buffer<float> reused;
    REQUIRE_OK(reused.open(*device, count));
    REQUIRE_OK(reused.upload(twos));
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::vector<float> sums(count);
    REQUIRE_OK(result.download(sums));
    for (const float sum : sums)
        REQUIRE(sum == 2.0f);
}

TEST_CASE("each command sees the writes of the commands before it") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    constexpr std::uint32_t count = 1u << 20;
    constexpr std::uint32_t steps = 64;
    Buffer<float> ones;
    Buffer<float> first;
    Buffer<float> second;
    Buffer<float> sum;
    REQUIRE_OK(ones.open(*device, count));
    REQUIRE_OK(first.open(*device, count));
    REQUIRE_OK(second.open(*device, count));
    REQUIRE_OK(sum.open(*device, count));
    REQUIRE_OK(ones.upload(std::vector<float>(count, 1.0f)));
    const std::vector<float> zeros(count, 0.0f);
    ComputePipeline add;
    REQUIRE_OK(add.open(*device, Add::main));
    Readback<float> readback;
    REQUIRE_OK(readback.open(*device, count));

    CommandList list(*device);
    list.upload(first, std::span<const float>(zeros));
    for (std::uint32_t step = 0; step < steps; ++step) {
        const Buffer<float>& from = step % 2 == 0 ? first : second;
        const Buffer<float>& to = step % 2 == 0 ? second : first;
        list.dispatch(add, {count}, Add::Arguments{from.getPtr(), ones.getPtr(), to.getPtr(), count});
    }
    list.beginConcurrent();
    list.dispatch(add, {count}, Add::Arguments{first.getPtr(), ones.getPtr(), second.getPtr(), count});
    list.dispatch(add, {count}, Add::Arguments{first.getPtr(), first.getPtr(), sum.getPtr(), count});
    list.endConcurrent();
    list.dispatch(add, {count}, Add::Arguments{second.getPtr(), sum.getPtr(), first.getPtr(), count});
    list.download(first, readback);
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::expected<std::span<const float>, Error> values = readback.read();
    REQUIRE(values.has_value());
    constexpr float expected = (steps + 1.0f) + 2.0f * steps;
    CHECK(std::ranges::count_if(*values, [](const float value) { return value != expected; }) == 0);
}

TEST_CASE("indirect dispatch reads its groups from a buffer") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    constexpr std::uint32_t count = 256;
    const std::vector<float> ones(count, 1.0f);
    Buffer<float> first;
    Buffer<float> result;
    Buffer<DispatchSize> groups;
    REQUIRE_OK(first.open(*device, count));
    REQUIRE_OK(result.open(*device, count));
    REQUIRE_OK(groups.open(*device, 1));
    REQUIRE_OK(first.upload(ones));
    const std::array<DispatchSize, 1> size{DispatchSize{count / 64, 1, 1}};
    ComputePipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, Add::main));

    CommandList list(*device);
    list.upload(groups, std::span<const DispatchSize>(size));
    list.dispatchIndirect(pipeline, groups.getPtr(), Add::Arguments{first.getPtr(), first.getPtr(), result.getPtr(), count});
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::vector<float> sums(count);
    REQUIRE_OK(result.download(sums));
    CHECK(std::ranges::all_of(sums, [](const float value) { return value == 2.0f; }));
}

TEST_CASE("atomics in atomic memory work next to plain pointer access") {
    static_assert(sizeof(Count::Arguments) == 32 && offsetof(Count::Arguments, counters) == 16);
    Device* device = Test::getDevice();
    REQUIRE(device);
    constexpr std::uint32_t count = 1000;
    constexpr std::uint32_t threshold = 600;
    std::vector<std::uint32_t> values(count);
    std::iota(values.begin(), values.end(), 0u);
    Buffer<std::uint32_t> input;
    Buffer<std::uint32_t> slots;
    Buffer<Atomic<std::uint32_t>> counters;
    REQUIRE_OK(input.open(*device, count));
    REQUIRE_OK(slots.open(*device, count));
    REQUIRE_OK(counters.open(*device, 2));
    REQUIRE_OK(input.upload(values));
    REQUIRE_OK(counters.upload(std::array<Atomic<std::uint32_t>, 2>{}));
    ComputePipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, Count::main));

    CommandList list(*device);
    list.dispatch(pipeline, {count}, Count::Arguments{input.getPtr(), slots.getPtr(), counters.getPtr(), count, threshold});
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::array<Atomic<std::uint32_t>, 2> totals{};
    REQUIRE_OK(counters.download(totals));
    REQUIRE(totals[0].value == count - threshold);
    CHECK(totals[1].value == count - 1);
    std::vector<std::uint32_t> written(count - threshold);
    REQUIRE_OK(slots.download(written));
    std::ranges::sort(written);
    for (std::uint32_t index = 0; index < written.size(); ++index)
        REQUIRE(written[index] == threshold + index);
}

TEST_CASE("command list uploads keep unaligned bytes") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    Buffer<std::uint8_t> bytes;
    REQUIRE_OK(bytes.open(*device, 16));
    const std::vector<std::uint8_t> zeros(16, 0);
    REQUIRE_OK(bytes.upload(zeros));
    const std::array<std::uint8_t, 7> pattern{1, 2, 3, 4, 5, 6, 7};
    CommandList list(*device);
    list.upload(bytes, std::span<const std::uint8_t>(pattern), 3);
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::vector<std::uint8_t> result(16);
    REQUIRE_OK(bytes.download(result));
    const std::vector<std::uint8_t> expected{0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0};
    CHECK(result == expected);
}

TEST_CASE("compute writes storage images and samples images through handles") {
    Device* device = Test::getDevice();
    REQUIRE(device);
    constexpr std::uint32_t size = 16;
    Image target;
    Image source;
    Sampler sampler;
    REQUIRE_OK(target.open(*device, {.width = size, .height = size, .usage = ImageUsage::Storage}));
    REQUIRE_OK(source.open(*device, {.width = 1, .height = 1}));
    REQUIRE_OK(Test::uploadTexels(source, std::array<std::uint32_t, 1>{0xff800000u}));
    REQUIRE_OK(sampler.open(*device, {.filter = Filter::Nearest}));
    ComputePipeline pipeline;
    REQUIRE_OK(pipeline.open(*device, StorageImage::main));

    CommandList list(*device);
    list.dispatch(pipeline, {size, size}, StorageImage::Arguments{target.getStorageHandle(), source.getSampledHandle(), sampler.getHandle(), size});
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::expected<std::vector<std::uint32_t>, Error> texels = Test::downloadTexels(target);
    REQUIRE_OK(texels);
    for (std::uint32_t y = 0; y < size; ++y)
        for (std::uint32_t x = 0; x < size; ++x)
            REQUIRE((*texels)[y * size + x] == (0xff000000u | 0x800000u | (y * 16u) << 8 | x * 16u));
}

TEST_CASE("dispatch counts threads and argument structs have the GPU layout") {
    static_assert(sizeof(Add::Arguments) == 28 && offsetof(Add::Arguments, count) == 24);
    static_assert(sizeof(Raster::Arguments) == 40 && offsetof(Raster::Arguments, targetSize) == 32);
    Device* device = Test::getDevice();
    REQUIRE(device);
    constexpr std::uint32_t count = 77;
    const std::vector<float> ones(count + 1, 1.0f);
    Buffer<float> first;
    Buffer<float> result;
    REQUIRE_OK(first.open(*device, count + 1));
    REQUIRE_OK(result.open(*device, count + 1));
    REQUIRE_OK(first.upload(ones));
    REQUIRE_OK(result.upload(std::vector<float>(count + 1, 0.0f)));
    ComputePipeline pipeline;
    CHECK(pipeline.open(*device, Shader{}).error().code == ErrorCode::InvalidArgument);
    CHECK(pipeline.open(*device, Raster::vertMain).error().code == ErrorCode::InvalidArgument);
    REQUIRE_OK(pipeline.open(*device, Add::main));

    CommandList list(*device);
    list.dispatch(pipeline, {count}, Add::Arguments{first.getPtr(), first.getPtr(), result.getPtr(), count});
    REQUIRE_OK(Test::execute(*device, std::move(list)));
    std::vector<float> sums(count + 1);
    REQUIRE_OK(result.download(sums));
    CHECK(std::all_of(sums.begin(), sums.begin() + count, [](const float value) { return value == 2.0f; }));
    CHECK(sums[count] == 0.0f);
}
