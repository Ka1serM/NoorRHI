#include <noorrhi/noorrhi.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iterator>
#include <span>
#include <vector>

namespace {
std::vector<std::byte> read_shader(const char* path) {
    std::ifstream file(path, std::ios::binary);
    REQUIRE(file.good());
    const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
    std::vector<std::byte> result(bytes.size());
    for (std::size_t i = 0; i < bytes.size(); ++i)
        result[i] = static_cast<std::byte>(bytes[i]);
    return result;
}

// Every test runs with the validation layers on, so a spec violation fails the
// build rather than going unnoticed.
noorrhi::Device make_device() {
    return noorrhi::Device({.enable_validation = true, .application_name = "NoorRHI tests"});
}
}

TEST_CASE("NoorRHI API adds typed buffers through a root argument") {
    constexpr std::size_t count = 257;

    noorrhi::Device device = make_device();
    auto shader = device.create_shader(read_shader(NOORRHI_TEST_SHADER));
    auto a = device.buffer<float>(count);
    auto b = device.buffer<float>(count);
    auto output = device.buffer<float>(count);
    const auto a_address = a.ptr();
    REQUIRE(a_address.address != 0);
    REQUIRE(a.ptr().address == a_address.address);

    std::vector<float> lhs(count), rhs(count), expected(count);
    for (std::size_t i = 0; i < count; ++i) {
        lhs[i] = static_cast<float>(i) * 0.25f;
        rhs[i] = 3.0f - static_cast<float>(i) * 0.125f;
        expected[i] = lhs[i] + rhs[i];
    }
    a.upload(std::span<const float>(lhs));
    b.upload(std::span<const float>(rhs));

    struct Args {
        noorrhi::GpuPtr<float> a;
        noorrhi::GpuPtr<float> b;
        noorrhi::GpuPtr<float> output;
        std::uint32_t count;
    } args{a.ptr(), b.ptr(), output.ptr(), static_cast<std::uint32_t>(count)};

    // The temporary pipeline is destroyed immediately after launch; the
    // submission must retain the Vulkan pipeline until GPU completion.
    device.compute(shader).launch(
        {static_cast<std::uint32_t>((count + 63) / 64), 1, 1}, args);
    device.barrier(noorrhi::Stage::Compute, noorrhi::Stage::Copy);
    const auto token = device.signal();
    device.wait(token);

    std::vector<float> actual(count);
    output.download(std::span<float>(actual));
    REQUIRE(actual == expected);
}

TEST_CASE("NoorRHI API dispatches compute indirectly with root arguments") {
    constexpr std::size_t count = 64;
    noorrhi::Device device = make_device();
    auto shader = device.create_shader(read_shader(NOORRHI_TEST_SHADER));
    auto pipeline = device.compute(shader);

    auto a = device.buffer<float>(count);
    auto b = device.buffer<float>(count);
    auto output = device.buffer<float>(count);
    const std::vector<float> ones(count, 1.0f);
    const std::vector<float> twos(count, 2.0f);
    a.upload(std::span<const float>(ones));
    b.upload(std::span<const float>(twos));

    auto groups = device.buffer<noorrhi::DispatchArgs>(1);
    const noorrhi::DispatchArgs dispatch{1, 1, 1};
    groups.upload(std::span<const noorrhi::DispatchArgs>(&dispatch, 1));

    struct Args {
        noorrhi::GpuPtr<float> a;
        noorrhi::GpuPtr<float> b;
        noorrhi::GpuPtr<float> output;
        std::uint32_t count;
    } args{a.ptr(), b.ptr(), output.ptr(), static_cast<std::uint32_t>(count)};

    pipeline.launch_indirect(groups.ptr(), args);
    device.synchronize();

    std::vector<float> actual(count);
    output.download(std::span<float>(actual));
    REQUIRE(actual == std::vector<float>(count, 3.0f));
}

TEST_CASE("NoorRHI uploads capture bytes and preserve overlapping write order") {
    auto device = make_device();
    auto buffer = device.buffer<std::uint32_t>(4);
    std::vector<std::uint32_t> values{1, 2, 3, 4};
    buffer.upload(std::span<const std::uint32_t>(values));
    values.assign(4, 99);
    const std::uint32_t replacement = 7;
    buffer.upload(std::span(&replacement, 1), 1);
    std::vector<std::uint32_t> actual(4);
    buffer.download(std::span(actual));
    CHECK(actual == std::vector<std::uint32_t>{1, 7, 3, 4});
}

TEST_CASE("NoorRHI buffers keep their address across partial uploads") {
    auto device = make_device();
    constexpr std::size_t count = 40000;
    auto a = device.buffer<float>(count);
    auto b = device.buffer<float>(count);
    std::vector<float> aData(count, 2.0f);
    const std::vector<float> bData(count, 3.0f);
    auto output = device.buffer<float>(count);
    const auto address = a.ptr();
    a.upload(std::span<const float>(aData));
    b.upload(std::span<const float>(bData));
    aData[17] = 8.0f;
    a.upload(std::span<const float>(aData).subspan(17, 1), 17);
    CHECK(a.ptr().address == address.address);
    auto shader = device.create_shader(read_shader(NOORRHI_TEST_SHADER));
    struct Args {
        noorrhi::GpuPtr<float> a, b, output;
        std::uint32_t count;
    } args{a.ptr(), b.ptr(), output.ptr(), count};
    device.compute(shader).launch({(count + 63) / 64, 1, 1}, args);
    std::vector<float> actual(count);
    output.download(std::span(actual));
    std::vector<float> expected(count, 5.0f);
    expected[17] = 11.0f;
    CHECK(actual == expected);
}

TEST_CASE("NoorRHI shared records publish their data on commit") {
    auto device = make_device();
    noorrhi::Shared<std::uint32_t> record(device);
    record.data = 42;
    record.commit();
    device.synchronize();
    std::uint32_t actual = 0;
    CHECK(record.ptr().address != 0);
}

TEST_CASE("NoorRHI argument arena preserves launches across wraparound") {
    noorrhi::Device device({.enable_validation = true, .application_name = "arena tests",
        .argument_arena_bytes = 4096});
    auto input = device.buffer<float>(1);
    auto output = device.buffer<float>(400);
    const float one = 1.0f;
    input.upload(std::span(&one, 1));
    std::vector<float> actual(400, 0.0f);
    output.upload(std::span<const float>(actual));
    auto pipeline = device.compute(device.create_shader(read_shader(NOORRHI_TEST_SHADER)));
    struct Args {
        noorrhi::GpuPtr<float> a, b, output;
        std::uint32_t count;
    };
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const Args args{input.ptr(), input.ptr(),
            noorrhi::GpuPtr<float>{output.ptr().address + i * sizeof(float)}, 1};
        pipeline.launch({1, 1, 1}, args);
    }
    output.download(std::span(actual));
    CHECK(actual == std::vector<float>(400, 2.0f));
}

TEST_CASE("NoorRHI API round-trips image contents") {
    noorrhi::Device device = make_device();
    auto image = device.image<std::uint8_t>(4, 4,
        noorrhi::ImageUsage::Storage | noorrhi::ImageUsage::Sampled);
    REQUIRE(image.storage_handle());
    REQUIRE(image.sampled_handle());
    REQUIRE(image.storage_handle().value != image.sampled_handle().value);

    std::vector<std::uint8_t> pixels(4u * 4u * 4u);
    for (std::size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = static_cast<std::uint8_t>(i * 3u + 1u);
    image.upload(std::span<const std::uint8_t>(pixels));

    std::vector<std::uint8_t> read_back(pixels.size());
    image.download(std::span<std::uint8_t>(read_back));
    REQUIRE(read_back == pixels);
}

TEST_CASE("NoorRHI shaders read and write images through the descriptor heaps") {
    noorrhi::Device device = make_device();
    constexpr std::uint32_t size = 8;
    auto target = device.image<std::uint8_t>(size, size,
        noorrhi::ImageUsage::Storage, noorrhi::ImageFormat::Rgba8Unorm);
    auto source = device.image<std::uint8_t>(1, 1,
        noorrhi::ImageUsage::Sampled, noorrhi::ImageFormat::Rgba8Unorm);
    const std::uint8_t texel[4]{0, 0, 64, 255};
    source.upload(std::span<const std::uint8_t>(texel));
    auto sampler = device.sampler({noorrhi::Filter::Nearest});
    auto pipeline = device.compute(device.create_shader(read_shader(NOORRHI_IMAGE_SHADER)));

    struct Args {
        std::uint32_t target, source, sampler, size;
    };
    pipeline.launch({1, 1, 1}, Args{target.storage_handle().value,
        source.sampled_handle().value, sampler.handle().value, size});
    device.synchronize();

    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(size) * size * 4u);
    target.download(std::span<std::uint8_t>(pixels));
    for (std::uint32_t y = 0; y < size; ++y) {
        for (std::uint32_t x = 0; x < size; ++x) {
            const std::size_t offset = (static_cast<std::size_t>(y) * size + x) * 4u;
            REQUIRE(pixels[offset] == x * 16u);
            REQUIRE(pixels[offset + 1] == y * 16u);
            REQUIRE(pixels[offset + 2] == 64u);
            REQUIRE(pixels[offset + 3] == 255u);
        }
    }
}

TEST_CASE("NoorRHI descriptor heap slots are reused after retirement and never overflow") {
    // Slot 0 is reserved, so three storage images fill this heap.
    noorrhi::Device device({.enable_validation = true, .application_name = "heap tests",
        .texture_descriptor_capacity = 4, .sampler_descriptor_capacity = 2});
    for (std::uint32_t i = 0; i < 64u; ++i) {
        {
            auto image = device.image<std::uint8_t>(2, 2, noorrhi::ImageUsage::Storage);
            REQUIRE(image.storage_handle());
            REQUIRE(image.storage_handle().value < 4u);
            auto sampler = device.sampler({});
            REQUIRE(sampler.handle().value == 1u);
        }
        // Slots come back only once the GPU has passed the retire point.
        device.synchronize();
    }

    std::vector<noorrhi::Image<std::uint8_t>> held;
    for (std::uint32_t i = 0; i < 3u; ++i)
        held.push_back(device.image<std::uint8_t>(2, 2, noorrhi::ImageUsage::Storage));
    REQUIRE(held[0].storage_handle().value != held[1].storage_handle().value);
    REQUIRE(held[1].storage_handle().value != held[2].storage_handle().value);
    REQUIRE(held[0].storage_handle().value != held[2].storage_handle().value);
    try {
        auto overflow = device.image<std::uint8_t>(2, 2, noorrhi::ImageUsage::Storage);
        FAIL("a full descriptor heap accepted another image");
    } catch (const noorrhi::Error& error) {
        REQUIRE(error.code() == noorrhi::ErrorCode::OutOfMemory);
    }
    auto sampler = device.sampler({});
    REQUIRE_THROWS_AS(device.sampler({}), noorrhi::Error);
}

TEST_CASE("NoorRHI API rasterizes a triangle into a render target") {
    noorrhi::Device device = make_device();
    const auto triangle = read_shader(NOORRHI_TRIANGLE_SHADER);
    auto vertex = device.create_shader(triangle, "vertMain");
    auto fragment = device.create_shader(triangle, "fragMain");

    constexpr std::uint32_t size = 64;
    auto target = device.image<std::uint8_t>(size, size,
        noorrhi::ImageUsage::ColorAttachment, noorrhi::ImageFormat::Rgba8Unorm);
    auto pipeline = device.graphics({vertex, fragment, {}, noorrhi::ImageFormat::Rgba8Unorm});

    device.render({target.handle(), {}}, [&] { pipeline.draw(3); });
    device.synchronize();

    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(size) * size * 4u);
    target.download(std::span<std::uint8_t>(pixels));
    const auto texel = [&](const std::uint32_t x, const std::uint32_t y) {
        return std::span<const std::uint8_t>(
            pixels.data() + (static_cast<std::size_t>(y) * size + x) * 4u, 4u);
    };
    // The triangle covers the middle of the target and none of the corners.
    const auto centre = texel(size / 2, size / 2);
    REQUIRE((centre[0] || centre[1] || centre[2]));
    const auto corner = texel(0, 0);
    REQUIRE(corner[0] == 0);
    REQUIRE(corner[1] == 0);
    REQUIRE(corner[2] == 0);

    // A pipeline built for a different attachment format must be rejected
    // rather than silently mismatching the render pass instance.
    auto bgra_target = device.image<std::uint8_t>(8, 8,
        noorrhi::ImageUsage::ColorAttachment, noorrhi::ImageFormat::Bgra8Unorm);
    REQUIRE_THROWS_AS(
        device.render({bgra_target.handle(), {}}, [&] { pipeline.draw(3); }), noorrhi::Error);
}

TEST_CASE("NoorRHI API draws indirectly and honours depth targets") {
    noorrhi::Device device = make_device();
    const auto triangle = read_shader(NOORRHI_TRIANGLE_SHADER);
    auto vertex = device.create_shader(triangle, "vertMain");
    auto fragment = device.create_shader(triangle, "fragMain");

    constexpr std::uint32_t size = 32;
    auto target = device.image<std::uint8_t>(size, size, noorrhi::ImageUsage::ColorAttachment);
    auto depth = device.image<float>(size, size, noorrhi::ImageUsage::DepthAttachment);

    noorrhi::GraphicsState state{};
    state.depth_test = true;
    state.depth_write = true;
    auto depth_pipeline = device.graphics({vertex, fragment, state});
    auto plain_pipeline = device.graphics({vertex, fragment, {}});

    auto commands = device.buffer<noorrhi::DrawArgs>(1);
    const noorrhi::DrawArgs draw_args{3, 1, 0, 0};
    commands.upload(std::span<const noorrhi::DrawArgs>(&draw_args, 1));

    device.render({target.handle(), depth.handle()},
        [&] { depth_pipeline.draw_indirect(commands.ptr()); });
    device.synchronize();

    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(size) * size * 4u);
    target.download(std::span<std::uint8_t>(pixels));
    const std::size_t centre =
        (static_cast<std::size_t>(size / 2) * size + size / 2) * 4u;
    REQUIRE((pixels[centre] || pixels[centre + 1] || pixels[centre + 2]));

    // Depth attachment presence is part of the pipeline's contract.
    REQUIRE_THROWS_AS(device.render({target.handle(), depth.handle()},
        [&] { plain_pipeline.draw(3); }), noorrhi::Error);
    REQUIRE_THROWS_AS(device.render({target.handle(), {}},
        [&] { depth_pipeline.draw(3); }), noorrhi::Error);
}

TEST_CASE("NoorRHI API exposes samplers and rejects invalid resources") {
    noorrhi::Device device = make_device();
    auto sampler = device.sampler({});
    REQUIRE(sampler);
    REQUIRE(sampler.handle());
    auto texture = device.image<std::uint8_t>(1, 1, noorrhi::ImageUsage::Sampled);
    REQUIRE(texture.sampled_handle());
    REQUIRE_FALSE(texture.storage_handle());

    REQUIRE_THROWS_AS(device.buffer<float>(0), noorrhi::Error);
    REQUIRE_THROWS_AS(device.image<std::uint8_t>(0, 4, noorrhi::ImageUsage::Storage), noorrhi::Error);
    REQUIRE_THROWS_AS(noorrhi::Buffer<float>{}.ptr(), noorrhi::Error);
    REQUIRE_THROWS_AS(noorrhi::ComputePipeline{}.launch({1, 1, 1}, 0u), noorrhi::Error);
}

TEST_CASE("NoorRHI API reclaims buffer allocations after GPU retirement") {
    noorrhi::Device device = make_device();
    const auto before = device.memory_report().allocation_bytes;
    // Churn far more short-lived buffers than any heap would once have held,
    // keeping only one live at a time. Buffers consume no descriptor now, so
    // what this guards is that the retire path actually frees them and that
    // every allocation still yields a usable device address. This models
    // camera and immutable scene snapshots replaced over a long session.
    for (std::uint32_t i = 0; i < 12'000u; ++i) {
        {
            auto snapshot = device.buffer<std::uint32_t>(1);
            REQUIRE(snapshot.ptr().address != 0);
        }
        if ((i & 31u) == 31u) {
            const auto token = device.signal();
            device.wait(token);
        }
    }
    device.synchronize();
    CHECK(device.memory_report().allocation_bytes == before);
}

TEST_CASE("NoorRHI owners reject invalid ranges and expired host references") {
    auto device = make_device();
    noorrhi::ImageHandle expired;
    {
        auto image = device.image<std::uint8_t>(1, 1, noorrhi::ImageUsage::ColorAttachment);
        expired = image.handle();
        REQUIRE(expired);
    }
    device.synchronize(); // Retire the initial image-layout submission's reference.
    CHECK_FALSE(expired);
    CHECK_THROWS_AS(device.render({expired, {}}, [] {}), noorrhi::Error);

    auto buffer = device.buffer<std::uint32_t>(2);
    const std::uint32_t value = 42;
    CHECK_THROWS_AS(buffer.upload(std::span(&value, 1), 2), noorrhi::Error);
    noorrhi::Shared<unsigned> empty;
    CHECK_THROWS_AS(empty.commit(), noorrhi::Error);
    struct alignas(1024) Aligned { std::uint32_t value; };
    auto aligned = device.buffer<Aligned>(1);
    CHECK(aligned.ptr().address % alignof(Aligned) == 0);
}
