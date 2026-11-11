#include "NoorRHI/Commands/TimestampQuery.h"
#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Vulkan/Vulkan.h"

namespace NoorRHI {

namespace Backend {

VulkanTimestampQuery::~VulkanTimestampQuery() {
    if (!device)
        return;
    device->retire([owner = device.get(), pool = this->pool] { owner->device->destroyQueryPool(pool); });
}

}

std::expected<void, Error> TimestampQuery::open(Device& device) {
    if (!device)
        return fail(ErrorCode::InvalidState, "the device is not open");
    auto& owner = static_cast<Backend::VulkanDevice&>(*device.getBackend());
    std::expected<vk::QueryPool, Error> pool = Backend::checkVulkan(owner.device->createQueryPool({{}, vk::QueryType::eTimestamp, 2}), "vkCreateQueryPool");
    if (!pool)
        return std::unexpected(std::move(pool.error()));
    auto result = std::make_shared<Backend::VulkanTimestampQuery>();
    result->device = owner.shared_from_this();
    result->pool = *pool;
    backend = std::move(result);
    return {};
}

namespace Backend {

double VulkanTimestampQuery::getMilliseconds() {
    std::array<std::uint64_t, 2> values{};
    const vk::Result result = device->device->getQueryPoolResults(pool, 0, 2, sizeof(values), values.data(), sizeof(values[0]), vk::QueryResultFlagBits::e64);
    if (result != vk::Result::eSuccess)
        return milliseconds;
    std::uint64_t ticks = values[1] - values[0];
    const std::uint32_t validBits = device->getQueue(queue).timestampValidBits;
    if (validBits != 0 && validBits < 64)
        ticks &= (std::uint64_t{1} << validBits) - 1;
    milliseconds = static_cast<double>(ticks) * device->timestampPeriod * 1.0e-6;
    return milliseconds;
}

}

}
