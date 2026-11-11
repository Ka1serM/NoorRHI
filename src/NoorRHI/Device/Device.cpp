#include "NoorRHI/Device/Device.h"

#include "NoorRHI/Core/Backend.h"

namespace NoorRHI {

const DeviceFeatures& Device::getFeatures() const {
    return backend->features;
}

const DeviceInfo& Device::getInfo() const {
    return backend->info;
}

std::expected<Token, Error> Device::submit(CommandList&& list, const std::span<const Token> waits) {
    const CommandList consumed = std::move(list);
    if (!backend || !consumed.getBackend())
        return fail(ErrorCode::InvalidState, "submit needs an open device and command list");
    return backend->submit(*consumed.getBackend(), waits);
}

std::expected<void, Error> Device::wait(const Token token) {
    return backend->wait(token);
}

bool Device::isFinished(const Token token) const {
    return backend->isFinished(token);
}

std::expected<void, Error> Device::waitIdle() {
    return backend->waitIdle();
}

}
