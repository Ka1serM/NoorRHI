#pragma once

#include <expected>
#include <memory>

#include "NoorRHI/Core/Error.h"

namespace NoorRHI {

class Device;
namespace Backend {
struct TimestampQuery;
}

class TimestampQuery {
public:
    [[nodiscard]] std::expected<void, Error> open(Device& device);

    double getMilliseconds() const;
    Backend::TimestampQuery* getBackend() const { return backend.get(); }
    explicit operator bool() const { return backend != nullptr; }

private:
    std::shared_ptr<Backend::TimestampQuery> backend;
};

}
