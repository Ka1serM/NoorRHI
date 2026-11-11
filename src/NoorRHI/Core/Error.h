#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <utility>

namespace NoorRHI {

enum class ErrorCode : std::uint8_t {
    Unsupported,
    InvalidArgument,
    InvalidShader,
    InvalidState,
    OutOfMemory,
    DeviceLost,
};

struct Error {
    ErrorCode code = ErrorCode::InvalidState;
    std::string message;
};

[[nodiscard]] inline std::unexpected<Error> fail(const ErrorCode code, std::string message) {
    return std::unexpected(Error{code, std::move(message)});
}

}
