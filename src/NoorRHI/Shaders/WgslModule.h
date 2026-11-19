#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <utility>

#include "NoorRHI/Core/Error.h"

namespace NoorRHI {

//! Edits WGSL from slangc, which writes each global declaration on one line and appends _N to each name it declares.
class WgslModule {
public:
    explicit WgslModule(std::string text) : text(std::move(text)) {}

    const std::string& getText() const { return text; }
    bool hasFunction(std::string_view name) const;
    std::vector<std::uint32_t> listBindings(std::uint32_t group) const;
    //! A variable at binding moves to target, and when target already has a variable of the same type, the two become one.
    [[nodiscard]] std::expected<void, Error> moveBinding(std::uint32_t group, std::uint32_t binding, std::uint32_t target);
    //! The named function takes the donor function of that name, and the names in the donor link to host functions and structs of the same name.
    [[nodiscard]] std::expected<void, Error> replaceFunction(std::string_view name, std::string_view donor);

private:
    struct Definition {
        std::size_t position = 0;
        std::string_view name;
    };
    struct Declaration {
        std::size_t begin = std::string::npos;
        std::size_t end = 0;
        std::string_view name;
        std::string_view type;
    };

    std::unordered_map<std::string_view, Definition> indexDefinitions() const;
    Declaration findDeclaration(std::uint32_t group, std::uint32_t binding) const;

    std::string text;
};

}
