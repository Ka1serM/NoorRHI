#include "NoorRHI/Shaders/WgslModule.h"

#include <algorithm>
#include <charconv>
#include <format>

namespace NoorRHI {

namespace {

bool isIdentifierStart(const char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

bool isIdentifierPart(const char c) {
    return isIdentifierStart(c) || (c >= '0' && c <= '9');
}

std::string_view identifierAt(const std::string_view source, const std::size_t position) {
    std::size_t end = position;
    while (end < source.size() && isIdentifierPart(source[end]))
        ++end;
    return source.substr(position, end - position);
}

std::string_view baseName(const std::string_view name) {
    const std::size_t separator = name.rfind('_');
    if (separator == std::string_view::npos || separator + 1 == name.size())
        return name;
    for (std::size_t index = separator + 1; index < name.size(); ++index)
        if (name[index] < '0' || name[index] > '9')
            return name;
    return name.substr(0, separator);
}

std::size_t findBlockEnd(const std::string_view source, const std::size_t from) {
    std::size_t depth = 0;
    for (std::size_t index = source.find('{', from); index < source.size(); ++index) {
        if (source[index] == '{')
            ++depth;
        else if (source[index] == '}' && --depth == 0)
            return index + 1;
    }
    return std::string_view::npos;
}

std::size_t findAtLineStart(const std::string_view source, const std::string_view pattern) {
    for (std::size_t position = source.find(pattern); position != std::string_view::npos; position = source.find(pattern, position + 1))
        if (position == 0 || source[position - 1] == '\n')
            return position;
    return std::string_view::npos;
}

template <class Lookup>
std::string renameIdentifiers(const std::string_view source, const Lookup& lookup) {
    std::string result;
    result.reserve(source.size());
    for (std::size_t index = 0; index < source.size();) {
        if (isIdentifierStart(source[index])) {
            const std::string_view word = identifierAt(source, index);
            result += lookup(word);
            index += word.size();
            continue;
        }
        const bool number = source[index] >= '0' && source[index] <= '9';
        const std::size_t end = number ? index + identifierAt(source, index).size() : index + 1;
        result.append(source.substr(index, end - index));
        index = end;
    }
    return result;
}

}

std::unordered_map<std::string_view, WgslModule::Definition> WgslModule::indexDefinitions() const {
    std::unordered_map<std::string_view, Definition> definitions;
    const std::string_view source = text;
    for (std::size_t line = 0; line < source.size();) {
        const std::string_view rest = source.substr(line);
        const std::size_t keyword = rest.starts_with("fn ") ? 3 : rest.starts_with("struct ") ? 7 : 0;
        if (keyword != 0) {
            const std::string_view name = identifierAt(source, line + keyword);
            if (const auto [found, added] = definitions.try_emplace(baseName(name), Definition{line, name}); !added)
                found->second.name = {};
        }
        const std::size_t next = source.find('\n', line);
        if (next == std::string_view::npos)
            break;
        line = next + 1;
    }
    return definitions;
}

bool WgslModule::hasFunction(const std::string_view name) const {
    const std::unordered_map<std::string_view, Definition> definitions = indexDefinitions();
    const auto found = definitions.find(name);
    return found != definitions.end() && !found->second.name.empty();
}

std::vector<std::uint32_t> WgslModule::listBindings(const std::uint32_t group) const {
    const std::string suffix = std::format(") @group({}) var ", group);
    constexpr std::string_view prefix = "@binding(";
    std::vector<std::uint32_t> bindings;
    const std::string_view source = text;
    for (std::size_t position = findAtLineStart(source, prefix); position != std::string_view::npos;) {
        const std::size_t number = position + prefix.size();
        const std::size_t close = source.find(')', number);
        std::uint32_t binding = 0;
        const bool parsed = std::from_chars(source.data() + number, source.data() + close, binding).ec == std::errc();
        if (parsed && source.substr(close).starts_with(suffix))
            bindings.push_back(binding);
        const std::size_t next = source.find(std::string_view("\n@binding("), number);
        position = next == std::string_view::npos ? next : next + 1;
    }
    return bindings;
}

WgslModule::Declaration WgslModule::findDeclaration(const std::uint32_t group, const std::uint32_t binding) const {
    const std::string prefix = std::format("@binding({}) @group({}) var ", binding, group);
    const std::string_view source = text;
    const std::size_t begin = findAtLineStart(source, prefix);
    if (begin == std::string_view::npos)
        return {};
    const std::size_t lineEnd = std::min(source.find('\n', begin), source.size());
    const std::string_view line = source.substr(begin + prefix.size(), lineEnd - begin - prefix.size());
    const std::size_t colon = line.find(" : ");
    const std::size_t semicolon = line.rfind(';');
    if (colon == std::string_view::npos || semicolon == std::string_view::npos || semicolon < colon)
        return {};
    return {begin, std::min(lineEnd + 1, source.size()), line.substr(0, colon), line.substr(colon + 3, semicolon - colon - 3)};
}

std::expected<void, Error> WgslModule::moveBinding(const std::uint32_t group, const std::uint32_t binding, const std::uint32_t target) {
    const Declaration moved = findDeclaration(group, binding);
    if (moved.begin == std::string::npos)
        return {};
    const Declaration kept = findDeclaration(group, target);
    if (kept.begin == std::string::npos) {
        const std::string from = std::format("@binding({})", binding);
        text.replace(moved.begin, from.size(), std::format("@binding({})", target));
        return {};
    }
    if (kept.type != moved.type)
        return fail(ErrorCode::InvalidShader, std::format("Group {} bindings {} and {} have different types.", group, binding, target));
    const std::string movedName(moved.name);
    const std::string keptName(kept.name);
    text.erase(moved.begin, moved.end - moved.begin);
    text = renameIdentifiers(text, [&](const std::string_view word) { return word == movedName ? std::string_view(keptName) : word; });
    return {};
}

std::expected<void, Error> WgslModule::replaceFunction(const std::string_view name, const std::string_view donor) {
    const std::unordered_map<std::string_view, Definition> definitions = indexDefinitions();
    const auto host = definitions.find(name);
    if (host == definitions.end() || host->second.name.empty())
        return fail(ErrorCode::InvalidShader, std::format("The shader has no function {} to replace.", name));
    const std::size_t hostEnd = findBlockEnd(text, host->second.position);
    const std::size_t donorBegin = findAtLineStart(donor, std::format("fn {}(", name));
    if (donorBegin == std::string_view::npos)
        return fail(ErrorCode::InvalidShader, std::format("The donor has no function {}.", name));
    const std::size_t donorEnd = findBlockEnd(donor, donorBegin);
    if (hostEnd == std::string::npos || donorEnd == std::string_view::npos)
        return fail(ErrorCode::InvalidShader, std::format("The function {} has no closed body.", name));
    const std::string replacement = renameIdentifiers(donor.substr(donorBegin, donorEnd - donorBegin), [&](const std::string_view word) {
        const auto found = definitions.find(word);
        return found == definitions.end() || found->second.name.empty() ? word : found->second.name;
    });
    text.replace(host->second.position, hostEnd - host->second.position, replacement);
    return {};
}

}
