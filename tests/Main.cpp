#include <cstdio>
#include <string_view>

#include "Test.h"

int main(const int argumentCount, char** arguments) {
    const std::string_view filter = argumentCount > 1 ? arguments[1] : "";
    int ran = 0;
    for (const Test::Case& testCase : Test::getCases()) {
        if (!testCase.name.contains(filter))
            continue;
        const int before = Test::getFailures();
        testCase.run();
        std::printf("%s %.*s\n", Test::getFailures() == before ? "pass" : "FAIL", static_cast<int>(testCase.name.size()), testCase.name.data());
        ++ran;
    }
    Test::closeDevice();
    std::printf("%d tests, %d failed checks\n", ran, Test::getFailures());
    return Test::getFailures() == 0 ? 0 : 1;
}
