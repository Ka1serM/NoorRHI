#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <expected>
#include <string_view>
#include <utility>
#include <vector>

#include "NoorRHI/NoorRHI.h"

namespace Benchmark {

using Result = std::expected<void, NoorRHI::Error>;

struct Case {
    std::string_view name;
    Result (*run)(NoorRHI::Device&);
};

inline std::vector<Case>& getCases() {
    static std::vector<Case> cases;
    return cases;
}

struct Registrar {
    Registrar(const std::string_view name, Result (*run)(NoorRHI::Device&)) { getCases().push_back({name, run}); }
};

struct Timing {
    double recordMs = -1.0;
    double submitMs = -1.0;
    double wallMs = -1.0;
    double gpuMs = -1.0;
};

inline double getNow() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline double getMedian(std::vector<double> values) {
    std::ranges::sort(values);
    return values[values.size() / 2];
}

//! Each sample records, submits and waits for one list; the two warm-up samples are not counted.
template <class Record>
std::expected<Timing, NoorRHI::Error> measure(NoorRHI::Device& device, const int samples, Record&& record) {
    std::vector<double> records;
    std::vector<double> submits;
    std::vector<double> walls;
    std::vector<double> gpus;
    for (int sample = -2; sample < samples; ++sample) {
        NoorRHI::TimestampQuery query;
        if (std::expected<void, NoorRHI::Error> opened = query.open(device); !opened)
            return std::unexpected(std::move(opened.error()));
        const double start = getNow();
        NoorRHI::CommandList list(device);
        list.beginTimestamp(query);
        record(list);
        list.endTimestamp(query);
        if (list.getError())
            return std::unexpected(*list.getError());
        const double recorded = getNow();
        std::expected<NoorRHI::Token, NoorRHI::Error> token = device.submit(std::move(list));
        if (!token)
            return std::unexpected(std::move(token.error()));
        const double submitted = getNow();
        if (std::expected<void, NoorRHI::Error> waited = device.wait(*token); !waited)
            return std::unexpected(std::move(waited.error()));
        const double finished = getNow();
        if (std::expected<void, NoorRHI::Error> idle = device.waitIdle(); !idle)
            return std::unexpected(std::move(idle.error()));
        if (sample < 0)
            continue;
        records.push_back(recorded - start);
        submits.push_back(submitted - recorded);
        walls.push_back(finished - start);
        gpus.push_back(query.getMilliseconds());
    }
    return Timing{getMedian(records), getMedian(submits), getMedian(walls), getMedian(gpus)};
}

template <class Run>
std::expected<double, NoorRHI::Error> measureWall(const int samples, Run&& run) {
    std::vector<double> walls;
    for (int sample = -1; sample < samples; ++sample) {
        const double start = getNow();
        if (std::expected<void, NoorRHI::Error> ran = run(); !ran)
            return std::unexpected(std::move(ran.error()));
        if (sample >= 0)
            walls.push_back(getNow() - start);
    }
    return getMedian(walls);
}

inline void report(const std::string_view name, const Timing& timing, const std::string_view extra = {}) {
    std::printf("BENCH|%.*s|%.4f|%.4f|%.4f|%.4f|%.*s\n", static_cast<int>(name.size()), name.data(), timing.wallMs, timing.gpuMs, timing.recordMs,
                timing.submitMs, static_cast<int>(extra.size()), extra.data());
    std::fflush(stdout);
}

}

#define BENCHMARK_JOIN2(a, b) a##b
#define BENCHMARK_JOIN(a, b) BENCHMARK_JOIN2(a, b)
#define BENCHMARK(name)                                                                                                                                        \
    static Benchmark::Result BENCHMARK_JOIN(benchmarkCase, __LINE__)(NoorRHI::Device & device);                                                                \
    static const Benchmark::Registrar BENCHMARK_JOIN(benchmarkRegistrar, __LINE__)(name, &BENCHMARK_JOIN(benchmarkCase, __LINE__));                            \
    static Benchmark::Result BENCHMARK_JOIN(benchmarkCase, __LINE__)(NoorRHI::Device & device)
#define TRY(expression)                                                                                                                                        \
    if (std::expected<void, NoorRHI::Error> tried = (expression); !tried)                                                                                      \
    return std::unexpected(std::move(tried.error()))
