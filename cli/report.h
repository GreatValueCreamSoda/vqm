// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "options.h"
#include "psnr_hvs.h"
#include "ssimulacra2.h"

namespace vqm::cli {

struct Pair {
    std::string id, original, distorted;
};
struct Frame {
    std::uint64_t index = 0;
    double score = 0;
    std::vector<double> features;
    std::vector<double> raw_norms;
    std::optional<double> distance_p3;
    std::optional<PsnrHvsScore> psnr_hvs;
};
struct Result {
    double score = 0, core_seconds = 0;
    std::vector<Frame> frames;
};
struct PairResult {
    std::vector<Result> metrics;
    double core_seconds = 0, decode_seconds = 0;
    std::uint64_t count = 0;

    [[nodiscard]] std::optional<double> ScoringFps() const {
        if (core_seconds <= 0) return std::nullopt;
        return static_cast<double>(count) / core_seconds;
    }
};

// Preserve the reference tuning output's channel-major norm order.
std::vector<double> RawNorms(const Ssimulacra2Result& result);
void PrintRawNorms(const Ssimulacra2Result& result);
void PrintClip(const Options& options, const Pair& pair, const PairResult& clip);

}  // namespace vqm::cli
