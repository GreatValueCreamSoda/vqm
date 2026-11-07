// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include "report.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>

#include "metrics/vmaf/v1/vmaf_v1_model.h"

namespace vqm::cli {
namespace {

struct FrameStats {
    double min = 0, max = 0;
    std::optional<double> finite_mean, finite_stddev;
    std::uint64_t perfect_frames = 0;
    std::uint64_t infinite_frames = 0;
};

double PerfectFrameScore(Metric metric, const Options& options) {
    switch (metric) {
        case Metric::Ssimulacra2:
        case Metric::VmafV0:
            return 100.0;
        case Metric::VmafV1:
            return internal::VmafV1Parameters(
                       options.vmaf_v1_model.value_or(VmafV1Model::Standard1080)
            )
                .clip[1];
        case Metric::Butteraugli:
            return 0.0;
        case Metric::Cvvdp:
            return 10.0;
        case Metric::PsnrHvs:
            return std::numeric_limits<double>::infinity();
    }
    throw std::logic_error("unknown metric");
}

FrameStats SummarizeFrames(double perfect_score, std::span<const Frame> frames) {
    FrameStats stats;
    stats.min = stats.max = frames.front().score;
    std::uint64_t finite_frames = 0;
    double mean = 0, squared_deviations = 0;
    for (const auto& frame : frames) {
        const auto score = frame.score;
        stats.min = std::min(stats.min, score);
        stats.max = std::max(stats.max, score);
        if (score == perfect_score) ++stats.perfect_frames;
        if (std::isinf(score)) continue;
        const double delta = score - mean;
        mean += delta / static_cast<double>(++finite_frames);
        squared_deviations += delta * (score - mean);
    }
    stats.infinite_frames = frames.size() - finite_frames;
    if (finite_frames != 0) {
        stats.finite_mean = mean;
        stats.finite_stddev =
            std::sqrt(squared_deviations / static_cast<double>(finite_frames));
    }
    return stats;
}

// std::quoted does not escape JSON control characters; paths and sample IDs
// may contain quotes, backslashes or tabs even though TSV fields cannot contain
// tabs.
void JsonString(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    std::cout << '"';
    for (const char raw : value) {
        const auto character = static_cast<unsigned char>(raw);
        if (character == '"' || character == '\\') {
            std::cout << '\\' << static_cast<char>(character);
        } else if (character < 32) {
            std::cout << "\\u00" << hex[character >> 4] << hex[character & 15];
        } else {
            std::cout << static_cast<char>(character);
        }
    }
    std::cout << '"';
}

void JsonFeatures(std::span<const double> features) {
    std::cout << ",\"features\":[";
    for (std::size_t i = 0; i < features.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << features[i];
    }
    std::cout << ']';
}

void JsonScore(double value) {
    if (std::isinf(value))
        JsonString("inf");
    else
        std::cout << value;
}

void JsonPsnrHvs(const PsnrHvsScore& score) {
    std::cout << ",\"y\":";
    JsonScore(score.y);
    std::cout << ",\"cb\":";
    JsonScore(score.cb);
    std::cout << ",\"cr\":";
    JsonScore(score.cr);
    std::cout << ",\"normalized_error\":[" << score.normalized_error[0] << ','
              << score.normalized_error[1] << ',' << score.normalized_error[2]
              << ']';
}

void JsonOptionalScore(std::optional<double> value) {
    if (value)
        JsonScore(*value);
    else
        std::cout << "null";
}

void JsonFrameStats(const FrameStats& stats) {
    std::cout << ",\"frame_stats\":{\"min\":";
    JsonScore(stats.min);
    std::cout << ",\"max\":";
    JsonScore(stats.max);
    std::cout << ",\"finite_mean\":";
    JsonOptionalScore(stats.finite_mean);
    std::cout << ",\"finite_stddev\":";
    JsonOptionalScore(stats.finite_stddev);
    std::cout << ",\"infinite_frames\":" << stats.infinite_frames
              << ",\"perfect_frames\":" << stats.perfect_frames << '}';
}

void PrintJson(
    Metric metric, const Pair& pair, const PairResult& clip,
    const Result& result, const std::optional<FrameStats>& stats
) {
    std::cout << "{\"sample_id\":";
    JsonString(pair.id);
    std::cout << ",\"metric\":";
    JsonString(MetricName(metric));
    std::cout << ",\"score\":";
    JsonScore(result.score);
    if (result.frames.size() == 1 && result.frames.front().psnr_hvs)
        JsonPsnrHvs(*result.frames.front().psnr_hvs);
    std::cout << ",\"frames\":" << clip.count
              << ",\"core_seconds\":" << result.core_seconds
              << ",\"decode_seconds\":" << clip.decode_seconds;
    if (stats) {
        std::cout << ",\"scoring_fps\":";
        JsonOptionalScore(clip.ScoringFps());
        JsonFrameStats(*stats);
    }
    std::cout << ",\"frame_results\":[";
    for (std::size_t i = 0; i < result.frames.size(); ++i) {
        if (i) std::cout << ',';
        const auto& frame = result.frames[i];
        std::cout << "{\"index\":" << frame.index << ",\"score\":";
        JsonScore(frame.score);
        if (!frame.features.empty()) JsonFeatures(frame.features);
        if (frame.psnr_hvs) JsonPsnrHvs(*frame.psnr_hvs);
        std::cout << '}';
    }
    std::cout << "]}\n" << std::flush;
}

void PrintRawFrame(const Frame& frame) {
    if (frame.psnr_hvs) {
        const auto& score = *frame.psnr_hvs;
        std::printf("%.12f,%.12f,%.12f\n", score.y, score.cb, score.cr);
        std::printf(
            "%.12f,%.12f,%.12f\n", score.normalized_error[0],
            score.normalized_error[1], score.normalized_error[2]
        );
    }
    if (frame.distance_p3) std::printf("%.8f\n", *frame.distance_p3);
    for (const auto value : frame.raw_norms) std::printf("%.12f,", value);
    if (!frame.raw_norms.empty()) std::printf("\n");
    for (const auto feature : frame.features) std::printf("%.12f,", feature);
    if (!frame.features.empty()) std::printf("\n");
}

void PrintText(
    const Options& options, Metric metric, const PairResult& clip,
    const Result& result
) {
    if (options.metrics.size() > 1) {
        const auto name = MetricName(metric);
        std::printf("%.*s: ", static_cast<int>(name.size()), name.data());
    }
    if (options.raw) {
        if (metric == Metric::Cvvdp) {
            std::printf(
                "frames %llu\n", static_cast<unsigned long long>(clip.count)
            );
        } else {
            for (const auto& frame : result.frames) PrintRawFrame(frame);
        }
    }
    std::printf("%.8f\n", result.score);
}

constexpr auto kVerboseRow = "{:<12} {:>15} {:>15} {:>15} {:>15} {:>8}\n";

void PrintVerbose(Metric metric, double score, const FrameStats& stats) {
    const auto format = [](std::optional<double> value) {
        return value ? std::format("{:.8g}", *value) : "n/a";
    };
    std::cout << std::format(
        kVerboseRow, MetricName(metric), format(score), format(stats.min),
        format(stats.max), format(stats.finite_stddev), stats.perfect_frames
    );
}

}  // namespace

std::vector<double> RawNorms(const Ssimulacra2Result& result) {
    std::vector<double> values;
    for (std::size_t channel = 0; channel < 3; ++channel)
        for (const auto& scale : result.scales)
            for (std::size_t norm = 0; norm < 2; ++norm) {
                values.push_back(scale.avg_ssim[channel * 2 + norm]);
                values.push_back(scale.avg_edgediff[channel * 4 + norm]);
                values.push_back(scale.avg_edgediff[channel * 4 + 2 + norm]);
            }
    return values;
}

// The reference tool's tuning build prints the 108 norms, channel-major.
void PrintRawNorms(const Ssimulacra2Result& msssim) {
    for (std::size_t c = 0; c < 3; ++c) {
        for (const auto& scale : msssim.scales) {
            for (std::size_t n = 0; n < 2; ++n) {
                std::printf(
                    "%.12f,%.12f,%.12f,", scale.avg_ssim[c * 2 + n],
                    scale.avg_edgediff[c * 4 + n],
                    scale.avg_edgediff[c * 4 + 2 + n]
                );
            }
        }
    }
    std::printf("\n");
}

void PrintClip(
    const Options& options, const Pair& pair, const PairResult& clip
) {
    if (options.verbose && !options.json) {
        const auto fps = clip.ScoringFps();
        std::cout << "Frames: " << clip.count << '\n'
                  << "Scoring FPS: "
                  << (fps ? std::format("{:.2f}", *fps) : "n/a") << '\n'
                  << std::format(
                         kVerboseRow, "Metric", "Aggregate", "Frame min",
                         "Frame max", "Finite stddev", "Perfect"
                     );
    }
    for (std::size_t i = 0; i < clip.metrics.size(); ++i) {
        const auto metric = options.metrics[i];
        const auto& result = clip.metrics[i];
        const auto stats =
            options.verbose
                ? std::optional(SummarizeFrames(
                      PerfectFrameScore(metric, options), result.frames
                  ))
                : std::nullopt;
        if (options.json)
            PrintJson(metric, pair, clip, result, stats);
        else if (stats)
            PrintVerbose(metric, result.score, *stats);
        else
            PrintText(options, metric, clip, result);
    }
}

}  // namespace vqm::cli
