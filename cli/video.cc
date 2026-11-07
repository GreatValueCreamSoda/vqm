// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#include "video.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <format>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "butteraugli.h"
#include "cvvdp.h"
#include "metrics/cvvdp/cvvdp_internal.h"
#include "psnr_hvs.h"
#include "vmaf.h"
#include "vmaf_v1.h"
#include "vqm.h"
#include "vqm_alpha.h"
#include "vqm_vulkan.h"

#ifdef VQM_CLI_NVDEC
#include "nvdec.h"
#endif

namespace vqm::cli {
namespace {
using Clock = std::chrono::steady_clock;
using image_io::SequenceReader;
using image_io::SequenceReadResult;

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

struct FrameStats {
    double min = 0, max = 0;
    std::optional<double> finite_mean, finite_stddev;
    std::uint64_t perfect_frames = 0;
    std::uint64_t infinite_frames = 0;
};

double PerfectFrameScore(Metric metric) {
    switch (metric) {
        case Metric::Ssimulacra2:
        case Metric::VmafV0:
        case Metric::VmafV1:
            return 100.0;
        case Metric::Butteraugli:
            return 0.0;
        case Metric::Cvvdp:
            return 10.0;
        case Metric::PsnrHvs:
            return std::numeric_limits<double>::infinity();
    }
    throw std::logic_error("unknown metric");
}

FrameStats SummarizeFrames(Metric metric, std::span<const Frame> frames) {
    FrameStats stats;
    stats.min = stats.max = frames.front().score;
    const auto perfect_score = PerfectFrameScore(metric);
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

double Seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::vector<Pair> ReadPairs(const Options& options) {
    if (options.pairs_file.empty())
        return {{"pair", options.original, options.distorted}};
    std::ifstream input(options.pairs_file);
    if (!input)
        throw std::runtime_error(
            "cannot open pairs file: " + options.pairs_file
        );
    std::vector<Pair> pairs;
    std::unordered_set<std::string> ids;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream row(line);
        Pair pair;
        if (!std::getline(row, pair.id, '\t') ||
            !std::getline(row, pair.original, '\t') ||
            !std::getline(row, pair.distorted) || pair.id.empty() ||
            pair.original.empty() || pair.distorted.empty() ||
            pair.distorted.find('\t') != std::string::npos)
            throw std::invalid_argument("expected ID, original, distorted TSV");
        if (!ids.insert(pair.id).second)
            throw std::invalid_argument("duplicate sample ID: " + pair.id);
        pairs.push_back(std::move(pair));
    }
    if (!input.eof()) throw std::runtime_error("could not read pairs file");
    if (pairs.empty()) throw std::invalid_argument("empty pair list");
    return pairs;
}

template <typename Reader>
void CheckReaders(const Reader& original, const Reader& distorted) {
    if (original.size() != distorted.size())
        throw std::invalid_argument("Image size mismatch");
    if (original.is_video() != distorted.is_video())
        throw std::invalid_argument(
            "inputs require two images or two Y4M sequences"
        );
    if (original.frame_rate() != distorted.frame_rate())
        throw std::invalid_argument("inputs have unequal frame rates");
}

auto FormatKey(const ImageFormat& format) {
    return std::tie(
        format.type, format.family, format.alpha, format.range, format.matrix,
        format.transfer, format.primaries, format.chroma_location,
        format.subsample_w, format.subsample_h, format.depth
    );
}

struct InputKey {
    Size size;
    ImageFormat original, distorted;
    image_io::FrameRate fps;
    bool video;

    bool operator==(const InputKey& other) const {
        return size == other.size &&
               FormatKey(original) == FormatKey(other.original) &&
               FormatKey(distorted) == FormatKey(other.distorted) &&
               fps == other.fps && video == other.video;
    }
};

template <typename Reader>
InputKey Key(
    const Options& options, const Reader& original, const Reader& distorted
) {
    InputKey key{original.size(), original.format(), distorted.format(),
                 original.frame_rate(), original.is_video()};
    options.input.ApplyTo(key.original);
    options.input.ApplyTo(key.distorted);
    return key;
}

void Add(Result& result, const std::optional<VmafFrame>& frame) {
    if (!frame) return;
    result.frames.push_back(
        {frame->index,
         frame->score,
         {frame->adm2, frame->motion2, frame->vif[0], frame->vif[1],
          frame->vif[2], frame->vif[3]},
         {},
         {},
         {}}
    );
}

void Add(Result& result, const std::optional<VmafV1Frame>& frame) {
    if (!frame) return;
    result.frames.push_back(
        {frame->index,
         frame->score,
         {frame->cambi, frame->speed_chroma_uv, frame->adm3, frame->motion3},
         {},
         {},
         {}}
    );
}

void AddCvvdpFrame(Result& result, double score) {
    result.frames.push_back({result.frames.size(), score, {}, {}, {}, {}});
}

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

double PoolPsnrHvs(std::span<const Frame> frames) {
    if (frames.size() == 1) return frames.front().score;

    double sum = 0.0;
    for (const auto& frame : frames) {
        const auto& [y, cb, cr] = frame.psnr_hvs->normalized_error;
        sum += 0.8 * y + 0.1 * (cb + cr);
    }
    if (sum == 0.0) return std::numeric_limits<double>::infinity();
    return -10.0 * std::log10(sum / static_cast<double>(frames.size()));
}

template <typename Pipeline, typename Config>
std::unique_ptr<Pipeline> MakePipeline(
    Config config, const VulkanDevice* device
) {
    config.gpu_resident_images = 0;
    config.streaming_inputs = device == nullptr;
    if (device) return std::make_unique<Pipeline>(config, *device);
    return std::make_unique<Pipeline>(config);
}

VmafFitMode VmafFit(const Options& options) {
    if (options.vmaf_scale)
        return options.vmaf_pad ? VmafFitMode::FitAndPad : VmafFitMode::Fit;
    return options.vmaf_pad ? VmafFitMode::Pad : VmafFitMode::Native;
}

// One geometry, format domain and frame rate per cached pipeline. Push consumes
// the current sources synchronously and retains its own temporal history.
class Scorer {
public:
    Scorer(
        const Options& options, Metric metric, const InputKey& key,
        const VulkanDevice* device
    )
        : raw_(options.raw), verbose_(options.verbose), video_(key.video) {
        if (metric == Metric::Ssimulacra2) {
            Ssimulacra2Config config{.size = key.size};
            config.half_planes = options.half_planes;
            config.chroma_filter = options.input.chroma_filter;
            ssimu_ = MakePipeline<Ssimulacra2Pipeline>(config, device);
        } else if (metric == Metric::Butteraugli) {
            ButteraugliConfig config{.size = key.size};
            config.chroma_filter = options.input.chroma_filter;
            config.hf_asymmetry = options.hf_asymmetry;
            config.intensity_target = options.intensity_target;
            butter_ = MakePipeline<ButteraugliPipeline>(config, device);
        } else if (metric == Metric::VmafV0) {
            VmafConfig config{.size = key.size};
            config.fit = VmafFit(options);
            v0_ = MakePipeline<VmafPipeline>(config, device);
        } else if (metric == Metric::VmafV1) {
            VmafV1Config config{.size = key.size, .input_format = key.original};
            config.fit = VmafFit(options);
            v1_ = MakePipeline<VmafV1Pipeline>(config, device);
        } else if (metric == Metric::PsnrHvs) {
            PsnrHvsConfig config{.size = key.size};
            psnr_hvs_ = MakePipeline<PsnrHvsPipeline>(config, device);
        } else {
            CvvdpConfig config{.size = key.size};
            config.display = options.cvvdp.ResolveDisplay();
            config.temporal = key.video;
            config.chroma_filter = options.input.chroma_filter;
            if (key.video) {
                config.fps_numerator = key.fps.numerator;
                config.fps_denominator = key.fps.denominator;
            }
            cvvdp_ = MakePipeline<CvvdpPipeline>(config, device);
            if (verbose_ && video_) internal::CvvdpFrameAccess::Enable(*cvvdp_);
        }
    }

    void Reset() {
        if (v0_) v0_->Reset();
        if (v1_) v1_->Reset();
        if (cvvdp_) cvvdp_->Reset();
    }

    template <typename Input>
    void Push(
        Result& result, std::uint64_t index, const Input& original,
        const Input& distorted
    ) {
        if (ssimu_) {
            const auto value =
                ComputeWorstBackground(*ssimu_, original, distorted);
            Frame frame{index, value.Score(), {}, {}, {}, {}};
            if (raw_) frame.raw_norms = RawNorms(value);
            result.frames.push_back(std::move(frame));
        } else if (butter_) {
            const auto value =
                ComputeWorstBackground(*butter_, original, distorted);
            result.frames.push_back(
                {index, value.score, {}, {}, value.distance_p3, {}}
            );
        } else if (v0_) {
            Add(result, v0_->Push(original, distorted));
        } else if (v1_) {
            Add(result, v1_->Push(original, distorted));
        } else if (psnr_hvs_) {
            const auto value = psnr_hvs_->Compute(original, distorted);
            result.frames.push_back({index, value.score, {}, {}, {}, value});
        } else if (video_) {
            cvvdp_->Push(original, distorted);
            if (verbose_)
                for (double score : internal::CvvdpFrameAccess::Scores(*cvvdp_))
                    AddCvvdpFrame(result, score);
        } else {
            result.score = cvvdp_->ComputeImage(original, distorted).score;
        }
    }

    void Finish(Result& result, std::uint64_t count) {
        if (cvvdp_) {
            if (video_) {
                const auto score = cvvdp_->Finalize();
                if (!score || score->frames != count)
                    throw std::runtime_error("empty or incomplete CVVDP result");
                result.score = score->score;
            }
            if (verbose_ && count == 1) AddCvvdpFrame(result, result.score);
            if (verbose_ && result.frames.size() != count)
                throw std::runtime_error("incomplete emitted CVVDP frames");
            return;
        }
        if (v0_) Add(result, v0_->Flush());
        if (v1_) Add(result, v1_->Flush());
        if (result.frames.size() != count)
            throw std::runtime_error("incomplete emitted frame results");
        if (v0_ || v1_) {
            const auto pooled = v0_ ? v0_->PooledScore() : v1_->PooledScore();
            if (!pooled) throw std::runtime_error("empty pooled VMAF result");
            result.score = *pooled;
            return;
        }
        if (psnr_hvs_) {
            result.score = PoolPsnrHvs(result.frames);
            return;
        }
        double sum = 0;
        for (const auto& frame : result.frames) sum += frame.score;
        result.score = sum / static_cast<double>(count);
    }

private:
    bool raw_;
    bool verbose_;
    bool video_;
    std::unique_ptr<Ssimulacra2Pipeline> ssimu_;
    std::unique_ptr<ButteraugliPipeline> butter_;
    std::unique_ptr<VmafPipeline> v0_;
    std::unique_ptr<VmafV1Pipeline> v1_;
    std::unique_ptr<CvvdpPipeline> cvvdp_;
    std::unique_ptr<PsnrHvsPipeline> psnr_hvs_;
};

class ScorerCache {
public:
    explicit ScorerCache(const Options& options) : options_(options) {}

    std::vector<Scorer> Take(const InputKey& key, const VulkanDevice* device) {
        auto scorers = std::exchange(scorers_, {});
        if (scorers.empty() || key_ != key || device_ != device) {
            scorers.clear();
            scorers.reserve(options_.metrics.size());
            for (const auto metric : options_.metrics)
                scorers.emplace_back(options_, metric, key, device);
            key_ = key;
            device_ = device;
        }
        return scorers;
    }

    void Store(std::vector<Scorer> scorers) { scorers_ = std::move(scorers); }

private:
    const Options& options_;
    const VulkanDevice* device_ = nullptr;
    std::optional<InputKey> key_;
    std::vector<Scorer> scorers_;
};

bool ValidScore(Metric metric, double value) {
    return std::isfinite(value) ||
           (metric == Metric::PsnrHvs && std::isinf(value) && value > 0.0);
}

bool ValidPsnrHvs(const PsnrHvsScore& value) {
    return ValidScore(Metric::PsnrHvs, value.y) &&
           ValidScore(Metric::PsnrHvs, value.cb) &&
           ValidScore(Metric::PsnrHvs, value.cr) &&
           std::all_of(
               value.normalized_error.begin(), value.normalized_error.end(),
               [](double error) { return std::isfinite(error) && error >= 0.0; }
           );
}

void ValidateResult(Metric metric, const Result& result) {
    if (!ValidScore(metric, result.score))
        throw std::runtime_error("non-finite clip score");
    for (std::size_t i = 0; i < result.frames.size(); ++i) {
        const auto& frame = result.frames[i];
        if (frame.index != i || !ValidScore(metric, frame.score) ||
            (frame.psnr_hvs && !ValidPsnrHvs(*frame.psnr_hvs)) ||
            !std::all_of(
                frame.features.begin(), frame.features.end(),
                [](double value) { return std::isfinite(value); }
            ))
            throw std::runtime_error(
                "incorrect or non-finite emitted frame result"
            );
    }
}

template <typename Reader, typename Input>
SequenceReadResult ReadPair(
    const Pair& pair, Reader& original, Reader& distorted, Input& ref,
    Input& test
) {
    std::string error;
    const auto status = original.Next(ref, error);
    if (status == SequenceReadResult::Error)
        throw std::runtime_error(pair.original + ": " + error);
    const auto test_status = distorted.Next(test, error);
    if (test_status == SequenceReadResult::Error)
        throw std::runtime_error(pair.distorted + ": " + error);
    if (status != test_status)
        throw std::invalid_argument("inputs have unequal frame counts");
    return status;
}

template <typename Input, typename OpenReader>
PairResult Score(
    const Options& options, const Pair& pair, ScorerCache& cache,
    const OpenReader& open_reader, const VulkanDevice* device
) {
    const auto open = Clock::now();
    auto original = open_reader(pair.original);
    auto distorted = open_reader(pair.distorted);
    CheckReaders(original, distorted);
    PairResult result;
    result.decode_seconds = Seconds(open);
    // Destroy failed scorers before the readers release borrowed frames.
    auto scorers = cache.Take(Key(options, original, distorted), device);
    result.metrics.resize(scorers.size());
    for (auto& scorer : scorers) scorer.Reset();
    Input ref, test;
    for (;;) {
        const auto decode = Clock::now();
        const auto status = ReadPair(pair, original, distorted, ref, test);
        result.decode_seconds += Seconds(decode);
        if (status == SequenceReadResult::End) break;
        options.input.ApplyTo(ref.format);
        options.input.ApplyTo(test.format);
        // NVDEC readers may reuse their buffers on the next read.
        for (std::size_t i = 0; i < scorers.size(); ++i) {
            const auto core = Clock::now();
            scorers[i].Push(result.metrics[i], result.count, ref, test);
            result.metrics[i].core_seconds += Seconds(core);
        }
        ++result.count;
    }
    if (result.count == 0)
        throw std::invalid_argument("input sequence has no frames");
    for (std::size_t i = 0; i < scorers.size(); ++i) {
        auto& report = result.metrics[i];
        const auto finish = Clock::now();
        scorers[i].Finish(report, result.count);
        report.core_seconds += Seconds(finish);
        result.core_seconds += report.core_seconds;
        ValidateResult(options.metrics[i], report);
    }
    cache.Store(std::move(scorers));
    return result;
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

bool IsY4m(std::string_view path) {
    std::ifstream file(std::string(path), std::ios::binary);
    std::array<char, 9> magic{};
    return file.read(magic.data(), magic.size()) &&
           std::string_view(magic.data(), magic.size()) == "YUV4MPEG2";
}

template <typename Input, typename OpenReader>
void RunPair(
    const Options& options, const Pair& pair, ScorerCache& cache,
    const OpenReader& open_reader, const VulkanDevice* device = nullptr
) {
    for (unsigned i = 0; i < options.warmup; ++i)
        Score<Input>(options, pair, cache, open_reader, device);
    std::vector<PairResult> measured;
    for (unsigned i = 0; i < options.iterations; ++i)
        measured.push_back(
            Score<Input>(options, pair, cache, open_reader, device)
        );
    std::sort(
        measured.begin(), measured.end(), [](const auto& a, const auto& b) {
            return a.core_seconds < b.core_seconds;
        }
    );
    const auto& clip = measured[measured.size() / 2];
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
                ? std::optional(SummarizeFrames(metric, result.frames))
                : std::nullopt;
        if (options.json)
            PrintJson(metric, pair, clip, result, stats);
        else if (stats)
            PrintVerbose(metric, result.score, *stats);
        else
            PrintText(options, metric, clip, result);
    }
}

#ifdef VQM_CLI_NVDEC
bool UseNvdec(const Options& options, const Pair& pair) {
    const bool original = IsCompressedVideo(pair.original);
    const bool distorted = pair.original == pair.distorted
                               ? original
                               : IsCompressedVideo(pair.distorted);
    if (original != distorted || (options.nvdec && !original))
        throw std::invalid_argument(
            "NVDEC requires two compressed video inputs"
        );
    return original;
}

void RunNvdecPair(
    const Options& options, const Pair& pair, ScorerCache& cache,
    NvdecDevice& device
) {
    RunPair<VulkanBufferView>(
        options, pair, cache,
        [&device](const std::string& path) { return NvdecReader(path, device); },
        &device.vulkan()
    );
}
#endif

}  // namespace

bool HasY4mInput(const Options& options) {
    return IsY4m(options.original) || IsY4m(options.distorted);
}

bool TryRunNvdecPair(const Options& options) {
#ifdef VQM_CLI_NVDEC
    const Pair pair{"pair", options.original, options.distorted};
    if (!UseNvdec(options, pair)) return false;
    NvdecDevice device;
    ScorerCache cache(options);
    RunNvdecPair(options, pair, cache, device);
    return true;
#else
    (void)options;
    return false;
#endif
}

int RunSequences(const Options& options) {
#ifdef VQM_CLI_NVDEC
    std::optional<NvdecDevice> nvdec;
#else
    if (options.nvdec)
        throw std::runtime_error(
            "--nvdec requires a CLI built with -DVQM_ENABLE_NVDEC=ON"
        );
#endif
    const auto pairs = ReadPairs(options);
    ScorerCache cache(options);
    std::cout << std::setprecision(17);
    for (const auto& pair : pairs) {
#ifdef VQM_CLI_NVDEC
        if (UseNvdec(options, pair)) {
            if (!nvdec) nvdec.emplace();
            RunNvdecPair(options, pair, cache, *nvdec);
            continue;
        }
#endif
        RunPair<Image>(options, pair, cache, [](const std::string& path) {
            return SequenceReader(path);
        });
    }
    return 0;
}

}  // namespace vqm::cli
