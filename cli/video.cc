// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#include "video.h"

#include "report.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
#include <variant>
#include <type_traits>

#include "butteraugli.h"
#include "cvvdp.h"
#include "metrics/cvvdp/cvvdp_internal.h"
#include "psnr_hvs.h"
#include "vmaf.h"
#include "vmaf_v1.h"
#include "vqm.h"
#include "vqm_alpha.h"
#include "vqm_vulkan.h"

#ifdef VQM_CLI_BESTSOURCE
#include "bestsource.h"
#endif

namespace vqm::cli {
namespace {
using Clock = std::chrono::steady_clock;
using image_io::SequenceReader;
using image_io::SequenceReadResult;

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

VmafFitMode VmafFit(const Options& options) {
    if (options.vmaf_scale)
        return options.vmaf_pad ? VmafFitMode::FitAndPad : VmafFitMode::Fit;
    return options.vmaf_pad ? VmafFitMode::Pad : VmafFitMode::Native;
}

using ScorerConfig = std::variant<
    Ssimulacra2Config, ButteraugliConfig, VmafConfig, VmafV1Config, CvvdpConfig,
    PsnrHvsConfig>;

ScorerConfig MakeScorerConfig(
    const Options& options, Metric metric, const InputKey& key
) {
    if (metric == Metric::Ssimulacra2)
        return MakeSsimulacra2Config(options, key.size);
    if (metric == Metric::Butteraugli)
        return MakeButteraugliConfig(options, key.size);
    if (metric == Metric::VmafV0) {
        VmafConfig config{.size = key.size};
        config.model = options.vmaf_v0_model.value_or(VmafModel::Standard1080);
        config.fit = VmafFit(options);
        config.neg = options.vmaf_v0_neg;
        config.transform = options.vmaf_v0_transform;
        return config;
    }
    if (metric == Metric::VmafV1) {
        VmafV1Config config{.size = key.size, .input_format = key.original};
        config.fit = VmafFit(options);
        config.hfr = options.vmaf_v1_hfr;
        config.model = options.vmaf_v1_model.value_or(VmafV1Model::Standard1080);
        return config;
    }
    if (metric == Metric::PsnrHvs) return PsnrHvsConfig{.size = key.size};
    CvvdpConfig config{.size = key.size};
    config.display = options.cvvdp.ResolveDisplay(key.size);
    config.temporal = key.video;
    config.chroma_filter = options.input.chroma_filter;
    if (key.video) {
        config.fps_numerator = key.fps.numerator;
        config.fps_denominator = key.fps.denominator;
    }
    return config;
}

MetricsConfig MakePipelineConfig(
    const Options& options, const InputKey& key, bool streaming
) {
    MetricsConfig config{.size = key.size,
                         .original_format = key.original,
                         .distorted_format = key.distorted,
                         .streaming_inputs = streaming};
    for (const auto metric : options.metrics) {
        std::visit(
            [&](const auto& value) {
                using Config = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<Config, Ssimulacra2Config>)
                    config.ssimulacra2 = value;
                else if constexpr (std::is_same_v<Config, ButteraugliConfig>)
                    config.butteraugli = value;
                else if constexpr (std::is_same_v<Config, VmafConfig>)
                    config.vmaf_v0 = value;
                else if constexpr (std::is_same_v<Config, VmafV1Config>)
                    config.vmaf_v1 = value;
                else if constexpr (std::is_same_v<Config, CvvdpConfig>)
                    config.cvvdp = value;
                else if constexpr (std::is_same_v<Config, PsnrHvsConfig>)
                    config.psnr_hvs = value;
            },
            MakeScorerConfig(options, metric, key)
        );
    }
    return config;
}

// One geometry, format domain and frame rate per cached pipeline. Push consumes
// the current sources synchronously and retains its own temporal history.
class Scorer {
public:
    Scorer(
        const Options& options, const InputKey& key, Metric metric,
        std::shared_ptr<MetricPipelines> pipelines
    )
        : pipelines_(std::move(pipelines)),
          raw_(options.raw),
          verbose_(options.verbose),
          video_(key.video),
          ssimu_(
              metric == Metric::Ssimulacra2 ? pipelines_->Ssimulacra2() : nullptr
          ),
          butter_(
              metric == Metric::Butteraugli ? pipelines_->Butteraugli() : nullptr
          ),
          v0_(metric == Metric::VmafV0 ? pipelines_->VmafV0() : nullptr),
          v1_(metric == Metric::VmafV1 ? pipelines_->VmafV1() : nullptr),
          cvvdp_(metric == Metric::Cvvdp ? pipelines_->Cvvdp() : nullptr),
          psnr_hvs_(
              metric == Metric::PsnrHvs ? pipelines_->PsnrHvs() : nullptr
          ) {
        if (cvvdp_ && verbose_ && video_)
            internal::CvvdpFrameAccess::Enable(*cvvdp_);
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
        if (v1_)
            while (const auto frame = v1_->Flush()) Add(result, frame);
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
    std::shared_ptr<MetricPipelines> pipelines_;
    bool raw_;
    bool verbose_;
    bool video_;
    Ssimulacra2Pipeline* ssimu_;
    ButteraugliPipeline* butter_;
    VmafPipeline* v0_;
    VmafV1Pipeline* v1_;
    CvvdpPipeline* cvvdp_;
    PsnrHvsPipeline* psnr_hvs_;
};

class ScorerCache {
public:
    explicit ScorerCache(const Options& options) : options_(options) {}

    std::vector<Scorer> Take(const InputKey& key, const VulkanDevice* device) {
        auto scorers = std::exchange(scorers_, {});
        if (scorers.empty() || key_ != key || device_ != device) {
            scorers.clear();
            scorers.reserve(options_.metrics.size());
            const auto config =
                MakePipelineConfig(options_, key, device == nullptr);
            const auto pipelines =
                device ? std::make_shared<MetricPipelines>(config, *device)
                       : std::make_shared<MetricPipelines>(config);
            for (const auto metric : options_.metrics)
                scorers.emplace_back(options_, key, metric, pipelines);
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
        // GPU readers may reuse their buffers on the next read.
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
    PrintClip(options, pair, clip);
}

#ifdef VQM_CLI_BESTSOURCE
struct VideoTracks {
    int original;
    int distorted;
};

std::optional<VideoTracks> BestSourceTracks(const Pair& pair) {
    const auto original = FindBestSourceTrack(pair.original);
    const auto distorted = pair.original == pair.distorted
                               ? original
                               : FindBestSourceTrack(pair.distorted);
    if (original.has_value() != distorted.has_value())
        throw std::invalid_argument(
            "BestSource requires two compressed video inputs"
        );
    if (!original) return std::nullopt;
    return VideoTracks{*original, *distorted};
}

void RunBestSourcePair(
    const Options& options, const Pair& pair, VideoTracks tracks,
    ScorerCache& cache, BestSourceDevice& device
) {
    RunPair<VulkanBufferView>(
        options, pair, cache,
        [&](const std::string& path) {
            const int track =
                path == pair.original ? tracks.original : tracks.distorted;
            return BestSourceReader(
                path, track, device, options.start_frame.value_or(0),
                options.frame_count
            );
        },
        &device.vulkan()
    );
}
#endif

}  // namespace

bool HasY4mInput(const Options& options) {
    return IsY4m(options.original) || IsY4m(options.distorted);
}

bool TryRunBestSourcePair(
    const Options& options, const std::optional<DeviceUuid>& device_uuid
) {
#ifdef VQM_CLI_BESTSOURCE
    const Pair pair{"pair", options.original, options.distorted};
    const auto tracks = BestSourceTracks(pair);
    if (!tracks) return false;
    BestSourceDevice device(device_uuid);
    ScorerCache cache(options);
    RunBestSourcePair(options, pair, *tracks, cache, device);
    return true;
#else
    (void)options;
    (void)device_uuid;
    return false;
#endif
}

int RunSequences(
    const Options& options, const std::optional<DeviceUuid>& device_uuid
) {
#ifdef VQM_CLI_BESTSOURCE
    std::optional<BestSourceDevice> device;
#else
    (void)device_uuid;
    if (options.start_frame || options.frame_count)
        throw std::invalid_argument(
            "--start-frame and --frame-count require -DVQM_ENABLE_BESTSOURCE=ON"
        );
#endif
    const auto pairs = ReadPairs(options);
    ScorerCache cache(options);
    std::cout << std::setprecision(17);
    for (const auto& pair : pairs) {
#ifdef VQM_CLI_BESTSOURCE
        if (const auto tracks = BestSourceTracks(pair)) {
            if (!device) device.emplace(device_uuid);
            RunBestSourcePair(options, pair, *tracks, cache, *device);
            continue;
        }
        if (options.start_frame || options.frame_count)
            throw std::invalid_argument(
                "--start-frame and --frame-count require two compressed video "
                "inputs"
            );
#endif
        RunPair<Image>(options, pair, cache, [](const std::string& path) {
            return SequenceReader(path);
        });
    }
    return 0;
}

}  // namespace vqm::cli
