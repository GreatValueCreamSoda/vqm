// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "bench.h"
#include "input.h"
#include "vmaf.h"
#include "vmaf_v1.h"

namespace bench_driver {

// A VMAF call advances an ordered sequence. Both images are prepared on
// every Push, even in resident mode; the previous blurred reference remains
// resident for motion. The final delayed frame is emitted by Flush.

template <typename Config>
Config VmafBenchmarkConfig(
    const BenchOptions& options, const vqm::Image& original
) {
    Config config;
    config.size = original.size;
    if constexpr (std::is_same_v<Config, vqm::VmafV1Config>)
        config.input_format = original.format;
    config.chroma_sampling =
        options.vmaf_chroma == "420"   ? vqm::VmafChromaSampling::Yuv420
        : options.vmaf_chroma == "444" ? vqm::VmafChromaSampling::Yuv444
                                       : vqm::VmafChromaSampling::Native;
    config.fit = options.vmaf_fit == "pad" ? vqm::VmafFitMode::FitAndPad
                                           : vqm::VmafFitMode::Native;
    if constexpr (std::is_same_v<Config, vqm::VmafConfig>) {
        config.model = options.vmaf_model == "4k" ? vqm::VmafModel::Standard4K
                                                  : vqm::VmafModel::Standard1080;
        config.neg = options.vmaf_neg;
        config.transform = options.vmaf_transform;
        config.temporal_weighting = options.vmaf_temporal;
    }
    config.streaming_inputs = options.resident == 0;
    config.gpu_resident_images = static_cast<std::size_t>(options.resident);
    return config;
}

// Score-only batches reuse one pipeline while geometry and native v1 coding
// stay compatible. Every pair remains its own sequence and starts after Reset.
template <typename Pipeline>
class VmafBatchPipeline {
public:
    template <typename Config>
    Pipeline& Prepare(const Config& config, const vqm::ImageFormat& format) {
        constexpr bool v1 = std::is_same_v<Config, vqm::VmafV1Config>;
        if (!pipeline_ || size_ != config.size ||
            (v1 && !vqm::internal::SameImageFormat(format_, format))) {
            pipeline_.emplace(config);
            size_ = config.size;
            format_ = format;
        } else {
            pipeline_->Reset();
        }
        return *pipeline_;
    }

private:
    std::optional<Pipeline> pipeline_;
    vqm::Size size_{};
    vqm::ImageFormat format_{};
};

struct VmafTotals {
    std::array<double, vqm::internal::kStageCount> stages{};
    double checksum = 0.0;
    double seconds = 0.0;
};

struct VmafPairMeasurement {
    double core_ms;
    double score;
};

// Upload outside the timer, advance the ordered sequence for warm-up and
// measured calls, then flush its delayed frame outside the timed interval.
template <typename Pipeline>
VmafPairMeasurement MeasureVmafSequence(
    Pipeline& pipeline, const BenchOptions& options, const vqm::Image& original,
    const vqm::Image& distorted, int first, int last, VmafTotals& totals
) {
    vqm::UploadedImage ref, dist;
    if (options.resident) {
        ref = pipeline.Upload(original);
        dist = pipeline.Upload(distorted);
    }
    std::vector<double> times;
    double score = 0.0;
    const int warmup = options.score_only ? 0 : options.warmup;
    const int iterations = options.score_only ? 1 : options.iterations;
    for (int i = 0; i < warmup + iterations; ++i) {
        const bool measured = i >= warmup;
        if (measured) vqm::internal::ResetTimings();
        const auto start = Clock::now();
        const auto result = options.resident
                                ? pipeline.Push(ref, dist)
                                : pipeline.Push(original, distorted);
        const double seconds = Seconds(Clock::now() - start);
        if (result) {
            score = result->score;
            totals.checksum += score;
        }
        if (measured) {
            times.push_back(seconds);
            totals.seconds += seconds;
            for (int stage = first; stage <= last; ++stage) {
                totals.stages[static_cast<std::size_t>(stage)] +=
                    vqm::internal::Timings()
                        .seconds[static_cast<std::size_t>(stage)];
            }
        }
    }
    if (const auto result = pipeline.Flush()) {
        score = result->score;
        totals.checksum += score;
    }
    if (options.resident) {
        pipeline.Release(ref);
        pipeline.Release(dist);
    }
    const double core_ms = 1000.0 * Median(times);
    return {core_ms, score};
}

template <typename Pipeline, typename Config>
int RunVmaf(const BenchOptions& options) {
    constexpr bool v1 = std::is_same_v<Config, vqm::VmafV1Config>;
    constexpr int first = static_cast<int>(
        v1 ? vqm::internal::Stage::VmafV1Luma : vqm::internal::Stage::VmafLuma
    );
    constexpr int last = static_cast<int>(
        v1 ? vqm::internal::Stage::VmafV1Score : vqm::internal::Stage::VmafScore
    );
    VmafTotals totals;
    std::vector<Row> rows;
    std::vector<double> cores;
    int failures = 0;
    VmafBatchPipeline<Pipeline> batch_pipeline;
    for (const Pair& pair : options.pairs) {
        vqm::Image original, distorted;
        std::string error;
        double decode_ms = 0.0;
        if (!MeasureDecode(pair, 1, &decode_ms, &original, &distorted, &error)) {
            std::fprintf(
                stderr, "%s: %s\n", pair.original.c_str(), error.c_str()
            );
            if (options.score_only) std::puts("ERROR");
            ++failures;
            continue;
        }
        try {
            ApplyOverrides(options, &original, &distorted);
            const auto config = VmafBenchmarkConfig<Config>(options, original);
            std::optional<Pipeline> own_pipeline;
            Pipeline& pipeline =
                options.score_only && !options.resident
                    ? batch_pipeline.Prepare(config, original.format)
                    : own_pipeline.emplace(config);
            const auto measured = MeasureVmafSequence(
                pipeline, options, original, distorted, first, last, totals
            );
            const double core_ms = measured.core_ms;
            const double score = measured.score;
            rows.push_back(
                {pair, static_cast<int>(original.size.width),
                 static_cast<int>(original.size.height), decode_ms, core_ms,
                 score}
            );
            cores.push_back(core_ms);
            if (options.score_only) {
                std::printf("%.8f\n", score);
            } else {
                std::printf(
                    "%s vs %s: decode %.3f ms, core %.6f ms, score %.8f\n",
                    pair.original.c_str(), pair.distorted.c_str(), decode_ms,
                    core_ms, score
                );
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s: %s\n", pair.original.c_str(), e.what());
            if (options.score_only) std::puts("ERROR");
            ++failures;
        }
    }
    if (rows.empty()) return 1;
    if (!options.score_only) {
        const double median_core = Median(cores);
        if (options.resident)
            std::printf(
                "resident: %.1f pairs/s, %.4f ms/pair\n", 1000.0 / median_core,
                median_core
            );
        std::printf(
            "stage timing over %d iterations (%zu pairs):\n", options.iterations,
            rows.size()
        );
        PrintStages(
            totals.stages,
            static_cast<double>(options.iterations) *
                static_cast<double>(rows.size()),
            totals.seconds, first, last, false
        );
        std::printf("checksum %.6f\n", totals.checksum);
    }
    if (!options.csv_path.empty() && !WritePairsCsv(options.csv_path, rows))
        return 1;
    if (!options.stages_csv_path.empty() &&
        !WriteStagesCsv(
            options.stages_csv_path, totals.stages,
            static_cast<double>(options.iterations) *
                static_cast<double>(rows.size()),
            totals.seconds, first, last, false
        ))
        return 1;
    return failures ? 1 : 0;
}

int RunVmafForMetric(const BenchOptions& options) {
    return (options.metric == Metric::VmafV1)
               ? RunVmaf<vqm::VmafV1Pipeline, vqm::VmafV1Config>(options)
               : RunVmaf<vqm::VmafPipeline, vqm::VmafConfig>(options);
}

}  // namespace bench_driver
