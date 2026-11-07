// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>

#include "bench.h"
#include "butteraugli.h"
#include "inmemory/vulkan_inputs.h"
#include "vqm_metrics.h"
#include "vqm_shared.h"

namespace bench_driver {

// Own the selected metric arrangement and its input dispatch. Construction
// stays outside measurements; Score always performs SSIMULACRA 2 then
// Butteraugli.
class MetricPairPipelines {
public:
    MetricPairPipelines(
        const BenchOptions& options, const DecodedPair& decoded,
        const std::optional<inmem::VulkanInputs>& borrowed, bool has_alpha
    )
        : pair(decoded), inputs(borrowed) {
        const bool streaming = !inputs.has_value();
        const auto size = pair.original.size;
        vqm::Ssimulacra2Config ssimu_config;
        ssimu_config.size = size;
        ssimu_config.gpu_resident_images = 0;
        ssimu_config.streaming_inputs = streaming;
        ssimu_config.half_planes = options.half_planes;
        ssimu_config.chroma_filter = options.input.chroma_filter;
        vqm::ButteraugliConfig butter_config;
        butter_config.size = size;
        butter_config.gpu_resident_images = 0;
        butter_config.streaming_inputs = streaming;
        butter_config.chroma_filter = options.input.chroma_filter;
        butter_config.hf_asymmetry = options.hf_asymmetry;
        butter_config.intensity_target = options.intensity_target;
        vqm::SharedMetricConfig shared_config;
        shared_config.size = size;
        shared_config.streaming_inputs = streaming;
        shared_config.half_planes = options.half_planes;
        shared_config.chroma_filter = options.input.chroma_filter;
        shared_config.hf_asymmetry = options.hf_asymmetry;
        shared_config.intensity_target = options.intensity_target;

        if (options.metric_pair == "unified") {
            vqm::MetricsConfig config;
            config.size = size;
            config.original_format = pair.original.format;
            config.distorted_format = pair.distorted.format;
            config.streaming_inputs = streaming;
            config.background = has_alpha ? 0.1f : 0.5f;
            config.ssimulacra2 = ssimu_config;
            config.butteraugli = butter_config;
            if (inputs)
                unified.emplace(config, inputs->device());
            else
                unified.emplace(config);
            if (has_alpha) {
                config.background = 0.9f;
                if (inputs)
                    unified_bright.emplace(config, inputs->device());
                else
                    unified_bright.emplace(config);
            }
        } else if (options.metric_pair == "shared") {
            if (inputs)
                shared.emplace(shared_config, inputs->device());
            else
                shared.emplace(shared_config);
        } else if (inputs) {
            ssimu.emplace(ssimu_config, inputs->device());
            butter.emplace(butter_config, inputs->device());
        } else {
            ssimu.emplace(ssimu_config);
            butter.emplace(butter_config);
        }
    }

    std::array<double, 2> Score(float background) {
        if (unified) {
            auto& metrics = background == 0.9f ? *unified_bright : *unified;
            const auto result =
                inputs ? metrics.Compute(inputs->original(), inputs->distorted())
                       : metrics.Compute(pair.original, pair.distorted);
            if (!result || !result->ssimulacra2 || !result->butteraugli)
                throw std::logic_error("unified metric pair was not ready");
            return {*result->ssimulacra2, *result->butteraugli};
        }
        if (shared) {
            if (inputs) {
                const double a =
                    shared
                        ->ComputeSsimulacra2(
                            inputs->original(), inputs->distorted(), background
                        )
                        .Score();
                const double b =
                    shared
                        ->ComputeButteraugli(
                            inputs->original(), inputs->distorted(), background
                        )
                        .score;
                return {a, b};
            }
            const double a = shared
                                 ->ComputeSsimulacra2(
                                     pair.original, pair.distorted, background
                                 )
                                 .Score();
            const double b = shared
                                 ->ComputeButteraugli(
                                     pair.original, pair.distorted, background
                                 )
                                 .score;
            return {a, b};
        }
        if (inputs) {
            const double a =
                ssimu
                    ->Compute(
                        inputs->original(), inputs->distorted(), background
                    )
                    .Score();
            const double b =
                butter
                    ->Compute(
                        inputs->original(), inputs->distorted(), background
                    )
                    .score;
            return {a, b};
        }
        const double a =
            ssimu->Compute(pair.original, pair.distorted, background).Score();
        const double b =
            butter->Compute(pair.original, pair.distorted, background).score;
        return {a, b};
    }

private:
    const DecodedPair& pair;
    const std::optional<inmem::VulkanInputs>& inputs;
    std::optional<vqm::SharedMetricWorkspace> shared;
    std::optional<vqm::Metrics> unified;
    std::optional<vqm::Metrics> unified_bright;
    std::optional<vqm::Ssimulacra2Pipeline> ssimu;
    std::optional<vqm::ButteraugliPipeline> butter;
};

// Fixed-count serial calls on the same image pair. Each iteration scores both
// metrics, so wall time includes shared-workspace activation and metadata
// republishing as well as both metric calls; construction and decode are out.
int RunMetricPair(const BenchOptions& options) {
    if (options.pairs.size() != 1) {
        std::fprintf(stderr, "--metric-pair requires exactly one pair\n");
        return 2;
    }
    try {
        DecodedPair pair;
        std::string error;
        if (!LoadOnce(options.pairs[0], pair, error))
            throw std::runtime_error(error);
        ApplyOverrides(options, &pair.original, &pair.distorted);
        if (pair.original.size != pair.distorted.size)
            throw std::invalid_argument("pair geometries differ");

        std::optional<inmem::VulkanInputs> inputs;
        if (options.vulkan_buffers)
            inputs.emplace(pair.original, pair.distorted);
        const auto size = pair.original.size;
        const bool has_alpha =
            pair.original.HasAlpha() || pair.distorted.HasAlpha();
        MetricPairPipelines pipelines(options, pair, inputs, has_alpha);
        const float background = has_alpha ? 0.1f : 0.5f;
        std::array<double, 2> checksum{}, result{};
        const auto accumulate = [&](const std::array<double, 2>& values) {
            for (std::size_t i = 0; i < values.size(); ++i)
                checksum[i] += values[i];
        };
        for (int i = 0; i < options.warmup; ++i)
            accumulate(pipelines.Score(background));
        vqm::internal::ResetTimings();
        const auto start = Clock::now();
        for (int i = 0; i < options.iterations; ++i) {
            result = pipelines.Score(background);
            accumulate(result);
        }
        const double seconds = Seconds(Clock::now() - start);
        std::array<double, vqm::internal::kStageCount> stages{};
        std::copy_n(
            vqm::internal::Timings().seconds, stages.size(), stages.begin()
        );
        if (has_alpha) {
            const auto bright = pipelines.Score(0.9f);
            result[0] = std::min(result[0], bright[0]);
            result[1] = std::max(result[1], bright[1]);
        }
        std::printf(
            "metric_pair %s input %s size %zux%zu warmup %d iterations %d\n",
            options.metric_pair.c_str(), inputs ? "vulkan-buffers" : "streaming",
            size.width, size.height, options.warmup, options.iterations
        );
        std::printf(
            "ssimulacra2_score %.8f\nbutteraugli_score %.8f\n", result[0],
            result[1]
        );
        std::printf(
            "ssimulacra2_checksum %.12f\nbutteraugli_checksum %.12f\n",
            checksum[0], checksum[1]
        );
        std::printf(
            "wall_seconds %.9f\nms_per_pair %.6f\npairs_per_second %.6f\n",
            seconds, 1000.0 * seconds / options.iterations,
            options.iterations / seconds
        );
        if (has_alpha)
            std::puts(
                "alpha: checksums use dark-background calls; scores include "
                "bright check"
            );
        constexpr int first =
            static_cast<int>(vqm::internal::Stage::Ssimulacra2Prepare);
        constexpr int last =
            static_cast<int>(vqm::internal::Stage::ButteraugliReduce);
        std::printf("stage timing over %d metric pairs:\n", options.iterations);
        PrintStages(stages, options.iterations, seconds, first, last, false);
        if (!options.stages_csv_path.empty() &&
            !WriteStagesCsv(
                options.stages_csv_path, stages, options.iterations, seconds,
                first, last, false
            )) {
            std::fprintf(
                stderr, "could not write %s\n", options.stages_csv_path.c_str()
            );
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "metric pair: %s\n", e.what());
        return 1;
    }
}

}  // namespace bench_driver
