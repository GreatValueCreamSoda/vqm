// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "bench.h"
#include "butteraugli.h"
#include "inmemory/vulkan_inputs.h"

namespace bench_driver {

struct ButteraugliWorkersResult {
    double score = 0.0;
    double checksum = 0.0;
    double seconds = 0.0;
    std::array<double, vqm::internal::kStageCount> stages{};
};

// Measure a fixed number of pairs across independent pipelines. Each pipeline
// has one submission in flight; the timer includes the initial submissions.
ButteraugliWorkersResult MeasureButteraugliWorkers(
    const BenchOptions& options, const vqm::ButteraugliConfig& config,
    const vqm::Image& original, const vqm::Image& distorted
) {
    struct Worker {
        explicit Worker(const vqm::ButteraugliConfig& config)
            : pipeline(config) {}
        vqm::ButteraugliPipeline pipeline;
        std::optional<vqm::UploadedImage> original;
        std::optional<vqm::UploadedImage> distorted;
    };
    std::vector<Worker> workers;
    workers.reserve(static_cast<std::size_t>(options.workers));
    for (int i = 0; i < options.workers; ++i) {
        workers.emplace_back(config);
        if (options.resident > 0) {
            workers.back().original = workers.back().pipeline.Upload(original);
            workers.back().distorted = workers.back().pipeline.Upload(distorted);
        }
    }
    const bool has_alpha = (original.HasAlpha() || distorted.HasAlpha());
    const float background = has_alpha ? 0.1f : 0.5f;
    const auto compute = [&](Worker& worker, float bg) {
        return worker.original
                   ? worker.pipeline
                         .Compute(*worker.original, *worker.distorted, bg)
                         .score
                   : worker.pipeline.Compute(original, distorted, bg).score;
    };
    const auto submit = [&](Worker& worker) {
        return worker.original
                   ? worker.pipeline.Submit(
                         *worker.original, *worker.distorted, background
                     )
                   : worker.pipeline.Submit(original, distorted, background);
    };
    ButteraugliWorkersResult result;
    for (int i = 0; i < options.warmup; ++i) {
        for (Worker& worker : workers)
            result.checksum += compute(worker, background);
    }
    vqm::internal::ResetTimings();
    const std::size_t active =
        std::min(workers.size(), static_cast<std::size_t>(options.iterations));
    std::vector<std::uint64_t> tokens(active);
    const Clock::time_point start = Clock::now();
    for (std::size_t worker = 0; worker < active; ++worker) {
        tokens[worker] = submit(workers[worker]);
    }
    for (int i = 0; i < options.iterations; ++i) {
        const std::size_t worker = static_cast<std::size_t>(i) % active;
        result.score = workers[worker].pipeline.Finish(tokens[worker]).score;
        result.checksum += result.score;
        if (i + static_cast<int>(active) < options.iterations) {
            tokens[worker] = submit(workers[worker]);
        }
    }
    result.seconds = Seconds(Clock::now() - start);
    for (int stage = static_cast<int>(vqm::internal::Stage::ButteraugliOpsin);
         stage <= static_cast<int>(vqm::internal::Stage::ButteraugliReduce);
         ++stage) {
        result.stages[static_cast<std::size_t>(stage)] =
            vqm::internal::Timings().seconds[static_cast<std::size_t>(stage)];
    }
    if (has_alpha)
        result.score = std::max(result.score, compute(workers.front(), 0.9f));
    return result;
}

void AccumulateButteraugliStages(
    std::array<double, vqm::internal::kStageCount>& target, const double* source
) {
    for (int stage = static_cast<int>(vqm::internal::Stage::ButteraugliOpsin);
         stage <= static_cast<int>(vqm::internal::Stage::ButteraugliReduce);
         ++stage) {
        target[static_cast<std::size_t>(stage)] +=
            source[static_cast<std::size_t>(stage)];
    }
}

void ReadyButteraugliBatch(
    const vqm::ButteraugliConfig& config,
    std::optional<vqm::ButteraugliPipeline>& pipeline, vqm::Size& size
) {
    if (pipeline && size == config.size) return;
    pipeline.emplace(config);
    size = config.size;
}

// The Butteraugli arm. Same flags as the SSIMULACRA 2 arm -- pairs,
// iterations, warmup, resident -- but its own stages, so it prints the stages
// this metric timed. Its reference chain is per pipeline and the metric is f32.
int RunButteraugli(const BenchOptions& options) {
    constexpr int kFirstButter =
        static_cast<int>(vqm::internal::Stage::ButteraugliOpsin);
    constexpr int kLastButter =
        static_cast<int>(vqm::internal::Stage::ButteraugliReduce);
    if (!options.score_only) {
        std::printf(
            "%-60s %9s %9s %10s\n", "pair", "decode_ms", "core_ms", "score"
        );
    }
    std::vector<Row> rows;
    std::vector<double> decodes;
    std::vector<double> cores;
    std::array<double, vqm::internal::kStageCount> stage_totals = {};
    double checksum = 0.0;
    double total_iterations = 0.0;
    double total_core_seconds = 0.0;
    int failures = 0;
    // A batch scorer reuses one pipeline per geometry: the pipeline owns the
    // device state and its arena, and paying that per pair would make the
    // metric a rounding error next to creation. The timed loop is still one
    // pair's Compute calls.
    const bool reuse_pipeline =
        options.score_only && options.resident == 0 && !options.vulkan_buffers;
    std::optional<vqm::ButteraugliPipeline> batch_pipeline;
    vqm::Size batch_size;
    PairDecoder decoder(options);
    for (const Pair& pair : options.pairs) {
        DecodedPair images;
        std::string decode_error;
        const bool decoded = decoder.Decode(pair, images, decode_error);
        if (!decoded) {
            std::fprintf(
                stderr, "%s: %s\n", pair.original.c_str(), decode_error.c_str()
            );
            if (options.score_only) std::puts("ERROR");
            ++failures;
            continue;
        }
        vqm::Image& original = images.original;
        vqm::Image& distorted = images.distorted;
        const double decode_ms = images.decode_ms;
        try {
            ApplyOverrides(options, &original, &distorted);
            vqm::ButteraugliConfig config;
            config.size = {original.size.width, original.size.height};
            config.chroma_filter = options.input.chroma_filter;
            config.hf_asymmetry = options.hf_asymmetry;
            config.intensity_target = options.intensity_target;
            config.streaming_inputs = options.resident == 0;
            if (options.vulkan_buffers) {
                config.gpu_resident_images = 0;
                config.streaming_inputs = false;
            }
            if (options.resident > 0) {
                config.gpu_resident_images =
                    static_cast<std::size_t>(options.resident);
            }
            if (options.workers > 1) {
                const ButteraugliWorkersResult result =
                    MeasureButteraugliWorkers(
                        options, config, original, distorted
                    );
                const double core_ms =
                    result.seconds * 1000.0 / options.iterations;
                checksum += result.checksum;
                total_iterations += options.iterations;
                total_core_seconds += result.seconds;
                AccumulateButteraugliStages(stage_totals, result.stages.data());
                rows.push_back(
                    Row{pair, static_cast<int>(original.size.width),
                        static_cast<int>(original.size.height), decode_ms,
                        core_ms, result.score}
                );
                decodes.push_back(decode_ms);
                cores.push_back(core_ms);
                std::printf(
                    "%-60s %9.1f %9.1f %10.6f\n",
                    (pair.original + " vs " + pair.distorted).c_str(), decode_ms,
                    core_ms, result.score
                );
                continue;
            }
            std::optional<inmem::VulkanInputs> vulkan_inputs;
            if (options.vulkan_buffers)
                vulkan_inputs.emplace(original, distorted);
            std::optional<vqm::ButteraugliPipeline> own_pipeline;
            if (reuse_pipeline) {
                ReadyButteraugliBatch(config, batch_pipeline, batch_size);
            } else if (vulkan_inputs) {
                own_pipeline.emplace(config, vulkan_inputs->device());
            } else {
                own_pipeline.emplace(config);
            }
            vqm::ButteraugliPipeline& pipeline =
                reuse_pipeline ? *batch_pipeline : *own_pipeline;
            vqm::UploadedImage resident_original;
            vqm::UploadedImage resident_distorted;
            if (options.resident > 0 || options.resident_reference) {
                resident_original = pipeline.Upload(original);
                if (options.resident > 0)
                    resident_distorted = pipeline.Upload(distorted);
            }
            std::vector<double> core_times;
            double pair_timed_seconds = 0.0;
            double score = 0.0;
            std::optional<vqm::ImageView> original_view;
            std::optional<vqm::ImageView> distorted_view;
            if (options.host_views) {
                original_view = BorrowImage(original);
                distorted_view = BorrowImage(distorted);
            }
            // Keep input representation selection common to the timed dark
            // call and the untimed bright-background check.
            const auto compute = [&](float alpha_background) {
                if (options.vulkan_buffers)
                    return pipeline.Compute(
                        vulkan_inputs->original(), vulkan_inputs->distorted(),
                        alpha_background
                    );
                if (options.host_views)
                    return ComputeHostViews(
                        pipeline, *original_view, *distorted_view,
                        alpha_background
                    );
                if (options.resident_reference)
                    return pipeline.Compute(
                        resident_original, distorted, alpha_background
                    );
                if (options.resident > 0)
                    return pipeline.Compute(
                        resident_original, resident_distorted, alpha_background
                    );
                return pipeline.Compute(original, distorted, alpha_background);
            };
            const bool has_alpha = (original.HasAlpha() || distorted.HasAlpha());
            const float background = has_alpha ? 0.1f : 0.5f;
            for (int i = 0; i < options.warmup + options.iterations; ++i) {
                const bool measured = i >= options.warmup;
                if (measured) vqm::internal::ResetTimings();
                const Clock::time_point start = Clock::now();
                const vqm::ButteraugliScore result = compute(background);
                const double seconds = Seconds(Clock::now() - start);
                checksum += result.score;
                if (!measured) continue;
                core_times.push_back(seconds);
                pair_timed_seconds += seconds;
                score = result.score;
                AccumulateButteraugliStages(
                    stage_totals, vqm::internal::Timings().seconds
                );
            }
            if (has_alpha) {
                // The CLI reports the worse (larger) score over the dark and
                // bright backgrounds; only the dark one is timed because the
                // background is a shader constant.
                const vqm::ButteraugliScore bright = compute(0.9f);
                score = std::max(score, bright.score);
            }
            if (options.resident > 0 || options.resident_reference) {
                pipeline.Release(resident_original);
                if (options.resident > 0) pipeline.Release(resident_distorted);
            }
            const double core_ms = 1000.0 * Median(core_times);
            rows.push_back(
                Row{pair, static_cast<int>(original.size.width),
                    static_cast<int>(original.size.height), decode_ms, core_ms,
                    score}
            );
            decodes.push_back(decode_ms);
            cores.push_back(core_ms);
            total_iterations += static_cast<double>(options.iterations);
            total_core_seconds += pair_timed_seconds;
            if (options.score_only) {
                std::printf("%.8f\n", score);
            } else {
                std::printf(
                    "%-60s %9.1f %9.1f %10.6f\n",
                    (pair.original + " vs " + pair.distorted).c_str(), decode_ms,
                    core_ms, score
                );
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s: %s\n", pair.original.c_str(), e.what());
            if (options.score_only) std::puts("ERROR");
            ++failures;
        }
    }
    if (decodes.empty()) {
        std::fprintf(stderr, "no pair could be measured\n");
        return 1;
    }
    if (!options.score_only) {
        const double median_decode = Median(decodes);
        const double median_core = Median(cores);
        std::printf(
            "aggregate: median decode %.1f ms, median core %.1f ms, median "
            "total %.1f ms\n",
            median_decode, median_core, median_decode + median_core
        );
        if (options.resident > 0 && median_core > 0.0) {
            std::printf(
                "resident: %.1f pairs/s, %.2f ms/pair\n", 1000.0 / median_core,
                median_core
            );
        }
        std::printf(
            "stage timing over %d iterations (%zu pairs):\n", options.iterations,
            decodes.size()
        );
        PrintStages(
            stage_totals, total_iterations, total_core_seconds, kFirstButter,
            kLastButter, options.workers > 1
        );
        std::printf("checksum %.6f\n", checksum);
    }
    if (!options.csv_path.empty() && !WritePairsCsv(options.csv_path, rows)) {
        std::fprintf(stderr, "could not write %s\n", options.csv_path.c_str());
        ++failures;
    }
    if (!options.stages_csv_path.empty() &&
        !WriteStagesCsv(
            options.stages_csv_path, stage_totals, total_iterations,
            total_core_seconds, kFirstButter, kLastButter, options.workers > 1
        )) {
        std::fprintf(
            stderr, "could not write %s\n", options.stages_csv_path.c_str()
        );
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}

}  // namespace bench_driver
