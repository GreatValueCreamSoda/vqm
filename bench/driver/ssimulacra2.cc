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
#include <string_view>
#include <utility>
#include <vector>

#include "bench.h"
#include "inmemory/vulkan_inputs.h"

namespace bench_driver {

// One metric call for the timing loop: resident handles when --resident is in
// play, the decoded images otherwise.
struct Ssimulacra2Scorer {
    vqm::Ssimulacra2Pipeline* pipeline = nullptr;
    const vqm::Image* original = nullptr;
    const vqm::Image* distorted = nullptr;
    const vqm::ImageView* original_view = nullptr;
    const vqm::ImageView* distorted_view = nullptr;
    const vqm::VulkanBufferView* original_buffer = nullptr;
    const vqm::VulkanBufferView* distorted_buffer = nullptr;
    const vqm::UploadedImage* resident_original = nullptr;
    const vqm::UploadedImage* alternate_original = nullptr;
    const vqm::UploadedImage* resident_distorted = nullptr;
    float background = 0.5f;
    bool use_alternate = false;

    [[nodiscard]] double operator()() {
        if (original_buffer != nullptr) {
            return pipeline
                ->Compute(*original_buffer, *distorted_buffer, background)
                .Score();
        }
        if (original_view != nullptr) {
            return ComputeHostViews(
                       *pipeline, *original_view, *distorted_view, background
            )
                .Score();
        }
        if (resident_original != nullptr) {
            const auto& selected =
                use_alternate ? *alternate_original : *resident_original;
            if (alternate_original) use_alternate = !use_alternate;
            if (resident_distorted == nullptr) {
                return pipeline->Compute(selected, *distorted, background)
                    .Score();
            }
            return pipeline->Compute(selected, *resident_distorted, background)
                .Score();
        }
        return pipeline->Compute(*original, *distorted, background).Score();
    }
};

constexpr int kSsimulacra2StageCount =
    static_cast<int>(vqm::internal::Stage::Ssimulacra2Score) + 1;

struct Ssimulacra2PairMeasurement {
    std::vector<double> core_times;
    std::array<double, kSsimulacra2StageCount> stages{};
    double timed_seconds = 0.0;
    double wall_seconds = 0.0;
    double score = 0.0;
};

// One pipeline owns all input representations. Preparation and upload remain
// outside the clock; only Compute/Score calls contribute to pair timing.
Ssimulacra2PairMeasurement MeasureSsimulacra2SerialPair(
    const BenchOptions& options, const vqm::Ssimulacra2Config& config,
    const vqm::Image& original, const vqm::Image& distorted, bool reuse_pipeline,
    std::optional<vqm::Ssimulacra2Pipeline>& batch_pipeline,
    vqm::Size& batch_size, double& checksum
) {
    Ssimulacra2PairMeasurement measurement;
    auto& core_times = measurement.core_times;
    auto& pair_stages = measurement.stages;
    auto& pair_timed_seconds = measurement.timed_seconds;
    auto& pair_wall_seconds = measurement.wall_seconds;
    auto& score = measurement.score;
    // One instance per pair: its buffers are allocated here and
    // reused by every iteration, so the timing covers only the
    // metric. With --resident the pair is parked in device memory
    // once and the timed iterations only score it. A batch score
    // keeps the pipeline across pairs of one geometry.
    std::optional<inmem::VulkanInputs> vulkan_inputs;
    if (options.vulkan_buffers) vulkan_inputs.emplace(original, distorted);
    std::optional<vqm::Ssimulacra2Pipeline> own_pipeline;
    if (reuse_pipeline) {
        if (!batch_pipeline || batch_size != config.size) {
            batch_pipeline.emplace(config);
            batch_size = config.size;
        }
    } else if (vulkan_inputs) {
        own_pipeline.emplace(config, vulkan_inputs->device());
    } else {
        own_pipeline.emplace(config);
    }
    vqm::Ssimulacra2Pipeline& pipeline =
        reuse_pipeline ? *batch_pipeline : *own_pipeline;
    std::optional<vqm::UploadedImage> resident_original;
    std::optional<vqm::UploadedImage> alternate_original;
    std::optional<vqm::UploadedImage> resident_distorted;
    if (options.resident > 0 || options.resident_reference) {
        resident_original = pipeline.Upload(original);
        if (options.resident > 0)
            resident_distorted = pipeline.Upload(distorted);
        alternate_original = pipeline.Upload(original);
    }
    Ssimulacra2Scorer run;
    run.pipeline = &pipeline;
    run.original = &original;
    run.distorted = &distorted;
    if (vulkan_inputs) {
        run.original_buffer = &vulkan_inputs->original();
        run.distorted_buffer = &vulkan_inputs->distorted();
    }
    std::optional<vqm::ImageView> original_view;
    std::optional<vqm::ImageView> distorted_view;
    if (options.host_views) {
        original_view = BorrowImage(original);
        distorted_view = BorrowImage(distorted);
        run.original_view = &*original_view;
        run.distorted_view = &*distorted_view;
    }
    const bool has_alpha = (original.HasAlpha() || distorted.HasAlpha());
    run.background = has_alpha ? 0.1f : 0.5f;
    if (resident_original) {
        run.resident_original = &*resident_original;
        run.alternate_original = &*alternate_original;
        if (resident_distorted) run.resident_distorted = &*resident_distorted;
    }
    for (int i = 0; i < options.warmup; ++i) checksum += run();
    vqm::internal::ResetTimings();
    const Clock::time_point wall_start = Clock::now();
    for (int i = 0; i < options.iterations; ++i) {
        const Clock::time_point start = Clock::now();
        score = run();
        const double seconds = Seconds(Clock::now() - start);
        core_times.push_back(seconds);
        pair_timed_seconds += seconds;
        checksum += score;
    }
    pair_wall_seconds = Seconds(Clock::now() - wall_start);
    // The bright-background scoring call below is intentionally
    // outside the timing loop, so capture stage totals first.
    for (int s = 0; s < kSsimulacra2StageCount; ++s) {
        pair_stages[static_cast<std::size_t>(s)] =
            vqm::internal::Timings().seconds[static_cast<std::size_t>(s)];
    }
    if (has_alpha) {
        // The CLI reports the worse of the dark and bright
        // backgrounds. Only the dark one is timed: the background
        // is a shader constant, so the call's cost is identical.
        run.background = 0.9f;
        const double bright = run();
        score = std::min(score, bright);
    }
    if (resident_original) pipeline.Release(*resident_original);
    if (alternate_original) pipeline.Release(*alternate_original);
    if (resident_distorted) pipeline.Release(*resident_distorted);
    return measurement;
}

// Independent pipelines overlap submissions. The wall clock begins before
// the first submission and ends after every measured result is collected.
Ssimulacra2PairMeasurement MeasureSsimulacra2WorkerPair(
    const BenchOptions& options, const vqm::Ssimulacra2Config& config,
    const vqm::Image& original, const vqm::Image& distorted, double& checksum
) {
    Ssimulacra2PairMeasurement measurement;
    auto& core_times = measurement.core_times;
    auto& pair_stages = measurement.stages;
    auto& pair_timed_seconds = measurement.timed_seconds;
    auto& pair_wall_seconds = measurement.wall_seconds;
    auto& score = measurement.score;
    // Independent workers, one submission each: the second pair is
    // already running while the first one's score is collected,
    // which is where the other half of the machine comes from.
    struct Worker {
        explicit Worker(const vqm::Ssimulacra2Config& c) : pipeline(c) {}
        vqm::Ssimulacra2Pipeline pipeline;
        std::optional<vqm::UploadedImage> original;
        std::optional<vqm::UploadedImage> alternate_original;
        std::optional<vqm::UploadedImage> distorted;
        const vqm::Image* source_original = nullptr;
        const vqm::Image* source_distorted = nullptr;
        float background = 0.5f;
        bool use_alternate = false;

        const vqm::UploadedImage& NextOriginal() {
            const auto& result = use_alternate ? *alternate_original : *original;
            if (alternate_original) use_alternate = !use_alternate;
            return result;
        }

        std::uint64_t Submit() {
            if (original) {
                return pipeline.Submit(NextOriginal(), *distorted, background);
            }
            return pipeline.Submit(
                *source_original, *source_distorted, background
            );
        }
    };
    std::vector<Worker> workers;
    workers.reserve(static_cast<std::size_t>(options.workers));
    const bool has_alpha = (original.HasAlpha() || distorted.HasAlpha());
    const float background = has_alpha ? 0.1f : 0.5f;
    for (int w = 0; w < options.workers; ++w) {
        workers.emplace_back(config);
        Worker& worker = workers.back();
        worker.source_original = &original;
        worker.source_distorted = &distorted;
        worker.background = background;
        if (options.resident > 0) {
            worker.original = worker.pipeline.Upload(original);
            worker.distorted = worker.pipeline.Upload(distorted);
            worker.alternate_original = worker.pipeline.Upload(original);
        }
    }
    for (int i = 0; i < options.warmup; ++i) {
        for (Worker& worker : workers) {
            double warm;
            if (worker.original) {
                warm =
                    worker.pipeline
                        .Compute(
                            worker.NextOriginal(), *worker.distorted, background
                        )
                        .Score();
            } else {
                warm = worker.pipeline.Compute(original, distorted, background)
                           .Score();
            }
            checksum += warm;
        }
    }
    vqm::internal::ResetTimings();
    const std::size_t active =
        std::min(workers.size(), static_cast<std::size_t>(options.iterations));
    std::vector<std::uint64_t> tokens(active);
    const Clock::time_point start = Clock::now();
    for (std::size_t w = 0; w < active; ++w) {
        tokens[w] = workers[w].Submit();
    }
    for (int i = 0; i < options.iterations; ++i) {
        const std::size_t w = static_cast<std::size_t>(i) % active;
        score = workers[w].pipeline.Finish(tokens[w]).Score();
        checksum += score;
        if (i + static_cast<int>(active) < options.iterations) {
            tokens[w] = workers[w].Submit();
        }
    }
    const double per_iteration =
        Seconds(Clock::now() - start) / options.iterations;
    pair_timed_seconds = per_iteration * options.iterations;
    pair_wall_seconds = pair_timed_seconds;
    for (int i = 0; i < options.iterations; ++i) {
        core_times.push_back(per_iteration);
    }
    for (int s = 0; s < kSsimulacra2StageCount; ++s) {
        pair_stages[static_cast<std::size_t>(s)] =
            vqm::internal::Timings().seconds[static_cast<std::size_t>(s)];
    }
    if (has_alpha) {
        Worker& worker = workers.front();
        const double bright =
            worker.original
                ? worker.pipeline
                      .Compute(*worker.original, *worker.distorted, 0.9f)
                      .Score()
                : worker.pipeline.Compute(original, distorted, 0.9f).Score();
        score = std::min(score, bright);
    }
    return measurement;
}

int RunSsimulacra2(const BenchOptions& options) {
    std::vector<Row> rows;
    double checksum = 0.0;
    double total_core_seconds = 0.0;
    double total_wall_seconds = 0.0;
    int failures = 0;
    // The SSIMULACRA 2 arm reports its own three stages only: Stage::Count also
    // holds the Butteraugli stages, and printing seven zero rows here would
    // break the protocol output's comparability, as would the stages CSV's
    // extra zero columns.
    std::array<double, vqm::internal::kStageCount> stage_totals = {};
    // Batch scoring keeps one pipeline per geometry alive across pairs; the
    // per-pair pipeline construction is the expensive part at small sizes.
    const bool reuse_pipeline = options.score_only && options.resident == 0 &&
                                options.workers <= 1 && !options.vulkan_buffers;
    std::optional<vqm::Ssimulacra2Pipeline> batch_pipeline;
    vqm::Size batch_size;
    PairDecoder decoder(options);
    if (!options.score_only) {
        std::printf(
            "%-60s %9s %9s %10s\n", "pair", "decode_ms", "core_ms", "score"
        );
    }
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

        std::vector<double> core_times;
        std::array<double, kSsimulacra2StageCount> pair_stages{};
        double pair_timed_seconds = 0.0;
        double pair_wall_seconds = 0.0;
        double score = 0.0;
        bool ok = true;
        try {
            vqm::Ssimulacra2Config config;
            config.size = {original.size.width, original.size.height};
            config.half_planes = options.half_planes;
            config.chroma_filter = options.input.chroma_filter;
            if (options.vulkan_buffers) {
                config.gpu_resident_images = 0;
                config.streaming_inputs = false;
            }
            if (options.resident > 0) {
                config.gpu_resident_images = std::max(
                    std::size_t{3}, static_cast<std::size_t>(options.resident)
                );
            }
            // The same overrides the CLI applies: replace what the decoder
            // declared, on both sides of the pair.
            ApplyOverrides(options, &original, &distorted);
            Ssimulacra2PairMeasurement measurement =
                options.workers <= 1
                    ? MeasureSsimulacra2SerialPair(
                          options, config, original, distorted, reuse_pipeline,
                          batch_pipeline, batch_size, checksum
                      )
                    : MeasureSsimulacra2WorkerPair(
                          options, config, original, distorted, checksum
                      );
            core_times = std::move(measurement.core_times);
            pair_stages = measurement.stages;
            pair_timed_seconds = measurement.timed_seconds;
            pair_wall_seconds = measurement.wall_seconds;
            score = measurement.score;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s: %s\n", pair.original.c_str(), e.what());
            ok = false;
        }
        if (!ok) {
            if (options.score_only) std::puts("ERROR");
            ++failures;
            continue;
        }

        Row row;
        row.pair = pair;
        row.width = static_cast<int>(original.size.width);
        row.height = static_cast<int>(original.size.height);
        row.decode_ms = decode_ms;
        row.core_ms = 1000.0 * Median(core_times);
        row.score = score;
        total_core_seconds += pair_timed_seconds;
        total_wall_seconds += pair_wall_seconds;
        for (int s = 0; s < kSsimulacra2StageCount; ++s) {
            stage_totals[static_cast<std::size_t>(s)] +=
                pair_stages[static_cast<std::size_t>(s)];
        }
        rows.push_back(row);
        if (options.score_only) {
            std::printf("%.8f\n", row.score);
        } else {
            std::printf(
                "%-60s %9.1f %9.1f %10.6f\n",
                (pair.original + " vs " + pair.distorted).c_str(), row.decode_ms,
                row.core_ms, row.score
            );
        }
    }

    if (rows.empty()) {
        std::fprintf(stderr, "no pair could be measured\n");
        return 1;
    }

    std::vector<double> decodes;
    std::vector<double> cores;
    for (const Row& row : rows) {
        decodes.push_back(row.decode_ms);
        cores.push_back(row.core_ms);
    }
    // Stage breakdown over all timed iterations of all measured pairs.
    const double total_iterations = static_cast<double>(rows.size()) *
                                    static_cast<double>(options.iterations);
    if (!options.score_only) {
        const double median_decode = Median(decodes);
        const double median_core = Median(cores);
        std::printf(
            "aggregate: median decode %.1f ms, median core %.1f ms, median "
            "total %.1f ms\n",
            median_decode, median_core, median_decode + median_core
        );
        if (options.resident > 0 && total_wall_seconds > 0.0) {
            std::printf(
                "resident: %.1f pairs/s, %.2f ms/pair (fixed-count wall "
                "clock)\n",
                total_iterations / total_wall_seconds,
                1000.0 * total_wall_seconds / total_iterations
            );
        }
        std::printf(
            "stage timing over %d iterations (%zu pairs):\n", options.iterations,
            rows.size()
        );
        PrintStages(
            stage_totals, total_iterations, total_core_seconds, 0,
            kSsimulacra2StageCount - 1, options.workers > 1
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
            total_core_seconds, 0, kSsimulacra2StageCount - 1,
            options.workers > 1
        )) {
        std::fprintf(
            stderr, "could not write %s\n", options.stages_csv_path.c_str()
        );
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}

}  // namespace bench_driver
