// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "bench.h"
#include "butteraugli.h"
#include "vqm_pool.h"

namespace bench_driver {

// The public pool API scores a fixed number of streamed pairs, including the
// initial submissions and final drains in the wall-clock interval.
template <typename Pool>
auto PoolResults(
    Pool& pool, std::span<const vqm::ImagePair> pairs, float background
) {
    if constexpr (std::is_same_v<Pool, vqm::Ssimulacra2Pool>) {
        return pool.ScoreMany(pairs, background);
    } else {
        return pool.ComputeMany(pairs, background);
    }
}

double PoolScore(double score) { return score; }
double PoolScore(vqm::ButteraugliScore score) { return score.score; }

template <typename Pool>
int MeasurePool(
    Pool& pool, const BenchOptions& options, const Pair& pair, double decode_ms,
    const vqm::Image& original, const vqm::Image& distorted
) {
    const std::size_t workers = pool.Workers();
    const std::size_t count = static_cast<std::size_t>(options.iterations);
    const std::size_t warmup =
        static_cast<std::size_t>(options.warmup) * workers;
    const vqm::ImagePair input{original, distorted};
    const std::vector<vqm::ImagePair> warm_pairs(warmup, input);
    const std::vector<vqm::ImagePair> timed_pairs(count, input);
    const float background =
        (original.HasAlpha() || distorted.HasAlpha()) ? 0.1f : 0.5f;
    const auto warm_scores = PoolResults(pool, warm_pairs, background);
    double checksum = 0.0;
    for (const auto& value : warm_scores) checksum += PoolScore(value);

    vqm::internal::ResetTimings();
    const Clock::time_point start = Clock::now();
    const auto timed_scores = PoolResults(pool, timed_pairs, background);
    const double seconds = Seconds(Clock::now() - start);
    std::array<double, vqm::internal::kStageCount> stages{};
    for (std::size_t index = 0; index < stages.size(); ++index) {
        stages[index] = vqm::internal::Timings().seconds[index];
    }
    for (const auto& value : timed_scores) checksum += PoolScore(value);
    double score = PoolScore(timed_scores.back());
    if ((original.HasAlpha() || distorted.HasAlpha())) {
        const std::array<vqm::ImagePair, 1> bright{{input}};
        const double candidate =
            PoolScore(PoolResults(pool, std::span(bright), 0.9f)[0]);
        score = (options.metric == Metric::Butteraugli)
                    ? std::max(score, candidate)
                    : std::min(score, candidate);
    }

    const double ms_per_call = 1000.0 * seconds / static_cast<double>(count);
    std::printf("%-60s %9s %9s %10s\n", "pair", "decode_ms", "core_ms", "score");
    std::printf(
        "%-60s %9.1f %9.1f %10.6f\n",
        (pair.original + " vs " + pair.distorted).c_str(), decode_ms,
        ms_per_call, score
    );
    std::printf(
        "pool: %.1f pairs/s, %.4f ms/pair (fixed-count wall clock)\n",
        static_cast<double>(count) / seconds, ms_per_call
    );
    const int first =
        (options.metric == Metric::Butteraugli)
            ? static_cast<int>(vqm::internal::Stage::ButteraugliOpsin)
            : static_cast<int>(vqm::internal::Stage::Ssimulacra2Prepare);
    const int last =
        (options.metric == Metric::Butteraugli)
            ? static_cast<int>(vqm::internal::Stage::ButteraugliReduce)
            : static_cast<int>(vqm::internal::Stage::Ssimulacra2Score);
    std::printf(
        "stage timing over %d iterations (1 pairs):\n", options.iterations
    );
    PrintStages(
        stages, static_cast<double>(count), seconds, first, last, workers > 1
    );
    std::printf("checksum %.6f\n", checksum);
    const Row row{pair,
                  static_cast<int>(original.size.width),
                  static_cast<int>(original.size.height),
                  decode_ms,
                  ms_per_call,
                  score};
    if (!options.csv_path.empty() && !WritePairsCsv(options.csv_path, {row})) {
        std::fprintf(stderr, "could not write %s\n", options.csv_path.c_str());
        return 1;
    }
    if (!options.stages_csv_path.empty() &&
        !WriteStagesCsv(
            options.stages_csv_path, stages, static_cast<double>(count), seconds,
            first, last, workers > 1
        )) {
        std::fprintf(
            stderr, "could not write %s\n", options.stages_csv_path.c_str()
        );
        return 1;
    }
    return 0;
}

int RunPool(const BenchOptions& options) {
    if (options.pairs.size() != 1) {
        std::fprintf(stderr, "--pool requires exactly one --pair\n");
        return 2;
    }
    const Pair& pair = options.pairs.front();
    vqm::Image original, distorted;
    double decode_ms = 0.0;
    std::string error;
    if (!MeasureDecode(pair, 3, &decode_ms, &original, &distorted, &error)) {
        std::fprintf(stderr, "%s: %s\n", pair.original.c_str(), error.c_str());
        return 1;
    }
    ApplyOverrides(options, &original, &distorted);
    try {
        const std::size_t workers =
            static_cast<std::size_t>(std::max(options.workers, 1));
        if (options.metric == Metric::Butteraugli) {
            vqm::ButteraugliConfig config;
            config.size = original.size;
            config.chroma_filter = options.input.chroma_filter;
            config.hf_asymmetry = options.hf_asymmetry;
            config.intensity_target = options.intensity_target;
            vqm::ButteraugliPool pool(config, workers);
            return MeasurePool(
                pool, options, pair, decode_ms, original, distorted
            );
        }
        vqm::Ssimulacra2Config config;
        config.size = original.size;
        config.chroma_filter = options.input.chroma_filter;
        config.half_planes = options.half_planes;
        vqm::Ssimulacra2Pool pool(config, workers);
        return MeasurePool(pool, options, pair, decode_ms, original, distorted);
    } catch (const std::exception& e) {
        std::fprintf(
            stderr, "%s vs %s: %s\n", pair.original.c_str(),
            pair.distorted.c_str(), e.what()
        );
        return 1;
    }
}

// One ring owns one pipeline and two distinct resident handles per listed
// pair. Rotating the original handle between consecutive calls forces both
// prepare dispatches even when the files contain identical pixels.
std::size_t ResidentRingInputBytes(vqm::Size size, std::size_t slots) {
    if (size.width == 0 ||
        size.height > std::numeric_limits<std::size_t>::max() / size.width ||
        size.width * size.height >
            std::numeric_limits<std::size_t>::max() / slots / 16) {
        throw std::overflow_error("resident ring size overflows size_t");
    }
    return size.width * size.height * slots * 16;
}

int RunResidentRing(const BenchOptions& options) {
    const std::size_t count = options.pairs.size();
    if (count < 2 || count > static_cast<std::size_t>(options.resident_ring)) {
        std::fprintf(
            stderr, "--resident-ring needs between 2 and %d pairs\n",
            options.resident_ring
        );
        return 2;
    }
    if (static_cast<std::size_t>(options.iterations) < count) {
        std::fprintf(
            stderr, "--iterations must cover every resident ring pair\n"
        );
        return 2;
    }
    struct Entry {
        vqm::UploadedImage original;
        vqm::UploadedImage distorted;
        bool has_alpha;
        double score = 0.0;
    };
    try {
        std::vector<Entry> entries;
        entries.reserve(count);
        std::optional<vqm::Ssimulacra2Pipeline> pipeline;
        vqm::Size size;
        std::size_t resident_bytes = 0;
        for (const Pair& pair : options.pairs) {
            DecodedPair decoded;
            std::string error;
            if (!LoadOnce(pair, decoded, error)) {
                std::fprintf(
                    stderr, "%s: %s\n", pair.original.c_str(), error.c_str()
                );
                return 1;
            }
            if (decoded.original.size != decoded.distorted.size) {
                std::fprintf(
                    stderr, "resident ring requires matching image geometries\n"
                );
                return 1;
            }
            if (!pipeline) {
                size = decoded.original.size;
                const std::size_t slots = 2 * count;
                resident_bytes = ResidentRingInputBytes(size, slots);
                vqm::Ssimulacra2Config config;
                config.size = size;
                config.gpu_resident_images = slots;
                config.streaming_inputs = false;
                config.half_planes = options.half_planes;
                config.chroma_filter = options.input.chroma_filter;
                pipeline.emplace(config);
            } else if (decoded.original.size != size) {
                std::fprintf(
                    stderr, "resident ring requires matching image geometries\n"
                );
                return 1;
            }
            ApplyOverrides(options, &decoded.original, &decoded.distorted);
            const bool alpha =
                decoded.original.HasAlpha() || decoded.distorted.HasAlpha();
            entries.push_back(
                {pipeline->Upload(decoded.original),
                 pipeline->Upload(decoded.distorted), alpha}
            );
        }
        const auto compute = [&](std::size_t index, float background) {
            const Entry& entry = entries[index];
            return pipeline->Compute(entry.original, entry.distorted, background)
                .Score();
        };
        // Score each pair outside the measured calls, including the bright
        // alpha background. Timed calls use the dark background only.
        for (std::size_t i = 0; i < count; ++i) {
            entries[i].score = compute(i, entries[i].has_alpha ? 0.1f : 0.5f);
            if (entries[i].has_alpha)
                entries[i].score = std::min(entries[i].score, compute(i, 0.9f));
        }
        double checksum = 0.0;
        const auto run = [&](std::size_t i) {
            const std::size_t index = i % count;
            return compute(index, entries[index].has_alpha ? 0.1f : 0.5f);
        };
        for (int i = 0; i < options.warmup; ++i)
            checksum += run(static_cast<std::size_t>(i));
        vqm::internal::ResetTimings();
        const auto start = Clock::now();
        for (int i = 0; i < options.iterations; ++i)
            checksum +=
                run(static_cast<std::size_t>(options.warmup) +
                    static_cast<std::size_t>(i));
        const double seconds = Seconds(Clock::now() - start);
        std::array<double, vqm::internal::kStageCount> stages{};
        std::copy_n(
            vqm::internal::Timings().seconds, stages.size(), stages.begin()
        );
        std::printf(
            "resident ring: %zu pairs, %zu slots, %zux%zu, %zu resident input "
            "bytes\n",
            count, 2 * count, size.width, size.height, resident_bytes
        );
        for (std::size_t i = 0; i < count; ++i) {
            std::printf(
                "pair[%zu] %s vs %s score %.8f\n", i,
                options.pairs[i].original.c_str(),
                options.pairs[i].distorted.c_str(), entries[i].score
            );
        }
        std::printf(
            "resident ring: %.1f pairs/s, %.4f ms/pair (fixed-count wall "
            "clock)\n",
            options.iterations / seconds, 1000.0 * seconds / options.iterations
        );
        std::printf(
            "stage timing over %d iterations (%zu pairs):\n", options.iterations,
            count
        );
        constexpr int first =
            static_cast<int>(vqm::internal::Stage::Ssimulacra2Prepare);
        constexpr int last =
            static_cast<int>(vqm::internal::Stage::Ssimulacra2Score);
        PrintStages(stages, options.iterations, seconds, first, last, false);
        std::printf("checksum %.6f\n", checksum);
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
        std::fprintf(stderr, "resident ring: %s\n", e.what());
        return 1;
    }
}

}  // namespace bench_driver
