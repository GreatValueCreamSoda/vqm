// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "bench.h"
#include "inmemory/vulkan_inputs.h"
#include "psnr_hvs.h"

namespace bench_driver {
namespace {

constexpr int kFirst = static_cast<int>(vqm::internal::Stage::PsnrHvsPrepare);
constexpr int kLast = static_cast<int>(vqm::internal::Stage::PsnrHvsScore);

struct Totals {
    std::array<double, vqm::internal::kStageCount> stages{};
    double checksum = 0.0;
    double seconds = 0.0;
    double calls = 0.0;

    void Accumulate(double elapsed, int iterations) {
        seconds += elapsed;
        calls += iterations;
        for (int stage = kFirst; stage <= kLast; ++stage) {
            const auto index = static_cast<std::size_t>(stage);
            stages[index] += vqm::internal::Timings().seconds[index];
        }
    }
};

vqm::PsnrHvsConfig ConfigFor(const BenchOptions& options, vqm::Size size) {
    vqm::PsnrHvsConfig config{.size = size};
    config.gpu_resident_images =
        options.resident_reference ? 1
                                   : static_cast<std::size_t>(options.resident);
    config.streaming_inputs = !options.vulkan_buffers && options.resident == 0;
    return config;
}

// Inputs are installed outside the measurement. There is no prepared-reference
// cache: every submission prepares both raw inputs, including resident calls.
class Worker {
public:
    Worker(
        const BenchOptions& options, const vqm::PsnrHvsConfig& config,
        const DecodedPair& images, const inmem::VulkanInputs* external,
        vqm::PsnrHvsPipeline* cached
    )
        : original_(images.original),
          distorted_(images.distorted),
          external_(external),
          pipeline_(cached) {
        if (!pipeline_) {
            if (external)
                owned_ = std::make_unique<vqm::PsnrHvsPipeline>(
                    config, external->device()
                );
            else
                owned_ = std::make_unique<vqm::PsnrHvsPipeline>(config);
            pipeline_ = owned_.get();
        }
        if (options.host_views) {
            original_view_ = BorrowImage(original_);
            distorted_view_ = BorrowImage(distorted_);
        }
        if (options.resident || options.resident_reference)
            resident_original_ = pipeline_->Upload(original_);
        if (options.resident)
            resident_distorted_ = pipeline_->Upload(distorted_);
    }

    [[nodiscard]] std::uint64_t Submit() {
        if (external_)
            return pipeline_->Submit(
                external_->original(), external_->distorted()
            );
        if (original_view_)
            return pipeline_->Submit(*original_view_, *distorted_view_);
        if (resident_distorted_)
            return pipeline_->Submit(*resident_original_, *resident_distorted_);
        if (resident_original_)
            return pipeline_->Submit(*resident_original_, distorted_);
        return pipeline_->Submit(original_, distorted_);
    }

    [[nodiscard]] vqm::PsnrHvsScore Finish(std::uint64_t token) {
        return pipeline_->Finish(token);
    }

    [[nodiscard]] vqm::PipelineMemoryUsage Memory() const {
        return pipeline_->GetMemoryUsage();
    }

private:
    const vqm::Image& original_;
    const vqm::Image& distorted_;
    const inmem::VulkanInputs* external_;
    std::unique_ptr<vqm::PsnrHvsPipeline> owned_;
    vqm::PsnrHvsPipeline* pipeline_;
    std::optional<vqm::ImageView> original_view_, distorted_view_;
    std::optional<vqm::UploadedImage> resident_original_, resident_distorted_;
};

void PrintMemory(const std::vector<std::unique_ptr<Worker>>& workers) {
    std::uint64_t bytes = 0;
    for (const auto& worker : workers) {
        const auto usage = worker->Memory();
        bytes += usage.arena.allocation_bytes + usage.staging.allocation_bytes;
        for (const auto& auxiliary : usage.auxiliary)
            bytes += auxiliary.allocation_bytes;
    }
    std::printf(
        "PSNR-HVS native coded planes, %zu workers, both inputs prepared per "
        "call\nowned buffer allocations %llu bytes\n",
        workers.size(), static_cast<unsigned long long>(bytes)
    );
}

struct Measurement {
    double core_ms = 0.0;
    vqm::PsnrHvsScore result;
};

Measurement MeasureSerial(Worker& worker, int iterations, Totals& totals) {
    Measurement measured;
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    vqm::internal::ResetTimings();
    const auto wall = Clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto start = Clock::now();
        measured.result = worker.Finish(worker.Submit());
        samples.push_back(1000.0 * Seconds(Clock::now() - start));
        totals.checksum += measured.result.score;
    }
    totals.Accumulate(Seconds(Clock::now() - wall), iterations);
    measured.core_ms = Median(samples);
    return measured;
}

Measurement MeasureWorkers(
    const std::vector<std::unique_ptr<Worker>>& workers, int iterations,
    Totals& totals
) {
    Measurement measured;
    const auto active =
        std::min(workers.size(), static_cast<std::size_t>(iterations));
    std::vector<std::uint64_t> tokens(active);
    vqm::internal::ResetTimings();
    // Include every initial submission and every final completion in the wall
    // clock. GPU stage times are cumulative and can overlap between workers.
    const auto wall = Clock::now();
    for (std::size_t index = 0; index < active; ++index)
        tokens[index] = workers[index]->Submit();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto index = static_cast<std::size_t>(iteration) % active;
        measured.result = workers[index]->Finish(tokens[index]);
        totals.checksum += measured.result.score;
        if (static_cast<std::size_t>(iteration) + active <
            static_cast<std::size_t>(iterations))
            tokens[index] = workers[index]->Submit();
    }
    const double elapsed = Seconds(Clock::now() - wall);
    totals.Accumulate(elapsed, iterations);
    measured.core_ms = 1000.0 * elapsed / iterations;
    return measured;
}

Measurement MeasurePair(
    const BenchOptions& options, const DecodedPair& images,
    vqm::PsnrHvsPipeline* cached, Totals& totals
) {
    const auto config = ConfigFor(options, images.original.size);
    std::optional<inmem::VulkanInputs> external;
    if (options.vulkan_buffers)
        external.emplace(images.original, images.distorted);
    const int count = options.score_only ? 1 : options.workers;
    std::vector<std::unique_ptr<Worker>> workers;
    for (int index = 0; index < count; ++index)
        workers.push_back(
            std::make_unique<Worker>(
                options, config, images, external ? &*external : nullptr, cached
            )
        );
    if (!options.score_only) PrintMemory(workers);
    const int warmup = options.score_only ? 0 : options.warmup;
    for (int iteration = 0; iteration < warmup; ++iteration)
        for (const auto& worker : workers)
            totals.checksum += worker->Finish(worker->Submit()).score;
    const int iterations = options.score_only ? 1 : options.iterations;
    return workers.size() == 1
               ? MeasureSerial(*workers.front(), iterations, totals)
               : MeasureWorkers(workers, iterations, totals);
}

void PrintRow(
    const BenchOptions& options, const Row& row, const vqm::PsnrHvsScore& score
) {
    if (options.score_only) {
        std::printf("%.8f\n", row.score);
        return;
    }
    std::printf(
        "%s vs %s: decode %.3f ms, core %.6f ms/pair, score %.8f dB "
        "(Y %.8f, Cb %.8f, Cr %.8f)\n",
        row.pair.original.c_str(), row.pair.distorted.c_str(), row.decode_ms,
        row.core_ms, row.score, score.y, score.cb, score.cr
    );
}

}  // namespace

int RunPsnrHvs(const BenchOptions& options) {
    Totals totals;
    std::vector<Row> rows;
    int failures = 0;
    std::optional<vqm::PsnrHvsPipeline> pipeline;
    vqm::Size current_size;
    PairDecoder decoder(options);
    // Batch scoring retains the geometry's allocation, while every call still
    // uploads and prepares the new raw input pair.
    const bool reuse = options.score_only && !options.resident &&
                       !options.resident_reference && !options.vulkan_buffers;
    for (const auto& pair : options.pairs) {
        try {
            DecodedPair images;
            std::string error;
            if (!decoder.Decode(pair, images, error))
                throw std::runtime_error(error);
            ApplyOverrides(options, &images.original, &images.distorted);
            if (images.original.size != images.distorted.size)
                throw std::invalid_argument("Image size mismatch");
            const auto config = ConfigFor(options, images.original.size);
            if (reuse && (!pipeline || current_size != config.size)) {
                pipeline.emplace(config);
                current_size = config.size;
            }
            const auto measured = MeasurePair(
                options, images, reuse ? &*pipeline : nullptr, totals
            );
            rows.push_back(
                {pair, images.width, images.height, images.decode_ms,
                 measured.core_ms, measured.result.score}
            );
            PrintRow(options, rows.back(), measured.result);
        } catch (const std::exception& error) {
            pipeline.reset();
            std::fprintf(
                stderr, "%s: %s\n", pair.original.c_str(), error.what()
            );
            if (options.score_only) std::puts("ERROR");
            ++failures;
        }
    }
    if (rows.empty()) return 1;
    const bool concurrent = !options.score_only && options.workers > 1;
    if (!options.score_only) {
        std::printf(
            "fixed-count wall clock: %.6f ms/pair, %.3f pairs/s\n",
            1000.0 * totals.seconds / totals.calls, totals.calls / totals.seconds
        );
        PrintStages(
            totals.stages, totals.calls, totals.seconds, kFirst, kLast,
            concurrent
        );
        std::printf("checksum %.8f\n", totals.checksum);
    }
    if (!options.csv_path.empty() && !WritePairsCsv(options.csv_path, rows))
        return 1;
    if (!options.stages_csv_path.empty() &&
        !WriteStagesCsv(
            options.stages_csv_path, totals.stages, totals.calls, totals.seconds,
            kFirst, kLast, concurrent
        ))
        return 1;
    return failures == 0 ? 0 : 1;
}

}  // namespace bench_driver
