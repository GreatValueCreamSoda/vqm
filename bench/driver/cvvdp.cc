// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bench.h"
#include "cvvdp.h"
#include "inmemory/vulkan_inputs.h"

namespace bench_driver {
namespace {

constexpr int kFirst = static_cast<int>(vqm::internal::Stage::CvvdpPrepare);
constexpr int kLast = static_cast<int>(vqm::internal::Stage::CvvdpScore);

struct Totals {
    std::array<double, vqm::internal::kStageCount> stages{};
    double checksum = 0.0;
    double seconds = 0.0;
    double frames = 0.0;

    void Accumulate(double elapsed, std::uint64_t frame_count) {
        seconds += elapsed;
        frames += static_cast<double>(frame_count);
        for (int stage = kFirst; stage <= kLast; ++stage) {
            const auto index = static_cast<std::size_t>(stage);
            stages[index] += vqm::internal::Timings().seconds[index];
        }
    }
};

vqm::CvvdpConfig ConfigFor(
    const BenchOptions& options, vqm::Size size,
    vqm::image_io::FrameRate fps = {}
) {
    vqm::CvvdpConfig config;
    config.size = size;
    config.display = options.cvvdp.ResolveDisplay(size);
    config.temporal = options.cvvdp_video;
    config.chroma_filter = options.input.chroma_filter;
    config.streaming_inputs = options.resident == 0 && !options.vulkan_buffers;
    config.gpu_resident_images = static_cast<std::size_t>(options.resident);
    if (options.cvvdp_video) {
        config.fps_numerator = fps.numerator;
        config.fps_denominator = fps.denominator;
    }
    return config;
}

void PrintConfiguration(
    const vqm::CvvdpConfig& config, const vqm::PipelineMemoryUsage& memory
) {
    std::printf(
        "CVVDP v0.5.7 base, %s, ppd %.9g, peak %.9g nit, contrast %.9g, ambient "
        "%.9g lux\n",
        config.temporal ? "ordered video" : "independent images",
        config.display.pixels_per_degree, config.display.peak_luminance,
        config.display.contrast, config.display.ambient_illuminance
    );
    if (config.temporal)
        std::printf(
            "frame rate %u/%u\n", config.fps_numerator, config.fps_denominator
        );
    std::uint64_t allocation_bytes =
        memory.arena.allocation_bytes + memory.staging.allocation_bytes;
    for (const auto& allocation : memory.auxiliary)
        allocation_bytes += allocation.allocation_bytes;
    std::printf(
        "owned buffer allocations %llu bytes (includes temporal history)\n",
        static_cast<unsigned long long>(allocation_bytes)
    );
}

struct StillMeasurement {
    double core_ms;
    double score;
};

StillMeasurement MeasureStill(
    const BenchOptions& options, vqm::CvvdpPipeline& pipeline,
    const vqm::Image& original, const vqm::Image& distorted, Totals& totals
) {
    vqm::UploadedImage ref, test;
    if (options.resident) {
        ref = pipeline.Upload(original);
        test = pipeline.Upload(distorted);
    }
    const int warmup = options.score_only ? 0 : options.warmup;
    const int iterations = options.score_only ? 1 : options.iterations;
    std::vector<double> samples;
    double score = 0.0;
    for (int iteration = 0; iteration < warmup + iterations; ++iteration) {
        pipeline.Reset();
        vqm::internal::ResetTimings();
        const auto start = Clock::now();
        const auto result = options.resident
                                ? pipeline.ComputeImage(ref, test)
                                : pipeline.ComputeImage(original, distorted);
        const double elapsed = Seconds(Clock::now() - start);
        score = result.score;
        totals.checksum += score;
        if (iteration < warmup) continue;
        samples.push_back(1000.0 * elapsed);
        totals.Accumulate(elapsed, result.frames);
    }
    if (options.resident) {
        pipeline.Release(ref);
        pipeline.Release(test);
    }
    return {Median(samples), score};
}

int RunStill(const BenchOptions& options) {
    Totals totals;
    std::vector<Row> rows;
    int failures = 0;
    std::optional<vqm::CvvdpPipeline> pipeline;
    vqm::Size current_size;
    PairDecoder decoder(options);
    for (const auto& pair : options.pairs) {
        try {
            DecodedPair decoded;
            std::string error;
            if (!decoder.Decode(pair, decoded, error))
                throw std::runtime_error(error);
            ApplyOverrides(options, &decoded.original, &decoded.distorted);
            if (decoded.original.size != decoded.distorted.size)
                throw std::invalid_argument("Image size mismatch");
            const auto config = ConfigFor(options, decoded.original.size);
            if (!pipeline || current_size != config.size) {
                pipeline.emplace(config);
                current_size = config.size;
                if (!options.score_only)
                    PrintConfiguration(config, pipeline->GetMemoryUsage());
            }
            const auto measurement = MeasureStill(
                options, *pipeline, decoded.original, decoded.distorted, totals
            );
            rows.push_back(
                {pair, decoded.width, decoded.height, decoded.decode_ms,
                 measurement.core_ms, measurement.score}
            );
            if (options.score_only)
                std::printf("%.8f\n", measurement.score);
            else
                std::printf(
                    "%s vs %s: decode %.3f ms, core %.6f ms/image, score %.8f\n",
                    pair.original.c_str(), pair.distorted.c_str(),
                    decoded.decode_ms, measurement.core_ms, measurement.score
                );
        } catch (const std::exception& exception) {
            pipeline.reset();
            std::fprintf(
                stderr, "%s: %s\n", pair.original.c_str(), exception.what()
            );
            if (options.score_only) std::puts("ERROR");
            ++failures;
        }
    }
    if (rows.empty()) return 1;
    if (!options.score_only) {
        std::printf(
            "fixed-count wall clock: %.6f ms/image, %.3f images/s\n",
            1000.0 * totals.seconds / totals.frames,
            totals.frames / totals.seconds
        );
        PrintStages(
            totals.stages, totals.frames, totals.seconds, kFirst, kLast, false
        );
        std::printf("checksum %.8f\n", totals.checksum);
    }
    if (!options.csv_path.empty() && !WritePairsCsv(options.csv_path, rows))
        return 1;
    if (!options.stages_csv_path.empty() &&
        !WriteStagesCsv(
            options.stages_csv_path, totals.stages, totals.frames,
            totals.seconds, kFirst, kLast, false
        ))
        return 1;
    return failures == 0 ? 0 : 1;
}

struct ClipMeasurement {
    vqm::CvvdpScore result;
    double core_seconds = 0.0;
    double decode_seconds = 0.0;
};

using ResidentClip = std::vector<std::unique_ptr<inmem::VulkanInputs>>;

ResidentClip UploadClip(const BenchOptions& options, const Pair& pair) {
    using vqm::image_io::SequenceReadResult;
    vqm::image_io::SequenceReader original(pair.original);
    vqm::image_io::SequenceReader distorted(pair.distorted);
    ResidentClip frames;
    vqm::Image ref, test;
    std::string error;
    for (;;) {
        const auto first = original.Next(ref, error);
        if (first == SequenceReadResult::Error)
            throw std::runtime_error(pair.original + ": " + error);
        const auto second = distorted.Next(test, error);
        if (second == SequenceReadResult::Error)
            throw std::runtime_error(pair.distorted + ": " + error);
        if (first != second)
            throw std::invalid_argument(
                "CVVDP inputs have unequal frame counts"
            );
        if (first == SequenceReadResult::End) break;
        ApplyOverrides(options, &ref, &test);
        frames.push_back(std::make_unique<inmem::VulkanInputs>(ref, test));
    }
    if (frames.empty())
        throw std::invalid_argument("CVVDP input sequence has no frames");
    return frames;
}

ClipMeasurement MeasureResidentClip(
    const ResidentClip& frames, vqm::CvvdpPipeline& pipeline
) {
    pipeline.Reset();
    vqm::internal::ResetTimings();
    const auto start = Clock::now();
    for (const auto& frame : frames)
        pipeline.Push(frame->original(), frame->distorted());
    const auto result = pipeline.Finalize();
    const double elapsed = Seconds(Clock::now() - start);
    if (!result)
        throw std::invalid_argument("CVVDP input sequence has no frames");
    return {*result, elapsed, 0.0};
}

// Each call reopens and consumes a real clip. Only Push/Finalize are core time;
// header parsing, file I/O, frame decoding and Reset are outside that interval.
ClipMeasurement MeasureClip(
    const BenchOptions& options, const Pair& pair, vqm::CvvdpPipeline& pipeline,
    vqm::image_io::FrameRate expected_fps
) {
    using vqm::image_io::SequenceReadResult;
    vqm::image_io::SequenceReader original(pair.original);
    vqm::image_io::SequenceReader distorted(pair.distorted);
    if (original.frame_rate() != expected_fps ||
        distorted.frame_rate() != expected_fps)
        throw std::invalid_argument(
            "CVVDP inputs have unequal or changed frame rates"
        );
    pipeline.Reset();
    vqm::internal::ResetTimings();
    ClipMeasurement measured;
    vqm::Image ref, test;
    std::string error;
    for (;;) {
        const auto decode_start = Clock::now();
        const auto a = original.Next(ref, error);
        if (a == SequenceReadResult::Error)
            throw std::runtime_error(pair.original + ": " + error);
        const auto b = distorted.Next(test, error);
        if (b == SequenceReadResult::Error)
            throw std::runtime_error(pair.distorted + ": " + error);
        if (a != b)
            throw std::invalid_argument(
                "CVVDP inputs have unequal frame counts"
            );
        if (a == SequenceReadResult::End) break;
        ApplyOverrides(options, &ref, &test);
        measured.decode_seconds += Seconds(Clock::now() - decode_start);
        const auto core_start = Clock::now();
        pipeline.Push(ref, test);
        measured.core_seconds += Seconds(Clock::now() - core_start);
    }
    const auto core_start = Clock::now();
    const auto result = pipeline.Finalize();
    measured.core_seconds += Seconds(Clock::now() - core_start);
    if (!result)
        throw std::invalid_argument("CVVDP input sequence has no frames");
    measured.result = *result;
    return measured;
}

int RunVideo(const BenchOptions& options) {
    Totals totals;
    int failures = 0;
    std::ofstream csv;
    if (!options.csv_path.empty()) {
        csv.open(options.csv_path);
        if (!csv) return 1;
        csv << "original,distorted,width,height,fps_numerator,fps_denominator,"
               "frames,decode_ms_per_clip,core_ms_per_clip,core_ms_per_frame,"
               "jod\n";
    }
    for (const auto& pair : options.pairs) {
        try {
            vqm::image_io::SequenceReader original(pair.original);
            vqm::image_io::SequenceReader distorted(pair.distorted);
            if (!original.is_video() || !distorted.is_video())
                throw std::invalid_argument(
                    "--cvvdp-video requires two progressive Y4M inputs"
                );
            if (original.size() != distorted.size())
                throw std::invalid_argument("Image size mismatch");
            if (original.frame_rate() != distorted.frame_rate())
                throw std::invalid_argument(
                    "CVVDP inputs have unequal frame rates"
                );
            const auto fps = original.frame_rate();
            const auto config = ConfigFor(options, original.size(), fps);
            const auto resident = options.vulkan_buffers
                                      ? UploadClip(options, pair)
                                      : ResidentClip{};
            auto pipeline =
                resident.empty()
                    ? vqm::CvvdpPipeline(config)
                    : vqm::CvvdpPipeline(config, resident.front()->device());
            if (!options.score_only)
                PrintConfiguration(config, pipeline.GetMemoryUsage());
            if (!options.score_only && options.vulkan_buffers)
                std::printf(
                    "preuploaded external inputs: %zu frame pairs "
                    "(outside pipeline allocations)\n",
                    resident.size()
                );
            const int warmup = options.score_only ? 0 : options.warmup;
            const int iterations = options.score_only ? 1 : options.iterations;
            std::vector<double> cores, decodes;
            vqm::CvvdpScore result;
            for (int iteration = 0; iteration < warmup + iterations;
                 ++iteration) {
                const auto measured =
                    options.vulkan_buffers
                        ? MeasureResidentClip(resident, pipeline)
                        : MeasureClip(options, pair, pipeline, fps);
                result = measured.result;
                totals.checksum += result.score;
                if (iteration < warmup) continue;
                cores.push_back(1000.0 * measured.core_seconds);
                decodes.push_back(1000.0 * measured.decode_seconds);
                totals.Accumulate(measured.core_seconds, result.frames);
            }
            const double core_ms = Median(cores);
            const double per_frame =
                core_ms / static_cast<double>(result.frames);
            if (options.score_only)
                std::printf("%.8f\n", result.score);
            else
                std::printf(
                    "%s vs %s: %llu frames, decode %.3f ms/clip, core %.6f "
                    "ms/clip, %.6f ms/frame, score %.8f\n",
                    pair.original.c_str(), pair.distorted.c_str(),
                    static_cast<unsigned long long>(result.frames),
                    Median(decodes), core_ms, per_frame, result.score
                );
            if (csv.is_open())
                csv << std::format(
                    "{},{},{},{},{},{},{},{:.6f},{:.6f},{:.6f},{:.8f}\n",
                    inmem::CsvField(pair.original),
                    inmem::CsvField(pair.distorted), original.size().width,
                    original.size().height, fps.numerator, fps.denominator,
                    result.frames, Median(decodes), core_ms, per_frame,
                    result.score
                );
        } catch (const std::exception& exception) {
            std::fprintf(
                stderr, "%s: %s\n", pair.original.c_str(), exception.what()
            );
            if (options.score_only) std::puts("ERROR");
            ++failures;
        }
    }
    if (totals.frames == 0.0) return 1;
    if (!options.score_only) {
        std::printf(
            "fixed-count wall clock: %.6f ms/frame, %.3f frames/s\n",
            1000.0 * totals.seconds / totals.frames,
            totals.frames / totals.seconds
        );
        std::puts(
            "stage timing per frame over complete reset clips (decode excluded):"
        );
        PrintStages(
            totals.stages, totals.frames, totals.seconds, kFirst, kLast, false
        );
        std::printf("checksum %.8f\n", totals.checksum);
    }
    if (!options.stages_csv_path.empty() &&
        !WriteStagesCsv(
            options.stages_csv_path, totals.stages, totals.frames,
            totals.seconds, kFirst, kLast, false
        ))
        return 1;
    if (csv.is_open()) {
        csv.close();
        if (!csv) return 1;
    }
    return failures == 0 ? 0 : 1;
}

}  // namespace

int RunCvvdp(const BenchOptions& options) {
    return options.cvvdp_video ? RunVideo(options) : RunStill(options);
}

}  // namespace bench_driver
