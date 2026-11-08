// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "bench.h"
#include "butteraugli.h"
#include "inmemory/prefetch.h"

namespace bench_driver {

// The two metrics expose the same Compute shape -- images or resident handles,
// plus the alpha background -- and differ only in the config fields they read
// and the stages they time, so the concurrent paths are written once.
struct Ssimulacra2Metric {
    using Pipeline = vqm::Ssimulacra2Pipeline;
    using Config = vqm::Ssimulacra2Config;
    static constexpr bool kAlternateReference = true;

    static void Configure(const BenchOptions& options, Config& config) {
        config.half_planes = options.half_planes;
        config.chroma_filter = options.input.chroma_filter;
        config.streaming_inputs = options.resident == 0;
        if (options.resident > 0) {
            config.gpu_resident_images = std::max(
                std::size_t{3}, static_cast<std::size_t>(options.resident)
            );
        }
    }
    static double Compute(
        Pipeline& pipeline, const vqm::Image& original,
        const vqm::Image& distorted, float background
    ) {
        return pipeline.Compute(original, distorted, background).Score();
    }
    static double Compute(
        Pipeline& pipeline, const vqm::UploadedImage& original,
        const vqm::UploadedImage& distorted, float background
    ) {
        return pipeline.Compute(original, distorted, background).Score();
    }
    // A transparency pair is scored on a dark and on a bright background and
    // the worse result is reported, which for SSIMULACRA 2 is the smaller.
    static double Worse(double dark, double bright) {
        return std::min(dark, bright);
    }
    static std::string_view Name() { return "ssimulacra2"; }
    static int FirstStage() { return 0; }
    // prepare + scale only: Stage::Ssimulacra2Score brackets the host-side
    // conversion of the scale readback (src/metrics/ssimulacra2/score.cc), which
    // the device does not pay, so it is excluded from the GPU stage time.
    static int LastStage() {
        return static_cast<int>(vqm::internal::Stage::Ssimulacra2Scale);
    }
};

struct ButteraugliMetric {
    using Pipeline = vqm::ButteraugliPipeline;
    using Config = vqm::ButteraugliConfig;
    static constexpr bool kAlternateReference = false;

    static void Configure(const BenchOptions& options, Config& config) {
        config.chroma_filter = options.input.chroma_filter;
        config.hf_asymmetry = options.hf_asymmetry;
        config.intensity_target = options.intensity_target;
        config.streaming_inputs = options.resident == 0;
        if (options.resident > 0) {
            config.gpu_resident_images =
                static_cast<std::size_t>(options.resident);
        }
    }
    static double Compute(
        Pipeline& pipeline, const vqm::Image& original,
        const vqm::Image& distorted, float background
    ) {
        return pipeline.Compute(original, distorted, background).score;
    }
    static double Compute(
        Pipeline& pipeline, const vqm::UploadedImage& original,
        const vqm::UploadedImage& distorted, float background
    ) {
        return pipeline.Compute(original, distorted, background).score;
    }
    // Butteraugli's score is the diffmap maximum, so the worse result is the
    // larger one.
    static double Worse(double dark, double bright) {
        return std::max(dark, bright);
    }
    static std::string_view Name() { return "butteraugli"; }
    static int FirstStage() {
        return static_cast<int>(vqm::internal::Stage::ButteraugliOpsin);
    }
    static int LastStage() {
        return static_cast<int>(vqm::internal::Stage::ButteraugliReduce);
    }
};

// One worker's device state for a concurrent batch or throughput run: its own
// pipeline -- therefore its own arena and its own submission in flight -- and,
// for a resident arm, the pair parked in device memory.
template <typename Ops>
struct WorkerState {
    using Pipeline = typename Ops::Pipeline;

    // (Re)creates the pipeline when the geometry changes: device setup and the
    // arena are per geometry, not per pair.
    Pipeline& Ready(const BenchOptions& options, vqm::Size wanted) {
        if (!pipeline || size != wanted) {
            typename Ops::Config config;
            config.size = wanted;
            Ops::Configure(options, config);
            pipeline.emplace(config);
            size = wanted;
            original.reset();
            alternate_original.reset();
            distorted.reset();
            use_alternate = false;
        }
        return *pipeline;
    }

    // The pair this worker holds resident, if the arm asked for it.
    void Park(Pipeline& target, const DecodedPair& item) {
        original = target.Upload(item.original);
        distorted = target.Upload(item.distorted);
        if constexpr (Ops::kAlternateReference) {
            alternate_original = target.Upload(item.original);
        }
        use_alternate = false;
    }

    const vqm::UploadedImage& NextOriginal() {
        const auto& result = use_alternate ? *alternate_original : *original;
        if (alternate_original) use_alternate = !use_alternate;
        return result;
    }

    std::optional<Pipeline> pipeline;
    vqm::Size size;
    std::optional<vqm::UploadedImage> original;
    std::optional<vqm::UploadedImage> alternate_original;
    std::optional<vqm::UploadedImage> distorted;
    bool use_alternate = false;
};

// One metric call on a worker's pipeline: the resident pair when the arm parks
// one, the decoded images otherwise.
template <typename Ops>
double CallOnce(
    WorkerState<Ops>& worker, const DecodedPair& item, float background
) {
    typename Ops::Pipeline& pipeline = *worker.pipeline;
    if (worker.original) {
        return Ops::Compute(
            pipeline, worker.NextOriginal(), *worker.distorted, background
        );
    }
    return Ops::Compute(pipeline, item.original, item.distorted, background);
}

// Throughput counts a complete CLI-style pair evaluation, including both
// alpha backgrounds. The caller's timer brackets this whole responsibility.
template <typename Ops>
double ScoreWindowPair(
    WorkerState<Ops>& worker, const DecodedPair& item, float background,
    bool has_alpha
) {
    const double dark = CallOnce(worker, item, background);
    return has_alpha ? Ops::Worse(dark, CallOnce(worker, item, 0.9f)) : dark;
}

// `--warmup` + `--iterations` calls and the median duration, with the CLI's
// alpha rule: the run reports the worse of the dark and the bright result.
// Uploads and releases of a resident arm stay outside the timed calls.
template <typename Ops>
void MeasureOnWorker(
    const BenchOptions& options, WorkerState<Ops>& worker,
    const DecodedPair& item, double* score, double* core_ms
) {
    typename Ops::Pipeline& pipeline = worker.Ready(
        options, {static_cast<std::size_t>(item.width),
                  static_cast<std::size_t>(item.height)}
    );
    if (options.resident > 0) worker.Park(pipeline, item);
    const bool has_alpha =
        (item.original.HasAlpha() || item.distorted.HasAlpha());
    const float background = has_alpha ? 0.1f : 0.5f;
    std::vector<double> core_times;
    core_times.reserve(static_cast<std::size_t>(options.iterations));
    double value = 0.0;
    for (int i = 0; i < options.warmup + options.iterations; ++i) {
        const Clock::time_point start = Clock::now();
        value = CallOnce(worker, item, background);
        const double seconds = Seconds(Clock::now() - start);
        if (i >= options.warmup) core_times.push_back(seconds);
    }
    // The CLI reports the worse of the dark and bright backgrounds. Only the
    // dark one is timed: the background is a shader constant, so the call's
    // cost is identical.
    if (has_alpha) value = Ops::Worse(value, CallOnce(worker, item, 0.9f));
    if (worker.original) {
        pipeline.Release(*worker.original);
        pipeline.Release(*worker.distorted);
        if (worker.alternate_original)
            pipeline.Release(*worker.alternate_original);
        worker.original.reset();
        worker.alternate_original.reset();
        worker.distorted.reset();
    }
    *score = value;
    *core_ms = 1000.0 * Median(core_times);
}

// The decode pool's size: --decode-threads when given, every hardware thread
// for a batch or throughput run, and at most four for the latency path (whose
// core_ms is a per-pair latency, not a saturated machine).
int DecodeThreads(const BenchOptions& options, bool batch) {
    if (options.decode_threads > 0) return options.decode_threads;
    const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
    return static_cast<int>(batch ? cores : std::min(4u, cores));
}

// The one-line-per-pair batch contract with `--workers` pipelines in this
// process: each worker owns its pipeline, so two pairs are in flight at once
// and the device is not waiting on the host between submissions.
template <typename Ops>
int RunBatchWorkers(const BenchOptions& options) {
    const int threads = std::max(1, options.workers);
    const int decode_threads = DecodeThreads(options, /*batch=*/true);
    std::vector<WorkerState<Ops>> workers(static_cast<std::size_t>(threads));
    const auto load =
        [&options](const Pair& pair, DecodedPair& item, std::string& message) {
            if (!LoadOnce(pair, item, message)) return false;
            ApplyOverrides(options, &item.original, &item.distorted);
            return true;
        };
    const auto score = [&options, &workers](
                           DecodedPair& item, std::size_t, int worker,
                           double* value, double* core_ms, std::string& message
                       ) {
        try {
            MeasureOnWorker(
                options, workers[static_cast<std::size_t>(worker)], item, value,
                core_ms
            );
            return true;
        } catch (const std::exception& e) {
            message = e.what();
            return false;
        }
    };
    std::vector<inmem::PairResult> results;
    std::string error;
    if (!inmem::RunBatch<DecodedPair>(
            options.pairs, threads, decode_threads, load, score, &results, &error
        )) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::vector<Row> rows;
    rows.reserve(options.pairs.size());
    int failures = 0;
    for (std::size_t index = 0; index < options.pairs.size(); ++index) {
        const inmem::PairResult& result = results[index];
        if (!result.ok) {
            std::fprintf(
                stderr, "%s: %s\n", options.pairs[index].original.c_str(),
                result.error.c_str()
            );
            std::puts("ERROR");
            ++failures;
            continue;
        }
        Row row;
        row.pair = options.pairs[index];
        row.width = result.width;
        row.height = result.height;
        row.decode_ms = result.decode_ms;
        row.core_ms = result.core_ms;
        row.score = result.score;
        rows.push_back(row);
        std::printf("%.8f\n", row.score);
    }
    if (rows.empty()) {
        std::fprintf(stderr, "no pair could be measured\n");
        return 1;
    }
    if (!options.csv_path.empty() && !WritePairsCsv(options.csv_path, rows)) {
        std::fprintf(stderr, "could not write %s\n", options.csv_path.c_str());
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}

// Decode once outside the timer, then keep `--workers` pipelines busy for a
// fixed count or `--throughput-seconds` and print one machine line. With a
// single worker the stage timer is read across the window, which is what makes
// the reciprocal GPU stage rate visible next to the achieved rate; it is a
// process-wide accumulator, so it is only read when there is one worker.
template <typename Ops>
int RunWindow(const BenchOptions& options) {
    const int threads =
        options.throughput_iterations > 0
            ? std::min(
                  std::max(1, options.workers), options.throughput_iterations
              )
            : std::max(1, options.workers);
    const int decode_threads = DecodeThreads(options, /*batch=*/true);
    const int distinct =
        std::min(threads, static_cast<int>(options.pairs.size()));

    std::vector<Pair> head(
        options.pairs.begin(), options.pairs.begin() + distinct
    );
    std::vector<DecodedPair> items(static_cast<std::size_t>(distinct));
    {
        inmem::Prefetch<DecodedPair> prefetch(head, decode_threads, LoadOnce);
        for (int index = 0; index < distinct; ++index) {
            std::size_t taken = 0;
            std::string message;
            if (!prefetch.Take(
                    items[static_cast<std::size_t>(index)], &taken, &message
                )) {
                std::fprintf(
                    stderr, "%s: %s\n",
                    head[static_cast<std::size_t>(index)].original.c_str(),
                    message.empty() ? "could not decode the pair"
                                    : message.c_str()
                );
                return 1;
            }
            DecodedPair& item = items[static_cast<std::size_t>(index)];
            ApplyOverrides(options, &item.original, &item.distorted);
        }
    }
    for (const DecodedPair& item : items) {
        if (item.width != items.front().width ||
            item.height != items.front().height) {
            std::fprintf(
                stderr, "throughput inputs must all have the same geometry\n"
            );
            return 1;
        }
    }

    // Device setup first, work second: every pipeline and (for a resident arm)
    // every upload happens before the first warm-up call, so the arenas are
    // allocated while the device is still empty. Interleaving them costs the
    // last pipeline its host-visible arena on a device whose resizable BAR heap
    // is nearly full, which is a slow fallback rather than a saving.
    std::vector<WorkerState<Ops>> workers(static_cast<std::size_t>(threads));
    for (int worker = 0; worker < threads; ++worker) {
        const DecodedPair& item =
            items[static_cast<std::size_t>(worker) % items.size()];
        typename Ops::Pipeline& pipeline =
            workers[static_cast<std::size_t>(worker)].Ready(
                options, {static_cast<std::size_t>(item.width),
                          static_cast<std::size_t>(item.height)}
            );
        if (options.resident > 0) {
            workers[static_cast<std::size_t>(worker)].Park(pipeline, item);
        }
    }
    // Warmup calls per pipeline, outside the measured interval.
    for (int worker = 0; worker < threads; ++worker) {
        const DecodedPair& item =
            items[static_cast<std::size_t>(worker) % items.size()];
        const int warmup =
            options.throughput_iterations > 0 ? options.warmup : 1;
        for (int iteration = 0; iteration < warmup; ++iteration)
            CallOnce(
                workers[static_cast<std::size_t>(worker)], item,
                (item.original.HasAlpha() || item.distorted.HasAlpha()) ? 0.1f
                                                                        : 0.5f
            );
    }

    std::atomic<bool> started{false};
    std::atomic<bool> stop{false};
    std::atomic<long long> start_ticks{0};
    std::mutex failure_mutex;
    std::string failure;
    std::vector<long long> calls(static_cast<std::size_t>(threads), 0);
    std::vector<double> busy(static_cast<std::size_t>(threads), 0.0);
    std::vector<double> checksums(static_cast<std::size_t>(threads), 0.0);

    const auto fail = [&](std::string message) {
        {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (failure.empty()) failure = std::move(message);
        }
        stop.store(true, std::memory_order_release);
    };

    const auto body = [&](int worker) {
        DecodedPair& item =
            items[static_cast<std::size_t>(worker) % items.size()];
        WorkerState<Ops>& state = workers[static_cast<std::size_t>(worker)];
        const bool has_alpha =
            (item.original.HasAlpha() || item.distorted.HasAlpha());
        const float background = has_alpha ? 0.1f : 0.5f;
        while (!started.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        const Clock::time_point window_start{
            Clock::duration(start_ticks.load(std::memory_order_acquire))};
        const Clock::time_point deadline =
            window_start +
            std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(options.throughput_seconds)
            );
        long long count = 0;
        const int quota = options.throughput_iterations / threads +
                          (worker < options.throughput_iterations % threads);
        double busy_here = 0.0;
        double sum = 0.0;
        while (!stop.load(std::memory_order_relaxed)) {
            const Clock::time_point begin = Clock::now();
            if (options.throughput_iterations > 0 ? count >= quota
                                                  : begin >= deadline)
                break;
            double value = 0.0;
            try {
                value = ScoreWindowPair(state, item, background, has_alpha);
            } catch (const std::exception& e) {
                fail(e.what());
                break;
            }
            busy_here += Seconds(Clock::now() - begin);
            sum += value;
            ++count;
        }
        calls[static_cast<std::size_t>(worker)] = count;
        busy[static_cast<std::size_t>(worker)] = busy_here;
        checksums[static_cast<std::size_t>(worker)] = sum;
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(threads));
    const bool timed_stages = threads == 1;
    if (timed_stages) vqm::internal::ResetTimings();
    for (int worker = 0; worker < threads; ++worker) {
        pool.emplace_back(body, worker);
    }
    // Published only now, so every worker's warmup call is outside the window.
    const Clock::time_point window_start = Clock::now();
    start_ticks.store(window_start.time_since_epoch().count());
    started.store(true, std::memory_order_release);
    for (std::thread& thread : pool) thread.join();
    const Clock::time_point window_end = Clock::now();

    inmem::ThroughputStats stats;
    stats.width = items.front().width;
    stats.height = items.front().height;
    stats.threads = threads;
    stats.seconds = Seconds(window_end - window_start);
    for (std::size_t worker = 0; worker < static_cast<std::size_t>(threads);
         ++worker) {
        stats.calls += calls[worker];
        stats.busy_seconds += busy[worker];
        stats.checksum += checksums[worker];
    }
    stats.fps = stats.seconds > 0.0
                    ? static_cast<double>(stats.calls) / stats.seconds
                    : 0.0;
    std::optional<double> gpu_ms;
    if (timed_stages && stats.calls > 0) {
        double stage_seconds = 0.0;
        for (int stage = Ops::FirstStage(); stage <= Ops::LastStage(); ++stage) {
            stage_seconds += vqm::internal::Timings()
                                 .seconds[static_cast<std::size_t>(stage)];
        }
        gpu_ms = 1000.0 * stage_seconds / static_cast<double>(stats.calls);
    }
    if (!failure.empty()) {
        std::fprintf(stderr, "%s\n", failure.c_str());
        return 1;
    }
    if (stats.calls == 0) {
        std::fprintf(
            stderr, "no metric call completed inside the throughput window\n"
        );
        return 1;
    }
    inmem::EmitThroughput(stats, Ops::Name(), gpu_ms);
    return 0;
}

int RunWindowForMetric(const BenchOptions& options) {
    return (options.metric == Metric::Butteraugli)
               ? RunWindow<ButteraugliMetric>(options)
               : RunWindow<Ssimulacra2Metric>(options);
}

int RunBatchWorkersForMetric(const BenchOptions& options) {
    return (options.metric == Metric::Butteraugli)
               ? RunBatchWorkers<ButteraugliMetric>(options)
               : RunBatchWorkers<Ssimulacra2Metric>(options);
}

}  // namespace bench_driver
