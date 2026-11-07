// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// Shared plumbing for the in-memory reference harnesses: argument parsing,
// the pair list, the timing rule (decode once outside the timer, median of the
// timed metric calls) and the CSV the benchmark runner reads.
//
// Header-only and free of both Vulkan and metric code, so it can be compiled
// against any reference tree with nothing but a C++20 compiler.

#ifndef VQM_BENCH_INMEMORY_DRIVER_H_
#define VQM_BENCH_INMEMORY_DRIVER_H_

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <format>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include "pairs.h"
#include "prefetch.h"

namespace inmem {

using Clock = std::chrono::steady_clock;

inline double Seconds(Clock::duration duration) {
    return std::chrono::duration<double>(duration).count();
}

// The deadline is added to Clock::now(); reject non-finite and unrepresentable
// durations before converting to clock ticks or constructing that deadline.
inline bool ValidWindowSeconds(double seconds) {
    const double remaining = Seconds(Clock::time_point::max() - Clock::now());
    return std::isfinite(seconds) && seconds > 0.0 && seconds < remaining - 1.0;
}

struct Options {
    std::vector<Pair> pairs;
    std::string pairs_file;
    std::string csv_path;
    std::string metric = "ssimu2";
    int iterations = 20;
    int warmup = 3;
    int limit = 0;
    // One line per pair on stdout -- the score, or ERROR -- in input order,
    // and nothing else. Batch scorers read this; reasons go to stderr.
    bool score_only = false;
    // In-process concurrency. 0 means the mode default (ResolveThreads and
    // ResolveDecodeThreads): a latency run stays one worker and four decode
    // threads whatever the pair count, so its per-pair core_ms is not measured
    // under contention; a batch or throughput run uses every hardware thread.
    int threads = 0;
    int decode_threads = 0;
    // Fixed-length throughput window. 0 runs the latency protocol; a positive
    // value runs RunThroughput instead and prints the machine line only.
    double throughput_seconds = 0.0;
    bool help = false;
};

// The pair count does not decide the defaults: a latency run is one worker and
// four decode threads even over 4292 pairs, because its core_ms is quoted as a
// per-pair latency. A batch (--score-only) or throughput run is a throughput
// measurement and takes the whole machine. Any options struct with a `threads`
// and a `decode_threads` field works, which is how the Vship driver -- whose
// options are its own -- shares the rule.
template <typename OptionsT>
inline int ResolveThreads(const OptionsT& options, bool batch) {
    if (options.threads > 0) return options.threads;
    if (!batch) return 1;
    const unsigned cores = std::thread::hardware_concurrency();
    return cores > 0 ? static_cast<int>(cores) : 1;
}

template <typename OptionsT>
inline int ResolveDecodeThreads(const OptionsT& options, bool batch) {
    if (options.decode_threads > 0) return options.decode_threads;
    if (!batch) return 4;
    const unsigned cores = std::thread::hardware_concurrency();
    return cores > 0 ? static_cast<int>(cores) : 4;
}

inline bool ParseInt(std::string_view text, int& out) {
    const char* first = text.data();
    const char* last = text.data() + text.size();
    const std::from_chars_result result = std::from_chars(first, last, out);
    return result.ec == std::errc{} && result.ptr == last;
}

inline bool ParseArgs(
    std::span<const std::string_view> args, Options& options, std::string* error
) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        const bool has_value = i + 1 < args.size();
        if (arg == "--help" || arg == "-h") {
            options.help = true;
            return true;
        }
        if (arg == "--score-only") {
            options.score_only = true;
            continue;
        }
        if (arg == "--pair") {
            if (i + 2 >= args.size()) {
                *error = "--pair needs two paths";
                return false;
            }
            options.pairs.push_back(
                Pair{std::string(args[i + 1]), std::string(args[i + 2])}
            );
            i += 2;
            continue;
        }
        if (!has_value) {
            *error = std::format("{} needs a value", arg);
            return false;
        }
        if (arg == "--pairs-file") {
            options.pairs_file = std::string(args[++i]);
        } else if (arg == "--csv") {
            options.csv_path = std::string(args[++i]);
        } else if (arg == "--metric") {
            options.metric = std::string(args[++i]);
            if (options.metric != "ssimu2" && options.metric != "butteraugli") {
                *error = std::format("bad --metric '{}'", options.metric);
                return false;
            }
        } else if (arg == "--iterations" || arg == "--warmup" ||
                   arg == "--limit") {
            const std::string_view text = args[++i];
            int value = 0;
            if (!ParseInt(text, value) || value < 0) {
                *error = std::format("bad value for {}", arg);
                return false;
            }
            if (arg == "--iterations")
                options.iterations = value;
            else if (arg == "--warmup")
                options.warmup = value;
            else
                options.limit = value;
        } else if (arg == "--threads" || arg == "--decode-threads") {
            const std::string_view text = args[++i];
            int value = 0;
            if (!ParseInt(text, value) || value < 1) {
                *error =
                    std::format("bad value for {} (expected at least 1)", arg);
                return false;
            }
            if (arg == "--threads")
                options.threads = value;
            else
                options.decode_threads = value;
        } else if (arg == "--throughput-seconds") {
            const std::string_view text = args[++i];
            double value = 0.0;
            const char* last = text.data() + text.size();
            const std::from_chars_result result =
                std::from_chars(text.data(), last, value);
            if (result.ec != std::errc{} || result.ptr != last ||
                !ValidWindowSeconds(value)) {
                *error = std::format(
                    "bad value for --throughput-seconds (expected a positive, "
                    "finite, representable number of seconds)"
                );
                return false;
            }
            options.throughput_seconds = value;
        } else {
            *error = std::format("unknown option '{}'", arg);
            return false;
        }
    }
    if (options.iterations < 1) {
        *error = "--iterations must be at least 1";
        return false;
    }
    if (options.warmup > std::numeric_limits<int>::max() - options.iterations) {
        *error = "--warmup plus --iterations is too large";
        return false;
    }
    return true;
}

// Loads --pairs-file (after any --pair entries) and applies --limit.
inline bool ResolvePairs(Options& options, std::string* error) {
    if (!options.pairs_file.empty()) {
        if (!ReadPairsFile(options.pairs_file, options.pairs, error))
            return false;
    }
    if (options.limit > 0 &&
        options.pairs.size() > static_cast<std::size_t>(options.limit)) {
        options.pairs.resize(static_cast<std::size_t>(options.limit));
    }
    return true;
}

inline double Median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const std::size_t middle = values.size() / 2;
    const std::ptrdiff_t index = static_cast<std::ptrdiff_t>(middle);
    std::nth_element(values.begin(), values.begin() + index, values.end());
    double median = values[middle];
    if (values.size() % 2 == 0) {
        std::nth_element(
            values.begin(), values.begin() + index - 1, values.begin() + index
        );
        median = 0.5 * (median + values[middle - 1]);
    }
    return median;
}

// Times three decodes and returns the median; the loader keeps the images
// from its last successful call. Decoding never enters the metric timer.
template <typename Loader>
bool TimedDecode(
    int attempts, Loader&& loader, double* decode_ms, std::string* error
) {
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(attempts));
    for (int attempt = 0; attempt < attempts; ++attempt) {
        const Clock::time_point start = Clock::now();
        const bool ok = loader(error);
        times.push_back(Seconds(Clock::now() - start));
        if (!ok) return false;
    }
    *decode_ms = 1000.0 * Median(times);
    return true;
}

inline std::string FormatScore(double value) {
    return std::format("{:.6f}", value);
}

// Preserve path fields containing delimiters, quotes, or newlines for CSV
// readers such as Python's csv.DictReader.
inline std::string CsvField(std::string_view value) {
    if (value.find_first_of(",\"\r\n") == std::string_view::npos) {
        return std::string(value);
    }
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (const char c : value) {
        if (c == '"') out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

struct Row {
    std::string original;
    std::string distorted;
    int width = 0;
    int height = 0;
    double decode_ms = 0.0;
    double core_ms = 0.0;
    double score = 0.0;
};

inline bool WriteCsv(const std::string& path, const std::vector<Row>& rows) {
    std::ofstream file(path);
    if (!file) return false;
    file << "original,distorted,width,height,decode_ms,core_ms,score\n";
    for (const Row& row : rows) {
        file << std::format(
            "{},{},{},{},{:.6f},{:.6f},{:.6f}\n", CsvField(row.original),
            CsvField(row.distorted), row.width, row.height, row.decode_ms,
            row.core_ms, row.score
        );
    }
    file.close();
    return static_cast<bool>(file);
}

// The batch contract: exactly one stdout line per pair, in input order. The
// score line is the value alone; a failed pair prints the literal ERROR and
// its reason goes to stderr.
inline void EmitPairScore(bool score_only, double score) {
    if (score_only) std::printf("%.8f\n", score);
}

inline void EmitPairError(bool score_only) {
    if (score_only) std::puts("ERROR");
}

// One pair's outcome in a batch run. The driver emits these in input order
// after the join, so the one-line-per-pair contract holds without a reorder
// thread; a failed pair keeps its message instead of a score.
struct PairResult {
    int width = 0;
    int height = 0;
    double decode_ms = 0.0;
    double core_ms = 0.0;
    double score = 0.0;
    std::string error;
    bool ok = false;
};

struct ThroughputStats {
    int width = 0;
    int height = 0;
    int threads = 0;
    long long calls = 0;
    double seconds = 0.0;
    double fps = 0.0;
    double checksum = 0.0;
    // Sum of the per-call metric durations over every worker, so the share of
    // the window the metric actually occupied is visible next to fps.
    double busy_seconds = 0.0;
};

// Decode ahead and distribute result slots across independent score workers.
template <typename Item, typename Loader, typename Attempt>
void RunPrefetchedBatch(
    const std::vector<Pair>& pairs, int threads, int decode_threads,
    Loader& load, const Attempt& attempt, std::vector<PairResult>* results
) {
    Prefetch<Item> prefetch(pairs, decode_threads, load);
    const auto body = [&](int worker) {
        Item item{};
        while (true) {
            std::size_t index = 0;
            std::string message;
            const bool delivered = prefetch.Take(item, &index, &message);
            if (!delivered) {
                // An empty error is the end of the list; anything else is
                // one pair that could not be decoded, still in its place.
                if (message.empty()) return;
                (*results)[index].error = message;
                continue;
            }
            attempt(item, index, worker);
        }
    };
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (int worker = 0; worker < threads; ++worker) {
        workers.emplace_back(body, worker);
    }
    for (std::thread& worker : workers) worker.join();
}

// Scores a list in input order with `threads` workers in one process.
//
// `load` is bool(const Pair&, Item&, std::string&) and runs on the decode
// threads, so it must be thread-safe; it fills Item::decode_ms and the item's
// width/height. `score` is bool(Item&, index, worker, score*, core_ms*,
// error) and runs on a worker; it may index per-worker state by `worker`.
// Item is named at the call site (`RunBatch<Item>(...)`).
//
// One worker and one decode thread load and score inline in input order. Every
// other shape decodes ahead through Prefetch and spawns the workers, each
// taking the next pair in input order and writing its own result slot.
template <typename Item, typename Loader, typename Scorer>
bool RunBatch(
    const std::vector<Pair>& pairs, int threads, int decode_threads, Loader load,
    Scorer score, std::vector<PairResult>* results, std::string* error
) {
    results->assign(pairs.size(), PairResult{});
    if (pairs.empty()) {
        *error = "no pair to score";
        return false;
    }
    threads = std::clamp(threads, 1, static_cast<int>(pairs.size()));
    decode_threads = std::max(1, decode_threads);

    const auto attempt = [&](Item& item, std::size_t index, int worker) {
        PairResult& result = (*results)[index];
        result.width = static_cast<int>(item.width);
        result.height = static_cast<int>(item.height);
        result.decode_ms = item.decode_ms;
        std::string message;
        double value = 0.0;
        double core_ms = 0.0;
        try {
            if (score(item, index, worker, &value, &core_ms, message)) {
                result.score = value;
                result.core_ms = core_ms;
                result.ok = true;
                return;
            }
            if (message.empty()) message = "the metric reported no score";
        } catch (const std::exception& e) {
            message = e.what();
        }
        result.error = message;
    };

    if (threads == 1 && decode_threads == 1) {
        for (std::size_t index = 0; index < pairs.size(); ++index) {
            Item item{};
            std::string message;
            bool loaded = false;
            try {
                loaded = load(pairs[index], item, message);
            } catch (const std::exception& e) {
                message = e.what();
            }
            if (!loaded) {
                (*results)[index].error =
                    message.empty() ? "could not decode the pair" : message;
                continue;
            }
            attempt(item, index, 0);
        }
    } else {
        RunPrefetchedBatch<Item>(
            pairs, threads, decode_threads, load, attempt, results
        );
    }

    long long scored = 0;
    for (const PairResult& result : *results) {
        if (result.ok) ++scored;
    }
    if (scored == 0) {
        *error = "no pair could be scored";
        return false;
    }
    return true;
}

// Runs a fixed-length throughput window: the head of the list is decoded
// before the timer, every worker warms the metric up once (uncounted), and the
// workers then call it back to back until the shared deadline. The window
// length, not the pair count, decides when it ends, so a single pair can
// saturate a device for the whole measurement.
template <typename Item, typename Loader, typename Scorer>
bool RunThroughput(
    const std::vector<Pair>& pairs, int threads, int decode_threads,
    double seconds, Loader load, Scorer score, ThroughputStats* stats,
    std::string* error
) {
    *stats = ThroughputStats{};
    if (pairs.empty()) {
        *error = "no pair to score";
        return false;
    }
    if (!ValidWindowSeconds(seconds)) {
        *error =
            "the throughput window must be a positive, finite, "
            "representable number of seconds";
        return false;
    }
    threads = std::max(1, threads);
    decode_threads = std::max(1, decode_threads);
    const int distinct = std::min(threads, static_cast<int>(pairs.size()));

    // Decode the head of the list before the timer. Workers share an item when
    // there are fewer pairs than workers: the metric only reads it.
    std::vector<Pair> head(pairs.begin(), pairs.begin() + distinct);
    std::vector<Item> items(static_cast<std::size_t>(distinct));
    {
        Prefetch<Item> prefetch(head, decode_threads, load);
        for (int index = 0; index < distinct; ++index) {
            std::size_t taken = 0;
            std::string message;
            const std::size_t slot = static_cast<std::size_t>(index);
            if (!prefetch.Take(items[slot], &taken, &message)) {
                *error =
                    message.empty()
                        ? std::format("could not decode {}", head[slot].original)
                        : message;
                return false;
            }
        }
    }
    stats->width = static_cast<int>(items.front().width);
    stats->height = static_cast<int>(items.front().height);
    stats->threads = threads;
    for (const Item& item : items) {
        if (item.width != items.front().width ||
            item.height != items.front().height) {
            *error = "throughput inputs must all have the same geometry";
            return false;
        }
    }

    std::atomic<bool> started{false};
    std::atomic<bool> stop{false};
    std::atomic<long long> start_ticks{0};
    // Workers that finished their warmup call. The window opens only when all
    // of them have: a warmup on a slow call (a CPU reference at 4K takes over
    // a second) would otherwise run inside the window and spend its whole
    // length, which reported no call at all.
    std::atomic<int> warmed{0};
    std::mutex failure_mutex;
    std::string failure;
    std::vector<long long> calls(static_cast<std::size_t>(threads), 0);
    std::vector<double> busy(static_cast<std::size_t>(threads), 0.0);
    std::vector<double> checksums(static_cast<std::size_t>(threads), 0.0);

    const auto fail = [&](std::string message) {
        if (message.empty()) message = "the metric reported no score";
        {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (failure.empty()) failure = std::move(message);
        }
        stop.store(true, std::memory_order_release);
    };

    const auto attempt = [&](Item& item, std::size_t slot, int worker,
                             double* value, double* core_ms,
                             std::string& message) {
        try {
            if (score(item, slot, worker, value, core_ms, message)) return true;
            fail(message);
        } catch (const std::exception& e) {
            fail(e.what());
        }
        return false;
    };

    const auto body = [&](int worker) {
        const std::size_t slot = static_cast<std::size_t>(worker) %
                                 static_cast<std::size_t>(distinct);
        Item& item = items[slot];
        std::string message;
        double warm_score = 0.0;
        double warm_ms = 0.0;
        if (!attempt(item, slot, worker, &warm_score, &warm_ms, message)) {
            warmed.fetch_add(1, std::memory_order_release);
            return;
        }
        warmed.fetch_add(1, std::memory_order_release);
        // Sleep rather than spin: the wait is one metric call long, and a
        // busy loop on every core would slow the warmups it is waiting for.
        while (!started.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        const Clock::time_point window_start{
            Clock::duration(start_ticks.load(std::memory_order_acquire))};
        const Clock::time_point deadline =
            window_start + std::chrono::duration_cast<Clock::duration>(
                               std::chrono::duration<double>(seconds)
                           );
        long long count = 0;
        double busy_here = 0.0;
        double sum = 0.0;
        while (!stop.load(std::memory_order_relaxed)) {
            const Clock::time_point begin = Clock::now();
            if (begin >= deadline) break;
            double value = 0.0;
            double core_ms = 0.0;
            if (!attempt(item, slot, worker, &value, &core_ms, message)) break;
            busy_here += Seconds(Clock::now() - begin);
            sum += value;
            ++count;
        }
        calls[static_cast<std::size_t>(worker)] = count;
        busy[static_cast<std::size_t>(worker)] = busy_here;
        checksums[static_cast<std::size_t>(worker)] = sum;
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (int worker = 0; worker < threads; ++worker) {
        workers.emplace_back(body, worker);
    }
    while (warmed.load(std::memory_order_acquire) < threads) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    // Published only now, so every worker's warmup call is outside the window.
    const Clock::time_point window_start = Clock::now();
    start_ticks.store(window_start.time_since_epoch().count());
    started.store(true, std::memory_order_release);
    for (std::thread& worker : workers) worker.join();
    const Clock::time_point window_end = Clock::now();

    stats->seconds = Seconds(window_end - window_start);
    for (std::size_t worker = 0; worker < static_cast<std::size_t>(threads);
         ++worker) {
        stats->calls += calls[worker];
        stats->busy_seconds += busy[worker];
        stats->checksum += checksums[worker];
    }
    stats->fps = stats->seconds > 0.0
                     ? static_cast<double>(stats->calls) / stats->seconds
                     : 0.0;
    if (!failure.empty()) {
        *error = std::move(failure);
        return false;
    }
    if (stats->calls == 0) {
        *error = "no metric call completed inside the throughput window";
        return false;
    }
    return true;
}

// The one machine-readable line a throughput run prints, on stdout, whatever
// --score-only says -- the human tables are what --score-only suppresses.
// `gpu_ms` is the stage timer's per-call GPU time for a single-worker run of
// our bench; its reciprocal is a stage-rate reference, not a throughput limit.
// The machine-readable field retains the historical name `ceiling_fps`.
inline void EmitThroughput(
    const ThroughputStats& stats, std::string_view metric,
    std::optional<double> gpu_ms
) {
    std::string line = std::format(
        "throughput metric={} size={}x{} threads={} calls={} seconds={:.6f} "
        "fps={:.6f} busy={:.6f} checksum={:.6f}",
        metric, stats.width, stats.height, stats.threads, stats.calls,
        stats.seconds, stats.fps, stats.busy_seconds, stats.checksum
    );
    if (gpu_ms) {
        const double gpu = *gpu_ms;
        line += std::format(
            " gpu_ms={:.6f} ceiling_fps={:.6f}", gpu,
            gpu > 0.0 ? 1000.0 / gpu : 0.0
        );
    }
    std::puts(line.c_str());
}

}  // namespace inmem

#endif  // VQM_BENCH_INMEMORY_DRIVER_H_
