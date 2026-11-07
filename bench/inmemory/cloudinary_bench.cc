// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// In-memory harness for Cloudinary's SSIMULACRA 2 reference. It decodes each
// pair once with the reference's own codec reader, outside the timer, then
// times ComputeSSIMULACRA2 calls in memory -- the same shape as
// bench/inmemory/libjxl_bench.cc, so the two rows are comparable.
//
// Built by tools/build_references.py against a copy of .reference/ssimulacra2
// under build/reference-src; the reference tree itself is never modified.

#include <algorithm>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "driver.h"
#include "lib/extras/codec.h"
#include "ssimulacra2.h"

namespace {

std::string Usage(std::string_view program) {
    return std::format(
        "Usage: {} --pairs-file PATH | --pair ORIG DIST ...\n"
        "          [--iterations N] [--warmup N] [--limit N] [--csv PATH]\n"
        "          [--score-only] [--threads N] [--decode-threads N]\n"
        "          [--throughput-seconds S] [--help]\n"
        "\n"
        "Times the Cloudinary reference's in-memory SSIMULACRA 2 call; the\n"
        "decode is measured separately and never enters the metric timer.\n"
        "--threads N scores with N workers in this process (--score-only and\n"
        "--throughput-seconds default to every hardware thread, the latency\n"
        "path to one); --decode-threads N sizes the decode pool.\n"
        "--throughput-seconds S runs a fixed S-second window over decoded\n"
        "pairs and prints one machine-readable throughput line.\n",
        program
    );
}

struct Decoded {
    std::unique_ptr<jxl::CodecInOut> original;
    std::unique_ptr<jxl::CodecInOut> distorted;
};

bool HasAlpha(const Decoded& decoded) {
    return decoded.original->Main().HasAlpha() ||
           decoded.distorted->Main().HasAlpha();
}

// One pair decoded for a batch or throughput run, with its own load time: the
// metric timer starts only after the load returns. `width` and `height` are
// what the driver reports as the measured geometry.
struct Item {
    Decoded decoded;
    double decode_ms = 0.0;
    int width = 0;
    int height = 0;
};

bool Load(const inmem::Pair& pair, Decoded* out, std::string* error) {
    auto original = std::make_unique<jxl::CodecInOut>();
    if (!SetFromFile(pair.original, jxl::extras::ColorHints(), original.get())) {
        *error = std::format("could not load {}", pair.original);
        return false;
    }
    auto distorted = std::make_unique<jxl::CodecInOut>();
    if (!SetFromFile(
            pair.distorted, jxl::extras::ColorHints(), distorted.get()
        )) {
        *error = std::format("could not load {}", pair.distorted);
        return false;
    }
    if (original->xsize() != distorted->xsize() ||
        original->ysize() != distorted->ysize()) {
        *error = "image size mismatch";
        return false;
    }
    out->original = std::move(original);
    out->distorted = std::move(distorted);
    return true;
}

// The driver's loader: one timed decode of both members, thread-safe because
// the reference reader keeps no shared state this harness touches.
bool LoadItem(const inmem::Pair& pair, Item& out, std::string& error) {
    const inmem::Clock::time_point start = inmem::Clock::now();
    if (!Load(pair, &out.decoded, &error)) return false;
    out.decode_ms = 1000.0 * inmem::Seconds(inmem::Clock::now() - start);
    out.width = static_cast<int>(out.decoded.original->xsize());
    out.height = static_cast<int>(out.decoded.original->ysize());
    return true;
}

double ScoreOne(const Decoded& decoded, float background) {
    if (!HasAlpha(decoded)) {
        return ComputeSSIMULACRA2(
                   decoded.original->Main(), decoded.distorted->Main()
        )
            .Score();
    }
    return ComputeSSIMULACRA2(
               decoded.original->Main(), decoded.distorted->Main(), background
    )
        .Score();
}

// Compare the worse of both backgrounds whenever either image has alpha.
double Score(const Decoded& decoded) {
    const double dark = ScoreOne(decoded, 0.1f);
    if (!HasAlpha(decoded)) return dark;
    return std::min(dark, ScoreOne(decoded, 0.9f));
}

// One metric call, with the reference's own exceptions turned into an error
// string so a worker thread never has to unwind into the driver.
bool ScoreOnce(Item& item, double* score, std::string* error) {
    try {
        *score = Score(item.decoded);
    } catch (const std::exception& e) {
        *error = e.what();
        return false;
    }
    return true;
}

// Score-only batches produce the complete reference score, including both
// backgrounds for alpha. Their internal time is not the one-background
// latency reported by the serial path.
bool MeasurePair(
    const inmem::Options& options, Item& item, double* score, double* core_ms,
    std::string& error
) {
    std::vector<double> core_times;
    core_times.reserve(static_cast<std::size_t>(options.iterations));
    double value = 0.0;
    for (int i = 0; i < options.warmup + options.iterations; ++i) {
        const inmem::Clock::time_point start = inmem::Clock::now();
        if (!ScoreOnce(item, &value, &error)) return false;
        const double seconds = inmem::Seconds(inmem::Clock::now() - start);
        if (i >= options.warmup) core_times.push_back(seconds);
    }
    *score = value;
    *core_ms = 1000.0 * inmem::Median(core_times);
    return true;
}

// One complete pair score for the throughput window, including both
// backgrounds for alpha, timed by the driver.
bool ScoreForWindow(
    Item& item, std::size_t, int, double* score, double* core_ms,
    std::string& error
) {
    *core_ms = 0.0;
    return ScoreOnce(item, score, &error);
}

// The fixed window: decode once outside the timer, then score until the
// deadline with `threads` workers. Only the machine line is printed.
int RunWindow(const inmem::Options& options, int threads, int decode_threads) {
    inmem::ThroughputStats stats;
    std::string error;
    if (!inmem::RunThroughput<Item>(
            options.pairs, threads, decode_threads, options.throughput_seconds,
            LoadItem, ScoreForWindow, &stats, &error
        )) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    inmem::EmitThroughput(stats, options.metric, std::nullopt);
    return 0;
}

// The batch path: `threads` workers score the list, results are emitted in
// input order afterwards. Scores use the same worst-background policy as the
// serial path.
int RunBatch(const inmem::Options& options, int threads, int decode_threads) {
    std::vector<inmem::PairResult> results;
    std::string error;
    const auto score = [&options](
                           Item& item, std::size_t, int, double* value,
                           double* core_ms, std::string& message
                       ) {
        return MeasurePair(options, item, value, core_ms, message);
    };
    if (!inmem::RunBatch<Item>(
            options.pairs, threads, decode_threads, LoadItem, score, &results,
            &error
        )) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::vector<inmem::Row> rows;
    rows.reserve(options.pairs.size());
    int failures = 0;
    for (std::size_t index = 0; index < options.pairs.size(); ++index) {
        const inmem::Pair& pair = options.pairs[index];
        const inmem::PairResult& result = results[index];
        if (!result.ok) {
            std::fprintf(
                stderr, "%s vs %s: %s\n", pair.original.c_str(),
                pair.distorted.c_str(), result.error.c_str()
            );
            inmem::EmitPairError(true);
            ++failures;
            continue;
        }
        inmem::Row row;
        row.original = pair.original;
        row.distorted = pair.distorted;
        row.width = result.width;
        row.height = result.height;
        row.decode_ms = result.decode_ms;
        row.core_ms = result.core_ms;
        row.score = result.score;
        rows.push_back(row);
        inmem::EmitPairScore(true, result.score);
    }
    if (rows.empty()) {
        std::fprintf(stderr, "no pair could be measured\n");
        return 1;
    }
    if (!options.csv_path.empty() && !inmem::WriteCsv(options.csv_path, rows)) {
        std::fprintf(stderr, "could not write %s\n", options.csv_path.c_str());
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string_view program = argc > 0 ? argv[0] : "cloudinary_bench";
    std::vector<std::string_view> args;
    args.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);

    inmem::Options options;
    std::string error;
    if (!inmem::ParseArgs(args, options, &error)) {
        std::fprintf(stderr, "%s\n%s", error.c_str(), Usage(program).c_str());
        return 2;
    }
    if (options.help) {
        std::fputs(Usage(program).c_str(), stdout);
        return 0;
    }
    if (options.metric != "ssimu2") {
        std::fprintf(stderr, "this harness scores SSIMULACRA 2 only\n");
        return 2;
    }
    if (!inmem::ResolvePairs(options, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }
    if (options.pairs.empty()) {
        std::fputs(Usage(program).c_str(), stderr);
        return 2;
    }

    const bool batch = options.score_only || options.throughput_seconds > 0.0;
    const int threads = inmem::ResolveThreads(options, batch);
    const int decode_threads = inmem::ResolveDecodeThreads(options, batch);
    if (options.throughput_seconds > 0.0) {
        return RunWindow(options, threads, decode_threads);
    }
    if (options.score_only) return RunBatch(options, threads, decode_threads);

    // Latency path: one pair at a time, three decodes for the median, and the
    // human table. --threads does not apply here -- a per-pair latency is not
    // measured under contention.
    std::printf("%-60s %9s %9s %10s\n", "pair", "decode_ms", "core_ms", "score");
    std::vector<inmem::Row> rows;
    rows.reserve(options.pairs.size());
    double checksum = 0.0;
    int failures = 0;
    for (const inmem::Pair& pair : options.pairs) {
        Decoded decoded;
        double decode_ms = 0.0;
        const bool loaded = inmem::TimedDecode(
            3,
            [&](std::string* load_error) {
                return Load(pair, &decoded, load_error);
            },
            &decode_ms, &error
        );
        if (!loaded) {
            std::fprintf(
                stderr, "%s vs %s: %s\n", pair.original.c_str(),
                pair.distorted.c_str(), error.c_str()
            );
            ++failures;
            continue;
        }
        double score = 0.0;
        std::vector<double> core_times;
        core_times.reserve(static_cast<std::size_t>(options.iterations));
        try {
            for (int i = 0; i < options.warmup + options.iterations; ++i) {
                const inmem::Clock::time_point start = inmem::Clock::now();
                score = ScoreOne(decoded, 0.1f);
                const double seconds =
                    inmem::Seconds(inmem::Clock::now() - start);
                checksum += score;
                if (i >= options.warmup) core_times.push_back(seconds);
            }
            if (HasAlpha(decoded)) {
                const double bright = ScoreOne(decoded, 0.9f);
                score = std::min(score, bright);
            }
        } catch (const std::exception& e) {
            std::fprintf(
                stderr, "%s vs %s: %s\n", pair.original.c_str(),
                pair.distorted.c_str(), e.what()
            );
            ++failures;
            continue;
        }
        inmem::Row row;
        row.original = pair.original;
        row.distorted = pair.distorted;
        row.width = static_cast<int>(decoded.original->xsize());
        row.height = static_cast<int>(decoded.original->ysize());
        row.decode_ms = decode_ms;
        row.core_ms = 1000.0 * inmem::Median(core_times);
        row.score = score;
        rows.push_back(row);
        std::printf(
            "%-60s %9.1f %9.1f %10s\n",
            (pair.original + " vs " + pair.distorted).c_str(), row.decode_ms,
            row.core_ms, inmem::FormatScore(score).c_str()
        );
    }
    if (rows.empty()) {
        std::fprintf(stderr, "no pair could be measured\n");
        return 1;
    }
    std::printf("checksum %.6f\n", checksum);
    if (!options.csv_path.empty() && !inmem::WriteCsv(options.csv_path, rows)) {
        std::fprintf(stderr, "could not write %s\n", options.csv_path.c_str());
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}
