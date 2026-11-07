// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// In-memory harness for libjxl's SSIMULACRA 2 and Butteraugli references. It
// decodes each pair once with libjxl's own reader, outside the timer, then
// times the reference entry point -- ComputeSSIMULACRA2 for SSIMULACRA 2 and
// jxl::ComputeScore on a JxlButteraugliComparator for Butteraugli, the same
// call sequence as tools/butteraugli_main.cc.
//
// Built by tools/build_references.py against a copy of .reference/libjxl under
// build/libjxl-src; the reference tree itself is never modified.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <jxl/cms.h>
#include <jxl/cms_interface.h>

#include "driver.h"
#include "lib/extras/codec.h"
#include "lib/extras/codec_in_out.h"
#include "lib/extras/dec/color_hints.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/enc_butteraugli_comparator.h"
#include "lib/jxl/enc_comparator.h"
#include "lib/jxl/image.h"
#include "tools/file_io.h"
#include "tools/no_memory_manager.h"
#include "tools/ssimulacra2.h"
#include "tools/thread_pool_internal.h"

namespace {

std::string Usage(std::string_view program) {
    return std::format(
        "Usage: {} --metric ssimu2|butteraugli --pairs-file PATH\n"
        "          [--pair ORIG DIST ...] [--iterations N] [--warmup N]\n"
        "          [--limit N] [--csv PATH] [--score-only] [--threads N]\n"
        "          [--decode-threads N] [--throughput-seconds S] [--help]\n"
        "\n"
        "Times libjxl's in-memory metric call; the decode is measured\n"
        "separately and never enters the metric timer.\n"
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

bool Load(
    const inmem::Pair& pair, jxl::ThreadPool* pool, Decoded* out,
    std::string* error
) {
    JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
    auto original = std::make_unique<jxl::CodecInOut>(memory_manager);
    auto distorted = std::make_unique<jxl::CodecInOut>(memory_manager);
    for (const auto& entry : {std::pair<const std::string*, jxl::CodecInOut*>{
                                  &pair.original, original.get()},
                              std::pair<const std::string*, jxl::CodecInOut*>{
                                  &pair.distorted, distorted.get()}}) {
        std::vector<std::uint8_t> bytes;
        if (!jpegxl::tools::ReadFile(*entry.first, &bytes)) {
            *error = std::format("could not read {}", *entry.first);
            return false;
        }
        if (!jxl::SetFromBytes(
                jxl::Bytes(bytes), jxl::extras::ColorHints(), entry.second, pool
            )) {
            *error = std::format("could not decode {}", *entry.first);
            return false;
        }
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

// One pair decoded for a batch or throughput run, with its own load time: the
// metric timer starts only after the load returns. `width` and `height` are
// what the driver reports as the measured geometry.
struct Item {
    Decoded decoded;
    double decode_ms = 0.0;
    int width = 0;
    int height = 0;
};

// The decode-thread loader. It hands SetFromBytes no thread pool: the pool
// would be shared by every decode thread and oversubscribe the machine, and
// the driver's decode threads already fill it.
bool LoadItem(const inmem::Pair& pair, Item& out, std::string& error) {
    const inmem::Clock::time_point start = inmem::Clock::now();
    if (!Load(pair, nullptr, &out.decoded, &error)) return false;
    out.decode_ms = 1000.0 * inmem::Seconds(inmem::Clock::now() - start);
    out.width = static_cast<int>(out.decoded.original->xsize());
    out.height = static_cast<int>(out.decoded.original->ysize());
    return true;
}

// Per-worker Butteraugli state. The comparator keeps the reference image and
// the diffmap is the comparison's scratch output, so neither can be shared
// between workers.
struct ButterState {
    explicit ButterState(const JxlCmsInterface& cms)
        : comparator(Params(), cms) {}

    static jxl::ButteraugliParams Params() {
        jxl::ButteraugliParams params;
        params.hf_asymmetry = 1.0f;
        params.xmul = 1.0f;
        params.intensity_target = 80.0f;
        return params;
    }

    jxl::JxlButteraugliComparator comparator;
    jxl::ImageF diffmap;
};

// One entry per worker id, or empty for SSIMULACRA 2, which keeps no state
// between calls.
using ButterPool = std::vector<std::unique_ptr<ButterState>>;

ButterPool MakeButterPool(const inmem::Options& options, int threads) {
    ButterPool pool;
    if (options.metric != "butteraugli") return pool;
    const JxlCmsInterface& cms = *JxlGetDefaultCms();
    pool.reserve(static_cast<std::size_t>(threads));
    for (int worker = 0; worker < threads; ++worker) {
        pool.push_back(std::make_unique<ButterState>(cms));
    }
    return pool;
}

// One metric call. SSIMULACRA 2 blends an alpha pair against dark and bright
// backgrounds and reports the worse score if either image has alpha;
// Butteraugli's comparator comes from the worker's slot.
bool CallOnce(
    const inmem::Options& options, Item& item, ButterState* state, double* score,
    std::string* error
) {
    const bool has_alpha = HasAlpha(item.decoded);
    if (options.metric != "butteraugli") {
        const auto compute = [&item, has_alpha](float background) {
            return has_alpha ? ComputeSSIMULACRA2(
                                   item.decoded.original->Main(),
                                   item.decoded.distorted->Main(), background
                               )
                             : ComputeSSIMULACRA2(
                                   item.decoded.original->Main(),
                                   item.decoded.distorted->Main()
                               );
        };
        jxl::StatusOr<Msssim> result = compute(0.1f);
        if (!result.ok()) {
            *error = std::format(
                "ComputeSSIMULACRA2 failed (code {})",
                static_cast<int>(result.status().code())
            );
            return false;
        }
        double value = std::move(result).value_().Score();
        if (has_alpha) {
            jxl::StatusOr<Msssim> bright = compute(0.9f);
            if (!bright.ok()) {
                *error = std::format(
                    "ComputeSSIMULACRA2 failed (code {})",
                    static_cast<int>(bright.status().code())
                );
                return false;
            }
            value = std::min(value, std::move(bright).value_().Score());
        }
        *score = value;
        return true;
    }
    const JxlCmsInterface& cms = *JxlGetDefaultCms();
    float distance = 0.0f;
    const jxl::Status status = jxl::ComputeScore(
        item.decoded.original->Main(), item.decoded.distorted->Main(),
        &state->comparator, cms, &distance, &state->diffmap, nullptr,
        /* ignore_alpha */ false
    );
    if (!status) {
        *error = std::format(
            "ComputeScore failed (code {})", static_cast<int>(status.code())
        );
        return false;
    }
    *score = static_cast<double>(distance);
    return true;
}

// Score-only batches produce the complete reference score, including both
// backgrounds for alpha. Their internal time is not the one-background
// latency reported by the serial path.
struct BatchScorer {
    const inmem::Options* options = nullptr;
    ButterPool* pool = nullptr;

    bool operator()(
        Item& item, std::size_t, int worker, double* score, double* core_ms,
        std::string& error
    ) const {
        ButterState* state =
            pool->empty() ? nullptr
                          : (*pool)[static_cast<std::size_t>(worker)].get();
        std::vector<double> core_times;
        core_times.reserve(static_cast<std::size_t>(options->iterations));
        double value = 0.0;
        for (int i = 0; i < options->warmup + options->iterations; ++i) {
            const inmem::Clock::time_point start = inmem::Clock::now();
            if (!CallOnce(*options, item, state, &value, &error)) return false;
            const double seconds = inmem::Seconds(inmem::Clock::now() - start);
            if (i >= options->warmup) core_times.push_back(seconds);
        }
        *score = value;
        *core_ms = 1000.0 * inmem::Median(core_times);
        return true;
    }
};

// One complete pair score for the window, including both backgrounds for
// alpha, timed by the driver.
struct WindowScorer {
    const inmem::Options* options = nullptr;
    ButterPool* pool = nullptr;

    bool operator()(
        Item& item, std::size_t, int worker, double* score, double* core_ms,
        std::string& error
    ) const {
        *core_ms = 0.0;
        ButterState* state =
            pool->empty() ? nullptr
                          : (*pool)[static_cast<std::size_t>(worker)].get();
        return CallOnce(*options, item, state, score, &error);
    }
};

bool RunSsimu2(
    const inmem::Options& options, const Decoded& decoded, double* score,
    double* core_ms, double* checksum, std::string* error
) {
    // Blend an alpha image against dark and bright
    // backgrounds and reports the worse score; only the dark call is timed
    // because the background is a parameter, not a different workload.
    const bool has_alpha = HasAlpha(decoded);
    const auto compute = [&decoded, has_alpha](float background) {
        return has_alpha
                   ? ComputeSSIMULACRA2(
                         decoded.original->Main(), decoded.distorted->Main(),
                         background
                     )
                   : ComputeSSIMULACRA2(
                         decoded.original->Main(), decoded.distorted->Main()
                     );
    };
    std::vector<double> core_times;
    core_times.reserve(static_cast<std::size_t>(options.iterations));
    for (int i = 0; i < options.warmup + options.iterations; ++i) {
        const inmem::Clock::time_point start = inmem::Clock::now();
        jxl::StatusOr<Msssim> result = compute(0.1f);
        if (!result.ok()) {
            *error = std::format(
                "ComputeSSIMULACRA2 failed (code {})",
                static_cast<int>(result.status().code())
            );
            return false;
        }
        const double value = std::move(result).value_().Score();
        const double seconds = inmem::Seconds(inmem::Clock::now() - start);
        *checksum += value;
        if (i >= options.warmup) {
            core_times.push_back(seconds);
            *score = value;
        }
    }
    if (has_alpha) {
        jxl::StatusOr<Msssim> bright = compute(0.9f);
        if (!bright.ok()) {
            *error = std::format(
                "ComputeSSIMULACRA2 failed (code {})",
                static_cast<int>(bright.status().code())
            );
            return false;
        }
        const double bright_score = std::move(bright).value_().Score();
        *score = std::min(*score, bright_score);
    }
    *core_ms = 1000.0 * inmem::Median(core_times);
    return true;
}

bool RunButter(
    const inmem::Options& options, const Decoded& decoded, double* score,
    double* core_ms, double* checksum, std::string* error
) {
    jxl::ButteraugliParams params;
    params.hf_asymmetry = 1.0f;
    params.xmul = 1.0f;
    params.intensity_target = 80.0f;
    const JxlCmsInterface& cms = *JxlGetDefaultCms();
    jxl::JxlButteraugliComparator comparator(params, cms);
    std::vector<double> core_times;
    core_times.reserve(static_cast<std::size_t>(options.iterations));
    jxl::ImageF diffmap;
    for (int i = 0; i < options.warmup + options.iterations; ++i) {
        float distance = 0.0f;
        const inmem::Clock::time_point start = inmem::Clock::now();
        const jxl::Status status = jxl::ComputeScore(
            decoded.original->Main(), decoded.distorted->Main(), &comparator,
            cms, &distance, &diffmap, nullptr, /* ignore_alpha */ false
        );
        const double seconds = inmem::Seconds(inmem::Clock::now() - start);
        if (!status) {
            *error = std::format(
                "ComputeScore failed (code {})", static_cast<int>(status.code())
            );
            return false;
        }
        *checksum += static_cast<double>(distance);
        if (i >= options.warmup) {
            core_times.push_back(seconds);
            *score = static_cast<double>(distance);
        }
    }
    *core_ms = 1000.0 * inmem::Median(core_times);
    return true;
}

// The fixed window: decode once outside the timer, then score until the
// deadline with `threads` workers. Only the machine line is printed.
int RunWindow(const inmem::Options& options, int threads, int decode_threads) {
    inmem::ThroughputStats stats;
    std::string error;
    ButterPool pool = MakeButterPool(options, threads);
    WindowScorer scorer{&options, &pool};
    if (!inmem::RunThroughput<Item>(
            options.pairs, threads, decode_threads, options.throughput_seconds,
            LoadItem, scorer, &stats, &error
        )) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    inmem::EmitThroughput(stats, options.metric, std::nullopt);
    return 0;
}

// The batch path: `threads` workers score the list and the rows are emitted in
// input order afterwards.
int RunBatchPath(
    const inmem::Options& options, int threads, int decode_threads
) {
    ButterPool pool = MakeButterPool(options, threads);
    BatchScorer scorer{&options, &pool};
    std::vector<inmem::PairResult> results;
    std::string error;
    if (!inmem::RunBatch<Item>(
            options.pairs, threads, decode_threads, LoadItem, scorer, &results,
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
    const std::string_view program = argc > 0 ? argv[0] : "libjxl_bench";
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
    if (options.score_only)
        return RunBatchPath(options, threads, decode_threads);

    // Latency path: one pair at a time and three decodes for the median. The
    // decode pool stays here -- this path is sequential, so one pool fills the
    // machine without oversubscribing it.
    jpegxl::tools::ThreadPoolInternal pool(4);
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
                return Load(pair, pool.get(), &decoded, load_error);
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
        double core_ms = 0.0;
        const bool measured =
            options.metric == "butteraugli"
                ? RunButter(
                      options, decoded, &score, &core_ms, &checksum, &error
                  )
                : RunSsimu2(
                      options, decoded, &score, &core_ms, &checksum, &error
                  );
        if (!measured) {
            std::fprintf(
                stderr, "%s vs %s: %s\n", pair.original.c_str(),
                pair.distorted.c_str(), error.c_str()
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
        row.core_ms = core_ms;
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
