// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// In-memory harness for fssimu2's C API (lib/libssimu2.so). It decodes each
// pair once through this repository's io reader into the 8-bit RGB planes the
// C API accepts, outside the timer, then times ssimulacra2_score calls.
//
// fssimu2's C ABI takes interleaved sRGB bytes -- src/ssimulacra2.zig feeds
// the caller's buffer straight to sRGBInterleavedToPlanarLinear -- while our
// Image is planar. The planes are interleaved once per pair here; passing the
// planar buffer would score a transposed-looking image and return a plausible
// but wrong number.
//
// Built by tools/build_references.py; the reference tree is never modified.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "driver.h"
#include "input.h"
#include "io.h"
#include "vqm.h"
#include "ssimu2.h"

namespace {

std::string Usage(std::string_view program) {
    return std::format(
        "Usage: {} --pairs-file PATH | --pair ORIG DIST ...\n"
        "          [--iterations N] [--warmup N] [--limit N] [--csv PATH]\n"
        "          [--score-only] [--threads N] [--decode-threads N]\n"
        "          [--throughput-seconds S] [--help]\n"
        "\n"
        "Times fssimu2's ssimulacra2_score on interleaved 8-bit RGB; the\n"
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
    std::vector<std::uint8_t> original;
    std::vector<std::uint8_t> distorted;
    int width = 0;
    int height = 0;
};

// fssimu2's C ABI is interleaved sRGB; our Image is planar (zimg order). The
// reference CLI's own decoder goes through libspng, which converts a 16-bit
// PNG to 8-bit by keeping the high byte of every sample (spng.c:3620-3624),
// drops alpha and replicates grey, so the expansion here is the same. A pair
// scored through this harness is therefore comparable with the CLI's score,
// not with the native-depth implementations.
bool Interleave(
    const vqm::Image& image, std::vector<std::uint8_t>* out, std::string* error
) {
    if (image.format.type != vqm::PixelType::Byte &&
        image.format.type != vqm::PixelType::Word) {
        *error = "fssimu2 driver needs 8- or 16-bit integer planes";
        return false;
    }
    const std::array<vqm::internal::PlaneGeometry, 4> layout =
        vqm::internal::PlaneLayout(image.format, image.size);
    const std::size_t bytes = layout[0].sample_bytes;
    if (bytes != 1 && bytes != 2) {
        *error = "unexpected sample size";
        return false;
    }
    const std::size_t pixels =
        static_cast<std::size_t>(image.size.width) * image.size.height;
    // libspng's 16 -> 8 conversion is `sample >> 8`, which on this platform's
    // little-endian layout is the second byte of the sample.
    const auto read = [&](std::size_t plane, std::size_t index) {
        const std::uint8_t* sample =
            image.data.data() + layout[plane].offset + index * bytes;
        return bytes == 1 ? sample[0] : sample[1];
    };
    out->resize(3 * pixels);
    if (image.format.family == vqm::ColorFamily::Rgb) {
        if (!layout[1].present() || !layout[2].present() ||
            layout[1].sample_bytes != bytes || layout[2].sample_bytes != bytes) {
            *error = "unexpected RGB plane layout";
            return false;
        }
        for (std::size_t i = 0; i < pixels; ++i) {
            (*out)[3 * i + 0] = read(0, i);
            (*out)[3 * i + 1] = read(1, i);
            (*out)[3 * i + 2] = read(2, i);
        }
        return true;
    }
    if (image.format.family == vqm::ColorFamily::Grey) {
        if (layout[1].present() || layout[2].present()) {
            *error = "unexpected grey plane layout";
            return false;
        }
        for (std::size_t i = 0; i < pixels; ++i) {
            const std::uint8_t value = read(0, i);
            (*out)[3 * i + 0] = value;
            (*out)[3 * i + 1] = value;
            (*out)[3 * i + 2] = value;
        }
        return true;
    }
    *error = "fssimu2 driver needs RGB or grey planes";
    return false;
}

bool Load(const inmem::Pair& pair, Decoded* out, std::string* error) {
    std::optional<vqm::Image> original =
        vqm::image_io::LoadImage(pair.original, *error);
    if (!original) return false;
    std::optional<vqm::Image> distorted =
        vqm::image_io::LoadImage(pair.distorted, *error);
    if (!distorted) return false;
    if (original->size != distorted->size) {
        *error = "image size mismatch";
        return false;
    }
    if (!Interleave(*original, &out->original, error)) return false;
    if (!Interleave(*distorted, &out->distorted, error)) return false;
    out->width = static_cast<int>(original->size.width);
    out->height = static_cast<int>(original->size.height);
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

bool LoadItem(const inmem::Pair& pair, Item& out, std::string& error) {
    const inmem::Clock::time_point start = inmem::Clock::now();
    if (!Load(pair, &out.decoded, &error)) return false;
    out.decode_ms = 1000.0 * inmem::Seconds(inmem::Clock::now() - start);
    out.width = out.decoded.width;
    out.height = out.decoded.height;
    return true;
}

// One metric call. fssimu2's C ABI allocates its workspace per call, so
// nothing is shared between workers.
bool ScoreOnce(Item& item, double* score, std::string* error) {
    double value = 0.0;
    const int rc = ssimulacra2_score(
        item.decoded.original.data(), item.decoded.distorted.data(),
        static_cast<unsigned>(item.decoded.width),
        static_cast<unsigned>(item.decoded.height), 3, &value
    );
    if (rc != SSIMU2_OK) {
        *error = std::format("ssimulacra2_score returned {}", rc);
        return false;
    }
    *score = value;
    return true;
}

// The batch scorer: `--warmup` + `--iterations` calls and the median duration,
// so a score-only row carries the same core_ms the latency path's does.
struct BatchScorer {
    const inmem::Options* options = nullptr;

    bool operator()(
        Item& item, std::size_t, int, double* score, double* core_ms,
        std::string& error
    ) const {
        std::vector<double> core_times;
        core_times.reserve(static_cast<std::size_t>(options->iterations));
        double value = 0.0;
        for (int i = 0; i < options->warmup + options->iterations; ++i) {
            const inmem::Clock::time_point start = inmem::Clock::now();
            if (!ScoreOnce(item, &value, &error)) return false;
            const double seconds = inmem::Seconds(inmem::Clock::now() - start);
            if (i >= options->warmup) core_times.push_back(seconds);
        }
        *score = value;
        *core_ms = 1000.0 * inmem::Median(core_times);
        return true;
    }
};

// The window scorer: exactly one metric call, timed by the driver.
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

// The batch path: `threads` workers score the list and the rows are emitted in
// input order afterwards.
int RunBatchPath(
    const inmem::Options& options, int threads, int decode_threads
) {
    std::vector<inmem::PairResult> results;
    std::string error;
    const BatchScorer scorer{&options};
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
    const std::string_view program = argc > 0 ? argv[0] : "fssimu2_bench";
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
    if (options.score_only)
        return RunBatchPath(options, threads, decode_threads);

    // Latency path: one pair at a time, three decodes for the median, and the
    // human table. --threads does not apply here.
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
        bool ok = true;
        try {
            for (int i = 0; i < options.warmup + options.iterations; ++i) {
                double value = 0.0;
                const inmem::Clock::time_point start = inmem::Clock::now();
                const int rc = ssimulacra2_score(
                    decoded.original.data(), decoded.distorted.data(),
                    static_cast<unsigned>(decoded.width),
                    static_cast<unsigned>(decoded.height), 3, &value
                );
                const double seconds =
                    inmem::Seconds(inmem::Clock::now() - start);
                if (rc != SSIMU2_OK) {
                    std::fprintf(
                        stderr, "%s vs %s: ssimulacra2_score returned %d\n",
                        pair.original.c_str(), pair.distorted.c_str(), rc
                    );
                    ok = false;
                    break;
                }
                checksum += value;
                if (i >= options.warmup) {
                    core_times.push_back(seconds);
                    score = value;
                }
            }
        } catch (const std::exception& e) {
            std::fprintf(
                stderr, "%s vs %s: %s\n", pair.original.c_str(),
                pair.distorted.c_str(), e.what()
            );
            ok = false;
        }
        if (!ok) {
            ++failures;
            continue;
        }
        inmem::Row row;
        row.original = pair.original;
        row.distorted = pair.distorted;
        row.width = decoded.width;
        row.height = decoded.height;
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
