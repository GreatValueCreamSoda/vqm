// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// Benchmark driver for Vship's metrics, shaped like bench/driver/ssimulacra2.cc
// so the two tools' output can sit side by side: same
// --pair/--iterations/--warmup/--csv flags, same row and aggregate format, and
// the same rule that only the metric call is timed (decoding is measured
// separately).
//
// Vship is never built in place: tools/build_references.py copies
// .reference/Vship under build/, builds libvship there (BACKEND=Cuda or
// Vulkan) and compiles this file against it. Vship is MIT with a no-AI clause:
// this tool only calls its public C API (src/VshipAPI.h), it does not copy or
// adapt any of its code.
//
// The colours passed to Vship are full-range, identity-matrix, BT.709-primaried
// sRGB planes, with each side's own sample depth (8 or 16 bits) -- the same
// decode class this repository's CLI pins for PNG input -- so both tools
// measure the same pixels.

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <format>
#include <iterator>
#include <map>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>

#include "VshipAPI.h"
#include "inmemory/driver.h"
#include "inmemory/pairs.h"
#include "inmemory/prefetch.h"
#include "input.h"
#include "io.h"
#include "vqm.h"

namespace {

using Clock = inmem::Clock;
using inmem::Median;
using inmem::Seconds;
using Pair = inmem::Pair;

enum class Metric { Butteraugli, Ssimulacra2, Cvvdp };

struct Options {
    std::vector<Pair> pairs;
    std::string pairs_file;
    std::string csv_path;
    Metric metric = Metric::Butteraugli;
    bool score_only = false;
    int iterations = 10;
    int warmup = 2;
    int gpu = 0;
    int qnorm = 2;  // 0 = qnorm, 1 = 3norm, 2 = infnorm
    int limit = 0;
    // Decode worker threads (0 = the mode default), and how many handlers this
    // process runs: Vship's handler is per thread, so `threads` handlers score
    // concurrently against one device.
    int decode_threads = 0;
    int threads = 0;
    // Fixed-length throughput window; 0 runs the latency protocol.
    double throughput_seconds = 0.0;
    int throughput_iterations = 0;
    float intensity = 80.0f;  // libjxl's Butteraugli intensity target
    bool help = false;
};

std::string Usage(std::string_view program) {
    return std::format(
        "Usage: {} [--pair ORIG DIST]... [--pairs-file PATH] [--limit N]\n"
        "          [--metric butteraugli|ssimu2|cvvdp] [--score-only]\n"
        "          [--iterations N] [--warmup N] [--gpu ID] [--qnorm N]\n"
        "          [--intensity NITS] [--csv PATH] [--decode-threads N]\n"
        "          [--threads N] [--throughput-seconds S | "
        "--throughput-iterations N] [--help]\n"
        "          --device-info\n"
        "\n"
        "Times Vship's metric call on the decoded pair; decoding is measured\n"
        "separately, exactly as bench/driver/ssimulacra2.cc does for this "
        "repository's\n"
        "backend. Butteraugli prints Vship's normQ, norm3 and norminf (score\n"
        "column = norminf); ssimu2 prints its single score.\n"
        "cvvdp scores independent RGB/sRGB stills, explicitly standard_4k, no\n"
        "resizing or heatmap. Both history and score are reset before every\n"
        "call; reset API overhead is included in core timing. This peer is a\n"
        "comparison arm, not the numerical oracle.\n"
        "--score-only prints one line per pair -- the score, or ERROR -- and\n"
        "nothing else, in input order; the reason for an ERROR goes to stderr.\n"
        "One handler is cached per thread per (metric, geometry, qnorm,\n"
        "intensity, gpu) for the whole process, so a batch pays the device\n"
        "setup once; --threads N scores with N handlers (the default for\n"
        "--score-only and --throughput-seconds is every hardware thread, for\n"
        "the latency path one).\n"
        "--throughput-seconds S runs a fixed S-second window over decoded\n"
        "pairs and prints one machine-readable throughput line.\n"
        "--throughput-iterations N instead completes N total evaluations,\n"
        "distributed across handlers after --warmup calls per handler. Setup,\n"
        "decoding and warmups are excluded; per-call uploads are included.\n",
        program
    );
}

// Names come from the Vship public API and need JSON escaping for the runner.
std::string JsonString(std::string_view text) {
    std::string result = "\"";
    for (char value : text) {
        const auto c = static_cast<unsigned char>(value);
        if (c == '"' || c == '\\') {
            result.push_back('\\');
            result.push_back(value);
        } else if (c < 0x20u) {
            result += std::format("\\u{:04x}", static_cast<unsigned>(c));
        } else {
            result.push_back(value);
        }
    }
    result.push_back('"');
    return result;
}

int PrintDeviceInfo() {
    int count = 0;
    if (Vship_GetDeviceCount(&count) != Vship_NoError || count <= 0) {
        std::fprintf(stderr, "vship: no device\n");
        return 1;
    }
    std::vector<std::string> devices;
    for (int id = 0; id < count; ++id) {
        Vship_DeviceInfo info{};
        if (Vship_GetDeviceInfo(&info, id) != Vship_NoError) {
            std::fprintf(stderr, "vship: cannot inspect device %d\n", id);
            return 1;
        }
        const char* end =
            std::find(std::begin(info.name), std::end(info.name), '\0');
        devices.push_back(
            std::format(
                "{{\"id\":{},\"name\":{}}}", id,
                JsonString(
                    std::string_view(
                        info.name, static_cast<std::size_t>(end - info.name)
                    )
                )
            )
        );
    }
    std::fputc('[', stdout);
    for (std::size_t i = 0; i < devices.size(); ++i) {
        if (i) std::fputc(',', stdout);
        std::fputs(devices[i].c_str(), stdout);
    }
    std::fputs("]\n", stdout);
    return 0;
}

bool ParseInt(std::string_view text, int& out) {
    const std::from_chars_result result =
        std::from_chars(text.data(), text.data() + text.size(), out);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

bool ParsePositiveFloat(std::string_view text, float& value) {
    const auto parsed =
        std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() &&
           std::isfinite(value) && value > 0.0f;
}

bool ParseArgs(std::span<const std::string_view> args, Options& options) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        const bool has_value = i + 1 < args.size();
        if (arg == "--help" || arg == "-h") {
            options.help = true;
            return true;
        }
        if (arg == "--pair") {
            if (i + 2 >= args.size()) {
                std::fprintf(stderr, "--pair needs two paths\n");
                return false;
            }
            options.pairs.push_back(
                Pair{std::string(args[i + 1]), std::string(args[i + 2])}
            );
            i += 2;
        } else if (arg == "--metric") {
            if (!has_value) return false;
            const std::string_view text = args[++i];
            if (text == "butteraugli") {
                options.metric = Metric::Butteraugli;
            } else if (text == "ssimu2") {
                options.metric = Metric::Ssimulacra2;
            } else if (text == "cvvdp") {
                options.metric = Metric::Cvvdp;
            } else {
                std::fprintf(
                    stderr, "bad --metric '%.*s'\n",
                    static_cast<int>(text.size()), text.data()
                );
                return false;
            }
        } else if (arg == "--iterations" || arg == "--warmup" ||
                   arg == "--gpu" || arg == "--qnorm") {
            if (!has_value) return false;
            int value = 0;
            const std::string_view text = args[++i];
            if (!ParseInt(text, value)) {
                std::fprintf(
                    stderr, "bad value for %.*s\n", static_cast<int>(arg.size()),
                    arg.data()
                );
                return false;
            }
            if (arg == "--iterations")
                options.iterations = value;
            else if (arg == "--warmup")
                options.warmup = value;
            else if (arg == "--gpu")
                options.gpu = value;
            else
                options.qnorm = value;
        } else if (arg == "--score-only") {
            options.score_only = true;
        } else if (arg == "--pairs-file") {
            if (!has_value) return false;
            options.pairs_file = std::string(args[++i]);
        } else if (arg == "--limit") {
            if (!has_value) return false;
            int value = 0;
            if (!ParseInt(args[++i], value) || value < 0) {
                std::fprintf(stderr, "bad value for --limit\n");
                return false;
            }
            options.limit = value;
        } else if (arg == "--decode-threads" || arg == "--threads") {
            if (!has_value) return false;
            int value = 0;
            if (!ParseInt(args[++i], value) || value < 0) {
                std::fprintf(
                    stderr, "bad value for %.*s\n", static_cast<int>(arg.size()),
                    arg.data()
                );
                return false;
            }
            if (arg == "--threads") {
                options.threads = value;
            } else {
                options.decode_threads = value;
            }
        } else if (arg == "--throughput-iterations") {
            if (!has_value) return false;
            if (!ParseInt(args[++i], options.throughput_iterations) ||
                options.throughput_iterations < 1) {
                std::fprintf(stderr, "invalid --throughput-iterations\n");
                return false;
            }
        } else if (arg == "--throughput-seconds") {
            if (!has_value) return false;
            const std::string_view text = args[++i];
            double value = 0.0;
            const char* last = text.data() + text.size();
            const std::from_chars_result result =
                std::from_chars(text.data(), last, value);
            if (result.ec != std::errc{} || result.ptr != last ||
                !inmem::ValidWindowSeconds(value)) {
                std::fprintf(
                    stderr,
                    "bad --throughput-seconds '%.*s' (expected a positive, "
                    "finite, representable number of seconds)\n",
                    static_cast<int>(text.size()), text.data()
                );
                return false;
            }
            options.throughput_seconds = value;
        } else if (arg == "--intensity" || arg == "--csv") {
            if (!has_value) return false;
            const std::string text(args[++i]);
            if (arg == "--csv") {
                options.csv_path = text;
                continue;
            }
            float value = 0.0f;
            if (!ParsePositiveFloat(text, value)) {
                std::fprintf(stderr, "bad value for --intensity\n");
                return false;
            }
            options.intensity = value;
        } else if (!arg.empty() && arg.front() == '-') {
            std::fprintf(
                stderr, "unknown option '%.*s'\n", static_cast<int>(arg.size()),
                arg.data()
            );
            return false;
        } else {
            std::fprintf(
                stderr, "unexpected argument '%.*s'\n",
                static_cast<int>(arg.size()), arg.data()
            );
            return false;
        }
    }
    if (options.throughput_iterations > 0 && options.throughput_seconds > 0.0) {
        std::fprintf(stderr, "choose fixed count or throughput window\n");
        return false;
    }
    if (options.iterations < 1 || options.warmup < 0 ||
        options.warmup > std::numeric_limits<int>::max() - options.iterations) {
        std::fprintf(stderr, "invalid --iterations or --warmup\n");
        return false;
    }
    if (options.gpu < 0 || options.qnorm < 0 || options.qnorm > 2) {
        std::fprintf(stderr, "invalid --gpu or --qnorm\n");
        return false;
    }
    if (!options.pairs_file.empty()) {
        std::string error;
        if (!inmem::ReadPairsFile(options.pairs_file, options.pairs, &error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return false;
        }
    }
    if (options.limit > 0 &&
        options.pairs.size() > static_cast<std::size_t>(options.limit)) {
        options.pairs.resize(static_cast<std::size_t>(options.limit));
    }
    if (options.pairs.empty()) {
        std::fprintf(
            stderr, "--pair ORIG DIST or --pairs-file PATH is required\n"
        );
        return false;
    }
    return true;
}

// Vship's sample enum names the depth, not just the storage width, so it has
// to come from the image descriptor. Only the depths its converters implement
// are accepted; PNG gives 8 or 16, y4m can give 9/10/12/14/16.
bool VshipSample(const vqm::ImageFormat& format, Vship_Sample_t* out) {
    if (format.type != vqm::PixelType::Byte &&
        format.type != vqm::PixelType::Word) {
        return false;
    }
    switch (vqm::internal::EffectiveDepth(format)) {
        case 8:
            *out = Vship_SampleUINT8;
            return true;
        case 9:
            *out = Vship_SampleUINT9;
            return true;
        case 10:
            *out = Vship_SampleUINT10;
            return true;
        case 12:
            *out = Vship_SampleUINT12;
            return true;
        case 14:
            *out = Vship_SampleUINT14;
            return true;
        case 16:
            *out = Vship_SampleUINT16;
            return true;
        default:
            return false;
    }
}

// The colours this repository's CLI pins for PNG input: full-range,
// identity-matrix (RGB), BT.709 primaries, sRGB transfer. The depth follows
// the decoded image.
Vship_Colorspace_t Colorspace(
    std::int64_t width, std::int64_t height, Vship_Sample_t sample
) {
    Vship_Colorspace_t out{};
    out.width = width;
    out.height = height;
    out.target_width = width;
    out.target_height = height;
    out.sample = sample;
    out.range = Vship_RangeFull;
    out.subsampling = Vship_ChromaSubsample_t{0, 0};
    out.chromaLocation = Vship_ChromaLoc_Left;
    out.colorFamily = Vship_ColorRGB;
    out.YUVMatrix = Vship_MATRIX_RGB;
    out.transferFunction = Vship_TRC_sRGB;
    out.primaries = Vship_PRIMARIES_BT709;
    out.crop = Vship_CropRectangle_t{0, 0, 0, 0};
    return out;
}

bool MeasureDecode(
    const Pair& pair, int attempts, double* decode_ms, vqm::Image* original,
    vqm::Image* distorted, std::string* error
) {
    std::vector<double> load_times;
    for (int attempt = 0; attempt < attempts; ++attempt) {
        const Clock::time_point start = Clock::now();
        std::optional<vqm::Image> loaded_original =
            vqm::image_io::LoadImage(pair.original, *error);
        std::optional<vqm::Image> loaded_distorted;
        if (loaded_original) {
            loaded_distorted = vqm::image_io::LoadImage(pair.distorted, *error);
        }
        if (loaded_original && loaded_distorted) {
            *original = std::move(*loaded_original);
            *distorted = std::move(*loaded_distorted);
        }
        load_times.push_back(Seconds(Clock::now() - start));
        if (!loaded_distorted) return false;
    }
    *decode_ms = 1000.0 * Median(load_times);
    return true;
}

// The three planes of a decoded 8/16-bit RGB image, which this repository
// stores planar and contiguous. Vship's C API has no alpha plane, so a fourth
// plane is accepted only when it is fully opaque: then the RGB pixels are the
// image, which is what the reference CLIs' alpha blend also yields. Greyscale
// is not expanded here; the corpus and the synthesised pairs are RGB. Line
// sizes are byte strides, which is what Vship's converters index with.
bool PlanePointers(
    const vqm::Image& image, const std::uint8_t* planes[3],
    std::int64_t line_size[3], std::string* error
) {
    if (image.format.family != vqm::ColorFamily::Rgb) {
        *error = "need RGB planes";
        return false;
    }
    const std::array<vqm::internal::PlaneGeometry, 4> layout =
        vqm::internal::PlaneLayout(image.format, image.size);
    const std::size_t bytes = layout[0].sample_bytes;
    if (bytes != 1 && bytes != 2) {
        *error = "unexpected sample size";
        return false;
    }
    if (!layout[1].present() || !layout[2].present() ||
        layout[1].sample_bytes != bytes || layout[2].sample_bytes != bytes) {
        *error = "need three RGB planes";
        return false;
    }
    if (layout[3].present()) {
        const std::size_t pixels = layout[0].width * layout[0].height;
        const std::uint32_t opaque = bytes == 1 ? 255u : 65535u;
        for (std::size_t i = 0; i < pixels; ++i) {
            const std::uint8_t* here =
                image.data.data() + layout[3].offset + i * bytes;
            std::uint32_t value = here[0];
            if (bytes == 2) {
                std::uint16_t word = 0;
                std::memcpy(&word, here, sizeof(word));
                value = word;
            }
            if (value != opaque) {
                *error =
                    "Vship has no alpha plane; a transparent image cannot be "
                    "scored by this driver";
                return false;
            }
        }
    }
    for (std::size_t plane = 0; plane < 3; ++plane) {
        planes[plane] = image.data.data() + layout[plane].offset;
        line_size[plane] = static_cast<std::int64_t>(layout[plane].width) *
                           static_cast<std::int64_t>(bytes);
    }
    return true;
}

struct Result {
    double norm_q = 0.0;
    double norm_3 = 0.0;
    double norm_inf = 0.0;
    double score = 0.0;
};

// One pair decoded for the prefetch queue, with its own load time. The metric
// timer starts after this returns.
struct DecodedPair {
    vqm::Image original;
    vqm::Image distorted;
    double decode_ms = 0.0;
};

bool LoadOnce(const Pair& pair, DecodedPair& out, std::string& error) {
    const Clock::time_point start = Clock::now();
    std::optional<vqm::Image> original =
        vqm::image_io::LoadImage(pair.original, error);
    if (!original) return false;
    std::optional<vqm::Image> distorted =
        vqm::image_io::LoadImage(pair.distorted, error);
    if (!distorted) return false;
    out.decode_ms = 1000.0 * Seconds(Clock::now() - start);
    out.original = std::move(*original);
    out.distorted = std::move(*distorted);
    return true;
}

// One handler per (metric, geometry, qnorm, intensity, gpu), kept for the life
// of the process. Vship's handler owns the device pipeline; creating one per
// pair would put device setup inside a batch's per-pair path, which is exactly
// what a corpus run must not measure.
struct HandlerKey {
    Metric metric = Metric::Butteraugli;
    int width = 0;
    int height = 0;
    int sample1 = 0;
    int sample2 = 0;
    int qnorm = 0;
    int gpu = 0;
    float intensity = 0.0f;

    [[nodiscard]] bool operator<(const HandlerKey& other) const {
        return std::tie(
                   metric, width, height, sample1, sample2, qnorm, gpu, intensity
               ) <
               std::tie(
                   other.metric, other.width, other.height, other.sample1,
                   other.sample2, other.qnorm, other.gpu, other.intensity
               );
    }
};

class HandlerCache {
public:
    HandlerCache() = default;
    HandlerCache(const HandlerCache&) = delete;
    HandlerCache& operator=(const HandlerCache&) = delete;
    ~HandlerCache() {
        for (auto& [key, handler] : butter_) {
            (void)key;
            if (handler.id != 0) Vship_ButteraugliFree(handler);
        }
        for (auto& [key, handler] : cvvdp_) {
            (void)key;
            if (handler != nullptr) Vship_FreeHandler(handler);
        }
        for (auto& [key, handler] : ssimu2_) {
            (void)key;
            if (handler.id != 0) Vship_SSIMU2Free(handler);
        }
    }

    Vship_ButteraugliHandler* Butter(
        const Vship_Colorspace_t& colorspace1,
        const Vship_Colorspace_t& colorspace2, const Options& options,
        std::string* error
    ) {
        HandlerKey key{
            Metric::Butteraugli, 0, 0, 0, 0, options.qnorm, options.gpu,
            options.intensity};
        key.width = static_cast<int>(colorspace1.width);
        key.height = static_cast<int>(colorspace1.height);
        key.sample1 = static_cast<int>(colorspace1.sample);
        key.sample2 = static_cast<int>(colorspace2.sample);
        const auto found = butter_.find(key);
        if (found != butter_.end()) return &found->second;
        Vship_ButteraugliHandler handler{};
        const Vship_Exception init = Vship_ButteraugliInit2(
            &handler, colorspace1, colorspace2, options.qnorm, options.intensity,
            options.gpu
        );
        if (init != Vship_NoError) {
            char message[1024] = {};
            Vship_GetErrorMessage(init, message, sizeof(message));
            *error = message;
            return nullptr;
        }
        return &butter_.emplace(key, handler).first->second;
    }

    Vship_SSIMU2Handler* Ssimu2(
        const Vship_Colorspace_t& colorspace1,
        const Vship_Colorspace_t& colorspace2, const Options& options,
        std::string* error
    ) {
        HandlerKey key{
            Metric::Ssimulacra2, 0, 0, 0, 0, options.qnorm, options.gpu,
            options.intensity};
        key.width = static_cast<int>(colorspace1.width);
        key.height = static_cast<int>(colorspace1.height);
        key.sample1 = static_cast<int>(colorspace1.sample);
        key.sample2 = static_cast<int>(colorspace2.sample);
        const auto found = ssimu2_.find(key);
        if (found != ssimu2_.end()) return &found->second;
        Vship_SSIMU2Handler handler{};
        const Vship_Exception init =
            Vship_SSIMU2Init2(&handler, colorspace1, colorspace2, options.gpu);
        if (init != Vship_NoError) {
            char message[1024] = {};
            Vship_GetErrorMessage(init, message, sizeof(message));
            *error = message;
            return nullptr;
        }
        return &ssimu2_.emplace(key, handler).first->second;
    }

    // Public C API only: .reference/Vship/src/VshipAPI.h:153-167,
    // 190-209. The fixed model, frame rate and disabled resizing are this
    // driver's explicit still-image workload; no peer mechanism is copied.
    Vship_Handle Cvvdp(
        const Vship_Colorspace_t& colorspace1,
        const Vship_Colorspace_t& colorspace2, const Options& options,
        std::string* error
    ) {
        HandlerKey key{Metric::Cvvdp, 0, 0, 0, 0, 0, options.gpu, 0};
        key.width = static_cast<int>(colorspace1.width);
        key.height = static_cast<int>(colorspace1.height);
        key.sample1 = static_cast<int>(colorspace1.sample);
        key.sample2 = static_cast<int>(colorspace2.sample);
        const auto found = cvvdp_.find(key);
        if (found != cvvdp_.end()) return found->second;
        Vship_Handle handler = nullptr;
        Vship_InitCVVDP_1 init{
            Vship_StructInitCVVDP_1, colorspace1, colorspace2, 1.0f, false,
            "standard_4k",           nullptr,     options.gpu};
        const auto result = Vship_InitHandler(&handler, &init);
        if (result != Vship_NoError) {
            char message[1024] = {};
            Vship_GetErrorMessage(result, message, sizeof(message));
            *error = message;
            return nullptr;
        }
        return cvvdp_.emplace(key, handler).first->second;
    }

private:
    std::map<HandlerKey, Vship_Handle> cvvdp_;
    std::map<HandlerKey, Vship_ButteraugliHandler> butter_;
    std::map<HandlerKey, Vship_SSIMU2Handler> ssimu2_;
};

// One pair decoded for a batch or throughput run, with the plane pointers,
// line sizes and colourspaces Vship's API takes. They are derived here, on the
// decode thread, so the timed call is Vship's metric only. `width`/`height`
// are what the driver reports as the measured geometry.
struct Item {
    DecodedPair decoded;
    const std::uint8_t* srcp1[3] = {};
    const std::uint8_t* srcp2[3] = {};
    std::int64_t line1[3] = {};
    std::int64_t line2[3] = {};
    Vship_Colorspace_t colorspace1{};
    Vship_Colorspace_t colorspace2{};
    int width = 0;
    int height = 0;
    double decode_ms = 0.0;
};

bool DescribeItem(
    const vqm::Image& original, const vqm::Image& distorted, Item& out,
    std::string& error
) {
    if (!PlanePointers(original, out.srcp1, out.line1, &error) ||
        !PlanePointers(distorted, out.srcp2, out.line2, &error)) {
        return false;
    }
    if (original.size != distorted.size) {
        error = "size mismatch";
        return false;
    }
    Vship_Sample_t sample1 = Vship_SampleUINT8;
    Vship_Sample_t sample2 = Vship_SampleUINT8;
    if (!VshipSample(original.format, &sample1) ||
        !VshipSample(distorted.format, &sample2)) {
        error = "unsupported sample type or depth";
        return false;
    }
    out.colorspace1 = Colorspace(
        static_cast<std::int64_t>(original.size.width),
        static_cast<std::int64_t>(original.size.height), sample1
    );
    out.colorspace2 = Colorspace(
        static_cast<std::int64_t>(distorted.size.width),
        static_cast<std::int64_t>(distorted.size.height), sample2
    );
    out.width = static_cast<int>(original.size.width);
    out.height = static_cast<int>(original.size.height);
    return true;
}

bool LoadItem(const inmem::Pair& pair, Item& out, std::string& error) {
    const auto start = Clock::now();
    if (!LoadOnce(pair, out.decoded, error)) return false;
    out.decode_ms = 1000.0 * Seconds(Clock::now() - start);
    return DescribeItem(out.decoded.original, out.decoded.distorted, out, error);
}

// One handler cache per worker, created in that worker on first use: Vship's
// handler owns the device pipeline and its detailed-error lookups are
// handler-scoped, so a handler is never shared between threads.
using HandlerGrid = std::vector<std::unique_ptr<HandlerCache>>;

HandlerCache& HandlerFor(HandlerGrid& grid, int worker) {
    std::unique_ptr<HandlerCache>& slot = grid[static_cast<std::size_t>(worker)];
    if (!slot) slot = std::make_unique<HandlerCache>();
    return *slot;
}

// One Vship metric call on a worker's handler.
bool ComputeOnce(
    HandlerCache& cache, Item& item, const Options& options, Result* out,
    std::string* error
) {
    if (options.metric == Metric::Cvvdp) {
        const auto handler =
            cache.Cvvdp(item.colorspace1, item.colorspace2, options, error);
        if (handler == nullptr) return false;
        // Both resets are part of the public contract. Calling each also
        // avoids relying on a particular release's interpretation of Reset.
        auto result = Vship_Reset(handler);
        if (result == Vship_NoError) result = Vship_ResetScore(handler);
        Vship_ScoreCVVDP score{Vship_StructScoreCVVDP, 0.0, nullptr, 0};
        if (result == Vship_NoError)
            result = Vship_ComputeHandler(
                handler, &score, item.srcp1, item.srcp2, item.line1, item.line2
            );
        if (result != Vship_NoError) {
            char message[1024] = {};
            Vship_GetDetailedLastErrorHandler(handler, message, sizeof(message));
            *error = message;
            return false;
        }
        *out = Result{0.0, 0.0, 0.0, score.score};
        return true;
    }
    if (options.metric == Metric::Butteraugli) {
        Vship_ButteraugliHandler* handler =
            cache.Butter(item.colorspace1, item.colorspace2, options, error);
        if (handler == nullptr) return false;
        Vship_ButteraugliScore score{};
        const Vship_Exception result = Vship_ComputeButteraugli(
            *handler, &score, nullptr, 0, item.srcp1, item.srcp2, item.line1,
            item.line2
        );
        if (result != Vship_NoError) {
            char message[1024] = {};
            Vship_ButteraugliGetDetailedLastError(
                *handler, message, sizeof(message)
            );
            *error = message;
            return false;
        }
        *out = Result{score.normQ, score.norm3, score.norminf, score.norminf};
        return true;
    }
    Vship_SSIMU2Handler* handler =
        cache.Ssimu2(item.colorspace1, item.colorspace2, options, error);
    if (handler == nullptr) return false;
    double score = 0.0;
    const Vship_Exception result = Vship_ComputeSSIMU2(
        *handler, &score, item.srcp1, item.srcp2, item.line1, item.line2
    );
    if (result != Vship_NoError) {
        char message[1024] = {};
        Vship_SSIMU2GetDetailedLastError(*handler, message, sizeof(message));
        *error = message;
        return false;
    }
    *out = Result{score, score, score, score};
    return true;
}

bool MeasureItem(
    HandlerCache& cache, Item& item, const Options& options,
    std::vector<double>& samples, Result& last, double& checksum,
    std::string& error
) {
    for (int iteration = 0; iteration < options.warmup + options.iterations;
         ++iteration) {
        const auto start = Clock::now();
        Result result;
        if (!ComputeOnce(cache, item, options, &result, &error)) return false;
        const double seconds = Seconds(Clock::now() - start);
        if (iteration >= options.warmup) samples.push_back(seconds);
        checksum += result.score;
        last = result;
    }
    return true;
}

// The fixed window: decode once outside the timer, then keep `threads`
// handlers busy for `options.throughput_seconds` and print one machine line.
int RunWindow(const Options& options, int threads, int decode_threads) {
    HandlerGrid handlers(static_cast<std::size_t>(threads));
    inmem::ThroughputStats stats;
    std::string error;
    const auto score = [&options, &handlers](
                           Item& item, std::size_t, int worker, double* value,
                           double* core_ms, std::string& message
                       ) {
        *core_ms = 0.0;
        Result result;
        if (!ComputeOnce(
                HandlerFor(handlers, worker), item, options, &result, &message
            )) {
            return false;
        }
        *value = result.score;
        return true;
    };
    if (!inmem::RunThroughput<Item>(
            options.pairs, threads, decode_threads, options.throughput_seconds,
            LoadItem, score, &stats, &error, options.throughput_iterations,
            options.throughput_iterations > 0 ? options.warmup : 1
        )) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    inmem::EmitThroughput(
        stats,
        options.metric == Metric::Butteraugli ? "butteraugli"
        : options.metric == Metric::Cvvdp     ? "cvvdp"
                                              : "ssimu2",
        std::nullopt
    );
    return 0;
}

// The batch path: `threads` handlers score the list and the rows are emitted in
// input order afterwards, exactly as the serial loop prints them.
int RunBatchPath(const Options& options, int threads, int decode_threads) {
    HandlerGrid handlers(static_cast<std::size_t>(threads));
    std::vector<Result> per_pair(options.pairs.size());
    const auto score = [&options, &handlers, &per_pair](
                           Item& item, std::size_t index, int worker,
                           double* value, double* core_ms, std::string& message
                       ) {
        HandlerCache& cache = HandlerFor(handlers, worker);
        std::vector<double> core_times;
        core_times.reserve(static_cast<std::size_t>(options.iterations));
        Result last;
        double checksum = 0.0;
        if (!MeasureItem(
                cache, item, options, core_times, last, checksum, message
            ))
            return false;
        *value = last.score;
        *core_ms = 1000.0 * inmem::Median(core_times);
        per_pair[index] = last;
        return true;
    };
    std::vector<inmem::PairResult> results;
    std::string error;
    if (!inmem::RunBatch<Item>(
            options.pairs, threads, decode_threads, LoadItem, score, &results,
            &error
        )) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::ofstream csv;
    if (!options.csv_path.empty()) {
        csv.open(options.csv_path);
        if (!csv) {
            std::fprintf(
                stderr, "could not write %s\n", options.csv_path.c_str()
            );
            return 1;
        }
        csv << "original,distorted,width,height,decode_ms,core_ms,score,"
               "normQ,norm3,norminf\n";
    }
    int failures = 0;
    for (std::size_t index = 0; index < options.pairs.size(); ++index) {
        const inmem::Pair& pair = options.pairs[index];
        const inmem::PairResult& result = results[index];
        if (!result.ok) {
            std::fprintf(
                stderr, "%s: %s\n", pair.original.c_str(), result.error.c_str()
            );
            std::puts("ERROR");
            ++failures;
            continue;
        }
        const Result& last = per_pair[index];
        std::printf("%.8f\n", last.score);
        if (csv) {
            csv << std::format(
                "{},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n",
                inmem::CsvField(pair.original), inmem::CsvField(pair.distorted),
                result.width, result.height, result.decode_ms, result.core_ms,
                last.score, last.norm_q, last.norm_3, last.norm_inf
            );
        }
    }
    if (!options.csv_path.empty()) {
        csv.close();
        if (!csv) {
            std::fprintf(
                stderr, "could not write %s\n", options.csv_path.c_str()
            );
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string_view program = argc > 0 ? argv[0] : "vship_bench";
    std::vector<std::string_view> args;
    args.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);

    if (args.size() == 1 && args[0] == "--device-info") {
        return PrintDeviceInfo();
    }

    Options options;
    if (!ParseArgs(args, options)) {
        std::fputs(Usage(program).c_str(), stderr);
        return 2;
    }
    if (options.help) {
        std::fputs(Usage(program).c_str(), stdout);
        return 0;
    }

    int devices = 0;
    if (Vship_GetDeviceCount(&devices) != Vship_NoError || devices == 0) {
        std::fprintf(stderr, "vship: no device\n");
        return 1;
    }

    // A throughput or batch run prints only its own lines: the device banner
    // belongs to the latency table, where a human reads it.
    const bool throughput =
        options.throughput_seconds > 0.0 || options.throughput_iterations > 0;
    const bool batch = options.score_only || throughput;
    const int threads = inmem::ResolveThreads(options, batch);
    const int decode_threads = inmem::ResolveDecodeThreads(options, batch);
    if (!options.score_only && !throughput) {
        Vship_DeviceInfo info{};
        if (Vship_GetDeviceInfo(&info, options.gpu) == Vship_NoError) {
            const Vship_Version version = Vship_GetVersion();
            std::printf(
                "vship %d.%d.%d, device %d: %s\n", version.major, version.minor,
                version.minorMinor, options.gpu, info.name
            );
        }
    }
    if (throughput) {
        return RunWindow(options, threads, decode_threads);
    }
    if (options.score_only)
        return RunBatchPath(options, threads, decode_threads);

    // Latency path: one handler on this thread, decoding ahead so the device is
    // not idle between pairs, and the human table.
    if (options.metric == Metric::Cvvdp) {
        std::puts(
            "CVVDP stills: standard_4k, fps=1, resize=false, history+score "
            "reset per call"
        );
        std::printf(
            "%-60s %9s %12s %12s\n", "pair", "decode_ms", "core_ms", "JOD"
        );
    } else {
        std::printf(
            "%-60s %9s %9s %10s %10s %10s\n", "pair", "decode_ms", "core_ms",
            "normQ", "norm3", "norminf"
        );
    }
    std::vector<double> decodes;
    std::vector<double> cores;
    double checksum = 0.0;
    int failures = 0;
    std::ofstream csv;
    if (!options.csv_path.empty()) {
        csv.open(options.csv_path);
        if (!csv) {
            std::fprintf(
                stderr, "could not write %s\n", options.csv_path.c_str()
            );
            return 1;
        }
        csv << "original,distorted,width,height,decode_ms,core_ms,score,"
               "normQ,norm3,norminf\n";
    }
    HandlerCache handlers;
    // Decode ahead on a small pool while the metric runs on this thread, so
    // one handler and one device serve the whole list and the GPU is not idle
    // between pairs.
    std::optional<inmem::Prefetch<DecodedPair>> prefetch;
    const int latency_decode_threads = options.decode_threads > 0
                                           ? options.decode_threads
                                           : std::min(4, decode_threads);
    if (latency_decode_threads > 1) {
        prefetch.emplace(options.pairs, latency_decode_threads, LoadOnce);
    }
    for (const Pair& pair : options.pairs) {
        vqm::Image original;
        vqm::Image distorted;
        std::string decode_error;
        double decode_ms = 0.0;
        bool decoded = false;
        if (prefetch) {
            DecodedPair pair_images;
            decoded = prefetch->Take(pair_images, &decode_error);
            if (decoded) {
                decode_ms = pair_images.decode_ms;
                original = std::move(pair_images.original);
                distorted = std::move(pair_images.distorted);
            }
        } else {
            decoded = MeasureDecode(
                pair, 3, &decode_ms, &original, &distorted, &decode_error
            );
        }
        if (!decoded) {
            std::fprintf(
                stderr, "%s: %s\n", pair.original.c_str(), decode_error.c_str()
            );
            ++failures;
            continue;
        }
        Item item;
        if (!DescribeItem(original, distorted, item, decode_error)) {
            std::fprintf(
                stderr, "%s: %s\n", pair.original.c_str(), decode_error.c_str()
            );
            ++failures;
            continue;
        }
        std::vector<double> core_times;
        Result last;
        double pair_checksum = 0.0;
        bool ok = true;
        try {
            std::string init_error;
            ok = MeasureItem(
                handlers, item, options, core_times, last, pair_checksum,
                init_error
            );
            if (!ok)
                std::fprintf(
                    stderr, "%s: %s\n", pair.original.c_str(), init_error.c_str()
                );
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s: %s\n", pair.original.c_str(), e.what());
            ok = false;
        }
        if (!ok) {
            ++failures;
            continue;
        }
        const double core_ms = 1000.0 * Median(core_times);
        decodes.push_back(decode_ms);
        cores.push_back(core_ms);
        checksum += pair_checksum;
        if (options.metric == Metric::Cvvdp) {
            std::printf(
                "%-60s %9.1f %12.6f %12.8f\n",
                (pair.original + " vs " + pair.distorted).c_str(), decode_ms,
                core_ms, last.score
            );
        } else {
            std::printf(
                "%-60s %9.1f %9.1f %10.6f %10.6f %10.6f\n",
                (pair.original + " vs " + pair.distorted).c_str(), decode_ms,
                core_ms, last.norm_q, last.norm_3, last.norm_inf
            );
        }
        if (csv) {
            csv << std::format(
                "{},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n",
                inmem::CsvField(pair.original), inmem::CsvField(pair.distorted),
                original.size.width, original.size.height, decode_ms, core_ms,
                last.score, last.norm_q, last.norm_3, last.norm_inf
            );
        }
    }
    if (decodes.empty()) {
        std::fprintf(stderr, "no pair could be measured\n");
        return 1;
    }
    const double median_decode = Median(decodes);
    const double median_core = Median(cores);
    std::printf(
        "aggregate: median decode %.1f ms, median core %.1f ms, median "
        "total %.1f ms\n",
        median_decode, median_core, median_decode + median_core
    );
    if (median_core > 0.0) {
        std::printf(
            "reciprocal latency: %.1f pairs/s, %.2f ms/pair\n",
            1000.0 / median_core, median_core
        );
    }
    std::printf("checksum %.6f\n", checksum);
    if (!options.csv_path.empty()) {
        csv.close();
        if (!csv) {
            std::fprintf(
                stderr, "could not write %s\n", options.csv_path.c_str()
            );
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
