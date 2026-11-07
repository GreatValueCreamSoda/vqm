// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "butteraugli.h"
#include "cvvdp.h"
#include "cvvdp_options.h"
#include "io.h"
#include "options.h"
#include "video.h"
#include "vqm.h"
#include "vqm_alpha.h"
#include "vulkan/vulkan_support.h"

namespace vqm::cli {
namespace {

struct MetricEntry {
    Metric metric;
    std::string_view name;
};

constexpr std::array kMetrics{
    MetricEntry{Metric::Ssimulacra2, "ssimulacra2"},
    MetricEntry{Metric::Butteraugli, "butteraugli"},
    MetricEntry{Metric::VmafV0, "vmaf-v0"},
    MetricEntry{Metric::VmafV1, "vmaf-v1"},
    MetricEntry{Metric::Cvvdp, "cvvdp"},
    MetricEntry{Metric::PsnrHvs, "psnr-hvs"},
};

bool HasMetric(std::span<const Metric> metrics, Metric metric) {
    return std::ranges::find(metrics, metric) != metrics.end();
}

void AppendMetric(std::vector<Metric>& metrics, Metric metric) {
    if (!HasMetric(metrics, metric)) metrics.push_back(metric);
}

std::optional<ParseError> ParseMetricList(
    std::string_view value, std::vector<Metric>& metrics
) {
    for (;;) {
        const auto separator = value.find(',');
        const auto name = value.substr(0, separator);
        if (name == "all") {
            for (const auto& entry : kMetrics)
                AppendMetric(metrics, entry.metric);
        } else {
            const auto entry =
                std::ranges::find(kMetrics, name, &MetricEntry::name);
            if (entry == kMetrics.end()) {
                return ParseError{std::format(
                    "invalid value for --metric: '{}' (expected "
                    "ssimulacra2|butteraugli|vmaf-v0|vmaf-v1|cvvdp|psnr-hvs|"
                    "all)",
                    name
                )};
            }
            AppendMetric(metrics, entry->metric);
        }
        if (separator == std::string_view::npos) return std::nullopt;
        value.remove_prefix(separator + 1);
    }
}

}  // namespace

std::string_view MetricName(Metric metric) {
    const auto entry = std::ranges::find(kMetrics, metric, &MetricEntry::metric);
    if (entry == kMetrics.end()) throw std::logic_error("unknown metric");
    return entry->name;
}

std::string Usage(std::string_view program) {
    return std::format(
        R"help(VQM - SSIMULACRA 2.1 / Butteraugli / VMAF / ColorVideoVDP / PSNR-HVS (Vulkan)
Usage: {} [--metric METRIC[,METRIC...]] [--raw|--json] [--verbose]
          [--half-planes] [--nvdec] [--iterations N] [--warmup N]
          [--chroma-filter KERNEL] [--intensity-target NITS]
          [--hf-asymmetry F]
          [--cvvdp-display standard-4k|standard-fhd|standard-hdr] [--cvvdp-ppd N]
          [--cvvdp-peak NITS] [--cvvdp-contrast RATIO] [--cvvdp-ambient LUX]
          [--input-matrix M] [--input-transfer T] [--input-primaries P] [--input-range R]
          original distorted

Batch clips: use --json --pairs-file PAIRS.tsv instead of positional paths.
Each TSV row contains sample ID, original path and distorted path, separated by tabs.
JSON prints one complete clip object per pair and metric, in selection order.
Each object contains ordered frame_results, per-metric core_seconds and the
pair's shared decode_seconds. --iterations repeats whole reset clips and
reports the iteration with median combined core time. --warmup runs untimed
clips first.

--metric selects one or more of ssimulacra2 (default), butteraugli, vmaf-v0,
vmaf-v1, cvvdp and psnr-hvs. Repeat --metric or use a comma-separated list.
The alias all selects all six metrics. Each selected metric must support the
input pair. Duplicate selections are scored once, in first-seen order.
Multiple metrics print labeled scores in text mode. --raw requires one metric.
--verbose (-v) prints an aligned table with clip scores and frame statistics.
Scoring FPS divides the clip's frame count by the combined core scoring time.
Aggregate shows each metric's clip score. Frame min and max use all frames.
Finite stddev is the population standard deviation of finite frame
scores. Perfect counts frames that reached the metric's best score.
With --json it adds scoring_fps, frame_stats and CVVDP's diagnostic frame
JOD values after temporal filtering. --verbose cannot be combined with --raw.

SSIMULACRA 2 prints a score in -inf..100, higher is better. Butteraugli prints
its diffmap maximum (0 for identical images, larger is worse). With --raw it
also prints its distance_p3 aggregate.
--intensity-target and --hf-asymmetry configure Butteraugli's libjxl parameters
(defaults 80 nits and 1.0).
CVVDP prints JOD (10 for identical, higher is better; negative values are valid).
VMAF v0.6.1 and v1 use their native input geometry and coded input formats.
SSIMULACRA 2, Butteraugli and VMAF average their frame scores.
CVVDP uses the original standard_4k display by default. Physical display/viewing
options are independent of image geometry. CVVDP reads all frames of progressive
Y4M inputs, using their matching rational frame rates and one final clip score.
It requires opaque inputs; PNG/JPEG pairs use the true single-image algorithm.
For CVVDP, linear RGB samples express absolute cd/m^2. SDR, PQ and HLG use
the configured physical display; special CL and ICtCp encodings are rejected.

PSNR-HVS reports dB (higher is better) from matching opaque native planar YUV
4:2:0, 4:2:2 or 4:4:4 inputs, at 8, 10 or 12 bits; every plane must be at least
8x8. RGB conversion is unavailable. Pixel codes are measured directly, without
range, matrix or transfer conversion. For clips, the CLI averages weighted
frame errors before converting to dB. The clip score is inf only when every
frame has zero measured error. JSON represents infinity as "inf".
--raw also prints PSNR-HVS Y, Cb and Cr dB, then normalized plane errors.

Inputs may be PNG (8/16-bit), JPEG or progressive YUV4MPEG2.
Every metric reads all Y4M frames and requires equal frame counts and frame rates.
Y4M chroma tags include Cmono, C420, C420jpeg, C420mpeg2, C420paldv, C422, C444,
C411 and C444alpha. Supported high-bit-depth tags use p9, p10, p12, p14 or p16.
Each metric also checks its own input requirements.
PNG and JPEG use sRGB with BT.709 primaries. Y4M defaults to the bt470bg matrix,
bt709 transfer and primaries, and limited range. Use --input-range full for
full-range Y4M because the XCOLORRANGE header extension is ignored.
Builds with NVDEC automatically decode compressed video inputs on an NVIDIA GPU
and keep decoded frames on that GPU for scoring. --nvdec requires both inputs
to be compressed videos. The clips must be progressive and have matching
dimensions and frame rates. Configure with -DVQM_ENABLE_NVDEC=OFF to build
without NVDEC.

--half-planes stores SSIMULACRA 2 intermediate planes as 16-bit floats and can
change scores. It requires device support for 16-bit storage.
--chroma-filter picks the chroma upsampling kernel: bilinear (default), bicubic
or lanczos
--input-matrix     rgb|bt709|fcc|bt470bg|st170m|st240m|ycgco|bt2020ncl|
                   bt2020cl|chromaticity-derived-ncl|chromaticity-derived-cl|ictcp
--input-transfer   bt709|bt601|bt470m|bt470bg|st240m|linear|log100|log316|xvycc|
                   bt1361|srgb|bt2020-10|bt2020-12|st2084|st428|arib-b67
--input-primaries  bt709|bt470m|bt470bg|st170m|st240m|film|bt2020|st428|st431-2|
                   st432-1|ebu3213e
--input-range      limited|full
The --input-* flags override what the decoder declared, on both images.

SSIMULACRA 2 returns a score in -inf..100, correlated to subjective visual quality:
     negative scores: extremely low quality, very strong distortion
     10 = very low quality (average output of cjxl -d 14 / -q 12 or libjpeg-turbo quality 14, 4:2:0)
     30 = low quality (average output of cjxl -d 9 / -q 20 or libjpeg-turbo quality 20, 4:2:0)
     50 = medium quality (average output of cjxl -d 5 / -q 45 or libjpeg-turbo quality 35, 4:2:0)
     70 = high quality (hard to notice artifacts without comparison to the original,
                        average output of cjxl -d 2.5 / -q 73 or libjpeg-turbo quality 70, 4:2:0)
     80 = very high quality (impossible to distinguish from the original in a side-by-side comparison at 1:1,
                             average output of cjxl -d 1.5 / -q 85 or libjpeg-turbo quality 85, 4:2:2)
     85 = excellent quality (impossible to distinguish from the original in a flip test at 1:1,
                             average output of cjxl -d 1 / -q 90 or libjpeg-turbo quality 90, 4:4:4)
     90 = visually lossless (impossible to distinguish from the original in a flicker test at 1:1,
                             average output of cjxl -d 0.5 / -q 95 or libjpeg-turbo quality 95, 4:4:4)
     100 = mathematically lossless
)help",
        program
    );
}

ParseResult ParseOptions(std::span<const std::string_view> args) {
    Options options;
    std::vector<std::string> positional;
    std::vector<Metric> metrics;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--raw") {
            options.raw = true;
        } else if (arg == "--json") {
            options.json = true;
        } else if (arg == "--verbose" || arg == "-v") {
            options.verbose = true;
        } else if (arg == "--nvdec") {
            options.nvdec = true;
        } else if (arg == "--pairs-file") {
            if (i + 1 >= args.size())
                return ParseError{"--pairs-file needs a value"};
            options.pairs_file = args[++i];
        } else if (arg == "--iterations" || arg == "--warmup") {
            if (i + 1 >= args.size())
                return ParseError{std::format("{} needs a value", arg)};
            const auto value = args[++i];
            unsigned count = 0;
            const auto parsed = std::from_chars(
                value.data(), value.data() + value.size(), count
            );
            if (parsed.ec != std::errc{} ||
                parsed.ptr != value.data() + value.size() ||
                (arg == "--iterations" && count == 0))
                return ParseError{
                    std::format("invalid value for {}: '{}'", arg, value)};
            if (arg == "--iterations")
                options.iterations = count;
            else
                options.warmup = count;
        } else if (arg == "--half-planes") {
            options.half_planes = true;
        } else if (arg == "--metric") {
            if (i + 1 >= args.size())
                return ParseError{"--metric needs a value"};
            if (auto error = ParseMetricList(args[++i], metrics))
                return std::move(*error);
        } else if (tool::IsCvvdpOption(arg)) {
            if (i + 1 >= args.size())
                return ParseError{std::format("{} needs a value", arg)};
            std::string error;
            if (!tool::ParseCvvdpOption(arg, args[++i], options.cvvdp, error))
                return ParseError{std::move(error)};
        } else if (arg == "--intensity-target" || arg == "--hf-asymmetry") {
            if (i + 1 >= args.size()) {
                return ParseError{std::format("{} needs a value", arg)};
            }
            const std::string_view value = args[++i];
            float parsed = 0.0f;
            std::size_t consumed = 0;
            try {
                parsed = std::stof(std::string(value), &consumed);
            } catch (const std::exception&) {
                return ParseError{std::format(
                    "invalid value for {}: '{}' (expected a finite positive "
                    "number)",
                    arg, value
                )};
            }
            if (consumed != value.size() || !std::isfinite(parsed) ||
                parsed <= 0.0f) {
                return ParseError{std::format(
                    "invalid value for {}: '{}' (expected a finite positive "
                    "number)",
                    arg, value
                )};
            }
            if (arg == "--intensity-target") {
                options.intensity_target = parsed;
            } else {
                options.hf_asymmetry = parsed;
            }
        } else if (arg == "--help" || arg == "-h") {
            return HelpRequested{};
        } else if (const auto input_option =
                       image_io::IdentifyInputColorOption(arg)) {
            if (i + 1 >= args.size()) {
                return ParseError{std::format("{} needs a value", arg)};
            }
            const std::string_view value = args[++i];
            std::string accepted;
            if (!image_io::ParseInputColorOption(
                    *input_option, value, options.input, accepted
                )) {
                return ParseError{std::format(
                    "invalid value for {}: '{}' (expected {})", arg, value,
                    accepted
                )};
            }
        } else if (!arg.empty() && arg.front() == '-') {
            return ParseError{std::format("unknown option '{}'", arg)};
        } else {
            positional.emplace_back(arg);
        }
    }
    if (!metrics.empty()) options.metrics = std::move(metrics);
    if (options.cvvdp.specified && !HasMetric(options.metrics, Metric::Cvvdp))
        return ParseError{"--cvvdp-* options require --metric cvvdp"};
    if (options.half_planes && !HasMetric(options.metrics, Metric::Ssimulacra2))
        return ParseError{"--half-planes requires ssimulacra2"};
    if (options.json && options.raw)
        return ParseError{"--json and --raw cannot be combined"};
    if (options.verbose && options.raw)
        return ParseError{"--verbose and --raw cannot be combined"};
    if (options.raw && options.metrics.size() != 1)
        return ParseError{"--raw requires exactly one metric"};
    if (!options.pairs_file.empty()) {
        if (!options.json) return ParseError{"--pairs-file requires --json"};
        if (!positional.empty())
            return ParseError{"--pairs-file replaces positional paths"};
        return options;
    }
    if (positional.size() != 2) {
        return ParseError{std::format(
            "expected exactly two image paths, got {}", positional.size()
        )};
    }
    options.original = std::move(positional[0]);
    options.distorted = std::move(positional[1]);
    return options;
}

}  // namespace vqm::cli

namespace {

// The reference tool's tuning build prints the 108 norms, channel-major.
void PrintRawNorms(const vqm::Ssimulacra2Result& msssim) {
    for (std::size_t c = 0; c < 3; ++c) {
        for (const auto& scale : msssim.scales) {
            for (std::size_t n = 0; n < 2; ++n) {
                std::printf(
                    "%.12f,%.12f,%.12f,", scale.avg_ssim[c * 2 + n],
                    scale.avg_edgediff[c * 4 + n],
                    scale.avg_edgediff[c * 4 + 2 + n]
                );
            }
        }
    }
    std::printf("\n");
}

int RunButteraugli(
    const vqm::cli::Options& options, const vqm::Image& original,
    const vqm::Image& distorted
) {
    if (options.half_planes) {
        std::fprintf(stderr, "--half-planes is ssimulacra2 only\n");
        return 2;
    }
    vqm::ButteraugliConfig config;
    config.size = original.size;
    config.chroma_filter = options.input.chroma_filter;
    config.hf_asymmetry = options.hf_asymmetry;
    config.intensity_target = options.intensity_target;
    vqm::ButteraugliPipeline pipeline(config);
    const vqm::ButteraugliScore result =
        vqm::ComputeWorstBackground(pipeline, original, distorted);
    if (options.raw) std::printf("%.8f\n", result.distance_p3);
    std::printf("%.8f\n", result.score);
    return 0;
}

int RunSsimulacra2(
    const vqm::cli::Options& options, const vqm::Image& original,
    const vqm::Image& distorted
) {
    vqm::Ssimulacra2Config config;
    config.size = original.size;
    config.half_planes = options.half_planes;
    config.chroma_filter = options.input.chroma_filter;
    vqm::Ssimulacra2Pipeline pipeline(config);
    const vqm::Ssimulacra2Result result =
        vqm::ComputeWorstBackground(pipeline, original, distorted);
    if (options.raw) PrintRawNorms(result);
    std::printf("%.8f\n", result.Score());
    return 0;
}

}  // namespace

namespace {

int RunCliMain(int argc, char** argv) try {
    const std::string_view program = argc > 0 ? argv[0] : "vqm";
    std::vector<std::string_view> args;
    args.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);

    const vqm::cli::ParseResult result = vqm::cli::ParseOptions(args);
    if (std::holds_alternative<vqm::cli::HelpRequested>(result)) {
        std::fputs(vqm::cli::Usage(program).c_str(), stdout);
        return 0;
    }
    if (const auto* parse_error = std::get_if<vqm::cli::ParseError>(&result)) {
        std::fprintf(
            stderr, "%s\n%s", parse_error->message.c_str(),
            vqm::cli::Usage(program).c_str()
        );
        return 2;
    }
    const vqm::cli::Options& options = std::get<vqm::cli::Options>(result);
    const auto metric = options.metrics.front();

    if (options.metrics.size() > 1 || options.nvdec || options.json ||
        options.verbose || metric == vqm::cli::Metric::Cvvdp ||
        metric == vqm::cli::Metric::PsnrHvs ||
        metric == vqm::cli::Metric::VmafV0 ||
        metric == vqm::cli::Metric::VmafV1 || options.iterations != 1 ||
        options.warmup != 0)
        return vqm::cli::RunSequences(options);

    if (vqm::cli::TryRunNvdecPair(options)) return 0;

    std::string error;
    std::optional<vqm::Image> original =
        vqm::image_io::LoadImage(options.original, error);
    if (!original) {
        std::fprintf(
            stderr, "Could not load original image: %s\n%s\n",
            options.original.c_str(), error.c_str()
        );
        return 1;
    }
    // No 8x8 pre-check here: Ssimulacra2Pipeline's constructor enforces
    // kMinDimension and throws with the same message this used to print.
    std::optional<vqm::Image> distorted =
        vqm::image_io::LoadImage(options.distorted, error);
    if (!distorted) {
        std::fprintf(
            stderr, "Could not load distorted image: %s\n%s\n",
            options.distorted.c_str(), error.c_str()
        );
        return 1;
    }
    if (original->size != distorted->size) {
        std::fprintf(stderr, "Image size mismatch\n");
        return 1;
    }

    options.input.ApplyTo(original->format);
    options.input.ApplyTo(distorted->format);

    if (vqm::cli::HasY4mInput(options)) return vqm::cli::RunSequences(options);

    if (metric == vqm::cli::Metric::Butteraugli) {
        return RunButteraugli(options, *original, *distorted);
    }
    return RunSsimulacra2(options, *original, *distorted);
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    const int status = RunCliMain(argc, argv);
    // Destroy automatic pipelines before releasing the retained device, while
    // driver exit handlers are still idle. Long video batches can initialize
    // additional driver components after the fallback exit handler is installed.
    try {
        vqm::internal::ShutdownDefaultVulkanContext();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CLI shutdown: %s\n", error.what());
        return 1;
    }
    return status;
}
