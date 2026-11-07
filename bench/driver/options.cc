// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <charconv>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <system_error>

#include "bench.h"
#include "io.h"

namespace bench_driver {

std::string Usage(std::string_view program) {
    return std::format(
        R"help(Usage: {} [--metric ssimulacra2|butteraugli|vmaf-v0|vmaf-v1|cvvdp|psnr-hvs] [--pair ORIG DIST]...
          [--pairs-file PATH] [--iterations N] [--warmup N] [--resident N] [--resident-ring N] [--resident-reference] [--workers N] [--csv PATH] [--stages-csv PATH]
          [--half-planes] [--score-only] [--host-views] [--vulkan-buffers] [--decode-threads N] [--pool]
          [--throughput-seconds S] [--vmaf-chroma native|420|444]
          [--vmaf-fit native|pad] [--vmaf-model 1080|4k] [--vmaf-neg] [--vmaf-transform] [--vmaf-no-temporal]
          [--metric-pair independent|shared|unified]
          [--cvvdp-video] [--cvvdp-display standard|compact]
          [--cvvdp-ppd N] [--cvvdp-peak NITS] [--cvvdp-contrast RATIO] [--cvvdp-ambient LUX]
          [--cvvdp-aspect W:H] [--cvvdp-diagonal INCHES] [--cvvdp-distance METRES]
          [--intensity-target NITS] [--hf-asymmetry F] [--device-info] [--help]

--resident N scores resident Vulkan images (N >= 2 slots, one per pair member)
instead of re-uploading the pair on every iteration.
SSIMULACRA 2 alternates duplicate reference slots so both input
pyramids are rebuilt each call; its resident mode needs at least
three internal slots, allocated automatically.
--resident-ring N preloads 2..N input pairs on one pipeline and
cycles them for a fixed number of calls, with uploads outside timing.
--resident-reference uploads only the original before timing and
also rebuilds both pyramids on every SSIMULACRA 2 call.
--metric-pair measures sequential SSIMULACRA 2 + Butteraugli using
independent pipelines, their shared workspace, or the Metrics API.
One pair and worker;
streaming by default, or preuploaded with --vulkan-buffers.
--host-views times borrowed host plane views and their upload.
--vulkan-buffers reads two preuploaded caller-owned Vulkan buffers.
--half-planes stores the intermediate planes as 16-bit floats (ssimulacra2 only).
--metric butteraugli times the Butteraugli backend instead and prints its stages.
--metric psnr-hvs reports dB (higher is better; zero weighted error is inf).
It measures matching opaque native planar YUV 4:2:0, 4:2:2 or 4:4:4, at 8,
10 or 12 bits; every plane must be at least 8x8. RGB conversion is unavailable.
Pixel codes are measured directly; input color overrides do not convert them.
Streaming, --resident, --resident-reference, --host-views, --vulkan-buffers and
fixed-count --workers are supported. Both inputs are prepared on every call.
PSNR-HVS --score-only uses one worker and retains a pipeline per geometry.
Y4M benchmark inputs use their first frame; the CLI scores complete clips.
--intensity-target/--hf-asymmetry are Butteraugli's libjxl parameters.
--input-matrix/--input-transfer/--input-primaries/--input-range and
--chroma-filter override the decoder's declaration, as in the CLI.
--throughput-seconds S --workers N runs a fixed S-second window with N
pipelines in one process and prints one machine-readable line instead of
the table; with one worker it also prints GPU stage time and its
reciprocal rate. --threads is an alias for --workers.
--vmaf-no-temporal forces VMAF v0 motion2 to zero, skips the motion
passes, and scores every frame immediately.
--metric cvvdp measures independent still pairs and reports JOD (identity 10).
CVVDP defaults to standard viewing geometry and derives pixels per degree
from the fitted input size. --cvvdp-ppd fixes pixels per degree.
Use --cvvdp-peak to set display brightness in nits (default 200).
--cvvdp-video instead reads whole progressive Y4M clips with matching rational
frame rates. Each warmup/iteration resets and scores one complete clip; decode
is outside core timing. It reports ms/clip and ms/frame separately. This mode
uses one worker and streaming inputs; warmup counts clips, not frames.
--pool runs a fixed-count streaming benchmark through the public
pool API; --workers controls its size. It accepts one pair.
)help",
        program
    );
}

bool ParseInt(std::string_view text, int& out) {
    const char* first = text.data();
    const char* last = text.data() + text.size();
    const std::from_chars_result result = std::from_chars(first, last, out);
    return result.ec == std::errc{} && result.ptr == last;
}

bool ParseArgs(
    std::span<const std::string_view> args, BenchOptions& options,
    std::string& error
) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        const bool has_value = i + 1 < args.size();
        if (arg == "--help" || arg == "-h") {
            options.help = true;
            return true;
        }
        if (arg == "--half-planes") {
            options.half_planes = true;
        } else if (arg == "--cvvdp-video") {
            options.cvvdp_video = true;
        } else if (vqm::tool::IsCvvdpOption(arg)) {
            if (!has_value) {
                error = std::format("{} needs a value", arg);
                return false;
            }
            if (!vqm::tool::ParseCvvdpOption(
                    arg, args[++i], options.cvvdp, error
                ))
                return false;
        } else if (arg == "--device-info") {
            options.device_info = true;
        } else if (arg == "--pool") {
            options.pool = true;
        } else if (arg == "--resident-reference") {
            options.resident_reference = true;
        } else if (arg == "--score-only") {
            options.score_only = true;
        } else if (arg == "--host-views") {
            options.host_views = true;
        } else if (arg == "--vulkan-buffers") {
            options.vulkan_buffers = true;
        } else if (arg == "--vmaf-chroma") {
            if (!has_value) {
                error = "--vmaf-chroma needs native, 420 or 444";
                return false;
            }
            const std::string_view value = args[++i];
            if (value != "native" && value != "420" && value != "444") {
                error = "--vmaf-chroma needs native, 420 or 444";
                return false;
            }
            options.vmaf_chroma = value;
        } else if (arg == "--vmaf-fit") {
            if (!has_value) {
                error = "--vmaf-fit needs native or pad";
                return false;
            }
            const std::string_view value = args[++i];
            if (value != "native" && value != "pad") {
                error = "--vmaf-fit needs native or pad";
                return false;
            }
            options.vmaf_fit = value;
        } else if (arg == "--vmaf-model") {
            if (!has_value) {
                error = "--vmaf-model needs 1080 or 4k";
                return false;
            }
            const std::string_view value = args[++i];
            if (value != "1080" && value != "4k") {
                error = "--vmaf-model needs 1080 or 4k";
                return false;
            }
            options.vmaf_model = value;
        } else if (arg == "--vmaf-neg") {
            options.vmaf_neg = true;
        } else if (arg == "--vmaf-transform") {
            options.vmaf_transform = true;
        } else if (arg == "--vmaf-no-temporal") {
            options.vmaf_temporal = false;
        } else if (arg == "--metric-pair") {
            if (!has_value) {
                error = "--metric-pair needs independent, shared or unified";
                return false;
            }
            const std::string_view mode = args[++i];
            if (mode != "independent" && mode != "shared" && mode != "unified") {
                error = "--metric-pair needs independent, shared or unified";
                return false;
            }
            options.metric_pair = mode;
        } else if (arg == "--metric") {
            if (!has_value) {
                error = "--metric needs a value";
                return false;
            }
            const std::string_view text = args[++i];
            if (text == "ssimulacra2") {
                options.metric = Metric::Ssimulacra2;
            } else if (text == "butteraugli") {
                options.metric = Metric::Butteraugli;
            } else if (text == "vmaf-v0") {
                options.metric = Metric::VmafV0;
            } else if (text == "vmaf-v1") {
                options.metric = Metric::VmafV1;
            } else if (text == "cvvdp") {
                options.metric = Metric::Cvvdp;
            } else if (text == "psnr-hvs") {
                options.metric = Metric::PsnrHvs;
            } else {
                error = std::format("invalid --metric '{}'", text);
                return false;
            }
        } else if (arg == "--intensity-target" || arg == "--hf-asymmetry") {
            if (!has_value) {
                error = std::format("{} needs a value", arg);
                return false;
            }
            const std::string_view text = args[++i];
            float parsed = 0.0f;
            const auto result =
                std::from_chars(text.data(), text.data() + text.size(), parsed);
            if (result.ec != std::errc{} ||
                result.ptr != text.data() + text.size() ||
                !std::isfinite(parsed) || parsed <= 0.0f) {
                error = std::format("invalid {} '{}'", arg, text);
                return false;
            }
            if (arg == "--intensity-target") {
                options.intensity_target = parsed;
            } else {
                options.hf_asymmetry = parsed;
            }
        } else if (const auto input_option =
                       vqm::image_io::IdentifyInputColorOption(arg)) {
            if (!has_value) {
                error = std::format("{} needs a value", arg);
                return false;
            }
            const std::string_view text = args[++i];
            std::string accepted;
            if (!vqm::image_io::ParseInputColorOption(
                    *input_option, text, options.input, accepted
                )) {
                error = std::format(
                    "invalid value for {}: '{}' (expected {})", arg, text,
                    accepted
                );
                return false;
            }
        } else if (arg == "--pair") {
            if (i + 2 >= args.size()) {
                error = "--pair needs an original and a distorted path";
                return false;
            }
            options.pairs.push_back(
                Pair{std::string(args[i + 1]), std::string(args[i + 2])}
            );
            i += 2;
        } else if (arg == "--pairs-file" || arg == "--csv" ||
                   arg == "--stages-csv") {
            if (!has_value) {
                error = std::format("{} needs a value", arg);
                return false;
            }
            std::string& target =
                arg == "--pairs-file"
                    ? options.pairs_file
                    : (arg == "--csv" ? options.csv_path
                                      : options.stages_csv_path);
            target = args[++i];
        } else if (arg == "--iterations" || arg == "--warmup" ||
                   arg == "--resident" || arg == "--resident-ring" ||
                   arg == "--workers" || arg == "--threads" ||
                   arg == "--decode-threads") {
            if (!has_value) {
                error = std::format("{} needs a value", arg);
                return false;
            }
            const std::string_view text = args[++i];
            int value = 0;
            if (!ParseInt(text, value) || value < 0) {
                error = std::format("invalid value for {}: '{}'", arg, text);
                return false;
            }
            if (arg == "--resident-ring" && value < 2) {
                error =
                    "--resident-ring needs a maximum pair count between 2 and "
                    "2047";
                return false;
            }
            if (arg == "--iterations")
                options.iterations = value;
            else if (arg == "--warmup")
                options.warmup = value;
            else if (arg == "--resident")
                options.resident = value;
            else if (arg == "--resident-ring")
                options.resident_ring = value;
            else if (arg == "--workers" || arg == "--threads")
                options.workers = value;
            else
                options.decode_threads = value;
        } else if (arg == "--throughput-seconds") {
            if (!has_value) {
                error = "--throughput-seconds needs a value";
                return false;
            }
            const std::string_view text = args[++i];
            double value = 0.0;
            const char* last = text.data() + text.size();
            const std::from_chars_result result =
                std::from_chars(text.data(), last, value);
            if (result.ec != std::errc{} || result.ptr != last ||
                !inmem::ValidWindowSeconds(value)) {
                error = std::format(
                    "invalid --throughput-seconds '{}' (expected a positive, "
                    "finite, representable number of seconds)",
                    text
                );
                return false;
            }
            options.throughput_seconds = value;
        } else {
            error = std::format("unknown option '{}'", arg);
            return false;
        }
    }
    if (options.iterations < 1) {
        error = "--iterations must be at least 1";
        return false;
    }
    if (options.metric == Metric::PsnrHvs && options.workers < 1) {
        error = "PSNR-HVS --workers must be at least 1";
        return false;
    }
    if (options.metric == Metric::PsnrHvs &&
        (options.pool || options.resident_ring || options.half_planes ||
         options.throughput_seconds != 0.0 || !options.metric_pair.empty())) {
        error =
            "PSNR-HVS supports fixed-count streaming, resident, host-view or "
            "Vulkan-buffer inputs; pool, ring, half-plane, throughput-window "
            "and metric-pair modes are unavailable";
        return false;
    }
    if (options.metric == Metric::PsnrHvs && options.score_only &&
        (!options.stages_csv_path.empty() || options.workers != 1)) {
        error =
            "PSNR-HVS --score-only requires one worker and cannot report "
            "timed stages";
        return false;
    }
    if ((options.cvvdp.specified || options.cvvdp_video) &&
        options.metric != Metric::Cvvdp) {
        error = "--cvvdp-* options require --metric cvvdp";
        return false;
    }
    if (const auto geometry_error = options.cvvdp.GeometryError();
        !geometry_error.empty()) {
        error = geometry_error;
        return false;
    }
    if (options.metric == Metric::Cvvdp &&
        (options.pool || options.resident_ring || options.resident_reference ||
         options.host_views || options.vulkan_buffers || options.half_planes ||
         options.workers != 1 || options.throughput_seconds != 0.0 ||
         !options.metric_pair.empty() ||
         (options.cvvdp_video && options.resident))) {
        error =
            "CVVDP supports one worker and streaming or still --resident "
            "inputs; "
            "--cvvdp-video requires streaming";
        return false;
    }
    if (options.metric == Metric::Cvvdp && options.score_only &&
        !options.stages_csv_path.empty()) {
        error = "CVVDP --score-only cannot report timed stages";
        return false;
    }
    if (!options.IsVmaf() && options.vmaf_chroma != "native") {
        error = "--vmaf-chroma requires a VMAF metric";
        return false;
    }
    if (!options.IsVmaf() && options.vmaf_fit != "native") {
        error = "--vmaf-fit requires a VMAF metric";
        return false;
    }
    if (!options.IsVmaf() && !options.vmaf_temporal) {
        error = "--vmaf-no-temporal requires a VMAF metric";
        return false;
    }
    if ((options.metric == Metric::VmafV1) &&
        (options.vmaf_model != "1080" || options.vmaf_neg ||
         options.vmaf_transform || !options.vmaf_temporal)) {
        error =
            "--vmaf-model, --vmaf-neg, --vmaf-transform and "
            "--vmaf-no-temporal require VMAF v0";
        return false;
    }
    if (!options.metric_pair.empty() &&
        (options.IsVmaf() || (options.metric == Metric::Butteraugli) ||
         options.pool || options.resident || options.resident_ring ||
         options.resident_reference || options.host_views ||
         options.score_only || options.workers != 1 ||
         options.throughput_seconds != 0.0 || !options.csv_path.empty())) {
        error =
            "--metric-pair requires one worker and streaming or "
            "--vulkan-buffers; other metric, input and output modes are "
            "unsupported";
        return false;
    }
    if (options.IsVmaf() &&
        (options.pool || options.resident_ring || options.resident_reference ||
         options.host_views || options.vulkan_buffers || options.half_planes ||
         options.workers != 1 || options.throughput_seconds != 0.0)) {
        error = "VMAF supports one worker and streaming or --resident inputs";
        return false;
    }
    if (options.IsVmaf() && options.score_only &&
        !options.stages_csv_path.empty()) {
        error = "VMAF --score-only cannot report timed stages";
        return false;
    }
    if (options.warmup > std::numeric_limits<int>::max() - options.iterations) {
        error = "--warmup plus --iterations is too large";
        return false;
    }
    if (options.resident == 1) {
        error = "--resident needs at least 2 slots, one per pair member";
        return false;
    }
    if (options.resident_ring == 1 || options.resident_ring > 2047) {
        error = "--resident-ring needs a maximum pair count between 2 and 2047";
        return false;
    }
    if (options.resident_ring > 0 &&
        ((options.metric == Metric::Butteraugli) || options.resident ||
         options.resident_reference || options.pool || options.score_only ||
         options.host_views || options.vulkan_buffers ||
         options.throughput_seconds != 0.0 || options.workers != 1 ||
         !options.csv_path.empty())) {
        error =
            "--resident-ring requires SSIMULACRA 2, one worker, and no "
            "other input or throughput mode or --csv";
        return false;
    }
    if ((options.metric == Metric::Butteraugli) && options.half_planes) {
        error = "--half-planes is available only for ssimulacra2";
        return false;
    }
    if (options.pool &&
        (options.resident != 0 || options.resident_reference ||
         options.score_only || options.host_views || options.vulkan_buffers ||
         options.throughput_seconds != 0.0)) {
        error =
            "--pool measures fixed-count streaming and cannot be combined "
            "with --resident, --resident-reference, --score-only, --host-views, "
            "--vulkan-buffers or "
            "--throughput-seconds";
        return false;
    }
    if (options.resident_reference &&
        (options.resident != 0 || options.workers != 1 ||
         options.throughput_seconds != 0.0)) {
        error =
            "--resident-reference requires one worker and no --resident or "
            "throughput mode";
        return false;
    }
    if (options.host_views &&
        (options.resident != 0 || options.resident_reference ||
         options.workers != 1 || options.throughput_seconds != 0.0)) {
        error = "--host-views needs one worker and streaming latency mode";
        return false;
    }
    if (options.vulkan_buffers &&
        (options.host_views || options.resident != 0 ||
         options.resident_reference || options.workers != 1 ||
         options.throughput_seconds != 0.0)) {
        error =
            "--vulkan-buffers needs one worker, no resident handles and latency "
            "mode";
        return false;
    }
    return true;
}

}  // namespace bench_driver
