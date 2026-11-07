// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "butteraugli.h"
#include "ssimulacra2.h"
#include "cvvdp_options.h"
#include "io.h"
#include "vmaf.h"
#include "vmaf_v1.h"
#include "vqm_device.h"

namespace vqm::cli {

enum class Metric { Ssimulacra2, Butteraugli, VmafV0, VmafV1, Cvvdp, PsnrHvs };

struct Options {
    bool raw = false;
    bool json = false;
    bool verbose = false;
    bool half_planes = false;
    std::optional<unsigned> device_index;
    std::optional<std::int64_t> start_frame;
    std::optional<std::int64_t> frame_count;
    bool vmaf_scale = false;
    bool vmaf_pad = false;
    std::optional<VmafModel> vmaf_v0_model;
    bool vmaf_v0_neg = false;
    bool vmaf_v0_transform = false;
    std::optional<VmafV1Model> vmaf_v1_model;
    bool vmaf_v1_hfr = false;
    std::vector<Metric> metrics{Metric::Ssimulacra2};
    tool::CvvdpOptions cvvdp;
    float hf_asymmetry = 1.0f;
    float intensity_target = 80.0f;
    image_io::InputColorOptions input;
    std::string original;
    std::string distorted;
    std::string pairs_file;
    unsigned iterations = 1;
    unsigned warmup = 0;
};

struct ParseError {
    std::string message;
};
struct HelpRequested {};
struct DeviceListRequested {};
using ParseResult =
    std::variant<Options, ParseError, HelpRequested, DeviceListRequested>;

[[nodiscard]] ParseResult ParseOptions(std::span<const std::string_view> args);
[[nodiscard]] std::string Usage(std::string_view program);
[[nodiscard]] std::string_view MetricName(Metric metric);
[[nodiscard]] Ssimulacra2Config MakeSsimulacra2Config(
    const Options& options, Size size
);
[[nodiscard]] ButteraugliConfig MakeButteraugliConfig(
    const Options& options, Size size
);

}  // namespace vqm::cli
