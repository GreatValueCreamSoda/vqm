// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#pragma once

#include <span>
#include <string>
#include <string_view>
#include <variant>

#include "cvvdp_options.h"
#include "io.h"

namespace vqm::cli {

enum class Metric { Ssimulacra2, Butteraugli, VmafV0, VmafV1, Cvvdp, PsnrHvs };

struct Options {
    bool raw = false;
    bool json = false;
    bool half_planes = false;
    bool nvdec = false;
    Metric metric = Metric::Ssimulacra2;
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
using ParseResult = std::variant<Options, ParseError, HelpRequested>;

[[nodiscard]] ParseResult ParseOptions(std::span<const std::string_view> args);
[[nodiscard]] std::string Usage(std::string_view program);

}  // namespace vqm::cli
