// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
#include "cvvdp_options.h"

#include <charconv>
#include <cmath>
#include <format>
#include <stdexcept>
#include <system_error>

namespace vqm::tool {

CvvdpDisplay CvvdpOptions::ResolveDisplay() const {
    CvvdpDisplay result = CvvdpDisplay::Standard4K();
    if (display == "standard-fhd") result = CvvdpDisplay::StandardFhd();
    if (display == "standard-hdr") result = CvvdpDisplay::StandardHdr();
    if (ppd) result.pixels_per_degree = *ppd;
    if (peak) result.peak_luminance = *peak;
    if (contrast) result.contrast = *contrast;
    if (ambient) result.ambient_illuminance = *ambient;
    return result;
}

bool IsCvvdpOption(std::string_view flag) {
    return flag == "--cvvdp-display" || flag == "--cvvdp-ppd" ||
           flag == "--cvvdp-peak" || flag == "--cvvdp-contrast" ||
           flag == "--cvvdp-ambient";
}

bool ParseCvvdpOption(
    std::string_view flag, std::string_view value, CvvdpOptions& options,
    std::string& error
) {
    if (flag == "--cvvdp-display") {
        if (value != "standard-4k" && value != "standard-fhd" &&
            value != "standard-hdr") {
            error =
                "--cvvdp-display requires standard-4k, standard-fhd or "
                "standard-hdr";
            return false;
        }
        options.display = value;
        options.specified = true;
        return true;
    }
    double number = 0.0;
    const auto parsed =
        std::from_chars(value.data(), value.data() + value.size(), number);
    const bool zero_allowed = flag == "--cvvdp-ambient";
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        !std::isfinite(number) || number < 0.0 ||
        (!zero_allowed && number == 0.0)) {
        error = std::format(
            "invalid value for {}: '{}' (expected a finite {} number)", flag,
            value, zero_allowed ? "nonnegative" : "positive"
        );
        return false;
    }
    if (flag == "--cvvdp-ppd")
        options.ppd = number;
    else if (flag == "--cvvdp-peak")
        options.peak = number;
    else if (flag == "--cvvdp-contrast")
        options.contrast = number;
    else if (flag == "--cvvdp-ambient")
        options.ambient = number;
    else
        throw std::invalid_argument("unrecognized CVVDP option");
    options.specified = true;
    return true;
}

}  // namespace vqm::tool
