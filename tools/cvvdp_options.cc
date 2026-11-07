// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
#include "cvvdp_options.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <numbers>
#include <stdexcept>
#include <system_error>

namespace vqm::tool {
namespace {

bool ValidNumber(double value, bool zero_allowed = false) {
    return std::isfinite(value) &&
           (value > 0.0 || (zero_allowed && value == 0.0));
}

std::optional<double> ParseNumber(
    std::string_view value, bool zero_allowed = false
) {
    double number = 0.0;
    const auto parsed =
        std::from_chars(value.data(), value.data() + value.size(), number);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        !ValidNumber(number, zero_allowed))
        return std::nullopt;
    return number;
}

std::optional<CvvdpOptions::AspectRatio> ParseAspect(std::string_view value) {
    const auto separator = value.find(':');
    if (separator == std::string_view::npos) return std::nullopt;
    const auto width = ParseNumber(value.substr(0, separator));
    const auto height = ParseNumber(value.substr(separator + 1));
    if (!width || !height) return std::nullopt;
    return CvvdpOptions::AspectRatio{*width, *height};
}

double PixelsPerDegree(
    Size content, CvvdpOptions::AspectRatio aspect, double diagonal_inches,
    double distance_metres
) {
    if (content.width == 0 || content.height == 0 ||
        !ValidNumber(aspect.width) || !ValidNumber(aspect.height) ||
        !ValidNumber(diagonal_inches) || !ValidNumber(distance_metres))
        throw std::invalid_argument("cvvdp: invalid display geometry");
    const double scale = std::max(aspect.width, aspect.height);
    const double width = aspect.width / scale;
    const double height = aspect.height / scale;
    const double metres = diagonal_inches * 0.0254 / std::hypot(width, height);
    // One square content pixel occupies the same physical pitch on both axes.
    // Aspect-preserving fit chooses the limiting display dimension.
    const double pitch = std::min(
        metres * width / static_cast<double>(content.width),
        metres * height / static_cast<double>(content.height)
    );
    // The original model measures the angle subtended by one central pixel.
    const double ppd =
        std::numbers::pi / (360.0 * std::atan2(0.5 * pitch, distance_metres));
    if (!std::isfinite(ppd) || ppd < 1.0 || ppd > 1e6)
        throw std::invalid_argument(
            std::format(
                "cvvdp: fitted display geometry yields {} PPD outside the "
                "supported range [1, 1000000]",
                ppd
            )
        );
    return ppd;
}

}  // namespace

std::string_view CvvdpOptions::GeometryError() const {
    if (ppd && (aspect || diagonal_inches || distance_metres))
        return "--cvvdp-ppd cannot be combined with --cvvdp-aspect, "
               "--cvvdp-diagonal or --cvvdp-distance";
    return {};
}

CvvdpDisplay CvvdpOptions::ResolveDisplay(Size content) const {
    if (const auto error = GeometryError(); !error.empty())
        throw std::invalid_argument(std::string(error));
    CvvdpDisplay result;
    const bool compact = display == "compact";
    result.pixels_per_degree =
        ppd ? *ppd
            : PixelsPerDegree(
                  content, aspect.value_or(AspectRatio{16.0, 9.0}),
                  diagonal_inches.value_or(compact ? 24.0 : 30.0),
                  distance_metres.value_or(compact ? 0.6 : 0.7472)
              );
    if (peak) result.peak_luminance = *peak;
    if (contrast) result.contrast = *contrast;
    if (ambient) result.ambient_illuminance = *ambient;
    return result;
}

bool IsCvvdpOption(std::string_view flag) {
    return flag == "--cvvdp-display" || flag == "--cvvdp-ppd" ||
           flag == "--cvvdp-aspect" || flag == "--cvvdp-diagonal" ||
           flag == "--cvvdp-distance" || flag == "--cvvdp-peak" ||
           flag == "--cvvdp-contrast" || flag == "--cvvdp-ambient";
}

bool ParseCvvdpOption(
    std::string_view flag, std::string_view value, CvvdpOptions& options,
    std::string& error
) {
    if (flag == "--cvvdp-display") {
        if (value != "standard" && value != "compact") {
            error = "--cvvdp-display requires standard or compact";
            return false;
        }
        options.display = value;
        options.specified = true;
        return true;
    }
    if (flag == "--cvvdp-aspect") {
        const auto aspect = ParseAspect(value);
        if (!aspect) {
            error = std::format(
                "invalid value for {}: '{}' (expected positive finite W:H)",
                flag, value
            );
            return false;
        }
        options.aspect = aspect;
        options.specified = true;
        return true;
    }
    const bool zero_allowed = flag == "--cvvdp-ambient";
    const auto number = ParseNumber(value, zero_allowed);
    if (!number) {
        error = std::format(
            "invalid value for {}: '{}' (expected a finite {} number)", flag,
            value, zero_allowed ? "nonnegative" : "positive"
        );
        return false;
    }
    if (flag == "--cvvdp-ppd")
        options.ppd = number;
    else if (flag == "--cvvdp-diagonal")
        options.diagonal_inches = number;
    else if (flag == "--cvvdp-distance")
        options.distance_metres = number;
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
