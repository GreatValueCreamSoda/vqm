// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "cvvdp.h"

namespace vqm::tool {

// Shared physical viewing options for the CLI and benchmark. Explicit numeric
// overrides apply after preset selection, independent of argument ordering.
struct CvvdpOptions {
    struct AspectRatio {
        double width;
        double height;
    };

    std::string display = "standard";
    std::optional<double> ppd;
    std::optional<AspectRatio> aspect;
    std::optional<double> diagonal_inches;
    std::optional<double> distance_metres;
    std::optional<double> peak;
    std::optional<double> contrast;
    std::optional<double> ambient;
    bool specified = false;

    [[nodiscard]] std::string_view GeometryError() const;
    [[nodiscard]] CvvdpDisplay ResolveDisplay(Size content) const;
};

[[nodiscard]] bool IsCvvdpOption(std::string_view flag);
[[nodiscard]] bool ParseCvvdpOption(
    std::string_view flag, std::string_view value, CvvdpOptions& options,
    std::string& error
);

}  // namespace vqm::tool
