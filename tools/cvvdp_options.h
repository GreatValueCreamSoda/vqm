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
    std::string display = "standard-4k";
    std::optional<double> ppd;
    std::optional<double> peak;
    std::optional<double> contrast;
    std::optional<double> ambient;
    bool specified = false;

    [[nodiscard]] CvvdpDisplay ResolveDisplay() const;
};

[[nodiscard]] bool IsCvvdpOption(std::string_view flag);
[[nodiscard]] bool ParseCvvdpOption(
    std::string_view flag, std::string_view value, CvvdpOptions& options,
    std::string& error
);

}  // namespace vqm::tool
