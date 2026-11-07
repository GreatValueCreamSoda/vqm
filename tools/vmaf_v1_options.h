// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#pragma once

#include <array>
#include <optional>
#include <string_view>
#include <utility>

#include "vmaf_v1.h"

namespace vqm::tool {

inline constexpr std::string_view kVmafV1ModelChoices =
    "1080p-3h|4k-1.5h|4k-3h|phone-5h|hdr-4k-1.5h";

inline std::optional<VmafV1Model> ParseVmafV1Model(std::string_view value) {
    using Model = VmafV1Model;
    constexpr std::array names{std::pair{"1080p-3h", Model::Standard1080},
                               std::pair{"4k-1.5h", Model::Standard4K},
                               std::pair{"4k-3h", Model::Tv4K},
                               std::pair{"phone-5h", Model::Phone},
                               std::pair{"hdr-4k-1.5h", Model::Hdr4K}};
    for (const auto& [name, model] : names)
        if (value == name) return model;
    return std::nullopt;
}

}  // namespace vqm::tool
