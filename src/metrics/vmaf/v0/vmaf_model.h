// Copyright (c) 2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

#pragma once

#include <array>

#include "vmaf.h"

namespace vqm::internal {

// Feature order is the pinned vmaf_float_v0.6.1 model's declared feature_names:
// ADM2, motion2, then VIF scales 0..3. These are floating-point feature values,
// not the integer extractor's values. This pure internal predictor is used by
// the temporal VMAF pipeline after feature extraction and motion lookahead.
struct VmafV061Features {
    double adm2 = 0.0;
    double motion2 = 0.0;
    std::array<double, 4> vif{};
};

// Evaluate one of Netflix's float v0.6.1 models. All six inputs must be
// finite; otherwise throws std::invalid_argument. Uses host double precision
// as the upstream model predictor does. Standard1080 and Standard4K select the
// 211- and 262-support-vector tables. Default options match libvmaf: no
// optional phone transform, clipping to [0,100]. Enabling the transform on
// Standard4K applies the 1080 phone polynomial as a derived extension because
// the 4K model declares none; clipping can be disabled independently for
// diagnostics.
[[nodiscard]] double VmafV061Predict(
    const VmafV061Features& features, bool enable_transform = false,
    bool enable_clip = true, VmafModel model = VmafModel::Standard1080
);

}  // namespace vqm::internal
