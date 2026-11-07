// Copyright (c) 2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
#pragma once

#include <array>

#include "vmaf_v1.h"

namespace vqm::internal {

// Features for the SDR and HDR v1 models, in model order.
// The model is trained on CAMBI, chroma SpEED UV, ADM3, and motion3 with
// specific extractor options; supplying v0 features does not implement v1.
struct VmafV1Features {
    double cambi = 0.0;
    double speed_chroma_uv = 0.0;
    double adm3 = 0.0;
    double motion3 = 0.0;
};

struct VmafV1ModelParameters {
    Size canvas;
    double chroma_prescale;
    std::array<double, 3> transform;
    std::array<double, 2> clip;
};

[[nodiscard]] const VmafV1ModelParameters& VmafV1Parameters(VmafV1Model model);

// Evaluates the selected v1 model, its chroma correction, default
// score transform and clipping. Disable clipping only for diagnostics.
// Throws std::invalid_argument for non-finite inputs.
[[nodiscard]] double VmafV1Predict(
    const VmafV1Features& features, bool enable_clip = true,
    VmafV1Model model = VmafV1Model::Standard1080
);

}  // namespace vqm::internal
