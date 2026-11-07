// Copyright (c) 2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
#pragma once

namespace vqm::internal {

// Features for vmaf_v1.0.16_3d0h, standard-range SDR, in model order.
// The model is trained on CAMBI, chroma SpEED UV, ADM3, and motion3 with
// specific extractor options; supplying v0 features does not implement v1.
struct VmafV1Features {
    double cambi = 0.0;
    double speed_chroma_uv = 0.0;
    double adm3 = 0.0;
    double motion3 = 0.0;
};

// Evaluates the pinned standard v1 model, its chroma correction, default
// score transform and clipping. Disable clipping only for diagnostics.
// Throws std::invalid_argument for non-finite inputs.
[[nodiscard]] double VmafV1Predict(
    const VmafV1Features& features, bool enable_clip = true
);

}  // namespace vqm::internal
