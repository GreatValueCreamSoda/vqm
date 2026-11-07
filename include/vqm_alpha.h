// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include "butteraugli.h"
#include "ssimulacra2.h"

namespace vqm {

// Match the CLI's alpha policy for images or views that expose `format`.
// Opaque inputs are scored once at the default background. An alpha-bearing
// image on either side is scored over 0.1 and 0.9 and the worse result is
// returned, so transparent inputs require two metric calls. The normal Compute
// overloads keep their existing single-background semantics.
template <typename Frame>
[[nodiscard]] Ssimulacra2Result ComputeWorstBackground(
    Ssimulacra2Pipeline& pipeline, const Frame& original, const Frame& distorted
) {
    if (original.format.alpha == AlphaType::None &&
        distorted.format.alpha == AlphaType::None) {
        return pipeline.Compute(original, distorted);
    }
    auto dark = pipeline.Compute(original, distorted, 0.1f);
    auto bright = pipeline.Compute(original, distorted, 0.9f);
    return dark.Score() <= bright.Score() ? dark : bright;
}

template <typename Frame>
[[nodiscard]] ButteraugliScore ComputeWorstBackground(
    ButteraugliPipeline& pipeline, const Frame& original, const Frame& distorted
) {
    if (original.format.alpha == AlphaType::None &&
        distorted.format.alpha == AlphaType::None) {
        return pipeline.Compute(original, distorted);
    }
    const auto dark = pipeline.Compute(original, distorted, 0.1f);
    const auto bright = pipeline.Compute(original, distorted, 0.9f);
    return dark.score >= bright.score ? dark : bright;
}

}  // namespace vqm
