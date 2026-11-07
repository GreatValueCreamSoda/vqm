// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// Butteraugli portions: Copyright (c) the JPEG XL Project Authors.
// BSD-3-Clause; see licenses/JPEG-XL-LICENSE.
//
// Test-only double-precision oracle for the libjxl butteraugli comparator.
// This is an independent transliteration of the comparator path of
// .reference/libjxl/lib/jxl/butteraugli/butteraugli.cc (BSD-3-Clause). The
// relevant line map is:
//   :75-83 ComputeKernel          :229-267 Blur (5-tap and FIR paths)
//   :184-196 image_ops.h Mirror   :330-378 XybLowFreqToVals/SuppressXByY
//   :581-985 MaltaUnit LF/full   :988-1107 MaltaDiffMapT
//   :1091-1130 masking helpers    :1166-1206 FuzzyErosion
//   :1268-1317 CombineChannels    :1320-1403 L2 diff terms
//   :1424-1539 OpsinDynamicsImage :1445-1610 SeparateFrequencies
//   :1733-1778 SubSample2x / AddSupersampled2x
//   :1792-1935 Make / Diffmap / DiffmapPsychoImage
//   :1954-2120 score, padded small-image entry, ButteraugliDiffmap
// distance_p3 follows lib/extras/metrics.cc:42-145. Everything is computed in
// double precision, single-threaded, so results are deterministic.

#pragma once

namespace vqm::test {

struct ButterRefParams {
    double hf_asymmetry = 1.0;
    double xmul = 1.0;
    double intensity_target = 80.0;
};

struct ButterRefResult {
    double score = 0.0;
    double distance_p3 = 0.0;
};

// Planes are row-major, linear Rec.709 RGB, width * height samples each, with
// no padding. Images smaller than 8x8 are edge-extended internally, exactly
// like ButteraugliDiffmapSmall.
ButterRefResult ButteraugliCpuReference(
    const float* r0, const float* g0, const float* b0, const float* r1,
    const float* g1, const float* b1, int width, int height,
    const ButterRefParams& params = {}
);

}  // namespace vqm::test
