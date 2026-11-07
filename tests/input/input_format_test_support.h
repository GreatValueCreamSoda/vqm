// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// The input surface: every datatype, transfer function, matrix coefficient,
// primaries set, chroma siting and subsampling this build accepts, checked by
// encoding the same content two ways and requiring the metric to see the same
// linear image.
//
// Each A/B test builds frame A in the format under test and frame B in a
// format whose conversion is trivial (linear RGB float), with B's samples
// computed independently in double by input_reference.h. The pair (A, B) must
// then behave like (B, B): the residual error norms are the deviation of the
// GPU conversion from the reference, so the assertion is
//
//     MaxNormDelta(Compute(A, B), Compute(B, B)) <= tolerance

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <limits>
#include <optional>
#include <source_location>
#include <string>
#include <vector>
#include "input.h"
#include "input_reference.h"
#include "io.h"
#include "support.h"
#include "vqm.h"
#include "vulkan/vulkan_support.h"

namespace vqm::test::input_format {

using vqm::AlphaType;
using vqm::ChromaFilter;
using vqm::ChromaLocation;
using vqm::ColorFamily;
using vqm::ColorPrimaries;
using vqm::Image;
using vqm::ImageFormat;
using vqm::MatrixCoefficients;
using vqm::PixelRange;
using vqm::PixelType;
using vqm::Size;
using vqm::Ssimulacra2Pipeline;
using vqm::Ssimulacra2Result;
using vqm::TransferCharacteristics;
namespace ref = vqm::test::ref;

constexpr std::size_t kSide = vqm::test::kSide;

// These comparisons run the GPU's float32 conversion against a double
// reference. Measured residual over every case at two significant figures:
// 1.3e-6 (constant luminance) to 5.4e-5 (ICtCp with PQ), so the bounds below
// are ~4x the worst case. Sensitivity, measured by perturbing one gamut entry:
// 0.1% of the entry passes, 0.3% fails 11 checks, 10% produces 1.4e-2 deltas.
// So these checks pin the conversion to a few tenths of a percent; tightening
// further would only track a device's transcendental rounding.
constexpr double kRounding = 2e-4;
// PQ's denominator (C2 - C3*x^pow) cancels catastrophically in float32 and the
// outer 1/M1 = 6.3 power amplifies it, so the residual there is the largest.
constexpr double kPqCurve = 4e-4;
// The HLG luminance OOTF amplifies through exp/log and a 1.2 power on the luma.
constexpr double kHlgCurve = 4e-4;

inline Size Side() { return {kSide, kSide}; }

// Content for the 4:4:4/RGB A/B tests: a fixed RGB triple everywhere, so the
// conversion is checked pointwise rather than through an average.
inline std::array<std::vector<float>, 4> Uniform(
    const ref::Vec3& rgb, const ref::Vec3& alpha = {1, 1, 1}
) {
    std::array<std::vector<float>, 4> values;
    for (std::size_t i = 0; i < kSide * kSide; ++i) {
        values[0].push_back(static_cast<float>(rgb[0]));
        values[1].push_back(static_cast<float>(rgb[1]));
        values[2].push_back(static_cast<float>(rgb[2]));
        values[3].push_back(static_cast<float>(alpha[0]));
    }
    return values;
}

inline ImageFormat LinearRgbFormat() {
    ImageFormat format;
    format.type = PixelType::Float;
    format.family = ColorFamily::Rgb;
    format.matrix = MatrixCoefficients::Rgb;
    format.transfer = TransferCharacteristics::Linear;
    format.primaries = ColorPrimaries::Bt709;
    return format;
}

inline Image Format(
    const ImageFormat& format, const std::array<std::vector<float>, 4>& values,
    Size size = Side()
) {
    return vqm::test::PackPlanes(format, size, values);
}

inline double MaxNormDelta(
    const Ssimulacra2Result& a, const Ssimulacra2Result& b
) {
    double worst = 0.0;
    if (a.scales.size() != b.scales.size()) return 1e9;
    for (std::size_t s = 0; s < a.scales.size(); ++s) {
        for (std::size_t n = 0; n < a.scales[s].avg_ssim.size(); ++n) {
            worst = std::max(
                worst,
                std::fabs(a.scales[s].avg_ssim[n] - b.scales[s].avg_ssim[n])
            );
        }
        for (std::size_t n = 0; n < a.scales[s].avg_edgediff.size(); ++n) {
            worst = std::max(
                worst,
                std::fabs(
                    a.scales[s].avg_edgediff[n] - b.scales[s].avg_edgediff[n]
                )
            );
        }
    }
    return worst;
}

// (a, b) must decode to the same linear image as (b, b).
inline void CheckAgrees(
    Ssimulacra2Pipeline& pipeline, const Image& a, const Image& b,
    double tolerance, const char* what, float background = 0.5f
) {
    const Ssimulacra2Result pair = pipeline.Compute(a, b, background);
    const Ssimulacra2Result identity = pipeline.Compute(b, b, background);
    const double delta = MaxNormDelta(pair, identity);
    if (!(delta <= tolerance)) {
        vqm::test::Fail(
            std::source_location::current(),
            std::format(
                "{}: conversion differs by {:.3e} (tolerance {:.3e}), "
                "score {:.8f}",
                what, delta, tolerance, pair.Score()
            )
        );
    }
}

inline Ssimulacra2Pipeline MakePipeline() {
    return Ssimulacra2Pipeline({{kSide, kSide}});
}

}  // namespace vqm::test::input_format
