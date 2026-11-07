// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// Butteraugli portions: Copyright (c) the JPEG XL Project Authors.
// BSD-3-Clause; see licenses/JPEG-XL-LICENSE.

// Host-side Butteraugli constants and tables. Every number names the libjxl
// line it comes from (.reference/libjxl/lib/jxl/butteraugli/butteraugli.cc,
// BSD-3-Clause); the shaders carry the same numbers as GLSL literals with the
// same citations, and the CPU oracle in tests/metrics/butteraugli/butter_ref.cc
// derives from the same source, so all three implementations stay one metric.

#include "metrics/butteraugli/butteraugli_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "butteraugli.h"
#include "handles.h"

namespace vqm::internal {
namespace {

// libjxl's ComputeKernel (butteraugli.cc:75-83): radius = max(1, int(2.25 *
// |sigma|)) with the taps exp(-i^2 / (2 sigma^2)) computed in double. libjxl
// truncates rather than carrying the full Gaussian tail, and the metric's
// numbers depend on that truncation, so the repository's MakeGaussian (which
// spans 3.5 sigma for SSIMULACRA 2's 13-tap kernel) is deliberately NOT used
// here: the two metrics read different kernels.
template <std::size_t N>
std::array<float, N> ButterKernel(double sigma) {
    constexpr double kAccuracy = 2.25;  // butteraugli.cc:78
    const int radius =
        std::max(1, static_cast<int>(kAccuracy * std::fabs(sigma)));
    if (static_cast<std::size_t>(2 * radius + 1) != N) {
        throw std::invalid_argument("butter: kernel tap count mismatch");
    }
    std::array<double, N> taps{};
    const double scaler = -1.0 / (2.0 * sigma * sigma);
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        const double value = std::exp(scaler * static_cast<double>(i) * i);
        taps[static_cast<std::size_t>(i + radius)] = value;
        sum += value;
    }
    std::array<float, N> out{};
    for (std::size_t i = 0; i < N; ++i)
        out[i] = static_cast<float>(taps[i] / sum);
    return out;
}

// Band constants, butteraugli.cc:60-71.
constexpr double kWMfMalta = 37.0819870399;
constexpr double kNorm1Mf = 130262059.556;
constexpr double kWMfMaltaX = 8246.75321353;
constexpr double kNorm1MfX = 1009002.70582;
constexpr double kWHfMalta = 18.7237414387;
constexpr double kNorm1Hf = 4498534.45232;
constexpr double kWHfMaltaX = 6923.99476109;
constexpr double kNorm1HfX = 8051.15833247;
constexpr double kWUhfMalta = 1.10039032555;
constexpr double kNorm1Uhf = 71.7800275169;
constexpr double kWUhfMaltaX = 173.5;
constexpr double kNorm1UhfX = 5.0;

// MaltaDiffMapT and its wrappers, butteraugli.cc:988-1108.
constexpr double kMaltaLen = 3.75;
constexpr double kMaltaMulli = 0.39905817637;     // 16-line unit (:1093)
constexpr double kMaltaMulliLf = 0.611612573796;  // LF unit (:1104)
constexpr double kMaltaWeightBig = 0.5;           // W_0GT1's kWeight0 (:998)
constexpr double kMaltaWeightSmall = 0.33;        // W_0LT1's kWeight1 (:999)

// One band's prefactors. `w_gt1`/`w_lt1` already carry the asymmetry split, so
// the shader only multiplies by sqrt(kWeight)/len and the norm.
void FillBand(
    ButterBands& bands, std::size_t index, double w_gt1, double w_lt1,
    double norm1, double mulli
) {
    const double denominator = 2.0 * kMaltaLen + 1.0;
    bands.norm1[index] = norm1;
    bands.norm2_gt1[index] =
        mulli * std::sqrt(kMaltaWeightBig * w_gt1) / denominator * norm1;
    bands.norm2_lt1[index] =
        mulli * std::sqrt(kMaltaWeightSmall * w_lt1) / denominator * norm1;
}

}  // namespace

void ValidateButteraugliConfig(const ButteraugliConfig& config) {
    if (!std::isfinite(config.hf_asymmetry) || config.hf_asymmetry <= 0.0f) {
        throw std::invalid_argument("hf_asymmetry must be finite and positive");
    }
    if (!std::isfinite(config.intensity_target) ||
        config.intensity_target <= 0.0f) {
        throw std::invalid_argument(
            "intensity_target must be finite and positive"
        );
    }
    // xmul weights the nonnegative X-channel differences before a square root.
    // Zero is meaningful (omit that channel); a negative weight is not.
    if (!std::isfinite(config.xmul) || config.xmul < 0.0f) {
        throw std::invalid_argument("xmul must be finite and nonnegative");
    }
    if (config.gpu_resident_images > kResidentIndexMask) {
        throw std::invalid_argument(
            "gpu_resident_images is limited to " +
            std::to_string(kResidentIndexMask) + " slots"
        );
    }
}

ButteraugliGeometry PlanButteraugliGeometry(Size base) {
    ButteraugliGeometry geometry;
    geometry.base = base;
    if (base.width < 8 || base.height < 8) {
        throw std::invalid_argument("minimum image size is 8x8 pixels");
    }
    geometry.sub =
        Size{base.width / 2 + base.width % 2, base.height / 2 + base.height % 2};
    geometry.levels = 1;
    geometry.fold = geometry.sub.width >= 8 && geometry.sub.height >= 8;
    if (geometry.fold) geometry.levels = 2;
    return geometry;
}

ButterTables MakeButterTables() {
    ButterTables tables;
    tables.opsin = ButterKernel<5>(1.2);          // :1471 kSigma
    tables.uhf = ButterKernel<7>(1.56416327805);  // :482 kSigmaUhf
    tables.mask = ButterKernel<13>(2.7);          // :1224 kRadius
    tables.hf = ButterKernel<15>(3.22489901262);  // :421 kSigmaHf
    tables.lf = ButterKernel<33>(7.15593339443);  // :406 kSigmaLf
    return tables;
}

ButterBands MakeButterBands(const ButteraugliParams& params) {
    ButterBands bands;
    // The asymmetry split, butteraugli.cc:1910-1928.
    const double asym = static_cast<double>(params.hf_asymmetry);
    const double root_asym = std::sqrt(asym);
    // The shader's norm2 formula folds sqrt(kWeight)/len in, so the split goes
    // into w_gt1/w_lt1 here: HF bands use sqrt(asymmetry), UHF bands asym.
    FillBand(
        bands, 0, kWUhfMalta * asym, kWUhfMalta / asym, kNorm1Uhf, kMaltaMulli
    );
    FillBand(
        bands, 1, kWUhfMaltaX * asym, kWUhfMaltaX / asym, kNorm1UhfX, kMaltaMulli
    );
    FillBand(
        bands, 2, kWHfMalta * root_asym, kWHfMalta / root_asym, kNorm1Hf,
        kMaltaMulliLf
    );
    FillBand(
        bands, 3, kWHfMaltaX * root_asym, kWHfMaltaX / root_asym, kNorm1HfX,
        kMaltaMulliLf
    );
    FillBand(bands, 4, kWMfMalta, kWMfMalta, kNorm1Mf, kMaltaMulliLf);
    FillBand(bands, 5, kWMfMaltaX, kWMfMaltaX, kNorm1MfX, kMaltaMulliLf);
    return bands;
}

}  // namespace vqm::internal
