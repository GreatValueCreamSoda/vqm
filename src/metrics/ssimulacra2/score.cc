// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include "ssimulacra2_internal.h"
#include "timing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

// The weighted sum of the 108 norms and the reference's monotone mapping to
// the quality range ending at 100 (poor matches may score below zero).

namespace vqm {
namespace {

constexpr double kScoreScale = 0.9562382616834844;
constexpr double kScoreLinear = 2.326765642916932;
constexpr double kScoreQuadratic = 0.020884521182843837;
constexpr double kScoreCubic = 6.248496625763138e-05;
constexpr double kScoreExponent = 0.6276336467831387;
constexpr double kPerfectScore = 100.0;
constexpr double kScoreDrop = 10.0;

// Tuned subjective weights, one row per (channel, scale): channels X, Y, B and
// scales 1:1 ... 1:32. Each row gives the 1-norm and 4-norm weight for the
// SSIM', ringing and blurring maps, in the order Score() accumulates them.
struct ScaleWeights {
    double ssim_1;
    double ring_1;
    double blur_1;
    double ssim_4;
    double ring_4;
    double blur_4;
};

constexpr std::array<
    ScaleWeights,
    internal::kSsimulacra2Channels * internal::kSsimulacra2MaxScales>
    kScaleWeights{{
        // X, 1:1
        {0.0, 0.0007376606707406586, 0.0, 0.0, 0.0007793481682867309, 0.0},
        // X, 1:2
        {0.0, 0.0004371155730107379, 0.0, 1.1041726426657346,
         0.00066284834129271, 0.00015231632783718752},
        // X, 1:4
        {0.0, 0.0016406437456599754, 0.0, 1.8422455520539298, 11.441172603757666,
         0.0},
        // X, 1:8
        {0.0007989109436015163, 0.000176816438078653, 0.0, 1.8787594979546387,
         10.94906990605142, 0.0},
        // X, 1:16
        {0.0007289346991508072, 0.9677937080626833, 0.0, 0.00014003424285435884,
         0.9981766977854967, 0.00031949755934435053},
        // X, 1:32
        {0.0004550992113792063, 0.0, 0.0, 0.0013648766163243398, 0.0, 0.0},
        // Y, 1:1
        {0.0, 0.0, 0.0, 7.466890328078848, 0.0, 17.445833984131262},
        // Y, 1:2
        {0.0006235601634041466, 0.0, 0.0, 6.683678146179332,
         0.00037724407979611296, 1.027889937768264},
        // Y, 1:4
        {225.20515300849274, 0.0, 0.0, 19.213238186143016, 0.0011401524586618361,
         0.001237755635509985},
        // Y, 1:8
        {176.39317598450694, 0.0, 0.0, 24.43300999870476, 0.28520802612117757,
         0.0004485436923833408},
        // Y, 1:16
        {0.0, 0.0, 0.0, 34.77906344483772, 44.835625328877896, 0.0},
        // Y, 1:32
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        // B, 1:1
        {0.0, 0.0008680556573291698, 0.0, 0.0, 0.0, 0.0},
        // B, 1:2
        {0.0, 0.0005313191874358747, 0.0, 0.00016533814161379112, 0.0, 0.0},
        // B, 1:4
        {0.0, 0.0, 0.0, 0.0004179171803251336, 0.0017290828234722833, 0.0},
        // B, 1:8
        {0.0020827005846636437, 0.0, 0.0, 8.826982764996862, 23.19243343998926,
         0.0},
        // B, 1:16
        {95.1080498811086, 0.9863978034400682, 0.9834382792465353,
         0.0012286405048278493, 171.2667255897307, 0.9807858872435379},
        // B, 1:32
        {0.0, 0.0, 0.0, 0.0005130064588990679, 0.0, 0.00010854057858411537},
    }};

static_assert(kScaleWeights.size() == 18);

// Entries at or below this are pruned: a weight that small cannot add more
// than a hundredth of a norm to the score's weighted sum. The 0.01 is Vship's
// threshold (.reference/Vship/src/Vulkan/ssimu2/score.hpp:4, weight_pruning);
// its scoring pass prunes one map at a time (:125-131), where the unit here is
// a whole (channel, level) pair, because one workgroup produces all six of its
// norms (see Ssimulacra2ChannelPruned).
constexpr double kWeightPruning = 0.01;

}  // namespace

namespace internal {

bool Ssimulacra2ChannelPruned(
    std::size_t channel, std::size_t scale, std::size_t scales
) {
    if (channel >= kSsimulacra2Channels || scale >= scales ||
        scales > kSsimulacra2MaxScales) {
        return false;
    }
    const ScaleWeights& w = kScaleWeights[channel * scales + scale];
    return std::max(
               {w.ssim_1, w.ring_1, w.blur_1, w.ssim_4, w.ring_4, w.blur_4}
           ) <= kWeightPruning;
}

}  // namespace internal

double Ssimulacra2Result::Score() const {
    VQM_TIMED(
        Ssimulacra2Score
    );  // spans the whole function; early returns still record.
    if (scales.size() > internal::kSsimulacra2MaxScales) {
        throw std::invalid_argument("Ssimulacra2Result has too many scales");
    }
    double error = 0.0;
    for (std::size_t c = 0; c < internal::kSsimulacra2Channels; ++c) {
        for (std::size_t s = 0; s < scales.size(); ++s) {
            // The reference consumes the weight table as one flat run, so a
            // channel starts at c * scales.size(), not at c *
            // kSsimulacra2MaxScales.
            const ScaleWeights& w = kScaleWeights[c * scales.size() + s];
            const Ssimulacra2Scale& m = scales[s];
            error += w.ssim_1 * std::abs(m.avg_ssim[internal::SsimNorm(c, 0)]);
            error +=
                w.ring_1 * std::abs(m.avg_edgediff[internal::RingNorm(c, 0)]);
            error +=
                w.blur_1 * std::abs(m.avg_edgediff[internal::BlurNorm(c, 0)]);
            error += w.ssim_4 * std::abs(m.avg_ssim[internal::SsimNorm(c, 1)]);
            error +=
                w.ring_4 * std::abs(m.avg_edgediff[internal::RingNorm(c, 1)]);
            error +=
                w.blur_4 * std::abs(m.avg_edgediff[internal::BlurNorm(c, 1)]);
        }
    }
    if (!std::isfinite(error)) {
        throw std::invalid_argument(
            "Ssimulacra2Result contains nonfinite or overflowing norms"
        );
    }
    const double scaled = error * kScoreScale;
    const double mapped = kScoreLinear * scaled -
                          kScoreQuadratic * scaled * scaled +
                          kScoreCubic * scaled * scaled * scaled;
    if (!std::isfinite(mapped)) {
        throw std::invalid_argument(
            "Ssimulacra2Result score mapping overflowed"
        );
    }
    if (mapped > 0.0) {
        return kPerfectScore - kScoreDrop * std::pow(mapped, kScoreExponent);
    }
    return kPerfectScore;
}

}  // namespace vqm
