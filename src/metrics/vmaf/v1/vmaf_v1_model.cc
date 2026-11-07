// Copyright (c) 2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Model evaluation follows Netflix/vmaf's normalization, chroma correction,
// RBF SVM, polynomial transform and clip in libvmaf/src/predict.c and svm.cpp
// at commit 4611e47a7330cd342498dcc130e1008efbeab6e4. The support vectors
// come from that commit's vmaf_v1.0.16_3d0h.json and
// hdr_vmaf_v1.1.8_1d5h_2160.json. The libsvm notice is in
// licenses/LIBSVM-LICENSE.

#include "vmaf_v1_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace vqm::internal {
namespace {

struct SupportVector {
    double coefficient;
    std::array<double, 4> values;
};

#include "vmaf_v1_model_data.inc"

struct Predictor {
    const std::array<double, 5>& slopes;
    const std::array<double, 5>& intercepts;
    double chroma_correction, gamma, rho;
    std::span<const SupportVector> support_vectors;
};

constexpr Predictor kSdr{kSdrSlopes, kSdrIntercepts, kSdrChromaCorrection,
                         kSdrGamma,  kSdrRho,        kSdrSupportVectors};
constexpr Predictor kHdr{kHdrSlopes, kHdrIntercepts, kHdrChromaCorrection,
                         kHdrGamma,  kHdrRho,        kHdrSupportVectors};

}  // namespace

const VmafV1ModelParameters& VmafV1Parameters(VmafV1Model model) {
    const auto index = static_cast<std::size_t>(model);
    if (index >= kModels.size())
        throw std::invalid_argument("vmaf v1: invalid model");
    return kModels[index];
}

double VmafV1Predict(
    const VmafV1Features& features, bool enable_clip, VmafV1Model model
) {
    const auto& parameters = VmafV1Parameters(model);
    const auto& predictor = model == VmafV1Model::Hdr4K ? kHdr : kSdr;
    const auto& [slopes, intercepts, correction, gamma, rho, vectors] =
        predictor;
    const std::array<double, 4> input{features.cambi, features.speed_chroma_uv,
                                      features.adm3, features.motion3};
    std::array<double, 4> normalized{};
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (!std::isfinite(input[i])) {
            throw std::invalid_argument(
                "vmaf v1: model features must be finite"
            );
        }
        normalized[i] = slopes[i + 1] * input[i] + intercepts[i + 1];
    }

    // Reference correction first denormalizes the guiding and guided values.
    // It applies only when chroma SpEED is exactly zero, so preserve the same
    // order of operations rather than testing the original argument directly.
    const double chroma = (normalized[1] - intercepts[2]) / slopes[2];
    if (chroma == 0.0) {
        const double adm = (normalized[2] - intercepts[3]) / slopes[3];
        const double corrected = -correction * adm + correction;
        normalized[1] = slopes[2] * corrected + intercepts[2];
    }

    double sum = 0.0;
    for (const SupportVector& vector : vectors) {
        double distance_squared = 0.0;
        for (std::size_t i = 0; i < normalized.size(); ++i) {
            const double difference = normalized[i] - vector.values[i];
            distance_squared += difference * difference;
        }
        sum += vector.coefficient * std::exp(-gamma * distance_squared);
    }
    const double raw = (sum - rho - intercepts[0]) / slopes[0];
    const auto& [p0, p1, p2] = parameters.transform;
    const double score = p0 + p1 * raw + p2 * raw * raw;
    return enable_clip
               ? std::clamp(score, parameters.clip[0], parameters.clip[1])
               : score;
}

}  // namespace vqm::internal
