// Copyright (c) 2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Model evaluation follows Netflix/vmaf's normalization, chroma correction,
// RBF SVM, polynomial transform and clip in libvmaf/src/predict.c and svm.cpp
// at commit 4611e47a7330cd342498dcc130e1008efbeab6e4. The support vectors
// come from that commit's vmaf_v1.0.16_3d0h.json. The libsvm notice is in
// licenses/LIBSVM-LICENSE.

#include "vmaf_v1_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace vqm::internal {
namespace {

struct SupportVector {
    double coefficient;
    std::array<double, 4> values;
};

#include "vmaf_v1_model_data.inc"

}  // namespace

double VmafV1Predict(const VmafV1Features& features, bool enable_clip) {
    const std::array<double, 4> input{features.cambi, features.speed_chroma_uv,
                                      features.adm3, features.motion3};
    std::array<double, 4> normalized{};
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (!std::isfinite(input[i])) {
            throw std::invalid_argument(
                "vmaf v1: model features must be finite"
            );
        }
        normalized[i] = kSlopes[i + 1] * input[i] + kIntercepts[i + 1];
    }

    // Reference correction first denormalizes the guiding and guided values.
    // It applies only when chroma SpEED is exactly zero, so preserve the same
    // order of operations rather than testing the original argument directly.
    const double chroma = (normalized[1] - kIntercepts[2]) / kSlopes[2];
    if (chroma == 0.0) {
        const double adm = (normalized[2] - kIntercepts[3]) / kSlopes[3];
        const double corrected = -kChromaCorrection * adm + kChromaCorrection;
        normalized[1] = kSlopes[2] * corrected + kIntercepts[2];
    }

    double sum = 0.0;
    for (const SupportVector& vector : kSupportVectors) {
        double distance_squared = 0.0;
        for (std::size_t i = 0; i < normalized.size(); ++i) {
            const double difference = normalized[i] - vector.values[i];
            distance_squared += difference * difference;
        }
        sum += vector.coefficient * std::exp(-kGamma * distance_squared);
    }
    const double raw = (sum - kRho - kIntercepts[0]) / kSlopes[0];
    const double score =
        kTransform[0] + kTransform[1] * raw + kTransform[2] * raw * raw;
    return enable_clip ? std::clamp(score, kClip[0], kClip[1]) : score;
}

}  // namespace vqm::internal
