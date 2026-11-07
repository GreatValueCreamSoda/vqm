// Copyright (c) 2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

// Predictor vectors were scored by the pinned libvmaf public API using the
// models at .reference/vmaf/model/vmaf_float_v0.6.1.json and
// vmaf_float_4k_v0.6.1.json. The explicit feature inputs below are
// deliberately independent of the feature implementations. See
// tests/metrics/vmaf/model/vmaf_model_oracle.cc for the reference program and
// reproduction.

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

#include "support.h"
#include "metrics/vmaf/v0/vmaf_model.h"

namespace {

struct ModelCase {
    vqm::internal::VmafV061Features features;
    double normal;
    double transform;
    double unclipped;
    double transform_unclipped;
};

// Each expected score is from the reference API's model evaluation with the
// indicated VMAF_MODEL_FLAG_* options (0, ENABLE_TRANSFORM, DISABLE_CLIP and
// both). Cases exercise feature ordering, sparse vectors, both clipping bounds
// and the optional phone transform/rectification.
constexpr std::array<ModelCase, 6> kCases{{
    {{1.0, 0.0, {1.0, 1.0, 1.0, 1.0}},
     97.428042642610308,
     100.0,
     97.428042642610308,
     102.96113796365687},
    {{0.962084, 3.894359729166667, {0.505712, 0.879061, 0.937873, 0.964301}},
     88.514360702568609,
     99.262161656565397,
     88.514360702568609,
     99.262161656565397},
    {{0.4, 30.0, {0.1, 0.2, 0.3, 0.4}},
     7.0190203255104677,
     13.477173301205278,
     7.0190203255104677,
     13.477173301205278},
    {{0.0, 0.0, {0.0, 0.0, 0.0, 0.0}},
     4.3366705109194772,
     9.0610969177989436,
     4.3366705109194772,
     9.0610969177989436},
    {{1.0, 100.0, {1.0, 1.0, 1.0, 1.0}},
     100.0,
     100.0,
     120.78141930868442,
     120.78141930868442},
    {{0.2, 0.5, {0.3, 0.55, 0.6, 0.81}},
     0.0,
     0.0,
     -38.092207950886106,
     -38.092207950886106},
}};

// Pinned libvmaf output for the same feature vectors with
// vmaf_float_4k_v0.6.1.json, flags 0 and DISABLE_CLIP. The 4K model has no
// score transform; the transform cases below exercise VQM's derived extension.
constexpr std::array<double, 6> kFourKNormal{{
    100.0,
    92.473529283744639,
    48.997401469258676,
    6.6757998038431206,
    100.0,
    0.0,
}};
constexpr std::array<double, 6> kFourKUnclipped{{
    100.8937736628905,
    92.473529283744639,
    48.997401469258676,
    6.6757998038431206,
    188.11728083649618,
    -17.631218614586668,
}};
constexpr std::array<double, 3> kPhoneTransform{{
    1.7067469200000001,
    1.7264384399999999,
    -0.0070530499999999999,
}};

}  // namespace

VQM_TEST(TestVmafFloatModelPrediction) {
    using vqm::internal::VmafV061Predict;
    for (const auto& test : kCases) {
        CHECK_NEAR(VmafV061Predict(test.features), test.normal, 1e-9);
        CHECK_NEAR(VmafV061Predict(test.features, true), test.transform, 1e-9);
        CHECK_NEAR(
            VmafV061Predict(test.features, false, false), test.unclipped, 1e-9
        );
        CHECK_NEAR(
            VmafV061Predict(test.features, true, false),
            test.transform_unclipped, 1e-9
        );
    }
}

VQM_TEST(TestVmafFloat4KModelPrediction) {
    using vqm::VmafModel;
    using vqm::internal::VmafV061Predict;
    for (std::size_t i = 0; i < kCases.size(); ++i) {
        const auto& features = kCases[i].features;
        CHECK_NEAR(
            VmafV061Predict(features, false, true, VmafModel::Standard4K),
            kFourKNormal[i], 1e-9
        );
        CHECK_NEAR(
            VmafV061Predict(features, false, false, VmafModel::Standard4K),
            kFourKUnclipped[i], 1e-9
        );
        const double raw = kFourKUnclipped[i];
        const double transformed = std::max(
            raw, kPhoneTransform[0] + kPhoneTransform[1] * raw +
                     kPhoneTransform[2] * raw * raw
        );
        CHECK_NEAR(
            VmafV061Predict(features, true, false, VmafModel::Standard4K),
            transformed, 1e-9
        );
        CHECK_NEAR(
            VmafV061Predict(features, true, true, VmafModel::Standard4K),
            std::clamp(transformed, 0.0, 100.0), 1e-9
        );
    }
}

VQM_TEST(TestVmafFloatModelRejectsUnknownModel) {
    using vqm::VmafModel;
    using vqm::internal::VmafV061Predict;
    const vqm::internal::VmafV061Features features{};
    CHECK_THROWS_AS(
        VmafV061Predict(features, false, true, static_cast<VmafModel>(99)),
        std::invalid_argument
    );
}

// These features and scores are from frame 0 of complete clips evaluated by
// the pinned libvmaf float extractor and model, not injected features. They
// pin the model's agreement with the full upstream pipeline. Reproduce with
// tests/make_vmaf_fixtures.py and build/vmaf-build/vmaf_reference as described
// in docs/vmaf.md.
VQM_TEST(TestVmafFloatModelReferenceVideoFrames) {
    using namespace vqm::internal;
    const VmafV061Features byte_frame{
        0.99665494006088773,
        0.0,
        {0.80599856716261997, 0.98281151596941074, 0.99870425428836973,
         0.99937769451174496}};
    const VmafV061Features low_bits_frame{
        0.99996509221431562,
        0.0,
        {0.99722822535314926, 0.99999266410602439, 0.99999693133046386,
         0.99999742555504145}};
    CHECK_NEAR(VmafV061Predict(byte_frame), 96.497334085832961, 1e-9);
    CHECK_NEAR(VmafV061Predict(low_bits_frame), 97.418560281082009, 1e-9);
}

VQM_TEST(TestVmafFloatModelRejectsNonfiniteFeatures) {
    using namespace vqm::internal;
    VmafV061Features input{};
    input.adm2 = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(VmafV061Predict(input), std::invalid_argument);
    input.adm2 = 0.0;
    input.motion2 = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(VmafV061Predict(input), std::invalid_argument);
    input.motion2 = 0.0;
    input.vif[3] = -std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(VmafV061Predict(input), std::invalid_argument);
}
