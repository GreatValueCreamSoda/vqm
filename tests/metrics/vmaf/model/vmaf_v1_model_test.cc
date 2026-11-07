// Copyright (c) 2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Golden scores come from the pinned libvmaf model via its public API.
// See tests/metrics/vmaf/model/vmaf_v1_model_oracle.cc for the reproduction
// program.

#include <array>
#include <limits>
#include <stdexcept>

#include "support.h"
#include "metrics/vmaf/v1/vmaf_v1_model.h"

VQM_TEST(TestVmafV1ModelAgainstLibvmaf) {
    using namespace vqm::internal;
    struct Case {
        VmafV1Features features;
        double score;
        double unclipped;
    };
    constexpr std::array<Case, 7> cases{{
        {{0.0, 0.0, 1.0, 0.0}, 100.0, 100.54849808402878},
        {{1.2, 3.3, 0.9, 5.0}, 72.085780615749599, 72.085780615749599},
        {{17.0, 45.0, 0.5, 18.0}, 0.0, -22.903316757376526},
        {{0.1, 0.0, 0.72, 10.0}, 11.471586003328635, 11.471586003328635},
        {{0.1, 1e-8, 0.72, 10.0}, 32.8582481518688, 32.8582481518688},
        {{0.0, 1.0, 1.0, 0.0}, 99.571416240843092, 99.571416240843092},
        {{10.0, 30.0, 0.55, 0.0}, 0.0, -33.334575806401681},
    }};
    for (const auto& test : cases) {
        CHECK_NEAR(VmafV1Predict(test.features), test.score, 1e-9);
        CHECK_NEAR(VmafV1Predict(test.features, false), test.unclipped, 1e-9);
    }
}

VQM_TEST(TestVmafV1ModelRejectsNonfinite) {
    using namespace vqm::internal;
    VmafV1Features features{};
    features.cambi = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(VmafV1Predict(features), std::invalid_argument);
    features = {};
    features.speed_chroma_uv = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(VmafV1Predict(features), std::invalid_argument);
    features = {};
    features.adm3 = -std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(VmafV1Predict(features), std::invalid_argument);
    features = {};
    features.motion3 = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(VmafV1Predict(features), std::invalid_argument);
}
