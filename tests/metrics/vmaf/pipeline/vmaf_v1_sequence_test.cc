// VMAF: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
#include <limits>
#include <stdexcept>
#include "support.h"
#include "metrics/vmaf/v1/vmaf_v1_internal.h"

VQM_TEST(TestVmafV1SequenceBoundariesAndPrediction) {
    using namespace vqm::internal;
    VmafV1Sequence sequence;
    CHECK_THROWS_AS(sequence.PooledScore(), std::logic_error);
    CHECK(!sequence.Flush());
    CHECK(!sequence.PooledScore());
    sequence.Reset();
    const VmafV1Spatial spatial{1.2, 3.3, 0.9};
    CHECK(!sequence.Push(spatial, 0));
    const auto first = sequence.Push(spatial, 5);
    CHECK(first.has_value());
    if (first) {
        CHECK(first->index == 0);
        CHECK_NEAR(first->features.motion3, 5, 0);
        // Pinned libvmaf 3H model at this feature point.
        CHECK_NEAR(first->score, 72.085780615749599, 1e-9);
    }
    const auto second = sequence.Push(spatial, 10);
    CHECK(second.has_value());
    if (second) {
        CHECK(second->index == 1);
        CHECK_NEAR(second->features.motion3, 5, 0);
        CHECK_NEAR(second->score, 72.085780615749599, 1e-9);
    }
    const auto third = sequence.Flush();
    CHECK(third.has_value());
    if (third) {
        CHECK(third->index == 2);
        CHECK_NEAR(third->features.motion3, 10, 0);
        CHECK_NEAR(third->score, VmafV1Predict({1.2, 3.3, 0.9, 10}), 1e-9);
    }
    CHECK(!sequence.Flush());
    CHECK_THROWS_AS(sequence.Push(spatial, 0), std::logic_error);
    const auto pooled = sequence.PooledScore();
    CHECK(pooled.has_value());
    if (pooled && first && second && third)
        CHECK_NEAR(
            *pooled, (first->score + second->score + third->score) / 3, 1e-9
        );
    sequence.Reset();
    CHECK(!sequence.Push(spatial, 0));
    const auto single = sequence.Flush();
    CHECK(single.has_value());
    if (single) CHECK_NEAR(single->features.motion3, 0, 0);
}

VQM_TEST(TestVmafV1SequenceRejectsInvalidWithoutAdvancing) {
    using namespace vqm::internal;
    VmafV1Sequence sequence;
    const VmafV1Spatial valid{1.2, 3.3, 0.9};
    auto invalid = valid;
    invalid.cambi = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(sequence.Push(invalid, 0), std::invalid_argument);
    CHECK(!sequence.Push(valid, 0));
    CHECK_THROWS_AS(sequence.Push(valid, -1), std::invalid_argument);
    const auto first = sequence.Push(valid, 5);
    CHECK(first.has_value());
    if (first) CHECK(first->index == 0);
}
