// VMAF motion: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include "support.h"
#include "metrics/vmaf/v1/vmaf_v1_internal.h"

using vqm::internal::VmafMotionFrame;
using vqm::internal::VmafV1MotionTimeline;

namespace {
void Frame(
    const std::optional<VmafMotionFrame>& result, std::uint64_t index,
    double value
) {
    CHECK(result.has_value());
    if (result) {
        CHECK(result->index == index);
        CHECK_NEAR(result->motion2, value, 1e-12);
    }
}
}  // namespace

VQM_TEST(TestVmafV1MotionTimeline) {
    VmafV1MotionTimeline t;
    CHECK(!t.Flush());
    CHECK(!t.Flush());
    CHECK_THROWS_AS(t.Push(0), std::logic_error);
    t.Reset();
    CHECK_THROWS_AS(t.Push(1), std::invalid_argument);
    CHECK(!t.Push(0));
    Frame(t.Flush(), 0, 0);
    t.Reset();
    CHECK(!t.Push(0));
    Frame(t.Push(40), 0, 18);
    Frame(t.Flush(), 1, 18);
    t.Reset();
    CHECK(!t.Push(0));
    CHECK_THROWS_AS(t.Push(-1), std::invalid_argument);
    CHECK_THROWS_AS(
        t.Push(std::numeric_limits<double>::infinity()), std::invalid_argument
    );
    // Exact integer motion3 from pinned libvmaf for four constant 10-bit
    // 64x48 frames with luma values 64, 71, 69, 81 and motion_max_val=18.
    Frame(t.Push(7), 0, 7);
    Frame(t.Push(2), 1, 2);
    Frame(t.Push(12), 2, 2);
    Frame(t.Flush(), 3, 12);
}
