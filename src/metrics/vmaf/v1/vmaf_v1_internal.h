// VMAF: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

#pragma once

#include <cstdint>
#include <optional>

#include "metrics/vmaf/features/vmaf_preprocess.h"
#include "metrics/vmaf/v1/vmaf_v1_model.h"

namespace vqm::internal {

// Motion3 timeline for the pinned v1.0.16 3H model's default temporal
// options (motion_max_val=18, no moving average, no five-frame window).
// Push receives the transition from the previous reference (zero for the
// first). Frame zero is stamped with the first transition, so no result is
// ready until Push 1. Following frames need one additional transition; Flush
// emits the last, or frame zero with value zero for a single-frame sequence.
class VmafV1MotionTimeline {
public:
    [[nodiscard]] std::optional<VmafMotionFrame> Push(double transition);
    [[nodiscard]] std::optional<VmafMotionFrame> Flush();
    void Reset() noexcept;

private:
    std::uint64_t frames_ = 0;
    double previous_transition_ = 0;
    bool finished_ = false;
};

// These features are supplied by independent GPU feature recorders; this
// class only handles delayed motion and model evaluation, not GPU scoring.
struct VmafV1Spatial {
    double cambi = 0;
    double speed_chroma_uv = 0;
    double adm3 = 0;
};
struct VmafV1Frame {
    std::uint64_t index = 0;
    VmafV1Features features{};
    double score = 0;
};
class VmafV1Sequence {
public:
    explicit VmafV1Sequence(bool clip = true) : clip_(clip) {}
    [[nodiscard]] std::optional<VmafV1Frame> Push(
        const VmafV1Spatial& spatial, double transition
    );
    [[nodiscard]] std::optional<VmafV1Frame> Flush();
    [[nodiscard]] std::optional<double> PooledScore() const;
    void Reset() noexcept;

private:
    [[nodiscard]] VmafV1Frame FinishFrame(
        std::uint64_t index, const VmafV1Spatial& spatial, double motion
    );
    VmafV1MotionTimeline motion_;
    std::optional<VmafV1Spatial> pending_;
    double score_sum_ = 0;
    std::uint64_t emitted_ = 0;
    bool finished_ = false;
    bool clip_;
};

}  // namespace vqm::internal
