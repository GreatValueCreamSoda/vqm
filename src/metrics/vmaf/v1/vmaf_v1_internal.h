// VMAF: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "metrics/vmaf/features/vmaf_preprocess.h"
#include "metrics/vmaf/v1/vmaf_v1_model.h"

namespace vqm::internal {

// Motion3 for pinned VMAF v1. Push receives the reference transition at the
// selected stride (one for standard, two for HFR), or zero before that stride.
class VmafV1MotionTimeline {
public:
    explicit VmafV1MotionTimeline(bool hfr = false) : hfr_(hfr) {}
    [[nodiscard]] std::optional<VmafMotionFrame> Push(double transition);
    [[nodiscard]] std::optional<VmafMotionFrame> Flush();
    [[nodiscard]] bool Drained() const noexcept {
        return finished_ && next_ == frames_;
    }
    void Reset() noexcept;

private:
    [[nodiscard]] double AdvanceHfr(std::uint64_t index, double transition);
    std::uint64_t frames_ = 0;
    std::uint64_t next_ = 0;
    double previous_transition_ = 0;
    double older_transition_ = 0;
    double hfr_motion_ = 0;
    double hfr_processed_ = 0;
    bool hfr_;
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
    explicit VmafV1Sequence(
        bool clip = true, VmafV1Model model = VmafV1Model::Standard1080,
        bool hfr = false
    )
        : motion_(hfr), clip_(clip), model_(model) {}
    [[nodiscard]] std::optional<VmafV1Frame> Push(
        const VmafV1Spatial& spatial, double transition
    );
    [[nodiscard]] std::optional<VmafV1Frame> Flush();
    [[nodiscard]] std::optional<double> PooledScore() const;
    void Reset() noexcept;

private:
    [[nodiscard]] std::optional<VmafV1Frame> FinishFrame(
        const std::optional<VmafMotionFrame>& ready
    );
    VmafV1MotionTimeline motion_;
    std::array<VmafV1Spatial, 3> pending_{};
    std::uint64_t submitted_ = 0;
    double score_sum_ = 0;
    std::uint64_t emitted_ = 0;
    bool clip_;
    VmafV1Model model_;
};

}  // namespace vqm::internal
