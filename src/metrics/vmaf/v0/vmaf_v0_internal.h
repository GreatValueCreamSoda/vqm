// Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "metrics/vmaf/v0/vmaf_model.h"
#include "metrics/vmaf/features/vmaf_preprocess.h"

namespace vqm::internal {

// The per-frame spatial features for the floating-point v0.6.1 model. Motion
// is supplied separately as the transition from the previous reference frame.
struct VmafV061Spatial {
    double adm2 = 0.0;
    std::array<double, 4> vif{};
};

struct VmafV061Frame {
    std::uint64_t index = 0;
    VmafV061Features features{};
    double score = 0.0;
};

// Combines ordered spatial features and reference-frame motion transitions.
// A valid first transition is zero. With temporal weighting, Push emits frame 0
// immediately, then each later frame when the next transition arrives, and
// Flush emits the final frame. Without it, motion2 is zero and every Push emits
// its own frame immediately.
// PooledScore returns the arithmetic mean of per-frame scores after Flush, or
// nullopt for an empty sequence. This is an internal building block, not a
// complete metric pipeline: ADM/VIF and source submission are supplied by its
// caller. A single instance is not thread-safe.
class VmafV061Sequence {
public:
    explicit VmafV061Sequence(
        bool transform = false, bool clip = true,
        VmafModel model = VmafModel::Standard1080, bool temporal_weighting = true
    );
    [[nodiscard]] std::optional<VmafV061Frame> Push(
        const VmafV061Spatial& spatial, double transition
    );
    [[nodiscard]] std::optional<VmafV061Frame> Flush();
    [[nodiscard]] std::optional<double> PooledScore() const;
    void Reset() noexcept;

private:
    [[nodiscard]] VmafV061Frame FinishFrame(
        std::uint64_t index, const VmafV061Spatial& spatial, double motion
    );

    VmafMotionTimeline motion_;
    std::optional<VmafV061Spatial> pending_;
    double score_sum_ = 0.0;
    std::uint64_t emitted_ = 0;
    bool finished_ = false;
    bool transform_;
    bool clip_;
    VmafModel model_;
    bool temporal_weighting_;
};

}  // namespace vqm::internal
