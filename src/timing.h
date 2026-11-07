// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <chrono>
#include <cstddef>
#include <string_view>

namespace vqm::internal {

// Stage names every build knows; the timers that record them are opt-in. The
// first three are SSIMULACRA 2's, followed by Butteraugli's and both VMAF
// pipelines' stages. A run reports only the stages its metric timed, so the
// array is shared. One
// bracket spans more than its name suggests: Diff covers the L2 terms *and*
// the mask's precompute, which share one dispatch (butter_l2.comp).
enum class Stage {
    Ssimulacra2Prepare = 0,
    Ssimulacra2Scale,
    Ssimulacra2Score,
    ButteraugliOpsin,
    ButteraugliSeparate,
    ButteraugliMalta,
    ButteraugliDiff,
    ButteraugliMask,
    ButteraugliCombine,
    ButteraugliReduce,
    VmafLuma,
    VmafVif,
    VmafAdm,
    VmafMotion,
    VmafScore,
    VmafV1Luma,
    VmafV1Adm,
    VmafV1Speed,
    VmafV1Cambi,
    VmafV1Motion,
    VmafV1Score,
    CvvdpPrepare,
    CvvdpTemporal,
    CvvdpSpatial,
    CvvdpScore,
    PsnrHvsPrepare,
    PsnrHvsBlocks,
    PsnrHvsReduce,
    PsnrHvsScore,
    Count,
};

inline constexpr std::size_t kStageCount =
    static_cast<std::size_t>(Stage::Count);

// Opt-in stage timing. Every use site wraps the timed work in its own braces:
// the timer lives until the end of the enclosing block.
#ifdef VQM_STAGE_TIMING

struct StageTiming {
    double seconds[kStageCount] = {};
};

// Process-wide accumulator; the benchmark resets it before each measurement.
// Reading it is the caller's business: the bench reads it with one worker and
// no submission in flight, because several pipelines finishing at once would
// race on the same doubles.
StageTiming& Timings();
void ResetTimings();
std::string_view StageName(Stage stage);

// Accumulating is not the caller's business: every pipeline in the process
// adds to the same struct, and a process may run several pipelines on several
// threads (the bench does), so these two take the accumulator's lock. One call
// per finished submission or per timed host stage, so the lock is nowhere near
// a per-sample cost.
void AddStageTimings(const StageTiming& delta);
void AddStageSeconds(Stage stage, double seconds);

class StageTimer {
public:
    explicit StageTimer(Stage stage)
        : stage_(stage), start_(std::chrono::steady_clock::now()) {}
    ~StageTimer() {
        AddStageSeconds(
            stage_, std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start_
                    )
                        .count()
        );
    }
    StageTimer(const StageTimer&) = delete;
    StageTimer& operator=(const StageTimer&) = delete;

private:
    Stage stage_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace vqm::internal

// clang-format off
#define VQM_CONCAT2(a, b) a##b
#define VQM_CONCAT(a, b) VQM_CONCAT2(a, b)
#define VQM_TIMED(stage)                                            \
    ::vqm::internal::StageTimer VQM_CONCAT(vqm_timer_, __LINE__)( \
        ::vqm::internal::Stage::stage)
// clang-format on

#else
#define VQM_TIMED(stage) ((void)0)

}  // namespace vqm::internal
#endif  // VQM_STAGE_TIMING
