// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include "timing.h"

#include <mutex>

#ifdef VQM_STAGE_TIMING

namespace vqm::internal {
namespace {

constexpr std::string_view kStageNames[kStageCount] = {"prepare",
                                                       "scale",
                                                       "score",
                                                       "opsin",
                                                       "separate",
                                                       "malta",
                                                       "diff",
                                                       "mask",
                                                       "combine",
                                                       "reduce",
                                                       "vmaf_luma",
                                                       "vmaf_vif",
                                                       "vmaf_adm",
                                                       "vmaf_motion",
                                                       "vmaf_score",
                                                       "vmaf_v1_luma",
                                                       "vmaf_v1_adm",
                                                       "vmaf_v1_speed",
                                                       "vmaf_v1_cambi",
                                                       "vmaf_v1_motion",
                                                       "vmaf_v1_score",
                                                       "cvvdp_prepare",
                                                       "cvvdp_temporal",
                                                       "cvvdp_spatial",
                                                       "cvvdp_score",
                                                       "psnr_hvs_prepare",
                                                       "psnr_hvs_blocks",
                                                       "psnr_hvs_reduce",
                                                       "psnr_hvs_score"};

StageTiming g_timings;
// Every pipeline in the process accumulates here, and the bench runs several
// pipelines on several threads, so the adds and the reset take this lock.
std::mutex g_timings_mutex;

}  // namespace

StageTiming& Timings() { return g_timings; }
void ResetTimings() {
    std::lock_guard<std::mutex> lock(g_timings_mutex);
    g_timings = StageTiming{};
}
void AddStageTimings(const StageTiming& delta) {
    std::lock_guard<std::mutex> lock(g_timings_mutex);
    for (std::size_t stage = 0; stage < kStageCount; ++stage) {
        g_timings.seconds[stage] += delta.seconds[stage];
    }
}
void AddStageSeconds(Stage stage, double seconds) {
    std::lock_guard<std::mutex> lock(g_timings_mutex);
    g_timings.seconds[static_cast<std::size_t>(stage)] += seconds;
}
std::string_view StageName(Stage stage) {
    return kStageNames[static_cast<std::size_t>(stage)];
}

}  // namespace vqm::internal

#endif  // VQM_STAGE_TIMING
