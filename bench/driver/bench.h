// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <array>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "inmemory/driver.h"
#include "inmemory/pairs.h"
#include "inmemory/prefetch.h"
#include "io.h"
#include "cvvdp_options.h"
#include "timing.h"
#include "vqm.h"
#include "vqm_views.h"

namespace bench_driver {

using Clock = inmem::Clock;
using inmem::Median;
using inmem::Seconds;
using Pair = inmem::Pair;

// One pair decoded for the prefetch queue: the metric never sees the decode.
struct DecodedPair {
    vqm::Image original;
    vqm::Image distorted;
    double decode_ms = 0.0;
    // What the batch and throughput runners report as the measured geometry.
    int width = 0;
    int height = 0;
};

struct Row {
    Pair pair;
    int width = 0;
    int height = 0;
    double decode_ms = 0.0;
    double core_ms = 0.0;
    double score = 0.0;
};

enum class Metric { Ssimulacra2, Butteraugli, VmafV0, VmafV1, Cvvdp, PsnrHvs };

struct BenchOptions {
    std::vector<Pair> pairs;
    vqm::tool::CvvdpOptions cvvdp;
    bool cvvdp_video = false;
    vqm::image_io::InputColorOptions input;
    std::string pairs_file;
    std::string csv_path;
    std::string stages_csv_path;
    // Native preserves coded YUV; explicit modes normalize to the chosen SDR
    // chroma layout before scoring.
    std::string vmaf_chroma = "native";
    // VMAF v0 only: keep the source geometry or fit it into the model canvas.
    std::string vmaf_fit = "native";
    // VMAF v0.6.1 predictor and feature/score options.
    std::string vmaf_model = "1080";
    bool vmaf_neg = false;
    bool vmaf_transform = false;
    bool vmaf_temporal = true;
    // Empty for the usual single-metric benchmark.
    std::string metric_pair;
    int iterations = 10;
    int warmup = 2;
    int resident = 0;
    // Maximum pairs preloaded for the distinct-input fixed-count arm.
    int resident_ring = 0;
    bool resident_reference = false;
    bool pool = false;
    // Number of independent pipelines scoring concurrently. Each worker owns
    // its arena and keeps at most one submission in flight.
    int workers = 1;
    bool half_planes = false;
    // Print one score per pair and nothing else, in input order. A batch
    // scorer reads this; the reason for an ERROR goes to stderr.
    bool score_only = false;
    // Exercise the borrowed strided-plane API, including its packing/upload.
    bool host_views = false;
    // Two caller-owned Vulkan buffers, uploaded once outside the timer.
    bool vulkan_buffers = false;
    // Decode worker threads for a score-only run (0 = the mode default). The
    // metric still runs on the workers; the pool only keeps pairs ready.
    int decode_threads = 0;
    // Fixed-length throughput window. 0 runs the latency protocol; a positive
    // value runs the throughput arm instead and prints one machine line.
    double throughput_seconds = 0.0;
    // One selected metric determines configuration, dispatch and timed stages.
    Metric metric = Metric::Ssimulacra2;
    [[nodiscard]] bool IsVmaf() const {
        return metric == Metric::VmafV0 || metric == Metric::VmafV1;
    }
    float hf_asymmetry = 1.0f;
    float intensity_target = 80.0f;
    bool help = false;
    bool device_info = false;
};

// Ordered decoding for a serial metric arm. Decode-ahead and synchronous
// median sampling share this policy; metric construction and timers stay in
// their owning drivers. The options' pair list must outlive this decoder.
class PairDecoder {
public:
    explicit PairDecoder(const BenchOptions& options);
    bool Decode(const Pair& pair, DecodedPair& out, std::string& error);

private:
    int attempts_;
    std::optional<inmem::Prefetch<DecodedPair>> prefetch_;
};

std::string Usage(std::string_view program);
bool ParseArgs(
    std::span<const std::string_view> args, BenchOptions& options,
    std::string& error
);
bool LoadOnce(const Pair& pair, DecodedPair& out, std::string& error);
bool MeasureDecode(
    const Pair& pair, int attempts, double* decode_ms, vqm::Image* original,
    vqm::Image* distorted, std::string* error
);
bool WritePairsCsv(const std::string& path, const std::vector<Row>& rows);
bool WriteStagesCsv(
    const std::string& path,
    const std::array<double, vqm::internal::kStageCount>& totals,
    double total_iterations, double total_core_seconds, int first_stage,
    int last_stage, bool concurrent
);
void PrintStages(
    const std::array<double, vqm::internal::kStageCount>& totals,
    double total_iterations, double total_core_seconds, int first_stage,
    int last_stage, bool concurrent
);
void ApplyOverrides(
    const BenchOptions& options, vqm::Image* original, vqm::Image* distorted
);

vqm::ImageView BorrowImage(const vqm::Image& image);

// The same harness builds against both the baseline without view overloads and
// the candidate with them. The baseline's public view helper makes these two
// packed Images before calling Compute, so keep that behavior for comparison.
template <typename Pipeline>
auto ComputeHostViews(
    Pipeline& pipeline, const vqm::ImageView& original,
    const vqm::ImageView& distorted, float background
) {
    if constexpr (requires {
                      pipeline.Compute(original, distorted, background);
                  }) {
        return pipeline.Compute(original, distorted, background);
    } else {
        const auto original_copy = vqm::CopyImage(original);
        const auto distorted_copy = vqm::CopyImage(distorted);
        return pipeline.Compute(original_copy, distorted_copy, background);
    }
}

int RunSsimulacra2(const BenchOptions& options);
int RunButteraugli(const BenchOptions& options);
int RunMetricPair(const BenchOptions& options);
int RunPool(const BenchOptions& options);
int PrintDeviceInfo();
int RunResidentRing(const BenchOptions& options);
int RunCvvdp(const BenchOptions& options);
int RunPsnrHvs(const BenchOptions& options);
int RunVmafForMetric(const BenchOptions& options);
int RunWindowForMetric(const BenchOptions& options);
int RunBatchWorkersForMetric(const BenchOptions& options);

}  // namespace bench_driver
