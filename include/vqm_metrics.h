// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

#include "butteraugli.h"
#include "cvvdp.h"
#include "psnr_hvs.h"
#include "vmaf.h"
#include "vmaf_v1.h"
#include "ssimulacra2.h"
#include "vqm_memory.h"
#include "vqm_memory_types.h"

namespace vqm {

// Select metrics by assigning their configuration. All inputs have the same
// geometry; each side has its own fixed format. A workspace is one worker and
// is not thread-safe. At least one metric must be selected.
//
// size, streaming_inputs and memory_policy below govern every metric. The
// corresponding fields in the individual configurations, their resident slot
// counts, input_bytes pools, VMAF background and VMAF v1 input_format are
// overridden. This interface uses caller images/views or borrowed Vulkan
// buffers rather than pipeline-local handles. Construct with VulkanDevice to
// use VulkanBufferView inputs; all metrics then use that device and queue.
// In Native VMAF v1 mode both input formats must match. PSNR-HVS requires
// matching opaque native YUV formats and does not use background.
struct MetricsConfig {
    Size size;
    ImageFormat original_format{};
    ImageFormat distorted_format{};
    bool streaming_inputs = true;
    MemoryPolicy memory_policy = MemoryPolicy::Automatic;
    float background = 0.5f;
    std::optional<Ssimulacra2Config> ssimulacra2 = std::nullopt;
    std::optional<ButteraugliConfig> butteraugli = std::nullopt;
    std::optional<VmafConfig> vmaf_v0 = std::nullopt;
    std::optional<VmafV1Config> vmaf_v1 = std::nullopt;
    std::optional<CvvdpConfig> cvvdp = std::nullopt;
    std::optional<PsnrHvsConfig> psnr_hvs = std::nullopt;
};

// One frame's selected scores; fields for unselected metrics are empty.
struct MetricsResult {
    std::uint64_t frame_index = 0;
    std::optional<double> ssimulacra2;
    std::optional<double> butteraugli;
    std::optional<double> vmaf_v0;
    std::optional<double> vmaf_v1;
    // Populated only by ComputeImage; sequences report CVVDP in Finalize.
    std::optional<double> cvvdp;
    // Combined PSNR-HVS in dB; positive infinity is a valid zero-error score.
    std::optional<double> psnr_hvs;
};

// Actual Vulkan allocations. When scratch is present, metric entries contain
// only private retained storage; spatial combines the two spatial metrics.
// Scratch contains the shared arena and optional staging allocation. A
// single-metric MetricPipelines instead reports its standalone allocations
// in the selected metric entry, with no scratch entry.
struct MetricsMemoryUsage {
    std::optional<PipelineMemoryUsage> spatial;
    std::optional<PipelineMemoryUsage> vmaf_v0;
    std::optional<PipelineMemoryUsage> vmaf_v1;
    std::optional<PipelineMemoryUsage> cvvdp;
    std::optional<PipelineMemoryUsage> psnr_hvs;
    std::optional<PipelineMemoryUsage> scratch;
};

// Device requirements for the independently allocated components of one
// worker. The scratch requirement counts the shared arena once. Use the
// scenarios with the budget helpers in vqm_memory.h.
struct MetricsMemoryRequirements {
    std::optional<MetricMemoryRequirements> spatial;
    std::optional<MetricMemoryRequirements> vmaf_v0;
    std::optional<MetricMemoryRequirements> vmaf_v1;
    std::optional<MetricMemoryRequirements> cvvdp;
    std::optional<MetricMemoryRequirements> psnr_hvs;
    std::optional<MetricMemoryRequirements> scratch;
};

// Owns selected pipelines and their memory without combining their results.
// Multiple metrics automatically share maximum-sized scratch; a single metric
// uses standalone storage. MetricsConfig normalization applies to both paths.
// Getters return nullptr for unselected metrics. Returned pointers are borrowed
// and remain valid across moves while the owning collection lives. Do not move
// individual pipelines out of the collection. Use them serially: an unfinished
// submission prevents other selected pipelines from using shared scratch.
// One collection is one worker and is not thread-safe.
class MetricPipelines {
public:
    explicit MetricPipelines(MetricsConfig config);
    MetricPipelines(MetricsConfig config, const VulkanDevice& device);
    ~MetricPipelines();
    MetricPipelines(MetricPipelines&&) noexcept;
    MetricPipelines& operator=(MetricPipelines&&) noexcept;
    MetricPipelines(const MetricPipelines&) = delete;
    MetricPipelines& operator=(const MetricPipelines&) = delete;

    [[nodiscard]] Ssimulacra2Pipeline* Ssimulacra2() noexcept;
    [[nodiscard]] ButteraugliPipeline* Butteraugli() noexcept;
    [[nodiscard]] VmafPipeline* VmafV0() noexcept;
    [[nodiscard]] VmafV1Pipeline* VmafV1() noexcept;
    [[nodiscard]] CvvdpPipeline* Cvvdp() noexcept;
    [[nodiscard]] PsnrHvsPipeline* PsnrHvs() noexcept;
    [[nodiscard]] MetricsMemoryUsage GetMemoryUsage() const;

private:
    friend class Metrics;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Plans the always-shared Metrics allocation path. Multi-metric
// MetricPipelines uses the same plan; for a single-metric collection, query
// that metric's normalized configuration with its individual overload.
[[nodiscard]] MetricsMemoryRequirements GetMemoryRequirements(
    const MetricsConfig& config
);
[[nodiscard]] MetricsMemoryRequirements GetMemoryRequirements(
    const MetricsConfig& config, const VulkanDevice& device
);
// Conservative number of such workers fitting the provided per-heap budgets.
// Shared scratch is counted once; every metric's private storage is counted.
[[nodiscard]] std::size_t MaxWorkersForBudgetWorstCase(
    const MetricsMemoryRequirements& requirements,
    std::span<const std::uint64_t> heap_budgets
);

// Finalization exposes clip metrics and any unconsumed tail. With HFR,
// penultimate_frame precedes last_frame when both are present. Repeated
// Finalize returns the same clip score with an empty tail.
struct MetricsFinalResult {
    std::optional<MetricsResult> last_frame;
    std::optional<CvvdpScore> cvvdp;
    std::uint64_t frames = 0;
    std::optional<MetricsResult> penultimate_frame;
};

class Metrics {
public:
    explicit Metrics(MetricsConfig config);
    Metrics(MetricsConfig config, const VulkanDevice& device);
    ~Metrics();
    Metrics(Metrics&&) noexcept;
    Metrics& operator=(Metrics&&) noexcept;
    Metrics(const Metrics&) = delete;
    Metrics& operator=(const Metrics&) = delete;

    // Consumes an ordered frame pair. VMAF lookahead can delay output; an
    // emitted result contains all selected scores for the same frame_index.
    // Flush emits the final pending result; call Reset for a new sequence.
    // CVVDP-only Compute emits frame-index metadata with empty quality fields;
    // its whole-clip score is available from Finalize.
    // Invalid geometry, format or packed input size is rejected before metric
    // processing and can be retried. A failure during metric processing,
    // including backend validation of Vulkan buffer layouts, requires Reset
    // before retrying, even if no GPU work was submitted.
    [[nodiscard]] std::optional<MetricsResult> Compute(
        const Image& original, const Image& distorted
    );
    [[nodiscard]] std::optional<MetricsResult> Compute(
        const ImageView& original, const ImageView& distorted
    );
    // Tightly packed planes with the configured formats. Each span must be
    // exactly RequiredBytes(format, size); it remains valid through the call.
    [[nodiscard]] std::optional<MetricsResult> Compute(
        std::span<const std::byte> original, std::span<const std::byte> distorted
    );
    [[nodiscard]] std::optional<MetricsResult> Compute(
        const VulkanBufferView& original, const VulkanBufferView& distorted
    );
    // Both sync overloads consume waits once on the first metric and submit
    // signals on the last. A later failure can leave waits consumed and
    // signals unscheduled. A signal denotes GPU completion, not score success.
    [[nodiscard]] std::optional<MetricsResult> Compute(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        const VulkanSubmitSync& sync
    );

    // Score one independent pair, including a one-frame VMAF flush. Requires
    // a new/reset sequence and resets it after successfully returning.
    [[nodiscard]] MetricsResult ComputeImage(
        const Image& original, const Image& distorted
    );
    [[nodiscard]] MetricsResult ComputeImage(
        const ImageView& original, const ImageView& distorted
    );
    [[nodiscard]] MetricsResult ComputeImage(
        std::span<const std::byte> original, std::span<const std::byte> distorted
    );
    [[nodiscard]] MetricsResult ComputeImage(
        const VulkanBufferView& original, const VulkanBufferView& distorted
    );
    [[nodiscard]] MetricsResult ComputeImage(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        const VulkanSubmitSync& sync
    );
    // Close input and emit one remaining result per call. Finalize may follow.
    [[nodiscard]] std::optional<MetricsResult> Flush();
    [[nodiscard]] MetricsFinalResult Finalize();
    void Reset();
    [[nodiscard]] MetricsMemoryUsage GetMemoryUsage() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vqm
