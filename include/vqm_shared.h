// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <memory>
#include <optional>
#include "butteraugli.h"
#include "ssimulacra2.h"
#include "vqm_memory_types.h"

namespace vqm {

// Shared, sequential scratch for the two non-temporal metrics. The input
// geometry is fixed, and each Compute call finishes before the next begins.
// It is not thread-safe; use one workspace per concurrent worker. Resident
// handles and AllocateImage are deliberately not exposed: their contents would
// alias the other metric.
struct SharedMetricConfig {
    Size size;
    bool streaming_inputs = true;
    bool half_planes = false;
    ChromaFilter chroma_filter = ChromaFilter::Bilinear;
    float hf_asymmetry = 1.0f;
    float xmul = 1.0f;
    float intensity_target = 80.0f;
    MemoryPolicy memory_policy = MemoryPolicy::Automatic;
    // Defaults to chroma_filter when not specified.
    std::optional<ChromaFilter> butter_chroma_filter = std::nullopt;
};

class SharedMetricWorkspace {
public:
    explicit SharedMetricWorkspace(SharedMetricConfig config);
    SharedMetricWorkspace(SharedMetricConfig config, const VulkanDevice& device);
    ~SharedMetricWorkspace();
    SharedMetricWorkspace(SharedMetricWorkspace&&) noexcept;
    SharedMetricWorkspace& operator=(SharedMetricWorkspace&&) noexcept;
    SharedMetricWorkspace(const SharedMetricWorkspace&) = delete;
    SharedMetricWorkspace& operator=(const SharedMetricWorkspace&) = delete;

    [[nodiscard]] Ssimulacra2Result ComputeSsimulacra2(
        const Image& original, const Image& distorted, float background = 0.5f
    );
    [[nodiscard]] Ssimulacra2Result ComputeSsimulacra2(
        const ImageView& original, const ImageView& distorted,
        float background = 0.5f
    );
    [[nodiscard]] Ssimulacra2Result ComputeSsimulacra2(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        float background = 0.5f
    );
    [[nodiscard]] Ssimulacra2Result ComputeSsimulacra2(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        const VulkanSubmitSync& sync, float background = 0.5f
    );
    [[nodiscard]] ButteraugliScore ComputeButteraugli(
        const Image& original, const Image& distorted, float background = 0.5f
    );
    [[nodiscard]] ButteraugliScore ComputeButteraugli(
        const ImageView& original, const ImageView& distorted,
        float background = 0.5f
    );
    [[nodiscard]] ButteraugliScore ComputeButteraugli(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        float background = 0.5f
    );
    [[nodiscard]] ButteraugliScore ComputeButteraugli(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        const VulkanSubmitSync& sync, float background = 0.5f
    );

    // Actual single shared allocation, counted once, and optional staging.
    [[nodiscard]] PipelineMemoryUsage GetMemoryUsage() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace vqm
