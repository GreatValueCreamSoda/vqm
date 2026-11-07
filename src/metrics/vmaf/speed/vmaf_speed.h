// VMAF SpEED: Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vulkan/vulkan.h>
#include "vqm_image.h"

namespace vqm::internal {

class VulkanRuntime;

// ---- Chroma filtering and downsampling ----

// Arena offsets in words. Input is a tightly packed FP32 chroma plane with
// sample / 2^(bits-8) - 128; temporary and output buffers must be disjoint.
struct VmafSpeedFilterBuffers {
    std::uint32_t input = 0;
    std::uint32_t scratch = 0;  // ScratchWords(size) words.
    std::uint32_t down = 0;     // OutputPixels(size) words.
    std::uint32_t result = 0;   // OutputPixels(size) words.
};

// SpEED's standard v1 prefilter for one chroma plane: anti-alias Gaussian,
// decimation by 16, then local Gaussian mean subtraction. This is one stage,
// not the SpEED feature. The caller owns all memory and submission.
class VmafSpeedFilter {
public:
    explicit VmafSpeedFilter(VulkanRuntime& runtime);
    ~VmafSpeedFilter();
    VmafSpeedFilter(const VmafSpeedFilter&) = delete;
    VmafSpeedFilter& operator=(const VmafSpeedFilter&) = delete;

    [[nodiscard]] static std::size_t ScratchWords(Size size);
    [[nodiscard]] static std::size_t OutputPixels(Size size);
    [[nodiscard]] static constexpr std::size_t PushBytes() noexcept {
        return 19 * sizeof(std::uint32_t);
    }
    void Record(Size size, const VmafSpeedFilterBuffers& buffers);
    void RecordBatch(
        Size size, const std::array<VmafSpeedFilterBuffers, 4>& buffers
    );

private:
    void RecordMany(Size size, std::span<const VmafSpeedFilterBuffers> buffers);
    VulkanRuntime* runtime_;
    std::array<VkPipeline, 2> pipelines_{};
};

// ---- Covariance statistics and pooling ----

// One or two pairs of prefiltered FP32 planes, in tightly packed arena words.
// Each region holds contiguous banks, one per channel pair. Each bank reserves
// size.width*size.height words per input, 2604 scratch words, PartialCount(size)
// partials, and 2 result words (raw SpEED score and status: 0 regular, 1
// singular, 2 invalid or nonconverged). The caller must reject status 2 and keep
// all regions disjoint and alive until work completes.
struct VmafSpeedStatsBuffers {
    std::uint32_t reference = 0;
    std::uint32_t distorted = 0;
    std::uint32_t scratch = 0;
    std::uint32_t partials = 0;
    std::uint32_t result = 0;
};

// Statistics stage after SpEED preprocessing, for U and/or V planes. Uses
// symmetric 25x25 FP32 covariance eigenvalues and a Cholesky variance solve,
// followed by mode-5 pooling. Scratch stores the lower-triangular factor.
// The caller combines channels, imputes singular channels, and clamps to 45.
class VmafSpeedStats {
public:
    // The wide eigen kernel is selected only when both device invocation limits
    // allow it. Disabling it keeps the portable 128-invocation path available.
    explicit VmafSpeedStats(VulkanRuntime& runtime, bool allow_wide = true);
    ~VmafSpeedStats();
    VmafSpeedStats(const VmafSpeedStats&) = delete;
    VmafSpeedStats& operator=(const VmafSpeedStats&) = delete;
    [[nodiscard]] static std::size_t PartialCount(Size size);
    [[nodiscard]] static constexpr std::size_t ScratchWords() noexcept {
        return 2604;
    }
    [[nodiscard]] static constexpr std::size_t PushBytes() noexcept {
        return 9 * sizeof(std::uint32_t);
    }
    void Record(
        Size size, const VmafSpeedStatsBuffers& buffers,
        std::uint32_t channels = 1
    );

private:
    VulkanRuntime* runtime_;
    std::array<VkPipeline, 4> pipelines_{};
};

// ---- UV feature composition ----

// Absolute arena offsets in words. Input is 4 tightly packed FP32 chroma
// planes in the libvmaf coded sample domain, already converted from the input
// YUV format. Scratch region lengths are defined by the corresponding stage:
// filter_scratch: 4*VmafSpeedFilter::ScratchWords(chroma_size),
// down: 4*VmafSpeedFilter::OutputPixels(chroma_size),
// reference_filtered/distorted_filtered: 2*VmafSpeedFilter::OutputPixels,
// stats_scratch: 2*VmafSpeedStats::ScratchWords(),
// partials: 2*VmafSpeedStats::PartialCount({width/16,height/16}),
// channel_results: 4 words, result: 2 words. All regions must be disjoint.
struct VmafSpeedBuffers {
    std::array<std::uint32_t, 2> reference{};
    std::array<std::uint32_t, 2> distorted{};
    std::uint32_t filter_scratch = 0;
    std::uint32_t down = 0;
    std::uint32_t reference_filtered = 0;
    std::uint32_t distorted_filtered = 0;
    std::uint32_t stats_scratch = 0;
    std::uint32_t partials = 0;
    std::uint32_t channel_results = 0;
    std::uint32_t result = 0;
};

// Standard v1.0.16 SpEED UV with default 1x prescale, model noise/floor,
// variance mode 5 and 45 cap. The chroma planes must be at least 80x80.
// Output is score, status: 0 regular, 1 singular channel(s) imputed per
// libvmaf, 2 numerical failure (the caller must not use that score).
// Caller owns the arena, descriptor binding, submission and readback.
class VmafSpeed {
public:
    explicit VmafSpeed(VulkanRuntime& runtime);
    ~VmafSpeed();
    VmafSpeed(const VmafSpeed&) = delete;
    VmafSpeed& operator=(const VmafSpeed&) = delete;
    [[nodiscard]] static constexpr std::size_t PushBytes() noexcept {
        return std::max(
            VmafSpeedFilter::PushBytes(), VmafSpeedStats::PushBytes()
        );
    }
    void Record(Size chroma_size, const VmafSpeedBuffers& buffers);

private:
    VulkanRuntime* runtime_;
    VmafSpeedFilter filter_;
    VmafSpeedStats stats_;
    VkPipeline combine_ = VK_NULL_HANDLE;
};

}  // namespace vqm::internal
