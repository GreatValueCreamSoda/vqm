// VMAF CAMBI: Copyright 2016-2020 Netflix, Inc.
// CAMBI pooling: Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vulkan/vulkan.h>
#include "vqm_image.h"

namespace vqm::internal {

class VulkanRuntime;

// ---- Preparation and texture mask ----

// Arena offsets, in 32-bit words. Each region holds width*height words and
// must be disjoint. Input contains coded luma in VmafLumaConverter's FP32
// domain (sample / 2^(depth-8) - 128). All samples must fit the stated depth.
// Processed samples and the binary mask are stored as exactly representable
// integer-valued floats. The derivative plane is scratch, not a result.
struct VmafCambiPrepareBuffers {
    std::uint32_t input = 0;
    std::uint32_t processed = 0;
    std::uint32_t derivative = 0;
    std::uint32_t mask = 0;
};

// CAMBI's same-geometry 10-bit conversion, optional anti-dithering for an
// encode below 10 bits, and 7x7 spatial texture mask. This is only the first
// part of CAMBI, not a score; encoding-size resampling is not implemented.
class VmafCambiPrepare {
public:
    explicit VmafCambiPrepare(VulkanRuntime& runtime);
    ~VmafCambiPrepare();
    VmafCambiPrepare(const VmafCambiPrepare&) = delete;
    VmafCambiPrepare& operator=(const VmafCambiPrepare&) = delete;
    [[nodiscard]] static std::size_t Pixels(Size size);
    [[nodiscard]] static std::uint32_t MaskThreshold(Size size);
    [[nodiscard]] static constexpr std::size_t PushBytes() noexcept {
        return 9 * sizeof(std::uint32_t);
    }
    void Record(
        Size size, std::uint32_t input_depth, std::uint32_t encoded_depth,
        const VmafCambiPrepareBuffers& b
    );

private:
    VulkanRuntime* runtime_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

// ---- Scale decimation ----

// Absolute arena offsets in 32-bit words for two separate integer-valued
// float planes (luma codes and spatial mask). All four regions must be
// disjoint and live until GPU completion.
struct VmafCambiDecimateBuffers {
    std::uint32_t pixels = 0;
    std::uint32_t mask = 0;
    std::uint32_t pixels_out = 0;
    std::uint32_t mask_out = 0;
};
// Decimate one CAMBI scale, picking sample [2*y,2*x] of each input plane.
// Output dimensions are ceil(width/2) by ceil(height/2), including odd sizes.
// Caller owns the source, arena, descriptor bindings, submission and readback.
class VmafCambiDecimate {
public:
    explicit VmafCambiDecimate(VulkanRuntime& runtime);
    ~VmafCambiDecimate();
    VmafCambiDecimate(const VmafCambiDecimate&) = delete;
    VmafCambiDecimate& operator=(const VmafCambiDecimate&) = delete;
    [[nodiscard]] static Size OutputSize(Size size);
    [[nodiscard]] static constexpr std::size_t PushBytes() noexcept {
        return 6 * sizeof(std::uint32_t);
    }
    void Record(Size size, const VmafCambiDecimateBuffers& buffers);

private:
    VulkanRuntime* runtime_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

// ---- Mode filtering and local contrast ----

// All offsets are arena words. Each plane holds size.width*size.height
// elements. Input and mask are integer-valued FP32, as produced by
// VmafCambiPrepare or decimation. Filtered and contrast remain available after
// recording; all regions must be disjoint and live through submission.
struct VmafCambiContrastBuffers {
    std::uint32_t input = 0;
    std::uint32_t mask = 0;
    std::uint32_t horizontal = 0;
    std::uint32_t filtered = 0;
    std::uint32_t contrast = 0;
};

// One CAMBI scale's separable mode filter and local contrast map, using the
// standard v1 SDR 3H model's BT.1886, .06 luminance threshold, .019 TVI,
// four contrast levels and 10-bit limited-range defaults. The map has
// non-negative FP32 values; top-k pooling and further scales are separate.
// Current bounds (<=1920x1080 and <=33-pixel odd windows) cover 1080p and
// 4K input using v1's 1080p high-resolution speedup.
class VmafCambiContrast {
public:
    explicit VmafCambiContrast(
        VulkanRuntime& runtime, bool allow_half = true,
        bool allow_subgroups = true, bool pq = false
    );
    ~VmafCambiContrast();
    VmafCambiContrast(const VmafCambiContrast&) = delete;
    VmafCambiContrast& operator=(const VmafCambiContrast&) = delete;
    [[nodiscard]] static std::size_t Pixels(
        Size size, unsigned window, bool pq = false
    );
    [[nodiscard]] static constexpr std::size_t PushBytes() noexcept {
        return 10 * sizeof(std::uint32_t);
    }
    void Record(Size size, unsigned window, const VmafCambiContrastBuffers& b);
    [[nodiscard]] bool UsesHalfSource() const noexcept { return half_source_; }
    [[nodiscard]] bool UsesSubgroupPipeline() const noexcept {
        return subgroup_pipeline_;
    }
    [[nodiscard]] static constexpr bool SupportsSubgroupRows(
        const VkPhysicalDeviceSubgroupProperties& properties
    ) noexcept {
        return properties.subgroupSize == 32 &&
               (properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
               (properties.supportedOperations &
                VK_SUBGROUP_FEATURE_BASIC_BIT) != 0;
    }

private:
    VulkanRuntime* runtime_;
    bool pq_;
    bool half_source_ = false;
    bool subgroup_pipeline_ = false;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

// ---- Exact top-k pooling ----

// Caller-owned disjoint arena word regions: values contains count finite,
// nonnegative FP32 contrast values; histogram reserves 256 words; state
// reserves 2; partials reserves PartialCount(count); and result reserves 1.
// Selection handles ties and signed zero exactly; the final sum is FP32
// instead of libvmaf's double, so the pooled mean can differ by rounding.
struct VmafCambiPoolBuffers {
    std::uint32_t values = 0;
    std::uint32_t histogram = 0;
    std::uint32_t state = 0;
    std::uint32_t partials = 0;
    std::uint32_t result = 0;
};

class VmafCambiPool {
public:
    explicit VmafCambiPool(VulkanRuntime& runtime);
    ~VmafCambiPool();
    VmafCambiPool(const VmafCambiPool&) = delete;
    VmafCambiPool& operator=(const VmafCambiPool&) = delete;
    [[nodiscard]] static std::size_t PartialCount(std::size_t count);
    // libvmaf clips and then truncates ratio*count to a positive integer.
    [[nodiscard]] static std::uint32_t SelectedCount(
        std::size_t count, double ratio = 0.6
    );
    [[nodiscard]] static constexpr std::size_t PushBytes() noexcept {
        return 19 * sizeof(std::uint32_t);
    }
    // Packed maps and per-scale partial arrays are contiguous in scale order;
    // histograms and state hold 5*256 and 5*2 words, and results 5 words.
    // Disabling the packed grid exercises the device-limit fallback in tests.
    void RecordBatch(
        const std::array<std::size_t, 5>& counts,
        const VmafCambiPoolBuffers& buffers, double ratio = 0.6,
        bool allow_packed = true
    );
    void Record(
        std::size_t count, const VmafCambiPoolBuffers& buffers,
        double ratio = 0.6
    );

private:
    VulkanRuntime* runtime_;
    std::array<VkPipeline, 5> pipelines_{};
};

// ---- Multiscale feature composition ----

// Offsets are absolute arena words, each region disjoint. `input` is FP32
// distorted coded luma (sample / 2^(bits-8) - 128). The following three
// preparation regions have original-size pixels; the five scale regions have
// Plan(size).max_scale_pixels each. Contrast stores all scale maps, histogram is
// 5*256 words, state is 5*2, partials stores all five scales'
// VmafCambiPool::PartialCount values, scale_scores is 5 words and result is one
// word. All memory must outlive GPU completion.
struct VmafCambiBuffers {
    std::uint32_t input = 0;
    std::uint32_t prepared = 0;
    std::uint32_t derivative = 0;
    std::uint32_t mask = 0;
    std::uint32_t scale_input = 0;
    std::array<std::uint32_t, 2> scale_mask{};
    std::uint32_t contrast_horizontal = 0;
    std::uint32_t filtered = 0;
    std::uint32_t contrast = 0;
    std::uint32_t pool_histogram = 0;
    std::uint32_t pool_state = 0;
    std::uint32_t pool_partials = 0;
    std::uint32_t scale_scores = 0;
    std::uint32_t result = 0;
};
struct VmafCambiPlan {
    Size max_scale_size{};
    std::size_t max_scale_pixels = 0;
    std::array<std::size_t, 5> scale_pixels{};
    std::size_t total_scale_pixels = 0;
    std::size_t total_partials = 0;
    unsigned window = 0;
    bool high_res_speedup = false;
};

// Standard SDR v1.0.16 CAMBI: same source/encode geometry, 1080p high-res
// speedup, .06 threshold, default top 60%, and max 17. Encode depth 0 defaults
// to input depth. PQ selects the HDR model's visibility thresholds.
// Plan rejects unsupported dimensions and windows. Caller owns the arena,
// submission and synchronization.
class VmafCambi {
public:
    explicit VmafCambi(VulkanRuntime& runtime, bool pq = false);
    ~VmafCambi();
    VmafCambi(const VmafCambi&) = delete;
    VmafCambi& operator=(const VmafCambi&) = delete;
    [[nodiscard]] static VmafCambiPlan Plan(Size size, bool pq = false);
    [[nodiscard]] static constexpr std::size_t PushBytes() noexcept {
        return VmafCambiPool::PushBytes();
    }
    void Record(
        Size size, std::uint32_t input_depth, std::uint32_t encoded_depth,
        const VmafCambiBuffers& buffers
    );

private:
    VulkanRuntime* runtime_;
    bool pq_;
    VmafCambiPrepare prepare_;
    VmafCambiDecimate decimate_;
    VmafCambiContrast contrast_;
    VmafCambiPool pool_;
    VkPipeline combine_ = VK_NULL_HANDLE;
};

}  // namespace vqm::internal
