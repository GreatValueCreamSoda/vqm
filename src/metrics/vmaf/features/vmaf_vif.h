// VMAF VIF: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <vulkan/vulkan.h>

#include "vqm_image.h"

namespace vqm::internal {

class VulkanRuntime;

// Absolute arena offsets in 32-bit words. Level 0 planes are already-filled
// luma in libvmaf's FP32 domain (code / 2^(bits-8) - 128). Levels 1..3 are
// distinct caller-allocated planes at floor(width/2^level) x
// floor(height/2^level). `scratch` reserves ScratchWords(size, fused) words,
// where fused reflects whether the scale-zero fused kernel is enabled;
// `partials` reserves 2 * PartialCount(size) words and `results` four words.
// All regions must remain disjoint while recording executes.
struct VmafVifBuffers {
    std::array<std::uint32_t, 4> reference{};
    std::array<std::uint32_t, 4> distorted{};
    std::uint32_t scratch = 0;
    std::uint32_t partials = 0;
    std::uint32_t results = 0;
};

// The default libvmaf FP32 VIF features (v0.6.1): four scales with kernel
// sizes 17, 9, 5 and 3, including prefilter-before-decimation and per-scale
// numerator/denominator reductions. Records four scalar scale scores to the
// arena. The caller owns arena bounds, descriptor binding, submission and
// readback; runtime must outlive this object and have at least PushBytes().
class VmafVif {
public:
    // The optional cap permits exercising lower device limits in tests; it
    // can only lower, never raise, the actual device's shared-memory limit.
    explicit VmafVif(
        VulkanRuntime& runtime, bool allow_fused = true,
        std::uint32_t shared_cap = std::numeric_limits<std::uint32_t>::max(),
        float enhancement_gain_limit = 100.0f
    );
    ~VmafVif();
    VmafVif(const VmafVif&) = delete;
    VmafVif& operator=(const VmafVif&) = delete;

    [[nodiscard]] static std::size_t PartialCount(Size size);
    // With fused scale zero, conservatively reserve two full-resolution
    // planes for the downsample fallback, even when that stage is fused too.
    // The scale-zero fallback needs five full-resolution planes.
    [[nodiscard]] static std::size_t ScratchWords(Size size, bool fused);
    // Source halo, five intermediate planes and reduction scratch use 29 KiB.
    [[nodiscard]] static constexpr bool SupportsFusedScale0(
        std::uint32_t shared_bytes
    ) noexcept {
        return shared_bytes >= 29696;
    }
    // Scale 1's 32x16 fused statistics tile is the largest at 22,528 bytes.
    [[nodiscard]] static constexpr bool SupportsFusedStats(
        std::uint32_t shared_bytes
    ) noexcept {
        return shared_bytes >= 22528;
    }
    // Scale 1 uses 40x24 vec2 source values and 40x8 intermediates.
    [[nodiscard]] static constexpr bool SupportsFusedDown(
        std::uint32_t shared_bytes
    ) noexcept {
        return shared_bytes >= 10240;
    }
    [[nodiscard]] static constexpr std::size_t PushBytes() noexcept {
        return 13 * sizeof(std::uint32_t);
    }
    void Record(
        Size size, const VmafVifBuffers& buffers, bool half_source = false
    );
    [[nodiscard]] bool SupportsHalfSource() const noexcept {
        return half_scale0_ != VK_NULL_HANDLE;
    }

private:
    struct ScalePipelines {
        VkPipeline vertical = VK_NULL_HANDLE;
        VkPipeline horizontal = VK_NULL_HANDLE;
        VkPipeline fused_down = VK_NULL_HANDLE;
        VkPipeline fused_stats = VK_NULL_HANDLE;
    };
    void DestroyPipelines() noexcept;

    VulkanRuntime* runtime_;
    bool fuse_scale0_;
    bool fuse_down_;
    bool fuse_stats_;
    float enhancement_gain_limit_;
    std::array<ScalePipelines, 4> scale_pipelines_{};
    VkPipeline reduce_ = VK_NULL_HANDLE;
    VkPipeline half_scale0_ = VK_NULL_HANDLE;
};

}  // namespace vqm::internal
