// VMAF ADM: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <vulkan/vulkan.h>

#include "vqm_image.h"
#include "vmaf_v1.h"

namespace vqm::internal {

class VulkanRuntime;

// One FP32, coded-luma-domain image and the four destination wavelet bands.
// Offsets are arena words. Every plane is tightly packed; the low and detail
// output dimensions are ceil(width/2) by ceil(height/2). H, V and D correspond
// to Netflix's adm_dwt_band_t_s. Every region must be disjoint and contained
// in the runtime's arena until the work completes.
struct VmafAdmWaveletBuffers {
    std::uint32_t source = 0;
    std::uint32_t low = 0;
    std::uint32_t horizontal = 0;
    std::uint32_t vertical = 0;
    std::uint32_t diagonal = 0;
};

// One Daubechies-2 analysis stage for ADM. Repeating it on the low band builds
// the four-scale pyramid; this stage alone does not compute ADM or a VMAF score.
// The caller owns the arena, command-buffer lifetime and any score readback.
class VmafAdmWavelet {
public:
    explicit VmafAdmWavelet(VulkanRuntime& runtime);
    ~VmafAdmWavelet();
    VmafAdmWavelet(const VmafAdmWavelet&) = delete;
    VmafAdmWavelet& operator=(const VmafAdmWavelet&) = delete;

    [[nodiscard]] static Size OutputSize(Size input);
    [[nodiscard]] static constexpr std::size_t PushBytes() noexcept {
        return 7 * sizeof(std::uint32_t);
    }
    void Record(Size input, const VmafAdmWaveletBuffers& buffers);

private:
    VulkanRuntime* runtime_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

// Absolute arena offsets, in 32-bit words. Levels 0 are source luma planes;
// levels 1..4 are distinct low-pass outputs with ceil(/2) dimensions. Each
// reference detail/restored/additive region holds three planes, each with
// BandPixels() words; partials holds 6 * PartialCount() words and results holds
// nine words: numerator and denominator for each of four scales, followed by
// ADM2. Both ADM2 and ADM3 consume distorted detail bands in registers during
// their fused wavelet/decouple pass. All used regions must be disjoint and
// remain alive until submission completes.
struct VmafAdmBuffers {
    std::array<std::uint32_t, 5> reference{};
    std::array<std::uint32_t, 5> distorted{};
    std::uint32_t reference_details = 0;
    std::uint32_t restored = 0;
    std::uint32_t additive = 0;
    std::uint32_t partials = 0;
    std::uint32_t results = 0;
};

// Default libvmaf float ADM2 (v0.6.1): four wavelet scales, directional
// decoupling, Watson97 contrast sensitivity, contrast masking and p=3 norms.
// The caller supplies FP32 coded luma, arena regions, command submission and
// readback. Input geometry must be at least 17x17 so every output band is at
// least 2x2, matching the valid neighbor reads in the reference implementation.
class VmafAdm {
public:
    explicit VmafAdm(VulkanRuntime& runtime, bool negative = false);
    ~VmafAdm();
    VmafAdm(const VmafAdm&) = delete;
    VmafAdm& operator=(const VmafAdm&) = delete;

    [[nodiscard]] static std::size_t BandPixels(Size input);
    [[nodiscard]] static std::size_t PartialCount(Size input);
    [[nodiscard]] static constexpr std::size_t PushBytes() noexcept {
        return 14 * sizeof(std::uint32_t);
    }
    void Record(Size input, const VmafAdmBuffers& buffers);

private:
    VulkanRuntime* runtime_;
    std::array<VkPipeline, 4> pipelines_{};
};

// ADM3 (v1.0.16 3H)

// Float approximation of the v1.0.16 SDR ADM3 feature. The buffers have the
// same layout as VmafAdmBuffers, except partials need 9*PartialCount words and
// results need 13 words: (DLM numerator, denominator, AIM numerator) for each
// scale, then ADM3. The source is still coded luma scaled to 8-bit values.
class VmafAdm3 {
public:
    explicit VmafAdm3(VulkanRuntime& runtime);
    ~VmafAdm3();
    VmafAdm3(const VmafAdm3&) = delete;
    VmafAdm3& operator=(const VmafAdm3&) = delete;
    static constexpr std::size_t PushBytes() noexcept {
        return VmafAdm::PushBytes();
    }
    void Record(
        Size input, const VmafAdmBuffers& buffers,
        VmafV1Model model = VmafV1Model::Standard1080
    );

private:
    VulkanRuntime* runtime_;
    std::array<VkPipeline, 4> pipelines_{};
};

}  // namespace vqm::internal
