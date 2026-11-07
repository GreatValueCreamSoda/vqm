// VMAF SpEED: Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include "support.h"
#include "metrics/vmaf/speed/vmaf_speed.h"
#include "vulkan/vulkan_runtime.h"
#include "vulkan/vulkan_support.h"

namespace {
using namespace vqm::internal;
class SpeedChannelsDevice {
public:
    explicit SpeedChannelsDevice(vqm::Size size = {320, 320})
        : size_(size), runtime_(AcquireVulkanContext(), Config(size)) {
        std::uint32_t offset = 0;
        const auto pixels = static_cast<std::uint32_t>(size.width * size.height);
        const auto output =
            static_cast<std::uint32_t>(VmafSpeedFilter::OutputPixels(size));
        const vqm::Size filtered{size.width / 16, size.height / 16};
        for (auto& value : b_.reference) {
            value = offset;
            offset += pixels;
        }
        for (auto& value : b_.distorted) {
            value = offset;
            offset += pixels;
        }
        b_.filter_scratch = offset;
        offset +=
            4 * static_cast<std::uint32_t>(VmafSpeedFilter::ScratchWords(size));
        b_.down = offset;
        offset += 4 * output;
        b_.reference_filtered = offset;
        offset += 2 * output;
        b_.distorted_filtered = offset;
        offset += 2 * output;
        b_.stats_scratch = offset;
        offset += 2 * static_cast<std::uint32_t>(VmafSpeedStats::ScratchWords());
        b_.partials = offset;
        offset +=
            2 *
            static_cast<std::uint32_t>(VmafSpeedStats::PartialCount(filtered));
        b_.channel_results = offset;
        offset += 4;
        b_.result = offset;
        offset += 2;
        runtime_.CreateArena(Bytes(offset));
        runtime_.CreateStaging(Bytes(4 * pixels + 2));
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, Bytes(offset));
        speed_.emplace(runtime_);
    }
    std::array<float, 2> Compute(
        std::uint32_t frame, bool low_bits, bool flat_u = false,
        bool flat_v = false
    ) {
        const auto pixels =
            static_cast<std::uint32_t>(size_.width * size_.height);
        for (std::uint32_t channel = 0; channel < 2; channel++) {
            const auto plane = channel + 1;
            const bool flat = channel == 0 ? flat_u : flat_v;
            for (std::uint32_t y = 0; y < size_.height; y++)
                for (std::uint32_t x = 0; x < size_.width; x++) {
                    const auto val =
                        16 + ((x * (7 + frame * 2) + y * 13 + (x / 7) * 23 +
                               (y / 11) * 31 + plane * 19 + frame * 33) %
                              220);
                    const auto dis =
                        (val + ((x * 13 + y * (3 + frame) + plane * 17) %
                                (41 + frame * 7))) %
                        256;
                    const float rf =
                        low_bits ? float((x + y + plane) % 4) * 0.25f : 0.0f;
                    const float df =
                        low_bits ? float((x * 3 + y + frame) % 4) * 0.25f : 0.0f;
                    const auto index = y * size_.width + x;
                    runtime_.staging_map()[channel * pixels + index] =
                        flat ? 0.0f : float(val) + rf - 128.0f;
                    runtime_.staging_map()[(channel + 2) * pixels + index] =
                        float(dis) + df - 128.0f;
                }
        }
        return ComputeStaged(pixels);
    }
    // Around the singularity threshold, one quantization step or a change
    // in covariance accumulation can alter libvmaf's channel imputation.
    std::array<float, 2> ComputeNearSingular(std::uint32_t amplitude) {
        const auto pixels =
            static_cast<std::uint32_t>(size_.width * size_.height);
        for (std::uint32_t channel = 0; channel < 2; ++channel) {
            const auto plane = channel + 1;
            for (std::uint32_t y = 0; y < size_.height; ++y) {
                for (std::uint32_t x = 0; x < size_.width; ++x) {
                    const auto i = y * size_.width + x;
                    const auto noise =
                        static_cast<int>(
                            Hash(x, y, plane, 0) % (2 * amplitude + 1)
                        ) -
                        static_cast<int>(amplitude);
                    const auto delta = static_cast<int>(
                                           Hash(x, y, plane, 1) % (amplitude + 1)
                                       ) -
                                       static_cast<int>(amplitude / 2);
                    runtime_.staging_map()[channel * pixels + i] =
                        static_cast<float>(noise) / 256.0f;
                    runtime_.staging_map()[(channel + 2) * pixels + i] =
                        static_cast<float>(noise + delta) / 256.0f;
                }
            }
        }
        return ComputeStaged(pixels);
    }

private:
    static std::uint32_t Hash(
        std::uint32_t x, std::uint32_t y, std::uint32_t plane, std::uint32_t salt
    ) {
        std::uint32_t z = x * 0x9e3779b9u ^ y * 0x85ebca6bu ^
                          plane * 0xc2b2ae35u ^ salt * 0x27d4eb2du;
        z ^= z >> 16;
        z *= 0x7feb352du;
        z ^= z >> 15;
        z *= 0x846ca68bu;
        return z ^ (z >> 16);
    }
    std::array<float, 2> ComputeStaged(std::uint32_t pixels) {
        runtime_.ResetCommandPool();
        runtime_.BeginCommandBuffer();
        const std::array copies{
            VulkanRuntime::CopyRange{0, b_.reference[0], pixels},
            VulkanRuntime::CopyRange{pixels, b_.reference[1], pixels},
            VulkanRuntime::CopyRange{2 * pixels, b_.distorted[0], pixels},
            VulkanRuntime::CopyRange{3 * pixels, b_.distorted[1], pixels}};
        runtime_.RecordStagedCopies(copies);
        runtime_.BindDescriptorSet();
        speed_->Record(size_, b_);
        runtime_.RecordReadback(b_.result, 4 * pixels, 2);
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        return {runtime_.staging_map()[4 * pixels],
                runtime_.staging_map()[4 * pixels + 1]};
    }

    static VulkanRuntime::Config Config(vqm::Size size) {
        VulkanRuntime::Config c;
        c.base_size = size;
        c.base_pixels = size.width * size.height;
        c.push_constant_size = VmafSpeed::PushBytes();
        c.streaming = false;
        return c;
    }
    vqm::Size size_;
    VulkanRuntime runtime_;
    std::optional<VmafSpeed> speed_;
    VmafSpeedBuffers b_{};
};
}  // namespace
VQM_TEST(TestVmafSpeedUVAgainstLibvmaf) {
    if (!vqm::test::GpuAvailable()) return;
    // Reference output from tests/metrics/vmaf/speed/vmaf_speed_stats_oracle.cc,
    // including the combination of U and V. Tolerance accounts for FP32
    // covariance, Jacobi eigenvectors, and reductions; broader numerical error
    // is not characterized.
    constexpr std::array<double, 4> byte{11.95609188079834, 13.781463623046875,
                                         24.095846176147461, 17.203107833862305};
    constexpr std::array<double, 4> high{11.955730438232422, 13.78204345703125,
                                         24.095909118652344, 17.203105926513672};
    SpeedChannelsDevice d;
    for (std::uint32_t frame = 0; frame < byte.size(); ++frame) {
        const auto a = d.Compute(frame, false), b = d.Compute(frame, true);
        CHECK_NEAR(a[0], byte[frame], 0.03);
        CHECK(a[1] == 0.0f);
        CHECK_NEAR(b[0], high[frame], 0.03);
        CHECK(b[1] == 0.0f);
    }
    // If only U's reference is singular, libvmaf imputes U from valid V.
    const auto one_flat = d.Compute(0, false, true, false);
    CHECK_NEAR(one_flat[0], 12.925295829772949, 0.03);
    CHECK(one_flat[1] == 1.0f);
    const auto both_flat = d.Compute(0, false, true, true);
    CHECK_NEAR(both_flat[0], 0.0, 1e-4);
    CHECK(both_flat[1] == 1.0f);
}

VQM_TEST(TestVmafSpeedUV1080AgainstLibvmaf) {
    if (!vqm::test::GpuAvailable()) return;
    // Full-feature oracle uses 1920x1080 YUV444 (1920x1080 chroma planes).
    SpeedChannelsDevice d({1920, 1080});
    const auto byte = d.Compute(0, false);
    const auto ten_bit = d.Compute(0, true);
    CHECK_NEAR(byte[0], 16.837528228759766, 0.03);
    CHECK(byte[1] == 0.0f);
    CHECK_NEAR(ten_bit[0], 16.837627410888672, 0.03);
    CHECK(ten_bit[1] == 0.0f);
}

VQM_TEST(TestVmafSpeedUVNearSingularAgainstLibvmaf) {
    if (!vqm::test::GpuAvailable()) return;
    // Pinned scalar reference output generated by
    // tests/metrics/vmaf/speed/vmaf_speed_singular_oracle.cc, 320x320 YUV444P16.
    // Low-amplitude cases exercise singular-channel imputation, regular
    // low-entropy zero scores, and regular nonzero scores. Status pins the
    // existing VQM singular classification; libvmaf's public API only exposes
    // the score.
    struct Sample {
        std::uint32_t amplitude;
        double score;
        float status;
    };
    constexpr std::array<Sample, 6> samples{{
        {1, 0.0, 1.0f},
        {16, 0.0, 1.0f},
        {128, 0.0, 0.0f},
        {640, 0.0, 0.0f},
        {768, 1.0981630086898804, 0.0f},
        {1024, 2.2217099666595459, 0.0f},
    }};
    SpeedChannelsDevice d;
    for (const auto& [amplitude, expected, status] : samples) {
        const auto result = d.ComputeNearSingular(amplitude);
        CHECK_NEAR(result[0], expected, 0.03);
        CHECK(result[1] == status);
    }
}
