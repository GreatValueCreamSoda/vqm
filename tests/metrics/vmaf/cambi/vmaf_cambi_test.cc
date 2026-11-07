// VMAF CAMBI: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include "support.h"
#include "metrics/vmaf/cambi/vmaf_cambi.h"
#include "vulkan/vulkan_runtime.h"
#include "vulkan/vulkan_support.h"
namespace {
using namespace vqm::internal;
class CambiDevice {
public:
    explicit CambiDevice(vqm::Size size)
        : size_(size),
          plan_(VmafCambi::Plan(size)),
          runtime_(AcquireVulkanContext(), Config(size)) {
        const auto pixels = static_cast<std::uint32_t>(size.width * size.height);
        const auto sp = static_cast<std::uint32_t>(plan_.max_scale_pixels);
        std::uint32_t offset = 0;
        b_.input = offset;
        offset += pixels;
        b_.prepared = offset;
        offset += pixels;
        b_.derivative = offset;
        offset += pixels;
        b_.mask = offset;
        offset += pixels;
        b_.scale_input = offset;
        offset += sp;
        for (auto& mask : b_.scale_mask) {
            mask = offset;
            offset += sp;
        }
        b_.contrast_horizontal = offset;
        offset += sp;
        b_.filtered = offset;
        offset += sp;
        b_.contrast = offset;
        offset += static_cast<std::uint32_t>(plan_.total_scale_pixels);
        b_.pool_histogram = offset;
        offset += 5 * 256;
        b_.pool_state = offset;
        offset += 5 * 2;
        b_.pool_partials = offset;
        offset += static_cast<std::uint32_t>(plan_.total_partials);
        b_.scale_scores = offset;
        offset += 5;
        b_.result = offset;
        offset++;
        runtime_.CreateArena(Bytes(offset));
        runtime_.CreateStaging(Bytes(pixels + 6));
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, Bytes(offset));
        cambi_.emplace(runtime_);
    }
    float Compute(
        std::uint32_t frame, std::uint32_t depth, std::uint32_t encoded_depth
    ) {
        const auto pixels = size_.width * size_.height;
        for (std::size_t y = 0; y < size_.height; y++)
            for (std::size_t x = 0; x < size_.width; x++) {
                auto val = frame == 0 ? 16 + (x * 219) / (size_.width - 1)
                           : frame == 1
                               ? 16 + (((x / 24) + (y / 32) * 3) % 11) * 19
                           : frame == 2 ? 64 + ((x / 16) % 24)
                                        : 128 + ((x / 4) % 16);
                float sample = static_cast<float>(val);
                if (depth > 8) {
                    const auto scale = std::size_t{1} << (depth - 8);
                    sample += static_cast<float>((x + y + frame) % scale) /
                              static_cast<float>(scale);
                }
                runtime_.staging_map()[y * size_.width + x] = sample - 128.0f;
            }
        runtime_.ResetCommandPool();
        runtime_.BeginCommandBuffer();
        const std::array copies{VulkanRuntime::CopyRange{0, b_.input, pixels}};
        runtime_.RecordStagedCopies(copies);
        runtime_.BindDescriptorSet();
        cambi_->Record(size_, depth, encoded_depth, b_);
        runtime_.RecordReadback(b_.scale_scores, pixels, 6);
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        return runtime_.staging_map()[pixels + 5];
    }

private:
    static VulkanRuntime::Config Config(vqm::Size size) {
        VulkanRuntime::Config c;
        c.base_size = size;
        c.base_pixels = size.width * size.height;
        c.push_constant_size = VmafCambi::PushBytes();
        c.streaming = false;
        return c;
    }
    vqm::Size size_;
    VmafCambiPlan plan_;
    VulkanRuntime runtime_;
    std::optional<VmafCambi> cambi_;
    VmafCambiBuffers b_{};
};
}  // namespace
VQM_TEST(TestVmafCambiPlan) {
    const auto p = VmafCambi::Plan({320, 320});
    CHECK(p.window == 7);
    CHECK(!p.high_res_speedup);
    CHECK(p.max_scale_pixels == 320 * 320);
    const auto q = VmafCambi::Plan({1920, 1080});
    CHECK(q.window == 17);
    CHECK(q.high_res_speedup);
    CHECK(q.max_scale_size.width == 960);
    CHECK(q.max_scale_size.height == 540);
    const auto odd = VmafCambi::Plan({321, 319});
    CHECK(
        (odd.scale_pixels ==
         std::array<std::size_t, 5>{102399, 25760, 6480, 1640, 420})
    );
    CHECK(odd.total_scale_pixels == 136699);
    CHECK(odd.total_partials == 536);
    CHECK_THROWS_AS(VmafCambi::Plan({64, 64}), std::invalid_argument);
    CHECK_THROWS_AS(
        VmafCambi::Plan({std::numeric_limits<std::size_t>::max(), 1080}),
        std::invalid_argument
    );
}
VQM_TEST(TestVmafCambiFeatureAgainstLibvmaf) {
    if (!vqm::test::GpuAvailable()) return;
    // Complete-feature oracles from
    // tests/metrics/vmaf/cambi/vmaf_cambi_oracle.cc at the pinned v1 feature
    // options. Includes nonzero score and encoded-depth override.
    CambiDevice small({320, 320});
    CHECK_NEAR(small.Compute(0, 8, 0), 0.0, 0.02);
    CHECK_NEAR(small.Compute(1, 10, 0), 0.0, 0.02);
    CHECK_NEAR(small.Compute(1, 10, 8), 0.75995860511063251, 0.02);
    CHECK_NEAR(small.Compute(2, 8, 0), 17.0, 0.02);
    CambiDevice odd({321, 319});
    CHECK_NEAR(odd.Compute(1, 12, 0), 3.8102786310941479, 0.03);
    CHECK_NEAR(odd.Compute(2, 12, 0), 11.423063720555481, 0.03);
    CHECK_NEAR(odd.Compute(1, 12, 8), 3.2910959641780724, 0.03);
    CambiDevice full({1920, 1080});
    CHECK_NEAR(full.Compute(0, 8, 0), 10.140030192791988, 0.03);
    CHECK_NEAR(full.Compute(0, 10, 0), 0.0, 0.02);
    CHECK_NEAR(full.Compute(1, 10, 8), 0.092913066678087103, 0.02);
    CHECK_NEAR(full.Compute(0, 12, 0), 3.2534804711950209, 0.03);
}
