// Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>

#include "support.h"
#include "metrics/vmaf/features/vmaf_adm.h"
#include "metrics/vmaf/features/vmaf_preprocess.h"
#include "metrics/vmaf/v0/vmaf_v0_internal.h"
#include "metrics/vmaf/features/vmaf_vif.h"
#include "vulkan/vulkan_runtime.h"

namespace {
using namespace vqm;
using namespace vqm::internal;

constexpr Size kSize{64, 48};
constexpr std::size_t kPixels = kSize.width * kSize.height;

struct Layout {
    std::uint32_t desc0 = 8 * kPixels;
    std::uint32_t desc1 = desc0 + kSourceDescWords;
    VmafVifBuffers vif{};
    VmafAdmBuffers adm{};
    VmafMotionBuffers motion{};
    std::uint32_t total = 0;

    Layout() {
        std::uint32_t cursor = desc1 + kSourceDescWords;
        std::size_t width = kSize.width;
        std::size_t height = kSize.height;
        for (std::size_t scale = 0; scale < 4; ++scale) {
            const auto pixels = static_cast<std::uint32_t>(width * height);
            vif.reference[scale] = cursor;
            cursor += pixels;
            vif.distorted[scale] = cursor;
            cursor += pixels;
            width /= 2;
            height /= 2;
        }
        vif.scratch = cursor;
        cursor += 5 * kPixels;
        vif.partials = cursor;
        cursor += static_cast<std::uint32_t>(2 * VmafVif::PartialCount(kSize));
        vif.results = cursor;
        cursor += 4;
        motion.luma = vif.reference[0];
        motion.score = cursor++;
        adm.results = cursor;
        cursor += 9;
        adm.reference[0] = vif.reference[0];
        adm.distorted[0] = vif.distorted[0];
        Size adm_size = kSize;
        for (std::size_t scale = 1; scale < 5; ++scale) {
            adm_size = VmafAdmWavelet::OutputSize(adm_size);
            const auto pixels =
                static_cast<std::uint32_t>(adm_size.width * adm_size.height);
            adm.reference[scale] = cursor;
            cursor += pixels;
            adm.distorted[scale] = cursor;
            cursor += pixels;
        }
        const auto bands =
            static_cast<std::uint32_t>(3 * VmafAdm::BandPixels(kSize));
        adm.reference_details = cursor;
        cursor += bands;
        adm.restored = cursor;
        cursor += bands;
        adm.additive = cursor;
        cursor += bands;
        adm.partials = cursor;
        cursor += static_cast<std::uint32_t>(6 * VmafAdm::PartialCount(kSize));
        motion.previous = cursor;
        cursor += kPixels;
        motion.blurred = cursor;
        cursor += kPixels;
        motion.partials = cursor;
        cursor += static_cast<std::uint32_t>(VmafMotion::PartialCount(kSize));
        total = cursor;
    }
};

// Exercises all feature recorders together with sequence scoring. This is an
// integration probe, not a public VMAF pipeline.
class FeatureProbe {
public:
    FeatureProbe()
        : layout_(), runtime_(AcquireVulkanContext(), Config(layout_)) {
        runtime_.CreateArena(Bytes(layout_.total));
        if (!runtime_.arena_visible())
            runtime_.CreateStaging(Bytes(Readback() + 14));
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, Bytes(layout_.total));
        luma_.emplace(runtime_);
        vif_.emplace(runtime_);
        motion_.emplace(runtime_);
        adm_.emplace(runtime_);
    }

    std::array<float, 14> Push(const Image& original, const Image& distorted) {
        runtime_.UploadStreaming(original, distorted);
        runtime_.PrepareSourceBindings({});
        runtime_.ResetCommandPool();
        runtime_.BeginCommandBuffer();
        (void)runtime_.PublishSources({});
        luma_->Record(
            0, layout_.desc0, layout_.vif.reference[0], kSize, original.format
        );
        luma_->Record(
            1, layout_.desc1, layout_.vif.distorted[0], kSize, distorted.format
        );
        vif_->Record(kSize, layout_.vif);
        motion_->Record(kSize, layout_.motion, previous_);
        adm_->Record(kSize, layout_.adm);
        if (runtime_.arena_visible()) {
            runtime_.ComputeToHostBarrier();
        } else {
            runtime_.RecordReadback(layout_.vif.results, Readback(), 14);
        }
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        const float* values = runtime_.arena_visible()
                                  ? runtime_.arena_map() + layout_.vif.results
                                  : runtime_.staging_map() + Readback();
        std::array<float, 14> result{};
        std::copy_n(values, result.size(), result.begin());
        std::swap(layout_.motion.previous, layout_.motion.blurred);
        previous_ = true;
        return result;
    }

private:
    static constexpr std::size_t Readback() {
        return 8 * kPixels + 2 * kSourceDescWords;
    }
    static VulkanRuntime::Config Config(const Layout& layout) {
        VulkanRuntime::Config config;
        config.stream_region = {0, 4 * kPixels};
        config.stream_desc = {layout.desc0, layout.desc1};
        config.staging_desc_offset = 8 * kPixels;
        config.staging_image_bytes = 16 * kPixels;
        config.base_pixels = kPixels;
        config.base_size = kSize;
        config.streaming = true;
        config.push_constant_size = VmafAdm::PushBytes();
        return config;
    }
    Layout layout_;
    VulkanRuntime runtime_;
    std::optional<VmafLumaConverter> luma_;
    std::optional<VmafVif> vif_;
    std::optional<VmafMotion> motion_;
    std::optional<VmafAdm> adm_;
    bool previous_ = false;
};

std::pair<Image, Image> Frame(std::size_t frame, bool noisy) {
    ImageFormat format;
    format.type = PixelType::Word;
    format.depth = 10;
    format.family = ColorFamily::Grey;
    format.matrix = MatrixCoefficients::Bt709;
    format.range = PixelRange::Limited;
    Image original{kSize, format, {}}, distorted{kSize, format, {}};
    original.data.resize(RequiredBytes(format, kSize));
    distorted.data.resize(RequiredBytes(format, kSize));
    for (std::size_t y = 0; y < kSize.height; ++y) {
        for (std::size_t x = 0; x < kSize.width; ++x) {
            const std::size_t base =
                16 +
                ((7 * x + 13 * y + 17 * frame + 29 * ((x / 7 + frame) % 2)) %
                 220);
            const auto ref = static_cast<std::uint16_t>(base << 2);
            const int noise =
                static_cast<int>((41 * x + 37 * y + 17 * frame) % 81) - 40;
            const auto noisy_code = static_cast<std::uint16_t>(
                std::clamp(static_cast<int>(base) + noise, 0, 255) * 4
            );
            const auto dist =
                noisy
                    ? noisy_code
                    : static_cast<std::uint16_t>(ref + 1 + (x + y + frame) % 3);
            const std::size_t offset = 2 * (y * kSize.width + x);
            std::memcpy(original.data.data() + offset, &ref, 2);
            std::memcpy(distorted.data.data() + offset, &dist, 2);
        }
    }
    return {std::move(original), std::move(distorted)};
}

struct Oracle {
    double adm2;
    std::array<double, 4> vif;
    double score;
};
// Full-precision libvmaf float output for make_vmaf_fixtures.py's 10-bit
// low-bits clip, generated by validation/vmaf_reference.cc at pinned revision
// 4611e47a7330cd342498dcc130e1008efbeab6e4.
constexpr std::array<Oracle, 4> kOracle{{
    {0.99996509221431562,
     {0.99722822535314926, 0.99999266410602439, 0.99999693133046386,
      0.99999742555504145},
     97.418560281082009},
    {0.99997434974491772,
     {0.99722448638939631, 0.99999370772159368, 0.99999654147338302,
      0.99999785586121348},
     100.0},
    {0.99997723137461125,
     {0.99721920443688683, 0.99999316267508476, 0.9999964128854345,
      0.99999925906243059},
     100.0},
    {0.9999796703727426,
     {0.99722978079403302, 0.99999369632917856, 0.99999630258731842,
      0.99999848520693191},
     100.0},
}};
// Same reference sequence, with the distorted luma perturbed by deterministic
// noise. These four scores do not hit the model's score clipping limits.
constexpr std::array<Oracle, 4> kNoisyOracle{{
    {0.95447211532900411,
     {0.27213160973403477, 0.59479317623447181, 0.70440189137446862,
      0.87947904597106463},
     70.312588648518869},
    {0.97267603243276934,
     {0.2707631197127271, 0.59608724001999702, 0.70531224308431095,
      0.87873462877624653},
     98.243758166617312},
    {0.94557848847289905,
     {0.26932741643619412, 0.59299580291846032, 0.70414188854560289,
      0.85243172468686357},
     91.517291991577622},
    {0.96159610465622314,
     {0.27085446819196851, 0.59507830163591346, 0.69975476140095272,
      0.85482214324506989},
     96.32054440928529},
}};
constexpr std::array kTransitions{0.0, 33.434216, 29.808189392089844,
                                  33.892772674560547};

}  // namespace

VQM_TEST(TestVmafCodedLumaFeaturesAndScoresTogether) {
    if (!vqm::test::GpuAvailable()) return;
    for (bool noisy : {false, true}) {
        FeatureProbe probe;
        VmafV061Sequence sequence;
        const auto& oracle = noisy ? kNoisyOracle : kOracle;
        std::size_t emitted = 0;
        for (std::size_t i = 0; i < oracle.size(); ++i) {
            const auto [original, distorted] = Frame(i, noisy);
            const auto gpu = probe.Push(original, distorted);
            VmafV061Spatial spatial;
            spatial.adm2 = gpu[13];
            CHECK_NEAR(gpu[13], oracle[i].adm2, 2e-6);
            for (std::size_t scale = 0; scale < 4; ++scale) {
                spatial.vif[scale] = gpu[scale];
                CHECK_NEAR(gpu[scale], oracle[i].vif[scale], 1e-4);
            }
            CHECK_NEAR(gpu[4], kTransitions[i], 1e-4);
            if (const auto result = sequence.Push(spatial, gpu[4])) {
                CHECK(result->index == emitted++);
                CHECK_NEAR(result->score, oracle[result->index].score, 0.01);
            }
        }
        const auto last = sequence.Flush();
        CHECK(last.has_value());
        if (last) {
            CHECK(last->index == emitted++);
            CHECK_NEAR(last->score, oracle[last->index].score, 0.01);
        }
        CHECK(emitted == oracle.size());
        const auto mean = sequence.PooledScore();
        CHECK(mean.has_value());
        double expected_mean = 0.0;
        for (const auto& frame : oracle) expected_mean += frame.score;
        expected_mean /= static_cast<double>(oracle.size());
        if (mean) CHECK_NEAR(*mean, expected_mean, 0.01);
    }
}
