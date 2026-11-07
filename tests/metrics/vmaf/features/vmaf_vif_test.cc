// VMAF VIF: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "support.h"
#include "metrics/vmaf/features/vmaf_vif.h"
#include "vqm_vulkan.h"
#include "vulkan/vulkan_runtime.h"
#include "vulkan/vulkan_support.h"

namespace {

using vqm::Size;
using vqm::internal::VmafVif;
using vqm::internal::VmafVifBuffers;
using vqm::internal::VulkanRuntime;

class VifDevice {
public:
    explicit VifDevice(
        Size size, bool allow_fused = true,
        std::uint32_t shared_cap = std::numeric_limits<std::uint32_t>::max(),
        bool allow_float16 = true
    )
        : size_(size),
          pixels_(size.width * size.height),
          partials_(VmafVif::PartialCount(size)),
          runtime_(Context(allow_float16), Config(size)) {
        std::size_t offset = 0;
        Size level = size;
        for (std::size_t i = 0; i < 4; ++i) {
            const std::size_t count = level.width * level.height;
            buffers_.reference[i] = static_cast<std::uint32_t>(offset);
            offset += count;
            buffers_.distorted[i] = static_cast<std::uint32_t>(offset);
            offset += count;
            level = {level.width / 2, level.height / 2};
        }
        guard_before_ = static_cast<std::uint32_t>(offset++);
        buffers_.scratch = static_cast<std::uint32_t>(offset);
        const bool fused =
            allow_fused &&
            VmafVif::SupportsFusedScale0(
                std::min(
                    runtime_.context().limits.maxComputeSharedMemorySize,
                    shared_cap
                )
            );
        offset += VmafVif::ScratchWords(size, fused);
        guard_after_ = static_cast<std::uint32_t>(offset++);
        buffers_.partials = static_cast<std::uint32_t>(offset);
        offset += 2 * partials_;
        buffers_.results = static_cast<std::uint32_t>(offset);
        offset += 4;
        runtime_.CreateArena(vqm::internal::Bytes(offset));
        runtime_.CreateStaging(vqm::internal::Bytes(2 * pixels_ + 6));
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, vqm::internal::Bytes(offset));
        vif_.emplace(runtime_, allow_fused, shared_cap);
    }

    std::array<float, 4> Compute(
        const std::vector<float>& reference, const std::vector<float>& distorted,
        bool half_source = false
    ) {
        if (reference.size() != pixels_ || distorted.size() != pixels_)
            throw std::invalid_argument("test geometry");
        std::copy(reference.begin(), reference.end(), runtime_.staging_map());
        std::copy(
            distorted.begin(), distorted.end(), runtime_.staging_map() + pixels_
        );
        constexpr float kGuard = -123456.5f;
        runtime_.staging_map()[2 * pixels_] = kGuard;
        runtime_.staging_map()[2 * pixels_ + 1] = kGuard;
        runtime_.ResetCommandPool();
        runtime_.BeginCommandBuffer();
        const std::array copies{
            VulkanRuntime::CopyRange{0, buffers_.reference[0], pixels_},
            VulkanRuntime::CopyRange{pixels_, buffers_.distorted[0], pixels_},
            VulkanRuntime::CopyRange{2 * pixels_, guard_before_, 1},
            VulkanRuntime::CopyRange{2 * pixels_ + 1, guard_after_, 1}};
        runtime_.RecordStagedCopies(copies);
        runtime_.BindDescriptorSet();
        vif_->Record(size_, buffers_, half_source);
        runtime_.RecordReadback(buffers_.results, 2 * pixels_, 4);
        runtime_.RecordReadback(guard_before_, 2 * pixels_ + 4, 1);
        runtime_.RecordReadback(guard_after_, 2 * pixels_ + 5, 1);
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        CHECK(runtime_.staging_map()[2 * pixels_ + 4] == kGuard);
        CHECK(runtime_.staging_map()[2 * pixels_ + 5] == kGuard);
        std::array<float, 4> result;
        std::copy_n(runtime_.staging_map() + 2 * pixels_, 4, result.begin());
        return result;
    }
    bool SupportsHalfSource() const { return vif_->SupportsHalfSource(); }

private:
    static std::shared_ptr<vqm::internal::VulkanContext> Context(
        bool allow_float16
    ) {
        auto context = vqm::internal::AcquireVulkanContext();
        if (allow_float16) return context;
        // Borrow the same handles without promising shaderFloat16 was enabled.
        // Physical support alone must not select a feature on a borrowed device.
        const vqm::VulkanDevice device{context->physical_device,
                                       context->device,
                                       context->queue,
                                       context->queue_family,
                                       context->sixteen_bit_storage,
                                       false};
        return vqm::internal::BorrowVulkanContext(device);
    }
    static VulkanRuntime::Config Config(Size size) {
        VulkanRuntime::Config config;
        config.base_size = size;
        config.base_pixels = size.width * size.height;
        config.push_constant_size = VmafVif::PushBytes();
        config.streaming = false;
        return config;
    }

    Size size_;
    std::size_t pixels_;
    std::size_t partials_;
    std::uint32_t guard_before_ = 0;
    std::uint32_t guard_after_ = 0;
    VulkanRuntime runtime_;
    VmafVifBuffers buffers_{};
    std::optional<VmafVif> vif_;
};

}  // namespace

VQM_TEST(TestVmafVifGeometry) {
    CHECK(!VmafVif::SupportsFusedScale0(20480));
    CHECK(!VmafVif::SupportsFusedScale0(29695));
    CHECK(VmafVif::SupportsFusedScale0(29696));
    CHECK(!VmafVif::SupportsFusedStats(22527));
    CHECK(VmafVif::SupportsFusedStats(22528));
    CHECK(!VmafVif::SupportsFusedDown(10239));
    CHECK(VmafVif::SupportsFusedDown(10240));
    CHECK(VmafVif::ScratchWords({17, 19}, true) == 2 * 17 * 19);
    CHECK(VmafVif::ScratchWords({17, 19}, false) == 5 * 17 * 19);
    CHECK_THROWS_AS(
        VmafVif::ScratchWords({15, 16}, true), std::invalid_argument
    );
    CHECK(VmafVif::PartialCount({16, 16}) == 1);
    CHECK(VmafVif::PartialCount({17, 19}) == 4);
    CHECK_THROWS_AS(VmafVif::PartialCount({15, 16}), std::invalid_argument);
    CHECK_THROWS_AS(
        VmafVif::PartialCount({std::numeric_limits<std::size_t>::max(), 16}),
        std::invalid_argument
    );
}

VQM_TEST(TestVmafVifAgainstLibvmaf) {
    if (!vqm::test::GpuAvailable()) return;
    // Pinned libvmaf float_vif output for the four 64x48 frames produced by
    // tests/make_vmaf_fixtures.py. The moving fixture has the same normalized
    // samples and oracle values at 8, 10, and 12 bits. The low-bit fixtures
    // exercise genuine 10- and 12-bit fractional values. On an RTX 5070,
    // maximum errors across these twelve comparisons were 3.25e-6, 1.61e-6,
    // 1.31e-6, and 1.26e-6 for scales 0..3. These fixtures do not characterize
    // drift for other content or devices.
    constexpr Size size{64, 48};
    constexpr std::array<std::array<double, 4>, 4> moving{{
        {{0.80599856716261997, 0.98281151596941074, 0.99870425428836973,
          0.99937769451174496}},
        {{0.80418852829714915, 0.98236972736456563, 0.99861397624422610,
          0.99965659473195390}},
        {{0.80537753864161310, 0.98290544456183004, 0.99880281959416306,
          0.99964036742723594}},
        {{0.80501375021837052, 0.98225946144622978, 0.99877530907866563,
          0.99957960037085702}},
    }};
    constexpr std::array<std::array<double, 4>, 4> low_bits{{
        {{0.99722822535314926, 0.99999266410602439, 0.99999693133046386,
          0.99999742555504145}},
        {{0.99722448638939631, 0.99999370772159368, 0.99999654147338302,
          0.99999785586121348}},
        {{0.99721920443688683, 0.99999316267508476, 0.99999641288543450,
          0.99999925906243059}},
        {{0.99722978079403302, 0.99999369632917856, 0.99999630258731842,
          0.99999848520693191}},
    }};
    constexpr std::array<std::array<double, 4>, 4> low_bits_12{{
        {{0.99578559434557501, 0.99906142081040872, 0.99968070493476580,
          0.99984215989185199}},
        {{0.99591415577924580, 0.99915724956904806, 0.99975490574707981,
          0.99991646435287795}},
        {{0.99568440155788385, 0.99902350568710041, 0.99966258226608407,
          0.99993674245500963}},
        {{0.99594281921963890, 0.99913168666211838, 0.99974865296708504,
          1.0000008910547460}},
    }};
    std::vector<float> reference(size.width * size.height);
    std::vector<float> distorted(reference.size());
    std::vector<float> subcode(reference.size());
    std::vector<float> subcode12(reference.size());
    VifDevice device(size);
    for (std::size_t frame = 0; frame < moving.size(); ++frame) {
        for (std::size_t y = 0; y < size.height; ++y) {
            for (std::size_t x = 0; x < size.width; ++x) {
                const std::size_t i = y * size.width + x;
                const std::size_t base =
                    16 +
                    ((7 * x + 13 * y + 17 * frame + 29 * ((x / 7 + frame) % 2)) %
                     220);
                reference[i] = static_cast<float>(base) - 128.0f;
                distorted[i] = static_cast<float>(std::min<std::size_t>(
                                   235, base + (x + y + frame) % 9
                               )) -
                               128.0f;
                subcode[i] = reference[i] +
                             static_cast<float>(1 + (x + y + frame) % 3) * 0.25f;
                subcode12[i] =
                    reference[i] +
                    static_cast<float>(1 + (x + y + frame) % 15) / 16.0f;
            }
        }
        const auto actual = device.Compute(reference, distorted);
        const auto actual_low_bits = device.Compute(reference, subcode);
        const auto actual_low_bits_12 = device.Compute(reference, subcode12);
        const auto identity = device.Compute(reference, reference);
        for (std::size_t scale = 0; scale < 4; ++scale) {
            CHECK_NEAR(actual[scale], moving[frame][scale], 1e-4);
            CHECK_NEAR(actual_low_bits[scale], low_bits[frame][scale], 1e-4);
            CHECK_NEAR(
                actual_low_bits_12[scale], low_bits_12[frame][scale], 1e-4
            );
            CHECK_NEAR(identity[scale], 1.0, 1e-5);
        }
        CHECK(actual_low_bits[0] < 0.999f);
    }
}

VQM_TEST(TestVmafVifOddGeometryAndLowVariance) {
    if (!vqm::test::GpuAvailable()) return;
    constexpr Size size{17, 19};
    std::vector<float> reference(size.width * size.height, -128.0f);
    VifDevice device(size);
    const auto identity = device.Compute(reference, reference);
    std::vector<float> distorted = reference;
    for (std::size_t i = 0; i < distorted.size(); ++i)
        distorted[i] += static_cast<float>(i % 4) * 0.25f;
    const auto differing = device.Compute(reference, distorted);
    for (std::size_t scale = 0; scale < identity.size(); ++scale) {
        CHECK_NEAR(identity[scale], 1.0, 1e-5);
        CHECK(std::isfinite(differing[scale]));
    }
}

VQM_TEST(TestVmafVifConstantHighLumaPreservesLowVariance) {
    if (!vqm::test::GpuAvailable()) return;
    constexpr Size size{64, 48};
    // Native 10-bit codes 914/913 become these exactly representable samples.
    // Both fields have zero variance despite their different DC levels. FP16
    // products round reference^2 upward by 3.75 before a FP32 sum can help,
    // incorrectly crossing VIF's variance-2 branch and producing VIF zero.
    constexpr float kReference = 914.0f / 4.0f - 128.0f;
    constexpr float kDistorted = 913.0f / 4.0f - 128.0f;
    const std::vector<float> reference(size.width * size.height, kReference);
    const std::vector<float> distorted(reference.size(), kDistorted);
    VifDevice fused(size), fallback(size, false);
    const auto full = fused.Compute(reference, distorted);
    const auto packed_source = fused.Compute(reference, distorted, true);
    const auto separate = fallback.Compute(reference, distorted);
    for (std::size_t scale = 0; scale < full.size(); ++scale) {
        CHECK_NEAR(full[scale], 1.0, 1e-5);
        CHECK_NEAR(packed_source[scale], 1.0, 1e-5);
        CHECK_NEAR(separate[scale], 1.0, 1e-5);
    }
}

VQM_TEST(TestVmafVifOddScaleAgainstLibvmaf) {
    if (!vqm::test::GpuAvailable()) return;
    // Pinned libvmaf float VIF for the 8-bit moving sequence generated by
    // tests/make_vmaf_fixtures.py. Both geometries produce an odd 9-pixel
    // dimension at scale 1, and 34x18 also produces a 17-pixel width. On an
    // RTX 5070 the largest absolute error across these frames and scales was
    // 1.50e-5 (18x18) and 3.84e-6 (34x18); this is not a general error bound.
    constexpr std::array<std::array<double, 4>, 4> expected18{{
        {{0.80441568490054105, 0.97867233441404877, 0.99783878692304906,
          0.99951558183165223}},
        {{0.80195710385326224, 0.97816538372204997, 0.99643289580159222,
          0.99414408124422970}},
        {{0.80702834162884207, 0.97772373227138343, 0.99705065994562514,
          0.99484194946311877}},
        {{0.80157644269653916, 0.97521006116904696, 0.99628812116749788,
          0.99792158546270626}},
    }};
    constexpr std::array<std::array<double, 4>, 4> expected34{{
        {{0.80294162242738643, 0.97922715020984119, 0.99825650892606743,
          0.99970672636823599}},
        {{0.80416705482481332, 0.98013550907892200, 0.99743710437691346,
          0.99735298880039669}},
        {{0.80287867013804204, 0.98020922810318845, 0.99755972250532554,
          0.99764583284161534}},
        {{0.80592578807152737, 0.97904039498353534, 0.99739752037107277,
          0.99932797912409699}},
    }};
    // Additional odd source sizes: pinned libvmaf with YUV444 inputs. The
    // largest absolute errors over these frames on RTX 5070 were 8.64e-6
    // (17x19) and 5.09e-6 (33x19); these are not general error bounds.
    constexpr std::array<std::array<double, 4>, 4> expected17{{
        {{0.80530046069134398, 0.97993765570672986, 0.9976839301887992,
          0.99952299249422738}},
        {{0.80034301286312759, 0.97936039037469957, 0.99614252767986722,
          0.99423127854280291}},
        {{0.80613758204174324, 0.98030752818023303, 0.99693633719895014,
          0.99487626563393183}},
        {{0.80005260567912995, 0.97713457337364029, 0.99619664102753047,
          0.99796728265123968}},
    }};
    constexpr std::array<std::array<double, 4>, 4> expected33{{
        {{0.80459145412055377, 0.98098694705045819, 0.99818618647808877,
          0.9996932215334966}},
        {{0.80497345406874565, 0.98071969126231529, 0.99739967095272686,
          0.99735874871702812}},
        {{0.8036554205189822, 0.98114000346473773, 0.99744574357956883,
          0.99764697244662182}},
        {{0.80655913576419935, 0.97976835888674474, 0.99716916796856148,
          0.99930614363758996}},
    }};
    for (const Size size :
         {Size{17, 19}, Size{18, 18}, Size{33, 19}, Size{34, 18}}) {
        VifDevice device(size);
        std::vector<float> reference(size.width * size.height);
        std::vector<float> distorted(reference.size());
        for (std::size_t frame = 0; frame < 4; ++frame) {
            for (std::size_t y = 0; y < size.height; ++y) {
                for (std::size_t x = 0; x < size.width; ++x) {
                    const std::size_t i = y * size.width + x;
                    const std::size_t base = 16 + ((7 * x + 13 * y + 17 * frame +
                                                    29 * ((x / 7 + frame) % 2)) %
                                                   220);
                    reference[i] = static_cast<float>(base) - 128.0f;
                    distorted[i] = static_cast<float>(std::min<std::size_t>(
                                       235, base + (x + y + frame) % 9
                                   )) -
                                   128.0f;
                }
            }
            const auto actual = device.Compute(reference, distorted);
            const auto& expected = size.width == 17   ? expected17
                                   : size.width == 18 ? expected18
                                   : size.width == 33 ? expected33
                                                      : expected34;
            for (std::size_t scale = 0; scale < 4; ++scale)
                CHECK_NEAR(actual[scale], expected[frame][scale], 1e-4);
        }
    }
}

VQM_TEST(TestVmafVifNearBlackAndHighContrastAgainstLibvmaf) {
    if (!vqm::test::GpuAvailable()) return;
    // One-frame 64x48 YUV420 inputs scored with the pinned float model.
    // The checkerboard alternates codes 16 and 235 in 3x3 blocks. On an RTX
    // 5070 the largest absolute error across these cases was 5.77e-7; this is
    // not a general error bound.
    constexpr Size size{64, 48};
    constexpr std::array<std::array<double, 4>, 4> expected{{
        {{0.99995811780293786, 1.0, 1.0, 1.0}},
        {{0.26247251033782959, 0.99870181083679199, 0.99995557467142737,
          0.99996121724446618}},
        {{0.0, 0.0, 0.87671188632162045, 0.86261280794171424}},
        {{0.99999992052714026, 0.99999992052714026, 1.0, 1.0}},
    }};
    VifDevice device(size);
    std::vector<float> reference(size.width * size.height);
    std::vector<float> distorted(reference.size());
    for (std::size_t pattern = 0; pattern < expected.size(); ++pattern) {
        for (std::size_t y = 0; y < size.height; ++y) {
            for (std::size_t x = 0; x < size.width; ++x) {
                const std::size_t i = y * size.width + x;
                const bool dark = (x / 3 + y / 3) % 2 != 0;
                if (pattern == 0) {
                    reference[i] = static_cast<float>(16 + (x + y) % 3) - 128.0f;
                    distorted[i] =
                        static_cast<float>(16 + (x + y + 1) % 3) - 128.0f;
                } else if (pattern == 1) {
                    reference[i] = 16.0f - 128.0f;
                    distorted[i] = (dark ? 16.0f : 235.0f) - 128.0f;
                } else if (pattern == 2) {
                    reference[i] = (dark ? 16.0f : 235.0f) - 128.0f;
                    distorted[i] = (dark ? 235.0f : 16.0f) - 128.0f;
                } else {
                    reference[i] = 16.0f - 128.0f;
                    distorted[i] = reference[i];
                }
            }
        }
        const auto actual = device.Compute(reference, distorted);
        for (std::size_t scale = 0; scale < 4; ++scale)
            CHECK_NEAR(actual[scale], expected[pattern][scale], 1e-4);
    }
}

VQM_TEST(TestVmafVifFusedFallback) {
    if (!vqm::test::GpuAvailable()) return;
    for (const Size size : {Size{16, 16}, Size{33, 17}, Size{64, 48}}) {
        const std::size_t count = size.width * size.height;
        std::vector<float> a(count), b(count);
        for (std::size_t i = 0; i < count; ++i) {
            a[i] = static_cast<float>((i * 37 + i / size.width * 13) % 1001) *
                       0.25f -
                   128.0f;
            b[i] = a[i] + static_cast<float>((i * 13) % 19) * 0.25f;
        }
        VifDevice fused(size), fallback(size, false),
            down_only(size, true, 23000), stats_fallback(size, true, 22000);
        const auto first = fused.Compute(a, b), second = fallback.Compute(a, b);
        const auto third = down_only.Compute(a, b);
        const auto fourth = stats_fallback.Compute(a, b);
        for (std::size_t i = 0; i < first.size(); ++i) {
            // The fused paths group 32 output columns per workgroup, while the
            // two-pass fallback reduces 16; the partial-sum order differs.
            CHECK(std::abs(first[i] - second[i]) < 1e-5f);
            CHECK(std::abs(first[i] - third[i]) < 1e-5f);
            CHECK(std::abs(first[i] - fourth[i]) < 1e-5f);
        }
    }
}

VQM_TEST(TestVmafVifHalfSourceRequiresEnabledFeature) {
    if (!vqm::test::GpuAvailable()) return;
    constexpr Size size{64, 48};
    std::vector<float> a(size.width * size.height), b(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        // Coded 10-bit luma is exactly representable in FP16, so selecting
        // the optional shared source storage must preserve every VIF result.
        a[i] = static_cast<float>((37 * i) % 1024) * 0.25f - 128.0f;
        b[i] = a[i] + static_cast<float>((13 * i) % 19) * 0.25f;
    }
    VifDevice enabled(size);
    VifDevice disabled(
        size, true, std::numeric_limits<std::uint32_t>::max(), false
    );
    CHECK(!disabled.SupportsHalfSource());
    const auto context = vqm::internal::AcquireVulkanContext();
    CHECK(
        enabled.SupportsHalfSource() ==
        (context->shader_float16 &&
         VmafVif::SupportsFusedScale0(
             context->limits.maxComputeSharedMemorySize
         ))
    );
    const auto full = enabled.Compute(a, b);
    const auto half = enabled.Compute(a, b, true);
    const auto fallback = disabled.Compute(a, b, true);
    for (std::size_t scale = 0; scale < full.size(); ++scale) {
        CHECK_NEAR(full[scale], half[scale], 1e-6);
        CHECK_NEAR(full[scale], fallback[scale], 1e-6);
    }
}
