// VMAF: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>
#include "input.h"
#include "support.h"
#include "vmaf_v1.h"
#include "vmaf_sdr_fixtures.h"
#include "vqm_memory.h"

namespace {
using namespace vqm;
constexpr Size kV1Size{512, 384};
struct V1Oracle {
    double cambi, speed, adm3, motion3, score;
};
#include "vmaf_v1_reference.inc"

std::uint16_t Sample(
    unsigned plane, std::size_t x, std::size_t y, std::size_t frame,
    bool distorted
) {
    int value;
    if (plane == 0) {
        value =
            80 + static_cast<int>((x / 8 * 5 + y / 9 * 3 + frame * 11) % 151);
        if (distorted) {
            value = ((value + 3) / 8) * 8;
            if ((x / 23 + y / 19 + frame) % 5 == 0) value += 2;
        }
    } else if (plane == 1) {
        value = 512 + static_cast<int>((x * 3 + y * 5 + frame * 29) % 241) - 120;
        if (distorted)
            value += static_cast<int>((x * 5 + y * 7 + frame * 13) % 65) - 32;
    } else {
        value = 512 + static_cast<int>((x * 7 + y * 2 + frame * 17) % 201) - 100;
        if (distorted)
            value += static_cast<int>((x * 11 + y * 3 + frame * 9) % 49) - 24;
    }
    return static_cast<std::uint16_t>(std::clamp(value, 0, 1023));
}
Image MakeFrame(const ImageFormat& format, std::size_t frame, bool distorted) {
    Image result{kV1Size, format, {}};
    result.data.resize(RequiredBytes(format, kV1Size));
    const auto planes = vqm::internal::PlaneLayout(format, kV1Size);
    for (unsigned plane = 0; plane < 3; ++plane) {
        for (std::size_t y = 0; y < planes[plane].height; ++y) {
            for (std::size_t x = 0; x < planes[plane].width; ++x) {
                const auto sample = Sample(plane, x, y, frame, distorted);
                const auto offset =
                    planes[plane].offset + 2 * (y * planes[plane].width + x);
                std::memcpy(
                    result.data.data() + offset, &sample, sizeof(sample)
                );
            }
        }
    }
    return result;
}
void CheckFrame(
    const VmafV1Frame& frame, const V1Oracle& expected, std::size_t index
) {
    CHECK(frame.index == index);
    CHECK_NEAR(frame.cambi, expected.cambi, 0.04);
    CHECK_NEAR(frame.speed_chroma_uv, expected.speed, 0.08);
    CHECK_NEAR(frame.adm3, expected.adm3, 0.005);
    CHECK_NEAR(frame.motion3, expected.motion3, 0.04);
    CHECK_NEAR(frame.score, expected.score, 0.35);
}
template <std::size_t N>
void RunVideo(
    VmafV1Pipeline& pipeline, const ImageFormat& format,
    const std::array<V1Oracle, N>& expected, bool resident
) {
    double expected_sum = 0.0;
    std::size_t received = 0;
    for (std::size_t frame = 0; frame < N; ++frame) {
        auto a = MakeFrame(format, frame, false);
        auto b = MakeFrame(format, frame, true);
        std::optional<VmafV1Frame> result;
        if (resident) {
            auto ref = pipeline.Upload(a);
            auto dist = pipeline.Upload(b);
            result = pipeline.Push(ref, dist);
            pipeline.Release(ref);
            pipeline.Release(dist);
        } else {
            result = pipeline.Push(a, b);
        }
        if (result) {
            CHECK(received < N);
            if (received < N) CheckFrame(*result, expected[received], received);
            ++received;
        }
        expected_sum += expected[frame].score;
    }
    if (const auto result = pipeline.Flush()) {
        CHECK(received < N);
        if (received < N) CheckFrame(*result, expected[received], received);
        ++received;
    }
    CHECK(received == N);
    const auto pooled = pipeline.PooledScore();
    CHECK(pooled.has_value());
    if (pooled) CHECK_NEAR(*pooled, expected_sum / static_cast<double>(N), 0.35);
}
}  // namespace

VQM_TEST(TestVmafV1PipelineReferenceSequences) {
    if (!vqm::test::GpuAvailable()) return;
    ImageFormat format;
    format.type = PixelType::Word;
    format.depth = 10;
    format.family = ColorFamily::Yuv;
    format.matrix = MatrixCoefficients::Bt709;
    format.range = PixelRange::Limited;
    VmafV1Config config;
    config.size = kV1Size;
    config.input_format = format;
    VmafV1Pipeline pipeline(config);
    RunVideo(pipeline, format, kOracle444Single, false);
    pipeline.Reset();
    RunVideo(pipeline, format, kOracle444Video, false);
    config.encoded_bit_depth = 8;
    VmafV1Pipeline encoded8(config);
    RunVideo(encoded8, format, kOracle444VideoEnc8, false);
    format.subsample_w = 1;
    format.subsample_h = 1;
    config.input_format = format;
    config.encoded_bit_depth = 0;
    VmafV1Pipeline subsampled(config);
    RunVideo(subsampled, format, kOracle420Single, false);
    subsampled.Reset();
    RunVideo(subsampled, format, kOracle420Video, true);
}

VQM_TEST(TestVmafV1StagedMemoryPreservesTemporalSequence) {
    if (!vqm::test::GpuAvailable()) {
        vqm::test::MarkSkip("no usable Vulkan device", true);
        return;
    }
    ImageFormat format;
    format.type = PixelType::Word;
    format.depth = 10;
    format.family = ColorFamily::Yuv;
    format.matrix = MatrixCoefficients::Bt709;
    format.range = PixelRange::Limited;
    format.subsample_w = 1;
    format.subsample_h = 1;
    VmafV1Config config;
    config.size = kV1Size;
    config.input_format = format;
    config.memory_policy = MemoryPolicy::Staged;
    if (!GetMemoryRequirements(config).staged.supported) {
        vqm::test::MarkSkip("no compatible staged memory type");
        return;
    }
    VmafV1Pipeline pipeline(config);
    CHECK(pipeline.GetMemoryUsage().uses_staging);
    RunVideo(pipeline, format, kOracle420Video, false);
}

// Compare the GPU's generic input path to an independently generated 16-bit
// limited-range BT.709 image, scored through the unchanged coded-YUV path.
VQM_TEST(TestVmafV1SdrInputConversion) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm::test::sdr;
    struct Case {
        PixelType type;
        ColorFamily family;
        AlphaType alpha;
        VmafChromaSampling sampling;
    };
    const std::array cases{
        Case{PixelType::Byte, ColorFamily::Rgb, AlphaType::None,
             VmafChromaSampling::Yuv444},
        Case{PixelType::Byte, ColorFamily::Rgb, AlphaType::Straight,
             VmafChromaSampling::Yuv420},
        Case{PixelType::Float, ColorFamily::Rgb, AlphaType::Premultiplied,
             VmafChromaSampling::Yuv444},
        Case{PixelType::Half, ColorFamily::Rgb, AlphaType::None,
             VmafChromaSampling::Yuv420},
        Case{PixelType::Word, ColorFamily::Rgb, AlphaType::None,
             VmafChromaSampling::Yuv444},
        Case{PixelType::Byte, ColorFamily::Grey, AlphaType::None,
             VmafChromaSampling::Yuv420},
    };
    for (const auto& c : cases) {
        const auto original = MakeSource(c.type, c.family, c.alpha, false);
        const auto distorted = MakeSource(c.type, c.family, c.alpha, true);
        const bool subsampled = c.sampling == VmafChromaSampling::Yuv420;
        const auto reference_original = ReferenceYuv(original, subsampled);
        const auto reference_distorted = ReferenceYuv(distorted, subsampled);
        VmafV1Config source_config;
        source_config.size = vqm::test::sdr::kSize;
        source_config.input_format = original.format;
        source_config.chroma_sampling = c.sampling;
        VmafV1Config reference_config;
        reference_config.size = vqm::test::sdr::kSize;
        reference_config.input_format = reference_original.format;
        VmafV1Pipeline source(source_config);
        VmafV1Pipeline reference(reference_config);
        const auto compare = [](const VmafV1Frame& a, const VmafV1Frame& b) {
            CHECK(a.index == b.index);
            CHECK_NEAR(a.cambi, b.cambi, 0.05);
            CHECK_NEAR(a.speed_chroma_uv, b.speed_chroma_uv, 0.05);
            CHECK_NEAR(a.adm3, b.adm3, 0.001);
            CHECK_NEAR(a.motion3, b.motion3, 0.01);
            CHECK_NEAR(a.score, b.score, 0.1);
        };
        CHECK(!source.Push(original, distorted));
        CHECK(!reference.Push(reference_original, reference_distorted));
        // Reuse an uploaded pair on the second call, exercising the resident
        // descriptor as well as the streaming descriptor for the same frame.
        const auto uploaded_original = source.Upload(original);
        const auto uploaded_distorted = source.Upload(distorted);
        const auto a = source.Push(uploaded_original, uploaded_distorted);
        const auto b = reference.Push(reference_original, reference_distorted);
        CHECK(a.has_value());
        CHECK(b.has_value());
        if (a && b) compare(*a, *b);
        source.Release(uploaded_original);
        source.Release(uploaded_distorted);
        const auto last_a = source.Flush();
        const auto last_b = reference.Flush();
        CHECK(last_a.has_value());
        CHECK(last_b.has_value());
        if (last_a && last_b) compare(*last_a, *last_b);
        source.Reset();
        CHECK(!source.Push(original, distorted));
        CHECK(source.Flush().has_value());
    }
}

VQM_TEST(TestVmafV1SdrMixedFormatsAcrossFrames) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm::test::sdr;
    const auto original = MakeSource(
        PixelType::Byte, ColorFamily::Rgb, AlphaType::Straight, false
    );
    const auto distorted =
        MakeSource(PixelType::Byte, ColorFamily::Rgb, AlphaType::Straight, true);
    constexpr double background = 0.25;
    const auto yuv_original = ReferenceYuv(original, false, background);
    const auto yuv_distorted = ReferenceYuv(distorted, false, background);
    VmafV1Config mixed_config;
    mixed_config.size = vqm::test::sdr::kSize;
    mixed_config.input_format = original.format;
    mixed_config.chroma_sampling = VmafChromaSampling::Yuv444;
    mixed_config.background = static_cast<float>(background);
    VmafV1Config reference_config;
    reference_config.size = vqm::test::sdr::kSize;
    reference_config.input_format = yuv_original.format;
    VmafV1Pipeline mixed(mixed_config);
    VmafV1Pipeline reference(reference_config);
    // The first pair mixes RGB and YUV; the second swaps which image is RGB.
    // Their canonical decoded frames are the same as both reference pairs.
    CHECK(!mixed.Push(original, yuv_distorted));
    CHECK(!reference.Push(yuv_original, yuv_distorted));
    const auto actual = mixed.Push(yuv_original, distorted);
    const auto expected = reference.Push(yuv_original, yuv_distorted);
    CHECK(actual.has_value());
    CHECK(expected.has_value());
    if (actual && expected) {
        CHECK_NEAR(actual->score, expected->score, 0.1);
        CHECK_NEAR(actual->adm3, expected->adm3, 0.001);
        CHECK_NEAR(actual->speed_chroma_uv, expected->speed_chroma_uv, 0.05);
    }
    const auto last_actual = mixed.Flush();
    const auto last_expected = reference.Flush();
    CHECK(last_actual.has_value());
    CHECK(last_expected.has_value());
    if (last_actual && last_expected) {
        CHECK_NEAR(last_actual->score, last_expected->score, 0.1);
        CHECK_NEAR(last_actual->motion3, last_expected->motion3, 0.01);
    }
}

VQM_TEST(TestVmafV1SdrOddGeometry420) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm::test::sdr;
    constexpr Size odd{321, 257};
    const auto original = MakeSource(
        PixelType::Float, ColorFamily::Rgb, AlphaType::None, false, odd
    );
    const auto distorted = MakeSource(
        PixelType::Float, ColorFamily::Rgb, AlphaType::None, true, odd
    );
    VmafV1Config config;
    config.size = odd;
    config.input_format = original.format;
    config.chroma_sampling = VmafChromaSampling::Yuv420;
    VmafV1Pipeline pipeline(config);
    CHECK(!pipeline.Push(original, original));
    const auto identical = pipeline.Flush();
    CHECK(identical.has_value());
    pipeline.Reset();
    CHECK(!pipeline.Push(original, distorted));
    const auto changed = pipeline.Flush();
    CHECK(changed.has_value());
    if (identical && changed) {
        CHECK(std::isfinite(identical->score));
        CHECK(std::isfinite(changed->score));
        CHECK(std::fabs(identical->score - changed->score) > 1e-5);
    }
}

VQM_TEST(TestVmafV1StagedSdrConversion) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm::test::sdr;
    const auto original =
        MakeSource(PixelType::Float, ColorFamily::Rgb, AlphaType::None, false);
    const auto distorted =
        MakeSource(PixelType::Float, ColorFamily::Rgb, AlphaType::None, true);
    VmafV1Config config;
    config.size = original.size;
    config.input_format = original.format;
    config.chroma_sampling = VmafChromaSampling::Yuv420;
    const auto plan = GetMemoryRequirements(config);
    if (!plan.direct.supported || !plan.staged.supported) {
        vqm::test::MarkSkip(
            "direct and staged memory policies are not both supported"
        );
        return;
    }
    config.memory_policy = MemoryPolicy::Direct;
    VmafV1Pipeline direct(config);
    config.memory_policy = MemoryPolicy::Staged;
    VmafV1Pipeline staged(config);
    CHECK(!direct.GetMemoryUsage().uses_staging);
    CHECK(staged.GetMemoryUsage().uses_staging);
    CHECK(!direct.Push(original, distorted));
    CHECK(!staged.Push(original, distorted));
    const auto a = direct.Flush();
    const auto b = staged.Flush();
    CHECK(a.has_value());
    CHECK(b.has_value());
    if (a && b) {
        CHECK_NEAR(a->cambi, b->cambi, 1e-6);
        CHECK_NEAR(a->speed_chroma_uv, b->speed_chroma_uv, 1e-6);
        CHECK_NEAR(a->adm3, b->adm3, 1e-6);
        CHECK_NEAR(a->score, b->score, 1e-6);
    }
}
