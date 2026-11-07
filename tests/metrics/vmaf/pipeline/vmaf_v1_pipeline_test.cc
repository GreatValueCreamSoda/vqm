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
#include "temporal_test_support.h"
#include "vmaf_v1.h"
#include "vmaf_sdr_fixtures.h"
#include "vqm_memory.h"

namespace {
using namespace vqm;
constexpr Size kV1Size{512, 384};
constexpr Size kViewingSize{640, 480};
constexpr Size kModelCanvas{1920, 1080};
struct V1Oracle {
    double cambi, speed, adm3, motion3, score;
};
#include "vmaf_v1_reference.inc"
#include "vmaf_v1_hdr_reference.inc"

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
Image MakeFrame(
    const ImageFormat& format, std::size_t frame, bool distorted,
    Size size = kV1Size
) {
    Image result{size, format, {}};
    result.data.resize(RequiredBytes(format, size));
    const auto planes = vqm::internal::PlaneLayout(format, size);
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

std::size_t HdrSample(
    unsigned plane, std::size_t x, std::size_t y, std::size_t width,
    std::size_t frame, bool distorted
) {
    if (plane == 0) {
        auto value =
            125 + (x + y / 3 + 63 * frame) % (width / 2) * 480 / (width / 2);
        if (distorted)
            value = (value + 2) / 5 * 5 +
                    ((x / 47 + y / 53 + frame) % 5 == 0 ? 1u : 0u);
        return value;
    }
    auto value =
        442 + (x / 5 * (5 + plane) + y / 7 * (7 - plane) + frame * 3) % 141;
    if (distorted)
        value = value - 10 + (x / 9 * 3 + y / 13 * 7 + frame * 7) % 21;
    return value;
}

Image MakeHdrFrame(
    const ImageFormat& format, Size size, std::size_t frame, bool distorted
) {
    Image result{size, format, {}};
    result.data.resize(RequiredBytes(format, size));
    const auto planes = vqm::internal::PlaneLayout(format, size);
    for (unsigned plane = 0; plane < 3; ++plane) {
        const auto& layout = planes[plane];
        for (std::size_t y = 0; y < layout.height; ++y) {
            for (std::size_t x = 0; x < layout.width; ++x) {
                auto value =
                    HdrSample(plane, x, y, layout.width, frame, distorted);
                if (format.depth == 16)
                    value = value * 64 + (x + 2 * y + frame) % 64;
                const auto code = static_cast<std::uint16_t>(value);
                const auto offset = layout.offset + 2 * (y * layout.width + x);
                std::memcpy(result.data.data() + offset, &code, sizeof(code));
            }
        }
    }
    return result;
}

Image PillarboxV1Frame(const Image& source) {
    Image result{kModelCanvas, source.format, {}};
    result.data.resize(RequiredBytes(source.format, kModelCanvas));
    const auto source_planes =
        vqm::internal::PlaneLayout(source.format, source.size);
    const auto target_planes =
        vqm::internal::PlaneLayout(source.format, kModelCanvas);
    for (std::size_t plane = 0; plane < 3; ++plane) {
        const auto& input = source_planes[plane];
        const auto& output = target_planes[plane];
        const std::vector<std::uint16_t> bar(
            plane == 0 ? 240u : 120u, plane == 0 ? 64u : 512u
        );
        const auto bar_bytes = bar.size() * sizeof(bar[0]);
        const auto input_bytes = input.width * sizeof(bar[0]);
        const auto output_bytes = output.width * sizeof(bar[0]);
        for (std::size_t y = 0; y < input.height; ++y) {
            const auto* from =
                source.data.data() + input.offset + y * input_bytes;
            auto* to = result.data.data() + output.offset + y * output_bytes;
            std::memcpy(to, bar.data(), bar_bytes);
            std::memcpy(to + bar_bytes, from, input_bytes);
            std::memcpy(to + bar_bytes + input_bytes, bar.data(), bar_bytes);
        }
    }
    return result;
}
void CheckFrame(
    const VmafV1Frame& frame, const V1Oracle& expected, std::size_t index,
    double speed_tolerance
) {
    CHECK(frame.index == index);
    CHECK_NEAR(frame.cambi, expected.cambi, 0.04);
    CHECK_NEAR(frame.speed_chroma_uv, expected.speed, speed_tolerance);
    CHECK_NEAR(frame.adm3, expected.adm3, 0.005);
    CHECK_NEAR(frame.motion3, expected.motion3, 0.04);
    CHECK_NEAR(frame.score, expected.score, 0.35);
}
template <std::size_t N>
void RunHdrVideo(
    Size size, MatrixCoefficients matrix, bool hfr,
    const std::array<V1Oracle, N>& expected
) {
    const ImageFormat format{
        .type = PixelType::Word,
        .family = ColorFamily::Yuv,
        .range = PixelRange::Limited,
        .matrix = matrix,
        .transfer = TransferCharacteristics::St2084,
        .primaries = ColorPrimaries::Bt2020,
        .subsample_w = 1,
        .subsample_h = 1,
        .depth = static_cast<std::uint8_t>(
            matrix == MatrixCoefficients::Bt2100ICtCp ? 16 : 10
        ),
    };
    VmafV1Pipeline pipeline(
        {
            .size = size,
            .input_format = format,
            .gpu_resident_images = 0,
            .model = VmafV1Model::Hdr4K,
            .hfr = hfr,
        }
    );
    std::size_t received = 0;
    double score_sum = 0.0;
    const auto check = [&](const VmafV1Frame& frame) {
        CHECK(received < N);
        if (received >= N) return;
        const auto& oracle = expected[received];
        CheckFrame(frame, oracle, received++, 0.025);
        CHECK_NEAR(frame.score, oracle.score, 0.05);
        score_sum += oracle.score;
    };
    for (std::size_t i = 0; i < N; ++i) {
        auto result = pipeline.Push(
            MakeHdrFrame(format, size, i, false),
            MakeHdrFrame(format, size, i, true)
        );
        if (result) check(*result);
    }
    while (const auto frame = pipeline.Flush()) check(*frame);
    CHECK(received == N);
    const auto pooled = pipeline.PooledScore();
    CHECK(pooled.has_value());
    if (pooled) CHECK_NEAR(*pooled, score_sum / static_cast<double>(N), 0.05);
}

template <std::size_t N>
void RunVideo(
    VmafV1Pipeline& pipeline, const ImageFormat& format,
    const std::array<V1Oracle, N>& expected, bool resident, Size size = kV1Size,
    double speed_tolerance = 0.08
) {
    double expected_sum = 0.0;
    std::size_t received = 0;
    for (std::size_t frame = 0; frame < N; ++frame) {
        auto a = MakeFrame(format, frame, false, size);
        auto b = MakeFrame(format, frame, true, size);
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
            if (received < N)
                CheckFrame(
                    *result, expected[received], received, speed_tolerance
                );
            ++received;
        }
        expected_sum += expected[frame].score;
    }
    while (const auto result = pipeline.Flush()) {
        CHECK(received < N);
        if (received < N)
            CheckFrame(*result, expected[received], received, speed_tolerance);
        ++received;
    }
    CHECK(received == N);
    const auto pooled = pipeline.PooledScore();
    CHECK(pooled.has_value());
    if (pooled) CHECK_NEAR(*pooled, expected_sum / static_cast<double>(N), 0.35);
}
}  // namespace

VQM_TEST(TestVmafV1HdrPipelineReferenceSequences) {
    if (!vqm::test::GpuAvailable()) return;
    RunHdrVideo(
        {1280, 720}, MatrixCoefficients::Bt2020Ncl, false, kOracle420HdrNclScaled
    );
    RunHdrVideo(
        {3840, 2160}, MatrixCoefficients::Bt2020Ncl, true, kOracle420HdrNclHfr
    );
    RunHdrVideo(
        {3840, 2160}, MatrixCoefficients::Bt2100ICtCp, false, kOracle420HdrIctcp
    );
}

VQM_TEST(TestVmafV1HfrPipelineReferenceSequences) {
    if (!vqm::test::GpuAvailable()) return;
    const ImageFormat format{
        .type = PixelType::Word,
        .family = ColorFamily::Yuv,
        .range = PixelRange::Limited,
        .matrix = MatrixCoefficients::Bt709,
        .subsample_w = 1,
        .subsample_h = 1,
        .depth = 10,
    };
    VmafV1Pipeline pipeline(
        {.size = kV1Size, .input_format = format, .hfr = true}
    );
    RunVideo(pipeline, format, kOracle420Hfr2, false);
    pipeline.Reset();
    RunVideo(pipeline, format, kOracle420Hfr3, true);
    pipeline.Reset();
    RunVideo(pipeline, format, kOracle420Hfr6, false);
}

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
    config.encoded_bit_depth = 0;
    config.size = kViewingSize;
    for (const auto& [model, oracle] :
         {std::pair{VmafV1Model::Standard4K, &kOracle444Close4K},
          std::pair{VmafV1Model::Tv4K, &kOracle444Tv4K},
          std::pair{VmafV1Model::Phone, &kOracle444Phone}}) {
        config.model = model;
        VmafV1Pipeline variant(config);
        // The 0.5x prescale amplifies covariance rounding on this periodic
        // fixture. Keep the same ADM, CAMBI, motion and final-score tolerances.
        const double speed_tolerance = model == VmafV1Model::Tv4K ? 0.16 : 0.08;
        RunVideo(variant, format, *oracle, false, config.size, speed_tolerance);
    }
    format.subsample_w = 1;
    format.subsample_h = 1;
    config.input_format = format;
    config.model = VmafV1Model::Standard1080;
    config.size = kV1Size;
    VmafV1Pipeline subsampled(config);
    RunVideo(subsampled, format, kOracle420Single, false);
    subsampled.Reset();
    RunVideo(subsampled, format, kOracle420Video, true);
}

VQM_TEST(TestVmafV1FitAndPadMatchesPadded420Canvas) {
    if (!vqm::test::GpuAvailable()) return;
    using vqm::test::temporal::Pair;
    using vqm::test::temporal::RunClip;
    // Same-height 4:3 input fits without interpolation. Its independently
    // padded reference has 240 luma and 120 chroma samples on each side.
    constexpr Size source_size{1440, 1080};
    ImageFormat format;
    format.type = PixelType::Word;
    format.depth = 10;
    format.family = ColorFamily::Yuv;
    format.matrix = MatrixCoefficients::Bt709;
    format.range = PixelRange::Limited;
    format.subsample_w = 1;
    format.subsample_h = 1;
    const std::array pairs{Pair{MakeFrame(format, 0, false, source_size),
                                MakeFrame(format, 0, true, source_size)},
                           Pair{MakeFrame(format, 1, false, source_size),
                                MakeFrame(format, 1, true, source_size)}};
    const std::array canvases{Pair{PillarboxV1Frame(pairs[0].original),
                                   PillarboxV1Frame(pairs[0].distorted)},
                              Pair{PillarboxV1Frame(pairs[1].original),
                                   PillarboxV1Frame(pairs[1].distorted)}};
    VmafV1Config config;
    config.size = source_size;
    config.input_format = format;
    config.gpu_resident_images = 0;
    config.fit = VmafFitMode::FitAndPad;
    VmafV1Pipeline fitted(config);
    config.size = kModelCanvas;
    config.fit = VmafFitMode::Native;
    VmafV1Pipeline native(config);
    const auto actual = RunClip<VmafV1Frame>(fitted, pairs);
    const auto expected = RunClip<VmafV1Frame>(native, canvases);
    CHECK(actual.size() == pairs.size());
    CHECK(expected.size() == canvases.size());
    if (actual.size() != pairs.size() || expected.size() != canvases.size())
        return;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        CHECK(actual[i].index == expected[i].index);
        CHECK_NEAR(actual[i].cambi, expected[i].cambi, 1e-6);
        CHECK_NEAR(actual[i].speed_chroma_uv, expected[i].speed_chroma_uv, 1e-6);
        CHECK_NEAR(actual[i].adm3, expected[i].adm3, 1e-6);
        CHECK_NEAR(actual[i].motion3, expected[i].motion3, 1e-6);
        CHECK_NEAR(actual[i].score, expected[i].score, 1e-6);
    }
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
