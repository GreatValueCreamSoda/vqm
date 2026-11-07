// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// VMAF: Copyright (c) Netflix, Inc. BSD-2-Clause-Patent;
// see licenses/NETFLIX-VMAF-LICENSE.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>

#include "support.h"
#include "vmaf.h"
#include "metrics/vmaf/v0/vmaf_model.h"
#include "metrics/vmaf/features/vmaf_preprocess.h"
#include "vmaf_sdr_fixtures.h"
#include "vqm_memory.h"
#include "vqm_views.h"

namespace {
using namespace vqm;
constexpr Size kSize{64, 48};
constexpr std::array kScores{70.312588648518869, 98.243758166617312,
                             91.517291991577622, 96.32054440928529};

std::pair<Image, Image> MakeFrame(std::size_t frame) {
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
            const int base = static_cast<int>(
                16 +
                ((7 * x + 13 * y + 17 * frame + 29 * ((x / 7 + frame) % 2)) %
                 220)
            );
            const int noise =
                static_cast<int>((41 * x + 37 * y + 17 * frame) % 81) - 40;
            const auto ref = static_cast<std::uint16_t>(base * 4);
            const auto dist =
                static_cast<std::uint16_t>(std::clamp(base + noise, 0, 255) * 4);
            const auto offset = 2 * (y * kSize.width + x);
            std::memcpy(original.data.data() + offset, &ref, 2);
            std::memcpy(distorted.data.data() + offset, &dist, 2);
        }
    }
    return {std::move(original), std::move(distorted)};
}

ImageView View(const Image& image) {
    ImageView view{image.size, image.format, {}};
    const auto layout = internal::PlaneLayout(image.format, image.size);
    for (std::size_t plane = 0; plane < layout.size(); ++plane) {
        if (!layout[plane].present()) continue;
        view.planes[plane] = {
            reinterpret_cast<const std::byte*>(image.data.data()) +
                layout[plane].offset,
            layout[plane].width * layout[plane].sample_bytes};
    }
    return view;
}

Image MakeCodedYuv(bool bt2020) {
    ImageFormat format;
    format.type = PixelType::Byte;
    format.family = ColorFamily::Yuv;
    format.range = PixelRange::Limited;
    format.matrix =
        bt2020 ? MatrixCoefficients::Bt2020Ncl : MatrixCoefficients::Bt709;
    format.transfer = TransferCharacteristics::Bt709;
    Image image{kSize, format, {}};
    image.data.resize(RequiredBytes(format, kSize));
    const auto planes = internal::PlaneLayout(format, kSize);
    for (std::size_t y = 0; y < kSize.height; ++y) {
        for (std::size_t x = 0; x < kSize.width; ++x) {
            const auto i = y * kSize.width + x;
            image.data[planes[0].offset + i] =
                static_cast<std::uint8_t>(32 + (3 * x + 5 * y) % 184);
            image.data[planes[1].offset + i] =
                static_cast<std::uint8_t>(48 + (7 * x + 11 * y) % 160);
            image.data[planes[2].offset + i] =
                static_cast<std::uint8_t>(48 + (13 * x + 17 * y) % 160);
        }
    }
    return image;
}

void CheckFrame(const std::optional<VmafFrame>& frame, std::size_t index) {
    CHECK(frame.has_value());
    if (frame) {
        CHECK(frame->index == index);
        CHECK_NEAR(frame->score, kScores[index], 0.01);
    }
}
}  // namespace

VQM_TEST(TestVmafPublicSequenceAndResidentInputs) {
    if (!vqm::test::GpuAvailable()) return;
    VmafPipeline pipeline({.size = kSize, .gpu_resident_images = 2});
    CHECK_THROWS_AS(pipeline.PooledScore(), std::logic_error);
    for (std::size_t i = 0; i < 4; ++i) {
        const auto [a, b] = MakeFrame(i);
        std::optional<VmafFrame> result;
        if (i == 0)
            result = pipeline.Push(a, b);
        else if (i == 1)
            result = pipeline.Push(View(a), View(b));
        else {
            const auto da = pipeline.Upload(a);
            const auto db = pipeline.Upload(b);
            result = pipeline.Push(da, db);
            pipeline.Release(da);
            pipeline.Release(db);
            CHECK_THROWS_AS(pipeline.Push(da, db), std::invalid_argument);
        }
        if (i == 1)
            CHECK(!result.has_value());
        else
            CheckFrame(result, i == 0 ? 0 : i - 1);
    }
    CheckFrame(pipeline.Flush(), 3);
    CHECK(!pipeline.Flush().has_value());
    const auto pooled = pipeline.PooledScore();
    CHECK(pooled.has_value());
    if (pooled)
        CHECK_NEAR(
            *pooled, (kScores[0] + kScores[1] + kScores[2] + kScores[3]) / 4,
            0.01
        );
    const auto [a, b] = MakeFrame(0);
    CHECK_THROWS_AS(pipeline.Push(a, b), std::logic_error);
    pipeline.Reset();
    CheckFrame(pipeline.Push(a, b), 0);
    CHECK(!pipeline.Flush().has_value());
    CHECK_NEAR(*pipeline.PooledScore(), kScores[0], 0.01);
}

VQM_TEST(TestVmafStagedMemoryPreservesTemporalSequence) {
    if (!vqm::test::GpuAvailable()) {
        vqm::test::MarkSkip("no usable Vulkan device", true);
        return;
    }
    VmafConfig config{.size = kSize};
    config.memory_policy = MemoryPolicy::Staged;
    if (!GetMemoryRequirements(config).staged.supported) {
        vqm::test::MarkSkip("no compatible staged memory type");
        return;
    }
    VmafPipeline pipeline(config);
    CHECK(pipeline.GetMemoryUsage().uses_staging);
    for (std::size_t i = 0; i < 4; ++i) {
        const auto [a, b] = MakeFrame(i);
        const auto frame = pipeline.Push(a, b);
        if (i == 1)
            CHECK(!frame.has_value());
        else
            CheckFrame(frame, i == 0 ? 0 : i - 1);
    }
    CheckFrame(pipeline.Flush(), 3);
}

VQM_TEST(TestVmafPublicValidation) {
    if (!vqm::test::GpuAvailable()) return;
    CHECK_THROWS_AS(VmafPipeline({.size = {16, 17}}), std::invalid_argument);
    VmafPipeline pipeline(
        {.size = kSize, .gpu_resident_images = 0, .streaming_inputs = false}
    );
    const auto [a, b] = MakeFrame(0);
    CHECK_THROWS_AS(pipeline.Push(a, b), std::invalid_argument);
    CHECK_THROWS_AS(pipeline.Upload(a), std::invalid_argument);
    CHECK(!pipeline.Flush());
    CHECK(!pipeline.PooledScore());
}

VQM_TEST(TestVmafV0SdrRgbInputConversion) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm::test::sdr;
    for (const auto type : {PixelType::Byte, PixelType::Float}) {
        const auto alpha = type == PixelType::Byte ? AlphaType::Straight
                                                   : AlphaType::Premultiplied;
        const auto original = MakeSource(type, ColorFamily::Rgb, alpha, false);
        const auto distorted = MakeSource(type, ColorFamily::Rgb, alpha, true);
        const auto ref_original = ReferenceYuv(original, false);
        const auto ref_distorted = ReferenceYuv(distorted, false);
        VmafPipeline source({.size = vqm::test::sdr::kSize});
        VmafPipeline reference({.size = vqm::test::sdr::kSize});
        const auto actual = source.Push(original, distorted);
        const auto expected = reference.Push(ref_original, ref_distorted);
        CHECK(actual.has_value());
        CHECK(expected.has_value());
        if (actual && expected) {
            CHECK(actual->index == expected->index);
            CHECK_NEAR(actual->score, expected->score, 0.1);
            CHECK_NEAR(actual->adm2, expected->adm2, 0.001);
            for (std::size_t i = 0; i < actual->vif.size(); ++i)
                CHECK_NEAR(actual->vif[i], expected->vif[i], 0.001);
        }
    }
}

VQM_TEST(TestVmafV0NativeSequenceCodingRecovery) {
    if (!vqm::test::GpuAvailable()) return;
    const auto [original, distorted] = MakeFrame(0);
    const auto rgb_original = vqm::test::sdr::MakeSource(
        PixelType::Float, ColorFamily::Rgb, AlphaType::None, false, kSize
    );
    const auto rgb_distorted = vqm::test::sdr::MakeSource(
        PixelType::Float, ColorFamily::Rgb, AlphaType::None, true, kSize
    );
    VmafPipeline pipeline({.size = kSize});
    VmafPipeline reference({.size = kSize});
    const auto first = pipeline.Push(original, distorted);
    const auto first_ref = reference.Push(original, distorted);
    CHECK(first.has_value());
    CHECK(first_ref.has_value());
    CHECK_THROWS_AS(
        pipeline.Push(rgb_original, rgb_distorted), std::invalid_argument
    );
    // The rejected pair must not advance the sequence or leave a command
    // pending: a compatible retry should match an uninterrupted pipeline.
    CHECK(!pipeline.Push(original, distorted));
    CHECK(!reference.Push(original, distorted));
    const auto next = pipeline.Push(original, distorted);
    const auto next_ref = reference.Push(original, distorted);
    CHECK(next.has_value());
    CHECK(next_ref.has_value());
    if (next && next_ref) {
        CHECK(next->index == next_ref->index);
        CHECK_NEAR(next->score, next_ref->score, 1e-8);
    }
    const auto last = pipeline.Flush();
    const auto last_ref = reference.Flush();
    CHECK(last.has_value());
    CHECK(last_ref.has_value());
    if (last && last_ref) CHECK_NEAR(last->score, last_ref->score, 1e-8);
    pipeline.Reset();
    // A new sequence may start in the canonical conversion domain; that
    // domain stays canonical when subsequent input happens to be coded YUV.
    const auto rgb_first = pipeline.Push(rgb_original, rgb_distorted);
    CHECK(rgb_first.has_value());
    CHECK(!pipeline.Push(original, distorted));
    CHECK(pipeline.Flush().has_value());
}

VQM_TEST(TestVmafV0MixedCodingStreamingMatchesResident) {
    if (!vqm::test::GpuAvailable()) return;
    const auto yuv709 = MakeCodedYuv(false);
    const auto yuv2020 = MakeCodedYuv(true);
    const auto rgb = vqm::test::sdr::MakeSource(
        PixelType::Float, ColorFamily::Rgb, AlphaType::None, false, kSize
    );
    const std::array pairs{std::pair{yuv709, yuv2020}, std::pair{yuv709, yuv709},
                           std::pair{rgb, yuv709}};
    const auto compare = [](const auto& actual, const auto& expected) {
        CHECK(actual.has_value() == expected.has_value());
        if (!actual || !expected) return;
        CHECK(actual->index == expected->index);
        CHECK_NEAR(actual->score, expected->score, 1e-9);
        CHECK_NEAR(actual->adm2, expected->adm2, 1e-9);
        CHECK_NEAR(actual->motion2, expected->motion2, 1e-9);
        for (std::size_t scale = 0; scale < actual->vif.size(); ++scale)
            CHECK_NEAR(actual->vif[scale], expected->vif[scale], 1e-9);
    };
    for (const auto policy : {MemoryPolicy::Automatic, MemoryPolicy::Staged}) {
        VmafConfig config{.size = kSize, .memory_policy = policy};
        if (policy == MemoryPolicy::Staged &&
            !GetMemoryRequirements(config).staged.supported)
            continue;
        for (const bool use_view : {false, true}) {
            VmafPipeline streaming(config), resident(config);
            for (const auto& [a, b] : pairs) {
                // The mixed first pair chooses conversion; later coded YUV
                // must keep uploading chroma in that same sequence domain.
                const auto da = resident.Upload(a);
                const auto db = resident.Upload(b);
                compare(
                    use_view ? streaming.Push(View(a), View(b))
                             : streaming.Push(a, b),
                    resident.Push(da, db)
                );
                resident.Release(da);
                resident.Release(db);
            }
            compare(streaming.Flush(), resident.Flush());
            CHECK_NEAR(*streaming.PooledScore(), *resident.PooledScore(), 1e-9);
        }
    }
}

VQM_TEST(TestVmafFitGeometry) {
    using vqm::internal::FitVmafInto;
    const auto check = [](Size source, Size target, Size content,
                          std::uint32_t x, std::uint32_t y) {
        const auto geometry = FitVmafInto(source, target);
        CHECK(geometry.content.width == content.width);
        CHECK(geometry.content.height == content.height);
        CHECK(geometry.x == x);
        CHECK(geometry.y == y);
    };
    check({1920, 1080}, {1920, 1080}, {1920, 1080}, 0, 0);
    check({1440, 1080}, {1920, 1080}, {1440, 1080}, 240, 0);
    check({3840, 1600}, {1920, 1080}, {1920, 800}, 0, 140);
    check({720, 1280}, {1920, 1080}, {608, 1080}, 656, 0);
    check({854, 480}, {1920, 1080}, {1920, 1079}, 0, 0);
    check({64, 48}, {128, 128}, {128, 96}, 0, 16);
    check({100, 100}, {128, 128}, {128, 128}, 0, 0);
}

VQM_TEST(TestVmafFitAndPadMatchesPaddedCanvas) {
    if (!vqm::test::GpuAvailable()) return;
    // VMAF's perfect-feature prediction is 97.428..., not 100. Compare the
    // fit path with a native pipeline fed the equivalent pre-padded canvas.
    ImageFormat format;
    format.type = PixelType::Word;
    format.depth = 10;
    format.family = ColorFamily::Grey;
    format.matrix = MatrixCoefficients::Bt709;
    format.range = PixelRange::Limited;
    const auto make_constant = [&](Size size, std::uint16_t code) {
        Image image{size, format, {}};
        image.data.resize(RequiredBytes(format, size));
        for (std::size_t i = 0; i < size.width * size.height; ++i)
            std::memcpy(image.data.data() + 2 * i, &code, 2);
        return image;
    };
    const auto make_canvas = [&](Size target, Size content, std::uint32_t x,
                                 std::uint32_t y) {
        Image image{target, format, {}};
        image.data.resize(RequiredBytes(format, target));
        constexpr std::uint16_t kBlack = 64;
        constexpr std::uint16_t kContent = 512;
        for (std::size_t row = 0; row < target.height; ++row) {
            for (std::size_t column = 0; column < target.width; ++column) {
                const bool inside = column >= x && column < x + content.width &&
                                    row >= y && row < y + content.height;
                const std::uint16_t code = inside ? kContent : kBlack;
                const std::size_t index = row * target.width + column;
                std::memcpy(image.data.data() + 2 * index, &code, 2);
            }
        }
        return image;
    };

    const auto source = make_constant(kSize, 512);
    struct Case {
        Size target;
        Size content;
        std::uint32_t x;
        std::uint32_t y;
    };
    constexpr std::array cases = {
        Case{{128, 96}, {128, 96}, 0, 0}, Case{{128, 128}, {128, 96}, 0, 16},
        Case{{128, 72}, {96, 72}, 16, 0}, Case{{64, 48}, {64, 48}, 0, 0}};
    for (const Case& test : cases) {
        const Image expected_source =
            make_canvas(test.target, test.content, test.x, test.y);
        VmafPipeline expected({.size = test.target});
        VmafPipeline fit(
            {.size = kSize,
             .fit = VmafFitMode::FitAndPad,
             .model_size = test.target}
        );
        const auto expected_frame =
            expected.Push(expected_source, expected_source);
        const auto actual_frame = fit.Push(source, source);
        CHECK(expected_frame.has_value());
        CHECK(actual_frame.has_value());
        if (expected_frame && actual_frame)
            CHECK_NEAR(actual_frame->score, expected_frame->score, 0.01);
    }
}

VQM_TEST(TestVmafFitAndPadAtNativeSizeMatchesNative) {
    if (!vqm::test::GpuAvailable()) return;
    const auto [original, distorted] = MakeFrame(0);
    VmafPipeline native({.size = kSize});
    VmafPipeline fit(
        {.size = kSize, .fit = VmafFitMode::FitAndPad, .model_size = kSize}
    );
    const auto expected = native.Push(original, distorted);
    const auto actual = fit.Push(original, distorted);
    CHECK(actual.has_value());
    CHECK(expected.has_value());
    if (actual && expected) {
        CHECK_NEAR(actual->score, expected->score, 1e-8);
        CHECK_NEAR(actual->adm2, expected->adm2, 1e-8);
        for (std::size_t i = 0; i < actual->vif.size(); ++i)
            CHECK_NEAR(actual->vif[i], expected->vif[i], 1e-8);
    }
}

VQM_TEST(TestVmafFitAndPadResidentMatchesStreaming) {
    if (!vqm::test::GpuAvailable()) return;
    const auto [original, distorted] = MakeFrame(0);
    const VmafConfig config{
        .size = kSize, .fit = VmafFitMode::FitAndPad, .model_size = {128, 128}};
    VmafPipeline streaming(config);
    VmafPipeline resident(config);
    const auto expected = streaming.Push(original, distorted);
    const auto ref = resident.Upload(original);
    const auto dist = resident.Upload(distorted);
    const auto actual = resident.Push(ref, dist);
    CHECK(actual.has_value());
    CHECK(expected.has_value());
    if (actual && expected) CHECK_NEAR(actual->score, expected->score, 1e-8);
    resident.Release(ref);
    resident.Release(dist);
}

VQM_TEST(TestVmafFitAndPadValidation) {
    if (!vqm::test::GpuAvailable()) return;
    CHECK_THROWS_AS(
        VmafPipeline(
            {.size = kSize,
             .fit = VmafFitMode::FitAndPad,
             .model_size = {16, 1080}}
        ),
        std::invalid_argument
    );
}

VQM_TEST(TestVmafPipelineSelects4KModel) {
    if (!vqm::test::GpuAvailable()) return;
    const auto [original, distorted] = MakeFrame(0);
    VmafPipeline pipeline({.size = kSize, .model = VmafModel::Standard4K});
    const auto frame = pipeline.Push(original, distorted);
    CHECK(frame.has_value());
    if (!frame) return;
    const vqm::internal::VmafV061Features features{frame->adm2, frame->motion2,
                                                   frame->vif};
    CHECK_NEAR(
        frame->score,
        vqm::internal::VmafV061Predict(
            features, false, true, VmafModel::Standard4K
        ),
        1e-9
    );
}

VQM_TEST(TestVmafNegChangesFeatures) {
    if (!vqm::test::GpuAvailable()) return;
    auto [original, distorted] = MakeFrame(0);
    for (std::size_t i = 0; i < original.size.width * original.size.height;
         ++i) {
        std::uint16_t reference = 0;
        std::memcpy(&reference, original.data.data() + 2 * i, 2);
        const auto base = static_cast<int>(reference / 4);
        const int enhanced = std::clamp(128 + 3 * (base - 128) / 2, 16, 235);
        const auto sample = static_cast<std::uint16_t>(enhanced * 4);
        std::memcpy(distorted.data.data() + 2 * i, &sample, 2);
    }
    VmafPipeline normal({.size = kSize});
    VmafPipeline neg({.size = kSize, .neg = true});
    const auto normal_frame = normal.Push(original, distorted);
    const auto neg_frame = neg.Push(original, distorted);
    CHECK(normal_frame.has_value());
    CHECK(neg_frame.has_value());
    if (normal_frame && neg_frame) {
        CHECK(std::fabs(normal_frame->adm2 - neg_frame->adm2) > 1e-6);
        CHECK(std::fabs(normal_frame->vif[0] - neg_frame->vif[0]) > 1e-6);
        CHECK(std::fabs(normal_frame->score - neg_frame->score) > 1e-6);
    }
}

VQM_TEST(TestVmafPipelineWithoutTemporalWeighting) {
    if (!vqm::test::GpuAvailable()) return;
    const auto [original, distorted] = MakeFrame(0);
    VmafPipeline pipeline({.size = kSize, .temporal_weighting = false});
    const auto frame = pipeline.Push(original, distorted);
    CHECK(frame.has_value());
    if (!frame) return;
    CHECK(frame->index == 0);
    CHECK(frame->motion2 == 0.0);
    const vqm::internal::VmafV061Features features{frame->adm2, 0.0, frame->vif};
    CHECK_NEAR(frame->score, vqm::internal::VmafV061Predict(features), 1e-9);
    CHECK(!pipeline.Flush());
    CHECK_NEAR(*pipeline.PooledScore(), frame->score, 1e-9);
}
