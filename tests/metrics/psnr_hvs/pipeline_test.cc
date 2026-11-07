// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
// CSF constants: Copyright 2001-2012 Xiph.Org and contributors.
// BSD-2-Clause; see licenses/XIPH-PSNR-HVS-LICENSE.

#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "support.h"
#include "psnr_hvs.h"
#include "vqm_memory.h"
#include "vqm_metrics.h"
#include "vqm_views.h"

namespace {
using namespace vqm;

Image ConstantYuv(
    Size size, std::uint8_t depth, std::uint8_t horizontal,
    std::uint8_t vertical, const std::array<std::uint16_t, 3>& codes
) {
    Image image;
    image.size = size;
    image.format.family = ColorFamily::Yuv;
    image.format.matrix = MatrixCoefficients::Bt709;
    image.format.transfer = TransferCharacteristics::Bt709;
    image.format.type = depth == 8 ? PixelType::Byte : PixelType::Word;
    image.format.depth = depth;
    image.format.subsample_w = horizontal;
    image.format.subsample_h = vertical;
    image.data.resize(RequiredBytes(image.format, size));
    const auto planes = internal::PlaneLayout(image.format, size);
    const std::size_t bytes = depth == 8 ? 1u : 2u;
    for (std::size_t plane = 0; plane < 3; ++plane) {
        const auto byte_code = static_cast<std::uint8_t>(codes[plane]);
        const void* sample_data = bytes == 1
                                      ? static_cast<const void*>(&byte_code)
                                      : static_cast<const void*>(&codes[plane]);
        for (std::size_t sample = 0;
             sample < planes[plane].width * planes[plane].height; ++sample) {
            std::memcpy(
                image.data.data() + planes[plane].offset + sample * bytes,
                sample_data, bytes
            );
        }
    }
    return image;
}

ImageView View(const Image& image) {
    ImageView view{image.size, image.format};
    const auto planes = internal::PlaneLayout(image.format, image.size);
    const std::size_t bytes = image.format.type == PixelType::Byte ? 1u : 2u;
    for (std::size_t plane = 0; plane < 3; ++plane) {
        view.planes[plane] = {reinterpret_cast<const std::byte*>(
                                  image.data.data() + planes[plane].offset
                              ),
                              planes[plane].width * bytes};
    }
    return view;
}

void CheckConstantError(const PsnrHvsScore& score, std::uint8_t depth) {
    // Independent closed form: a constant offset d changes only the DC
    // coefficient, by 8*d in an orthonormal 8x8 DCT. Dividing squared block
    // error by 64 leaves (d*CSF[0])^2. Masking never changes the DC term.
    // CSF DC weights: pinned libvmaf 4611e47, xiph/psnr_hvs.c:163-201.
    constexpr std::array weights{1.6193873005, 1.91113096927, 2.03871978502};
    constexpr std::array deltas{3.0, 5.0, 7.0};
    const double peak = static_cast<double>((1u << depth) - 1u);
    double combined = 0.0;
    const std::array decibels{score.y, score.cb, score.cr};
    for (std::size_t plane = 0; plane < 3; ++plane) {
        const double weighted = deltas[plane] * weights[plane] / peak;
        const double error = weighted * weighted;
        CHECK_NEAR(score.normalized_error[plane], error, error * 2e-5);
        CHECK_NEAR(decibels[plane], -10.0 * std::log10(error), 1e-4);
        combined += (plane == 0 ? 0.8 : 0.1) * error;
    }
    CHECK_NEAR(score.score, -10.0 * std::log10(combined), 1e-4);
}

void CheckIdentity(const PsnrHvsScore& score) {
    CHECK(std::isinf(score.score) && score.score > 0.0);
    CHECK(std::isinf(score.y) && score.y > 0.0);
    CHECK(std::isinf(score.cb) && score.cb > 0.0);
    CHECK(std::isinf(score.cr) && score.cr > 0.0);
    CHECK((score.normalized_error == std::array<double, 3>{}));
}

void CheckConstantFormats(std::uint8_t depth) {
    for (const auto sampling : {Size{0, 0}, Size{1, 0}, Size{1, 1}}) {
        const auto original = ConstantYuv(
            {32, 32}, depth, static_cast<std::uint8_t>(sampling.width),
            static_cast<std::uint8_t>(sampling.height), {64, 96, 128}
        );
        auto distorted = ConstantYuv(
            original.size, depth, original.format.subsample_w,
            original.format.subsample_h, {67, 101, 135}
        );
        PsnrHvsPipeline pipeline({.size = original.size});
        CheckIdentity(pipeline.Compute(original, original));
        CheckConstantError(pipeline.Compute(original, distorted), depth);
        CheckConstantError(
            pipeline.Compute(View(original), View(distorted)), depth
        );
        const auto resident = pipeline.Upload(original);
        const auto comparison = pipeline.Upload(distorted);
        CheckConstantError(pipeline.Compute(resident, comparison), depth);
        CheckConstantError(pipeline.Compute(resident, distorted), depth);
        pipeline.Release(comparison);
        pipeline.Release(resident);
    }
}

}  // namespace

VQM_TEST(TestPsnrHvsConstantNumericalOracle) {
    if (!test::GpuAvailable()) return;
    for (const auto depth : std::array<std::uint8_t, 3>{8, 10, 12})
        CheckConstantFormats(depth);
}

VQM_TEST(TestPsnrHvsTwoStageReductionAndStaging) {
    if (!test::GpuAvailable()) return;
    // More than 512 luma blocks requires the second reduction dispatch.
    const auto original = ConstantYuv({256, 256}, 10, 1, 1, {64, 96, 128});
    const auto distorted = ConstantYuv({256, 256}, 10, 1, 1, {67, 101, 135});
    PsnrHvsPipeline pipeline(
        {.size = original.size, .memory_policy = MemoryPolicy::Staged}
    );
    CheckConstantError(pipeline.Compute(original, distorted), 10);
    CheckIdentity(pipeline.Compute(original, original));
    const auto a = pipeline.Upload(original);
    const auto b = pipeline.Upload(distorted);
    CheckConstantError(pipeline.Compute(a, b), 10);
    const auto requirements = GetMemoryRequirements(
        PsnrHvsConfig{.size = original.size,
                      .memory_policy = MemoryPolicy::Staged}
    );
    CHECK(
        pipeline.GetMemoryUsage().arena.buffer_bytes ==
        requirements.staged.arena.buffer_bytes
    );
    CHECK(
        pipeline.GetMemoryUsage().staging.buffer_bytes ==
        requirements.staged.staging.buffer_bytes
    );
}

VQM_TEST(TestPsnrHvsSingleFrequencyMaskingOracle) {
    if (!test::GpuAvailable()) return;
    const auto original = ConstantYuv({8, 8}, 8, 0, 0, {128, 128, 128});
    auto distorted = original;
    constexpr std::array signs{1, -1, -1, 1, 1, -1, -1, 1};
    const auto planes = internal::PlaneLayout(original.format, original.size);
    for (std::size_t plane = 0; plane < 3; ++plane) {
        for (std::size_t sample = 0; sample < 64; ++sample) {
            distorted.data[planes[plane].offset + sample] =
                static_cast<std::uint8_t>(128 + 32 * signs[sample / 8]);
        }
    }
    PsnrHvsPipeline pipeline({.size = original.size});
    const auto score = pipeline.Compute(original, distorted);
    // This pattern has one AC coefficient (4,0), equal to 8*amplitude.
    // Each quadrant and the full block have mean 128, giving a variance
    // ratio (16/15)/(64/63) = 1.05. Evaluate the reference masking formula
    // analytically, independently of production tables and GPU reductions.
    // CSF weights: libvmaf 4611e47, xiph/psnr_hvs.c:163-201.
    constexpr std::array weights{1.00227514334, 1.05017074788, 1.01397751469};
    double combined = 0.0;
    for (std::size_t plane = 0; plane < 3; ++plane) {
        const double threshold =
            32.0 * std::sqrt(1.05) / (4.0 * weights[plane] * 0.3885746225901003);
        const double error =
            (256.0 - threshold) * weights[plane] / (8.0 * 255.0);
        CHECK_NEAR(score.normalized_error[plane], error * error, 1e-7);
        combined += (plane == 0 ? 0.8 : 0.1) * error * error;
    }
    CHECK_NEAR(score.score, -10.0 * std::log10(combined), 1e-4);
}

VQM_TEST(TestPsnrHvsCombinedMetricsFrameAlignment) {
    if (!test::GpuAvailable()) return;
    const auto original = ConstantYuv({32, 32}, 8, 1, 1, {64, 96, 128});
    const auto distorted = ConstantYuv({32, 32}, 8, 1, 1, {67, 101, 135});
    PsnrHvsPipeline standalone({.size = original.size});
    const double expected = standalone.Compute(original, distorted).score;
    MetricsConfig config;
    config.size = original.size;
    config.original_format = original.format;
    config.distorted_format = distorted.format;
    config.psnr_hvs = PsnrHvsConfig{
        .size = {7, 8}, .input_bytes = 4096, .streaming_inputs = false};
    config.vmaf_v0 = VmafConfig{.size = original.size};
    Metrics metrics(config);
    // VMAF v0 emits its zero-motion frame 0 immediately. Frame 1 waits
    // for lookahead or Flush; its PSNR-HVS score must stay associated.
    const auto first = metrics.Compute(original, original);
    CHECK(first && first->frame_index == 0 && first->psnr_hvs);
    if (first && first->psnr_hvs) CHECK(std::isinf(*first->psnr_hvs));
    CHECK(!metrics.Compute(original, distorted));
    const auto last = metrics.Finalize();
    CHECK(last.frames == 2);
    CHECK(
        last.last_frame && last.last_frame->frame_index == 1 &&
        last.last_frame->psnr_hvs
    );
    if (last.last_frame && last.last_frame->psnr_hvs)
        CHECK_NEAR(*last.last_frame->psnr_hvs, expected, 1e-6);
    CHECK(!metrics.Finalize().last_frame);
    CHECK_THROWS_AS(metrics.Compute(original, distorted), std::logic_error);
    metrics.Reset();
    const auto single = metrics.ComputeImage(View(original), View(distorted));
    CHECK(single.psnr_hvs);
    if (single.psnr_hvs) CHECK_NEAR(*single.psnr_hvs, expected, 1e-6);
    const auto memory = metrics.GetMemoryUsage().psnr_hvs;
    const auto planned = GetMemoryRequirements(config).psnr_hvs;
    CHECK(memory && planned);
    if (memory && planned) {
        const auto& scenario =
            memory->uses_staging ? planned->staged : planned->direct;
        CHECK(memory->arena.buffer_bytes == scenario.arena.buffer_bytes);
    }
}

VQM_TEST(TestPsnrHvsTrailingPixelsAndSymmetry) {
    if (!test::GpuAvailable()) return;
    const auto original = ConstantYuv({20, 20}, 8, 0, 0, {64, 96, 128});
    auto distorted = original;
    const auto planes = internal::PlaneLayout(original.format, original.size);
    // Windows start at 0 and 7 and end at 14. Rows 15..19 are uncovered.
    for (const auto& plane : planes) {
        if (!plane.present()) continue;
        std::memset(
            distorted.data.data() + plane.offset + 15 * plane.width, 255,
            5 * plane.width
        );
    }
    PsnrHvsPipeline pipeline({.size = original.size});
    CheckIdentity(pipeline.Compute(original, distorted));
    distorted.data[0] = 255;
    const auto forward = pipeline.Compute(original, distorted);
    const auto reverse = pipeline.Compute(distorted, original);
    CHECK(std::isfinite(forward.score));
    CHECK(forward.normalized_error[0] > 0.0);
    CHECK_NEAR(forward.score, reverse.score, 1e-6);
    CHECK(forward.normalized_error == reverse.normalized_error);
}

VQM_TEST(TestPsnrHvsSubmissionAndHandleContracts) {
    if (!test::GpuAvailable()) return;
    const auto image = ConstantYuv({32, 32}, 8, 1, 1, {64, 96, 128});
    PsnrHvsPipeline pipeline({.size = image.size});
    const auto resident = pipeline.Upload(image);
    const auto token = pipeline.Submit(resident, resident);
    CHECK_THROWS_AS(pipeline.Submit(image, image), std::logic_error);
    CHECK_THROWS_AS(pipeline.Upload(image), std::logic_error);
    CHECK_THROWS_AS(pipeline.Release(resident), std::logic_error);
    CHECK_THROWS_AS(pipeline.AllocateImage(image.format), std::logic_error);
    CHECK_THROWS_AS(pipeline.Finish(token + 1), std::invalid_argument);
    CheckIdentity(pipeline.Finish(token));
    CHECK_THROWS_AS(pipeline.Finish(token), std::invalid_argument);
    pipeline.Release(resident);
    CHECK_THROWS_AS(pipeline.Compute(resident, resident), std::invalid_argument);
    const auto replacement = pipeline.Upload(image);
    CHECK_THROWS_AS(
        pipeline.Compute(resident, replacement), std::invalid_argument
    );
    CheckIdentity(pipeline.Compute(replacement, replacement));
    PsnrHvsPipeline foreign({.size = image.size});
    CHECK_THROWS_AS(
        foreign.Compute(replacement, replacement), std::invalid_argument
    );
}

VQM_TEST(TestPsnrHvsDisabledStreamingAndMappedInputs) {
    if (!test::GpuAvailable()) return;
    const auto original = ConstantYuv({32, 32}, 12, 1, 1, {64, 96, 128});
    const auto distorted = ConstantYuv({32, 32}, 12, 1, 1, {67, 101, 135});
    PsnrHvsPipeline pipeline(
        {.size = original.size,
         .input_bytes = 2 * original.data.size() + 128,
         .streaming_inputs = false}
    );
    CHECK_THROWS_AS(
        pipeline.Compute(original, distorted), std::invalid_argument
    );
    CHECK_THROWS_AS(
        pipeline.Compute(View(original), View(distorted)), std::invalid_argument
    );
    const auto resident = pipeline.Upload(original);
    CHECK_THROWS_AS(
        pipeline.Compute(resident, distorted), std::invalid_argument
    );
    CheckIdentity(pipeline.Compute(resident, resident));
    if (pipeline.GetMemoryUsage().uses_staging) {
        CHECK_THROWS_AS(
            pipeline.AllocateImage(original.format), std::runtime_error
        );
        return;
    }
    const auto a = pipeline.AllocateImage(original.format);
    const auto b = pipeline.AllocateImage(distorted.format);
    CHECK(a.bytes == original.data.size());
    CHECK(b.bytes == distorted.data.size());
    std::memcpy(
        pipeline.MapImage(a), original.data.data(), original.data.size()
    );
    std::memcpy(
        pipeline.MapImage(b), distorted.data.data(), distorted.data.size()
    );
    CheckConstantError(pipeline.Compute(a, b), 12);
    const auto token = pipeline.Submit(a, b);
    CHECK_THROWS_AS(pipeline.MapImage(a), std::logic_error);
    CHECK_THROWS_AS(pipeline.Release(a), std::logic_error);
    CheckConstantError(pipeline.Finish(token), 12);
    pipeline.Release(a);
    CHECK_THROWS_AS(pipeline.Compute(a, b), std::invalid_argument);
    pipeline.Release(b);
}

VQM_TEST(TestPsnrHvsRejectsUnsupportedInputs) {
    if (!test::GpuAvailable()) return;
    const auto image = ConstantYuv({32, 32}, 8, 1, 1, {64, 96, 128});
    PsnrHvsPipeline pipeline({.size = image.size});
    auto invalid = image;
    invalid.data.clear();
    CHECK_THROWS_AS(pipeline.Compute(invalid, image), std::invalid_argument);
    invalid = image;
    invalid.format.family = ColorFamily::Rgb;
    CHECK_THROWS_AS(pipeline.Upload(invalid), std::invalid_argument);
    invalid = image;
    invalid.format.range = PixelRange::Limited;
    CHECK_THROWS_AS(pipeline.Compute(image, invalid), std::invalid_argument);
    CHECK_THROWS_AS(
        pipeline.Compute(image, test::MakeGradient(0)), std::invalid_argument
    );
    PsnrHvsPipeline small({.size = {8, 8}});
    const auto subsampled = ConstantYuv({8, 8}, 8, 1, 1, {64, 96, 128});
    CHECK_THROWS_AS(
        small.Compute(subsampled, subsampled), std::invalid_argument
    );
    CHECK_THROWS_AS(PsnrHvsPipeline({.size = {7, 8}}), std::invalid_argument);
}
