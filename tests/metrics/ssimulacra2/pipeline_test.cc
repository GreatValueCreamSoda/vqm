// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// Ssimulacra2Pipeline-level tests: golden scores, validation, repeatability and
// the resident-image pool.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "butteraugli.h"
#include "handles.h"
#include "vqm.h"
#include "vqm_views.h"
#include "support.h"
#include "vulkan/vulkan_runtime.h"

namespace {

vqm::Ssimulacra2Pipeline MakePipeline(std::size_t width, std::size_t height) {
    return vqm::Ssimulacra2Pipeline({{width, height}});
}

bool SameMsssim(
    const vqm::Ssimulacra2Result& a, const vqm::Ssimulacra2Result& b
) {
    if (a.scales.size() != b.scales.size()) {
        return false;
    }
    for (std::size_t s = 0; s < a.scales.size(); ++s) {
        if (a.scales[s].avg_ssim != b.scales[s].avg_ssim ||
            a.scales[s].avg_edgediff != b.scales[s].avg_edgediff) {
            return false;
        }
    }
    return true;
}

}  // namespace

VQM_TEST(TestMixedResidentReference) {
    if (!vqm::test::GpuAvailable()) return;
    const auto original =
        vqm::test::MakeRampImage(32, 32, true, vqm::PixelType::Byte);
    auto distorted = original;
    std::memset(distorted.data.data(), 0, 32 * 32);
    auto other = original;
    std::memset(other.data.data() + 32 * 32, 0, 32 * 32);
    const auto check = [&](auto& pipeline, auto score) {
        const double dark = score(pipeline.Compute(original, distorted, 0.1f));
        const double bright = score(pipeline.Compute(original, distorted, 0.9f));
        const double other_dark = score(pipeline.Compute(original, other, 0.1f));
        const auto reference = pipeline.Upload(original);
        CHECK_NEAR(
            score(pipeline.Compute(reference, distorted, 0.1f)), dark, 1e-12
        );
        CHECK_NEAR(
            score(pipeline.Compute(reference, distorted, 0.1f)), dark, 1e-12
        );
        CHECK_NEAR(
            score(pipeline.Compute(reference, other, 0.1f)), other_dark, 1e-12
        );
        CHECK_NEAR(
            score(pipeline.Compute(reference, distorted, 0.9f)), bright, 1e-12
        );
        const auto token = pipeline.Submit(reference, distorted);
        CHECK_THROWS_AS(
            (void)pipeline.Submit(reference, distorted), std::logic_error
        );
        (void)pipeline.Finish(token);
        auto short_image = distorted;
        short_image.data.pop_back();
        CHECK_THROWS_AS(
            (void)pipeline.Submit(reference, short_image), std::invalid_argument
        );
        pipeline.Release(reference);
        CHECK_THROWS_AS(
            (void)pipeline.Compute(reference, distorted), std::invalid_argument
        );
        const auto replacement = pipeline.Upload(distorted);
        CHECK_NEAR(
            score(pipeline.Compute(replacement, distorted)),
            score(pipeline.Compute(distorted, distorted)), 1e-12
        );
    };
    vqm::Ssimulacra2Pipeline ssimu({.size = original.size});
    check(ssimu, [](const vqm::Ssimulacra2Result& value) {
        return value.Score();
    });
    vqm::ButteraugliPipeline butter({.size = original.size});
    check(butter, [](const vqm::ButteraugliScore& value) {
        return value.score;
    });
    vqm::Ssimulacra2Config ssimu_config;
    ssimu_config.size = original.size;
    ssimu_config.streaming_inputs = false;
    vqm::Ssimulacra2Pipeline ssimu_no_stream(ssimu_config);
    const auto ssimu_ref = ssimu_no_stream.Upload(original);
    CHECK_THROWS_AS(
        (void)ssimu_no_stream.Compute(ssimu_ref, distorted),
        std::invalid_argument
    );
    vqm::ButteraugliConfig butter_config;
    butter_config.size = original.size;
    butter_config.streaming_inputs = false;
    vqm::ButteraugliPipeline butter_no_stream(butter_config);
    const auto butter_ref = butter_no_stream.Upload(original);
    CHECK_THROWS_AS(
        (void)butter_no_stream.Compute(butter_ref, distorted),
        std::invalid_argument
    );
}

VQM_TEST(TestIdenticalImages) {
    if (!vqm::test::GpuAvailable()) return;
    const vqm::Image img = vqm::test::MakeGradient(0);
    vqm::Ssimulacra2Pipeline pipeline =
        MakePipeline(vqm::test::kSide, vqm::test::kSide);
    const vqm::Ssimulacra2Result msssim = pipeline.Compute(img, img);
    // 32x32 yields scales 32, 16, 8 and 4 (the size check runs before the
    // downsample of the next scale, as in the reference).
    CHECK(msssim.scales.size() == 4);
    // The GPU maps divide the difference of numerator and denominator, so
    // equal planes cancel before any division rounds and a discrete-GPU driver
    // reports exactly 100. A software Vulkan ICD (llvmpipe) can still differ in
    // the last ulp of a norm, which the steep score mapping turns into
    // 99.94-99.98 (measured). The bound covers both drivers.
    const double score = msssim.Score();
    CHECK(score >= 99.5);
    CHECK(score <= 100.0);
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(
        vqm::internal::AcquireVulkanContext()->physical_device, &properties
    );
    if (properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
        CHECK(score == 100.0);
    }
}

VQM_TEST(TestGoldenScores) {
    if (!vqm::test::GpuAvailable()) return;
    const vqm::Image orig = vqm::test::MakeGradient(0);
    const vqm::Image dist_b = vqm::test::MakeGradient(1);
    const vqm::Image dist_c = vqm::test::MakeGradient(2);
    // Regression values for this exact-math implementation. The reference
    // binary scores these 93.09970849 and 92.11674825; the remaining gap comes
    // from the documented approximations (sRGB polynomial, 6-ulp cube root) and
    // is covered in the README. The MCOS correlation check in validation/ is
    // what ties these numbers to subjective quality.
    //
    // The metric has one backend now, so the tolerance covers the device's
    // rounding against exact math only: the measured Vulkan-vs-exact-math
    // spread on these pairs is at most 1.2e-4, and the rest of the allowance is
    // the pruned-map move the paragraph below records. The accuracy gate
    // remains the MCOS correlation in validation/.
    //
    // The values are those of the pruned channels
    // (src/metrics/ssimulacra2/pipeline.cc, Ssimulacra2ChannelPruned): this
    // 32x32 pair runs four scales, so the weight table's flat run gives it X at
    // 1:1, Y at 1:2 and B at 1:32 with weights at most 0.01, and those three
    // maps are no longer computed. That moved these two scores by -8.5e-4 and
    // -1.1e-3, and the 300-pair CID22 drift against the unpruned build is
    // mean 2.7e-4 over a maximum of 8.1e-4.
    vqm::Ssimulacra2Pipeline pipeline =
        MakePipeline(vqm::test::kSide, vqm::test::kSide);
    CHECK_NEAR(pipeline.Compute(orig, dist_b).Score(), 93.17505696, 5e-3);
    CHECK_NEAR(pipeline.Compute(orig, dist_c).Score(), 92.12553218, 5e-3);
}

// Caller-owned image memory: declare the budget, get exactly the bytes the
// format needs, write them yourself, score two of them. It has to agree with
// the copy-in path.
VQM_TEST(TestCallerOwnedImageBuffers) {
    if (!vqm::test::GpuAvailable()) return;
    const vqm::Image orig = vqm::test::MakeGradient(0);
    const vqm::Image dist = vqm::test::MakeGradient(2);
    const std::size_t per_image = vqm::RequiredBytes(orig.format, orig.size);
    const double expected = MakePipeline(vqm::test::kSide, vqm::test::kSide)
                                .Compute(orig, dist)
                                .Score();

    vqm::Ssimulacra2Config config;
    config.size = {vqm::test::kSide, vqm::test::kSide};
    config.input_bytes = 2 * per_image;
    vqm::Ssimulacra2Pipeline pipeline(config);

    const vqm::ImageBuffer a = pipeline.AllocateImage(orig.format);
    const vqm::ImageBuffer b = pipeline.AllocateImage(dist.format);
    CHECK(a.bytes == per_image);
    CHECK(b.bytes == per_image);
    std::memcpy(pipeline.MapImage(a), orig.data.data(), orig.data.size());
    std::memcpy(pipeline.MapImage(b), dist.data.data(), dist.data.size());
    CHECK_NEAR(pipeline.Compute(a, b).Score(), expected, 1e-12);

    const std::uint64_t token = pipeline.Submit(a, b);
    // In flight: the mapping and the pool must not move under the GPU.
    CHECK_THROWS_AS(pipeline.Release(a), std::logic_error);
    CHECK_THROWS_AS(pipeline.MapImage(a), std::logic_error);
    CHECK_THROWS_AS(pipeline.AllocateImage(orig.format), std::logic_error);
    CHECK_NEAR(pipeline.Finish(token).Score(), expected, 1e-12);
    pipeline.Release(a);
    CHECK_THROWS_AS(pipeline.Compute(a, b), std::invalid_argument);
    pipeline.Release(b);
    // Two buffers are all the pool has, so a third allocation must refuse.
    const vqm::ImageBuffer c = pipeline.AllocateImage(orig.format);
    const vqm::ImageBuffer d = pipeline.AllocateImage(orig.format);
    CHECK_THROWS_AS(pipeline.AllocateImage(orig.format), std::invalid_argument);
    pipeline.Release(c);
    pipeline.Release(d);
}

VQM_TEST(TestInputFormatValidationAcrossPaths) {
    using namespace vqm;
    if (!test::GpuAvailable()) return;
    for (const Size size : {Size{32, 32}, Size{33, 17}}) {
        Ssimulacra2Config config{.size = size};
        const std::size_t image_bytes = RequiredBytes(ImageFormat{}, size);
        config.input_bytes = 2 * image_bytes + 63;  // Align the second start.
        Ssimulacra2Pipeline ssimu(config);
        ButteraugliConfig butter_config{.size = size};
        butter_config.input_bytes = config.input_bytes;
        ButteraugliPipeline butter(butter_config);
        std::vector<ImageFormat> invalid;
        ImageFormat format;
        format.subsample_w = 1;  // RGB must never have subsampled G/B planes.
        invalid.push_back(format);
        format = {};
        format.family = ColorFamily::Yuv;  // Its RGB matrix is incompatible.
        invalid.push_back(format);
        format.matrix = MatrixCoefficients::Bt709;
        format.subsample_w = 1;
        format.subsample_h = 1;
        if (size.width % 2 != 0 || size.height % 2 != 0)
            invalid.push_back(format);
        format = {};
        format.range = static_cast<PixelRange>(255);
        invalid.push_back(format);
        format = {};
        format.alpha = static_cast<AlphaType>(255);
        invalid.push_back(format);
        for (const auto& bad_format : invalid) {
            const Image image{size, bad_format, {}};
            const ImageView view{size, bad_format, {}};
            const auto rejects = [&](auto& pipeline) {
                CHECK_THROWS_AS(
                    pipeline.Compute(image, image), std::invalid_argument
                );
                CHECK_THROWS_AS(
                    pipeline.Compute(view, view), std::invalid_argument
                );
                CHECK_THROWS_AS(pipeline.Upload(image), std::invalid_argument);
                CHECK_THROWS_AS(
                    pipeline.AllocateImage(bad_format), std::invalid_argument
                );
            };
            rejects(ssimu);
            rejects(butter);
        }
        // Rejected formats must not consume any of the caller-owned pool.
        if (!ssimu.GetMemoryUsage().uses_staging) {
            const auto a = ssimu.AllocateImage(ImageFormat{});
            const auto b = ssimu.AllocateImage(ImageFormat{});
            CHECK(a.bytes + b.bytes == 2 * image_bytes);
        }
    }
}

// A resident-only caller can refuse the streaming input region: the pixel
// overloads then throw and the resident path still produces the same score.
VQM_TEST(TestStreamingInputsDisabled) {
    if (!vqm::test::GpuAvailable()) return;
    const vqm::Image orig = vqm::test::MakeGradient(0);
    const vqm::Image dist = vqm::test::MakeGradient(1);
    const double expected = MakePipeline(vqm::test::kSide, vqm::test::kSide)
                                .Compute(orig, dist)
                                .Score();

    vqm::Ssimulacra2Config config;
    config.size = {vqm::test::kSide, vqm::test::kSide};
    config.gpu_resident_images = 2;
    config.streaming_inputs = false;
    vqm::Ssimulacra2Pipeline pipeline(config);

    CHECK_THROWS_AS(pipeline.Compute(orig, dist), std::invalid_argument);
    const vqm::UploadedImage a = pipeline.Upload(orig);
    const vqm::UploadedImage b = pipeline.Upload(dist);
    CHECK_NEAR(pipeline.Compute(a, b).Score(), expected, 1e-12);
    const std::uint64_t token = pipeline.Submit(a, b);
    CHECK_NEAR(pipeline.Finish(token).Score(), expected, 1e-12);
    CHECK_THROWS_AS(pipeline.Finish(token), std::invalid_argument);
}

VQM_TEST(TestPipelineValidation) {
    if (!vqm::test::GpuAvailable()) return;
    const vqm::Image img = vqm::test::MakeGradient(0);
    vqm::Ssimulacra2Pipeline pipeline({{vqm::test::kSide, vqm::test::kSide}});

    vqm::Image other = img;
    other.size.width = vqm::test::kSide / 2;
    CHECK_THROWS_AS((void)pipeline.Compute(img, other), std::invalid_argument);

    vqm::Image short_data = img;
    short_data.data.resize(10);
    CHECK_THROWS_AS(
        (void)pipeline.Compute(short_data, img), std::invalid_argument
    );

    vqm::Image bad_format = img;
    bad_format.format.subsample_w = 1;
    CHECK_THROWS_AS(
        (void)pipeline.Compute(bad_format, img), std::invalid_argument
    );

    // The geometry is validated before any device work, so an impossible size
    // names the minimum rather than failing on the device.
    bool size_reported = false;
    try {
        vqm::Ssimulacra2Pipeline bad_geometry({{4, 8}});
    } catch (const std::invalid_argument& e) {
        size_reported = std::string(e.what()).find("8x8") != std::string::npos;
    }
    CHECK(size_reported);

    vqm::Ssimulacra2Result too_many_scales;
    too_many_scales.scales.resize(7);
    CHECK_THROWS_AS(too_many_scales.Score(), std::invalid_argument);
}

VQM_TEST(TestPipelineRepeatable) {
    if (!vqm::test::GpuAvailable()) return;
    const vqm::Image orig = vqm::test::MakeGradient(0);
    const vqm::Image dist = vqm::test::MakeGradient(1);
    vqm::Ssimulacra2Pipeline pipeline =
        MakePipeline(vqm::test::kSide, vqm::test::kSide);
    const vqm::Ssimulacra2Result first = pipeline.Compute(orig, dist);
    const vqm::Ssimulacra2Result second = pipeline.Compute(orig, dist);
    CHECK(SameMsssim(first, second));
}

VQM_TEST(TestPipelineInterleaved) {
    if (!vqm::test::GpuAvailable()) return;
    const vqm::Image orig = vqm::test::MakeGradient(0);
    const vqm::Image dist_a = vqm::test::MakeGradient(1);
    const vqm::Image dist_b = vqm::test::MakeGradient(2);
    vqm::Ssimulacra2Pipeline pipeline =
        MakePipeline(vqm::test::kSide, vqm::test::kSide);
    const vqm::Ssimulacra2Result first = pipeline.Compute(orig, dist_a);
    (void)pipeline.Compute(orig, dist_b);
    const vqm::Ssimulacra2Result again = pipeline.Compute(orig, dist_a);
    CHECK(SameMsssim(first, again));
}

VQM_TEST(TestStreamingSubmissionValidation) {
    if (!vqm::test::GpuAvailable()) return;
    const vqm::Image orig = vqm::test::MakeGradient(0);
    const vqm::Image dist = vqm::test::MakeGradient(1);
    vqm::Image oversized = dist;
    // Exceed even the largest possible four-float-plane streaming slot.
    oversized.data.resize(20 * orig.size.width * orig.size.height);
    vqm::Image short_data = orig;
    short_data.data.resize(1);
    vqm::Image wrong_size = orig;
    --wrong_size.size.width;

    const auto check = [&](auto& pipeline, auto score) {
        const auto resident = pipeline.Upload(orig);
        const auto resident_dist = pipeline.Upload(dist);
        const vqm::UploadedImage invalid_resident{};
        const vqm::ImageBuffer invalid_buffer{};
        CHECK_THROWS_AS((void)pipeline.Upload(orig), std::invalid_argument);
        CHECK_THROWS_AS(
            (void)pipeline.Submit(orig, oversized), std::invalid_argument
        );
        CHECK_THROWS_AS(
            (void)pipeline.Submit(short_data, dist), std::invalid_argument
        );
        CHECK_THROWS_AS(
            (void)pipeline.Submit(wrong_size, dist), std::invalid_argument
        );
        const double expected = score(pipeline.Compute(orig, dist));
        const auto token = pipeline.Submit(orig, dist);
        // An attempted identity submission must neither replace the first
        // pair nor alter its inputs before the pending token is finished.
        CHECK_THROWS_AS((void)pipeline.Submit(orig, orig), std::logic_error);
        CHECK_THROWS_AS(
            (void)pipeline.Submit(orig, oversized), std::logic_error
        );
        CHECK_THROWS_AS(
            (void)pipeline.Compute(orig, oversized), std::logic_error
        );
        CHECK_THROWS_AS((void)pipeline.Upload(orig), std::logic_error);
        CHECK_THROWS_AS((void)pipeline.Upload(oversized), std::logic_error);
        CHECK_THROWS_AS(pipeline.Release(resident), std::logic_error);
        CHECK_THROWS_AS(pipeline.Release(invalid_resident), std::logic_error);
        CHECK_THROWS_AS(
            (void)pipeline.Submit(resident, resident_dist), std::logic_error
        );
        CHECK_THROWS_AS(
            (void)pipeline.Submit(invalid_resident, resident), std::logic_error
        );
        CHECK_THROWS_AS(
            (void)pipeline.Compute(invalid_resident, resident), std::logic_error
        );
        CHECK_THROWS_AS(
            (void)pipeline.AllocateImage(orig.format), std::logic_error
        );
        CHECK_THROWS_AS(
            (void)pipeline.MapImage(invalid_buffer), std::logic_error
        );
        CHECK_THROWS_AS(pipeline.Release(invalid_buffer), std::logic_error);
        CHECK_THROWS_AS(
            (void)pipeline.Submit(invalid_buffer, invalid_buffer),
            std::logic_error
        );
        CHECK_THROWS_AS(
            (void)pipeline.Compute(invalid_buffer, invalid_buffer),
            std::logic_error
        );
        CHECK_NEAR(score(pipeline.Finish(token)), expected, 1e-12);
        CHECK_THROWS_AS((void)pipeline.Finish(token), std::invalid_argument);
        CHECK_THROWS_AS(
            (void)pipeline.Submit(invalid_resident, resident),
            std::invalid_argument
        );
        CHECK_THROWS_AS(
            (void)pipeline.Submit(invalid_buffer, invalid_buffer),
            std::invalid_argument
        );
        pipeline.Release(invalid_resident);
        pipeline.Release(invalid_buffer);
        pipeline.Release(resident);
        pipeline.Release(resident_dist);
    };

    vqm::Ssimulacra2Pipeline ssimu({orig.size});
    check(ssimu, [](const vqm::Ssimulacra2Result& result) {
        return result.Score();
    });
    vqm::ButteraugliPipeline butter({orig.size});
    check(butter, [](const vqm::ButteraugliScore& result) {
        return result.score;
    });
}

VQM_TEST(TestResidentImages) {
    if (!vqm::test::GpuAvailable()) return;
    const vqm::Image orig = vqm::test::MakeGradient(0);
    const vqm::Image dist = vqm::test::MakeGradient(1);
    vqm::Ssimulacra2Pipeline pipeline =
        MakePipeline(vqm::test::kSide, vqm::test::kSide);
    const vqm::UploadedImage a = pipeline.Upload(orig);
    const vqm::UploadedImage b = pipeline.Upload(dist);
    CHECK_NEAR(
        pipeline.Compute(a, b).Score(), pipeline.Compute(orig, dist).Score(),
        1e-5
    );

    // A released handle must not alias the image that later reuses its slot.
    pipeline.Release(a);
    CHECK_THROWS_AS((void)pipeline.Compute(a, b), std::invalid_argument);

    // One slot: a second upload has nowhere to go until the first is released.
    vqm::Ssimulacra2Config config;
    config.size = {vqm::test::kSide, vqm::test::kSide};
    config.gpu_resident_images = 1;
    vqm::Ssimulacra2Pipeline single(config);
    const vqm::UploadedImage first = single.Upload(orig);
    CHECK_THROWS_AS((void)single.Upload(dist), std::invalid_argument);
    single.Release(first);
}

VQM_TEST(TestResidentHandleContracts) {
    if (!vqm::test::GpuAvailable()) return;
    const vqm::Image original = vqm::test::MakeGradient(0);
    const vqm::Image distorted = vqm::test::MakeGradient(1);
    const auto check = [&](auto& pipeline, auto score) {
        const auto old = pipeline.Upload(original);
        const auto other = pipeline.Upload(distorted);
        const double before = score(pipeline.Compute(old, other));
        CHECK_THROWS_AS((void)pipeline.Upload(original), std::invalid_argument);

        auto wrong_size = other;
        --wrong_size.size.width;
        CHECK_THROWS_AS(
            (void)pipeline.Submit(old, wrong_size), std::invalid_argument
        );

        pipeline.Release(old);
        const auto replacement = pipeline.Upload(distorted);
        CHECK_THROWS_AS(
            (void)pipeline.Submit(old, other), std::invalid_argument
        );
        const double expected = score(pipeline.Compute(distorted, distorted));
        const double after = score(pipeline.Compute(replacement, other));
        CHECK_NEAR(after, expected, 1e-12);
        CHECK(std::fabs(before - after) > 1e-5);
    };

    vqm::Ssimulacra2Pipeline ssimu({original.size});
    check(ssimu, [](const vqm::Ssimulacra2Result& result) {
        return result.Score();
    });
    vqm::ButteraugliPipeline butter({original.size});
    check(butter, [](const vqm::ButteraugliScore& result) {
        return result.score;
    });
}

VQM_TEST(TestPipelineHandleOwnershipAndMoves) {
    if (!vqm::test::GpuAvailable()) return;
    const auto original = vqm::test::MakeGradient(0);
    const auto distorted = vqm::test::MakeGradient(1);
    const auto check = [&](auto first, auto second, auto score) {
        const auto a = first.Upload(original);
        const auto b = second.Upload(distorted);
        CHECK(a.slot != b.slot);
        CHECK_THROWS_AS(second.Compute(a, b), std::invalid_argument);
        // A foreign release is ignored and must not release the local image.
        second.Release(a);
        const double identity = score(second.Compute(b, b));
        const auto token_a = first.Submit(a, a);
        const auto token_b = second.Submit(b, b);
        CHECK(token_a != token_b);
        CHECK_THROWS_AS(second.Finish(token_a), std::invalid_argument);
        CHECK_NEAR(score(second.Finish(token_b)), identity, 1e-12);
        const double original_identity = score(first.Finish(token_a));

        // Moving a pipeline preserves ownership of its live handles.
        auto moved = std::move(first);
        CHECK_NEAR(score(moved.Compute(a, a)), original_identity, 1e-12);
        auto assigned = std::move(second);
        assigned = std::move(moved);
        CHECK_NEAR(score(assigned.Compute(a, a)), original_identity, 1e-12);
        CHECK_THROWS_AS(assigned.Compute(b, b), std::invalid_argument);
        assigned.Release(a);
    };
    check(
        vqm::Ssimulacra2Pipeline({original.size}),
        vqm::Ssimulacra2Pipeline({original.size}),
        [](const vqm::Ssimulacra2Result& result) { return result.Score(); }
    );
    check(
        vqm::ButteraugliPipeline({original.size}),
        vqm::ButteraugliPipeline({original.size}),
        [](const vqm::ButteraugliScore& result) { return result.score; }
    );

    // Handle identities are unique across metric implementations as well.
    vqm::Ssimulacra2Pipeline ssimu({original.size});
    vqm::ButteraugliPipeline butter({original.size});
    const auto ssimu_image = ssimu.Upload(original);
    const auto butter_image = butter.Upload(distorted);
    CHECK_THROWS_AS(
        butter.Compute(ssimu_image, butter_image), std::invalid_argument
    );
    CHECK_THROWS_AS(
        ssimu.Compute(butter_image, ssimu_image), std::invalid_argument
    );
}

VQM_TEST(TestInputPoolExactSizeAndOwnership) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm;
    constexpr Size size{33, 17};
    const auto format = test::PinnedFormat(AlphaType::None, PixelType::Byte);
    const std::size_t bytes = RequiredBytes(format, size);
    CHECK(bytes == 1692);

    const auto check = [&](auto first, auto second) {
        ImageBuffer a;
        try {
            a = first.AllocateImage(format);
        } catch (const std::runtime_error& error) {
            CHECK(
                std::string_view(error.what())
                    .find("caller-owned image buffers") != std::string_view::npos
            );
            return;
        }
        const ImageBuffer b = second.AllocateImage(format);
        CHECK(a.bytes == bytes);
        CHECK(b.bytes == bytes);
        CHECK(a.slot != b.slot);
        CHECK(first.MapImage(a) != nullptr);
        CHECK_THROWS_AS(second.MapImage(a), std::invalid_argument);
        CHECK_THROWS_AS(second.Submit(a, b), std::invalid_argument);
        second.Release(a);
        CHECK(second.MapImage(b) != nullptr);
        CHECK_THROWS_AS(first.AllocateImage(format), std::invalid_argument);
        auto moved = std::move(first);
        CHECK(moved.MapImage(a) != nullptr);
        moved.Release(a);
        second.Release(b);
    };
    Ssimulacra2Config ssimu_config;
    ssimu_config.size = size;
    ssimu_config.input_bytes = bytes;
    check(Ssimulacra2Pipeline(ssimu_config), Ssimulacra2Pipeline(ssimu_config));
    ButteraugliConfig butter_config;
    butter_config.size = size;
    butter_config.input_bytes = bytes;
    check(
        ButteraugliPipeline(butter_config), ButteraugliPipeline(butter_config)
    );
}

VQM_TEST(TestInputHandleBeyondOldGenerationLimit) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm;
    constexpr Size size{8, 8};
    const auto format = test::PinnedFormat(AlphaType::None, PixelType::Byte);
    Ssimulacra2Config config;
    config.size = size;
    config.input_bytes = RequiredBytes(format, size);
    config.gpu_resident_images = 0;
    config.streaming_inputs = false;
    Ssimulacra2Pipeline pipeline(config);
    ImageBuffer stale;
    try {
        stale = pipeline.AllocateImage(format);
    } catch (const std::runtime_error& error) {
        CHECK(
            std::string_view(error.what()).find("caller-owned image buffers") !=
            std::string_view::npos
        );
        return;
    }
    pipeline.Release(stale);
    ImageBuffer current;
    constexpr std::size_t old_generation_limit = 1u << 20;
    for (std::size_t i = 0; i < old_generation_limit; ++i) {
        current = pipeline.AllocateImage(format);
        if (i + 1 != old_generation_limit) pipeline.Release(current);
    }
    CHECK(current.slot != stale.slot);
    CHECK(pipeline.MapImage(current) != nullptr);
    CHECK_THROWS_AS(pipeline.MapImage(stale), std::invalid_argument);
    pipeline.Release(current);
}

VQM_TEST(TestInputPoolAlignsStartsAndReusesGaps) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm;
    constexpr Size size{33, 17};
    const auto format = test::PinnedFormat(AlphaType::None, PixelType::Byte);
    const std::size_t bytes = RequiredBytes(format, size);
    const std::size_t second_offset = (bytes + 63) / 64 * 64;
    Ssimulacra2Config config;
    config.size = size;
    config.input_bytes = second_offset + bytes;
    Ssimulacra2Pipeline pipeline(config);
    ImageBuffer first;
    try {
        first = pipeline.AllocateImage(format);
    } catch (const std::runtime_error& error) {
        CHECK(
            std::string_view(error.what()).find("caller-owned image buffers") !=
            std::string_view::npos
        );
        return;
    }
    const ImageBuffer second = pipeline.AllocateImage(format);
    auto* first_memory = pipeline.MapImage(first);
    auto* second_memory = pipeline.MapImage(second);
    const auto first_address = reinterpret_cast<std::uintptr_t>(first_memory);
    const auto second_address = reinterpret_cast<std::uintptr_t>(second_memory);
    CHECK(second_address - first_address == second_offset);
    std::memset(second_memory, 0x5a, bytes);
    CHECK_THROWS_AS(pipeline.AllocateImage(format), std::invalid_argument);
    pipeline.Release(first);
    const ImageBuffer replacement = pipeline.AllocateImage(format);
    CHECK(pipeline.MapImage(replacement) == first_memory);
    CHECK(pipeline.MapImage(second)[bytes - 1] == std::byte{0x5a});
    pipeline.Release(replacement);
    pipeline.Release(second);
}

VQM_TEST(TestInputHandleCapacity) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm;
    constexpr Size size{8, 8};
    auto format = test::PinnedFormat(AlphaType::None, PixelType::Byte);
    format.family = ColorFamily::Grey;
    format.matrix = MatrixCoefficients::Unspecified;
    const std::size_t bytes = RequiredBytes(format, size);
    CHECK(bytes == 64);
    Ssimulacra2Config config;
    config.size = size;
    config.input_bytes = bytes * 4096;
    config.gpu_resident_images = 0;
    config.streaming_inputs = false;
    Ssimulacra2Pipeline pipeline(config);
    std::vector<ImageBuffer> buffers;
    buffers.reserve(4095);
    try {
        buffers.push_back(pipeline.AllocateImage(format));
    } catch (const std::runtime_error& error) {
        CHECK(
            std::string_view(error.what()).find("caller-owned image buffers") !=
            std::string_view::npos
        );
        return;
    }
    for (std::size_t i = 1; i < 4095; ++i) {
        buffers.push_back(pipeline.AllocateImage(format));
    }
    CHECK_THROWS_AS(pipeline.AllocateImage(format), std::invalid_argument);
    const auto old = buffers[1024];
    pipeline.Release(old);
    const auto replacement = pipeline.AllocateImage(format);
    CHECK(replacement.slot != old.slot);
    CHECK_THROWS_AS(pipeline.MapImage(old), std::invalid_argument);
    CHECK(pipeline.MapImage(replacement) != nullptr);
}

VQM_TEST(TestResidentReferencePyramidReuse) {
    if (!vqm::test::GpuAvailable()) return;
    using vqm::PixelType;
    const auto original =
        vqm::test::MakeRampImage(32, 32, true, PixelType::Byte);
    auto distorted = original;
    distorted.data[0] = 220;
    vqm::Ssimulacra2Pipeline pipeline({original.size});
    const auto a = pipeline.Upload(original);
    const auto b = pipeline.Upload(distorted);
    const auto original_identity = pipeline.Compute(original, original, 0.9f);
    const auto distorted_identity = pipeline.Compute(distorted, distorted, 0.9f);
    for (float background : {0.1f, 0.9f, 0.9f}) {
        const auto expected = pipeline.Compute(original, distorted, background);
        const auto actual = pipeline.Compute(a, b, background);
        CHECK(SameMsssim(actual, expected));
        CHECK(SameMsssim(pipeline.Compute(a, b, background), expected));
    }
    // Changing reference, streaming, and slot reuse must each invalidate the
    // prepared reference. Compare against the matching streamed result: a
    // software Vulkan implementation may score identity slightly below 100.
    CHECK(SameMsssim(pipeline.Compute(b, b, 0.9f), distorted_identity));
    CHECK(SameMsssim(pipeline.Compute(a, a, 0.9f), original_identity));
    CHECK(SameMsssim(
        pipeline.Compute(distorted, distorted, 0.9f), distorted_identity
    ));
    CHECK(SameMsssim(pipeline.Compute(a, a, 0.9f), original_identity));
    pipeline.Release(b);
    const auto updated_distorted = pipeline.Upload(original);
    CHECK(SameMsssim(
        pipeline.Compute(a, updated_distorted, 0.9f), original_identity
    ));
    pipeline.Release(a);
    const auto replaced = pipeline.Upload(distorted);
    CHECK(SameMsssim(
        pipeline.Compute(replaced, replaced, 0.9f), distorted_identity
    ));
}
VQM_TEST(TestSmallGeometries) {
    if (!vqm::test::GpuAvailable()) return;
    // The smallest legal geometry still yields two scales (the pyramid halves
    // only after the first entry), so the last scale always runs the prepare
    // dispatch with the downsample bit cleared: 8x8 ends at 4x4, 13x9 at 7x5.
    // Include full and partial 64-pixel prepare tiles: their inactive lanes
    // must stay within the shared pyramid ladder at every emitted level.
    const std::array cases{
        std::pair{vqm::Size{8, 8}, 2u}, std::pair{vqm::Size{13, 9}, 2u},
        std::pair{vqm::Size{64, 64}, 5u}, std::pair{vqm::Size{65, 65}, 5u},
        std::pair{vqm::Size{129, 97}, 5u}};
    for (const auto& [size, scales] : cases) {
        const vqm::Image img =
            vqm::test::MakeRampImage(size.width, size.height, false);
        vqm::Ssimulacra2Pipeline pipeline({{size.width, size.height}});
        const vqm::Ssimulacra2Result first = pipeline.Compute(img, img);
        CHECK(first.scales.size() == scales);
        const double score = first.Score();
        CHECK(score >= 99.5);
        CHECK(score <= 100.0);
        const vqm::Ssimulacra2Result second = pipeline.Compute(img, img);
        CHECK(SameMsssim(first, second));
    }
}

VQM_TEST(TestHalfPlanes) {
    if (!vqm::test::GpuAvailable() || !vqm::test::HalfPlanesAvailable()) {
        return;
    }
    // The half path stores the planes as 16-bit floats, which rounds them to
    // ~5e-4 relative. That is the whole cost of the mode: the transfer
    // function, alpha and the norm sums stay f32, and identical images still
    // cancel exactly (100.00000000). What moves is the ringing/blurring map,
    // a difference of two nearly equal terms, so the rounding shows up in the
    // tail of its 4-norms -- always upwards, by at most 7.9e-5 here.
    //
    // Measured on the two pairs below: score -0.057 (gradient_C) and -0.031
    // (gradient_B), max |norm delta| 7.87e-5. The bounds leave room for a
    // second device's cube root and summation order; a delta above them is a
    // defect in the half path, not a tolerance to relax.
    constexpr double kScoreTolerance = 0.1;
    constexpr double kNormTolerance = 2e-4;
    const vqm::Image orig = vqm::test::MakeGradient(0);
    const std::array<vqm::Image, 2> distorted{vqm::test::MakeGradient(1),
                                              vqm::test::MakeGradient(2)};
    vqm::Ssimulacra2Pipeline exact(
        MakePipeline(vqm::test::kSide, vqm::test::kSide)
    );
    vqm::Ssimulacra2Config config;
    config.size = {vqm::test::kSide, vqm::test::kSide};
    config.half_planes = true;
    vqm::Ssimulacra2Pipeline half(config);
    for (const vqm::Image& dist : distorted) {
        const vqm::Ssimulacra2Result exact_msssim = exact.Compute(orig, dist);
        const vqm::Ssimulacra2Result half_msssim = half.Compute(orig, dist);
        CHECK(half_msssim.scales.size() == exact_msssim.scales.size());
        CHECK_NEAR(half_msssim.Score(), exact_msssim.Score(), kScoreTolerance);
        for (std::size_t s = 0; s < exact_msssim.scales.size(); ++s) {
            const auto& exact_scale = exact_msssim.scales[s];
            const auto& half_scale = half_msssim.scales[s];
            for (std::size_t n = 0; n < exact_scale.avg_ssim.size(); ++n) {
                CHECK_NEAR(
                    half_scale.avg_ssim[n], exact_scale.avg_ssim[n],
                    kNormTolerance
                );
            }
            for (std::size_t n = 0; n < exact_scale.avg_edgediff.size(); ++n) {
                CHECK_NEAR(
                    half_scale.avg_edgediff[n], exact_scale.avg_edgediff[n],
                    kNormTolerance
                );
            }
        }
    }
}
