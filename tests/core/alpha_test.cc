// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <algorithm>

#include "vqm_alpha.h"
#include "vqm_views.h"
#include "vqm_vulkan.h"
#include "support.h"

namespace {

// Keep the helper's Vulkan overloads instantiated in the public API test.
[[maybe_unused]] double VulkanWorst(
    vqm::Ssimulacra2Pipeline& pipeline, const vqm::VulkanBufferView& original,
    const vqm::VulkanBufferView& distorted
) {
    return vqm::ComputeWorstBackground(pipeline, original, distorted).Score();
}

}  // namespace

VQM_TEST(TestWorstBackgroundHelper) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm;
    const auto original = test::MakeRampImage(32, 32, true, PixelType::Byte);
    auto distorted = original;
    for (std::size_t i = 0; i < 128; ++i) distorted.data[i] ^= 0x55u;

    Ssimulacra2Pipeline ssimu({original.size});
    const double dark = ssimu.Compute(original, distorted, 0.1f).Score();
    const double bright = ssimu.Compute(original, distorted, 0.9f).Score();
    CHECK_NEAR(
        ComputeWorstBackground(ssimu, original, distorted).Score(),
        std::min(dark, bright), 1e-9
    );
    const auto view = [](const Image& image) {
        ImageView result{image.size, image.format, {}};
        constexpr std::size_t plane_bytes = 32 * 32;
        for (std::size_t plane = 0; plane < 4; ++plane) {
            result.planes[plane] = {reinterpret_cast<const std::byte*>(
                                        image.data.data() + plane * plane_bytes
                                    ),
                                    32};
        }
        return result;
    };
    CHECK_NEAR(
        ComputeWorstBackground(ssimu, view(original), view(distorted)).Score(),
        std::min(dark, bright), 1e-9
    );

    ButteraugliPipeline butter({original.size});
    const auto butter_dark = butter.Compute(original, distorted, 0.1f);
    const auto butter_bright = butter.Compute(original, distorted, 0.9f);
    const auto expected =
        butter_dark.score >= butter_bright.score ? butter_dark : butter_bright;
    const auto actual = ComputeWorstBackground(butter, original, distorted);
    CHECK_NEAR(actual.score, expected.score, 1e-9);
    CHECK_NEAR(actual.distance_p3, expected.distance_p3, 1e-9);
    const auto via_view =
        ComputeWorstBackground(butter, view(original), view(distorted));
    CHECK_NEAR(via_view.score, expected.score, 1e-9);
    CHECK_NEAR(via_view.distance_p3, expected.distance_p3, 1e-9);

    const auto opaque = test::MakeGradient(0);
    const auto other = test::MakeGradient(1);
    CHECK_NEAR(
        ComputeWorstBackground(ssimu, opaque, other).Score(),
        ssimu.Compute(opaque, other).Score(), 1e-9
    );
}

VQM_TEST(TestWorstBackgroundWithAlphaOnEitherSide) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm;

    auto opaque = test::MakeRampImage(32, 32, false, PixelType::Byte);
    std::fill(opaque.data.begin(), opaque.data.end(), 128u);
    auto transparent = opaque;
    transparent.format.alpha = AlphaType::Straight;
    transparent.data.resize(RequiredBytes(transparent.format, transparent.size));

    const auto view = [](const Image& image) {
        ImageView result{image.size, image.format, {}};
        const std::size_t count = image.HasAlpha() ? 4 : 3;
        constexpr std::size_t plane_bytes = 32 * 32;
        for (std::size_t plane = 0; plane < count; ++plane) {
            result.planes[plane] = {reinterpret_cast<const std::byte*>(
                                        image.data.data() + plane * plane_bytes
                                    ),
                                    32};
        }
        return result;
    };

    Ssimulacra2Pipeline ssimu({opaque.size});
    ButteraugliPipeline butter({opaque.size});
    for (const auto& pair :
         {std::pair{&opaque, &transparent}, std::pair{&transparent, &opaque}}) {
        const auto& original = *pair.first;
        const auto& distorted = *pair.second;
        const double dark = ssimu.Compute(original, distorted, 0.1f).Score();
        const double bright = ssimu.Compute(original, distorted, 0.9f).Score();
        const double expected = std::min(dark, bright);
        CHECK(expected < 90.0);
        CHECK_NEAR(
            ComputeWorstBackground(ssimu, original, distorted).Score(), expected,
            1e-9
        );
        CHECK_NEAR(
            ComputeWorstBackground(ssimu, view(original), view(distorted))
                .Score(),
            expected, 1e-9
        );

        const auto butter_dark = butter.Compute(original, distorted, 0.1f);
        const auto butter_bright = butter.Compute(original, distorted, 0.9f);
        const auto butter_expected = butter_dark.score >= butter_bright.score
                                         ? butter_dark
                                         : butter_bright;
        CHECK(butter_expected.score > 0.0);
        const auto actual = ComputeWorstBackground(butter, original, distorted);
        CHECK_NEAR(actual.score, butter_expected.score, 1e-9);
        CHECK_NEAR(actual.distance_p3, butter_expected.distance_p3, 1e-9);
        const auto via_view =
            ComputeWorstBackground(butter, view(original), view(distorted));
        CHECK_NEAR(via_view.score, butter_expected.score, 1e-9);
        CHECK_NEAR(via_view.distance_p3, butter_expected.distance_p3, 1e-9);
    }
}
