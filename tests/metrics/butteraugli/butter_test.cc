// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// Butteraugli portions: Copyright (c) the JPEG XL Project Authors.
// BSD-3-Clause; see licenses/JPEG-XL-LICENSE.

// Butteraugli tests: identity, agreement with the double-precision CPU oracle
// in tests/metrics/butteraugli/butter_ref.cc (a line-cited port of libjxl's
// comparator), the 1x1 anchor that validates the oracle itself against libjxl's
// published test, and the small geometries at the fold boundary.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include "butter_ref.h"
#include "butteraugli.h"
#include "input.h"
#include "vqm.h"
#include "vqm_memory.h"
#include "support.h"

namespace {

using vqm::Image;
using vqm::Size;
using vqm::test::ButterRefParams;
using vqm::test::ButterRefResult;

// The sRGB EOTF. The fixtures are sRGB-encoded bytes with the identity matrix
// and BT.709 primaries, which is exactly what the pipeline's sRGB decode class
// inverts; the CPU oracle takes the linear values directly.
double SrgbToLinear(double c) {
    if (c <= 0.04045) return c / 12.92;
    return std::pow((c + 0.055) / 1.055, 2.4);
}

struct LinearPair {
    std::vector<float> r0, g0, b0, r1, g1, b1;
    int width = 0;
    int height = 0;
};

// Extracts the three colour planes of one sRGB byte image as linear floats.
void ToLinear(
    const Image& image, std::vector<float>* r, std::vector<float>* g,
    std::vector<float>* b
) {
    const std::size_t pixels = image.size.width * image.size.height;
    if (image.format.type != vqm::PixelType::Byte ||
        image.format.family != vqm::ColorFamily::Rgb) {
        vqm::test::Fail(
            std::source_location::current(), "butter fixtures must be 8-bit RGB"
        );
    }
    // Image's planes are planar with each start aligned to four bytes.
    r->assign(pixels, 0.0f);
    g->assign(pixels, 0.0f);
    b->assign(pixels, 0.0f);
    const auto layout = vqm::internal::PlaneLayout(image.format, image.size);
    const std::uint8_t* planes[3] = {image.data.data() + layout[0].offset,
                                     image.data.data() + layout[1].offset,
                                     image.data.data() + layout[2].offset};
    std::vector<float>* out[3] = {r, g, b};
    for (std::size_t c = 0; c < 3; ++c) {
        for (std::size_t i = 0; i < pixels; ++i) {
            (*out[c])[i] =
                static_cast<float>(SrgbToLinear(planes[c][i] / 255.0));
        }
    }
}

ButterRefResult Reference(const Image& original, const Image& distorted) {
    LinearPair pair;
    ToLinear(original, &pair.r0, &pair.g0, &pair.b0);
    ToLinear(distorted, &pair.r1, &pair.g1, &pair.b1);
    pair.width = static_cast<int>(original.size.width);
    pair.height = static_cast<int>(original.size.height);
    return ButteraugliCpuReference(
        pair.r0.data(), pair.g0.data(), pair.b0.data(), pair.r1.data(),
        pair.g1.data(), pair.b1.data(), pair.width, pair.height,
        ButterRefParams{}
    );
}

vqm::ButteraugliScore Gpu(const Image& original, const Image& distorted) {
    vqm::ButteraugliConfig config;
    config.size = original.size;
    config.streaming_inputs = true;
    vqm::ButteraugliPipeline pipeline(config);
    return pipeline.Compute(original, distorted);
}

enum class SpatialPattern { Checker, Chroma, TransposedChecker };

struct SpatialPair {
    Image original;
    Image distorted;
};

std::array<int, 3> SpatialRgb(int x, int y, SpatialPattern pattern) {
    const int checker = (x / 3 + y / 3) % 2;
    if (pattern == SpatialPattern::Chroma) {
        const int value = 80 + 96 * checker;
        const int opposite = (x / 2 + y / 5) % 2 != 0 ? 64 : -64;
        const int blue = (y / 3 + x / 7) % 2 != 0 ? 16 : -16;
        return {value + opposite, value - opposite, value + blue};
    }
    const int value = 64 + 128 * checker;
    return {value, value + x % 7 - 3, value + y % 5 - 2};
}

// Integer spatial detail exposes frequency separation and directional Malta
// errors that smooth 32x32 gradients miss. The checker matches the standalone
// bench/experiments/butteraugli_drift/make_checker.py reproducer. The chroma
// fixture combines opposing R/G with HF luminance, activating SuppressXByY.
SpatialPair MakeSpatialPair(Size size, SpatialPattern pattern) {
    Image original{size, {}, {}};
    original.format.type = vqm::PixelType::Byte;
    original.data.resize(vqm::RequiredBytes(original.format, size));
    Image distorted = original;
    const auto planes = vqm::internal::PlaneLayout(original.format, size);
    for (std::size_t index = 0; index < size.width * size.height; ++index) {
        int x = static_cast<int>(index % size.width);
        int y = static_cast<int>(index / size.width);
        if (pattern == SpatialPattern::TransposedChecker) std::swap(x, y);
        const auto rgb = SpatialRgb(x, y, pattern);
        const int noise = (13 * x + 7 * y + 3 * x * y) % 11 - 5;
        const int change = noise + ((x / 8 + y / 8) % 2 != 0 ? 5 : -5);
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const std::size_t offset = planes[channel].offset + index;
            original.data[offset] = static_cast<std::uint8_t>(rgb[channel]);
            distorted.data[offset] = static_cast<std::uint8_t>(
                std::clamp(rgb[channel] + change, 0, 255)
            );
        }
    }
    return {std::move(original), std::move(distorted)};
}

void CheckSpatialReference(Size size, SpatialPattern pattern) {
    const auto pair = MakeSpatialPair(size, pattern);
    const auto reference = Reference(pair.original, pair.distorted);
    vqm::ButteraugliPipeline pipeline({.size = size});
    const auto streamed = pipeline.Compute(pair.original, pair.distorted);
    CHECK(reference.score > 0.0);
    CHECK_NEAR(streamed.score, reference.score, 0.001 * reference.score);
    CHECK_NEAR(
        streamed.distance_p3, reference.distance_p3,
        0.001 * reference.distance_p3
    );

    // Repeated resident calls cover the cached reference as well as streaming.
    const auto original = pipeline.Upload(pair.original);
    const auto distorted = pipeline.Upload(pair.distorted);
    for (int repeat = 0; repeat < 2; ++repeat) {
        const auto resident = pipeline.Compute(original, distorted);
        CHECK_NEAR(resident.score, streamed.score, 1e-12);
        CHECK_NEAR(resident.distance_p3, streamed.distance_p3, 1e-12);
    }
}

}  // namespace

VQM_TEST(TestButterInvalidParameters) {
    using vqm::ButteraugliConfig;
    using vqm::ButteraugliPipeline;
    constexpr float inf = std::numeric_limits<float>::infinity();
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    for (float value : {0.0f, -1.0f, inf, -inf, nan}) {
        ButteraugliConfig config{.size = {32, 32}};
        config.hf_asymmetry = value;
        CHECK_THROWS_AS(ButteraugliPipeline(config), std::invalid_argument);
        config.hf_asymmetry = 1.0f;
        config.intensity_target = value;
        CHECK_THROWS_AS(ButteraugliPipeline(config), std::invalid_argument);
    }
    for (float value : {-1.0f, inf, -inf, nan}) {
        ButteraugliConfig config{.size = {32, 32}};
        config.xmul = value;
        CHECK_THROWS_AS(ButteraugliPipeline(config), std::invalid_argument);
    }
    if (vqm::test::GpuAvailable()) {
        ButteraugliConfig config{.size = {32, 32}};
        config.xmul = 0.0f;
        ButteraugliPipeline pipeline(config);
        const Image image = vqm::test::MakeGradient(0);
        CHECK(pipeline.Compute(image, image).score == 0.0);
    }
}

// The oracle validates itself against libjxl's own published anchor
// (butteraugli_test.cc:60-69: two 1x1 grays 0.01 apart in G, expected 2.5 +/-
// 0.5). Our closed-form evaluation of the constant padded 8x8 case gives
// 2.065718 and the port gives 2.065716, so libjxl's band is loose rather than
// the port being off; the test pins the tighter number so drift is visible.
VQM_TEST(TestButterReferenceAnchor) {
    const float r0 = 0.5f;
    const float g0 = 0.5f;
    const float b0 = 0.5f;
    const float g1 = 0.49f;
    const ButterRefResult result = vqm::test::ButteraugliCpuReference(
        &r0, &g0, &b0, &r0, &g1, &b0, 1, 1, ButterRefParams{}
    );
    CHECK(std::fabs(result.score - 2.5) <= 0.5);
    CHECK_NEAR(result.score, 2.065716, 2e-6);
    CHECK_NEAR(result.distance_p3, 2.065716, 2e-6);
}

VQM_TEST(TestButterIdentical) {
    if (!vqm::test::GpuAvailable()) return;
    const Image image = vqm::test::MakeGradient(0);
    const vqm::ButteraugliScore score = Gpu(image, image);
    // Identical inputs cancel exactly: the maps divide the difference of
    // numerator and denominator, so no rounding survives.
    CHECK(score.score == 0.0);
    CHECK(score.distance_p3 == 0.0);
}

// The resident reference's pixels are immutable, but its decoded colors also
// depend on the compositing background. Changing it must rebuild the chain.
VQM_TEST(TestButterReferenceBackground) {
    if (!vqm::test::GpuAvailable()) return;
    const Image image =
        vqm::test::MakeRampImage(32, 32, true, vqm::PixelType::Byte);
    vqm::ButteraugliPipeline pipeline({.size = image.size});
    const auto resident = pipeline.Upload(image);
    CHECK(pipeline.Compute(resident, resident, 0.1f).score == 0.0);
    CHECK(pipeline.Compute(resident, resident, 0.9f).score == 0.0);
    CHECK(pipeline.Compute(resident, resident, 0.9f).score == 0.0);
}

// Keep the same reference chain alive while changing the distortion, then
// return to the first distortion. This catches scratch storage overwriting the
// cached reference between calls, including on partially filled edge tiles.
VQM_TEST(TestButterCachedReferenceOddGeometry) {
    if (!vqm::test::GpuAvailable()) return;
    const Image original =
        vqm::test::MakeRampImage(129, 97, false, vqm::PixelType::Byte);
    Image a = original;
    Image b = original;
    const auto planes =
        vqm::internal::PlaneLayout(original.format, original.size);
    const std::size_t pixels = original.size.width * original.size.height;
    for (std::size_t i = 0; i < pixels; ++i) {
        if (i % 3 == 0) a.data[planes[0].offset + i] /= 2;
        if (i % 5 == 0) b.data[planes[1].offset + i] /= 3;
    }
    // Fresh streaming pipelines cannot reuse a resident reference chain.
    const auto expected_a = Gpu(original, a);
    const auto expected_b = Gpu(original, b);
    CHECK(expected_a.score > 0.0);
    CHECK(expected_b.score > 0.0);
    CHECK(std::fabs(expected_a.score - expected_b.score) > 1e-5);

    for (auto policy : {vqm::MemoryPolicy::Automatic, vqm::MemoryPolicy::Direct,
                        vqm::MemoryPolicy::Staged}) {
        vqm::ButteraugliConfig config{.size = original.size};
        config.memory_policy = policy;
        config.gpu_resident_images = 3;
        const auto plan = vqm::GetMemoryRequirements(config);
        if (policy == vqm::MemoryPolicy::Direct && !plan.direct.supported)
            continue;
        if (policy == vqm::MemoryPolicy::Staged && !plan.staged.supported)
            continue;
        if (!plan.direct.supported && !plan.staged.supported) continue;
        vqm::ButteraugliPipeline pipeline(config);
        const auto initial = pipeline.GetMemoryUsage();
        const auto& selected = initial.uses_staging ? plan.staged : plan.direct;
        CHECK(selected.supported);
        CHECK(initial.arena.allocation_bytes == selected.arena.allocation_bytes);
        CHECK(
            initial.staging.allocation_bytes == selected.staging.allocation_bytes
        );

        const auto ref = pipeline.Upload(original);
        const auto resident_a = pipeline.Upload(a);
        const auto resident_b = pipeline.Upload(b);
        const auto check = [](const auto& actual, const auto& expected) {
            CHECK_NEAR(actual.score, expected.score, 1e-9);
            CHECK_NEAR(actual.distance_p3, expected.distance_p3, 1e-9);
        };
        check(pipeline.Compute(ref, resident_a), expected_a);
        check(pipeline.Compute(ref, resident_b), expected_b);
        check(pipeline.Compute(ref, resident_a), expected_a);
        check(pipeline.Compute(ref, a), expected_a);
        check(pipeline.Compute(ref, b), expected_b);
        check(pipeline.Compute(ref, a), expected_a);
        CHECK(pipeline.Compute(ref, ref).score == 0.0);

        const auto after = pipeline.GetMemoryUsage();
        CHECK(after.arena.allocation_bytes == initial.arena.allocation_bytes);
        CHECK(
            after.staging.allocation_bytes == initial.staging.allocation_bytes
        );
    }
}

// A MapImage pointer can be retained and written without another MapImage call.
// Its handle generation alone cannot identify the contents of the reference.
VQM_TEST(TestButterMappedReferenceMutation) {
    if (!vqm::test::GpuAvailable()) return;
    const Image image =
        vqm::test::MakeRampImage(32, 32, true, vqm::PixelType::Byte);
    vqm::ButteraugliConfig config;
    config.size = image.size;
    config.input_bytes = image.data.size();
    vqm::ButteraugliPipeline pipeline(config);
    vqm::ImageBuffer buffer;
    try {
        buffer = pipeline.AllocateImage(image.format);
    } catch (const std::runtime_error& error) {
        CHECK(
            std::string_view(error.what()).find("caller-owned image buffers") !=
            std::string_view::npos
        );
        return;
    }
    std::byte* const pixels = pipeline.MapImage(buffer);
    std::memcpy(pixels, image.data.data(), image.data.size());
    CHECK(pipeline.Compute(buffer, buffer).score == 0.0);
    std::memset(pixels, 0, image.size.width * image.size.height);
    CHECK(pipeline.Compute(buffer, buffer).score == 0.0);
}

// The corrected GPU path against the unchanged double-precision oracle, at
// 0.1% of score and distance_p3. On RTX 5070 / driver 615.78.08, gradient_B
// gives GPU 3.77568078 / 2.92297296 against oracle 3.775690714 / 2.922971662;
// gradient_C gives GPU 2.95148849 / 2.35224729 against oracle
// 2.951491534 / 2.352248402. The former percent-scale residual came from
// frequency separation and Malta errors. See docs/divergence.md.
//
// libjxl cross-check (built from .reference/libjxl with highway/lcms/skcms/
// sjpeg, jxl-internal, one driver calling ButteraugliInterface directly): the
// *oracle* reproduces libjxl to 9.1e-6 (3.775690714 against 3.775699854) and
// 2.1e-5 (2.951491534 against 2.951512814) on these two pairs, so the oracle
// is a faithful port. It is the stand-in for the reference wherever the
// libjxl build is not available.
VQM_TEST(TestButterAgainstReference) {
    if (!vqm::test::GpuAvailable()) return;
    struct Case {
        int original;
        int distorted;
    };
    const Case cases[] = {{0, 1}, {0, 2}};
    for (const Case& c : cases) {
        const Image a = vqm::test::MakeGradient(c.original);
        const Image b = vqm::test::MakeGradient(c.distorted);
        const ButterRefResult reference = Reference(a, b);
        const vqm::ButteraugliScore gpu = Gpu(a, b);
        CHECK(reference.score > 0.0);
        const double score_tolerance = 0.001 * reference.score;
        const double p3_tolerance = 0.001 * reference.distance_p3;
        if (std::fabs(gpu.score - reference.score) > score_tolerance ||
            std::fabs(gpu.distance_p3 - reference.distance_p3) > p3_tolerance) {
            vqm::test::Fail(
                std::source_location::current(),
                std::format(
                    "butter pair {} vs {}: gpu score {:.6f} p3 {:.6f}, "
                    "reference {:.6f} / {:.6f} (tolerance 0.1%)",
                    c.original, c.distorted, gpu.score, gpu.distance_p3,
                    reference.score, reference.distance_p3
                )
            );
        }
    }
}

VQM_TEST(TestButterSpatialFrequencyResidual) {
    if (!vqm::test::GpuAvailable()) return;
    CheckSpatialReference({64, 64}, SpatialPattern::Checker);
}

VQM_TEST(TestButterSpatialChromaSuppression) {
    if (!vqm::test::GpuAvailable()) return;
    CheckSpatialReference({65, 47}, SpatialPattern::Chroma);
}

VQM_TEST(TestButterSpatialTransposedOddTiles) {
    if (!vqm::test::GpuAvailable()) return;
    CheckSpatialReference({65, 47}, SpatialPattern::TransposedChecker);
}

// This implementation's own numbers for the fixtures, so a change that moves
// them is deliberate. Measured after correcting frequency separation and Malta;
// see docs/divergence.md for the reference comparison.
VQM_TEST(TestButterGoldens) {
    if (!vqm::test::GpuAvailable()) return;
    const Image a = vqm::test::MakeGradient(0);
    const Image b = vqm::test::MakeGradient(1);
    const vqm::ButteraugliScore score = Gpu(a, b);
    CHECK_NEAR(score.score, 3.77568078, 5e-3);
    CHECK_NEAR(score.distance_p3, 2.92297296, 5e-3);
}

VQM_TEST(TestButterStreamingDisabled) {
    if (!vqm::test::GpuAvailable()) return;
    const Image original = vqm::test::MakeGradient(0);
    const Image distorted = vqm::test::MakeGradient(1);
    const vqm::ButteraugliScore expected = Gpu(original, distorted);

    vqm::ButteraugliConfig config;
    config.size = original.size;
    config.streaming_inputs = false;
    // The fixtures' byte planes are 64-byte aligned, as are arena carve-outs.
    config.input_bytes = original.data.size() + distorted.data.size();
    vqm::ButteraugliPipeline pipeline(config);
    CHECK_THROWS_AS(
        pipeline.Compute(original, distorted), std::invalid_argument
    );

    const auto a = pipeline.Upload(original);
    const auto b = pipeline.Upload(distorted);
    const auto resident = pipeline.Compute(a, b);
    CHECK_NEAR(resident.score, expected.score, 1e-12);
    CHECK_NEAR(resident.distance_p3, expected.distance_p3, 1e-12);
    CHECK(pipeline.Compute(a, a).score == 0.0);

    vqm::ImageBuffer in_a;
    try {
        in_a = pipeline.AllocateImage(original.format);
    } catch (const std::runtime_error& error) {
        // A non-host-visible arena supports the resident path above, but does
        // not support caller-owned buffers.
        CHECK(
            std::string_view(error.what()).find("caller-owned image buffers") !=
            std::string_view::npos
        );
        return;
    }
    const auto in_b = pipeline.AllocateImage(distorted.format);
    std::memcpy(
        pipeline.MapImage(in_a), original.data.data(), original.data.size()
    );
    std::memcpy(
        pipeline.MapImage(in_b), distorted.data.data(), distorted.data.size()
    );
    const auto buffers = pipeline.Compute(in_a, in_b);
    CHECK_NEAR(buffers.score, expected.score, 1e-12);
    CHECK_NEAR(buffers.distance_p3, expected.distance_p3, 1e-12);
    CHECK(pipeline.Compute(in_a, in_a).score == 0.0);
}

VQM_TEST(TestButterOddSubsampleAgainstReference) {
    if (!vqm::test::GpuAvailable()) return;
    // Constant colors isolate the subsample's treatment of a missing tap.
    // Cover odd columns, odd rows and their corner, with an even control.
    for (const Size size :
         std::array{Size{32, 16}, Size{33, 16}, Size{32, 17}, Size{33, 17}}) {
        Image original{size, {}, {}};
        original.format.type = vqm::PixelType::Byte;
        original.data.resize(vqm::RequiredBytes(original.format, size));
        Image distorted = original;
        const auto layout =
            vqm::internal::PlaneLayout(original.format, original.size);
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const auto value = static_cast<std::uint8_t>(64 + 32 * channel);
            std::fill_n(
                original.data.data() + layout[channel].offset,
                size.width * size.height, value
            );
            std::fill_n(
                distorted.data.data() + layout[channel].offset,
                size.width * size.height, value + 10
            );
        }
        const auto expected = Reference(original, distorted);
        const auto actual = Gpu(original, distorted);
        CHECK_NEAR(actual.score, expected.score, 0.05 * expected.score);
        CHECK_NEAR(
            actual.distance_p3, expected.distance_p3, 0.05 * expected.distance_p3
        );
    }
}

// 8x8 runs one level (its 4x4 subsample is below Butteraugli's 8x8 minimum),
// 15x15 is the boundary where the second level appears, and 16x16 folds.
VQM_TEST(TestButterSmallGeometries) {
    if (!vqm::test::GpuAvailable()) return;
    const std::size_t sides[] = {8, 15, 16, 17};
    for (const std::size_t side : sides) {
        Image a = vqm::test::MakeRampImage(side, side, false);
        Image b = a;
        // Perturb one pixel so the metric has something to measure.
        const float changed = 0.5f;
        std::memcpy(b.data.data(), &changed, sizeof(changed));
        const vqm::ButteraugliScore score = Gpu(a, b);
        CHECK(std::isfinite(score.score));
        CHECK(score.score > 0.0);
        const vqm::ButteraugliScore same = Gpu(a, a);
        CHECK(same.score == 0.0);
    }
}
