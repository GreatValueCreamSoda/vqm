// Independent fixtures generated with original ColorVideoVDP v0.5.7.
#include "support.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string_view>

#include "cvvdp.h"
#include "metrics/cvvdp/cvvdp_internal.h"
#include "vqm_views.h"

#include "cvvdp_reference.inc"

namespace vqm::test {
namespace {
float Sample(
    std::size_t x, std::size_t y, std::size_t ch, std::size_t frame,
    int artifact, bool distorted
) {
    if (artifact == 2) return distorted ? .5001f : .5f;
    if (artifact == 3) return distorted ? 1.0f : 0.0f;
    if (artifact == 5) return ch == (distorted ? 0u : 1u) ? 1.0f : 0.0f;
    if (artifact == 6) {
        const std::size_t active = distorted ? (frame % 2 ? 2u : 0u) : 1u;
        return ch == active ? 1.0f : 0.0f;
    }
    int code = 20 + static_cast<int>(
                        (x * (3 + 2 * ch) + y * (5 + 2 * ch) +
                         frame * (11 + 4 * ch) + ch * 37) %
                        211
                    );
    if (distorted && artifact == 0) {
        code = ((code + 3) / 8) * 8;
        if ((x / 7 + y / 5 + frame + ch) % 3 == 0) code += 5;
    }
    float value = static_cast<float>(std::clamp(code, 0, 255)) / 255.0f;
    if (distorted && artifact == 4)
        value = std::clamp(value + (frame % 2 ? .08f : -.08f), 0.0f, 1.0f);
    return value;
}
Image Input(
    const CvvdpReferenceCase& fixture, std::size_t frame, bool distorted
) {
    ImageFormat format;
    const std::string_view colorspace(fixture.color_space);
    if (colorspace == "BT.2020-PQ" || colorspace == "BT.2020-HLG") {
        format.primaries = ColorPrimaries::Bt2020;
        format.transfer = colorspace == "BT.2020-PQ"
                              ? TransferCharacteristics::St2084
                              : TransferCharacteristics::AribB67;
    }
    if (colorspace == "Display P3 Apple")
        format.primaries = ColorPrimaries::St432_1;
    const bool linear = colorspace == "BT.709-linear";
    if (linear) format.transfer = TransferCharacteristics::Linear;
    std::array<std::vector<float>, 4> values;
    for (std::size_t ch = 0; ch < 3; ++ch) {
        for (std::size_t y = 0; y < fixture.height; ++y) {
            for (std::size_t x = 0; x < fixture.width; ++x)
                values[ch].push_back(
                    Sample(x, y, ch, frame, fixture.artifact, distorted) *
                    (linear ? 1000.0f : 1.0f)
                );
        }
    }
    return PackPlanes(format, {fixture.width, fixture.height}, values);
}
CvvdpConfig Config(const CvvdpReferenceCase& fixture) {
    CvvdpConfig config;
    config.size = {fixture.width, fixture.height};
    const std::string_view display(fixture.display);
    if (display == "standard_fhd") config.display = CvvdpDisplay::StandardFhd();
    if (display.starts_with("standard_hdr"))
        config.display = CvvdpDisplay::StandardHdr();
    config.temporal = fixture.fps_num != 0;
    if (config.temporal) {
        config.fps_numerator = fixture.fps_num;
        config.fps_denominator = fixture.fps_den;
    }
    return config;
}

template <std::size_t N>
void CheckFilters(
    const std::array<double, N>& expected, std::uint32_t num,
    std::uint32_t den = 1
) {
    CvvdpConfig config;
    config.fps_numerator = num;
    config.fps_denominator = den;
    const auto actual = internal::CvvdpTemporalFilters(config);
    CHECK(actual.size() == N);
    for (std::size_t i = 0; i < N; ++i) CHECK_NEAR(actual[i], expected[i], 2e-7);
}
}  // namespace

VQM_TEST(TestCvvdpTemporalCoefficients) {
    CheckFilters(kCvvdpFilters24_1, 24);
    CheckFilters(kCvvdpFilters30_1, 30);
    CheckFilters(kCvvdpFilters60_1, 60);
    CheckFilters(kCvvdpFilters120_1, 120);
    CheckFilters(kCvvdpFilters24000_1001, 24000, 1001);
    CheckFilters(kCvvdpFilters30000_1001, 30000, 1001);
    CheckFilters(kCvvdpFilters60000_1001, 60000, 1001);
}
VQM_TEST(TestCvvdpModelSemantics) {
    CHECK(internal::CvvdpJod(0) == 10.0);
    CHECK(internal::CvvdpSafePow(0, .25) == 0.0);
    CHECK(internal::CvvdpJod(1e6) < 0);
    CHECK_NEAR(
        internal::CvvdpJod(.1 - 1e-9), internal::CvvdpJod(.1 + 1e-9), 1e-8
    );
    CvvdpConfig config;
    config.fps_denominator = 0;
    CHECK_THROWS_AS(internal::CvvdpTemporalTaps(config), std::invalid_argument);
}
VQM_TEST(TestCvvdpOriginalOracle) {
    if (!GpuAvailable()) return;
    for (const auto& fixture : kCvvdpReferences) {
        CvvdpPipeline pipeline(Config(fixture));
        if (fixture.fps_num == 0) {
            CHECK_NEAR(
                pipeline
                    .ComputeImage(
                        Input(fixture, 0, false), Input(fixture, 0, true)
                    )
                    .score,
                fixture.score, 1e-3
            );
            continue;
        }
        for (std::size_t frame = 0; frame < fixture.frames; ++frame)
            pipeline.Push(
                Input(fixture, frame, false), Input(fixture, frame, true)
            );
        const auto score = pipeline.Finalize();
        CHECK(score.has_value());
        CHECK(score->frames == fixture.frames);
        CHECK_NEAR(score->score, fixture.score, 1e-3);
        CHECK(pipeline.Finalize()->score == score->score);
    }
}
VQM_TEST(TestCvvdpLifecycleAndMemory) {
    if (!GpuAvailable()) return;
    CvvdpConfig config;
    config.size = {32, 32};
    const auto planned = GetMemoryRequirements(config);
    CvvdpPipeline pipeline(config);
    const auto actual = pipeline.GetMemoryUsage();
    const auto& path = actual.uses_staging ? planned.staged : planned.direct;
    CHECK(path.supported);
    CHECK(path.arena.allocation_bytes == actual.arena.allocation_bytes);
    CHECK(path.auxiliary.size() == 2);
    CHECK(actual.auxiliary.size() == 2);
    CHECK(
        path.auxiliary[0].allocation_bytes ==
        actual.auxiliary[0].allocation_bytes
    );
    const auto a = MakeGradient(0), b = MakeGradient(1);
    const auto original = pipeline.Upload(a), distorted = pipeline.Upload(b);
    const double still = pipeline.ComputeImage(a, b).score;
    CHECK_NEAR(pipeline.ComputeImage(original, distorted).score, still, 1e-7);
    CHECK(!pipeline.Finalize());
    CHECK_THROWS_AS(pipeline.Push(a, b), std::logic_error);
    pipeline.Reset();
    pipeline.Push(original, distorted);
    CHECK_THROWS_AS(pipeline.ComputeImage(a, b), std::logic_error);
    auto bad = b;
    bad.data.clear();
    CHECK_THROWS_AS(pipeline.Push(a, bad), std::invalid_argument);
    CHECK_NEAR(pipeline.Finalize()->score, still, 1e-7);
    pipeline.Reset();
    CHECK(pipeline.ComputeImage(original, original).score == 10.0);
    pipeline.Release(original);
    CHECK_THROWS_AS(
        pipeline.ComputeImage(original, distorted), std::invalid_argument
    );
    auto invalid = a;
    invalid.format = PinnedFormat(AlphaType::None);
    invalid.data.resize(RequiredBytes(invalid.format, invalid.size));
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(invalid.data.data(), &nan, sizeof(nan));
    CHECK_THROWS_AS(pipeline.ComputeImage(invalid, b), std::invalid_argument);
    CHECK_THROWS_AS(pipeline.ComputeImage(a, b), std::logic_error);
    pipeline.Reset();
    CHECK(pipeline.ComputeImage(a, a).score == 10.0);
    config.temporal = false;
    CvvdpPipeline image_only(config);
    CHECK(image_only.GetMemoryUsage().auxiliary.empty());
    CHECK_THROWS_AS(image_only.Push(a, b), std::logic_error);
    auto alpha = MakeRampImage(32, 32, true);
    CHECK_THROWS_AS(image_only.ComputeImage(alpha, b), std::invalid_argument);
}
}  // namespace vqm::test
