// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include "input_format_test_support.h"

using namespace vqm::test::input_format;

namespace {

// Deterministic content on a `grid`-level ladder, so a sample is exactly
// representable in a sample type of log2(grid) + 1 bits or more.
inline std::array<std::vector<float>, 4> Ladder(int grid, int variant) {
    std::array<std::vector<float>, 4> values;
    const double top = static_cast<double>(grid);
    for (std::size_t y = 0; y < kSide; ++y) {
        for (std::size_t x = 0; x < kSide; ++x) {
            const int r =
                static_cast<int>((x * 7) % static_cast<std::size_t>(grid + 1));
            const int g =
                static_cast<int>((y * 13) % static_cast<std::size_t>(grid + 1));
            const int b = static_cast<int>(
                (x * 11 + y * 5) % static_cast<std::size_t>(grid + 1)
            );
            const int delta = variant == 0 ? 0 : (variant == 1 ? 1 : -1);
            const auto clamp = [grid](int v) { return std::clamp(v, 0, grid); };
            values[0].push_back(static_cast<float>(clamp(r + delta) / top));
            values[1].push_back(static_cast<float>(clamp(g + 2 * delta) / top));
            values[2].push_back(static_cast<float>(clamp(b + 3 * delta) / top));
        }
    }
    return values;
}

// Per-pixel content: a diagonal ramp, so every A/B case compares structured
// images. Uniform frames are near-degenerate for this metric -- their error
// norms saturate at ~1e-4 no matter how accurate the conversion is, which
// makes them useless for catching a 1e-3 conversion error.
template <typename F>
std::array<std::vector<float>, 4> Pixels(F f) {
    std::array<std::vector<float>, 4> values;
    for (std::size_t y = 0; y < kSide; ++y) {
        for (std::size_t x = 0; x < kSide; ++x) {
            const ref::Vec3 v = f(x, y);
            values[0].push_back(static_cast<float>(v[0]));
            values[1].push_back(static_cast<float>(v[1]));
            values[2].push_back(static_cast<float>(v[2]));
        }
    }
    return values;
}

// The ramp every A/B case sweeps, in [0.02, 0.98].
inline double Ramp(std::size_t x, std::size_t y) {
    const double t =
        static_cast<double>(x + y) / (2.0 * static_cast<double>(kSide - 1));
    return 0.02 + 0.96 * t;
}

}  // namespace

// --- datatypes, depth and range --------------------------------------------

VQM_TEST(TestHalfConversionBoundaries) {
    // Binary16 subnormals are integer multiples of 2^-24; this expectation
    // does not use either conversion helper to calculate the other one's result.
    CHECK_NEAR(vqm::test::HalfToFloat(0x0001), std::ldexp(1.0, -24), 0.0);
    CHECK_NEAR(vqm::test::HalfToFloat(0x0200), std::ldexp(512.0, -24), 0.0);
    CHECK_NEAR(vqm::test::HalfToFloat(0x03ff), std::ldexp(1023.0, -24), 0.0);
    CHECK_NEAR(vqm::test::HalfToFloat(0x0400), std::ldexp(1.0, -14), 0.0);
    CHECK_NEAR(vqm::test::HalfToFloat(0x8001), -std::ldexp(1.0, -24), 0.0);
    CHECK(std::signbit(vqm::test::HalfToFloat(0x8000)));
    CHECK(vqm::test::FloatToHalf(std::ldexp(1.0f, -24)) == 0x0001);
    CHECK(vqm::test::FloatToHalf(std::ldexp(1023.0f, -24)) == 0x03ff);
    CHECK(vqm::test::FloatToHalf(std::ldexp(1.0f, -14)) == 0x0400);
}

VQM_TEST(TestImageFormatEnumValidation) {
    const ImageFormat base;
    const auto rejects = [](const ImageFormat& format) {
        CHECK_THROWS_AS(
            vqm::internal::ValidateImage(format, {32, 32}, "image"),
            std::invalid_argument
        );
    };
    auto format = base;
    format.type = static_cast<PixelType>(255);
    rejects(format);
    format = base;
    format.family = static_cast<ColorFamily>(255);
    rejects(format);
    format = base;
    format.alpha = static_cast<AlphaType>(255);
    rejects(format);
    format = base;
    format.range = static_cast<PixelRange>(255);
    rejects(format);
    format.type = PixelType::Byte;
    format.depth = 7;
    rejects(format);
    CHECK_THROWS_AS(
        vqm::internal::PlaneScaleOffset(format, false, false),
        std::invalid_argument
    );
    format = base;
    format.chroma_location = static_cast<ChromaLocation>(255);
    rejects(format);
    format = base;
    format.transfer = static_cast<TransferCharacteristics>(255);
    rejects(format);
    format = base;
    format.primaries = static_cast<ColorPrimaries>(255);
    rejects(format);
    format = base;
    format.matrix = static_cast<MatrixCoefficients>(255);
    rejects(format);
    // Meaningless grayscale metadata may be unspecified, but must still be
    // a known enum value before it reaches the descriptor packing code.
    format.family = ColorFamily::Grey;
    rejects(format);
}

VQM_TEST(TestRequiredBytesBoundaries) {
    ImageFormat format = vqm::test::PinnedFormat(AlphaType::None);
    format.type = PixelType::Byte;
    CHECK(vqm::RequiredBytes(format, {3, 1}) == 12);
    format.type = PixelType::Word;
    CHECK(vqm::RequiredBytes(format, {3, 1}) == 24);

    format.type = PixelType::Byte;
    format.family = ColorFamily::Yuv;
    format.subsample_w = 1;
    format.subsample_h = 1;
    CHECK(vqm::RequiredBytes(format, {8, 4}) == 48);
    format.alpha = AlphaType::Straight;
    CHECK(vqm::RequiredBytes(format, {8, 4}) == 80);

    for (const std::uint8_t shift :
         {std::uint8_t{3}, std::uint8_t{64}, std::uint8_t{255}}) {
        format.subsample_w = shift;
        CHECK_THROWS_AS(
            (void)vqm::RequiredBytes(format, {8, 8}), std::invalid_argument
        );
        format.subsample_w = 0;
        format.subsample_h = shift;
        CHECK_THROWS_AS(
            (void)vqm::RequiredBytes(format, {8, 8}), std::invalid_argument
        );
        format.subsample_h = 0;
    }

    format.type = static_cast<PixelType>(255);
    CHECK_THROWS_AS(
        (void)vqm::RequiredBytes(format, {8, 8}), std::invalid_argument
    );

    const std::size_t max = std::numeric_limits<std::size_t>::max();
    format.type = PixelType::Byte;
    format.family = ColorFamily::Grey;
    format.alpha = AlphaType::None;
    CHECK(vqm::RequiredBytes(format, {max - 3, 1}) == max - 3);
    CHECK_THROWS_AS(
        (void)vqm::RequiredBytes(format, {max - 2, 1}), std::overflow_error
    );
    CHECK_THROWS_AS(
        (void)vqm::RequiredBytes(format, {max, 2}), std::overflow_error
    );
    format.type = PixelType::Float;
    CHECK_THROWS_AS(
        (void)vqm::RequiredBytes(format, {max / 2, 1}), std::overflow_error
    );
    format.type = PixelType::Byte;
    format.family = ColorFamily::Rgb;
    format.alpha = AlphaType::Straight;
    CHECK_THROWS_AS(
        (void)vqm::RequiredBytes(format, {max / 4, 1}), std::overflow_error
    );
}

VQM_TEST(TestScaleOffsetValidation) {
    ImageFormat format = vqm::test::PinnedFormat(AlphaType::None);
    format.type = PixelType::Byte;
    format.depth = 8;
    format.range = PixelRange::Limited;
    const auto luma = vqm::internal::PlaneScaleOffset(format, false, false);
    const auto chroma = vqm::internal::PlaneScaleOffset(format, true, false);
    CHECK(luma.scale == static_cast<float>(1.0 / 219.0));
    CHECK(luma.offset == static_cast<float>(-16.0 / 219.0));
    CHECK(chroma.scale == static_cast<float>(1.0 / 224.0));
    CHECK(chroma.offset == static_cast<float>(-128.0 / 224.0));

    format.depth = 7;
    CHECK_THROWS_AS(
        (void)vqm::internal::PlaneScaleOffset(format, false, false),
        std::invalid_argument
    );
    CHECK_THROWS_AS(
        (void)vqm::internal::PlaneScaleOffset(format, true, false),
        std::invalid_argument
    );
    format.range = PixelRange::Full;
    format.depth = 255;
    CHECK_THROWS_AS(
        (void)vqm::internal::PlaneScaleOffset(format, false, false),
        std::invalid_argument
    );
}

VQM_TEST(TestFloatHalfWordAgree) {
    if (!vqm::test::GpuAvailable()) return;
    Ssimulacra2Pipeline pipeline = MakePipeline();
    const ImageFormat base = vqm::test::PinnedFormat(AlphaType::None);

    // A 1024-level ladder is exact in binary16, so Half must reproduce Float
    // bit for bit.
    ImageFormat half = base;
    half.type = PixelType::Half;
    const auto half_values = Ladder(1024, 0);
    CheckAgrees(
        pipeline, Format(half, half_values), Format(base, half_values),
        kRounding, "half vs float"
    );

    // 16-bit integers: the ladder steps are exactly representable as codes, so
    // only the decode's reciprocal rounding separates them from Float.
    ImageFormat word = base;
    word.type = PixelType::Word;
    word.depth = 16;
    const auto word_values = Ladder(65535, 0);
    CheckAgrees(
        pipeline, Format(word, word_values), Format(base, word_values),
        kRounding, "word16 vs float"
    );
}

VQM_TEST(TestIntegerQuantisationAgrees) {
    if (!vqm::test::GpuAvailable()) return;
    Ssimulacra2Pipeline pipeline = MakePipeline();
    const ImageFormat base = vqm::test::PinnedFormat(AlphaType::None);

    // 8-bit full range: 255 levels against the float reference.
    ImageFormat byte = base;
    byte.type = PixelType::Byte;
    const auto byte_values = Ladder(255, 0);
    CheckAgrees(
        pipeline, Format(byte, byte_values), Format(base, byte_values),
        kRounding, "byte vs float"
    );

    // 10-bit, full range.
    ImageFormat word10 = base;
    word10.type = PixelType::Word;
    word10.depth = 10;
    const auto word10_values = Ladder(1023, 0);
    CheckAgrees(
        pipeline, Format(word10, word10_values), Format(base, word10_values),
        kRounding, "word10 vs float"
    );

    // Limited range against full range at the same normalised content: this
    // one is no longer a rounding comparison, the two decimations really do
    // differ by up to 1/(2*219) of a code step.
    ImageFormat limited = byte;
    limited.range = PixelRange::Limited;
    const Image full_frame = Format(byte, byte_values);
    const Image limited_frame = Format(limited, byte_values);
    // Not a rounding comparison: the two decimations differ by up to half a
    // limited-range code step (1/438 of full scale), which the metric reports
    // at ~4e-3. A wrong limited-range mapping is off by orders of magnitude.
    const double limited_delta = MaxNormDelta(
        pipeline.Compute(limited_frame, full_frame),
        pipeline.Compute(full_frame, full_frame)
    );
    if (!(limited_delta <= 1e-2)) {
        vqm::test::Fail(
            std::source_location::current(),
            std::format("limited vs full: {:.3e}", limited_delta)
        );
    }
}

// --- transfer functions ----------------------------------------------------

VQM_TEST(TestTransferFunctions) {
    if (!vqm::test::GpuAvailable()) return;
    Ssimulacra2Pipeline pipeline = MakePipeline();
    const ImageFormat base = vqm::test::PinnedFormat(AlphaType::None);

    struct Case {
        TransferCharacteristics transfer;
        bool upper;  // sweep the top half of the curve instead of the bottom
    };
    // Every curve is swept across half its domain; the log and PQ curves are
    // steepest in the top half, so those get both halves.
    const Case cases[] = {
        {TransferCharacteristics::Bt709, false},
        {TransferCharacteristics::Bt601, false},
        {TransferCharacteristics::Bt2020_10, false},
        {TransferCharacteristics::Bt2020_12, false},
        {TransferCharacteristics::St240M, false},
        {TransferCharacteristics::Bt470M, false},
        {TransferCharacteristics::Bt470Bg, false},
        {TransferCharacteristics::Srgb, false},
        {TransferCharacteristics::Log100, true},
        {TransferCharacteristics::Log316, true},
        {TransferCharacteristics::Iec61966_2_4, true},
        {TransferCharacteristics::Bt1361, true},
        {TransferCharacteristics::St428, false},
        {TransferCharacteristics::Linear, false},
        {TransferCharacteristics::St2084, true},
        {TransferCharacteristics::St2084, false},
        {TransferCharacteristics::AribB67, true},
        {TransferCharacteristics::AribB67, false},
    };
    // The ramp mapped into the half of the curve the case sweeps.
    const auto sample = [](const Case& c, std::size_t x, std::size_t y) {
        const double v = Ramp(x, y);
        return c.upper ? 0.5 + 0.5 * v : 0.5 * v;
    };
    for (const Case& c : cases) {
        ImageFormat format = base;
        format.transfer = c.transfer;
        // Sweep the curve: the sample rides the ramp, so every part of the
        // transfer function contributes to the comparison.
        const Image a = Format(format, Pixels([&](std::size_t x, std::size_t y) {
                                   const double v = sample(c, x, y);
                                   return ref::Vec3{v, v, v};
                               }));
        const Image b =
            Format(LinearRgbFormat(), Pixels([&](std::size_t x, std::size_t y) {
                       const double v = sample(c, x, y);
                       // HLG's display-referred decode is the luminance OOTF,
                       // which for neutral input reduces to pow(v, 1.2) times
                       // 1000/peak.
                       const double linear =
                           c.transfer == TransferCharacteristics::AribB67
                               ? ref::HlgToLinear(
                                     {v, v, v}, 0.2126, 0.7152, 0.0722, 10.0
                                 )[0]
                               : ref::DecodeTransfer(c.transfer, v);
                       return ref::Vec3{linear, linear, linear};
                   }));
        const double tolerance =
            c.transfer == TransferCharacteristics::St2084
                ? kPqCurve
                : (c.transfer == TransferCharacteristics::AribB67 ? kHlgCurve
                                                                  : kRounding);
        CheckAgrees(
            pipeline, a, b, tolerance,
            std::format("transfer {}", static_cast<int>(c.transfer)).c_str()
        );
    }

    // A chromatic triple pins the luminance OOTF: zimg scales the whole triple
    // by pow(luma, 0.2), which differs from a per-channel 1.2 power for every
    // non-neutral pixel, and neutral content cannot tell the two apart.
    {
        const ref::Vec3 hlg{0.4, 0.55, 0.7};
        ImageFormat format = base;
        format.transfer = TransferCharacteristics::AribB67;
        const Image a = Format(format, Uniform(hlg));
        // BT.709 primaries supply the OOTF weights.
        const Image b = Format(
            LinearRgbFormat(),
            Uniform(ref::HlgToLinear(hlg, 0.2126, 0.7152, 0.0722, 10.0))
        );
        CheckAgrees(pipeline, a, b, kHlgCurve, "transfer HLG chromatic");
    }

    // The sRGB path keeps this implementation's IEC constants.
    const double expected =
        0.4 <= 0.04045 ? 0.4 / 12.92 : std::pow((0.4 + 0.055) / 1.055, 2.4);
    CHECK_NEAR(
        ref::DecodeTransfer(TransferCharacteristics::Srgb, 0.4), expected, 1e-6
    );
}

// --- matrix coefficients ---------------------------------------------------

VQM_TEST(TestMatrixCoefficients) {
    if (!vqm::test::GpuAvailable()) return;
    Ssimulacra2Pipeline pipeline = MakePipeline();
    struct Case {
        MatrixCoefficients matrix;
        ColorPrimaries primaries;
        bool derived;
    };
    const Case cases[] = {
        {MatrixCoefficients::Bt709, ColorPrimaries::Bt709, false},
        {MatrixCoefficients::Bt470Bg, ColorPrimaries::Bt709, false},
        {MatrixCoefficients::St170M, ColorPrimaries::Bt709, false},
        {MatrixCoefficients::St240M, ColorPrimaries::Bt709, false},
        {MatrixCoefficients::Fcc, ColorPrimaries::Bt709, false},
        {MatrixCoefficients::YCgCo, ColorPrimaries::Bt709, false},
        {MatrixCoefficients::Bt2020Ncl, ColorPrimaries::Bt2020, false},
        {MatrixCoefficients::Bt2020Ncl, ColorPrimaries::Bt470Bg, false},
        {MatrixCoefficients::ChromaticityDerivedNcl, ColorPrimaries::Bt2020,
         true},
        {MatrixCoefficients::ChromaticityDerivedNcl, ColorPrimaries::Bt470Bg,
         true},
        {MatrixCoefficients::ChromaticityDerivedNcl, ColorPrimaries::Ebu3213E,
         true},
    };
    for (const Case& c : cases) {
        ImageFormat format;
        format.type = PixelType::Float;
        format.family = ColorFamily::Yuv;
        format.matrix = c.matrix;
        format.transfer = TransferCharacteristics::Linear;
        format.primaries = c.primaries;
        const auto plane = [](std::size_t x, std::size_t y) {
            const double t = Ramp(x, y);
            return ref::Vec3{0.1 + 0.8 * t, 0.25 - 0.4 * t, -0.25 + 0.5 * t};
        };
        const Image a = Format(format, Pixels(plane));
        const ref::Mat3 ncl = c.derived ? ref::NclToRgbFromPrimaries(c.primaries)
                                        : ref::NclToRgb(c.matrix);
        const ref::Mat3 to_target =
            ref::Gamut(c.primaries, ColorPrimaries::Bt709);
        const Image b =
            Format(LinearRgbFormat(), Pixels([&](std::size_t x, std::size_t y) {
                       return ref::Mul(to_target, ref::Mul(ncl, plane(x, y)));
                   }));
        const std::string label = std::format(
            "matrix {} primaries {}", static_cast<int>(c.matrix),
            static_cast<int>(c.primaries)
        );
        CheckAgrees(pipeline, a, b, kRounding, label.c_str());
    }
}

// --- primaries -------------------------------------------------------------

VQM_TEST(TestPrimaries) {
    if (!vqm::test::GpuAvailable()) return;
    Ssimulacra2Pipeline pipeline = MakePipeline();
    const ImageFormat linear = LinearRgbFormat();

    const ColorPrimaries cases[] = {
        ColorPrimaries::Bt470M,  ColorPrimaries::Bt470Bg,
        ColorPrimaries::St170M,  ColorPrimaries::St240M,
        ColorPrimaries::Film,    ColorPrimaries::Bt2020,
        ColorPrimaries::St428,   ColorPrimaries::St431_2,
        ColorPrimaries::St432_1, ColorPrimaries::Ebu3213E,
    };
    for (ColorPrimaries primaries : cases) {
        ImageFormat format = linear;
        format.primaries = primaries;
        const auto sample = [](std::size_t x, std::size_t y) {
            const double t = Ramp(x, y);
            return ref::Vec3{0.1 + 0.8 * t, 0.9 - 0.8 * t, 0.05 + 0.5 * t};
        };
        const ref::Mat3 to_target = ref::Gamut(primaries, ColorPrimaries::Bt709);
        const Image a = Format(format, Pixels(sample));
        const Image b = Format(linear, Pixels([&](std::size_t x, std::size_t y) {
                                   return ref::Mul(to_target, sample(x, y));
                               }));
        CheckAgrees(
            pipeline, a, b, kRounding,
            std::format("primaries {}", static_cast<int>(primaries)).c_str()
        );
    }
}
