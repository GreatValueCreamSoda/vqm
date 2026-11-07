// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include "input_format_test_support.h"

using namespace vqm::test::input_format;

// --- subsampling and chroma filters ---------------------------------------

namespace {

// 8-bit-exact Y ramp with a sharp chroma edge, so the upsampling kernel is
// visible in the result.
std::array<std::vector<float>, 4> SubsampledContent(
    unsigned subsample_w, unsigned subsample_h
) {
    std::array<std::vector<float>, 4> values;
    const std::size_t chroma_w = kSide >> subsample_w;
    const std::size_t chroma_h = kSide >> subsample_h;
    for (std::size_t y = 0; y < kSide; ++y) {
        for (std::size_t x = 0; x < kSide; ++x) {
            values[0].push_back(
                static_cast<float>(x) / static_cast<float>(kSide - 1)
            );
        }
    }
    for (std::size_t y = 0; y < chroma_h; ++y) {
        for (std::size_t x = 0; x < chroma_w; ++x) {
            const double u = x < chroma_w / 2 ? 0.2 : 0.8;
            const double v = y < chroma_h / 2 ? 0.7 : 0.3;
            values[1].push_back(static_cast<float>(u));
            values[2].push_back(static_cast<float>(v));
        }
    }
    return values;
}

// The reference 4:4:4 chroma: the same planes upsampled with `filter`.
std::array<std::vector<float>, 4> UpsampledContent(
    unsigned subsample_w, unsigned subsample_h, ChromaFilter filter,
    ChromaLocation location
) {
    std::array<std::vector<float>, 4> values =
        SubsampledContent(subsample_w, subsample_h);
    const std::size_t chroma_w = kSide >> subsample_w;
    const std::size_t chroma_h = kSide >> subsample_h;
    // Keep the chroma source separate: the destination planes are full size
    // and would otherwise alias the samples still being read.
    const std::vector<float> source_u = values[1];
    const std::vector<float> source_v = values[2];
    values[1].assign(kSide * kSide, 0.0f);
    values[2].assign(kSide * kSide, 0.0f);

    const double shift_x = ref::ChromaShift(location, true, subsample_w);
    const double shift_y = ref::ChromaShift(location, false, subsample_h);
    const double scale_x = static_cast<double>(1u << subsample_w);
    const double scale_y = static_cast<double>(1u << subsample_h);
    for (std::size_t y = 0; y < kSide; ++y) {
        const auto ty = ref::AxisTaps(filter, y, scale_y, shift_y, chroma_h);
        for (std::size_t x = 0; x < kSide; ++x) {
            const auto tx = ref::AxisTaps(filter, x, scale_x, shift_x, chroma_w);
            double u = 0.0;
            double v = 0.0;
            for (const auto& [iy, wy] : ty) {
                for (const auto& [ix, wx] : tx) {
                    const std::size_t index = iy * chroma_w + ix;
                    u += wx * wy * static_cast<double>(source_u[index]);
                    v += wx * wy * static_cast<double>(source_v[index]);
                }
            }
            values[1][y * kSide + x] = static_cast<float>(u);
            values[2][y * kSide + x] = static_cast<float>(v);
        }
    }
    return values;
}

ImageFormat YuvFormat(
    unsigned subsample_w, unsigned subsample_h, ChromaLocation location
) {
    ImageFormat format;
    format.type = PixelType::Float;
    format.family = ColorFamily::Yuv;
    format.matrix = MatrixCoefficients::Bt709;
    format.transfer = TransferCharacteristics::Linear;
    format.primaries = ColorPrimaries::Bt709;
    format.subsample_w = static_cast<std::uint8_t>(subsample_w);
    format.subsample_h = static_cast<std::uint8_t>(subsample_h);
    format.chroma_location = location;
    return format;
}

Ssimulacra2Pipeline MakeFilterPipeline(ChromaFilter filter) {
    vqm::Ssimulacra2Config config;
    config.size = Side();
    config.chroma_filter = filter;
    return Ssimulacra2Pipeline(config);
}

}  // namespace

// --- the absolute code mapping ---------------------------------------------

VQM_TEST(TestFullRangeByteChroma) {
    if (!vqm::test::GpuAvailable()) return;
    Ssimulacra2Pipeline pipeline = MakePipeline();
    // Build the frame from *codes*, not from decoded values: PackPlanes inverts
    // whatever scale/offset the implementation uses, so it cannot check the
    // absolute mapping. zimg's integer_offset centres a chroma plane on the
    // code midpoint in both ranges, so 8-bit full-range U=128 is 0.0.
    ImageFormat byte_format = YuvFormat(0, 0, ChromaLocation::Left);
    byte_format.type = PixelType::Byte;
    Image coded;
    coded.size = Side();
    coded.format = byte_format;
    coded.data.assign(vqm::RequiredBytes(byte_format, coded.size), 0u);
    const std::array<vqm::internal::PlaneGeometry, 4> layout =
        vqm::internal::PlaneLayout(byte_format, coded.size);
    std::fill(
        coded.data.begin() + static_cast<std::ptrdiff_t>(layout[0].offset),
        coded.data.begin() +
            static_cast<std::ptrdiff_t>(layout[0].offset + kSide * kSide),
        static_cast<std::uint8_t>(128)
    );  // Y = 128 -> 128/255
    std::fill(
        coded.data.begin() + static_cast<std::ptrdiff_t>(layout[1].offset),
        coded.data.begin() +
            static_cast<std::ptrdiff_t>(layout[1].offset + kSide * kSide),
        static_cast<std::uint8_t>(128)
    );  // U = 128 -> midpoint -> 0.0
    std::fill(
        coded.data.begin() + static_cast<std::ptrdiff_t>(layout[2].offset),
        coded.data.begin() +
            static_cast<std::ptrdiff_t>(layout[2].offset + kSide * kSide),
        static_cast<std::uint8_t>(255)
    );  // V = 255 -> +0.498
    const Image reference = Format(
        YuvFormat(0, 0, ChromaLocation::Left),
        Uniform({128.0 / 255.0, 0.0, 127.0 / 255.0})
    );
    CheckAgrees(pipeline, coded, reference, kRounding, "full-range byte chroma");
}

VQM_TEST(TestSubsamplingMatchesReference) {
    if (!vqm::test::GpuAvailable()) return;

    struct Case {
        unsigned sub_w;
        unsigned sub_h;
        ChromaFilter filter;
        ChromaLocation location;
    };
    const Case cases[] = {
        {1, 1, ChromaFilter::Bilinear, ChromaLocation::Left},
        {1, 1, ChromaFilter::Bicubic, ChromaLocation::Left},
        {1, 1, ChromaFilter::Lanczos, ChromaLocation::Left},
        {1, 0, ChromaFilter::Bilinear, ChromaLocation::Left},
        {2, 0, ChromaFilter::Bilinear, ChromaLocation::Left},
        // Every siting the API accepts, so a sign error on TOP/BOTTOM cannot
        // pass: the reference derives the phase independently.
        {1, 1, ChromaFilter::Bilinear, ChromaLocation::Center},
        {1, 1, ChromaFilter::Bilinear, ChromaLocation::TopLeft},
        {1, 1, ChromaFilter::Bilinear, ChromaLocation::Top},
        {1, 1, ChromaFilter::Bilinear, ChromaLocation::BottomLeft},
        {1, 1, ChromaFilter::Bilinear, ChromaLocation::Bottom},
        {1, 1, ChromaFilter::Bicubic, ChromaLocation::Bottom},
    };
    for (const Case& c : cases) {
        Ssimulacra2Pipeline pipeline = MakeFilterPipeline(c.filter);
        const ChromaLocation location = c.location;
        const Image a = Format(
            YuvFormat(c.sub_w, c.sub_h, location),
            SubsampledContent(c.sub_w, c.sub_h)
        );
        const Image b = Format(
            YuvFormat(0, 0, location),
            UpsampledContent(c.sub_w, c.sub_h, c.filter, location)
        );
        CheckAgrees(
            pipeline, a, b, kRounding,
            std::format(
                "subsampling {}x{} filter {} siting {}", c.sub_w, c.sub_h,
                static_cast<int>(c.filter), static_cast<int>(c.location)
            )
                .c_str()
        );
    }
}

VQM_TEST(TestChromaFilterAndLocationReachTheShader) {
    if (!vqm::test::GpuAvailable()) return;
    const ImageFormat format = YuvFormat(1, 1, ChromaLocation::Left);
    const Image content = Format(format, SubsampledContent(1, 1));
    const Image reference =
        Format(YuvFormat(1, 1, ChromaLocation::Left), SubsampledContent(1, 1));

    // A pair whose two sides use different chroma phases has content that the
    // upsampling kernel cannot reproduce identically, so the three filters and
    // the two chroma locations must give different scores.
    const Image other =
        Format(YuvFormat(1, 1, ChromaLocation::Left), SubsampledContent(1, 1));

    const double bilinear = MakeFilterPipeline(ChromaFilter::Bilinear)
                                .Compute(content, other)
                                .Score();
    const double bicubic = MakeFilterPipeline(ChromaFilter::Bicubic)
                               .Compute(content, other)
                               .Score();
    const double lanczos = MakeFilterPipeline(ChromaFilter::Lanczos)
                               .Compute(content, other)
                               .Score();
    // The pair is identical here, so use a distorted side instead: shift the
    // chroma content by one column.
    std::array<std::vector<float>, 4> shifted = SubsampledContent(1, 1);
    std::rotate(shifted[1].begin(), shifted[1].begin() + 1, shifted[1].end());
    const Image distorted = Format(format, shifted);
    const double f_bilinear = MakeFilterPipeline(ChromaFilter::Bilinear)
                                  .Compute(content, distorted)
                                  .Score();
    const double f_bicubic = MakeFilterPipeline(ChromaFilter::Bicubic)
                                 .Compute(content, distorted)
                                 .Score();
    const double f_lanczos = MakeFilterPipeline(ChromaFilter::Lanczos)
                                 .Compute(content, distorted)
                                 .Score();
    CHECK(std::fabs(f_bilinear - f_bicubic) > 1e-3);
    CHECK(std::fabs(f_bilinear - f_lanczos) > 1e-3);
    CHECK(std::fabs(f_bicubic - f_lanczos) > 1e-3);
    (void)bilinear;
    (void)bicubic;
    (void)lanczos;
    (void)reference;

    // Chroma siting: a frame whose chroma is 0 at index 0 and 1 elsewhere must
    // score differently with CENTER than with LEFT.
    std::array<std::vector<float>, 4> edge = SubsampledContent(1, 1);
    for (std::size_t i = 0; i < edge[1].size(); ++i) {
        edge[1][i] = i == 0 ? 0.0f : 1.0f;
    }
    const Image left = Format(YuvFormat(1, 1, ChromaLocation::Left), edge);
    const Image center = Format(YuvFormat(1, 1, ChromaLocation::Center), edge);
    Ssimulacra2Pipeline pipeline = MakePipeline();
    const double left_score = pipeline.Compute(left, left).Score();
    const double center_score = pipeline.Compute(center, center).Score();
    const auto context = vqm::internal::AcquireVulkanContext();
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context->physical_device, &properties);
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
        // Software Vulkan can leave a small, content-dependent identity error.
        CHECK(left_score >= 99.5 && left_score <= 100.0);
        CHECK(center_score >= 99.5 && center_score <= 100.0);
    } else {
        CHECK_NEAR(left_score, center_score, 1e-9);  // hardware arithmetic pin
    }

    const double left_vs_center = pipeline.Compute(left, center).Score();
    const double left_vs_left = pipeline.Compute(left, left).Score();
    CHECK_NEAR(left_vs_left, left_score, 1e-9);
    CHECK_NEAR(pipeline.Compute(center, center).Score(), center_score, 1e-9);
    CHECK(std::fabs(left_vs_center - left_vs_left) > 1e-3);
}

// --- constant luminance and ICtCp -----------------------------------------

VQM_TEST(TestConstantLuminanceAndIctCp) {
    if (!vqm::test::GpuAvailable()) return;
    Ssimulacra2Pipeline pipeline = MakePipeline();
    const ref::Vec3 yuv{0.5, 0.1, -0.2};

    struct ClCase {
        MatrixCoefficients matrix;
        ColorPrimaries primaries;
    };
    const ClCase cl_cases[] = {
        {MatrixCoefficients::Bt2020Cl, ColorPrimaries::Bt2020},
        {MatrixCoefficients::ChromaticityDerivedCl, ColorPrimaries::Bt709},
        {MatrixCoefficients::ChromaticityDerivedCl, ColorPrimaries::Bt470Bg},
    };
    for (const ClCase& c : cl_cases) {
        ImageFormat format;
        format.type = PixelType::Float;
        format.family = ColorFamily::Yuv;
        format.matrix = c.matrix;
        format.transfer = TransferCharacteristics::Bt709;
        format.primaries = c.primaries;
        const Image a = Format(format, Uniform(yuv));
        double kr = 0.2627;
        double kb = 0.0593;
        if (c.matrix == MatrixCoefficients::ChromaticityDerivedCl) {
            ref::YuvConstantsFromPrimaries(c.primaries, kr, kb);
        }
        const ref::Vec3 rgb = ref::ClToRgb(kr, kb, yuv);
        const Image b = Format(
            LinearRgbFormat(),
            Uniform(
                ref::Mul(ref::Gamut(c.primaries, ColorPrimaries::Bt709), rgb)
            )
        );
        CheckAgrees(
            pipeline, a, b, kRounding,
            std::format("CL matrix {}", static_cast<int>(c.matrix)).c_str()
        );
    }

    // ICtCp: ICtCp -> LMS -> PQ -> RGB(BT.2020) -> linear Rec.709.
    struct IctCpCase {
        TransferCharacteristics transfer;
        double scale;
    };
    const IctCpCase ictcp_cases[] = {
        {TransferCharacteristics::St2084, 100.0},
        {TransferCharacteristics::AribB67, 10.0},
    };
    // Chromatic enough that a per-channel decode of LMS would be visibly
    // different from the luminance OOTF zimg applies.
    const ref::Vec3 ictcp{0.5, 0.15, -0.12};
    for (const IctCpCase& c : ictcp_cases) {
        ImageFormat format;
        format.type = PixelType::Float;
        format.family = ColorFamily::Yuv;
        format.matrix = MatrixCoefficients::Bt2100ICtCp;
        format.transfer = c.transfer;
        format.primaries = ColorPrimaries::Bt2020;
        const Image a = Format(format, Uniform(ictcp));

        const ref::Vec3 lms = ref::Mul(ref::IctCpToLms(c.transfer), ictcp);
        ref::Vec3 decoded{};
        if (c.transfer == TransferCharacteristics::AribB67) {
            // BT.2020 luma weights, applied to the LMS triple as zimg does.
            decoded = ref::HlgToLinear(lms, 0.2627, 0.6780, 0.0593, c.scale);
        } else {
            for (std::size_t i = 0; i < 3; ++i) {
                decoded[i] = ref::DecodeTransfer(c.transfer, lms[i]);
            }
        }
        const ref::Vec3 rgb2020 = ref::Mul(ref::LmsToRgb(), decoded);
        const ref::Vec3 rgb = ref::Mul(
            ref::Gamut(ColorPrimaries::Bt2020, ColorPrimaries::Bt709), rgb2020
        );
        const Image b = Format(LinearRgbFormat(), Uniform(rgb));
        CheckAgrees(
            pipeline, a, b,
            c.transfer == TransferCharacteristics::AribB67 ? kHlgCurve
                                                           : kRounding,
            std::format("ictcp transfer {}", static_cast<int>(c.transfer))
                .c_str()
        );
    }
}

// --- alpha types -----------------------------------------------------------

VQM_TEST(TestAlphaTypes) {
    if (!vqm::test::GpuAvailable()) return;
    Ssimulacra2Pipeline pipeline = MakePipeline();

    // Straight alpha A over colour S equals premultiplied colour S*A.
    std::array<std::vector<float>, 4> straight;
    std::array<std::vector<float>, 4> premultiplied;
    for (std::size_t y = 0; y < kSide; ++y) {
        for (std::size_t x = 0; x < kSide; ++x) {
            const double a = static_cast<double>((x + 3 * y) % 7) / 6.0;
            const double s[3] = {
                static_cast<double>(x) / (kSide - 1),
                static_cast<double>(y) / (kSide - 1),
                static_cast<double>((x * 7 + y * 11) % 64) / 63.0};
            for (std::size_t c = 0; c < 3; ++c) {
                straight[c].push_back(static_cast<float>(s[c]));
                premultiplied[c].push_back(static_cast<float>(s[c] * a));
            }
            straight[3].push_back(static_cast<float>(a));
            premultiplied[3].push_back(static_cast<float>(a));
        }
    }

    ImageFormat straight_format = vqm::test::PinnedFormat(AlphaType::Straight);
    ImageFormat premultiplied_format =
        vqm::test::PinnedFormat(AlphaType::Premultiplied);
    const Image a = Format(straight_format, straight);
    const Image b = Format(premultiplied_format, premultiplied);
    for (const float background : {0.1f, 0.9f}) {
        CheckAgrees(pipeline, a, b, kRounding, "alpha types", background);
    }

    // Premultiplied with a zero alpha must not divide by zero.
    std::array<std::vector<float>, 4> zero = premultiplied;
    for (std::vector<float>& plane : zero) {
        std::fill(plane.begin(), plane.end(), 0.0f);
    }
    const Image degenerate = Format(premultiplied_format, zero);
    const Ssimulacra2Result result = pipeline.Compute(degenerate, degenerate);
    for (const auto& scale : result.scales) {
        for (const double v : scale.avg_ssim) CHECK(std::isfinite(v));
        for (const double v : scale.avg_edgediff) CHECK(std::isfinite(v));
    }
    const double score = result.Score();
    CHECK(std::isfinite(score));
    CHECK(score <= 100.0);
}

// --- rejections ------------------------------------------------------------

namespace {

// `format` must be refused with a message containing `expected`.
void CheckRejects(
    Ssimulacra2Pipeline& pipeline, const ImageFormat& format,
    const char* expected
) {
    Image image;
    image.size = Side();
    image.format = format;
    image.data.resize(vqm::RequiredBytes(format, image.size));
    std::string message;
    try {
        (void)pipeline.Compute(image, image);
    } catch (const std::invalid_argument& e) {
        message = e.what();
    }
    if (message.find(expected) == std::string::npos) {
        vqm::test::Fail(
            std::source_location::current(),
            std::format("expected \"{}\", got \"{}\"", expected, message)
        );
    }
}

}  // namespace

VQM_TEST(TestRejectedFormats) {
    if (!vqm::test::GpuAvailable()) return;
    Ssimulacra2Pipeline pipeline = MakePipeline();
    const ImageFormat rgb = vqm::test::PinnedFormat(AlphaType::None);

    {  // A YUV image cannot claim RGB matrix coefficients.
        ImageFormat f = rgb;
        f.family = ColorFamily::Yuv;
        f.matrix = MatrixCoefficients::Rgb;
        CheckRejects(pipeline, f, "YUV images cannot declare RGB");
    }
    {  // RGB cannot be subsampled.
        ImageFormat f = rgb;
        f.subsample_w = 1;
        CheckRejects(pipeline, f, "subsampled RGB images are not supported");
    }
    {  // Neither can greyscale.
        ImageFormat f = rgb;
        f.family = ColorFamily::Grey;
        f.matrix = MatrixCoefficients::Bt709;
        f.subsample_h = 1;
        CheckRejects(
            pipeline, f, "subsampled greyscale images are not supported"
        );
    }
    {  // RGB cannot claim a YUV matrix.
        ImageFormat f = rgb;
        f.matrix = MatrixCoefficients::Bt709;
        CheckRejects(pipeline, f, "RGB images cannot declare YUV");
    }
    {  // Constant luminance needs the BT.709 curve.
        ImageFormat f = rgb;
        f.family = ColorFamily::Yuv;
        f.matrix = MatrixCoefficients::Bt2020Cl;
        f.transfer = TransferCharacteristics::Srgb;
        f.primaries = ColorPrimaries::Bt2020;
        CheckRejects(pipeline, f, "constant-luminance matrix coefficients");
    }
    {  // ICtCp needs PQ or HLG and BT.2020 primaries.
        ImageFormat f = rgb;
        f.family = ColorFamily::Yuv;
        f.matrix = MatrixCoefficients::Bt2100ICtCp;
        f.transfer = TransferCharacteristics::Srgb;
        f.primaries = ColorPrimaries::Bt2020;
        CheckRejects(pipeline, f, "ICtCp requires");
    }
    {  // An unset transfer function cannot be linearised.
        ImageFormat f = rgb;
        f.transfer = TransferCharacteristics::Unspecified;
        CheckRejects(pipeline, f, "transfer characteristics are required");
    }
    {  // Nor can unset primaries be mapped to Rec.709.
        ImageFormat f = rgb;
        f.primaries = ColorPrimaries::Unspecified;
        CheckRejects(pipeline, f, "color primaries are required");
    }
    {  // No conversion is defined for this matrix.
        ImageFormat f = rgb;
        f.family = ColorFamily::Yuv;
        f.matrix = MatrixCoefficients::St2085YDzDx;
        CheckRejects(pipeline, f, "unsupported matrix coefficients");
    }
    {  // Depth cannot exceed the pixel type.
        ImageFormat f = rgb;
        f.type = PixelType::Byte;
        f.depth = 12;
        CheckRejects(pipeline, f, "bit depth exceeds the pixel type");
    }
    {  // Limited range needs at least 8 bits.
        ImageFormat f = rgb;
        f.type = PixelType::Byte;
        f.depth = 7;
        f.range = PixelRange::Limited;
        CheckRejects(
            pipeline, f, "limited range requires a bit depth of at least 8"
        );
    }
    {  // A truncated buffer is refused with the expected size.
        Image short_image = Format(rgb, Uniform({0.5, 0.5, 0.5}));
        short_image.data.pop_back();
        std::string message;
        try {
            (void)pipeline.Compute(short_image, short_image);
        } catch (const std::invalid_argument& e) {
            message = e.what();
        }
        CHECK(message.find("data size does not match") != std::string::npos);
        CHECK(
            message.find(std::to_string(vqm::RequiredBytes(rgb, Side()))) !=
            std::string::npos
        );
    }
}

// --- YUV4MPEG2 -------------------------------------------------------------

VQM_TEST(TestY4mHighBitDepthLittleEndian) {
    const char* path = "y4m_word_test.y4m";
    std::FILE* file = std::fopen(path, "wb");
    CHECK(file != nullptr);
    if (!file) return;
    std::fputs("YUV4MPEG2 W2 H2 C444p10\nFRAME\n", file);
    const std::array<std::uint16_t, 12> samples = {
        64, 257, 512, 940, 100, 333, 601, 900, 256, 500, 700, 1000};
    for (std::uint16_t value : samples) {
        std::fputc(static_cast<int>(value & 255u), file);
        std::fputc(static_cast<int>(value >> 8u), file);
    }
    std::fclose(file);
    std::string error;
    const auto loaded = vqm::image_io::LoadImage(path, error);
    std::remove(path);
    CHECK(loaded.has_value());
    if (!loaded) return;
    CHECK(loaded->format.type == PixelType::Word);
    CHECK(loaded->format.depth == 10);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        std::uint16_t value = 0;
        std::memcpy(&value, loaded->data.data() + 2 * i, sizeof(value));
        CHECK(value == samples[i]);
    }
}

VQM_TEST(TestY4mReader) {
    if (!vqm::test::GpuAvailable()) return;
    constexpr std::size_t side = 16;
    constexpr std::size_t chroma_side = side / 2;
    const char* path = "y4m_test.y4m";

    // A 4:2:0 8-bit frame with JPEG (centred) chroma siting.
    std::vector<unsigned char> luma(side * side);
    std::vector<unsigned char> u_plane(chroma_side * chroma_side);
    std::vector<unsigned char> v_plane(chroma_side * chroma_side);
    for (std::size_t y = 0; y < side; ++y) {
        for (std::size_t x = 0; x < side; ++x) {
            luma[y * side + x] =
                static_cast<unsigned char>((20 + x * 13 + y * 7) & 0xFF);
        }
    }
    for (std::size_t i = 0; i < u_plane.size(); ++i) {
        u_plane[i] = static_cast<unsigned char>(
            i % chroma_side < chroma_side / 2 ? 60 : 200
        );
        v_plane[i] = static_cast<unsigned char>(
            i / chroma_side < chroma_side / 2 ? 180 : 90
        );
    }
    {
        std::FILE* file = std::fopen(path, "wb");
        CHECK(file != nullptr);
        std::fputs("YUV4MPEG2 W16 H16 F25:1 Ip A1:1 C420jpeg\nFRAME\n", file);
        for (const unsigned char value : luma) std::fputc(value, file);
        for (const unsigned char value : u_plane) std::fputc(value, file);
        for (const unsigned char value : v_plane) std::fputc(value, file);
        std::fclose(file);
    }

    std::string error;
    const std::optional<Image> loaded = vqm::image_io::LoadImage(path, error);
    std::remove(path);
    if (!loaded) {
        vqm::test::Fail(
            std::source_location::current(), std::format("y4m: {}", error)
        );
        return;
    }
    CHECK(loaded->size.width == side && loaded->size.height == side);
    CHECK(loaded->format.subsample_w == 1 && loaded->format.subsample_h == 1);
    CHECK(loaded->format.chroma_location == ChromaLocation::Center);
    CHECK(loaded->format.range == PixelRange::Limited);

    // The reference 4:4:4 frame: the y4m's samples decoded to [0, 1] and its
    // chroma upsampled with the same bilinear kernel, in float.
    std::array<std::vector<float>, 4> values;
    for (std::size_t i = 0; i < luma.size(); ++i) {
        values[0].push_back(
            static_cast<float>((static_cast<double>(luma[i]) - 16.0) / 219.0)
        );
    }
    std::vector<double> u_norm;
    std::vector<double> v_norm;
    for (std::size_t i = 0; i < u_plane.size(); ++i) {
        u_norm.push_back((static_cast<double>(u_plane[i]) - 128.0) / 224.0);
        v_norm.push_back((static_cast<double>(v_plane[i]) - 128.0) / 224.0);
    }
    const double shift = ref::ChromaShift(ChromaLocation::Center, true, 1);
    const double shift_y = ref::ChromaShift(ChromaLocation::Center, false, 1);
    for (std::size_t y = 0; y < side; ++y) {
        const auto ty =
            ref::AxisTaps(ChromaFilter::Bilinear, y, 2.0, shift_y, chroma_side);
        for (std::size_t x = 0; x < side; ++x) {
            const auto tx = ref::AxisTaps(
                ChromaFilter::Bilinear, x, 2.0, shift, chroma_side
            );
            double u = 0.0;
            double v = 0.0;
            for (const auto& [iy, wy] : ty) {
                for (const auto& [ix, wx] : tx) {
                    u += wx * wy * u_norm[iy * chroma_side + ix];
                    v += wx * wy * v_norm[iy * chroma_side + ix];
                }
            }
            values[1].push_back(static_cast<float>(u));
            values[2].push_back(static_cast<float>(v));
        }
    }

    ImageFormat reference_format;
    reference_format.type = PixelType::Float;
    reference_format.family = ColorFamily::Yuv;
    reference_format.matrix = MatrixCoefficients::Bt470Bg;
    reference_format.transfer = TransferCharacteristics::Bt709;
    reference_format.primaries = ColorPrimaries::Bt709;
    const Image reference = Format(reference_format, values, {side, side});

    Ssimulacra2Pipeline pipeline({{side, side}});
    CheckAgrees(pipeline, *loaded, reference, kRounding, "y4m round trip");

    // An unknown chroma tag is refused with a named reason.
    {
        std::FILE* file = std::fopen(path, "wb");
        CHECK(file != nullptr);
        std::fputs("YUV4MPEG2 W16 H16 Cbogus\nFRAME\n", file);
        for (const unsigned char value : luma) std::fputc(value, file);
        std::fclose(file);
    }
    std::string tag_error;
    const std::optional<Image> rejected =
        vqm::image_io::LoadImage(path, tag_error);
    std::remove(path);
    CHECK(!rejected.has_value());
    CHECK(tag_error == "unsupported y4m chroma tag");
}
