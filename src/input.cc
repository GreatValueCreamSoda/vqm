// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "input.h"

namespace vqm {
namespace {

constexpr std::size_t SampleBytes(PixelType type) noexcept {
    switch (type) {
        case PixelType::Byte:
            return 1;
        case PixelType::Word:
        case PixelType::Half:
            return 2;
        case PixelType::Float:
            return 4;
    }
    return 0;
}

constexpr std::size_t TypeDepth(PixelType type) noexcept {
    switch (type) {
        case PixelType::Byte:
            return 8;
        case PixelType::Word:
        case PixelType::Half:
            return 16;
        case PixelType::Float:
            return 32;
    }
    return 0;
}

constexpr bool HasChroma(ColorFamily family) noexcept {
    return family != ColorFamily::Grey;
}

constexpr bool KnownMatrix(MatrixCoefficients matrix) noexcept {
    switch (matrix) {
        case MatrixCoefficients::Rgb:
        case MatrixCoefficients::Bt709:
        case MatrixCoefficients::Unspecified:
        case MatrixCoefficients::Fcc:
        case MatrixCoefficients::Bt470Bg:
        case MatrixCoefficients::St170M:
        case MatrixCoefficients::St240M:
        case MatrixCoefficients::YCgCo:
        case MatrixCoefficients::Bt2020Ncl:
        case MatrixCoefficients::Bt2020Cl:
        case MatrixCoefficients::St2085YDzDx:
        case MatrixCoefficients::ChromaticityDerivedNcl:
        case MatrixCoefficients::ChromaticityDerivedCl:
        case MatrixCoefficients::Bt2100ICtCp:
        case MatrixCoefficients::St2128IptC2:
        case MatrixCoefficients::YCgCoRe:
        case MatrixCoefficients::YCgCoRo:
            return true;
    }
    return false;
}

constexpr bool KnownPrimaries(ColorPrimaries primaries) noexcept {
    switch (primaries) {
        case ColorPrimaries::Bt709:
        case ColorPrimaries::Unspecified:
        case ColorPrimaries::Bt470M:
        case ColorPrimaries::Bt470Bg:
        case ColorPrimaries::St170M:
        case ColorPrimaries::St240M:
        case ColorPrimaries::Film:
        case ColorPrimaries::Bt2020:
        case ColorPrimaries::St428:
        case ColorPrimaries::St431_2:
        case ColorPrimaries::St432_1:
        case ColorPrimaries::Ebu3213E:
            return true;
    }
    return false;
}

// Every plane starts on a 32-bit boundary: the shader addresses a plane as
// words of the arena and derives the word from the sample index, which only
// holds while the plane's base offset is a multiple of the sample width's
// largest alignment (4).
std::size_t CheckedAdd(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b) {
        throw std::overflow_error("image layout exceeds size_t");
    }
    return a + b;
}

std::size_t CheckedMultiply(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("image layout exceeds size_t");
    }
    return a * b;
}

std::size_t AlignWord(std::size_t bytes) {
    return CheckedAdd(bytes, 3u) / 4u * 4u;
}

}  // namespace

namespace internal {

std::array<PlaneGeometry, 4> PlaneLayout(const ImageFormat& format, Size size) {
    std::array<PlaneGeometry, 4> layout{};
    const std::size_t sample_bytes = SampleBytes(format.type);
    if (sample_bytes == 0) {
        throw std::invalid_argument("invalid pixel type");
    }
    if (format.subsample_w > 2 || format.subsample_h > 2) {
        throw std::invalid_argument(
            "subsampling greater than 4x is not supported"
        );
    }
    const std::size_t chroma_width = size.width >> format.subsample_w;
    const std::size_t chroma_height = size.height >> format.subsample_h;
    std::size_t cursor = 0;
    for (std::size_t plane = 0; plane < layout.size(); ++plane) {
        const bool chroma = plane == 1 || plane == 2;
        const bool present = plane == 0 ||
                             (chroma && HasChroma(format.family)) ||
                             (plane == 3 && format.alpha != AlphaType::None);
        if (!present) continue;
        PlaneGeometry& geometry = layout[plane];
        geometry.width = chroma ? chroma_width : size.width;
        geometry.height = chroma ? chroma_height : size.height;
        geometry.sample_bytes = sample_bytes;
        geometry.offset = cursor;
        const std::size_t pixels =
            CheckedMultiply(geometry.width, geometry.height);
        const std::size_t bytes = CheckedMultiply(pixels, sample_bytes);
        cursor = AlignWord(CheckedAdd(cursor, bytes));
    }
    return layout;
}

std::uint8_t EffectiveDepth(const ImageFormat& format) {
    return format.depth != 0 ? format.depth
                             : static_cast<std::uint8_t>(TypeDepth(format.type));
}

void ValidateImage(const ImageFormat& format, Size size, const char* name) {
    const auto fail = [name](const char* message) {
        throw std::invalid_argument(std::string(name) + ": " + message);
    };
    // Validate enum membership before arithmetic or descriptor bit packing.
    // External callers may pass numeric zimg values that this API does not know.
    if (SampleBytes(format.type) == 0) fail("invalid pixel type");
    if (format.range != PixelRange::Limited && format.range != PixelRange::Full)
        fail("invalid pixel range");
    if (format.alpha != AlphaType::None && format.alpha != AlphaType::Straight &&
        format.alpha != AlphaType::Premultiplied)
        fail("invalid alpha type");
    if (static_cast<std::uint8_t>(format.chroma_location) >
        static_cast<std::uint8_t>(ChromaLocation::Bottom))
        fail("invalid chroma location");
    if (!KnownMatrix(format.matrix)) fail("invalid matrix coefficients");
    if (!KnownPrimaries(format.primaries)) fail("invalid color primaries");
    const std::uint8_t depth = EffectiveDepth(format);
    if (format.depth > TypeDepth(format.type)) {
        fail("bit depth exceeds the pixel type");
    }
    if (format.subsample_w > 2 || format.subsample_h > 2) {
        fail("subsampling greater than 4x is not supported");
    }
    if (size.width % (std::size_t{1} << format.subsample_w) != 0 ||
        size.height % (std::size_t{1} << format.subsample_h) != 0) {
        fail("image dimensions must be divisible by the subsampling");
    }
    if (format.range == PixelRange::Limited && depth < 8) {
        fail("limited range requires a bit depth of at least 8");
    }
    switch (format.family) {
        case ColorFamily::Grey:
            if (format.subsample_w != 0 || format.subsample_h != 0) {
                fail("subsampled greyscale images are not supported");
            }
            if (format.matrix == MatrixCoefficients::Rgb) {
                fail("greyscale images cannot declare RGB matrix coefficients");
            }
            break;
        case ColorFamily::Rgb:
            if (format.subsample_w != 0 || format.subsample_h != 0) {
                fail("subsampled RGB images are not supported");
            }
            if (format.matrix != MatrixCoefficients::Rgb &&
                format.matrix != MatrixCoefficients::Unspecified) {
                fail("RGB images cannot declare YUV matrix coefficients");
            }
            break;
        case ColorFamily::Yuv:
            if (format.matrix == MatrixCoefficients::Rgb) {
                fail("YUV images cannot declare RGB matrix coefficients");
            }
            if (format.matrix == MatrixCoefficients::Unspecified) {
                fail("matrix coefficients are required for YUV input");
            }
            break;
        default:
            fail("invalid color family");
    }
    // Every family needs a transfer function: it is what linearises the
    // samples before the metric sees them.
    if (const char* reason = TransferError(format.transfer)) fail(reason);
    // Greyscale is neutral by definition, so its primaries carry no meaning
    // and are not required (zimg does not convert greyscale colour either).
    if (format.family != ColorFamily::Grey &&
        format.primaries == ColorPrimaries::Unspecified) {
        fail("color primaries are required");
    }
    if (const char* reason = ModeError(format)) fail(reason);
}

void CheckImage(const Image& image, Size size, const char* name) {
    if (image.size.width != size.width || image.size.height != size.height) {
        throw std::invalid_argument(
            std::string(name) +
            ": image size does not match the pipeline geometry"
        );
    }
    ValidateImage(image.format, image.size, name);
    const std::size_t expected = RequiredBytes(image.format, image.size);
    if (image.data.size() != expected) {
        throw std::invalid_argument(
            std::string(name) +
            ": data size does not match the image format (expected " +
            std::to_string(expected) + " bytes)"
        );
    }
}

}  // namespace internal

std::size_t RequiredBytes(const ImageFormat& format, Size size) {
    const std::array<internal::PlaneGeometry, 4> layout =
        internal::PlaneLayout(format, size);
    std::size_t total = 0;
    for (const internal::PlaneGeometry& plane : layout) {
        if (plane.present()) {
            total =
                plane.offset + plane.width * plane.height * plane.sample_bytes;
        }
    }
    return AlignWord(total);
}

}  // namespace vqm

namespace vqm::internal {
namespace {

// --- zimg constants (colorspace/colorspace_param.h, colorspace/gamma.cpp) ---

constexpr double kFccKr = 0.3;
constexpr double kFccKb = 0.11;
constexpr double kSt240MKr = 0.212;
constexpr double kSt240MKb = 0.087;
constexpr double kRec601Kr = 0.299;
constexpr double kRec601Kb = 0.114;
constexpr double kRec709Kr = 0.2126;
constexpr double kRec709Kb = 0.0722;
constexpr double kRec2020Kr = 0.2627;
constexpr double kRec2020Kb = 0.0593;

using Xy = std::array<double, 2>;

constexpr Xy kRec470MPrimaries[3] = {
    {0.670, 0.330}, {0.210, 0.710}, {0.140, 0.080}};
constexpr Xy kRec470BgPrimaries[3] = {
    {0.640, 0.330}, {0.290, 0.600}, {0.150, 0.060}};
constexpr Xy kSmpteCPrimaries[3] = {
    {0.630, 0.340}, {0.310, 0.595}, {0.155, 0.070}};
constexpr Xy kRec709Primaries[3] = {
    {0.640, 0.330}, {0.300, 0.600}, {0.150, 0.060}};
constexpr Xy kFilmPrimaries[3] = {
    {0.681, 0.319}, {0.243, 0.692}, {0.145, 0.049}};
constexpr Xy kRec2020Primaries[3] = {
    {0.708, 0.292}, {0.170, 0.797}, {0.131, 0.046}};
constexpr Xy kDciP3Primaries[3] = {
    {0.680, 0.320}, {0.265, 0.690}, {0.150, 0.060}};
constexpr Xy kEbu3213EPrimaries[3] = {
    {0.630, 0.340}, {0.295, 0.605}, {0.155, 0.077}};

constexpr Xy kIlluminantC = {0.31, 0.316};
constexpr Xy kIlluminantDci = {0.314, 0.351};
constexpr Xy kIlluminantD65 = {0.3127, 0.3290};
constexpr Xy kIlluminantE = {1.0 / 3.0, 1.0 / 3.0};

// gamma.cpp: BT.709 OETF constants, the two values the CL op and the
// xvYCC/BT.1361 branches need.
constexpr double kRec709Alpha = 1.09929682680944;
constexpr double kRec709Beta = 0.018053968510807;

double Rec709Oetf(double x) {
    x = std::max(x, 0.0);
    return x < kRec709Beta
               ? x * 4.5
               : kRec709Alpha * std::pow(x, 0.45) - (kRec709Alpha - 1.0);
}

// --- matrix helpers, in zimg's algebra ---

double Det2(double a00, double a01, double a10, double a11) {
    return a00 * a11 - a01 * a10;
}

double Determinant(const Matrix3& m) {
    double det = 0.0;
    det += m[0] * Det2(m[4], m[5], m[7], m[8]);
    det -= m[1] * Det2(m[3], m[5], m[6], m[8]);
    det += m[2] * Det2(m[3], m[4], m[6], m[7]);
    return det;
}

Matrix3 Inverse(const Matrix3& m) {
    const double det = Determinant(m);
    Matrix3 r{};
    r[0] = Det2(m[4], m[5], m[7], m[8]) / det;
    r[1] = Det2(m[2], m[1], m[8], m[7]) / det;
    r[2] = Det2(m[1], m[2], m[4], m[5]) / det;
    r[3] = Det2(m[5], m[3], m[8], m[6]) / det;
    r[4] = Det2(m[0], m[2], m[6], m[8]) / det;
    r[5] = Det2(m[2], m[0], m[5], m[3]) / det;
    r[6] = Det2(m[3], m[4], m[6], m[7]) / det;
    r[7] = Det2(m[1], m[0], m[7], m[6]) / det;
    r[8] = Det2(m[0], m[1], m[3], m[4]) / det;
    return r;
}

std::array<double, 3> Mul(const Matrix3& m, const std::array<double, 3>& v) {
    return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}

double Dot(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<double, 3> Cross(
    const std::array<double, 3>& a, const std::array<double, 3>& b
) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

Matrix3 Identity() { return {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}; }

std::array<double, 3> XyToXyz(const Xy& xy) {
    return {xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]};
}

// --- primaries tables ---

void PrimariesXy(ColorPrimaries primaries, Xy out[3]) {
    const Xy* source = nullptr;
    switch (primaries) {
        case ColorPrimaries::Bt470M:
            source = kRec470MPrimaries;
            break;
        case ColorPrimaries::Bt470Bg:
            source = kRec470BgPrimaries;
            break;
        case ColorPrimaries::St170M:
        case ColorPrimaries::St240M:
            source = kSmpteCPrimaries;
            break;
        case ColorPrimaries::Bt709:
            source = kRec709Primaries;
            break;
        case ColorPrimaries::Film:
            source = kFilmPrimaries;
            break;
        case ColorPrimaries::Bt2020:
            source = kRec2020Primaries;
            break;
        case ColorPrimaries::St431_2:
        case ColorPrimaries::St432_1:
            source = kDciP3Primaries;
            break;
        case ColorPrimaries::Ebu3213E:
            source = kEbu3213EPrimaries;
            break;
        case ColorPrimaries::St428:
        case ColorPrimaries::Unspecified:
            break;
    }
    if (source == nullptr) {
        throw std::invalid_argument(
            "color primaries do not define chromaticities"
        );
    }
    std::copy(source, source + 3, out);
}

std::array<double, 3> WhitePoint(ColorPrimaries primaries) {
    switch (primaries) {
        case ColorPrimaries::Bt470M:
        case ColorPrimaries::Film:
            return XyToXyz(kIlluminantC);
        case ColorPrimaries::St428:
            return XyToXyz(kIlluminantE);
        case ColorPrimaries::St431_2:
            return XyToXyz(kIlluminantDci);
        default:
            return XyToXyz(kIlluminantD65);
    }
}

// Columns: R G B, rows X Y Z.
Matrix3 PrimariesXyz(ColorPrimaries primaries) {
    Xy xy[3];
    PrimariesXy(primaries, xy);
    const std::array<double, 3> r = XyToXyz(xy[0]);
    const std::array<double, 3> g = XyToXyz(xy[1]);
    const std::array<double, 3> b = XyToXyz(xy[2]);
    // zimg builds the rows then transposes.
    return {r[0], g[0], b[0], r[1], g[1], b[1], r[2], g[2], b[2]};
}

// zimg's ncl_rgb_to_yuv_matrix_from_primaries: the two standard sets use their
// published constants rather than the chromaticity-derived ones (which differ
// in the fourth digit), and everything else is derived from the primaries.
void YuvConstantsFromPrimaries(
    ColorPrimaries primaries, double& kr, double& kb
) {
    if (primaries == ColorPrimaries::Bt709) {
        kr = kRec709Kr;
        kb = kRec709Kb;
        return;
    }
    if (primaries == ColorPrimaries::Bt2020) {
        kr = kRec2020Kr;
        kb = kRec2020Kb;
        return;
    }
    Xy xy[3];
    PrimariesXy(primaries, xy);
    const std::array<double, 3> r = XyToXyz(xy[0]);
    const std::array<double, 3> g = XyToXyz(xy[1]);
    const std::array<double, 3> b = XyToXyz(xy[2]);
    const std::array<double, 3> white = WhitePoint(primaries);
    const std::array<double, 3> x_rgb = {r[0], g[0], b[0]};
    const std::array<double, 3> y_rgb = {r[1], g[1], b[1]};
    const std::array<double, 3> z_rgb = {r[2], g[2], b[2]};
    const double denominator = Dot(x_rgb, Cross(y_rgb, z_rgb));
    kr = Dot(white, Cross(g, b)) / denominator;
    kb = Dot(white, Cross(r, g)) / denominator;
}

Matrix3 RgbToYuvFromKrKb(double kr, double kb) {
    const double kg = 1.0 - kr - kb;
    const double uscale = 1.0 / (2.0 - 2.0 * kb);
    const double vscale = 1.0 / (2.0 - 2.0 * kr);
    return {kr,
            kg,
            kb,
            -kr * uscale,
            -kg * uscale,
            (1.0 - kb) * uscale,
            (1.0 - kr) * vscale,
            -kg * vscale,
            -kb * vscale};
}

void YuvConstants(MatrixCoefficients matrix, double& kr, double& kb) {
    switch (matrix) {
        case MatrixCoefficients::Fcc:
            kr = kFccKr;
            kb = kFccKb;
            return;
        case MatrixCoefficients::St240M:
            kr = kSt240MKr;
            kb = kSt240MKb;
            return;
        case MatrixCoefficients::Bt470Bg:
        case MatrixCoefficients::St170M:
            kr = kRec601Kr;
            kb = kRec601Kb;
            return;
        case MatrixCoefficients::Bt709:
            kr = kRec709Kr;
            kb = kRec709Kb;
            return;
        case MatrixCoefficients::Bt2020Ncl:
        case MatrixCoefficients::Bt2020Cl:
            kr = kRec2020Kr;
            kb = kRec2020Kb;
            return;
        default:
            throw std::invalid_argument(
                "matrix coefficients have no luma weights"
            );
    }
}

Matrix3 RgbToYuvMatrix(MatrixCoefficients matrix) {
    if (matrix == MatrixCoefficients::YCgCo) {
        return {0.25, 0.5, 0.25, -0.25, 0.5, -0.25, 0.5, 0.0, -0.5};
    }
    if (matrix == MatrixCoefficients::Bt2100ICtCp) {
        // Not reachable through the public surface; the BT.2100 LMS matrix is
        // asked for by LmsToRgbMatrix() instead.
        throw std::invalid_argument("matrix coefficients are not NCL");
    }
    double kr = 0.0;
    double kb = 0.0;
    YuvConstants(matrix, kr, kb);
    return RgbToYuvFromKrKb(kr, kb);
}

Matrix3 YuvToRgbFromPrimaries(ColorPrimaries primaries) {
    if (primaries == ColorPrimaries::Bt709) {
        return Inverse(RgbToYuvMatrix(MatrixCoefficients::Bt709));
    }
    if (primaries == ColorPrimaries::Bt2020) {
        return Inverse(RgbToYuvMatrix(MatrixCoefficients::Bt2020Ncl));
    }
    double kr = 0.0;
    double kb = 0.0;
    YuvConstantsFromPrimaries(primaries, kr, kb);
    return Inverse(RgbToYuvFromKrKb(kr, kb));
}

Matrix3 LmsToIctCp(TransferCharacteristics transfer) {
    if (transfer == TransferCharacteristics::AribB67) {
        return {0.5,
                0.5,
                0.0,
                3625.0 / 4096.0,
                -7465.0 / 4096.0,
                3840.0 / 4096.0,
                9500.0 / 4096.0,
                -9212.0 / 4096.0,
                -288.0 / 4096.0};
    }
    return {0.5,
            0.5,
            0.0,
            6610.0 / 4096.0,
            -13613.0 / 4096.0,
            7003.0 / 4096.0,
            17933.0 / 4096.0,
            -17390.0 / 4096.0,
            -543.0 / 4096.0};
}

Matrix3 RgbToXyz(ColorPrimaries primaries) {
    if (primaries == ColorPrimaries::St428) return Identity();
    const Matrix3 m = PrimariesXyz(primaries);
    const std::array<double, 3> s = Mul(Inverse(m), WhitePoint(primaries));
    Matrix3 r{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            r[row * 3 + column] = m[row * 3 + column] * s[column];
        }
    }
    return r;
}

Matrix3 XyzToRgb(ColorPrimaries primaries) {
    if (primaries == ColorPrimaries::St428) return Identity();
    return Inverse(RgbToXyz(primaries));
}

bool IsBt709Curve(TransferCharacteristics transfer) {
    switch (transfer) {
        case TransferCharacteristics::Bt709:
        case TransferCharacteristics::Bt601:
        case TransferCharacteristics::Bt2020_10:
        case TransferCharacteristics::Bt2020_12:
            return true;
        default:
            return false;
    }
}

}  // namespace

const char* TransferError(TransferCharacteristics transfer) {
    switch (transfer) {
        case TransferCharacteristics::Bt709:
        case TransferCharacteristics::Bt470M:
        case TransferCharacteristics::Bt470Bg:
        case TransferCharacteristics::Bt601:
        case TransferCharacteristics::St240M:
        case TransferCharacteristics::Linear:
        case TransferCharacteristics::Log100:
        case TransferCharacteristics::Log316:
        case TransferCharacteristics::Iec61966_2_4:
        case TransferCharacteristics::Bt1361:
        case TransferCharacteristics::Srgb:
        case TransferCharacteristics::Bt2020_10:
        case TransferCharacteristics::Bt2020_12:
        case TransferCharacteristics::St2084:
        case TransferCharacteristics::St428:
        case TransferCharacteristics::AribB67:
            return nullptr;
        case TransferCharacteristics::Unspecified:
            return "transfer characteristics are required";
    }
    return "unsupported transfer characteristics";
}

const char* ModeError(const ImageFormat& format) {
    // A greyscale image is neutral by definition: its matrix coefficients and
    // primaries carry no meaning, so only the transfer function applies.
    if (format.family == ColorFamily::Grey) return nullptr;
    switch (format.matrix) {
        case MatrixCoefficients::Rgb:
        case MatrixCoefficients::Unspecified:
        case MatrixCoefficients::Bt709:
        case MatrixCoefficients::Fcc:
        case MatrixCoefficients::Bt470Bg:
        case MatrixCoefficients::St170M:
        case MatrixCoefficients::St240M:
        case MatrixCoefficients::YCgCo:
        case MatrixCoefficients::Bt2020Ncl:
        case MatrixCoefficients::ChromaticityDerivedNcl:
            break;
        case MatrixCoefficients::Bt2020Cl:
        case MatrixCoefficients::ChromaticityDerivedCl:
            if (!IsBt709Curve(format.transfer)) {
                return "constant-luminance matrix coefficients require the "
                       "BT.709 transfer";
            }
            break;
        case MatrixCoefficients::Bt2100ICtCp:
            if (!(format.transfer == TransferCharacteristics::St2084 ||
                  format.transfer == TransferCharacteristics::AribB67) ||
                format.primaries != ColorPrimaries::Bt2020) {
                return "ICtCp requires the ST 2084 or ARIB B67 transfer and "
                       "BT.2020 primaries";
            }
            break;
        default:
            return "unsupported matrix coefficients";
    }
    // Chromaticity-derived luma weights and the HLG luminance OOTF are read
    // off the primaries' chromaticities, which XYZ primaries do not have.
    if (format.family != ColorFamily::Grey &&
        format.primaries == ColorPrimaries::St428) {
        const bool derived =
            format.matrix == MatrixCoefficients::ChromaticityDerivedNcl ||
            format.matrix == MatrixCoefficients::ChromaticityDerivedCl;
        if (derived || format.transfer == TransferCharacteristics::AribB67) {
            return "XYZ color primaries do not define the chromaticity this "
                   "conversion needs";
        }
    }
    return nullptr;
}

TransferDecode SelectTransfer(TransferCharacteristics transfer) {
    switch (transfer) {
        case TransferCharacteristics::Linear:
            return {0, 1.0f};
        case TransferCharacteristics::Bt709:
        case TransferCharacteristics::Bt601:
        case TransferCharacteristics::Bt2020_10:
        case TransferCharacteristics::Bt2020_12:
        case TransferCharacteristics::St240M:
            return {1, 1.0f};
        case TransferCharacteristics::Bt470M:
            return {2, 1.0f};
        case TransferCharacteristics::Bt470Bg:
            return {3, 1.0f};
        case TransferCharacteristics::Srgb:
            return {4, 1.0f};
        case TransferCharacteristics::Log100:
            return {5, 1.0f};
        case TransferCharacteristics::Log316:
            return {6, 1.0f};
        case TransferCharacteristics::Iec61966_2_4:
            return {7, 1.0f};
        case TransferCharacteristics::Bt1361:
            return {8, 1.0f};
        case TransferCharacteristics::St2084:
            return {9, 10000.0f / 100.0f};
        case TransferCharacteristics::St428:
            return {10, 1.0f};
        case TransferCharacteristics::AribB67:
            return {11, 1000.0f / 100.0f};
        case TransferCharacteristics::Unspecified:
            break;
    }
    throw std::invalid_argument("unsupported transfer characteristics");
}

OpMode SelectMode(const ImageFormat& format) {
    // Greyscale bypasses the matrix stage entirely: plane 0 is replicated.
    if (format.family == ColorFamily::Grey) return OpMode::Matrix;
    switch (format.matrix) {
        case MatrixCoefficients::Bt2020Cl:
        case MatrixCoefficients::ChromaticityDerivedCl:
            return OpMode::ConstantLuminance;
        case MatrixCoefficients::Bt2100ICtCp:
            return OpMode::ICtCp;
        default:
            return OpMode::Matrix;
    }
}

ScaleOffset PlaneScaleOffset(
    const ImageFormat& format, bool chroma, bool alpha
) {
    const bool is_float =
        format.type == PixelType::Half || format.type == PixelType::Float;
    const int depth = EffectiveDepth(format);
    // This helper can also be used while constructing inputs, before
    // ValidateImage runs. Reject invalid depths before the integer shifts.
    if (depth < 1 || depth > static_cast<int>(TypeDepth(format.type))) {
        throw std::invalid_argument("invalid bit depth for pixel type");
    }
    if (format.range != PixelRange::Limited &&
        format.range != PixelRange::Full) {
        throw std::invalid_argument("invalid pixel range");
    }
    if (format.range == PixelRange::Limited && depth < 8) {
        throw std::invalid_argument(
            "limited range requires a bit depth of at least 8"
        );
    }
    if (is_float) return {1.0f, 0.0f};

    const bool full = alpha || format.range == PixelRange::Full;
    const bool ycgco = format.matrix == MatrixCoefficients::YCgCo;
    const double numeric_max = static_cast<double>((1u << depth) - 1u);
    const double range =
        full ? numeric_max
             : static_cast<double>(
                   chroma && !ycgco ? 224u << (depth - 8) : 219u << (depth - 8)
               );
    // zimg's integer_offset tests the chroma flag first: a chroma plane is
    // centred on the midpoint in *both* ranges, so full-range chroma decodes
    // to [-0.5, 0.5) rather than [0, 1).
    double offset = 0.0;
    if (alpha) {
        offset = 0.0;
    } else if (chroma) {
        offset = static_cast<double>(1u << (depth - 1));
    } else if (!full) {
        offset = static_cast<double>(16u << (depth - 8));
    }
    return {static_cast<float>(1.0 / range),
            static_cast<float>(-offset / range)};
}

void LumaCoefficients(
    const ImageFormat& format, double& kr, double& kg, double& kb
) {
    kr = 0.0;
    kg = 1.0;
    kb = 0.0;
    // Only two stages read these: the constant-luminance op (which takes them
    // from its matrix coefficients) and the HLG luminance OOTF (which takes
    // them off the primaries, as zimg does). Every other combination leaves
    // the neutral triple, which nothing reads.
    const OpMode mode = SelectMode(format);
    if (format.family == ColorFamily::Grey) return;
    if (mode == OpMode::ConstantLuminance) {
        if (format.matrix == MatrixCoefficients::ChromaticityDerivedCl) {
            YuvConstantsFromPrimaries(format.primaries, kr, kb);
        } else {
            YuvConstants(format.matrix, kr, kb);
        }
    } else if (format.transfer == TransferCharacteristics::AribB67) {
        YuvConstantsFromPrimaries(format.primaries, kr, kb);
    } else {
        return;
    }
    kg = 1.0 - kr - kb;
}

Matrix3 NclToRgbMatrix(const ImageFormat& format, bool& identity) {
    identity = true;
    if (format.family == ColorFamily::Grey) return Identity();
    switch (format.matrix) {
        case MatrixCoefficients::Rgb:
        case MatrixCoefficients::Unspecified:
            return Identity();
        case MatrixCoefficients::ChromaticityDerivedNcl:
            identity = false;
            return YuvToRgbFromPrimaries(format.primaries);
        default:
            break;
    }
    identity = false;
    return Inverse(RgbToYuvMatrix(format.matrix));
}

Matrix3 GamutMatrix(ColorPrimaries from, ColorPrimaries to) {
    return Multiply(XyzToRgb(to), RgbToXyz(from));
}

Matrix3 IctCpToLmsMatrix(TransferCharacteristics transfer) {
    return Inverse(LmsToIctCp(transfer));
}

Matrix3 LmsToRgbMatrix() {
    // zimg's ncl_rgb_to_yuv_matrix(REC_2100_LMS): the H.273 LMS matrix, used
    // as the RGB->"YUV" of the LMS colour family, so LMS -> RGB inverts it.
    const Matrix3 lms_to_rgb{1688.0 / 4096.0, 2146.0 / 4096.0, 262.0 / 4096.0,
                             683.0 / 4096.0,  2951.0 / 4096.0, 462.0 / 4096.0,
                             99.0 / 4096.0,   309.0 / 4096.0,  3688.0 / 4096.0};
    return Inverse(lms_to_rgb);
}

Matrix3 Multiply(const Matrix3& a, const Matrix3& b) {
    Matrix3 r{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double accum = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                accum += a[i * 3 + k] * b[k * 3 + j];
            }
            r[i * 3 + j] = accum;
        }
    }
    return r;
}

std::array<float, 4> ConstantLuminanceConstants(double kr, double kb) {
    const double nb = Rec709Oetf(1.0 - kb);
    const double pb = 1.0 - Rec709Oetf(kb);
    const double nr = Rec709Oetf(1.0 - kr);
    const double pr = 1.0 - Rec709Oetf(kr);
    return {static_cast<float>(nb), static_cast<float>(pb),
            static_cast<float>(nr), static_cast<float>(pr)};
}

}  // namespace vqm::internal
