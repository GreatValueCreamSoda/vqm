// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "vqm.h"

// Independent double-precision reference for the input conversions, written
// from the standards rather than from the implementation, so the A/B tests in
// input_format_test.cc check the GPU against something other than itself.

namespace vqm::test::ref {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<double, 9>;  // row-major

inline Mat3 Identity() { return {1, 0, 0, 0, 1, 0, 0, 0, 1}; }

inline Vec3 Mul(const Mat3& m, const Vec3& v) {
    return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}

inline Mat3 Mul(const Mat3& a, const Mat3& b) {
    Mat3 r{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                sum += a[i * 3 + k] * b[k * 3 + j];
            }
            r[i * 3 + j] = sum;
        }
    }
    return r;
}

inline double Det2(double a, double b, double c, double d) {
    return a * d - b * c;
}

inline Mat3 Inverse(const Mat3& m) {
    const double det = m[0] * Det2(m[4], m[5], m[7], m[8]) -
                       m[1] * Det2(m[3], m[5], m[6], m[8]) +
                       m[2] * Det2(m[3], m[4], m[6], m[7]);
    Mat3 r{};
    r[0] = Det2(m[4], m[5], m[7], m[8]) / det;
    r[1] = -Det2(m[1], m[2], m[7], m[8]) / det;
    r[2] = Det2(m[1], m[2], m[4], m[5]) / det;
    r[3] = -Det2(m[3], m[5], m[6], m[8]) / det;
    r[4] = Det2(m[0], m[2], m[6], m[8]) / det;
    r[5] = -Det2(m[0], m[2], m[3], m[5]) / det;
    r[6] = Det2(m[3], m[4], m[6], m[7]) / det;
    r[7] = -Det2(m[0], m[1], m[6], m[7]) / det;
    r[8] = Det2(m[0], m[1], m[3], m[4]) / det;
    return r;
}

// --- transfer functions, display-referred like the metric ---

inline double Rec709Oetf(double x) {
    constexpr double kAlpha = 1.09929682680944;
    constexpr double kBeta = 0.018053968510807;
    x = std::fmax(x, 0.0);
    return x < kBeta ? 4.5 * x : kAlpha * std::pow(x, 0.45) - (kAlpha - 1.0);
}

inline double Rec709InverseOetf(double x) {
    constexpr double kAlpha = 1.09929682680944;
    constexpr double kBeta = 0.018053968510807;
    x = std::fmax(x, 0.0);
    return x < 4.5 * kBeta ? x / 4.5
                           : std::pow((x + (kAlpha - 1.0)) / kAlpha, 1.0 / 0.45);
}

inline double AribB67InverseOetf(double x) {
    x = std::fmax(x, 0.0);
    return x <= 0.5
               ? x * x / 3.0
               : (std::exp((x - 0.55991073) / 0.17883277) + 0.28466892) / 12.0;
}

// zimg's display-referred to_linear for `transfer`, including the peak
// luminance factor of PQ (10000/100) and HLG (1000/100). HLG's luminance OOTF
// needs the whole triple and is handled by the caller.
inline double DecodeTransfer(TransferCharacteristics transfer, double v) {
    constexpr double kM1 = 0.1593017578125;
    constexpr double kM2 = 78.84375;
    constexpr double kC1 = 0.8359375;
    constexpr double kC2 = 18.8515625;
    constexpr double kC3 = 18.6875;
    switch (transfer) {
        case TransferCharacteristics::Linear:
            return v;
        case TransferCharacteristics::Bt709:
        case TransferCharacteristics::Bt601:
        case TransferCharacteristics::Bt2020_10:
        case TransferCharacteristics::Bt2020_12:
        case TransferCharacteristics::St240M:
            return v < 0.0 ? 0.0 : std::pow(v, 2.4);
        case TransferCharacteristics::Bt470M:
            return v < 0.0 ? 0.0 : std::pow(v, 2.2);
        case TransferCharacteristics::Bt470Bg:
            return v < 0.0 ? 0.0 : std::pow(v, 2.8);
        case TransferCharacteristics::Srgb:
            return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
        case TransferCharacteristics::Log100:
            return v <= 0.0 ? 0.01 : std::pow(10.0, 2.0 * (v - 1.0));
        case TransferCharacteristics::Log316:
            return v <= 0.0 ? 0.00316227766 : std::pow(10.0, 2.5 * (v - 1.0));
        case TransferCharacteristics::Iec61966_2_4:
            if (v >= 0.0 && v <= 1.0) return std::pow(v, 2.4);
            return std::copysign(Rec709InverseOetf(std::fabs(v)), v);
        case TransferCharacteristics::Bt1361:
            if (v >= 0.0 && v <= 1.0) return std::pow(v, 2.4);
            return v >= 0.0 ? Rec709InverseOetf(v)
                            : -Rec709InverseOetf(-4.0 * v) / 4.0;
        case TransferCharacteristics::St2084: {
            if (v <= 0.0) return 0.0;
            const double xpow = std::pow(v, 1.0 / kM2);
            const double num = std::fmax(xpow - kC1, 0.0);
            const double den = std::fmax(kC2 - kC3 * xpow, 1.17549435e-38);
            return 100.0 * std::pow(num / den, 1.0 / kM1);
        }
        case TransferCharacteristics::St428:
            return v < 0.0 ? 0.0 : (52.37 / 48.0) * std::pow(v, 2.6);
        case TransferCharacteristics::AribB67:
            // Per-channel inverse OETF with the luminance OOTF; the caller
            // supplies the luma weights, this is the single-channel shape.
            return AribB67InverseOetf(v);
        case TransferCharacteristics::Unspecified:
            return v;
    }
    return v;
}

// zimg's AribB67InverseOperationC, the display-referred HLG decode: per-channel
// inverse OETF, then the *luminance* OOTF across the triple, then the
// peak-luminance factor. For neutral input this equals pow(v, 1.2) per
// channel; for every chromatic pixel it does not.
inline Vec3 HlgToLinear(
    const Vec3& v, double kr, double kg, double kb, double scale
) {
    Vec3 c = {AribB67InverseOetf(v[0]), AribB67InverseOetf(v[1]),
              AribB67InverseOetf(v[2])};
    const double luma =
        std::fmax(kr * c[0] + kg * c[1] + kb * c[2], 1.17549435e-38);
    const double ys = std::pow(luma, 0.2);
    return {c[0] * ys * scale, c[1] * ys * scale, c[2] * ys * scale};
}

// --- matrix coefficients ---

inline void YuvConstants(MatrixCoefficients matrix, double& kr, double& kb) {
    switch (matrix) {
        case MatrixCoefficients::Fcc:
            kr = 0.3;
            kb = 0.11;
            return;
        case MatrixCoefficients::St240M:
            kr = 0.212;
            kb = 0.087;
            return;
        case MatrixCoefficients::Bt470Bg:
        case MatrixCoefficients::St170M:
            kr = 0.299;
            kb = 0.114;
            return;
        case MatrixCoefficients::Bt709:
            kr = 0.2126;
            kb = 0.0722;
            return;
        default:
            kr = 0.2627;
            kb = 0.0593;
            return;
    }
}

inline Mat3 RgbToYuv(double kr, double kb) {
    const double kg = 1.0 - kr - kb;
    const double u = 1.0 / (2.0 - 2.0 * kb);
    const double v = 1.0 / (2.0 - 2.0 * kr);
    return {
        kr,      kg,     kb, -kr * u, -kg * u, (1.0 - kb) * u, (1.0 - kr) * v,
        -kg * v, -kb * v};
}

// Encoded YUV -> encoded RGB for `matrix`.
inline Mat3 NclToRgb(MatrixCoefficients matrix) {
    if (matrix == MatrixCoefficients::YCgCo) {
        // YCgCo: Y = R/4 + G/2 + B/4, Cg = -R/4 + G/2 - B/4, Co = R/2 - B/2.
        return {1.0, -1.0, 1.0, 1.0, 1.0, 0.0, 1.0, -1.0, -1.0};
    }
    double kr = 0.0;
    double kb = 0.0;
    YuvConstants(matrix, kr, kb);
    return Inverse(RgbToYuv(kr, kb));
}

// --- primaries ---

struct Xy {
    double x;
    double y;
};

inline Vec3 XyToXyz(const Xy& xy) {
    return {xy.x / xy.y, 1.0, (1.0 - xy.x - xy.y) / xy.y};
}

inline bool PrimariesXy(ColorPrimaries primaries, Xy out[3]) {
    switch (primaries) {
        case ColorPrimaries::Bt470M:
            out[0] = {0.670, 0.330};
            out[1] = {0.210, 0.710};
            out[2] = {0.140, 0.080};
            return true;
        case ColorPrimaries::Bt470Bg:
            out[0] = {0.640, 0.330};
            out[1] = {0.290, 0.600};
            out[2] = {0.150, 0.060};
            return true;
        case ColorPrimaries::St170M:
        case ColorPrimaries::St240M:
            out[0] = {0.630, 0.340};
            out[1] = {0.310, 0.595};
            out[2] = {0.155, 0.070};
            return true;
        case ColorPrimaries::Bt709:
            out[0] = {0.640, 0.330};
            out[1] = {0.300, 0.600};
            out[2] = {0.150, 0.060};
            return true;
        case ColorPrimaries::Film:
            out[0] = {0.681, 0.319};
            out[1] = {0.243, 0.692};
            out[2] = {0.145, 0.049};
            return true;
        case ColorPrimaries::Bt2020:
            out[0] = {0.708, 0.292};
            out[1] = {0.170, 0.797};
            out[2] = {0.131, 0.046};
            return true;
        case ColorPrimaries::St431_2:
        case ColorPrimaries::St432_1:
            out[0] = {0.680, 0.320};
            out[1] = {0.265, 0.690};
            out[2] = {0.150, 0.060};
            return true;
        case ColorPrimaries::Ebu3213E:
            out[0] = {0.630, 0.340};
            out[1] = {0.295, 0.605};
            out[2] = {0.155, 0.077};
            return true;
        case ColorPrimaries::St428:
        case ColorPrimaries::Unspecified:
            return false;
    }
    return false;
}

inline Xy WhitePoint(ColorPrimaries primaries) {
    switch (primaries) {
        case ColorPrimaries::Bt470M:
        case ColorPrimaries::Film:
            return {0.31, 0.316};  // Illuminant C
        case ColorPrimaries::St428:
            return {1.0 / 3.0, 1.0 / 3.0};
        case ColorPrimaries::St431_2:
            return {0.314, 0.351};  // DCI
        default:
            return {0.3127, 0.3290};  // D65
    }
}

inline Mat3 RgbToXyz(ColorPrimaries primaries) {
    if (primaries == ColorPrimaries::St428) return Identity();
    Xy xy[3];
    if (!PrimariesXy(primaries, xy)) return Identity();
    const Vec3 r = XyToXyz(xy[0]);
    const Vec3 g = XyToXyz(xy[1]);
    const Vec3 b = XyToXyz(xy[2]);
    const Mat3 m{r[0], g[0], b[0], r[1], g[1], b[1], r[2], g[2], b[2]};
    const Vec3 s = Mul(Inverse(m), XyToXyz(WhitePoint(primaries)));
    Mat3 out{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            out[row * 3 + column] = m[row * 3 + column] * s[column];
        }
    }
    return out;
}

inline Mat3 Gamut(ColorPrimaries from, ColorPrimaries to) {
    return Mul(Inverse(RgbToXyz(to)), RgbToXyz(from));
}

inline double Dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline Vec3 Cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

// zimg reads the two standard sets off their published constants and derives
// everything else with ITU-T H.265 Annex E, Eq (E-22) to (E-27).
inline void YuvConstantsFromPrimaries(
    ColorPrimaries primaries, double& kr, double& kb
) {
    if (primaries == ColorPrimaries::Bt709) {
        kr = 0.2126;
        kb = 0.0722;
        return;
    }
    if (primaries == ColorPrimaries::Bt2020) {
        kr = 0.2627;
        kb = 0.0593;
        return;
    }
    Xy xy[3];
    if (!PrimariesXy(primaries, xy)) {
        kr = 0.2126;
        kb = 0.0722;
        return;
    }
    const Vec3 r = XyToXyz(xy[0]);
    const Vec3 g = XyToXyz(xy[1]);
    const Vec3 b = XyToXyz(xy[2]);
    const Vec3 white = XyToXyz(WhitePoint(primaries));
    const Vec3 x_rgb{r[0], g[0], b[0]};
    const Vec3 y_rgb{r[1], g[1], b[1]};
    const Vec3 z_rgb{r[2], g[2], b[2]};
    const double denominator = Dot(x_rgb, Cross(y_rgb, z_rgb));
    kr = Dot(white, Cross(g, b)) / denominator;
    kb = Dot(white, Cross(r, g)) / denominator;
}

// Encoded YUV -> encoded RGB for the chromaticity-derived matrix.
inline Mat3 NclToRgbFromPrimaries(ColorPrimaries primaries) {
    if (primaries == ColorPrimaries::Bt709) {
        return NclToRgb(MatrixCoefficients::Bt709);
    }
    if (primaries == ColorPrimaries::Bt2020) {
        return NclToRgb(MatrixCoefficients::Bt2020Ncl);
    }
    double kr = 0.0;
    double kb = 0.0;
    YuvConstantsFromPrimaries(primaries, kr, kb);
    return Inverse(RgbToYuv(kr, kb));
}

// --- constant luminance and ICtCp ---

inline Vec3 ClToRgb(double kr, double kb, const Vec3& yuv) {
    const double kg = 1.0 - kr - kb;
    const double nb = Rec709Oetf(1.0 - kb);
    const double pb = 1.0 - Rec709Oetf(kb);
    const double nr = Rec709Oetf(1.0 - kr);
    const double pr = 1.0 - Rec709Oetf(kr);
    const double b_minus_y = yuv[1] * 2.0 * (yuv[1] < 0.0 ? nb : pb);
    const double r_minus_y = yuv[2] * 2.0 * (yuv[2] < 0.0 ? nr : pr);
    const double b = Rec709InverseOetf(b_minus_y + yuv[0]);
    const double r = Rec709InverseOetf(r_minus_y + yuv[0]);
    const double yl = Rec709InverseOetf(yuv[0]);
    return {r, (yl - kr * r - kb * b) / kg, b};
}

inline Mat3 LmsToIctCp(TransferCharacteristics transfer) {
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

inline Mat3 IctCpToLms(TransferCharacteristics transfer) {
    return Inverse(LmsToIctCp(transfer));
}

inline Mat3 LmsToRgb() {
    // ITU-T H.273: M_RGBtoLMS, so LMS -> RGB is its inverse.
    return Inverse(
        Mat3{1688.0 / 4096.0, 2146.0 / 4096.0, 262.0 / 4096.0, 683.0 / 4096.0,
             2951.0 / 4096.0, 462.0 / 4096.0, 99.0 / 4096.0, 309.0 / 4096.0,
             3688.0 / 4096.0}
    );
}

// --- chroma upsampling, zimg's resampler ---

inline double Sinc(double x) {
    constexpr double kPi = 3.14159265358979323846;
    return x == 0.0 ? 1.0 : std::sin(x * kPi) / (x * kPi);
}

inline double FilterKernel(ChromaFilter filter, double x) {
    x = std::fabs(x);
    switch (filter) {
        case ChromaFilter::Bilinear:
            return std::fmax(1.0 - x, 0.0);
        case ChromaFilter::Bicubic:
            if (x < 1.0) return 1.0 - 2.5 * x * x + 1.5 * x * x * x;
            if (x < 2.0) return 2.0 - 4.0 * x + 2.5 * x * x - 0.5 * x * x * x;
            return 0.0;
        case ChromaFilter::Lanczos:
            return x < 3.0 ? Sinc(x) * Sinc(x / 3.0) : 0.0;
    }
    return 0.0;
}

inline int Taps(ChromaFilter filter) {
    switch (filter) {
        case ChromaFilter::Bilinear:
            return 2;
        case ChromaFilter::Bicubic:
            return 4;
        case ChromaFilter::Lanczos:
            return 6;
    }
    return 2;
}

// One axis of the resampler: (mirrored index, normalised weight) pairs for
// output position `out_pos`, sampling a plane of `dim` samples.
inline std::vector<std::pair<std::size_t, double>> AxisTaps(
    ChromaFilter filter, std::size_t out_pos, double scale, double shift,
    std::size_t dim
) {
    const int taps = Taps(filter);
    const double pos = (static_cast<double>(out_pos) + 0.5) / scale + shift;
    const double start = pos - taps / 2.0;
    // round half up, as zimg keeps round(x - 1) == round(x) - 1.
    const double base = (start < 0.0 ? std::floor(start + 0.5)
                                     : std::floor(start + 0.49999999999999994)) +
                        0.5;
    double total = 0.0;
    std::vector<std::pair<std::size_t, double>> taps_out;
    for (int j = 0; j < taps; ++j) {
        total += FilterKernel(filter, base + j - pos);
    }
    for (int j = 0; j < taps; ++j) {
        const double xpos = base + j;
        const double mirrored =
            xpos < 0.0 ? -xpos
                       : (xpos >= static_cast<double>(dim)
                              ? 2.0 * static_cast<double>(dim) - xpos
                              : xpos);
        const double clamped =
            std::clamp(mirrored, 0.0, static_cast<double>(dim) - 1.0);
        taps_out.emplace_back(
            static_cast<std::size_t>(std::floor(clamped)),
            FilterKernel(filter, xpos - pos) / total
        );
    }
    return taps_out;
}

inline double ChromaShift(
    ChromaLocation location, bool horizontal, unsigned subsample
) {
    const double ratio = 1.0 / static_cast<double>(1u << subsample);
    const bool left = location == ChromaLocation::Left ||
                      location == ChromaLocation::TopLeft ||
                      location == ChromaLocation::BottomLeft;
    const bool top =
        location == ChromaLocation::TopLeft || location == ChromaLocation::Top;
    const bool bottom = location == ChromaLocation::BottomLeft ||
                        location == ChromaLocation::Bottom;
    if (horizontal) return left ? 0.5 - 0.5 * ratio : 0.0;
    if (top) return 0.5 - 0.5 * ratio;
    if (bottom) return -(0.5 - 0.5 * ratio);
    return 0.0;
}

}  // namespace vqm::test::ref
