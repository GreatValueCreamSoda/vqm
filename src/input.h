// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "vqm_image.h"

// Geometry and validation of the decoded input buffer: where each plane of an
// Image starts and how large it is. The shader addresses the same layout, so
// the rules here (plane order, 4-byte plane alignment) are part of the GPU
// interface.

namespace vqm::internal {

// Compare declared formats exactly. An implicit depth (0) and the pixel type's
// explicit full depth remain distinct, even when they decode identically.
[[nodiscard]] constexpr bool SameImageFormat(
    const ImageFormat& a, const ImageFormat& b
) noexcept {
    return a.type == b.type && a.family == b.family && a.alpha == b.alpha &&
           a.range == b.range && a.matrix == b.matrix &&
           a.transfer == b.transfer && a.primaries == b.primaries &&
           a.chroma_location == b.chroma_location &&
           a.subsample_w == b.subsample_w && a.subsample_h == b.subsample_h &&
           a.depth == b.depth;
}

struct PlaneGeometry {
    std::size_t width = 0;         // samples per row
    std::size_t height = 0;        // rows
    std::size_t sample_bytes = 0;  // 1, 2 or 4
    std::size_t offset = 0;        // bytes from the start of Image::data
    [[nodiscard]] constexpr bool present() const noexcept {
        return sample_bytes != 0;
    }
};

// One entry per plane in zimg order. Absent planes have sample_bytes == 0.
[[nodiscard]] std::array<PlaneGeometry, 4> PlaneLayout(
    const ImageFormat& format, Size size
);

// `depth`, or the pixel type's full depth when the format leaves it at 0.
[[nodiscard]] std::uint8_t EffectiveDepth(const ImageFormat& format);

// Throws std::invalid_argument("<name>: <reason>") unless every field of
// `format` is a combination this implementation decodes.
void ValidateImage(const ImageFormat& format, Size size, const char* name);

// Validates one image against a pipeline geometry: its size, every field of
// its format, and that `data` holds exactly RequiredBytes. Throws
// std::invalid_argument("<name>: <reason>") otherwise.
void CheckImage(const Image& image, Size size, const char* name);

}  // namespace vqm::internal

// The colours part of the input pipeline: zimg's depth quantisation, transfer
// functions, matrix coefficients and primaries. Every formula is a translit-
// eration of .reference/zimg (depth/quantize.h, colorspace/gamma.cpp,
// colorspace/colorspace_param.cpp, colorspace/operation_impl.cpp) with
// zimg's display-referred defaults and without chromatic adaptation.
//
// The host builds the constants here; the shader (src/vulkan/shaders/color.glsl)
// applies them. Nothing in this module touches the device.

namespace vqm::internal {

// Which conversion the encoded triple needs before the transfer function.
enum class OpMode : std::uint8_t {
    Matrix = 0,             // NCL YUV -> RGB, or identity for RGB/GREY
    ConstantLuminance = 1,  // Rec.2020 CL op, already linear on output
    ICtCp = 2,              // ICtCp -> LMS, then the transfer applies to LMS
};

struct ScaleOffset {
    float scale = 1.0f;
    float offset = 0.0f;
};

// The transfer decode the shader runs, and the peak-luminance factor zimg
// applies after it (10000/peak for PQ, 1000/peak for HLG, 1 otherwise).
struct TransferDecode {
    std::uint8_t id = 0;
    float to_linear_scale = 1.0f;
};

// Row-major 3x3 in double, like zimg's Matrix3x3.
using Matrix3 = std::array<double, 9>;

// nullptr when `format`'s colours combination is decodable, otherwise the
// reason, without the caller's "<name>: " prefix.
[[nodiscard]] const char* TransferError(TransferCharacteristics transfer);
[[nodiscard]] const char* ModeError(const ImageFormat& format);

[[nodiscard]] TransferDecode SelectTransfer(TransferCharacteristics transfer);
[[nodiscard]] OpMode SelectMode(const ImageFormat& format);

// scale/offset mapping a raw sample to a normalised float, for plane 0
// (luma/R), for a chroma plane, or for alpha (always full range).
[[nodiscard]] ScaleOffset PlaneScaleOffset(
    const ImageFormat& format, bool chroma, bool alpha
);

// The kr/kg/kb the active mode needs: from the matrix coefficients for
// constant luminance, from the primaries for the HLG luminance OOTF.
void LumaCoefficients(
    const ImageFormat& format, double& kr, double& kg, double& kb
);

// Encoded YUV -> encoded RGB. Sets `identity` when the triple passes through.
[[nodiscard]] Matrix3 NclToRgbMatrix(const ImageFormat& format, bool& identity);

// Linear RGB in `from` -> linear RGB in `to`, no chromatic adaptation.
[[nodiscard]] Matrix3 GamutMatrix(ColorPrimaries from, ColorPrimaries to);

// ICtCp -> LMS for the matching PQ/HLG variant of the BT.2100 matrix.
[[nodiscard]] Matrix3 IctCpToLmsMatrix(TransferCharacteristics transfer);

// LMS -> linear RGB. zimg models REC_2100_LMS as the LMS planes with this
// matrix as their RGB->"YUV", so the plane conversion is its inverse.
[[nodiscard]] Matrix3 LmsToRgbMatrix();

[[nodiscard]] Matrix3 Multiply(const Matrix3& a, const Matrix3& b);

// The four corner offsets of the constant-luminance op, evaluated from the
// BT.709 OETF (zimg's CLToRGBOperationC constructor).
[[nodiscard]] std::array<float, 4> ConstantLuminanceConstants(
    double kr, double kb
);

}  // namespace vqm::internal
