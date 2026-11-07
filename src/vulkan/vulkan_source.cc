// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// Host-side source descriptors and the matching prepare shader variants.
// Device ownership and feature negotiation live in vulkan_support.cc.

#include "vulkan_support.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "input.h"

namespace vqm::internal {
namespace {

// enums0 bit allocation, mirrored by src/vulkan/shaders/color.glsl.
constexpr std::uint32_t kTypeShift = 0;
constexpr std::uint32_t kFamilyShift = 2;
constexpr std::uint32_t kAlphaShift = 4;
constexpr std::uint32_t kSubsampleWShift = 6;
constexpr std::uint32_t kSubsampleHShift = 8;
constexpr std::uint32_t kTransferShift = 10;
constexpr std::uint32_t kFilterShift = 15;
constexpr std::uint32_t kModeShift = 17;

// enums1 flags.
constexpr std::uint32_t kIdentity1 = 1u << 0;
constexpr std::uint32_t kIdentity2 = 1u << 1;

void StoreMatrix(const Matrix3& matrix, float out[9]) {
    for (std::size_t i = 0; i < 9; ++i) {
        out[i] = static_cast<float>(matrix[i]);
    }
}

// zimg's chroma_offset_w/chroma_offset_h, negated as its resampler applies
// them. Both axes are zero when that axis is not subsampled, because the
// offset is then "centred" by definition.
void ChromaSiting(const ImageFormat& format, double& shift_x, double& shift_y) {
    const double ratio_w = 1.0 / static_cast<double>(1u << format.subsample_w);
    const double ratio_h = 1.0 / static_cast<double>(1u << format.subsample_h);
    const double left = 0.5 - 0.5 * ratio_w;
    const double top = 0.5 - 0.5 * ratio_h;
    const double bottom = -(0.5 - 0.5 * ratio_h);
    shift_x = 0.0;
    shift_y = 0.0;
    switch (format.chroma_location) {
        case ChromaLocation::Left:
            shift_x = left;
            break;
        case ChromaLocation::Center:
            break;
        case ChromaLocation::TopLeft:
            shift_x = left;
            shift_y = top;
            break;
        case ChromaLocation::Top:
            shift_y = top;
            break;
        case ChromaLocation::BottomLeft:
            shift_x = left;
            shift_y = bottom;
            break;
        case ChromaLocation::Bottom:
            shift_y = bottom;
            break;
    }
}

}  // namespace

SourceDesc BuildSourceDesc(
    const ImageFormat& format, const std::array<std::uint32_t, 4>& plane_words,
    ChromaFilter chroma_filter
) {
    SourceDesc desc{};
    std::memcpy(desc.plane_word, plane_words.data(), sizeof(desc.plane_word));

    const OpMode mode = SelectMode(format);
    const TransferDecode transfer = SelectTransfer(format.transfer);
    const bool grey = format.family == ColorFamily::Grey;
    const bool chroma_subsampled =
        format.family == ColorFamily::Yuv &&
        (format.subsample_w != 0 || format.subsample_h != 0);

    desc.enums0 =
        (static_cast<std::uint32_t>(format.type) << kTypeShift) |
        (static_cast<std::uint32_t>(format.family) << kFamilyShift) |
        (static_cast<std::uint32_t>(format.alpha) << kAlphaShift) |
        (static_cast<std::uint32_t>(format.subsample_w) << kSubsampleWShift) |
        (static_cast<std::uint32_t>(format.subsample_h) << kSubsampleHShift) |
        (static_cast<std::uint32_t>(transfer.id) << kTransferShift) |
        (static_cast<std::uint32_t>(chroma_filter) << kFilterShift) |
        (static_cast<std::uint32_t>(mode) << kModeShift);

    // Only a YUV image has a chroma plane class; the shader's plane_class()
    // applies the same rule, so scale[1]/offset[1] are unused elsewhere.
    const ScaleOffset luma = PlaneScaleOffset(format, false, false);
    const ScaleOffset chroma =
        PlaneScaleOffset(format, format.family == ColorFamily::Yuv, false);
    const ScaleOffset alpha = PlaneScaleOffset(format, false, true);
    desc.scale[0] = luma.scale;
    desc.scale[1] = chroma.scale;
    desc.scale[2] = alpha.scale;
    desc.offset[0] = luma.offset;
    desc.offset[1] = chroma.offset;
    desc.offset[2] = alpha.offset;

    std::uint32_t flags = 0;
    switch (mode) {
        case OpMode::ConstantLuminance: {
            // The CL op consumes Y/U/V directly, so matrix1 stays identity.
            StoreMatrix(
                Matrix3{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
                desc.matrix1
            );
            flags |= kIdentity1;
            break;
        }
        case OpMode::ICtCp:
            StoreMatrix(IctCpToLmsMatrix(format.transfer), desc.matrix1);
            break;
        case OpMode::Matrix: {
            bool identity = false;
            const Matrix3 matrix = NclToRgbMatrix(format, identity);
            StoreMatrix(matrix, desc.matrix1);
            if (identity) flags |= kIdentity1;
            break;
        }
    }

    if (grey || format.primaries == ColorPrimaries::Bt709) {
        flags |= kIdentity2;
        StoreMatrix(
            Matrix3{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, desc.matrix2
        );
    } else if (mode == OpMode::ICtCp) {
        // zimg's graph: ICtCp -> LMS, transfer, LMS -> RGB (the H.273 LMS
        // matrix), then the gamut conversion off the BT.2020 primaries.
        StoreMatrix(
            Multiply(
                GamutMatrix(format.primaries, ColorPrimaries::Bt709),
                LmsToRgbMatrix()
            ),
            desc.matrix2
        );
    } else {
        StoreMatrix(
            GamutMatrix(format.primaries, ColorPrimaries::Bt709), desc.matrix2
        );
    }

    double kr = 0.0;
    double kg = 1.0;
    double kb = 0.0;
    LumaCoefficients(format, kr, kg, kb);
    desc.kr = static_cast<float>(kr);
    desc.kg = static_cast<float>(kg);
    desc.kb = static_cast<float>(kb);
    desc.to_linear_scale = transfer.to_linear_scale;
    desc.enums1 = flags;

    double shift_x = 0.0;
    double shift_y = 0.0;
    if (chroma_subsampled) ChromaSiting(format, shift_x, shift_y);
    desc.shift_x = static_cast<float>(shift_x);
    desc.shift_y = static_cast<float>(shift_y);

    if (mode == OpMode::ConstantLuminance) {
        const std::array<float, 4> corner = ConstantLuminanceConstants(kr, kb);
        desc.cl_nb = corner[0];
        desc.cl_pb = corner[1];
        desc.cl_nr = corner[2];
        desc.cl_pr = corner[3];
    }
    return desc;
}

}  // namespace vqm::internal

namespace vqm::internal {

SourceDecodeVariant SelectSourceDecodeVariant(
    const ImageFormat& format, ChromaFilter chroma_filter
) {
    SourceDecodeVariant variant;
    variant.chroma = format.subsample_w != 0 || format.subsample_h != 0 ? 1 : 0;
    if (SelectMode(format) != OpMode::Matrix) {
        variant.transfer = 0;
    } else if (format.transfer == TransferCharacteristics::Srgb) {
        variant.transfer = 1;
    } else if (format.transfer == TransferCharacteristics::Linear) {
        variant.transfer = 2;
    } else {
        variant.transfer = 0;
    }
    variant.filter =
        variant.chroma != 0 ? static_cast<std::uint8_t>(chroma_filter) : 0;
    return variant;
}

std::size_t SourceDecodeVariantIndex(SourceDecodeVariant variant) {
    for (std::size_t i = 0; i < kSourceDecodeVariants.size(); ++i) {
        if (kSourceDecodeVariants[i] == variant) return i;
    }
    throw std::invalid_argument("vulkan: no prepare kernel for this format");
}

}  // namespace vqm::internal
