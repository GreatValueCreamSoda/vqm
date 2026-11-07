// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vqm {

struct VulkanDevice;
struct VulkanBufferView;
struct VulkanSubmitSync;
struct ImageView;

struct Size {
    std::size_t width = 0;
    std::size_t height = 0;
    [[nodiscard]] constexpr bool operator==(const Size&) const noexcept =
        default;
};

// Input description enums. The numeric values are zimg's, so a caller that
// already holds zimg enums can pass them through unchanged.
enum class PixelType : std::uint8_t { Byte = 0, Word = 1, Half = 2, Float = 3 };
enum class PixelRange : std::uint8_t { Limited = 0, Full = 1 };
enum class ColorFamily : std::uint8_t { Grey = 0, Rgb = 1, Yuv = 2 };
enum class AlphaType : std::uint8_t {
    None = 0,
    Straight = 1,
    Premultiplied = 2
};
enum class ChromaLocation : std::uint8_t {
    Left = 0,
    Center = 1,
    TopLeft = 2,
    Top = 3,
    BottomLeft = 4,
    Bottom = 5
};
enum class MatrixCoefficients : std::uint8_t {
    Rgb = 0,
    Bt709 = 1,
    Unspecified = 2,
    Fcc = 4,
    Bt470Bg = 5,
    St170M = 6,
    St240M = 7,
    YCgCo = 8,
    Bt2020Ncl = 9,
    Bt2020Cl = 10,
    St2085YDzDx = 11,
    ChromaticityDerivedNcl = 12,
    ChromaticityDerivedCl = 13,
    Bt2100ICtCp = 14,
    St2128IptC2 = 15,
    YCgCoRe = 16,
    YCgCoRo = 17
};
enum class TransferCharacteristics : std::uint8_t {
    Bt709 = 1,
    Unspecified = 2,
    Bt470M = 4,
    Bt470Bg = 5,
    Bt601 = 6,
    St240M = 7,
    Linear = 8,
    Log100 = 9,
    Log316 = 10,
    Iec61966_2_4 = 11,
    Bt1361 = 12,
    Srgb = 13,
    Bt2020_10 = 14,
    Bt2020_12 = 15,
    St2084 = 16,
    St428 = 17,
    AribB67 = 18
};
enum class ColorPrimaries : std::uint8_t {
    Bt709 = 1,
    Unspecified = 2,
    Bt470M = 4,
    Bt470Bg = 5,
    St170M = 6,
    St240M = 7,
    Film = 8,
    Bt2020 = 9,
    St428 = 10,
    St431_2 = 11,
    St432_1 = 12,
    Ebu3213E = 22
};
// Upsampling kernel for subsampled chroma planes.
enum class ChromaFilter : std::uint8_t {
    Bilinear = 0,
    Bicubic = 1,
    Lanczos = 2
};

// Everything needed to interpret an Image's samples: the zimg input surface.
// Planar only, no field parity, full planes (no active-window cropping).
//
// Decoding is display-referred with zimg's defaults: scene_referred,
// approximate_gamma and chromatic_adaptation off, peak luminance 100 cd/m^2.
// Every sample is decoded straight to float, so the depth field never selects
// a low-precision path.
struct ImageFormat {
    PixelType type = PixelType::Float;
    ColorFamily family = ColorFamily::Rgb;
    AlphaType alpha = AlphaType::None;
    PixelRange range = PixelRange::Full;
    MatrixCoefficients matrix = MatrixCoefficients::Rgb;
    TransferCharacteristics transfer = TransferCharacteristics::Srgb;
    ColorPrimaries primaries = ColorPrimaries::Bt709;
    ChromaLocation chroma_location = ChromaLocation::Left;
    std::uint8_t subsample_w = 0;  // log2, 0..2
    std::uint8_t subsample_h = 0;  // log2, 0..2
    std::uint8_t depth = 0;        // 0 = the pixel type's full depth
};

// A decoded image: zimg plane order (Y/R, U/G, V/B, alpha) held in one packed
// buffer. There is no row padding and no header, each plane starts on a 4-byte
// boundary so the shader can address a plane as 32-bit words:
//
//   plane 0        width * height samples
//   plane 1, 2     (width >> subsample_w) * (height >> subsample_h) samples,
//                  present when family is not Grey
//   plane 3        width * height samples, present when alpha is not None
//
// A sample is Byte/Word/Half/Float = 1/2/2/4 bytes in the host's byte order.
// Integer samples are subject to `range` and `depth`; float samples are
// normalised and ignore `range`. Call RequiredBytes() to size `data`.
struct Image {
    Size size;
    ImageFormat format;
    std::vector<std::uint8_t> data;
    [[nodiscard]] bool HasAlpha() const noexcept {
        return format.alpha != AlphaType::None;
    }
};

// Bytes `data` needs for `format` and `size`, including the 4-byte alignment
// of each plane. Throws std::invalid_argument for an invalid pixel type or
// subsampling and std::overflow_error if the layout does not fit in size_t.
[[nodiscard]] std::size_t RequiredBytes(const ImageFormat& format, Size size);

// Device-side copy of one image, produced by a metric pipeline's Upload. Treat
// as opaque: `slot` identifies one upload in one pipeline. A handle from another
// pipeline or an earlier upload cannot alias a later occupant of a slot.
struct UploadedImage {
    std::uint64_t slot = 0;  // 0 == invalid
    Size size;
};

// Device memory reserved for one input image by a metric pipeline's
// AllocateImage. `bytes` is exactly RequiredBytes(format,
// size) for the format it was allocated with; write that many bytes into
// MapImage(buffer), in the plane layout Image documents, then score two of them
// with Compute or Submit. `slot` identifies this allocation in this pipeline;
// handles from other pipelines and released allocations cannot alias it.
struct ImageBuffer {
    std::uint64_t slot = 0;  // 0 == invalid
    Size size;
    std::size_t bytes = 0;
};

}  // namespace vqm
