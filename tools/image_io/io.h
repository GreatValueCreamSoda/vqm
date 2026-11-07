// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "vqm_image.h"

namespace vqm::image_io {

// Loads PNG (8/16-bit), baseline/progressive JPEG or YUV4MPEG2 (first frame
// only), or returns std::nullopt and fills `error` for decoder errors.
// PNG/JPEG samples are interpreted as sRGB; embedded ICC profiles, gAMA, cHRM
// and sBIT chunks are ignored. YUV4MPEG2 defaults to limited range, bt470bg
// matrix and bt709 transfer. The CLI's --input-* flags override these defaults.
// Layout overflow and allocation failures throw; decoder resources are released
// in both the returned-error and exception paths.
// JPEG scans that require synthesized missing pixels are rejected; an absent
// end marker alone and benign metadata warnings retain libjpeg's behavior.
std::optional<Image> LoadImage(std::string_view path, std::string& error);

// Exact fixed frame rate declared by a Y4M stream. A zero numerator means no
// timing was declared (including independent PNG/JPEG images).
struct FrameRate {
    std::uint32_t numerator = 0;
    std::uint32_t denominator = 1;
    [[nodiscard]] bool operator==(const FrameRate&) const noexcept = default;
};

enum class SequenceReadResult { Frame, End, Error };

// Bounded-memory ordered reader. PNG/JPEG produce one frame; Y4M produces all
// frames and preserves its rational timing. Only progressive Y4M is accepted.
// Constructor errors throw std::runtime_error; Next distinguishes clean EOF
// from malformed/truncated input. An error is terminal. The image returned by
// Next owns its pixels and is independent of subsequent reads.
class SequenceReader {
public:
    explicit SequenceReader(std::string_view path);
    ~SequenceReader();
    SequenceReader(SequenceReader&&) noexcept;
    SequenceReader& operator=(SequenceReader&&) noexcept;
    SequenceReader(const SequenceReader&) = delete;
    SequenceReader& operator=(const SequenceReader&) = delete;

    [[nodiscard]] Size size() const noexcept;
    [[nodiscard]] const ImageFormat& format() const noexcept;
    [[nodiscard]] FrameRate frame_rate() const noexcept;
    [[nodiscard]] bool is_video() const noexcept;
    [[nodiscard]] SequenceReadResult Next(Image& image, std::string& error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Colour declarations and reconstruction settings shared by command-line tools.
// Only explicit overrides replace what the decoder declared; the chroma filter
// is passed separately to the selected metric's configuration.
struct InputColorOptions {
    std::optional<MatrixCoefficients> matrix;
    std::optional<TransferCharacteristics> transfer;
    std::optional<ColorPrimaries> primaries;
    std::optional<PixelRange> range;
    ChromaFilter chroma_filter = ChromaFilter::Bilinear;

    void ApplyTo(ImageFormat& format) const;
};

enum class InputColorOption { Matrix, Transfer, Primaries, Range, ChromaFilter };

// Recognize only the shared colour flags. The caller owns argument consumption
// and missing-value errors, keeping these options in its normal parse order.
[[nodiscard]] std::optional<InputColorOption> IdentifyInputColorOption(
    std::string_view flag
);

// Parse one recognized flag. `accepted` lists the supported spellings when the
// value is invalid; callers retain their own error presentation.
[[nodiscard]] bool ParseInputColorOption(
    InputColorOption option, std::string_view value, InputColorOptions& options,
    std::string& accepted
);

}  // namespace vqm::image_io
