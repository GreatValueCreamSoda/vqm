// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <format>
#include <source_location>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "input.h"
#include "vqm.h"
#include "vulkan/vulkan_support.h"

namespace vqm::test {

struct TestCase {
    std::string_view name;
    void (*run)();
};

void RegisterTest(std::string_view name, void (*run)());
void Fail(std::source_location where, std::string_view message);
void MarkSkip(std::string_view reason, bool missing_gpu = false);
int RunAll(bool require_gpu);

// Registers a test in its translation unit's static initialization.
struct Registrar {
    Registrar(std::string_view name, void (*run)()) { RegisterTest(name, run); }
};

}  // namespace vqm::test

#define VQM_TEST(name)                                                \
    static void name();                                               \
    static const ::vqm::test::Registrar vqm_reg_##name{#name, &name}; \
    static void name()

#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            ::vqm::test::Fail(std::source_location::current(), #cond); \
        }                                                              \
    } while (false)

#define CHECK_NEAR(actual, expected, tolerance)                            \
    do {                                                                   \
        const double vqm_actual_ = (actual);                               \
        const double vqm_expected_ = (expected);                           \
        const double vqm_tolerance_ = (tolerance);                         \
        if (!(std::fabs(vqm_actual_ - vqm_expected_) <= vqm_tolerance_)) { \
            ::vqm::test::Fail(                                             \
                std::source_location::current(),                           \
                std::format(                                               \
                    "{} = {:.12g}, expected {:.12g} +/- {:.12g}", #actual, \
                    vqm_actual_, vqm_expected_, vqm_tolerance_             \
                )                                                          \
            );                                                             \
        }                                                                  \
    } while (false)

#define CHECK_THROWS_AS(expr, Exception)           \
    do {                                           \
        bool vqm_thrown_ = false;                  \
        try {                                      \
            (void)(expr);                          \
        } catch (const Exception&) {               \
            vqm_thrown_ = true;                    \
        } catch (...) {                            \
        }                                          \
        if (!vqm_thrown_) {                        \
            ::vqm::test::Fail(                     \
                std::source_location::current(),   \
                #expr " did not throw " #Exception \
            );                                     \
        }                                          \
    } while (false)

// Shared fixtures for the test suite. Everything here is inline so several test
// translation units can use it.

namespace vqm::test {

inline constexpr std::size_t kSide = 32;

inline int BaseRed(std::size_t x) {
    return static_cast<int>((x * 255) / (kSide - 1));
}
inline int BaseGreen(std::size_t y) {
    return static_cast<int>((y * 255) / (kSide - 1));
}
inline int BaseBlue(std::size_t x, std::size_t y) {
    return static_cast<int>(((x * 7 + y * 11) * 13) & 255);
}

// The format the CLI decoders pin for sRGB PNG/JPEG input.
inline vqm::ImageFormat PinnedFormat(
    AlphaType alpha, PixelType type = PixelType::Float
) {
    vqm::ImageFormat format;
    format.type = type;
    format.family = ColorFamily::Rgb;
    format.alpha = alpha;
    format.matrix = MatrixCoefficients::Rgb;
    format.transfer = TransferCharacteristics::Srgb;
    format.primaries = ColorPrimaries::Bt709;
    return format;
}

// Round-to-nearest-even float -> binary16, for building Half fixtures.
inline std::uint16_t FloatToHalf(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::int32_t exponent =
        static_cast<std::int32_t>((bits >> 23) & 0xFFu);
    std::uint32_t mantissa = bits & 0x7FFFFFu;
    if (exponent == 0xFF) {
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    const std::int32_t biased = exponent - 127 + 15;
    if (biased >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
    if (biased <= 0) {
        if (biased < -10) return static_cast<std::uint16_t>(sign);
        mantissa |= 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - biased);
        std::uint32_t half = mantissa >> shift;
        const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (half & 1u) != 0u)) {
            ++half;
        }
        return static_cast<std::uint16_t>(sign | half);
    }
    std::uint32_t half =
        (static_cast<std::uint32_t>(biased) << 10) | (mantissa >> 13);
    const std::uint32_t remainder = mantissa & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u) != 0u)) {
        ++half;
    }
    return static_cast<std::uint16_t>(sign | half);
}

inline float HalfToFloat(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
    const std::uint32_t exponent = (value >> 10) & 0x1Fu;
    std::uint32_t mantissa = value & 0x3FFu;
    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            // Subnormal: normalise it.
            std::int32_t shift = 0;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x3FFu;
            const std::int32_t biased = 127 - 14 - shift;
            bits = sign | (static_cast<std::uint32_t>(biased) << 23) |
                   (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// Quantises normalised [0, 1] values (the metric's decoded domain) to the
// format's sample type, depth and range: the inverse of the shader's decode.
inline vqm::Image PackPlanes(
    const vqm::ImageFormat& format, vqm::Size size,
    const std::array<std::vector<float>, 4>& values
) {
    vqm::Image img;
    img.size = size;
    img.format = format;
    img.data.assign(vqm::RequiredBytes(format, size), 0u);
    const std::array<internal::PlaneGeometry, 4> layout =
        internal::PlaneLayout(format, size);
    const std::uint8_t depth = internal::EffectiveDepth(format);
    // Only integer samples have a code range to stay inside; a float sample is
    // the value itself, including the linear values above 1 that PQ and HLG
    // decode to and the negative ones an out-of-gamut conversion can produce.
    const bool integer =
        format.type == PixelType::Byte || format.type == PixelType::Word;
    const double max_code =
        integer ? static_cast<double>((1u << depth) - 1u) : 0.0;
    for (std::size_t plane = 0; plane < 4; ++plane) {
        if (!layout[plane].present()) continue;
        const std::size_t samples = layout[plane].width * layout[plane].height;
        if (values[plane].size() < samples) {
            throw std::invalid_argument(
                "PackPlanes: a plane of values is shorter than the layout"
            );
        }
        const internal::ScaleOffset so = internal::PlaneScaleOffset(
            format,
            (plane == 1 || plane == 2) && format.family == ColorFamily::Yuv,
            plane == 3
        );
        std::uint8_t* base = img.data.data() + layout[plane].offset;
        for (std::size_t i = 0; i < samples; ++i) {
            const double decoded = values[plane][i];
            const double raw = (decoded - so.offset) / so.scale;
            const double code = integer ? std::clamp(raw, 0.0, max_code) : raw;
            switch (format.type) {
                case PixelType::Byte: {
                    const auto sample =
                        static_cast<std::uint8_t>(std::lround(code));
                    std::memcpy(base + i, &sample, 1);
                    break;
                }
                case PixelType::Word: {
                    const auto sample =
                        static_cast<std::uint16_t>(std::lround(code));
                    std::memcpy(base + i * 2, &sample, 2);
                    break;
                }
                case PixelType::Half: {
                    const std::uint16_t sample =
                        FloatToHalf(static_cast<float>(code));
                    std::memcpy(base + i * 2, &sample, 2);
                    break;
                }
                case PixelType::Float: {
                    const auto sample = static_cast<float>(code);
                    std::memcpy(base + i * 4, &sample, 4);
                    break;
                }
            }
        }
    }
    return img;
}

inline vqm::Image MakeImage(
    std::vector<float> r, std::vector<float> g, std::vector<float> b
) {
    std::array<std::vector<float>, 4> values{
        std::move(r), std::move(g), std::move(b), {}};
    return PackPlanes(PinnedFormat(AlphaType::None), {kSide, kSide}, values);
}

// int variant: 0 = gradient, 1 = gradient_B, 2 = gradient_C. An 8-bit image,
// so the shader sees exactly the floats the old float planes carried.
inline vqm::Image MakeGradient(int variant) {
    vqm::Image img;
    img.size = {kSide, kSide};
    img.format = PinnedFormat(AlphaType::None, PixelType::Byte);
    img.data.assign(vqm::RequiredBytes(img.format, img.size), 0u);
    const std::array<internal::PlaneGeometry, 4> layout =
        internal::PlaneLayout(img.format, img.size);
    std::uint8_t* planes[3] = {img.data.data() + layout[0].offset,
                               img.data.data() + layout[1].offset,
                               img.data.data() + layout[2].offset};
    for (std::size_t y = 0; y < kSide; ++y) {
        for (std::size_t x = 0; x < kSide; ++x) {
            int r8 = BaseRed(x);
            int g8 = BaseGreen(y);
            int b8 = BaseBlue(x, y);
            if (variant == 1 && ((x + y) & 1) == 0) r8 = std::min(r8 + 16, 255);
            if (variant == 2) {
                r8 &= 0xF8;
                g8 &= 0xF8;
                b8 &= 0xF8;
            }
            const std::size_t i = y * kSide + x;
            planes[0][i] = static_cast<std::uint8_t>(r8);
            planes[1][i] = static_cast<std::uint8_t>(g8);
            planes[2][i] = static_cast<std::uint8_t>(b8);
        }
    }
    return img;
}

inline vqm::Image MakeRampImage(
    std::size_t width, std::size_t height, bool with_alpha,
    PixelType type = PixelType::Float
) {
    std::array<std::vector<float>, 4> values;
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            values[0].push_back(
                static_cast<float>(x) / static_cast<float>(width - 1)
            );
            values[1].push_back(
                static_cast<float>(y) / static_cast<float>(height - 1)
            );
            values[2].push_back(
                static_cast<float>((x * 7 + y * 11) % 256) / 255.0f
            );
            if (with_alpha) {
                values[3].push_back(static_cast<float>((x + 3 * y) % 7) / 6.0f);
            }
        }
    }
    return PackPlanes(
        PinnedFormat(with_alpha ? AlphaType::Straight : AlphaType::None, type),
        {width, height}, values
    );
}

inline double RelativeError(double actual, double expected) {
    return std::fabs(actual - expected) / std::max(std::fabs(expected), 1e-12);
}

// Every avg_ssim and avg_edgediff entry of every scale within `tolerance`
// relative, and the score within twice that: the score mapping is steeper
// than the norms, so the norms are the meaningful bound.
inline bool NormsNear(
    const vqm::Ssimulacra2Result& a, const vqm::Ssimulacra2Result& b,
    double tolerance
) {
    if (a.scales.size() != b.scales.size()) return false;
    for (std::size_t s = 0; s < a.scales.size(); ++s) {
        for (std::size_t n = 0; n < a.scales[s].avg_ssim.size(); ++n) {
            if (RelativeError(a.scales[s].avg_ssim[n], b.scales[s].avg_ssim[n]) >
                tolerance) {
                return false;
            }
        }
        for (std::size_t n = 0; n < a.scales[s].avg_edgediff.size(); ++n) {
            if (RelativeError(
                    a.scales[s].avg_edgediff[n], b.scales[s].avg_edgediff[n]
                ) > tolerance) {
                return false;
            }
        }
    }
    return RelativeError(a.Score(), b.Score()) <= 2.0 * tolerance;
}

// The metric has no CPU backend, so every device test starts here.
inline bool GpuAvailable() {
    if (!internal::AcquireVulkanContext()) {
        MarkSkip("no usable Vulkan device", true);
        return false;
    }
    return true;
}

// True when the device can run the 16-bit plane path.
inline bool HalfPlanesAvailable() {
    const auto context = internal::AcquireVulkanContext();
    if (!context) return GpuAvailable();
    if (!context->sixteen_bit_storage) {
        MarkSkip("16-bit planes unavailable");
        return false;
    }
    // Construction errors on supported hardware are test failures, not skips.
    vqm::Ssimulacra2Pipeline probe({{8, 8}, 2, true});
    return true;
}

}  // namespace vqm::test
