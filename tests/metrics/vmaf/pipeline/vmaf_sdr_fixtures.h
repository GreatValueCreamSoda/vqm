// VMAF: Copyright (c) Netflix, Inc. BSD-2-Clause-Patent;
// see licenses/NETFLIX-VMAF-LICENSE.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "input.h"
#include "support.h"
#include "vqm_image.h"

namespace vqm::test::sdr {

// The CPU oracle below deliberately starts with linear BT.709 inputs. This
// keeps the expected display transfer, RGB matrix and 2x2 chroma box independent
// of the GPU implementation while avoiding a fixture-specific color profile.
constexpr Size kSize{320, 256};

inline void WriteSample(
    Image& image, unsigned plane, std::size_t index, float v
) {
    const auto p = internal::PlaneLayout(image.format, image.size);
    if (image.format.type == PixelType::Byte) {
        image.data[p[plane].offset + index] = static_cast<std::uint8_t>(
            std::nearbyint(std::clamp(v, 0.0f, 1.0f) * 255.0f)
        );
    } else if (image.format.type == PixelType::Word ||
               image.format.type == PixelType::Half) {
        const auto sample =
            image.format.type == PixelType::Half
                ? FloatToHalf(v)
                : static_cast<std::uint16_t>(
                      std::nearbyint(std::clamp(v, 0.0f, 1.0f) * 65535.0f)
                  );
        std::memcpy(
            image.data.data() + p[plane].offset + index * sizeof(sample),
            &sample, sizeof(sample)
        );
    } else {
        std::memcpy(
            image.data.data() + p[plane].offset + index * sizeof(float), &v,
            sizeof(v)
        );
    }
}

inline float ReadSample(const Image& image, unsigned plane, std::size_t index) {
    const auto p = internal::PlaneLayout(image.format, image.size);
    if (image.format.type == PixelType::Byte)
        return image.data[p[plane].offset + index] / 255.0f;
    if (image.format.type == PixelType::Word ||
        image.format.type == PixelType::Half) {
        std::uint16_t sample;
        std::memcpy(
            &sample,
            image.data.data() + p[plane].offset + index * sizeof(sample),
            sizeof(sample)
        );
        return image.format.type == PixelType::Half ? HalfToFloat(sample)
                                                    : sample / 65535.0f;
    }
    float v;
    std::memcpy(
        &v, image.data.data() + p[plane].offset + index * sizeof(float),
        sizeof(v)
    );
    return v;
}

inline Image MakeSource(
    PixelType type, ColorFamily family, AlphaType alpha, bool distorted,
    Size size = kSize
) {
    ImageFormat format;
    format.type = type;
    format.family = family;
    format.alpha = alpha;
    if (family == ColorFamily::Grey) format.matrix = MatrixCoefficients::Bt709;
    format.transfer = TransferCharacteristics::Linear;
    format.primaries = ColorPrimaries::Bt709;
    Image image{size, format, {}};
    image.data.resize(RequiredBytes(format, size));
    const unsigned channels = family == ColorFamily::Grey ? 1 : 3;
    for (std::size_t y = 0; y < size.height; ++y) {
        for (std::size_t x = 0; x < size.width; ++x) {
            const std::size_t index = y * size.width + x;
            const float opacity =
                static_cast<float>(64 + ((x / 7 + y / 11) % 192)) / 255.0f;
            if (alpha != AlphaType::None) WriteSample(image, 3, index, opacity);
            for (unsigned channel = 0; channel < channels; ++channel) {
                unsigned value =
                    20 + static_cast<unsigned>(
                             (x * (channel + 3) + y * (channel + 5) +
                              (x / 9) * (y / 11) + channel * 37) %
                             210
                         );
                if (distorted) value = (value / 7) * 7;
                float sample = static_cast<float>(value) / 255.0f;
                if (alpha == AlphaType::Premultiplied) sample *= opacity;
                WriteSample(image, channel, index, sample);
            }
        }
    }
    return image;
}

inline double Encode709(double linear) {
    return std::pow(std::max(0.0, linear), 1.0 / 2.4);
}

inline std::array<double, 3> ExpectedYuv(
    const Image& source, std::size_t index, double background
) {
    const double alpha = source.format.alpha == AlphaType::None
                             ? 1.0
                             : ReadSample(source, 3, index);
    std::array<double, 3> rgb{};
    for (unsigned channel = 0; channel < 3; ++channel) {
        double sample = ReadSample(
            source, source.format.family == ColorFamily::Grey ? 0 : channel,
            index
        );
        if (source.format.alpha == AlphaType::Premultiplied)
            sample = alpha == 0 ? 0 : sample / alpha;
        sample = sample * alpha + (1 - alpha) * background;
        rgb[channel] = Encode709(sample);
    }
    const double y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    return {16.0 + 219.0 * y, 128.0 + (112.0 / 0.9278) * (rgb[2] - y),
            128.0 + (112.0 / 0.7874) * (rgb[0] - y)};
}

inline Image ReferenceYuv(
    const Image& source, bool subsampled, double background = 0.5
) {
    ImageFormat format;
    format.type = PixelType::Word;
    format.depth = 16;
    format.family = ColorFamily::Yuv;
    format.range = PixelRange::Limited;
    format.matrix = MatrixCoefficients::Bt709;
    format.transfer = TransferCharacteristics::Bt709;
    format.primaries = ColorPrimaries::Bt709;
    format.subsample_w = subsampled ? 1 : 0;
    format.subsample_h = subsampled ? 1 : 0;
    Image result{source.size, format, {}};
    result.data.resize(RequiredBytes(format, result.size));
    const auto planes = internal::PlaneLayout(format, result.size);
    auto store = [&](unsigned plane, std::size_t index, double value) {
        const auto code = static_cast<std::uint16_t>(
            std::nearbyint(std::clamp(value * 256.0, 0.0, 65535.0))
        );
        std::memcpy(
            result.data.data() + planes[plane].offset + index * sizeof(code),
            &code, sizeof(code)
        );
    };
    for (std::size_t y = 0; y < source.size.height; y += 2) {
        for (std::size_t x = 0; x < source.size.width; x += 2) {
            std::array<double, 3> sum{};
            for (unsigned iy = 0; iy < 2; ++iy) {
                for (unsigned ix = 0; ix < 2; ++ix) {
                    const auto index = (y + iy) * source.size.width + x + ix;
                    const auto color = ExpectedYuv(source, index, background);
                    store(0, index, color[0]);
                    for (unsigned p = 1; p < 3; ++p) {
                        sum[p] += color[p];
                        if (!subsampled) store(p, index, color[p]);
                    }
                }
            }
            if (subsampled) {
                const auto index = (y / 2) * (source.size.width / 2) + x / 2;
                store(1, index, sum[1] / 4);
                store(2, index, sum[2] / 4);
            }
        }
    }
    return result;
}

}  // namespace vqm::test::sdr
