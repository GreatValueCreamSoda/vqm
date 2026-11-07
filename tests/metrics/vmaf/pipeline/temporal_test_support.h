// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <span>
#include <vector>

#include "support.h"

namespace vqm::test::temporal {

struct Pair {
    Image original;
    Image distorted;
};

// Uneven transitions, a freeze, and a larger final jump distinguish the
// look-ahead minimum from either adjacent transition and from stale history.
inline constexpr std::array kOffsets{0, 7, 5, 17, 17, 21, 18, 27, 11};

inline Image MakeImage(Size size) {
    ImageFormat format;
    format.type = PixelType::Word;
    format.depth = 10;
    format.family = ColorFamily::Yuv;
    format.matrix = MatrixCoefficients::Bt709;
    format.range = PixelRange::Limited;
    Image image{size, format, {}};
    image.data.resize(RequiredBytes(format, size));
    return image;
}

inline void FillImage(Image& image, std::size_t frame, bool distorted) {
    const auto planes = internal::PlaneLayout(image.format, image.size);
    for (std::size_t plane = 0; plane < 3; ++plane) {
        for (std::size_t y = 0; y < planes[plane].height; ++y) {
            for (std::size_t x = 0; x < planes[plane].width; ++x) {
                int value = plane == 0
                                ? 160 + static_cast<int>((3 * x + 5 * y) % 480) +
                                      4 * kOffsets[frame % kOffsets.size()]
                                : 320 + static_cast<int>(
                                            (7 * x + 11 * y + 97 * plane) % 320
                                        );
                if (distorted) {
                    const auto noise =
                        static_cast<int>((13 * x + 17 * y + 19 * plane) % 31) -
                        15;
                    value += static_cast<int>(frame % 5 + 1) * noise;
                }
                const auto sample = static_cast<std::uint16_t>(value);
                const auto offset =
                    planes[plane].offset + 2 * (y * planes[plane].width + x);
                std::memcpy(image.data.data() + offset, &sample, sizeof(sample));
            }
        }
    }
}

inline std::vector<Pair> MakeClip(Size size, std::size_t count) {
    std::vector<Pair> pairs;
    for (std::size_t i = 0; i < count; ++i) {
        Pair pair{MakeImage(size), MakeImage(size)};
        FillImage(pair.original, i, false);
        FillImage(pair.distorted, i, true);
        pairs.push_back(std::move(pair));
    }
    return pairs;
}

template <typename Frame, typename Pipeline>
std::vector<Frame> RunClip(Pipeline& pipeline, std::span<const Pair> pairs) {
    std::vector<Frame> frames;
    for (const auto& pair : pairs) {
        if (const auto frame = pipeline.Push(pair.original, pair.distorted))
            frames.push_back(*frame);
    }
    if (const auto last = pipeline.Flush()) frames.push_back(*last);
    return frames;
}

template <typename Frame, typename Pipeline>
void CheckPooled(const Pipeline& pipeline, std::span<const Frame> frames) {
    const auto pooled = pipeline.PooledScore();
    CHECK(pooled.has_value() == !frames.empty());
    if (!pooled) return;
    const double sum = std::accumulate(
        frames.begin(), frames.end(), 0.0,
        [](double value, const Frame& frame) { return value + frame.score; }
    );
    CHECK_NEAR(*pooled, sum / static_cast<double>(frames.size()), 1e-12);
}

inline double ExpectedMotion(std::size_t frame, std::size_t count, bool v1) {
    if (count == 1 || (frame == 0 && !v1)) return 0.0;
    const auto previous = frame == 0 ? 1 : frame;
    const auto left = std::abs(kOffsets[previous] - kOffsets[previous - 1]);
    if (frame + 1 == count) return static_cast<double>(left);
    const auto right = std::abs(kOffsets[frame + 1] - kOffsets[frame]);
    return static_cast<double>(std::min(left, right));
}

}  // namespace vqm::test::temporal
