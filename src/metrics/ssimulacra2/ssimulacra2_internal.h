// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "ssimulacra2.h"

namespace vqm::internal {

// ---- SSIMULACRA 2 pyramid geometry ----
inline constexpr std::size_t kSsimulacra2Channels =
    3;  // X, Y, B / R, G, B planes
inline constexpr std::size_t kSsimulacra2MaxScales = 6;  // 1:1 ... 1:32
inline constexpr std::size_t kSsimulacra2MinDimension =
    8;  // smaller images yield no scale

// ---- separable Gaussian blur ----
inline constexpr double kSsimulacra2GaussianSigma =
    1.5;  // sigma for every norm plane
inline constexpr double kSsimulacra2GaussianSupport =
    3.5;  // taps span +/- support * sigma

// ---- error-map norm indices ----
inline constexpr std::size_t kSsimNorms = 2;       // 1-norm, 4-norm
inline constexpr std::size_t kEdgeNorms = 4;       // ringing 1/4, blurring 1/4
inline constexpr std::size_t kBlurNormOffset = 2;  // blurring sits after ringing

constexpr std::size_t SsimNorm(std::size_t channel, std::size_t norm) {
    return channel * kSsimNorms + norm;
}
constexpr std::size_t RingNorm(std::size_t channel, std::size_t norm) {
    return channel * kEdgeNorms + norm;
}
constexpr std::size_t BlurNorm(std::size_t channel, std::size_t norm) {
    return channel * kEdgeNorms + kBlurNormOffset + norm;
}

// True when every one of the six weights this (channel, scale) pair carries in
// the score's weight table is at most the pruning threshold, so the maps of
// that pair are omitted as a measured approximation. The answer depends on
// `scales` because Score() reads the weight table as one
// flat run of `scales` rows per channel, so the same (channel, scale) pair
// carries different weights on a 4-level pyramid than on a 6-level one.
//
// The rule is Vship's, and finer there: it decides each map on its own from
// that map's 1-norm and 4-norm weights
// (.reference/Vship/src/Vulkan/ssimu2/score.hpp:125-131), drops a whole
// (plane, scale) dispatch only when all three maps of the pair go (:181), and
// suppresses the downsample store on the same conjunction (main.hpp:32-42).
// Its weight table is indexed with a fixed six-scale stride because its
// pyramid is always six levels; ours is a flat run of `scales` rows, which is
// why the mask has to be rebuilt per geometry. One workgroup here produces all
// six norms of a pair at once, so the pair is the unit we can drop -- that is
// what this answers, and BuildChannelMasks in src/metrics/ssimulacra2/vulkan.cc
// is its only caller. Vship was consulted for the mechanism only.
bool Ssimulacra2ChannelPruned(
    std::size_t channel, std::size_t scale, std::size_t scales
);

// Symmetric separable Gaussian kernel; weights[i] is the tap at offset
// (i - radius) and the weights sum to 1.
struct GaussianKernel {
    int radius = 0;
    std::vector<float> weights;

    [[nodiscard]] std::size_t TapCount() const noexcept {
        return weights.size();
    }
};

// sigma = 1.5 is what the metric uses; any positive sigma works.
// Taps span +/- ceil(3.5 * sigma).
GaussianKernel MakeGaussian(double sigma);

}  // namespace vqm::internal

namespace vqm::internal {

// Scale pyramid for one geometry: sizes[0] is the base size, sizes[s + 1] is
// the 2x downsample of sizes[s].
struct Ssimulacra2ScaleLayout {
    std::size_t count = 0;
    const Size* sizes = nullptr;
};

// Owning scale geometry shared by construction and memory planning.
struct Ssimulacra2ScalePlan {
    std::size_t count = 0;
    std::array<Size, kSsimulacra2MaxScales> sizes{};
};
[[nodiscard]] Ssimulacra2ScalePlan PlanSsimulacra2Scales(Size size);

// Whole-metric implementation on a GPU. Owns its Vulkan objects; like
// Ssimulacra2Pipeline it is not thread-safe and serves one geometry.
class Ssimulacra2GpuPipeline {
public:
    virtual ~Ssimulacra2GpuPipeline() = default;
    virtual void RequireIdle() const = 0;
    virtual void ActivateShared() = 0;
    // Uploads one image and returns its resident slot index, or std::nullopt
    // when the resident pool is exhausted. The slot keeps the image's format,
    // so Submit needs nothing else. `capacity` is the facade's
    // current pool size, used only for the exhaustion check.
    virtual std::optional<std::uint32_t> Upload(
        const Image& image, std::size_t capacity
    ) = 0;
    virtual void Release(std::uint32_t slot) = 0;
    // Split submission: Submit records and queues the pair without waiting,
    // Finish waits for it and returns the score. One submission may be in
    // flight per pipeline; overlap comes from holding several pipelines (one
    // per worker), because a single pair leaves about half the SM issue slots
    // idle and a second pipeline fills them.
    [[nodiscard]] virtual std::uint64_t Submit(
        const Image& original, const Image& distorted, float background
    ) = 0;
    [[nodiscard]] virtual std::uint64_t Submit(
        const ImageView& original, const ImageView& distorted, float background
    ) = 0;
    [[nodiscard]] virtual std::uint64_t Submit(
        std::uint32_t original, std::uint32_t distorted, float background
    ) = 0;
    [[nodiscard]] virtual std::uint64_t Submit(
        std::uint32_t original, const Image& distorted, float background
    ) = 0;
    [[nodiscard]] virtual Ssimulacra2Result Finish(std::uint64_t token) = 0;
    [[nodiscard]] virtual PipelineMemoryUsage GetMemoryUsage() const = 0;

    // Caller-owned input memory, carved from Ssimulacra2Config::input_bytes.
    [[nodiscard]] virtual ImageBuffer AllocateImage(
        const ImageFormat& format
    ) = 0;
    [[nodiscard]] virtual std::byte* MapImage(ImageBuffer buffer) = 0;
    virtual void Release(ImageBuffer buffer) = 0;
    [[nodiscard]] virtual std::uint64_t Submit(
        const ImageBuffer& original, const ImageBuffer& distorted,
        float background
    ) = 0;
    [[nodiscard]] virtual std::uint64_t Submit(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        float background
    ) = 0;
    [[nodiscard]] virtual std::uint64_t Submit(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        const VulkanSubmitSync& sync, float background
    ) = 0;
};

}  // namespace vqm::internal
