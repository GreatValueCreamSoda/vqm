// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// Butteraugli portions: Copyright (c) the JPEG XL Project Authors.
// BSD-3-Clause; see licenses/JPEG-XL-LICENSE.

// Butteraugli, as libjxl defines it. This header holds the internal contract
// between the public facade (src/metrics/butteraugli/pipeline.cc) and the Vulkan
// backend (src/metrics/butteraugli/vulkan.cc), plus every constant and host-side
// table the shaders consume. The metric is libjxl's "ButteraugliComparator" path
// (BSD-3-Clause; .reference/libjxl/lib/jxl/butteraugli/butteraugli.cc),
// translated here with the source line named next to every constant. Nothing is
// copied from .reference/Vship* (MIT with a no-AI clause); those checkouts were
// read for mechanisms only, and bench/notes/system/vship-notes.md records the
// comparison.
//
// Shapes, as libjxl computes them:
//   - two levels: the base size and one 2x2 box subsample of the *linear rgb*
//     (ButteraugliDiffmapSmall pads below 8x8; the pipeline enforces 8x8 like
//     SSIMULACRA 2 and the CPU reference implements the padding for its 1x1
//     anchor test). ButteraugliComparator::Diffmap (:1829) folds exactly one
//     sub-level at :1852-1857 (`sub_->DiffmapOpsinDynamicsImage` then
//     `AddSupersampled2x(subresult, 0.5, result)`) -- the deeper sub_ chain
//     Make builds is never consulted -- so the fold is
//     result = 0.85*base + 0.5*sub[y/2][x/2] (AddSupersampled2x, :1768).
//   - per level: opsin dynamics (:1468), the LF/MF/HF/UHF separation (:404),
//     six Malta band terms (:1910-1928), the L2 terms (:1320), the
//     psycho-visual mask (:1110) and one combine (:1268). No final blur.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "butteraugli.h"

namespace vqm {
struct ButteraugliConfig;
}

namespace vqm::internal {

// ButteraugliParams, butteraugli.h:34-44, with libjxl's defaults. Vship
// defaults intensity_target to 203 nits; libjxl's 80 is the reference value
// both the CPU oracle and the shaders use, and the public API can change it.
struct ButteraugliParams {
    float hf_asymmetry = 1.0f;
    float xmul = 1.0f;
    float intensity_target = 80.0f;
};

// The two levels a geometry runs. `fold` is true when the sub level is at
// least 8x8 and therefore contributes (butteraugli.cc:1852-1857).
struct ButteraugliGeometry {
    Size base{};
    Size sub{};
    std::uint32_t levels = 0;  // 1 or 2
    bool fold = false;
};

// Butteraugli requires at least 8x8 pixels. Below that size the upstream
// comparator returns a zero diffmap.
ButteraugliGeometry PlanButteraugliGeometry(Size base);
void ValidateButteraugliConfig(const ButteraugliConfig& config);

// The result: the canonical score is the diffmap's maximum
// (ButteraugliScoreFromDiffmap, :1954-1963); distance_p3 is libjxl's
// ComputeDistanceP(p=3) aggregate (lib/extras/metrics.cc:42-145), the number
// cjxl reports next to it.
struct ButteraugliResult {
    double score = 0.0;
    double distance_p3 = 0.0;
};

// One side of the pair. `generation` comes from the caller's handle and keys
// the cached reference chain: a handle whose slot was released and re-uploaded
// carries a new generation, so a stale cached reference can never be scored.
struct ButteraugliSource {
    enum class Kind : std::uint8_t {
        Stream,       // copied into the arena's source region
        Resident,     // an Upload() slot
        ImageBuffer,  // a caller-owned carve-out
    };
    Kind kind = Kind::Stream;
    std::uint32_t index = 0;
    std::uint64_t generation = 0;
};

// Host-side tables the shaders read once. All values are computed in double
// and stored as float, which is the precision policy: fp32 arithmetic on the
// device, double-precision constants (kernel taps, Malta weights) from here.
struct ButterTables {
    // Normalized separable Gaussian taps per blur, computed with libjxl's
    // ComputeKernel rule (:78-88): radius = max(1, int(2.25*sigma)) in double,
    // then normalized to sum 1.
    std::array<float, 5> opsin{};  // sigma 1.2, 5 taps, mirrored borders
    std::array<float, 7> uhf{};    // sigma 1.56416327805
    std::array<float, 13> mask{};  // sigma 2.7
    std::array<float, 15> hf{};    // sigma 3.22489901262
    std::array<float, 33> lf{};    // sigma 7.15593339443
};

[[nodiscard]] ButterTables MakeButterTables();

// The six Malta band terms with the params applied, in libjxl's order
// (butteraugli.cc:1910-1928): UHF Y, UHF X, HF Y, HF X, MF Y, MF X. Each
// entry carries the two prefactors the shader divides by (norm1 + absval):
//   norm2 = mulli * sqrt(kWeight * w) / (2*len + 1) * norm1
// with kWeight 0.5 for the too-big term and 0.33 for the too-small one, len
// 3.75, mulli 0.39905817637 (16-line unit) or 0.611612573796 (LF unit).
// Everything is computed in double here and stored as float.
struct ButterBands {
    double norm1[6]{};
    double norm2_gt1[6]{};
    double norm2_lt1[6]{};
};

[[nodiscard]] ButterBands MakeButterBands(const ButteraugliParams& params);

// The GPU backend for one geometry. Like Ssimulacra2GpuPipeline it owns its
// Vulkan objects, is not thread-safe, serves one geometry and keeps one
// submission in flight; the contracts (Finish on an unknown token throws
// std::invalid_argument, a second Submit/Upload/... while one is in flight
// throws std::logic_error) are the same.
class ButteraugliGpuPipeline {
public:
    virtual ~ButteraugliGpuPipeline() = default;
    virtual void RequireIdle() const = 0;
    virtual void ActivateShared() = 0;

    [[nodiscard]] virtual std::uint64_t Submit(
        const Image& original, const Image& distorted, float background
    ) = 0;
    [[nodiscard]] virtual std::uint64_t Submit(
        const ImageView& original, const ImageView& distorted, float background
    ) = 0;
    [[nodiscard]] virtual std::uint64_t Submit(
        const ButteraugliSource& original, const ButteraugliSource& distorted,
        float background
    ) = 0;
    [[nodiscard]] virtual std::uint64_t Submit(
        const ButteraugliSource& original, const Image& distorted,
        float background
    ) = 0;
    [[nodiscard]] virtual ButteraugliResult Finish(std::uint64_t token) = 0;
    [[nodiscard]] virtual PipelineMemoryUsage GetMemoryUsage() const = 0;

    virtual std::optional<std::uint32_t> Upload(
        const Image& image, std::size_t capacity
    ) = 0;
    virtual void ReleaseSlot(std::uint32_t slot) = 0;

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

// Null only when no Vulkan device could be acquired.

}  // namespace vqm::internal
