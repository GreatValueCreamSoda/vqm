// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// Butteraugli portions: Copyright (c) the JPEG XL Project Authors.
// BSD-3-Clause; see licenses/JPEG-XL-LICENSE.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "metrics/butteraugli/butteraugli_internal.h"
#include "vulkan/vulkan_runtime.h"
#include "vulkan/memory_plan.h"

namespace vqm::internal::butteraugli_layout {

// Region table fields, one 16-word row per level (butter_common.glsl).
enum : std::uint32_t {
    R_CHAIN0 = 0,
    R_CHAIN1 = 1,
    R_SCRATCH_A = 2,
    R_SCRATCH_B = 3,
    R_AC = 4,
    R_DIFFMAP = 5,
    R_SUB0 = 6,
    R_SUB1 = 7,
    R_WIDTH = 8,
    R_HEIGHT = 9,
    R_AREA = 10,
    R_SUBW = 11,
    R_SUBH = 12,
    R_PARTIALS = 13,
    kRowWords = 16,
};
inline constexpr std::uint32_t kRows = 2;
inline constexpr std::size_t kTableWords = kRowWords * kRows;

// Blur tap counts and where their normalized taps live in the arena.
inline constexpr std::uint32_t kOpsinTaps = 5;
inline constexpr std::uint32_t kUhfTaps = 7;
inline constexpr std::uint32_t kMaskTaps = 13;
inline constexpr std::uint32_t kHfTaps = 15;
inline constexpr std::uint32_t kLfTaps = 33;
inline constexpr std::uint32_t kKernelOpsin = 0;
inline constexpr std::uint32_t kKernelUhf = kKernelOpsin + kOpsinTaps;
inline constexpr std::uint32_t kKernelMask = kKernelUhf + kUhfTaps;
inline constexpr std::uint32_t kKernelHf = kKernelMask + kMaskTaps;
inline constexpr std::uint32_t kKernelLf = kKernelHf + kHfTaps;
inline constexpr std::uint32_t kKernelFloats = kKernelLf + kLfTaps;

// One chain is 12 contiguous planes: lf 3, mf 3, hf 2 (the UHF input), hff 2
// (the post-UHF high frequency) and uhf 2.
inline constexpr std::uint32_t kChainPlanes = 12;

struct LevelLayout {
    Size size{};
    std::uint32_t area = 0;
    std::uint32_t chain[2]{};
    std::uint32_t scratch_a = 0;
    std::uint32_t scratch_b = 0;
    std::uint32_t ac = 0;
    std::uint32_t diffmap = 0;
};

struct Layout {
    LevelLayout level[2];
    std::uint32_t levels = 0;
    std::uint32_t sub0 = 0;
    std::uint32_t sub1 = 0;
    std::uint32_t sub_area = 0;
    std::uint32_t partials = 0;
    std::uint32_t partial_groups = 0;
    std::uint32_t final_result = 0;
    std::uint32_t input_pool = 0;
    std::uint32_t input_pool_size = 0;
    std::uint32_t table_word = 0;
    std::array<std::uint32_t, kTableWords> table{};
    std::array<std::uint32_t, 2> stream_region{};
    std::array<std::uint32_t, 2> stream_desc{};
    std::vector<std::uint32_t> resident;
    VkDeviceSize bytes = 0;
};

struct ButterLayoutPlan {
    std::vector<PersistentRange> persistent_ranges;
    Layout layout;
    std::size_t base_pixels = 0;
    std::size_t staging_desc = 0;
    std::size_t staging_kernels = 0;
    std::size_t staging_table = 0;
    std::size_t staging_partials = 0;
    StagingSources staging{};
};

// Shared by construction and memory queries; no Vulkan objects are created.
ButterLayoutPlan BuildButterLayout(
    const ButteraugliGeometry& geometry, const VulkanContext& context,
    bool streaming, std::size_t resident_images, std::size_t input_bytes,
    ChromaFilter chroma_filter
);

}  // namespace vqm::internal::butteraugli_layout

namespace vqm::internal {

// Uses the same metric layout and dispatch checks as pipeline construction.
[[nodiscard]] BufferMemoryPlan PlanButteraugliMemory(
    const ButteraugliConfig& config, const VulkanContext& context
);

}  // namespace vqm::internal
