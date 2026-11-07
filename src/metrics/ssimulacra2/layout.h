// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "metrics/ssimulacra2/ssimulacra2_internal.h"
#include "vulkan/vulkan_runtime.h"
#include "vulkan/memory_plan.h"

namespace vqm::internal::ssimulacra2_layout {

inline constexpr std::size_t kChannels = 3;
inline constexpr std::uint32_t kWorkgroup = 16;
inline constexpr std::uint32_t kSsimSums = 6;
inline constexpr std::uint32_t kEdgeSums = 12;
inline constexpr std::uint32_t kScaleSums = kSsimSums + kEdgeSums;
inline constexpr std::uint32_t kSsimBlockOffset = kSsimSums;
// The level table the shaders read through binding 2 (see level_u in
// src/metrics/ssimulacra2/shaders/ssimu2_common.glsl): one row of eight words
// per level, at ArenaRegions::levels.
inline constexpr std::uint32_t kLevelFields = 8;
inline constexpr std::size_t kLevelWords = kLevelFields * kSsimulacra2MaxScales;

struct ScaleVariant {
    std::uint32_t rows;
    std::uint32_t cols;
    std::uint32_t lanes;  // local_size_y; the workgroup is 16 x lanes
    std::uint32_t shared_bytes;
};

// Arena regions. Everything binding 0 addresses is in arena floats; the plane
// sets of binding 1 are in plane-buffer elements. Both numberings start at 0,
// so `img` is a plane offset while every other field is an arena offset.
//
// Allocation order: source[2], the descriptor block, the level table, the
// resident slots, the norm partials, the norms, then the plane region last.
// Every f32 region starts on a 16-float (64 B) boundary, and the plane region
// starts on a minStorageBufferOffsetAlignment boundary because it is a
// descriptor offset rather than a buffer index.
struct ArenaRegions {
    // One XYB plane set per image per level, three channel planes of that
    // level's own area. Levels past the geometry's scale count hold no plane:
    // nothing writes or reads them.
    std::uint32_t img[2][kSsimulacra2MaxScales]{};
    std::array<std::uint32_t, 2> source{};       // arena floats: 4 base planes
    std::array<std::uint32_t, 2> source_desc{};  // arena floats: one each
    std::vector<std::uint32_t> resident;  // arena floats: 4 base-sized planes
    std::uint32_t levels = 0;             // arena floats: the level table, as
                                          // words
    std::uint32_t partials = 0;
    std::uint32_t norms = 0;
    // Pixels per channel of each level, which is also that level's plane
    // channel stride (the planes of a level are three runs of this many
    // values).
    std::uint32_t channel_floats[kSsimulacra2MaxScales]{};
    std::uint32_t plane_floats = 0;  // plane region size, in plane elements
    VkDeviceSize plane_base_bytes = 0;
    VkDeviceSize plane_bytes = 0;
    VkDeviceSize bytes = 0;  // whole arena buffer
};

struct Ssimulacra2LayoutPlan {
    ArenaRegions regions;
    ScaleVariant scale_tile{};
    std::size_t base_pixels = 0;
    std::size_t levels = 0;
    std::array<std::uint32_t, kSsimulacra2MaxScales> level_groups{};
    std::array<std::uint32_t, kSsimulacra2MaxScales> level_partials{};
    std::size_t staging_norm_offset = 0;
    std::size_t staging_desc_offset = 0;
    std::size_t staging_level_offset = 0;
    std::uint32_t input_pool = 0;
    std::uint32_t input_pool_size = 0;
    StagingSources staging{};
};

// Shared by construction and memory queries; no Vulkan objects are created.
Ssimulacra2LayoutPlan BuildSsimulacra2Layout(
    const std::vector<Size>& sizes, const VulkanContext& context,
    std::size_t resident_images, bool half_planes, bool streaming,
    std::size_t input_bytes, ChromaFilter chroma_filter
);

}  // namespace vqm::internal::ssimulacra2_layout

namespace vqm::internal {

// Uses the same metric layout and dispatch checks as pipeline construction.
[[nodiscard]] BufferMemoryPlan PlanSsimulacra2Memory(
    const Ssimulacra2Config& config, const VulkanContext& context
);

}  // namespace vqm::internal
