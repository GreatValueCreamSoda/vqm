// Copyright 2001-2012 Xiph.Org and contributors.
// BSD-2-Clause; see licenses/XIPH-PSNR-HVS-LICENSE.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "psnr_hvs.h"
#include "vulkan/memory_plan.h"
#include "vulkan/vulkan_runtime.h"

namespace vqm::internal {

void ValidatePsnrHvsConfig(const PsnrHvsConfig& config);
void ValidatePsnrHvsFormat(const ImageFormat& format, Size size);

namespace psnr_hvs_layout {

// Read-only FP32 metadata: Y/Cb/Cr CSF tables, then the 8x8 DCT basis.
inline constexpr std::uint32_t kTableWords = 192 + 64;
inline constexpr std::uint32_t kPrepareLanes = 128;
inline constexpr std::uint32_t kBlockLanes = 64;
inline constexpr std::uint32_t kReduceLanes = 128;
// Target input items per first-level workgroup; final groups use the same
// shader.
inline constexpr std::uint32_t kReduceItemsPerGroup = 512;

// Mirrors psnr_hvs_common.glsl; offsets are four-byte arena words.
struct Push {
    std::uint32_t desc_base = 0;
    std::uint32_t plane = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t reference = 0;
    std::uint32_t distorted = 0;
    std::uint32_t destination = 0;
    std::uint32_t blocks_x = 0;
    std::uint32_t blocks = 0;
    std::uint32_t tables = 0;
};
static_assert(sizeof(Push) == 40);

struct Layout {
    std::uint32_t tables = 0;
    std::array<std::uint32_t, 2> prepared{};
    std::array<std::uint32_t, 3> partials{};
    std::array<std::uint32_t, 3> reduction{};
    std::uint32_t result = 0;
    std::uint32_t plane_stride = 0;
    std::uint32_t input_pool = 0;
    std::uint32_t input_pool_size = 0;
    std::array<std::uint32_t, 2> stream_region{};
    std::array<std::uint32_t, 2> stream_desc{};
    std::vector<std::uint32_t> resident;
    VkDeviceSize bytes = 0;
    StagingSources staging{};
    std::size_t staging_tables = 0;
    std::size_t staging_desc = 0;
    std::size_t staging_result = 0;
    std::size_t staging_words = 0;
};

[[nodiscard]] Layout BuildLayout(
    const PsnrHvsConfig& config, const VulkanContext& context
);

}  // namespace psnr_hvs_layout

[[nodiscard]] BufferMemoryPlan PlanPsnrHvsMemory(
    const PsnrHvsConfig& config, const VulkanContext& context
);

}  // namespace vqm::internal
