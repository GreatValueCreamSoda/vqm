// Copyright 2001-2012 Xiph.Org and contributors.
// BSD-2-Clause; see licenses/XIPH-PSNR-HVS-LICENSE.

#include "metrics/psnr_hvs/layout.h"

#include <limits>
#include <stdexcept>

#include "handles.h"
#include "input.h"

namespace vqm::internal {

void ValidatePsnrHvsConfig(const PsnrHvsConfig& config) {
    ValidateMemoryPolicy(config.memory_policy);
    constexpr auto maximum = std::numeric_limits<std::uint32_t>::max();
    if (config.size.width < 8 || config.size.height < 8 ||
        config.size.width > maximum / config.size.height ||
        config.size.width * config.size.height > maximum / 16u) {
        throw std::invalid_argument(
            "psnr_hvs: dimensions must be at least 8 and fit arena addressing"
        );
    }
    if (config.gpu_resident_images > kResidentIndexMask)
        throw std::invalid_argument("psnr_hvs: too many resident images");
}

void ValidatePsnrHvsFormat(const ImageFormat& format, Size size) {
    ValidateImage(format, size, "psnr_hvs");
    const std::uint8_t depth = EffectiveDepth(format);
    const bool integer =
        (format.type == PixelType::Byte && depth == 8) ||
        (format.type == PixelType::Word && (depth == 10 || depth == 12));
    const bool sampling = (format.subsample_w == 0 && format.subsample_h == 0) ||
                          (format.subsample_w == 1 && format.subsample_h <= 1);
    if (format.family != ColorFamily::Yuv || format.alpha != AlphaType::None ||
        !integer || !sampling) {
        throw std::invalid_argument(
            "psnr_hvs: opaque 8-bit byte or 10/12-bit word YUV420/422/444 "
            "inputs are required"
        );
    }
    if ((size.width >> format.subsample_w) < 8 ||
        (size.height >> format.subsample_h) < 8) {
        throw std::invalid_argument(
            "psnr_hvs: every plane must contain a complete 8x8 block"
        );
    }
}

namespace psnr_hvs_layout {
namespace {

std::uint32_t AddRegion(std::uint64_t& cursor, std::uint64_t words) {
    constexpr auto maximum = std::numeric_limits<std::uint32_t>::max();
    if (words > maximum - 15u)
        throw std::invalid_argument("psnr_hvs: region exceeds arena addressing");
    const std::uint64_t aligned = (words + 15u) / 16u * 16u;
    if (aligned > maximum - cursor)
        throw std::invalid_argument("psnr_hvs: layout exceeds arena addressing");
    const auto result = static_cast<std::uint32_t>(cursor);
    cursor += aligned;
    return result;
}

void CheckDispatch(const PsnrHvsConfig& config, const VulkanContext& context) {
    const auto& limits = context.limits;
    if (limits.maxComputeWorkGroupInvocations < kReduceLanes ||
        limits.maxComputeWorkGroupSize[0] < kReduceLanes ||
        limits.maxComputeWorkGroupSize[1] < 8 ||
        limits.maxComputeSharedMemorySize < 4096 ||
        limits.maxPushConstantsSize < sizeof(Push) ||
        Groups(config.size.width, 16) > limits.maxComputeWorkGroupCount[0] ||
        Groups(config.size.height, 8) > limits.maxComputeWorkGroupCount[1] ||
        (config.size.width - 8) / 7 + 1 > limits.maxComputeWorkGroupCount[0] ||
        (config.size.height - 8) / 7 + 1 > limits.maxComputeWorkGroupCount[1] ||
        limits.maxComputeWorkGroupCount[2] < 1) {
        throw std::invalid_argument(
            "psnr_hvs: compute dispatch exceeds device limits"
        );
    }
}

}  // namespace

Layout BuildLayout(const PsnrHvsConfig& config, const VulkanContext& context) {
    ValidatePsnrHvsConfig(config);
    CheckDispatch(config, context);
    Layout layout;
    const std::uint64_t pixels = config.size.width * config.size.height;
    const std::uint64_t blocks =
        ((config.size.width - 8) / 7 + 1) * ((config.size.height - 8) / 7 + 1);
    const auto reduction_groups =
        Groups(static_cast<std::size_t>(blocks), kReduceItemsPerGroup);
    if (reduction_groups > context.limits.maxComputeWorkGroupCount[0])
        throw std::invalid_argument(
            "psnr_hvs: reduction dispatch exceeds device limits"
        );
    std::uint64_t cursor = 0;
    layout.tables = AddRegion(cursor, kTableWords);
    layout.plane_stride = static_cast<std::uint32_t>(pixels);
    for (auto& offset : layout.prepared) offset = AddRegion(cursor, 3u * pixels);
    for (auto& offset : layout.partials) offset = AddRegion(cursor, blocks);
    for (auto& offset : layout.reduction)
        offset = AddRegion(cursor, reduction_groups);
    layout.result = AddRegion(cursor, 3);
    const std::uint64_t input_words =
        config.input_bytes / 4u + (config.input_bytes % 4u != 0);
    layout.input_pool = AddRegion(cursor, input_words);
    layout.input_pool_size = static_cast<std::uint32_t>(input_words);
    for (std::size_t image = 0; image < 2; ++image) {
        layout.stream_region[image] =
            config.streaming_inputs
                ? AddRegion(cursor, kSourceFloatsPerPixel * pixels)
                : 0;
        layout.stream_desc[image] = AddRegion(cursor, kSourceDescWords);
    }
    layout.resident.reserve(config.gpu_resident_images);
    for (std::size_t slot = 0; slot < config.gpu_resident_images; ++slot)
        layout.resident.push_back(
            AddRegion(cursor, kSourceFloatsPerPixel * pixels)
        );
    if (cursor > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::invalid_argument("psnr_hvs: arena exceeds host addressing");
    layout.bytes = Bytes(static_cast<std::size_t>(cursor));
    if (layout.bytes > context.limits.maxStorageBufferRange)
        throw std::invalid_argument("psnr_hvs: arena exceeds storage range");
    layout.staging = PlanStagingSources(
        static_cast<std::size_t>(pixels), config.streaming_inputs,
        config.gpu_resident_images
    );
    layout.staging_tables = layout.staging.reserved_floats;
    layout.staging_desc = layout.staging_tables + kTableWords;
    layout.staging_result = layout.staging_desc + 2u * kSourceDescWords;
    layout.staging_words = layout.staging_result + 3u;
    if (layout.staging_words >
        std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::invalid_argument("psnr_hvs: staging exceeds host addressing");
    layout.persistent_ranges = {
        {Bytes(layout.tables), Bytes(kTableWords), true}};
    return layout;
}

}  // namespace psnr_hvs_layout

BufferMemoryPlan PlanPsnrHvsMemory(
    const PsnrHvsConfig& config, const VulkanContext& context
) {
    const auto layout = psnr_hvs_layout::BuildLayout(config, context);
    return {
        layout.bytes, Bytes(layout.staging_words), {}, layout.persistent_ranges};
}

}  // namespace vqm::internal
