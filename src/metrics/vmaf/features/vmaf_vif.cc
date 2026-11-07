// VMAF VIF: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Default four-scale filter/decimation order follows libvmaf vif.c:150-205
// at revision 4611e47a7330cd342498dcc130e1008efbeab6e4.

#include "vmaf_vif.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "vulkan/vulkan_runtime.h"

namespace vqm::internal {
namespace {

struct VifPush {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t reference;
    std::uint32_t distorted;
    std::uint32_t scratch;
    std::uint32_t reference_out;
    std::uint32_t distorted_out;
    std::uint32_t partials;
    std::uint32_t results;
    std::uint32_t scale;
    std::uint32_t op;
    std::uint32_t groups;
    float gain_limit;
};
static_assert(sizeof(VifPush) == VmafVif::PushBytes());

void CheckGeometry(Size size) {
    constexpr std::size_t kMaxDimension =
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() / 2);
    // Four levels, radius-eight mirror at scale 0, and five scratch planes.
    if (size.width < 16 || size.height < 16 || size.width > kMaxDimension ||
        size.height > kMaxDimension ||
        size.width >
            std::numeric_limits<std::uint32_t>::max() / 5u / size.height) {
        throw std::invalid_argument("vmaf vif: invalid image geometry");
    }
}

void CheckRange(std::uint32_t offset, std::size_t count) {
    if (count > std::numeric_limits<std::uint32_t>::max() - offset) {
        throw std::invalid_argument("vmaf vif: arena offset overflow");
    }
}

}  // namespace

VmafVif::VmafVif(
    VulkanRuntime& runtime, bool allow_fused, std::uint32_t shared_cap,
    float enhancement_gain_limit
)
    : runtime_(&runtime),
      fuse_scale0_(
          allow_fused &&
          SupportsFusedScale0(
              std::min(
                  runtime.context().limits.maxComputeSharedMemorySize, shared_cap
              )
          )
      ),
      fuse_down_(
          allow_fused &&
          SupportsFusedDown(
              std::min(
                  runtime.context().limits.maxComputeSharedMemorySize, shared_cap
              )
          )
      ),
      fuse_stats_(
          allow_fused &&
          SupportsFusedStats(
              std::min(
                  runtime.context().limits.maxComputeSharedMemorySize, shared_cap
              )
          )
      ),
      enhancement_gain_limit_(enhancement_gain_limit) {
    if (!std::isfinite(enhancement_gain_limit) || enhancement_gain_limit <= 0.0f)
        throw std::invalid_argument("vmaf vif: invalid enhancement gain limit");
    try {
        for (std::size_t scale = 0; scale < scale_pipelines_.size(); ++scale) {
            auto& pipelines = scale_pipelines_[scale];
            const auto suffix = std::to_string(scale);
            pipelines.vertical =
                runtime_->CreatePipeline("vmaf_vif_vertical_" + suffix);
            pipelines.horizontal =
                runtime_->CreatePipeline("vmaf_vif_horizontal_" + suffix);
            if (scale == 0) {
                if (fuse_scale0_)
                    pipelines.fused_stats =
                        runtime_->CreatePipeline("vmaf_vif_fused0");
            } else {
                if (fuse_down_)
                    pipelines.fused_down = runtime_->CreatePipeline(
                        "vmaf_vif_down_fused_" + suffix
                    );
                if (fuse_stats_)
                    pipelines.fused_stats =
                        runtime_->CreatePipeline("vmaf_vif_fused_" + suffix);
            }
        }
        reduce_ = runtime_->CreatePipeline("vmaf_vif_reduce");
        // Shared f16vec2 storage declares SPIR-V Float16, which requires the
        // enabled shaderFloat16 feature, independently of 16-bit buffer storage.
        if (fuse_scale0_ && runtime_->context().shader_float16)
            half_scale0_ = runtime_->CreatePipeline("vmaf_vif_fused0_half");
    } catch (...) {
        DestroyPipelines();
        throw;
    }
}

VmafVif::~VmafVif() { DestroyPipelines(); }

void VmafVif::DestroyPipelines() noexcept {
    for (const auto& pipelines : scale_pipelines_) {
        runtime_->DestroyPipelines(
            std::array{pipelines.vertical, pipelines.horizontal,
                       pipelines.fused_down, pipelines.fused_stats}
        );
    }
    runtime_->DestroyPipelines(std::array{reduce_, half_scale0_});
}

std::size_t VmafVif::PartialCount(Size size) {
    CheckGeometry(size);
    const std::size_t count =
        ((size.width + 15) / 16) * ((size.height + 15) / 16);
    if (count > (std::numeric_limits<std::uint32_t>::max() - 255u) / 2u) {
        throw std::invalid_argument("vmaf vif: too many workgroups");
    }
    return count;
}

std::size_t VmafVif::ScratchWords(Size size, bool fused) {
    CheckGeometry(size);
    return (fused ? 2u : 5u) * size.width * size.height;
}

void VmafVif::Record(
    Size size, const VmafVifBuffers& buffers, bool half_source
) {
    const std::size_t max_groups = PartialCount(size);
    const auto& limits = runtime_->context().limits;
    if (Groups(size.width, 16) > limits.maxComputeWorkGroupCount[0] ||
        Groups(size.height, 16) > limits.maxComputeWorkGroupCount[1]) {
        throw std::invalid_argument("vmaf vif: geometry exceeds device limits");
    }
    CheckRange(buffers.scratch, ScratchWords(size, fuse_scale0_));
    CheckRange(buffers.partials, 2 * max_groups);
    CheckRange(buffers.results, 4);
    std::array<Size, 4> sizes{size};
    for (std::size_t scale = 0; scale < sizes.size(); ++scale) {
        if (scale > 0)
            sizes[scale] = {sizes[scale - 1].width / 2,
                            sizes[scale - 1].height / 2};
        const auto pixels = sizes[scale].width * sizes[scale].height;
        CheckRange(buffers.reference[scale], pixels);
        CheckRange(buffers.distorted[scale], pixels);
    }
    runtime_->ComputeBarrier();
    for (std::uint32_t scale = 0; scale < 4; ++scale) {
        const auto& pipelines = scale_pipelines_[scale];
        if (scale > 0) {
            const Size previous = sizes[scale - 1];
            const VifPush down{static_cast<std::uint32_t>(previous.width),
                               static_cast<std::uint32_t>(previous.height),
                               buffers.reference[scale - 1],
                               buffers.distorted[scale - 1],
                               buffers.scratch,
                               buffers.reference[scale],
                               buffers.distorted[scale],
                               buffers.partials,
                               buffers.results,
                               scale,
                               0,
                               0,
                               enhancement_gain_limit_};
            if (fuse_down_) {
                runtime_->Dispatch(
                    pipelines.fused_down, &down, sizeof(down),
                    Groups(sizes[scale].width, 16),
                    Groups(sizes[scale].height, 8), 1
                );
            } else {
                runtime_->Dispatch(
                    pipelines.vertical, &down, sizeof(down),
                    Groups(previous.width, 16), Groups(sizes[scale].height, 16),
                    1
                );
                runtime_->ComputeBarrier();
                runtime_->Dispatch(
                    pipelines.horizontal, &down, sizeof(down),
                    Groups(sizes[scale].width, 16),
                    Groups(sizes[scale].height, 16), 1
                );
            }
            runtime_->ComputeBarrier();
        }
        const Size current = sizes[scale];
        const bool fused_zero = scale == 0 && fuse_scale0_;
        const auto stats_tile =
            fused_zero ? 32u : (scale > 0 && fuse_stats_ ? 32u : 16u);
        const auto groups_x = Groups(current.width, stats_tile);
        const auto groups_y = Groups(current.height, 16);
        const VifPush stats{static_cast<std::uint32_t>(current.width),
                            static_cast<std::uint32_t>(current.height),
                            buffers.reference[scale],
                            buffers.distorted[scale],
                            buffers.scratch,
                            0,
                            0,
                            buffers.partials,
                            buffers.results,
                            scale,
                            1,
                            static_cast<std::uint32_t>(groups_x * groups_y),
                            enhancement_gain_limit_};
        if (fused_zero) {
            runtime_->Dispatch(
                half_source && SupportsHalfSource() ? half_scale0_
                                                    : pipelines.fused_stats,
                &stats, sizeof(stats), groups_x, groups_y, 1
            );
        } else if (fuse_stats_ && scale > 0) {
            runtime_->Dispatch(
                pipelines.fused_stats, &stats, sizeof(stats), groups_x, groups_y,
                1
            );
        } else {
            runtime_->Dispatch(
                pipelines.vertical, &stats, sizeof(stats), groups_x, groups_y, 1
            );
            runtime_->ComputeBarrier();
            runtime_->Dispatch(
                pipelines.horizontal, &stats, sizeof(stats), groups_x, groups_y,
                1
            );
        }
        runtime_->ComputeBarrier();
        runtime_->Dispatch(reduce_, &stats, sizeof(stats), 1, 1, 1);
        // The next downsample touches neither partials nor scale results. Its
        // final barrier orders this reduction's reads before the next statistic
        // pass reuses partials; no extra dependency is needed at this boundary.
    }
    runtime_->ComputeBarrier();
}

}  // namespace vqm::internal
