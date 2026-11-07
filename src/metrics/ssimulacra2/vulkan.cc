// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// The SSIMULACRA 2 metric on a Vulkan compute queue. One Compute call records
// the entire scale pyramid -- upload, one prepare dispatch per uncached image
// that builds every level (decode, transfer and primaries to linear, XYB, and
// the 2x2 averages chained inside one workgroup), one scale dispatch per level
// and channel, and one reduction over every level -- into one command buffer and
// submits it once, so the only host traffic after submission is the 18 norm
// sums per scale (nine dispatches at six scales, or eight when the resident
// reference is cached).
//
// All shaders touch device-local memory only: the decoded planes reach the
// arena before the first dispatch, every plane stays there for the whole
// pyramid, and the norm block is read back afterwards. Two upload paths exist:
// when the device exposes device-local memory the host may write (resizable
// BAR) the planes are copied straight into the arena, otherwise they go
// through a mapped staging buffer and vkCmdCopyBuffer. Images passed to
// Upload() stay in a resident slot and skip the upload path entirely, which is
// what resident submissions use.
//
// One dispatch per level and channel
// (src/metrics/ssimulacra2/shaders/scale.comp) does everything the traffic
// allows to fuse: it loads the two source tiles into shared memory, filters all
// five blur quantities out of them, evaluates the two error maps and reduces
// their six norm partials. Neither the five blurred quantities nor the two error
// maps ever reach device memory, which is what the earlier blur/ssim/edge_diff
// dispatches spent about 3.4 GB per pair on. The level's plane offsets, size and
// norm block travel as push constants rather than as level-table loads: they are
// what every tile address hangs off, and at 4K two resident workgroups per SM
// cannot hide that latency.
//
// Precision: XYB's cube root uses pow(v, 1/3) and the norm sums accumulate in
// float with a hierarchical reduction, so scores come out close to, not
// bit-identical with, an evaluation in double. The host converts the sums to
// double for the 1-norm mean and the 4-norm fourth root.
//
// Everything that is not this metric lives in src/vulkan/vulkan_runtime.{h,cc}:
// the arena and staging buffers, the descriptor scheme, the command buffer and
// its one-in-flight token, dispatch/barrier wrappers, the upload paths and the
// stage timers. layout/ssimulacra2_layout holds memory planning. This file keeps
// the level table and channel
// masks, the push-constant block, the three pipelines and the recording order.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "handles.h"
#include "metrics/ssimulacra2/layout.h"
#include "metrics/ssimulacra2/ssimulacra2_internal.h"
#include "metrics/ssimulacra2/vulkan.h"
#include "vulkan/vulkan_runtime.h"
#include "vulkan/vulkan_support.h"

namespace vqm::internal {

using ssimulacra2_layout::ArenaRegions;
using ssimulacra2_layout::BuildSsimulacra2Layout;
using ssimulacra2_layout::kChannels;
using ssimulacra2_layout::kLevelFields;
using ssimulacra2_layout::kLevelWords;
using ssimulacra2_layout::kScaleSums;
using ssimulacra2_layout::kSsimBlockOffset;
using ssimulacra2_layout::kWorkgroup;
using ssimulacra2_layout::ScaleVariant;

namespace {

// Names are the stable embedded SPIR-V identifiers, independent of source paths.
std::string Ssimulacra2PrepareShaderName(
    SourceDecodeVariant variant, bool half
) {
    std::string name = "prepare_c" + std::to_string(variant.chroma) + "_t" +
                       std::to_string(variant.transfer);
    if (variant.chroma != 0) name += "_f" + std::to_string(variant.filter);
    if (half) name += "_half";
    return name;
}

// std430 mirror of the push-constant block in
// src/metrics/ssimulacra2/shaders/ssimu2_common.glsl.
struct PushParams {
    // Word offset of the level table in binding 2. Everything that varies by
    // level -- plane offsets, sizes, norm blocks, the scale grid -- is there.
    std::uint32_t level_base;
    // Levels this geometry runs; the pyramid's shape is a property of the
    // image size, so every dispatch of a pipeline carries it.
    std::uint32_t scale_count;
    // prepare: word offset of this image's descriptor, and which side of the
    // pair this dispatch converts.
    std::uint32_t desc_base;
    std::uint32_t image;
    float background;  // prepare: alpha blend target
    // scale: one level's planes, geometry and norm block. Every address the
    // tile loop computes hangs off these, so they travel as push constants
    // rather than as level-table loads: the dynamic loads measured 0.5 ms of
    // the 4K scale stage, whose two resident workgroups leave no latency to
    // hide them behind.
    std::uint32_t plane[2];  // image 0, image 1 plane offsets
    std::uint32_t stride;    // plane channel stride: the level's area
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t partials;  // this level's norm partials
    std::uint32_t groups;    // scale workgroups of this level
    std::uint32_t groups_x;
    // Channels whose maps this backend computes: scale reads bit c of its
    // level, prepare and reduce read bit level * 3 + channel of the pyramid.
    // A cleared bit is a (channel, level) pair whose six weights are all at
    // most kWeightPruning, so their omission is a measured approximation. The
    // rule is Vship's (see BuildChannelMasks and
    // src/metrics/ssimulacra2/ssimulacra2_internal.h).
    std::uint32_t channel_mask;
    float weights[16];  // Gaussian taps, 13 are used
};
static_assert(sizeof(PushParams) == 120);

}  // namespace

class VulkanSsimulacra2Pipeline final : public Ssimulacra2GpuPipeline {
public:
    VulkanSsimulacra2Pipeline(
        std::shared_ptr<VulkanContext> context, std::vector<Size> sizes,
        const GaussianKernel& kernel, std::size_t resident_images,
        bool half_planes, ChromaFilter chroma_filter, bool streaming_inputs,
        std::size_t input_bytes, MemoryPolicy memory_policy,
        std::shared_ptr<SharedMetricStorage> shared = {}
    );
    ~VulkanSsimulacra2Pipeline() override;

    VulkanSsimulacra2Pipeline(const VulkanSsimulacra2Pipeline&) = delete;
    VulkanSsimulacra2Pipeline& operator=(const VulkanSsimulacra2Pipeline&) =
        delete;

    void RequireIdle() const override { runtime_->RequireIdle(); }
    [[nodiscard]] PipelineMemoryUsage GetMemoryUsage() const override {
        return runtime_->GetMemoryUsage();
    }
    [[nodiscard]] std::uint64_t Submit(
        const Image& original, const Image& distorted, float background
    ) override;
    [[nodiscard]] std::uint64_t Submit(
        const ImageView& original, const ImageView& distorted, float background
    ) override;
    [[nodiscard]] std::uint64_t Submit(
        std::uint32_t original, std::uint32_t distorted, float background
    ) override;
    [[nodiscard]] std::uint64_t Submit(
        std::uint32_t original, const Image& distorted, float background
    ) override;
    [[nodiscard]] Ssimulacra2Result Finish(std::uint64_t token) override;
    [[nodiscard]] ImageBuffer AllocateImage(const ImageFormat& format) override;
    [[nodiscard]] std::byte* MapImage(ImageBuffer buffer) override;
    void Release(ImageBuffer buffer) override;
    [[nodiscard]] std::uint64_t Submit(
        const ImageBuffer& original, const ImageBuffer& distorted,
        float background
    ) override;
    [[nodiscard]] std::uint64_t Submit(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        float background
    ) override;
    [[nodiscard]] std::uint64_t Submit(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        const VulkanSubmitSync& sync, float background
    ) override;
    std::optional<std::uint32_t> Upload(
        const Image& image, std::size_t capacity
    ) override;
    void Release(std::uint32_t slot) override;

private:
    void CreatePipelines();
    void BuildLevelTable();
    void BuildChannelMasks();
    void PublishLevels();
    void ScaleStage();
    void Record(
        const std::array<VulkanRuntime::Source, 2>& sources, float background,
        bool reuse_reference
    );
    [[nodiscard]] std::uint64_t Submit(
        const std::array<VulkanRuntime::Source, 2>& sources, float background,
        const VulkanSubmitSync* sync = nullptr
    );
    [[nodiscard]] Ssimulacra2Result ReadResult(const float* sums) const;
    void Destroy() noexcept;

    std::shared_ptr<VulkanContext> ctx_;
    std::unique_ptr<VulkanRuntime> runtime_;
    std::vector<Size> sizes_;
    // Tile geometry of the scale dispatch, chosen from the device's limits.
    ScaleVariant scale_tile_{};
    // 16-bit plane storage (Ssimulacra2Config::half_planes): picks the *_half
    // shader blobs and halves the plane region.
    bool half_ = false;
    std::array<float, 16> weights_{};
    // Ssimulacra2Config::streaming_inputs: without it no source region is
    // reserved and the pixel overloads refuse, which saves 32 B per pixel of
    // arena and the same again of staging.
    bool streaming_ = true;
    std::size_t base_pixels_ = 0;
    ArenaRegions regions_;
    // Levels this geometry runs (sizes_.size()) and, per level, the scale
    // workgroup count; the level table words the shaders read.
    std::size_t levels_ = 0;
    std::array<std::uint32_t, kSsimulacra2MaxScales> level_groups_{};
    // Each level's norm partials, relative to regions_.partials.
    std::array<std::uint32_t, kSsimulacra2MaxScales> level_partials_{};
    std::array<std::uint32_t, kLevelWords> level_words_{};
    // Per level, the channels whose maps are computed (bits 0-2), and the same
    // 18 bits for the whole pyramid.
    std::array<std::uint32_t, kSsimulacra2MaxScales> level_channel_mask_{};
    std::uint32_t channel_mask_bits_ = 0;
    ChromaFilter chroma_filter_ = ChromaFilter::Bilinear;

    // Layout of the readback and the one-time level-table copy inside the
    // staging buffer. Only the non-BAR device uses them.
    std::size_t staging_norm_offset_ = 0;   // floats
    std::size_t staging_desc_offset_ = 0;   // floats
    std::size_t staging_level_offset_ = 0;  // floats

    // One compute pipeline per prepare source class
    // (src/vulkan/vulkan_support.h).
    std::array<VkPipeline, kSourceDecodeVariants.size()> prepare_{};
    VkPipeline scale_ = VK_NULL_HANDLE;
    VkPipeline reduce_ = VK_NULL_HANDLE;
    // Only resident inputs are immutable under the public API. A cached slot
    // is invalidated when it is released or reused for another image.
    bool reference_valid_ = false;
    std::uint32_t reference_slot_ = 0;
    float reference_background_ = 0.0f;
};

VulkanSsimulacra2Pipeline::VulkanSsimulacra2Pipeline(
    std::shared_ptr<VulkanContext> context, std::vector<Size> sizes,
    const GaussianKernel& kernel, std::size_t resident_images, bool half_planes,
    ChromaFilter chroma_filter, bool streaming_inputs, std::size_t input_bytes,
    MemoryPolicy memory_policy, std::shared_ptr<SharedMetricStorage> shared
)
    : ctx_(std::move(context)),
      sizes_(std::move(sizes)),
      half_(half_planes),
      streaming_(streaming_inputs),
      chroma_filter_(chroma_filter) {
    try {
        if (half_ && !ctx_->sixteen_bit_storage) {
            throw std::runtime_error(
                "vulkan: device does not support 16-bit storage; "
                "Ssimulacra2Config::half_planes needs it"
            );
        }
        // Exactly the 13 taps scale.comp's vec4 window chains consume
        // (kTapCount there); a longer kernel has taps this backend cannot
        // filter, a shorter one changes the window shapes.
        if (kernel.TapCount() != 13) {
            throw std::invalid_argument(
                "vulkan backend supports the 13-tap Gaussian only"
            );
        }
        std::copy(
            kernel.weights.begin(), kernel.weights.end(), weights_.begin()
        );

        const auto plan = BuildSsimulacra2Layout(
            sizes_, *ctx_, resident_images, half_, streaming_, input_bytes,
            chroma_filter_
        );
        regions_ = plan.regions;
        scale_tile_ = plan.scale_tile;
        base_pixels_ = plan.base_pixels;
        levels_ = plan.levels;
        level_groups_ = plan.level_groups;
        level_partials_ = plan.level_partials;
        staging_norm_offset_ = plan.staging_norm_offset;
        staging_desc_offset_ = plan.staging_desc_offset;
        staging_level_offset_ = plan.staging_level_offset;

        VulkanRuntime::Config runtime_config;
        runtime_config.stream_region = regions_.source;
        runtime_config.stream_desc = regions_.source_desc;
        runtime_config.resident_slots = regions_.resident;
        runtime_config.input_pool = plan.input_pool;
        runtime_config.input_pool_size = plan.input_pool_size;
        runtime_config.staging_desc_offset = staging_desc_offset_;
        runtime_config.staging_image_bytes = plan.staging.image_bytes;
        runtime_config.base_pixels = base_pixels_;
        runtime_config.base_size = sizes_[0];
        runtime_config.streaming = streaming_;
        runtime_config.memory_policy = memory_policy;
        runtime_config.chroma_filter = chroma_filter_;
        runtime_config.push_constant_size = sizeof(PushParams);
        runtime_config.persistent_ranges = plan.persistent_ranges;
        runtime_ = std::make_unique<VulkanRuntime>(
            ctx_, runtime_config, std::move(shared)
        );

        runtime_->CreateArena(regions_.bytes);
        if (!runtime_->arena_visible()) {
            runtime_->CreateStaging(Bytes(staging_level_offset_ + kLevelWords));
        }

        BuildLevelTable();
        BuildChannelMasks();

        runtime_->CreateCommandObjects();
        runtime_->CreateDescriptors(
            regions_.plane_base_bytes, regions_.plane_bytes
        );
        CreatePipelines();
        PublishLevels();
        runtime_->InitializeSharedState();
    } catch (...) {
        Destroy();
        throw;
    }
}

VulkanSsimulacra2Pipeline::~VulkanSsimulacra2Pipeline() { Destroy(); }

void VulkanSsimulacra2Pipeline::Destroy() noexcept {
    if (runtime_ && runtime_->pending()) {
        try {
            runtime_->WaitFence();
        } catch (...) {
            // A lost device cannot complete the work; destruction must not
            // throw.
        }
    }
    if (runtime_) {
        runtime_->DestroyPipelines(prepare_);
        runtime_->DestroyPipelines(std::array{scale_, reduce_});
    }
    prepare_.fill(VK_NULL_HANDLE);
    scale_ = reduce_ = VK_NULL_HANDLE;
    runtime_.reset();
}

void VulkanSsimulacra2Pipeline::CreatePipelines() {
    // The plane shaders exist in an f32 and a 16-bit build; reduce reads f32
    // norm sums either way.
    const std::string suffix = half_ ? "_half" : "";
    for (std::size_t i = 0; i < kSourceDecodeVariants.size(); ++i) {
        prepare_[i] = runtime_->CreatePipeline(
            Ssimulacra2PrepareShaderName(kSourceDecodeVariants[i], half_)
        );
    }
    scale_ = runtime_->CreatePipeline(
        "scale_" + std::to_string(scale_tile_.rows) + "x" +
        std::to_string(scale_tile_.cols) + "x" +
        std::to_string(scale_tile_.lanes) + suffix
    );
    reduce_ = runtime_->CreatePipeline("reduce");
}

void VulkanSsimulacra2Pipeline::BuildLevelTable() {
    level_words_.fill(0u);
    for (std::size_t level = 0; level < levels_; ++level) {
        std::uint32_t* row = level_words_.data() + level * kLevelFields;
        row[0] = regions_.img[0][level];
        row[1] = regions_.img[1][level];
        row[2] = regions_.partials + level_partials_[level];
        row[3] = regions_.norms + static_cast<std::uint32_t>(level * kScaleSums);
        row[4] = static_cast<std::uint32_t>(sizes_[level].width);
        row[5] = static_cast<std::uint32_t>(sizes_[level].height);
        row[6] = level_groups_[level];
    }
}

void VulkanSsimulacra2Pipeline::BuildChannelMasks() {
    // A (channel, level) pair whose six weights are all at most
    // kWeightPruning contributes nothing the score can see, so the backend
    // skips its maps: prepare does not store that plane, scale returns before
    // its tile, and reduce writes the norm straight to zero. On identical
    // inputs every map value is exactly zero anyway, so a pruned pair removes
    // only zeros and the score stays exactly 100.
    //
    // Provenance: dropping maps whose weights cannot reach the score is
    // Vship's idea. It decides each map on its own from that map's 1-norm and
    // 4-norm weights (.reference/Vship/src/Vulkan/ssimu2/score.hpp:125-131),
    // leaves a whole (plane, scale) dispatch out only when all three of its
    // maps go (:181), and suppresses the downsample store on the same
    // conjunction (main.hpp:32-42). Our workgroup computes all six norms of a
    // pair at once, so the pair is the unit we can drop. The mask itself is
    // ours: the peer indexes a fixed six-scale stride because its pyramid is
    // always six levels, while our weight table is a flat run of scales.size()
    // rows per channel (Ssimulacra2Result::Score), so this rebuilds the mask per
    // geometry from Ssimulacra2ChannelPruned.
    level_channel_mask_.fill(0u);
    channel_mask_bits_ = 0u;
    for (std::size_t level = 0; level < levels_; ++level) {
        for (std::size_t channel = 0; channel < kChannels; ++channel) {
            if (Ssimulacra2ChannelPruned(channel, level, levels_)) continue;
            const auto level_word = static_cast<std::uint32_t>(level);
            const auto channel_bit = static_cast<std::uint32_t>(channel);
            level_channel_mask_[level] |= 1u << channel_bit;
            channel_mask_bits_ |= 1u << (3u * level_word + channel_bit);
        }
    }
}

void VulkanSsimulacra2Pipeline::PublishLevels() {
    // The table is a property of the geometry, so it travels to the device
    // once, before the first compute submission. Everything after this reads
    // it in place: only the descriptors and the pixels are rewritten per pair.
    const auto* words =
        reinterpret_cast<const std::uint8_t*>(level_words_.data());
    const std::size_t bytes = kLevelWords * sizeof(std::uint32_t);
    if (runtime_->arena_visible()) {
        std::memcpy(
            reinterpret_cast<std::uint8_t*>(runtime_->arena_map()) +
                Bytes(regions_.levels),
            words, bytes
        );
        return;
    }
    std::memcpy(
        reinterpret_cast<std::uint8_t*>(runtime_->staging_map()) +
            Bytes(staging_level_offset_),
        words, bytes
    );
    runtime_->PublishStaged(regions_.levels, staging_level_offset_, kLevelWords);
}

void VulkanSsimulacra2Pipeline::ScaleStage() {
    // One dispatch per level, with that level's own grid: a level's grid is a
    // fraction of the base level's, so one grid covering all of them launches
    // four times the workgroups that have work (146 880 against 32 691 at 4K,
    // which measured 2.30 ms against 1.63 ms on the scale stage), and a flat
    // index decoded inside the shader did not pay either (2.05 ms). The
    // dispatches are independent -- each reads its own level's planes, all
    // written by the prepare pass -- so no barrier separates them.
    runtime_->OpenStage(Stage::Ssimulacra2Scale);
    for (std::size_t level = 0; level < levels_; ++level) {
        PushParams params{};
        params.level_base = regions_.levels;
        params.scale_count = static_cast<std::uint32_t>(levels_);
        params.plane[0] = regions_.img[0][level];
        params.plane[1] = regions_.img[1][level];
        params.stride = regions_.channel_floats[level];
        params.width = static_cast<std::uint32_t>(sizes_[level].width);
        params.height = static_cast<std::uint32_t>(sizes_[level].height);
        params.partials = regions_.partials +
                          static_cast<std::uint32_t>(level_partials_[level]);
        params.groups = level_groups_[level];
        params.groups_x = Groups(sizes_[level].width, scale_tile_.cols);
        params.channel_mask = level_channel_mask_[level];
        std::copy(weights_.begin(), weights_.end(), params.weights);
        runtime_->Dispatch(
            scale_, &params, sizeof(params), params.groups_x,
            Groups(sizes_[level].height, scale_tile_.rows), 3
        );
    }
    runtime_->ComputeBarrier();

    // One reduction covers every level: y is the level, x the norm, and each
    // level's own workgroup count bounds the sum.
    PushParams reduce_params{};
    reduce_params.level_base = regions_.levels;
    reduce_params.scale_count = static_cast<std::uint32_t>(levels_);
    reduce_params.channel_mask = channel_mask_bits_;
    runtime_->Dispatch(
        reduce_, &reduce_params, sizeof(reduce_params), kScaleSums,
        static_cast<std::uint32_t>(levels_), 1
    );
    runtime_->CloseStage();
    runtime_->ComputeBarrier();
}

void VulkanSsimulacra2Pipeline::Record(
    const std::array<VulkanRuntime::Source, 2>& sources, float background,
    bool reuse_reference
) {
    runtime_->BeginCommandBuffer();
    runtime_->BindDescriptorSet();

    // 1. Prepare: the upload DMA, sRGB -> linear, XYB and the 2x downsample of
    // every scale, one dispatch per image per scale. The stage timer opens
    // before the copies so its share includes the transfer.
    runtime_->OpenStage(Stage::Ssimulacra2Prepare);
    const auto formats = runtime_->PublishSources(sources);

    // 1b. One prepare dispatch per image builds that image's whole pyramid:
    // the base level's raw decode and colour conversion, the XYB plane of
    // every level, and the 2x2 averages that chain one level to the next
    // inside a single workgroup. The scale stage below then reads the levels.
    for (std::size_t image = 0; image < 2; ++image) {
        if (image == 0 && reuse_reference) continue;
        PushParams params{};
        params.level_base = regions_.levels;
        params.scale_count = static_cast<std::uint32_t>(levels_);
        // A float offset is a word offset: both are four bytes, and binding 2
        // is the same arena viewed as words.
        params.desc_base = regions_.source_desc[image];
        params.image = static_cast<std::uint32_t>(image);
        params.channel_mask = channel_mask_bits_;
        params.background = background;
        // A workgroup owns 32 base pixels per side, so the grid is the 2x2
        // block grid -- the next level's size -- rounded up to it.
        // Both internal sources use the arena, already bound at the start.
        // Only a caller buffer needs the second source descriptor set.
        if (image == 1 &&
            (sources[0].kind == VulkanRuntime::Source::Kind::External ||
             sources[1].kind == VulkanRuntime::Source::Kind::External)) {
            runtime_->BindSource(image);
        }
        runtime_->Dispatch(
            prepare_[SourceDecodeVariantIndex(
                SelectSourceDecodeVariant(*formats[image], chroma_filter_)
            )],
            &params, sizeof(params),
            Groups((sizes_[0].width + 1) / 2, kWorkgroup),
            Groups((sizes_[0].height + 1) / 2, kWorkgroup), 1
        );
    }
    runtime_->CloseStage();
    runtime_->ComputeBarrier();

    ScaleStage();

    // 2. Publish the norm block to the host: on the BAR path the host reads
    // the arena directly, otherwise the block is copied out past the upload
    // slots.
    if (runtime_->arena_visible()) {
        runtime_->ComputeToHostBarrier();
    } else {
        runtime_->RecordReadback(
            regions_.norms, staging_norm_offset_, sizes_.size() * kScaleSums
        );
    }

    runtime_->EndCommandBuffer();
}

Ssimulacra2Result VulkanSsimulacra2Pipeline::ReadResult(
    const float* sums
) const {
    Ssimulacra2Result msssim;
    for (std::size_t s = 0; s < sizes_.size(); ++s) {
        const float* scale_sums = sums + s * kScaleSums;
        const double one_per_pixel =
            1.0 / (static_cast<double>(sizes_[s].width) *
                   static_cast<double>(sizes_[s].height));
        Ssimulacra2Scale scale;
        for (std::size_t c = 0; c < 3; ++c) {
            const std::size_t ssim = c * 2;
            scale.avg_ssim[ssim + 0] =
                one_per_pixel * static_cast<double>(scale_sums[ssim + 0]);
            scale.avg_ssim[ssim + 1] = std::sqrt(
                std::sqrt(
                    one_per_pixel * static_cast<double>(scale_sums[ssim + 1])
                )
            );
            const std::size_t edge = kSsimBlockOffset + c * 4;
            scale.avg_edgediff[c * 4 + 0] =
                one_per_pixel * static_cast<double>(scale_sums[edge + 0]);
            scale.avg_edgediff[c * 4 + 1] = std::sqrt(
                std::sqrt(
                    one_per_pixel * static_cast<double>(scale_sums[edge + 1])
                )
            );
            scale.avg_edgediff[c * 4 + 2] =
                one_per_pixel * static_cast<double>(scale_sums[edge + 2]);
            scale.avg_edgediff[c * 4 + 3] = std::sqrt(
                std::sqrt(
                    one_per_pixel * static_cast<double>(scale_sums[edge + 3])
                )
            );
        }
        msssim.scales.push_back(scale);
    }
    return msssim;
}

std::uint64_t VulkanSsimulacra2Pipeline::Submit(
    const Image& original, const Image& distorted, float background
) {
    runtime_->RequireIdle();
    if (!streaming_) {
        throw std::invalid_argument(
            "vulkan: Ssimulacra2Config::streaming_inputs is false, so the pixel "
            "overloads are unavailable; use Upload() and Compute() with "
            "resident handles"
        );
    }
    // Fills the staging buffer and the descriptors, then records; the copies
    // are inside the pair's own command buffer, so this is still one
    // submission and the next pipeline can start recording immediately.
    runtime_->UploadStreaming(original, distorted);
    return Submit(std::array<VulkanRuntime::Source, 2>{}, background);
}

std::uint64_t VulkanSsimulacra2Pipeline::Submit(
    const ImageView& original, const ImageView& distorted, float background
) {
    runtime_->RequireIdle();
    if (!streaming_) {
        throw std::invalid_argument("vulkan: streaming_inputs is false");
    }
    runtime_->UploadStreaming(original, distorted);
    return Submit(std::array<VulkanRuntime::Source, 2>{}, background);
}

std::uint64_t VulkanSsimulacra2Pipeline::Submit(
    std::uint32_t original, std::uint32_t distorted, float background
) {
    return Submit(
        std::array<VulkanRuntime::Source, 2>{
            VulkanRuntime::Source{VulkanRuntime::Source::Kind::Resident,
                                  original},
            VulkanRuntime::Source{VulkanRuntime::Source::Kind::Resident,
                                  distorted}},
        background
    );
}

std::uint64_t VulkanSsimulacra2Pipeline::Submit(
    std::uint32_t original, const Image& distorted, float background
) {
    runtime_->RequireIdle();
    if (!streaming_) {
        throw std::invalid_argument("vulkan: streaming_inputs is false");
    }
    runtime_->UploadStreamingDistorted(distorted);
    return Submit(
        std::array<VulkanRuntime::Source, 2>{
            VulkanRuntime::Source{VulkanRuntime::Source::Kind::Resident,
                                  original},
            VulkanRuntime::Source{VulkanRuntime::Source::Kind::Stream, 0}},
        background
    );
}

Ssimulacra2Result VulkanSsimulacra2Pipeline::Finish(std::uint64_t token) {
    runtime_->FinishSubmission(token);
    const float* sums = runtime_->arena_visible()
                            ? runtime_->arena_map() + regions_.norms
                            : runtime_->staging_map() + staging_norm_offset_;
    return ReadResult(sums);
}

std::optional<std::uint32_t> VulkanSsimulacra2Pipeline::Upload(
    const Image& image, std::size_t capacity
) {
    const auto uploaded = runtime_->UploadResident(image, capacity);
    if (uploaded && *uploaded == reference_slot_) reference_valid_ = false;
    return uploaded;
}

void VulkanSsimulacra2Pipeline::Release(std::uint32_t slot) {
    if (slot == reference_slot_) reference_valid_ = false;
    runtime_->ReleaseResident(slot);
}

ImageBuffer VulkanSsimulacra2Pipeline::AllocateImage(const ImageFormat& format) {
    return runtime_->AllocateImage(format);
}

std::byte* VulkanSsimulacra2Pipeline::MapImage(ImageBuffer buffer) {
    return runtime_->MapImage(buffer);
}

void VulkanSsimulacra2Pipeline::Release(ImageBuffer buffer) {
    runtime_->ReleaseInput(buffer);
}

std::uint64_t VulkanSsimulacra2Pipeline::Submit(
    const ImageBuffer& original, const ImageBuffer& distorted, float background
) {
    return Submit(runtime_->ResolveInputs(original, distorted), background);
}

std::uint64_t VulkanSsimulacra2Pipeline::Submit(
    const VulkanBufferView& original, const VulkanBufferView& distorted,
    float background
) {
    runtime_->RequireIdle();
    if (ctx_->owns_device) {
        throw std::invalid_argument(
            "vulkan: external buffers require a caller-owned Vulkan device"
        );
    }
    return Submit(
        runtime_->ResolveExternalInputs(original, distorted), background
    );
}

std::uint64_t VulkanSsimulacra2Pipeline::Submit(
    const VulkanBufferView& original, const VulkanBufferView& distorted,
    const VulkanSubmitSync& sync, float background
) {
    runtime_->RequireIdle();
    if (ctx_->owns_device) {
        throw std::invalid_argument(
            "vulkan: external buffers require a caller-owned Vulkan device"
        );
    }
    VulkanRuntime::ValidateSubmitSync(sync);
    return Submit(
        runtime_->ResolveExternalInputs(original, distorted), background, &sync
    );
}

std::uint64_t VulkanSsimulacra2Pipeline::Submit(
    const std::array<VulkanRuntime::Source, 2>& sources, float background,
    const VulkanSubmitSync* sync
) {
    runtime_->RequireIdle();
    runtime_->PrepareSourceBindings(sources);
    const std::uint64_t token = runtime_->BeginSubmission();
    const bool resident =
        sources[0].kind == VulkanRuntime::Source::Kind::Resident;
    const bool reuse = resident && reference_valid_ &&
                       sources[0].index == reference_slot_ &&
                       background == reference_background_;
    reference_valid_ = false;
    Record(sources, background, reuse);
    if (sync != nullptr)
        runtime_->QueueSubmit(*sync);
    else
        runtime_->QueueSubmit();
    reference_valid_ = resident;
    reference_slot_ = sources[0].index;
    reference_background_ = background;
    return token;
}

std::unique_ptr<Ssimulacra2GpuPipeline> CreateSharedVulkanSsimulacra2Pipeline(
    const Ssimulacra2ScaleLayout& scales, const GaussianKernel& kernel,
    const Ssimulacra2Config& config, std::shared_ptr<VulkanContext> context,
    std::shared_ptr<SharedMetricStorage> storage
) {
    std::vector<Size> sizes(scales.sizes, scales.sizes + scales.count);
    return std::make_unique<VulkanSsimulacra2Pipeline>(
        std::move(context), std::move(sizes), kernel, 0, config.half_planes,
        config.chroma_filter, config.streaming_inputs, 0, config.memory_policy,
        std::move(storage)
    );
}

std::unique_ptr<Ssimulacra2GpuPipeline> CreateVulkanSsimulacra2Pipeline(
    const Ssimulacra2ScaleLayout& scales, const GaussianKernel& kernel,
    std::size_t resident_images, bool half_planes, ChromaFilter chroma_filter,
    bool streaming_inputs, std::size_t input_bytes, MemoryPolicy memory_policy,
    const VulkanDevice* borrowed_device
) {
    std::shared_ptr<VulkanContext> context =
        AcquireVulkanContext(borrowed_device);
    if (!context) return nullptr;

    std::vector<Size> sizes(scales.sizes, scales.sizes + scales.count);
    if (sizes.empty()) {
        throw std::invalid_argument("vulkan backend needs at least one scale");
    }
    for (const Size& size : sizes) {
        if (size.width == 0 || size.height == 0) {
            throw std::invalid_argument("vulkan backend needs non-zero sizes");
        }
    }
    return std::make_unique<VulkanSsimulacra2Pipeline>(
        std::move(context), std::move(sizes), kernel, resident_images,
        half_planes, chroma_filter, streaming_inputs, input_bytes, memory_policy
    );
}

}  // namespace vqm::internal
