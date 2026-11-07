// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// Butteraugli on the Vulkan compute queue, built on
// src/vulkan/vulkan_runtime.{h,cc} so every upload path, submission and barrier
// discipline is shared with SSIMULACRA 2. The metric is libjxl's comparator
// path; the constant list and its source lines are in
// src/metrics/butteraugli/tables.cc and repeated in the shaders.
//
// One pair records, in this order (each barrier is a true dependency):
//   1. the reference chain once per handle: opsin dynamics and the LF / HF /
//      UHF separation for level 0 and, when it exists, the 2x subsampled
//      level 1 (butteraugli.cc:1792-1822 Make);
//   2. the distorted chain the same way, plus the subsampled rgb level 1
//      consumes (Diffmap :1829; its one-sub-level fold is :1852-1857);
//   3. the diff stages, level 1 first so level 0 can fold it: ac = 0, the six
//      Malta bands, the L2 terms, the mask and its blur, the erosion with the
//      mask-to-error term, and the combine (:1894-1952);
//   4. one reduction over the final diffmap: the max (the score,
//      :1954-1964) and the three powered sums behind distance_p3.
//
// The reference chain is cached: (kind, slot, generation) identifies the
// handle, so a released-and-reuploaded slot cannot serve a stale reference.
// Streaming Compute() re-uploads and rebuilds it every call.
//
// Memory: the arena holds, per level, both 12-plane chains (lf 3, mf 3, hf 2,
// hff 2, uhf 2). Scratch A and B reuse six distorted-chain planes. AC is
// shared between levels and holds the subsampled RGB inputs before the diff
// stages; the coarse diffmap reuses one distorted-chain plane.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "metrics/butteraugli/butteraugli_internal.h"
#include "metrics/butteraugli/layout.h"
#include "metrics/butteraugli/vulkan.h"
#include "vulkan/vulkan_runtime.h"
#include "vulkan/vulkan_support.h"

namespace vqm::internal {

using butteraugli_layout::BuildButterLayout;
using butteraugli_layout::kHfTaps;
using butteraugli_layout::kKernelFloats;
using butteraugli_layout::kKernelHf;
using butteraugli_layout::kKernelLf;
using butteraugli_layout::kKernelMask;
using butteraugli_layout::kKernelOpsin;
using butteraugli_layout::kKernelUhf;
using butteraugli_layout::kLfTaps;
using butteraugli_layout::kMaskTaps;
using butteraugli_layout::kOpsinTaps;
using butteraugli_layout::kTableWords;
using butteraugli_layout::kUhfTaps;
using butteraugli_layout::Layout;
using butteraugli_layout::LevelLayout;

namespace {

// std430 mirror of the push block in
// src/metrics/butteraugli/shaders/butter_common.glsl.
struct ButterPush {
    std::uint32_t level_base;
    std::uint32_t level;
    std::uint32_t op;
    std::uint32_t taps;
    std::uint32_t kernel;
    std::uint32_t image;
    std::uint32_t z_count;
    std::uint32_t desc_base;
    std::uint32_t a, b, c, d;
    float background;
    float p0, p1, p2, p3;
    float q0, q1, q2, q3;
};
static_assert(sizeof(ButterPush) == 84);

// The two source enumerations are mapped by value in Submit; keep them in step.
static_assert(
    static_cast<int>(ButteraugliSource::Kind::Stream) ==
            static_cast<int>(VulkanRuntime::Source::Kind::Stream) &&
        static_cast<int>(ButteraugliSource::Kind::Resident) ==
            static_cast<int>(VulkanRuntime::Source::Kind::Resident) &&
        static_cast<int>(ButteraugliSource::Kind::ImageBuffer) ==
            static_cast<int>(VulkanRuntime::Source::Kind::ImageBuffer),
    "source kinds must correspond"
);

// Operation codes of the blur shader (butter_blur.comp).
constexpr std::uint32_t kBlurPlain = 0;
constexpr std::uint32_t kBlurLf = 1;
constexpr std::uint32_t kBlurMf = 2;
constexpr std::uint32_t kBlurUhf = 3;

std::string OpsinName(SourceDecodeVariant variant) {
    std::string name = "butter_opsin_c" + std::to_string(variant.chroma) + "_t" +
                       std::to_string(variant.transfer);
    if (variant.chroma != 0) {
        name += "_f" + std::to_string(variant.filter);
    }
    return name;
}

// Pipelines this backend owns, in creation order.
enum PipelineSlot : std::size_t {
    kSlotBlur7 = 12,
    kSlotBlur13 = 13,
    kSlotBlur15 = 14,
    kSlotBlur33 = 15,
    kSlotMalta = 16,
    kSlotMaltaLf = 17,
    kSlotL2 = 18,
    kSlotErode = 19,
    kSlotReduce = 20,
    kPipelineCount = 21,
};

class VulkanButteraugliPipeline final : public ButteraugliGpuPipeline {
public:
    VulkanButteraugliPipeline(
        std::shared_ptr<VulkanContext> context,
        const ButteraugliGeometry& geometry, const ButteraugliParams& params,
        std::size_t resident_images, ChromaFilter chroma_filter,
        bool streaming_inputs, std::size_t input_bytes,
        MemoryPolicy memory_policy,
        std::shared_ptr<SharedMetricStorage> shared = {}
    );
    ~VulkanButteraugliPipeline() override;

    VulkanButteraugliPipeline(const VulkanButteraugliPipeline&) = delete;
    VulkanButteraugliPipeline& operator=(const VulkanButteraugliPipeline&) =
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
        const ButteraugliSource& original, const ButteraugliSource& distorted,
        float background
    ) override;
    [[nodiscard]] std::uint64_t Submit(
        const ButteraugliSource& original, const Image& distorted,
        float background
    ) override;
    [[nodiscard]] ButteraugliResult Finish(std::uint64_t token) override;

    std::optional<std::uint32_t> Upload(
        const Image& image, std::size_t capacity
    ) override;
    void ReleaseSlot(std::uint32_t slot) override;

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

private:
    void CreatePipelines();
    void PublishTables(const ButterTables& tables);
    void RecordOpsin(
        std::uint32_t level, std::uint32_t level_op, std::uint32_t image,
        const ImageFormat& format, std::uint32_t desc_base, float background
    );
    void RecordSeparation(std::uint32_t level, std::uint32_t image);
    void RecordChain(
        std::uint32_t image, const ImageFormat& format, std::uint32_t desc_base,
        float background
    );
    void RecordDiff(std::uint32_t level, bool fold);
    void RecordReduce();
    void Record(
        const std::array<VulkanRuntime::Source, 2>& sources, float background,
        bool reuse_reference
    );
    [[nodiscard]] std::uint64_t SubmitResolved(
        const std::array<VulkanRuntime::Source, 2>& sources,
        std::uint64_t reference_generation, float background,
        const VulkanSubmitSync* sync = nullptr
    );
    void Dispatch(
        std::size_t slot, ButterPush push, std::uint32_t gx, std::uint32_t gy,
        std::uint32_t gz
    );
    [[nodiscard]] ButteraugliResult ReadResult();
    void Destroy() noexcept;

    static std::uint32_t GroupsFor(std::size_t extent, std::size_t size) {
        return static_cast<std::uint32_t>((extent + size - 1) / size);
    }

    std::shared_ptr<VulkanContext> ctx_;
    std::unique_ptr<VulkanRuntime> runtime_;
    ButteraugliGeometry geometry_;
    ButteraugliParams params_;
    ButterBands bands_{};
    ChromaFilter chroma_filter_ = ChromaFilter::Bilinear;
    bool streaming_ = true;
    std::size_t base_pixels_ = 0;
    Layout layout_;
    std::array<VkPipeline, kPipelineCount> pipelines_{};

    bool reference_valid_ = false;
    ButteraugliSource reference_key_{};
    float reference_background_ = 0.0f;

    std::size_t staging_desc_ = 0;
    std::size_t staging_kernels_ = 0;
    std::size_t staging_table_ = 0;
    std::size_t staging_partials_ = 0;
};

VulkanButteraugliPipeline::VulkanButteraugliPipeline(
    std::shared_ptr<VulkanContext> context, const ButteraugliGeometry& geometry,
    const ButteraugliParams& params, std::size_t resident_images,
    ChromaFilter chroma_filter, bool streaming_inputs, std::size_t input_bytes,
    MemoryPolicy memory_policy, std::shared_ptr<SharedMetricStorage> shared
)
    : ctx_(std::move(context)),
      geometry_(geometry),
      params_(params),
      chroma_filter_(chroma_filter),
      streaming_(streaming_inputs) {
    try {
        bands_ = MakeButterBands(params_);
        const auto plan = BuildButterLayout(
            geometry_, *ctx_, streaming_, resident_images, input_bytes,
            chroma_filter_
        );
        layout_ = plan.layout;
        base_pixels_ = plan.base_pixels;
        staging_desc_ = plan.staging_desc;
        staging_kernels_ = plan.staging_kernels;
        staging_table_ = plan.staging_table;
        staging_partials_ = plan.staging_partials;

        VulkanRuntime::Config config;
        config.stream_region = layout_.stream_region;
        config.stream_desc = layout_.stream_desc;
        config.resident_slots = layout_.resident;
        config.input_pool = layout_.input_pool;
        config.input_pool_size = layout_.input_pool_size;
        config.staging_desc_offset = staging_desc_;
        config.staging_image_bytes = plan.staging.image_bytes;
        config.base_pixels = base_pixels_;
        config.base_size = geometry_.base;
        config.streaming = streaming_;
        config.memory_policy = memory_policy;
        config.chroma_filter = chroma_filter_;
        config.push_constant_size = sizeof(ButterPush);
        config.persistent_ranges = plan.persistent_ranges;
        runtime_ =
            std::make_unique<VulkanRuntime>(ctx_, config, std::move(shared));

        runtime_->CreateArena(layout_.bytes);
        if (!runtime_->arena_visible()) {
            runtime_->CreateStaging(
                Bytes(staging_partials_ + 4u * layout_.partial_groups)
            );
        }
        runtime_->CreateCommandObjects();
        // Binding 1 is unused by every Butteraugli shader; point it at the
        // arena so the descriptor is still valid.
        runtime_->CreateDescriptors(0, layout_.bytes);
        CreatePipelines();
        PublishTables(MakeButterTables());
        runtime_->InitializeSharedState();
    } catch (...) {
        Destroy();
        throw;
    }
}

VulkanButteraugliPipeline::~VulkanButteraugliPipeline() { Destroy(); }

void VulkanButteraugliPipeline::Destroy() noexcept {
    if (runtime_ && runtime_->pending()) {
        try {
            runtime_->WaitFence();
        } catch (...) {
            // A lost device cannot complete the work; destruction must not
            // throw.
        }
    }
    if (runtime_) runtime_->DestroyPipelines(pipelines_);
    pipelines_.fill(VK_NULL_HANDLE);
    runtime_.reset();
}

void VulkanButteraugliPipeline::CreatePipelines() {
    for (std::size_t i = 0; i < kSourceDecodeVariants.size(); ++i) {
        pipelines_[i] =
            runtime_->CreatePipeline(OpsinName(kSourceDecodeVariants[i]));
    }
    pipelines_[kSlotBlur7] = runtime_->CreatePipeline("butter_blur_7");
    pipelines_[kSlotBlur13] = runtime_->CreatePipeline("butter_blur_13");
    pipelines_[kSlotBlur15] = runtime_->CreatePipeline("butter_blur_15");
    pipelines_[kSlotBlur33] = runtime_->CreatePipeline("butter_blur_33");
    pipelines_[kSlotMalta] = runtime_->CreatePipeline("butter_malta");
    pipelines_[kSlotMaltaLf] = runtime_->CreatePipeline("butter_malta_lf");
    pipelines_[kSlotL2] = runtime_->CreatePipeline("butter_l2");
    pipelines_[kSlotErode] = runtime_->CreatePipeline("butter_erode");
    pipelines_[kSlotReduce] = runtime_->CreatePipeline("butter_reduce");
}

void VulkanButteraugliPipeline::PublishTables(const ButterTables& tables) {
    std::array<float, kKernelFloats> taps{};
    std::copy(tables.opsin.begin(), tables.opsin.end(), taps.begin());
    std::copy(tables.uhf.begin(), tables.uhf.end(), taps.begin() + kKernelUhf);
    std::copy(
        tables.mask.begin(), tables.mask.end(), taps.begin() + kKernelMask
    );
    std::copy(tables.hf.begin(), tables.hf.end(), taps.begin() + kKernelHf);
    std::copy(tables.lf.begin(), tables.lf.end(), taps.begin() + kKernelLf);

    const std::size_t table_bytes = kTableWords * sizeof(std::uint32_t);
    if (runtime_->arena_visible()) {
        std::memcpy(
            runtime_->arena_map(), taps.data(), kKernelFloats * sizeof(float)
        );
        std::memcpy(
            reinterpret_cast<std::uint8_t*>(runtime_->arena_map()) +
                Bytes(layout_.table_word),
            layout_.table.data(), table_bytes
        );
        return;
    }
    std::memcpy(
        runtime_->staging_map() + staging_kernels_, taps.data(),
        kKernelFloats * sizeof(float)
    );
    std::memcpy(
        reinterpret_cast<std::uint8_t*>(runtime_->staging_map()) +
            Bytes(staging_table_),
        layout_.table.data(), table_bytes
    );
    const std::array<VulkanRuntime::CopyRange, 2> copies{
        VulkanRuntime::CopyRange{staging_kernels_, 0, kKernelFloats},
        VulkanRuntime::CopyRange{staging_table_, layout_.table_word,
                                 kTableWords}};
    runtime_->RequireIdle();
    runtime_->ResetCommandPool();
    runtime_->BeginCommandBuffer();
    runtime_->RecordStagedCopies(copies);
    runtime_->EndCommandBuffer();
    runtime_->SubmitAndWait();
}

void VulkanButteraugliPipeline::Dispatch(
    std::size_t slot, ButterPush push, std::uint32_t gx, std::uint32_t gy,
    std::uint32_t gz
) {
    push.level_base = layout_.table_word;
    runtime_->Dispatch(pipelines_[slot], &push, sizeof(push), gx, gy, gz);
    runtime_->ComputeBarrier();
}

void VulkanButteraugliPipeline::RecordOpsin(
    std::uint32_t level, std::uint32_t level_op, std::uint32_t image,
    const ImageFormat& format, std::uint32_t desc_base, float background
) {
    const LevelLayout& out = layout_.level[level];
    ButterPush push{};
    push.level = level;
    push.op = level_op;
    push.image = image;
    push.kernel = kKernelOpsin;
    push.taps = kOpsinTaps;
    push.desc_base = desc_base;
    push.q2 = params_.intensity_target;
    push.background = background;
    Dispatch(
        SourceDecodeVariantIndex(
            SelectSourceDecodeVariant(format, chroma_filter_)
        ),
        push, GroupsFor((out.size.width + 1) / 2, 16u),
        GroupsFor((out.size.height + 1) / 2, 16u), 1
    );
}

void VulkanButteraugliPipeline::RecordSeparation(
    std::uint32_t level, std::uint32_t image
) {
    const LevelLayout& out = layout_.level[level];
    const std::uint32_t chain = out.chain[image];
    const std::uint32_t gx = GroupsFor(out.size.width, 32u);
    const std::uint32_t gy = GroupsFor(out.size.height, 32u);

    // LF: blur xyb, keep mf = xyb - lf, three channels (SeparateLFAndMF).
    ButterPush lf{};
    lf.level = level;
    lf.op = kBlurLf;
    lf.kernel = kKernelLf;
    lf.taps = kLfTaps;
    lf.a = out.scratch_a;
    lf.b = chain;
    lf.c = out.scratch_b;
    lf.z_count = 3;
    Dispatch(kSlotBlur33, lf, gx, gy, 3);

    // HF: blur mf, keep hf = mf_before - raw_blur, then adjust mf's range.
    // Three channels (SeparateMFAndHF, whose B channel has no hf).
    ButterPush hf{};
    hf.level = level;
    hf.op = kBlurMf;
    hf.kernel = kKernelHf;
    hf.taps = kHfTaps;
    hf.a = out.scratch_b;
    hf.b = chain + 3u * out.area;
    hf.c = chain + 6u * out.area;
    hf.z_count = 3;
    Dispatch(kSlotBlur15, hf, gx, gy, 3);

    // UHF: suppress HF X by raw HF Y, then blur/split HF and UHF. The shader
    // reads immutable planes 6/7 and writes separate planes 8..11, so sibling
    // Y reads cannot race with this dispatch's outputs (SeparateHFAndUHF).
    ButterPush uhf{};
    uhf.level = level;
    uhf.op = kBlurUhf;
    uhf.kernel = kKernelUhf;
    uhf.taps = kUhfTaps;
    uhf.a = chain + 6u * out.area;
    uhf.b = chain + 8u * out.area;
    uhf.c = chain + 10u * out.area;
    uhf.z_count = 2;
    Dispatch(kSlotBlur7, uhf, gx, gy, 2);
}

void VulkanButteraugliPipeline::RecordChain(
    std::uint32_t image, const ImageFormat& format, std::uint32_t desc_base,
    float background
) {
    for (std::uint32_t level = 0; level < layout_.levels; ++level) {
        // Level 1 consumes level 0's subsampled rgb, so it never decodes.
        runtime_->OpenStage(Stage::ButteraugliOpsin);
        RecordOpsin(
            level, level == 0 ? 0u : 1u, image, format, desc_base, background
        );
        runtime_->CloseStage();
        runtime_->OpenStage(Stage::ButteraugliSeparate);
        RecordSeparation(level, image);
        runtime_->CloseStage();
    }
}

void VulkanButteraugliPipeline::RecordDiff(std::uint32_t level, bool fold) {
    const LevelLayout& out = layout_.level[level];
    const std::uint32_t area = out.area;
    const std::uint32_t chain0 = out.chain[0];
    const std::uint32_t chain1 = out.chain[1];
    const std::uint32_t ac = out.ac;

    // AC also holds the temporary subsampled inputs during chain construction
    // and is shared between levels. Finish their shader reads before each
    // transfer overwrite; compute-to-compute barriers do not order transfers.
    runtime_->Barrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT
    );
    // ac starts at zero; dc is derived in the combine from the lf planes.
    runtime_->FillZero(ac, 3u * static_cast<std::size_t>(area));
    runtime_->Barrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
    );

    const auto band = [this](std::uint32_t index) {
        ButterPush push{};
        push.p0 = static_cast<float>(bands_.norm2_gt1[index]);
        push.p1 = static_cast<float>(bands_.norm2_lt1[index]);
        push.p2 = static_cast<float>(bands_.norm1[index]);
        return push;
    };

    runtime_->OpenStage(Stage::ButteraugliMalta);
    // Malta UHF (16-line unit): bands 0 (Y) and 1 (X).
    ButterPush uhf = band(0);
    uhf.level = level;
    uhf.a = chain0 + 10u * area;
    uhf.b = chain1 + 10u * area;
    uhf.z_count = 2;
    uhf.q0 = static_cast<float>(bands_.norm2_gt1[1]);
    uhf.q1 = static_cast<float>(bands_.norm2_lt1[1]);
    uhf.q2 = static_cast<float>(bands_.norm1[1]);
    Dispatch(
        kSlotMalta, uhf, GroupsFor(out.size.width, 32u),
        GroupsFor(out.size.height, 32u), 2
    );

    // Malta LF on HF (bands 2, 3) and on MF (bands 4, 5).
    // The HF bands read the post-UHF hf (chain + 8 planes); chain + 6 was the
    // UHF blur's input and is dead by now.
    ButterPush hf = band(2);
    hf.level = level;
    hf.a = chain0 + 8u * area;
    hf.b = chain1 + 8u * area;
    hf.z_count = 2;
    hf.q0 = static_cast<float>(bands_.norm2_gt1[3]);
    hf.q1 = static_cast<float>(bands_.norm2_lt1[3]);
    hf.q2 = static_cast<float>(bands_.norm1[3]);
    Dispatch(
        kSlotMaltaLf, hf, GroupsFor(out.size.width, 32u),
        GroupsFor(out.size.height, 32u), 2
    );

    ButterPush mf = band(4);
    mf.level = level;
    mf.a = chain0 + 3u * area;
    mf.b = chain1 + 3u * area;
    mf.z_count = 2;
    mf.q0 = static_cast<float>(bands_.norm2_gt1[5]);
    mf.q1 = static_cast<float>(bands_.norm2_lt1[5]);
    mf.q2 = static_cast<float>(bands_.norm1[5]);
    Dispatch(
        kSlotMaltaLf, mf, GroupsFor(out.size.width, 32u),
        GroupsFor(out.size.height, 32u), 2
    );

    runtime_->CloseStage();
    runtime_->OpenStage(Stage::ButteraugliDiff);
    // The L2 terms and the mask's input for both images, one pass.
    ButterPush l2{};
    l2.level = level;
    l2.q0 = params_.hf_asymmetry;
    Dispatch(kSlotL2, l2, GroupsFor(area, 256u), 1, 1);

    runtime_->CloseStage();
    runtime_->OpenStage(Stage::ButteraugliMask);
    // Blur the two pre-masks with sigma 2.7 (13 taps).
    ButterPush blur{};
    blur.level = level;
    blur.op = kBlurPlain;
    blur.kernel = kKernelMask;
    blur.taps = kMaskTaps;
    blur.a = out.scratch_a;
    blur.b = out.scratch_b;
    blur.z_count = 2;
    Dispatch(
        kSlotBlur13, blur, GroupsFor(out.size.width, 32u),
        GroupsFor(out.size.height, 32u), 2
    );

    runtime_->CloseStage();
    runtime_->OpenStage(Stage::ButteraugliCombine);
    // The tiled erosion feeds the combine directly, folding the sub level's
    // diffmap on level 0.
    ButterPush combine{};
    combine.level = level;
    combine.op = 1;
    combine.q1 = params_.xmul;
    combine.q3 = fold ? 0.5f : 0.0f;
    combine.a = fold ? layout_.level[1].diffmap : 0;
    Dispatch(
        kSlotErode, combine, GroupsFor(out.size.width, 32u),
        GroupsFor(out.size.height, 32u), 1
    );
    runtime_->CloseStage();
}

void VulkanButteraugliPipeline::RecordReduce() {
    runtime_->OpenStage(Stage::ButteraugliReduce);
    ButterPush push{};
    push.a = layout_.partials;
    Dispatch(kSlotReduce, push, layout_.partial_groups, 1, 1);
    // Second stage: the whole reduction becomes four floats.
    ButterPush fold{};
    fold.op = 1;
    fold.a = layout_.partials;
    fold.b = layout_.final_result;
    fold.q0 = static_cast<float>(layout_.partial_groups);
    Dispatch(kSlotReduce, fold, 1, 1, 1);
    runtime_->CloseStage();
}

void VulkanButteraugliPipeline::Record(
    const std::array<VulkanRuntime::Source, 2>& sources, float background,
    bool reuse_reference
) {
    runtime_->BeginCommandBuffer();
    runtime_->BindDescriptorSet();
    const auto formats = runtime_->PublishSources(sources);

    // The descriptor block is always the two stream_desc slots, whatever
    // buffer backs each side, so the shaders' desc_base is that offset -- not
    // the plane base WriteSourceDesc derives the plane words from.
    const std::array<std::uint32_t, 2> desc_bases{layout_.stream_desc[0],
                                                  layout_.stream_desc[1]};

    // 1. The reference chain, unless this immutable resident image is cached.
    if (!reuse_reference) {
        RecordChain(0u, *formats[0], desc_bases[0], background);
    }

    // 2. The distorted chain. Internal images both read the already-bound
    // arena; external images need the second source descriptor set.
    if (sources[0].kind == VulkanRuntime::Source::Kind::External ||
        sources[1].kind == VulkanRuntime::Source::Kind::External) {
        runtime_->BindSource(1);
    }
    RecordChain(1u, *formats[1], desc_bases[1], background);

    // 3. The diff stages, coarsest first so level 0 can fold level 1.
    for (std::uint32_t level = layout_.levels; level > 0; --level) {
        const std::uint32_t index = level - 1;
        RecordDiff(index, geometry_.fold && index == 0);
    }

    // 4. The score.
    RecordReduce();
    if (runtime_->arena_visible()) {
        runtime_->ComputeToHostBarrier();
    } else {
        runtime_->RecordReadback(layout_.final_result, staging_partials_, 4);
    }
    runtime_->EndCommandBuffer();
}

ButteraugliResult VulkanButteraugliPipeline::ReadResult() {
    const float* result = runtime_->arena_visible()
                              ? runtime_->arena_map() + layout_.final_result
                              : runtime_->staging_map() + staging_partials_;
    double entry[4] = {};
    for (std::size_t i = 0; i < 4; ++i) {
        entry[i] = static_cast<double>(result[i]);
    }
    const double pixels = static_cast<double>(layout_.level[0].size.width) *
                          static_cast<double>(layout_.level[0].size.height);
    double distance = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        distance += std::pow(
            entry[i + 1] / pixels, 1.0 / (3.0 * static_cast<double>(1u << i))
        );
    }
    return ButteraugliResult{entry[0], distance / 3.0};
}

ButteraugliResult VulkanButteraugliPipeline::Finish(std::uint64_t token) {
    runtime_->FinishSubmission(token);
    return ReadResult();
}

std::uint64_t VulkanButteraugliPipeline::Submit(
    const ButteraugliSource& original, const ButteraugliSource& distorted,
    float background
) {
    std::array<VulkanRuntime::Source, 2> sources{};
    const std::array<const ButteraugliSource*, 2> handles{&original, &distorted};
    for (std::size_t image = 0; image < 2; ++image) {
        sources[image].kind =
            static_cast<VulkanRuntime::Source::Kind>(handles[image]->kind);
        sources[image].index = handles[image]->index;
    }
    return SubmitResolved(sources, original.generation, background);
}

std::uint64_t VulkanButteraugliPipeline::SubmitResolved(
    const std::array<VulkanRuntime::Source, 2>& sources,
    std::uint64_t reference_generation, float background,
    const VulkanSubmitSync* sync
) {
    runtime_->RequireIdle();
    runtime_->PrepareSourceBindings(sources);
    const std::uint64_t token = runtime_->BeginSubmission();
    // Only resident pixels are immutable. Mapped and external buffers can be
    // modified without changing their handles, so they never supply a cache key.
    const auto& reference = sources[0];
    const bool cacheable =
        reference.kind == VulkanRuntime::Source::Kind::Resident;
    const bool reuse = reference_valid_ && cacheable &&
                       reference_key_.index == reference.index &&
                       reference_key_.generation == reference_generation &&
                       reference_background_ == background;
    // Recording or submission can fail. Publish the key only after successful
    // queue submission, so a retry cannot reuse a chain that was never built.
    reference_valid_ = false;
    Record(sources, background, reuse);
    if (sync != nullptr)
        runtime_->QueueSubmit(*sync);
    else
        runtime_->QueueSubmit();
    reference_valid_ = cacheable;
    if (cacheable) {
        reference_key_ = {ButteraugliSource::Kind::Resident, reference.index,
                          reference_generation};
        reference_background_ = background;
    }
    return token;
}

std::uint64_t VulkanButteraugliPipeline::Submit(
    const Image& original, const Image& distorted, float background
) {
    runtime_->RequireIdle();
    if (!streaming_) {
        throw std::invalid_argument(
            "vulkan: streaming_inputs is false, so the pixel overloads are "
            "unavailable; use Upload() and Compute() with resident handles"
        );
    }
    runtime_->UploadStreaming(original, distorted);
    ButteraugliSource a;
    ButteraugliSource b;
    a.kind = ButteraugliSource::Kind::Stream;
    b.kind = ButteraugliSource::Kind::Stream;
    return Submit(a, b, background);
}

std::uint64_t VulkanButteraugliPipeline::Submit(
    const ImageView& original, const ImageView& distorted, float background
) {
    runtime_->RequireIdle();
    if (!streaming_) {
        throw std::invalid_argument("vulkan: streaming_inputs is false");
    }
    runtime_->UploadStreaming(original, distorted);
    return Submit(ButteraugliSource{}, ButteraugliSource{}, background);
}

std::uint64_t VulkanButteraugliPipeline::Submit(
    const ButteraugliSource& original, const Image& distorted, float background
) {
    runtime_->RequireIdle();
    if (!streaming_) {
        throw std::invalid_argument("vulkan: streaming_inputs is false");
    }
    runtime_->UploadStreamingDistorted(distorted);
    return Submit(original, ButteraugliSource{}, background);
}

std::uint64_t VulkanButteraugliPipeline::Submit(
    const ImageBuffer& original, const ImageBuffer& distorted, float background
) {
    const auto resolved = runtime_->ResolveInputs(original, distorted);
    const std::array<const ImageBuffer*, 2> handles{&original, &distorted};
    std::array<ButteraugliSource, 2> sources{};
    for (std::size_t image = 0; image < 2; ++image) {
        sources[image].kind = ButteraugliSource::Kind::ImageBuffer;
        sources[image].index = resolved[image].index;
        sources[image].generation = handles[image]->slot >> 12;
    }
    return Submit(sources[0], sources[1], background);
}

std::uint64_t VulkanButteraugliPipeline::Submit(
    const VulkanBufferView& original, const VulkanBufferView& distorted,
    float background
) {
    runtime_->RequireIdle();
    if (ctx_->owns_device) {
        throw std::invalid_argument(
            "vulkan: external buffers require a caller-owned Vulkan device"
        );
    }
    return SubmitResolved(
        runtime_->ResolveExternalInputs(original, distorted), 0, background
    );
}

std::uint64_t VulkanButteraugliPipeline::Submit(
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
    return SubmitResolved(
        runtime_->ResolveExternalInputs(original, distorted), 0, background,
        &sync
    );
}

std::optional<std::uint32_t> VulkanButteraugliPipeline::Upload(
    const Image& image, std::size_t capacity
) {
    return runtime_->UploadResident(image, capacity);
}

void VulkanButteraugliPipeline::ReleaseSlot(std::uint32_t slot) {
    runtime_->ReleaseResident(slot);
}

ImageBuffer VulkanButteraugliPipeline::AllocateImage(const ImageFormat& format) {
    return runtime_->AllocateImage(format);
}

std::byte* VulkanButteraugliPipeline::MapImage(ImageBuffer buffer) {
    return runtime_->MapImage(buffer);
}

void VulkanButteraugliPipeline::Release(ImageBuffer buffer) {
    runtime_->ReleaseInput(buffer);
}

}  // namespace

std::unique_ptr<ButteraugliGpuPipeline> CreateVulkanButteraugliPipeline(
    const ButteraugliGeometry& geometry, const ButteraugliParams& params,
    std::size_t resident_images, ChromaFilter chroma_filter,
    bool streaming_inputs, std::size_t input_bytes, MemoryPolicy memory_policy,
    const VulkanDevice* borrowed_device
) {
    std::shared_ptr<VulkanContext> context =
        AcquireVulkanContext(borrowed_device);
    if (!context) return nullptr;
    return std::make_unique<VulkanButteraugliPipeline>(
        std::move(context), geometry, params, resident_images, chroma_filter,
        streaming_inputs, input_bytes, memory_policy
    );
}

std::unique_ptr<ButteraugliGpuPipeline> CreateSharedVulkanButteraugliPipeline(
    const ButteraugliGeometry& geometry, const ButteraugliParams& params,
    const ButteraugliConfig& config, std::shared_ptr<VulkanContext> context,
    std::shared_ptr<SharedMetricStorage> storage
) {
    return std::make_unique<VulkanButteraugliPipeline>(
        std::move(context), geometry, params, 0, config.chroma_filter,
        config.streaming_inputs, 0, config.memory_policy, std::move(storage)
    );
}

}  // namespace vqm::internal
