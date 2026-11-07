// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// VMAF: Copyright (c) Netflix, Inc. BSD-2-Clause-Patent;
// see licenses/NETFLIX-VMAF-LICENSE.

#include "vmaf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "handles.h"
#include "input.h"
#include "metrics/vmaf/features/vmaf_adm.h"
#include "metrics/vmaf/features/vmaf_preprocess.h"
#include "metrics/vmaf/vmaf_memory.h"
#include "metrics/vmaf/v0/vmaf_v0_internal.h"
#include "metrics/vmaf/features/vmaf_vif.h"
#include "vqm_views.h"
#include "vqm_vulkan.h"
#include "vulkan/vulkan_runtime.h"
#include "vulkan/vulkan_support.h"

// Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Motion2 uses the default v0.6.1 timeline, and libvmaf predicts each frame
// before computing the arithmetic mean (resource/doc/faq.md at revision
// 4611e47a7330cd342498dcc130e1008efbeab6e4).
// Forcing motion2 to zero is a derived VQM scoring mode, not libvmaf behavior.

namespace vqm::internal {

VmafV061Sequence::VmafV061Sequence(
    bool transform, bool clip, VmafModel model, bool temporal_weighting
)
    : transform_(transform),
      clip_(clip),
      model_(model),
      temporal_weighting_(temporal_weighting) {}

VmafV061Frame VmafV061Sequence::FinishFrame(
    std::uint64_t index, const VmafV061Spatial& spatial, double motion
) {
    const VmafV061Features features{spatial.adm2, motion, spatial.vif};
    const double score = VmafV061Predict(features, transform_, clip_, model_);
    score_sum_ += score;
    ++emitted_;
    return {index, features, score};
}

std::optional<VmafV061Frame> VmafV061Sequence::Push(
    const VmafV061Spatial& spatial, double transition
) {
    if (finished_) throw std::logic_error("vmaf: sequence is finished");
    if (!std::isfinite(spatial.adm2) ||
        !std::all_of(spatial.vif.begin(), spatial.vif.end(), [](double value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument("vmaf: spatial features must be finite");
    }
    if (!temporal_weighting_) return FinishFrame(emitted_, spatial, 0.0);
    const auto ready = motion_.Push(transition);
    if (!ready) {
        pending_ = spatial;
        return std::nullopt;
    }
    if (ready->index == 0) {
        return FinishFrame(0, spatial, ready->motion2);
    }
    if (!pending_) throw std::logic_error("vmaf: missing pending frame");
    const auto result = FinishFrame(ready->index, *pending_, ready->motion2);
    pending_ = spatial;
    return result;
}

std::optional<VmafV061Frame> VmafV061Sequence::Flush() {
    if (finished_) return std::nullopt;
    if (!temporal_weighting_) {
        finished_ = true;
        return std::nullopt;
    }
    const auto ready = motion_.Flush();
    finished_ = true;
    if (!ready) return std::nullopt;
    if (!pending_) throw std::logic_error("vmaf: missing pending frame");
    const auto result = FinishFrame(ready->index, *pending_, ready->motion2);
    pending_.reset();
    return result;
}

std::optional<double> VmafV061Sequence::PooledScore() const {
    if (!finished_)
        throw std::logic_error("vmaf: flush sequence before pooling");
    if (emitted_ == 0) return std::nullopt;
    return score_sum_ / static_cast<double>(emitted_);
}

void VmafV061Sequence::Reset() noexcept {
    motion_.Reset();
    pending_.reset();
    score_sum_ = 0.0;
    emitted_ = 0;
    finished_ = false;
}

}  // namespace vqm::internal

namespace vqm {
namespace {
using namespace internal;

// The GPU writes these consecutive regions into one readback: four VIF
// values, a motion value, and ADM's eight scale terms and final score.
struct ReadbackLayout {
    static constexpr std::uint32_t kVifWords = 4;
    static constexpr std::uint32_t kMotion = kVifWords;
    static constexpr std::uint32_t kAdmStart = kMotion + 1;
    static constexpr std::uint32_t kAdmWords = 9;
    static constexpr std::uint32_t kAdmScore = kAdmStart + kAdmWords - 1;
    static constexpr std::uint32_t kWords = kAdmStart + kAdmWords;
};

// If either input needs color conversion, convert both to the same coded
// BT.709 space. Native luma can otherwise be read directly without rounding.
bool SameNativeCoding(const ImageFormat& a, const ImageFormat& b) {
    return IsVmafNativeInput(a) && IsVmafNativeInput(b) &&
           a.family == b.family && a.range == b.range && a.matrix == b.matrix &&
           a.transfer == b.transfer && a.primaries == b.primaries;
}

// VMAF's feature domain is coded 8-bit luma minus 128. Limited-range black is
// 16 -> -112 in that domain; full-range black is 0 -> -128. Explicit SDR
// conversion always emits limited-range BT.709, so its canvas is -112.
float BlackCode(const ImageFormat& format, bool native) {
    return native && format.range == PixelRange::Full ? -128.0f : -112.0f;
}

Size ModelCanvas(VmafModel model) {
    if (model == VmafModel::Standard4K) return {3840, 2160};
    if (model == VmafModel::Standard1080) return {1920, 1080};
    throw std::invalid_argument("vmaf: unknown model");
}

void ValidateDispatchLimits(
    Size size, const VkPhysicalDeviceLimits& limits, bool motion
) {
    // Luma and VIF use a 16x16 grid; motion uses 32x8. ADM uses
    // downsampled geometry and every reduction uses a single workgroup.
    // ADM's six 256-float shared arrays are the largest unconditional block;
    // VIF selects its shared-memory fused shaders only when supported.
    constexpr std::uint32_t kMinSharedBytes = 6 * 256 * sizeof(float);
    if (limits.maxComputeWorkGroupSize[0] < 256 ||
        limits.maxComputeWorkGroupSize[1] < 16 ||
        limits.maxComputeWorkGroupSize[2] < 1 ||
        limits.maxComputeWorkGroupInvocations < 256 ||
        limits.maxComputeSharedMemorySize < kMinSharedBytes ||
        limits.maxPushConstantsSize <
            std::max(
                {VmafAdm::PushBytes(), VmafSdrConverter::PushBytes(),
                 VmafResizer::PushBytes()}
            ) ||
        Groups(size.width, 16) > limits.maxComputeWorkGroupCount[0] ||
        Groups(size.height, motion ? 8 : 16) >
            limits.maxComputeWorkGroupCount[1] ||
        limits.maxComputeWorkGroupCount[2] < 1) {
        throw std::invalid_argument(
            "vmaf: compute dispatch exceeds device limits"
        );
    }
}

struct Layout {
    std::array<std::uint32_t, 2> source{};
    std::array<std::uint32_t, 2> desc{};
    std::vector<std::uint32_t> resident;
    VmafVifBuffers vif{};
    VmafAdmBuffers adm{};
    VmafMotionBuffers motion{};
    Size base_size{};
    Size target_size{};
    Size content_size{};
    std::uint32_t content_x = 0;
    std::uint32_t content_y = 0;
    std::array<std::uint32_t, 2> luma{};
    std::uint32_t resize_scratch = 0;
    bool fit = false;
    bool horizontal_resize = false;
    bool temporal_weighting = false;
    std::uint32_t total = 0;
    StagingSources staging{};
    std::size_t readback = 0;

    explicit Layout(const VmafConfig& config, std::uint32_t shared_memory)
        : temporal_weighting(config.temporal_weighting) {
        const Size source_size = config.size;
        base_size = source_size;
        if (config.model_size.width == 0 && config.model_size.height != 0)
            throw std::invalid_argument("vmaf: incomplete model_size");
        if (config.model_size.height == 0 && config.model_size.width != 0)
            throw std::invalid_argument("vmaf: incomplete model_size");
        const Size model_canvas = ModelCanvas(config.model);
        if (config.fit != VmafFitMode::Native &&
            config.fit != VmafFitMode::FitAndPad)
            throw std::invalid_argument("vmaf: invalid fit mode");
        target_size = config.fit == VmafFitMode::FitAndPad
                          ? (config.model_size.width == 0 ? model_canvas
                                                          : config.model_size)
                          : source_size;
        if (config.chroma_sampling != VmafChromaSampling::Native &&
            config.chroma_sampling != VmafChromaSampling::Yuv420 &&
            config.chroma_sampling != VmafChromaSampling::Yuv444)
            throw std::invalid_argument("vmaf: invalid chroma sampling");
        if (!std::isfinite(config.background) || config.background < 0.0f ||
            config.background > 1.0f)
            throw std::invalid_argument("vmaf: invalid alpha background");
        if (source_size.width < 17 || source_size.height < 17 ||
            source_size.width >
                std::numeric_limits<std::uint32_t>::max() / source_size.height ||
            target_size.width < 17 || target_size.height < 17 ||
            target_size.width >
                std::numeric_limits<std::uint32_t>::max() / target_size.height) {
            throw std::invalid_argument(
                "vmaf: dimensions must be at least 17 and fit in 32 bits"
            );
        }
        if (config.gpu_resident_images > kResidentIndexMask)
            throw std::invalid_argument("vmaf: too many resident images");
        const std::size_t source_pixels = source_size.width * source_size.height;
        const std::size_t target_pixels = target_size.width * target_size.height;
        // Bound multiplications and the 32-bit arena indices before calculating
        // any sizes, including on 32-bit hosts.
        if (source_pixels > std::numeric_limits<std::uint32_t>::max() / 64u ||
            target_pixels > std::numeric_limits<std::uint32_t>::max() / 64u)
            throw std::invalid_argument(
                "vmaf: image geometry exceeds arena addressing"
            );
        if (config.fit == VmafFitMode::FitAndPad) {
            const auto geometry = FitVmafInto(source_size, target_size);
            content_size = geometry.content;
            content_x = geometry.x;
            content_y = geometry.y;
        } else {
            content_size = target_size;
        }
        fit = config.fit == VmafFitMode::FitAndPad && source_size != target_size;
        horizontal_resize = fit && content_size.width != source_size.width;
        std::uint64_t cursor = 0;
        auto add = [&](std::uint64_t count) {
            if (count > std::numeric_limits<std::uint32_t>::max() - cursor)
                throw std::invalid_argument(
                    "vmaf: image geometry exceeds arena addressing"
                );
            const auto result = static_cast<std::uint32_t>(cursor);
            cursor += count;
            return result;
        };
        if (config.streaming_inputs) {
            source[0] = add(4 * source_pixels);
            source[1] = add(4 * source_pixels);
        }
        desc[0] = add(kSourceDescWords);
        desc[1] = add(kSourceDescWords);
        resident.reserve(config.gpu_resident_images);
        for (std::size_t i = 0; i < config.gpu_resident_images; ++i)
            resident.push_back(add(4 * source_pixels));
        if (fit) {
            luma[0] = add(source_pixels);
            luma[1] = add(source_pixels);
            if (horizontal_resize) {
                const std::uint64_t scratch =
                    static_cast<std::uint64_t>(content_size.width) *
                    source_size.height;
                resize_scratch = add(scratch);
            }
        }
        Size level = target_size;
        for (std::size_t scale = 0; scale < 4; ++scale) {
            const std::size_t count = level.width * level.height;
            vif.reference[scale] = add(count);
            vif.distorted[scale] = add(count);
            level.width /= 2;
            level.height /= 2;
        }
        vif.scratch =
            add(VmafVif::ScratchWords(
                target_size, VmafVif::SupportsFusedScale0(shared_memory)
            ));
        vif.partials = add(2 * VmafVif::PartialCount(target_size));
        // These allocations must follow ReadbackLayout's contiguous order.
        vif.results = add(ReadbackLayout::kVifWords);
        motion.luma = vif.reference[0];
        motion.score = add(1);
        adm.results = add(ReadbackLayout::kAdmWords);
        adm.reference[0] = vif.reference[0];
        adm.distorted[0] = vif.distorted[0];
        level = target_size;
        for (std::size_t scale = 1; scale < 5; ++scale) {
            level = VmafAdmWavelet::OutputSize(level);
            const std::size_t count = level.width * level.height;
            adm.reference[scale] = add(count);
            adm.distorted[scale] = add(count);
        }
        const auto bands = 3 * VmafAdm::BandPixels(target_size);
        adm.reference_details = add(bands);
        adm.restored = add(bands);
        adm.additive = add(bands);
        adm.partials = add(6 * VmafAdm::PartialCount(target_size));
        if (config.temporal_weighting) {
            motion.previous = add(target_pixels);
            motion.blurred = add(target_pixels);
            motion.partials = add(VmafMotion::PartialCount(target_size));
        }
        if (!fit) {
            luma[0] = vif.reference[0];
            luma[1] = vif.distorted[0];
        }
        total = static_cast<std::uint32_t>(cursor);
        staging = PlanStagingSources(
            source_pixels, config.streaming_inputs, config.gpu_resident_images
        );
        readback = staging.reserved_floats + 2 * kSourceDescWords;
    }

    [[nodiscard]] BufferMemoryPlan MemoryPlan(
        const VulkanContext& context
    ) const {
        ValidateDispatchLimits(
            base_size, context.limits,
            temporal_weighting && target_size == base_size
        );
        if (target_size != base_size)
            ValidateDispatchLimits(
                target_size, context.limits, temporal_weighting
            );
        const auto arena_bytes = Bytes(total);
        if (arena_bytes > context.limits.maxStorageBufferRange) {
            throw std::invalid_argument(
                "vmaf: image geometry exceeds device storage buffer range"
            );
        }
        return {arena_bytes, Bytes(readback + ReadbackLayout::kWords)};
    }
};

std::optional<VmafFrame> PublicFrame(const std::optional<VmafV061Frame>& frame) {
    if (!frame) return std::nullopt;
    return VmafFrame{frame->index, frame->features.adm2, frame->features.motion2,
                     frame->features.vif, frame->score};
}

}  // namespace

namespace internal {
BufferMemoryPlan PlanVmafMemory(
    const VmafConfig& config, const VulkanContext& context
) {
    return Layout(config, context.limits.maxComputeSharedMemorySize)
        .MemoryPlan(context);
}
}  // namespace internal

struct VmafPipeline::Impl {
    explicit Impl(VmafConfig pipeline_config, const VulkanDevice* device)
        : config(pipeline_config),
          context(RequireVulkanContext(device)),
          layout(pipeline_config, context->limits.maxComputeSharedMemorySize),
          runtime(context, RuntimeConfig()),
          sequence(
              pipeline_config.transform, pipeline_config.clip,
              pipeline_config.model, pipeline_config.temporal_weighting
          ) {
        const auto memory = layout.MemoryPlan(*context);
        runtime.CreateArena(memory.arena_bytes);
        if (!runtime.arena_visible())
            runtime.CreateStaging(
                static_cast<std::size_t>(memory.fallback_staging_bytes)
            );
        runtime.CreateCommandObjects();
        runtime.CreateDescriptors(0, memory.arena_bytes);
        luma.emplace(runtime);
        sdr.emplace(runtime);
        vif.emplace(
            runtime, true, std::numeric_limits<std::uint32_t>::max(),
            pipeline_config.neg ? 1.0f : 100.0f
        );
        adm.emplace(runtime, pipeline_config.neg);
        if (pipeline_config.temporal_weighting) motion.emplace(runtime);
        if (layout.fit) resizer.emplace(runtime);
        resident.Resize(pipeline_config.gpu_resident_images);
    }

    VulkanRuntime::Config RuntimeConfig() {
        VulkanRuntime::Config out;
        out.stream_region = layout.source;
        out.stream_desc = layout.desc;
        out.resident_slots = layout.resident;
        const std::size_t pixels = config.size.width * config.size.height;
        out.staging_desc_offset = layout.staging.reserved_floats;
        out.staging_image_bytes = layout.staging.image_bytes;
        out.base_pixels = pixels;
        out.base_size = config.size;
        out.streaming = config.streaming_inputs;
        out.memory_policy = config.memory_policy;
        out.push_constant_size = std::max(
            {VmafAdm::PushBytes(), VmafSdrConverter::PushBytes(),
             VmafResizer::PushBytes()}
        );
        return out;
    }

    void RequireOpen() const {
        if (finished)
            throw std::logic_error("vmaf: sequence is finished; call Reset");
    }
    void RequireStreaming() const {
        if (!config.streaming_inputs)
            throw std::invalid_argument("vmaf: streaming input is disabled");
    }
    bool UsesNativeCoding(const ImageFormat& a, const ImageFormat& b) const {
        // Motion history must stay in the code domain chosen by frame zero.
        const bool native =
            config.chroma_sampling == VmafChromaSampling::Native &&
            (previous ? native_sequence : SameNativeCoding(a, b));
        if (previous && native &&
            (!SameNativeCoding(a, b) || !SameNativeCoding(native_format, a)))
            throw std::invalid_argument(
                "vmaf: input coding changed in a native sequence; call Reset "
                "or select explicit chroma conversion"
            );
        return native;
    }
    std::array<std::uint8_t, 2> StreamingPlaneMasks(
        const ImageFormat& a, const ImageFormat& b
    ) const {
        // Conversion reconstructs RGB from all source planes, including when
        // Native mode selected conversion for a mixed pair or an earlier frame.
        constexpr std::uint8_t kAllPlanes = 0x0f;
        constexpr std::uint8_t kLuma = 0x01;
        return UsesNativeCoding(a, b) ? std::array{kLuma, kLuma}
                                      : std::array{kAllPlanes, kAllPlanes};
    }
    UploadedImage Upload(const Image& image) {
        RequireOpen();
        ValidateVmafSdrFormat(image.format);
        CheckImage(image, config.size, "image");
        return resident.Upload(image.size, [this, &image](std::size_t capacity) {
            return runtime.UploadResident(image, capacity);
        });
    }
    void Release(UploadedImage image) {
        resident.Release(image, [this](std::uint32_t slot) {
            runtime.ReleaseResident(slot);
        });
    }
    std::optional<VmafFrame> Run(
        const std::array<VulkanRuntime::Source, 2>& sources,
        const VulkanSubmitSync* sync = nullptr
    ) {
        RequireOpen();
        if (sync) VulkanRuntime::ValidateSubmitSync(*sync);
        runtime.PrepareSourceBindings(sources);
        const auto token = runtime.BeginSubmission();
        runtime.BeginCommandBuffer();
        const auto formats = runtime.PublishSources(sources);
        const bool native = UsesNativeCoding(*formats[0], *formats[1]);
        const bool half_vif_source = native && !layout.fit &&
                                     VmafLumaDepth(*formats[0]) <= 12u &&
                                     VmafLumaDepth(*formats[1]) <= 12u;
#ifdef VQM_STAGE_TIMING
        runtime.OpenStage(Stage::VmafLuma);
#endif
        if (native) {
            luma->Record(
                0, layout.desc[0], layout.luma[0], layout.base_size, *formats[0]
            );
            luma->Record(
                1, layout.desc[1], layout.luma[1], layout.base_size, *formats[1]
            );
        } else {
            sdr->Record(
                0, layout.desc[0], layout.luma[0], 0, 0, layout.base_size,
                *formats[0], false, false, config.background
            );
            sdr->Record(
                1, layout.desc[1], layout.luma[1], 0, 0, layout.base_size,
                *formats[1], false, false, config.background
            );
        }
        if (layout.fit) {
            runtime.ComputeBarrier();
            const Size vertical_source{layout.content_size.width,
                                       layout.base_size.height};
            const VmafFitGeometry geometry{layout.content_size, layout.content_x,
                                           layout.content_y};
            for (std::size_t image = 0; image < 2; ++image) {
                std::uint32_t source = layout.luma[image];
                if (layout.horizontal_resize) {
                    resizer->RecordHorizontal(
                        layout.luma[image], layout.resize_scratch,
                        layout.base_size, vertical_source
                    );
                    runtime.ComputeBarrier();
                    source = layout.resize_scratch;
                }
                const std::uint32_t destination = image == 0
                                                      ? layout.vif.reference[0]
                                                      : layout.vif.distorted[0];
                resizer->RecordVertical(
                    source, destination, vertical_source, geometry,
                    layout.target_size, BlackCode(*formats[image], native)
                );
                runtime.ComputeBarrier();
            }
        }
#ifdef VQM_STAGE_TIMING
        runtime.CloseStage();
        runtime.OpenStage(Stage::VmafVif);
#endif
        vif->Record(layout.target_size, layout.vif, half_vif_source);
#ifdef VQM_STAGE_TIMING
        runtime.CloseStage();
        runtime.OpenStage(Stage::VmafAdm);
#endif
        adm->Record(layout.target_size, layout.adm);
#ifdef VQM_STAGE_TIMING
        runtime.CloseStage();
        if (config.temporal_weighting) runtime.OpenStage(Stage::VmafMotion);
#endif
        if (config.temporal_weighting)
            motion->Record(layout.target_size, layout.motion, previous);
#ifdef VQM_STAGE_TIMING
        if (config.temporal_weighting) runtime.CloseStage();
#endif
        if (runtime.arena_visible())
            runtime.ComputeToHostBarrier();
        else
            runtime.RecordReadback(
                layout.vif.results, layout.readback, ReadbackLayout::kWords
            );
        runtime.EndCommandBuffer();
        if (sync)
            runtime.QueueSubmit(*sync);
        else
            runtime.QueueSubmit();
        runtime.FinishSubmission(token);
        const float* values = runtime.arena_visible()
                                  ? runtime.arena_map() + layout.vif.results
                                  : runtime.staging_map() + layout.readback;
        VmafV061Spatial spatial;
        spatial.adm2 = values[ReadbackLayout::kAdmScore];
        std::copy_n(values, ReadbackLayout::kVifWords, spatial.vif.begin());
        const double transition =
            config.temporal_weighting ? values[ReadbackLayout::kMotion] : 0.0;
        VQM_TIMED(VmafScore);
        const auto result = sequence.Push(spatial, transition);
        if (config.temporal_weighting)
            std::swap(layout.motion.previous, layout.motion.blurred);
        if (!previous) {
            native_sequence = native;
            if (native) native_format = *formats[0];
        }
        previous = true;
        return PublicFrame(result);
    }

    VmafConfig config;
    std::shared_ptr<VulkanContext> context;
    Layout layout;
    VulkanRuntime runtime;
    std::optional<VmafLumaConverter> luma;
    std::optional<VmafSdrConverter> sdr;
    std::optional<VmafVif> vif;
    std::optional<VmafAdm> adm;
    std::optional<VmafMotion> motion;
    std::optional<VmafResizer> resizer;
    ResidentTable resident;
    VmafV061Sequence sequence;
    ImageFormat native_format{};
    bool native_sequence = false;
    bool previous = false;
    bool finished = false;
};

VmafPipeline::VmafPipeline(VmafConfig config)
    : impl_(std::make_unique<Impl>(config, nullptr)) {}
VmafPipeline::VmafPipeline(VmafConfig config, const VulkanDevice& device)
    : impl_(std::make_unique<Impl>(config, &device)) {}
VmafPipeline::~VmafPipeline() = default;
VmafPipeline::VmafPipeline(VmafPipeline&&) noexcept = default;
VmafPipeline& VmafPipeline::operator=(VmafPipeline&&) noexcept = default;
PipelineMemoryUsage VmafPipeline::GetMemoryUsage() const {
    return impl_->runtime.GetMemoryUsage();
}
UploadedImage VmafPipeline::Upload(const Image& image) {
    return impl_->Upload(image);
}
void VmafPipeline::Release(UploadedImage image) { impl_->Release(image); }
std::optional<VmafFrame> VmafPipeline::Push(const Image& a, const Image& b) {
    impl_->RequireOpen();
    impl_->RequireStreaming();
    ValidateVmafSdrFormat(a.format);
    ValidateVmafSdrFormat(b.format);
    impl_->runtime.UploadStreaming(
        a, b, impl_->StreamingPlaneMasks(a.format, b.format)
    );
    return impl_->Run({});
}
std::optional<VmafFrame> VmafPipeline::Push(
    const ImageView& a, const ImageView& b
) {
    impl_->RequireOpen();
    impl_->RequireStreaming();
    ValidateVmafSdrFormat(a.format);
    ValidateVmafSdrFormat(b.format);
    impl_->runtime.UploadStreaming(
        a, b, impl_->StreamingPlaneMasks(a.format, b.format)
    );
    return impl_->Run({});
}
std::optional<VmafFrame> VmafPipeline::Push(
    const UploadedImage& a, const UploadedImage& b
) {
    impl_->RequireOpen();
    const auto& first =
        impl_->resident.Require(a, impl_->config.size, "original");
    const auto& second =
        impl_->resident.Require(b, impl_->config.size, "distorted");
    return impl_->Run(
        {VulkanRuntime::Source{VulkanRuntime::Source::Kind::Resident,
                               first.device, nullptr},
         VulkanRuntime::Source{VulkanRuntime::Source::Kind::Resident,
                               second.device, nullptr}}
    );
}
std::optional<VmafFrame> VmafPipeline::Push(
    const VulkanBufferView& a, const VulkanBufferView& b
) {
    impl_->RequireOpen();
    ValidateVmafSdrFormat(a.format);
    ValidateVmafSdrFormat(b.format);
    return impl_->Run(impl_->runtime.ResolveExternalInputs(a, b));
}
std::optional<VmafFrame> VmafPipeline::Push(
    const VulkanBufferView& a, const VulkanBufferView& b,
    const VulkanSubmitSync& sync
) {
    impl_->RequireOpen();
    ValidateVmafSdrFormat(a.format);
    ValidateVmafSdrFormat(b.format);
    return impl_->Run(impl_->runtime.ResolveExternalInputs(a, b), &sync);
}
std::optional<VmafFrame> VmafPipeline::Flush() {
    VQM_TIMED(VmafScore);
    auto result = impl_->sequence.Flush();
    impl_->finished = true;
    return PublicFrame(result);
}
std::optional<double> VmafPipeline::PooledScore() const {
    return impl_->sequence.PooledScore();
}
void VmafPipeline::Reset() {
    impl_->sequence.Reset();
    impl_->previous = false;
    impl_->finished = false;
}
}  // namespace vqm
