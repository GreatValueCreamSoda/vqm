// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// VMAF: Copyright (c) Netflix, Inc. BSD-2-Clause-Patent;
// see licenses/NETFLIX-VMAF-LICENSE.
#include "vmaf_v1.h"
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
#include <vector>
#include "handles.h"
#include "input.h"
#include "metrics/vmaf/features/vmaf_adm.h"
#include "metrics/vmaf/cambi/vmaf_cambi.h"
#include "metrics/vmaf/features/vmaf_preprocess.h"
#include "metrics/vmaf/features/vmaf_hdr.h"
#include "metrics/vmaf/vmaf_memory.h"
#include "metrics/vmaf/speed/vmaf_speed.h"
#include "metrics/vmaf/v1/vmaf_v1_internal.h"
#include "vqm_views.h"
#include "vqm_vulkan.h"
#include "vulkan/vulkan_runtime.h"
#include "vulkan/vulkan_shared_storage.h"
#include "vulkan/vulkan_support.h"

// VMAF motion: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Default v1 motion3 timeline follows libvmaf integer_motion.c:370-451 at
// revision 4611e47a7330cd342498dcc130e1008efbeab6e4.

namespace vqm::internal {

std::optional<VmafMotionFrame> VmafV1MotionTimeline::Push(double transition) {
    if (finished_) throw std::logic_error("vmaf motion3: sequence is finished");
    if (!std::isfinite(transition) || transition < 0.0)
        throw std::invalid_argument("vmaf motion3: invalid transition");
    const auto stride = hfr_ ? 2u : 1u;
    if (frames_ < stride && transition != 0.0)
        throw std::invalid_argument("vmaf motion3: missing reference history");
    if (frames_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("vmaf motion3: too many frames");
    const auto index = frames_++;
    if (index < stride) return std::nullopt;
    const auto current = std::min(transition, 18.0);
    const double score = hfr_         ? AdvanceHfr(index, current)
                         : index == 1 ? current
                                      : std::min(previous_transition_, current);
    older_transition_ = previous_transition_;
    previous_transition_ = current;
    return VmafMotionFrame{next_++, score};
}

double VmafV1MotionTimeline::AdvanceHfr(std::uint64_t index, double transition) {
    if (index == 2) {
        hfr_motion_ = hfr_processed_ = transition;
        return transition;
    }
    // Compute motion3 for the previous frame. Keep it until that frame's
    // spatial features reach the head of the two-frame output delay.
    const auto score = hfr_motion_;
    const auto processed =
        index == 3 ? transition : std::min(older_transition_, transition);
    hfr_motion_ = (processed + hfr_processed_) / 2.0;
    hfr_processed_ = processed;
    return score;
}

std::optional<VmafMotionFrame> VmafV1MotionTimeline::Flush() {
    finished_ = true;
    if (Drained()) return std::nullopt;
    double score = previous_transition_;
    if (hfr_) {
        score = next_ + 1 < frames_
                    ? hfr_motion_
                    : (previous_transition_ + hfr_processed_) / 2.0;
    }
    return VmafMotionFrame{next_++, score};
}
void VmafV1MotionTimeline::Reset() noexcept {
    *this = VmafV1MotionTimeline(hfr_);
}

}  // namespace vqm::internal

// VMAF: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

namespace vqm::internal {

std::optional<VmafV1Frame> VmafV1Sequence::FinishFrame(
    const std::optional<VmafMotionFrame>& ready
) {
    if (!ready) return std::nullopt;
    const auto& spatial = pending_[ready->index % pending_.size()];
    const VmafV1Features features{spatial.cambi, spatial.speed_chroma_uv,
                                  spatial.adm3, ready->motion2};
    const double score = VmafV1Predict(features, clip_, model_);
    score_sum_ += score;
    ++emitted_;
    return VmafV1Frame{ready->index, features, score};
}
std::optional<VmafV1Frame> VmafV1Sequence::Push(
    const VmafV1Spatial& spatial, double transition
) {
    if (!std::isfinite(spatial.cambi) ||
        !std::isfinite(spatial.speed_chroma_uv) || !std::isfinite(spatial.adm3))
        throw std::invalid_argument("vmaf v1: spatial features must be finite");
    const auto ready = motion_.Push(transition);
    pending_[submitted_++ % pending_.size()] = spatial;
    return FinishFrame(ready);
}
std::optional<VmafV1Frame> VmafV1Sequence::Flush() {
    return FinishFrame(motion_.Flush());
}
std::optional<double> VmafV1Sequence::PooledScore() const {
    if (!motion_.Drained())
        throw std::logic_error("vmaf v1: flush sequence before pooling");
    if (emitted_ == 0) return std::nullopt;
    return score_sum_ / static_cast<double>(emitted_);
}
void VmafV1Sequence::Reset() noexcept {
    motion_.Reset();
    submitted_ = 0;
    score_sum_ = 0;
    emitted_ = 0;
}

}  // namespace vqm::internal

namespace vqm {
namespace {
using namespace internal;

// The GPU writes adjacent ADM, SpEED, CAMBI and motion results so they can
// be read back together. ADM and SpEED also include intermediate/status words.
struct ReadbackLayout {
    static constexpr std::uint32_t kAdmWords = 13;
    static constexpr std::uint32_t kAdmScore = kAdmWords - 1;
    static constexpr std::uint32_t kSpeedScore = kAdmWords;
    static constexpr std::uint32_t kSpeedWords = 2;
    static constexpr std::uint32_t kSpeedStatus = kSpeedScore + 1;
    static constexpr std::uint32_t kCambi = kSpeedScore + kSpeedWords;
    static constexpr std::uint32_t kMotion = kCambi + 1;
    static constexpr std::uint32_t kWords = kMotion + 1;
};

constexpr std::size_t RequiredPushBytes() {
    return std::max(
        {VmafAdm3::PushBytes(), VmafCambi::PushBytes(), VmafSpeed::PushBytes(),
         VmafSpeedPrescaler::PushBytes(), VmafMotion::PushBytes(),
         VmafLumaConverter::PushBytes(), VmafSdrConverter::PushBytes(),
         VmafResizer::PushBytes(), VmafHdrConverter::PushBytes()}
    );
}

void ValidateDispatchLimits(
    Size source_size, Size size, Size chroma_size, const VmafCambiPlan& cambi,
    const VkPhysicalDeviceLimits& limits, bool hdr
) {
    // SpEED covariance emits 1250 groups for each of its two channels, and
    // the batched filter uses four independent planes in dispatch Z. CAMBI
    // pooling can fall back from a packed X grid to the largest scale by 5.
    // Contrast emits 16x8 outputs per workgroup on its largest scale.
    const Size filtered{chroma_size.width / 16, chroma_size.height / 16};
    const auto speed_partials = VmafSpeedStats::PartialCount(filtered);
    const auto cambi_partials =
        VmafCambiPool::PartialCount(cambi.max_scale_pixels);
    const auto groups_x =
        std::max<std::size_t>({Groups(source_size.width, 16),
                               Groups(size.width, 16), 2500, speed_partials,
                               cambi_partials});
    const auto groups_y = std::max<std::uint32_t>(
        {Groups(source_size.height, 16), Groups(size.height, 8),
         Groups(cambi.max_scale_size.height, hdr ? 2u : 8u), 5}
    );
    // The 256x1 and 16x16 shaders are always used. The optional 512-wide
    // SpEED eigen shader is selected only when the device supports it.
    // SDR CAMBI contrast's fixed shared arrays occupy 16,004 bytes.
    // The packed PQ variant fits in the same limit.
    constexpr std::uint32_t kMinSharedBytes =
        (48 * 40 + 8 * 256 + 1 + 8 * 4) * 4;
    if (limits.maxComputeWorkGroupSize[0] < 256 ||
        limits.maxComputeWorkGroupSize[1] < 16 ||
        limits.maxComputeWorkGroupSize[2] < 1 ||
        limits.maxComputeWorkGroupInvocations < 256 ||
        limits.maxComputeSharedMemorySize < kMinSharedBytes ||
        limits.maxPushConstantsSize < RequiredPushBytes() ||
        groups_x > limits.maxComputeWorkGroupCount[0] ||
        groups_y > limits.maxComputeWorkGroupCount[1] ||
        limits.maxComputeWorkGroupCount[2] < 4) {
        throw std::invalid_argument(
            "vmaf v1: compute dispatch exceeds device limits"
        );
    }
}

using FramePlanes = std::array<std::array<std::uint32_t, 3>, 2>;

std::size_t CheckedPixels(Size size) {
    if (size.width < 17 || size.height < 17 ||
        size.width > std::numeric_limits<std::uint32_t>::max() / size.height)
        throw std::invalid_argument("vmaf v1: invalid image geometry");
    const std::size_t pixels = size.width * size.height;
    if (pixels > std::numeric_limits<std::uint32_t>::max() / 64u)
        throw std::invalid_argument(
            "vmaf v1: geometry exceeds arena addressing"
        );
    return pixels;
}

Size SubsampledSize(Size size, unsigned width_shift, unsigned height_shift) {
    const auto width_bias = (std::size_t{1} << width_shift) - 1;
    const auto height_bias = (std::size_t{1} << height_shift) - 1;
    return {(size.width + width_bias) >> width_shift,
            (size.height + height_bias) >> height_shift};
}

struct PlaneFit {
    Size source, canvas;
    VmafFitGeometry geometry;

    [[nodiscard]] Size HorizontalSize() const {
        return {geometry.content.width, source.height};
    }
    [[nodiscard]] std::uint64_t ScratchWords() const {
        return geometry.content.width == source.width
                   ? 0
                   : static_cast<std::uint64_t>(geometry.content.width) *
                         source.height;
    }
};

struct FrameFit {
    PlaneFit luma, chroma;
    FramePlanes source{};
    std::uint32_t scratch = 0;
};

struct Layout {
    std::array<std::uint32_t, 2> source{}, desc{};
    std::vector<std::uint32_t> resident;
    VmafAdmBuffers adm{};
    VmafSpeedBuffers speed{};
    VmafCambiBuffers cambi{};
    VmafMotionBuffers motion{};
    std::uint32_t older_motion = 0;
    Size base_size{};
    Size feature_size{};
    Size chroma_size{};
    Size speed_size{};
    std::array<std::uint32_t, 4> chroma_source{};
    std::optional<FrameFit> fit;
    VmafCambiPlan cambi_plan{};
    std::optional<VmafHdrBuffers> hdr;
    std::uint32_t depth = 0;
    bool native_input = false;
    bool chroma420 = false;
    std::uint32_t total = 0;
    StagingSources staging{};
    std::size_t readback = 0;

    explicit Layout(const VmafV1Config& config) {
        base_size = config.size;
        const auto source_pixels = CheckedPixels(base_size);
        const auto& model = VmafV1Parameters(config.model);
        const bool hdr_model = config.model == VmafV1Model::Hdr4K;
        if (hdr_model && (config.fit != VmafFitMode::Native ||
                          config.chroma_sampling != VmafChromaSampling::Native))
            throw std::invalid_argument(
                "vmaf v1 HDR: model controls scaling and chroma subsampling"
            );
        const auto geometry =
            hdr_model ? VmafFrameGeometry{VmafHdrGeometry(
                                              base_size, model.canvas.height
                                          ),
                                          {}}
                      : PlanVmafGeometry(base_size, model.canvas, config.fit);
        feature_size = geometry.canvas;
        const auto size = feature_size;
        const auto pixels = CheckedPixels(size);
        const auto& fmt = config.input_format;
        if (config.chroma_sampling != VmafChromaSampling::Native &&
            config.chroma_sampling != VmafChromaSampling::Yuv420 &&
            config.chroma_sampling != VmafChromaSampling::Yuv444)
            throw std::invalid_argument("vmaf v1: invalid chroma sampling");
        const bool native_mode =
            !hdr_model && config.chroma_sampling == VmafChromaSampling::Native;
        if (native_mode) {
            ValidateVmafSdrFormat(fmt);
            ValidateImage(fmt, base_size, "vmaf v1 input format");
        }
        if (!std::isfinite(config.background) || config.background < 0.0f ||
            config.background > 1.0f)
            throw std::invalid_argument("vmaf v1: invalid alpha background");
        if (config.encoded_bit_depth > 16 ||
            (config.encoded_bit_depth > 0 && config.encoded_bit_depth < 6))
            throw std::invalid_argument("vmaf v1: invalid encoded bit depth");
        if (native_mode &&
            (fmt.subsample_w > 2 || fmt.subsample_h > 2 ||
             base_size.width % (std::size_t{1} << fmt.subsample_w) != 0 ||
             base_size.height % (std::size_t{1} << fmt.subsample_h) != 0))
            throw std::invalid_argument("vmaf v1: invalid subsampling geometry");
        if (config.gpu_resident_images > kResidentIndexMask)
            throw std::invalid_argument("vmaf v1: too many resident images");
        const bool native_yuv = native_mode && fmt.family == ColorFamily::Yuv &&
                                IsVmafNativeInput(fmt);
        std::uint8_t subsample_w = 1;
        std::uint8_t subsample_h = 1;
        if (config.chroma_sampling == VmafChromaSampling::Yuv444) {
            subsample_w = 0;
            subsample_h = 0;
        } else if (config.chroma_sampling == VmafChromaSampling::Native &&
                   native_yuv) {
            subsample_w = fmt.subsample_w;
            subsample_h = fmt.subsample_h;
        }
        native_input = native_yuv;
        chroma420 = subsample_w == 1 && subsample_h == 1;
        depth = native_input ? VmafLumaDepth(fmt) : 16;
        const auto source_chroma =
            SubsampledSize(base_size, subsample_w, subsample_h);
        chroma_size = SubsampledSize(size, subsample_w, subsample_h);
        if (!hdr_model && base_size != size) {
            fit.emplace();
            fit->luma = {base_size, size, geometry.fit};
            // Project the luma rectangle onto the coded chroma grid. Keep the
            // final chroma sample when a fitted edge has an odd luma extent.
            fit->chroma = {
                source_chroma,
                chroma_size,
                {SubsampledSize(geometry.fit.content, subsample_w, subsample_h),
                 geometry.fit.x >> subsample_w, geometry.fit.y >> subsample_h}};
        }
        // Match the pinned SpEED extractor's nearest-integer dimensions.
        speed_size = {
            static_cast<std::size_t>(
                static_cast<double>(chroma_size.width) * model.chroma_prescale +
                0.5
            ),
            static_cast<std::size_t>(
                static_cast<double>(chroma_size.height) * model.chroma_prescale +
                0.5
            )};
        const std::size_t speed_pixels = speed_size.width * speed_size.height;
        const std::size_t speed_scratch =
            VmafSpeedFilter::ScratchWords(speed_size);
        const std::size_t speed_output =
            VmafSpeedFilter::OutputPixels(speed_size);
        const Size filtered{speed_size.width / 16, speed_size.height / 16};
        const auto speed_partials = VmafSpeedStats::PartialCount(filtered);
        cambi_plan = VmafCambi::Plan(size, hdr_model);
        std::uint64_t cursor = 0;
        auto add = [&](std::uint64_t count) {
            if (count > std::numeric_limits<std::uint32_t>::max() - cursor)
                throw std::invalid_argument(
                    "vmaf v1: geometry exceeds arena addressing"
                );
            const auto at = static_cast<std::uint32_t>(cursor);
            cursor += count;
            return at;
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
        if (hdr_model) {
            hdr.emplace();
            hdr->source = add(source_pixels);
            hdr->horizontal =
                add(std::max(base_size.width, size.width) * base_size.height);
            const auto working_pixels = std::min(base_size.width, size.width) *
                                        std::min(base_size.height, size.height);
            for (auto& plane : hdr->working) plane = add(working_pixels);
        }
        if (fit) {
            const auto chroma_pixels =
                source_chroma.width * source_chroma.height;
            for (auto& image : fit->source)
                image = {add(source_pixels), add(chroma_pixels),
                         add(chroma_pixels)};
            fit->scratch = add(
                std::max(fit->luma.ScratchWords(), fit->chroma.ScratchWords())
            );
        }
        adm.reference[0] = add(pixels);
        adm.distorted[0] = add(pixels);
        Size level = size;
        for (std::size_t i = 1; i < 5; ++i) {
            level = VmafAdmWavelet::OutputSize(level);
            const std::size_t count = level.width * level.height;
            adm.reference[i] = add(count);
            adm.distorted[i] = add(count);
        }
        const auto bands = 3 * VmafAdm::BandPixels(size);
        adm.reference_details = add(bands);
        adm.restored = add(bands);
        adm.additive = add(bands);
        adm.partials = add(9 * VmafAdm::PartialCount(size));
        // These allocations must follow ReadbackLayout's contiguous order.
        adm.results = add(ReadbackLayout::kAdmWords);
        speed.result = add(ReadbackLayout::kSpeedWords);
        cambi.result = add(1);
        motion.score = add(1);
        for (std::size_t i = 0; i < 2; ++i) {
            speed.reference[i] = add(speed_pixels);
            speed.distorted[i] = add(speed_pixels);
        }
        chroma_source = {speed.reference[0], speed.reference[1],
                         speed.distorted[0], speed.distorted[1]};
        if (speed_size != chroma_size) {
            for (auto& plane : chroma_source)
                plane = add(chroma_size.width * chroma_size.height);
        }
        speed.filter_scratch = add(4 * speed_scratch);
        speed.down = add(4 * speed_output);
        speed.reference_filtered = add(2 * speed_output);
        speed.distorted_filtered = add(2 * speed_output);
        speed.stats_scratch = add(2 * VmafSpeedStats::ScratchWords());
        speed.partials = add(2 * speed_partials);
        speed.channel_results = add(4);
        cambi.input = adm.distorted[0];
        cambi.prepared = add(pixels);
        cambi.derivative = add(pixels);
        cambi.mask = add(pixels);
        const auto scale_pixels = cambi_plan.max_scale_pixels;
        cambi.scale_input = add(scale_pixels);
        cambi.scale_mask[0] = add(scale_pixels);
        cambi.scale_mask[1] = add(scale_pixels);
        cambi.contrast_horizontal = add(scale_pixels);
        cambi.filtered = add(scale_pixels);
        cambi.contrast = add(cambi_plan.total_scale_pixels);
        cambi.pool_histogram = add(5 * 256);
        cambi.pool_state = add(5 * 2);
        cambi.pool_partials = add(cambi_plan.total_partials);
        cambi.scale_scores = add(5);
        motion.luma = adm.reference[0];
        motion.previous = add(pixels);
        motion.blurred = add(pixels);
        if (config.hfr) older_motion = add(pixels);
        motion.partials = add(VmafMotion::PartialCount(size));
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
            base_size, feature_size, speed_size, cambi_plan, context.limits,
            hdr.has_value()
        );
        const auto arena_bytes = Bytes(total);
        if (arena_bytes > context.limits.maxStorageBufferRange) {
            throw std::invalid_argument(
                "vmaf v1: image exceeds device storage buffer range"
            );
        }
        return {arena_bytes,
                Bytes(readback + ReadbackLayout::kWords),
                {},
                {{Bytes(motion.previous),
                  Bytes(
                      (older_motion != 0 ? 3u : 2u) * feature_size.width *
                      feature_size.height
                  ),
                  false}}};
    }
};
std::optional<::vqm::VmafV1Frame> PublicFrame(
    const std::optional<internal::VmafV1Frame>& frame
) {
    if (!frame) return std::nullopt;
    return ::vqm::VmafV1Frame{frame->index,
                              frame->features.cambi,
                              frame->features.speed_chroma_uv,
                              frame->features.adm3,
                              frame->features.motion3,
                              frame->score};
}
}  // namespace

namespace internal {
BufferMemoryPlan PlanVmafV1Memory(
    const VmafV1Config& config, const VulkanContext& context
) {
    return Layout(config).MemoryPlan(context);
}
}  // namespace internal

struct VmafV1Pipeline::Impl {
    explicit Impl(
        VmafV1Config pipeline_config, const VulkanDevice* device,
        std::shared_ptr<SharedMetricStorage> shared = {}
    )
        : config(pipeline_config),
          layout(pipeline_config),
          context(
              shared ? shared->backing().shared_context()
                     : RequireVulkanContext(device)
          ),
          runtime(context, RuntimeConfig(), std::move(shared)),
          sequence(config.clip, config.model, config.hfr) {
        const auto memory = layout.MemoryPlan(*context);
        runtime.CreateArena(memory.arena_bytes);
        if (!runtime.arena_visible())
            runtime.CreateStaging(
                static_cast<std::size_t>(memory.fallback_staging_bytes)
            );
        runtime.CreateCommandObjects();
        runtime.CreateDescriptors(0, memory.arena_bytes);
        luma.emplace(runtime);
        if (layout.hdr)
            hdr.emplace(runtime);
        else if (!layout.native_input)
            sdr.emplace(runtime);
        if (layout.fit) resizer.emplace(runtime);
        adm.emplace(runtime);
        if (layout.speed_size != layout.chroma_size)
            speed_prescaler.emplace(runtime);
        speed.emplace(runtime);
        cambi.emplace(runtime, layout.hdr.has_value());
        resident.Resize(config.gpu_resident_images);
        runtime.InitializeSharedState();
        motion.emplace(runtime, runtime.BindPersistentPlanes());
    }
    VulkanRuntime::Config RuntimeConfig() {
        VulkanRuntime::Config out;
        out.stream_region = layout.source;
        out.stream_desc = layout.desc;
        out.resident_slots = layout.resident;
        const auto pixels = config.size.width * config.size.height;
        out.staging_desc_offset = layout.staging.reserved_floats;
        out.staging_image_bytes = layout.staging.image_bytes;
        out.base_pixels = pixels;
        out.base_size = config.size;
        out.streaming = config.streaming_inputs;
        out.memory_policy = config.memory_policy;
        out.persistent_ranges = layout.MemoryPlan(*context).persistent_ranges;
        out.push_constant_size = RequiredPushBytes();
        return out;
    }
    void RequireOpen() const {
        if (finished)
            throw std::logic_error("vmaf v1: sequence finished; call Reset");
    }
    void ValidateFormat(const ImageFormat& format) const {
        if (layout.hdr) {
            ValidateVmafHdrFormat(format);
            ValidateImage(format, config.size, "vmaf v1 HDR input format");
            return;
        }
        ValidateVmafSdrFormat(format);
        if (config.chroma_sampling == VmafChromaSampling::Native &&
            !SameImageFormat(format, config.input_format))
            throw std::invalid_argument(
                "vmaf v1: input format differs from sequence format"
            );
    }
    void RequireStreaming() const {
        if (!config.streaming_inputs)
            throw std::invalid_argument("vmaf v1: streaming input is disabled");
    }
    UploadedImage Upload(const Image& image) {
        RequireOpen();
        ValidateFormat(image.format);
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
    void RecordHdrPlanes(
        const std::array<const ImageFormat*, 2>& formats,
        const FramePlanes& planes
    ) {
        for (std::size_t image = 0; image < planes.size(); ++image) {
            const auto& format = *formats[image];
            const bool canonical =
                format.matrix == MatrixCoefficients::Bt2100ICtCp &&
                format.range == PixelRange::Limited && format.subsample_w == 1 &&
                format.subsample_h == 1 &&
                format.chroma_location == ChromaLocation::Left &&
                config.size == layout.feature_size;
            if (canonical) {
                for (std::uint32_t plane = 0; plane < 3; ++plane)
                    luma->RecordPlane(
                        image, layout.desc[image], planes[image][plane],
                        config.size, format, plane
                    );
            } else {
                hdr->Record(
                    image, layout.desc[image], format, config.size,
                    layout.feature_size, *layout.hdr, planes[image]
                );
            }
        }
    }
    void RecordInputPlanes(
        const std::array<const ImageFormat*, 2>& formats,
        const FramePlanes& planes
    ) {
        if (hdr) {
            RecordHdrPlanes(formats, planes);
            return;
        }
        if (layout.native_input) {
            for (std::uint32_t plane = 0; plane < 3; ++plane)
                for (std::size_t image = 0; image < planes.size(); ++image)
                    luma->RecordPlane(
                        image, layout.desc[image], planes[image][plane],
                        config.size, *formats[image], plane
                    );
            return;
        }
        for (std::size_t image = 0; image < planes.size(); ++image) {
            const auto& [y, cb, cr] = planes[image];
            sdr->Record(
                image, layout.desc[image], y, cb, cr, config.size,
                *formats[image], layout.chroma420, true, config.background
            );
        }
    }
    void RecordPlaneFit(
        std::uint32_t source, std::uint32_t destination, const PlaneFit& plane,
        float pad
    ) {
        const auto horizontal = plane.HorizontalSize();
        if (horizontal.width != plane.source.width) {
            const auto scratch = layout.fit->scratch;
            resizer->RecordHorizontal(source, scratch, plane.source, horizontal);
            runtime.ComputeBarrier();
            source = scratch;
        }
        resizer->RecordVertical(
            source, destination, horizontal, plane.geometry, plane.canvas, pad,
            true
        );
        // The same scratch is reused for every plane.
        runtime.ComputeBarrier();
    }
    void RecordFit(const FramePlanes& output) {
        const auto& fit = *layout.fit;
        const float black =
            layout.native_input && config.input_format.range == PixelRange::Full
                ? -128.0f
                : -112.0f;
        runtime.ComputeBarrier();
        for (std::size_t image = 0; image < output.size(); ++image) {
            RecordPlaneFit(
                fit.source[image][0], output[image][0], fit.luma, black
            );
            for (std::size_t plane = 1; plane < 3; ++plane)
                RecordPlaneFit(
                    fit.source[image][plane], output[image][plane], fit.chroma, 0
                );
        }
    }
    std::optional<::vqm::VmafV1Frame> Run(
        const std::array<VulkanRuntime::Source, 2>& sources,
        const VulkanSubmitSync* sync = nullptr
    ) {
        RequireOpen();
        if (sync) VulkanRuntime::ValidateSubmitSync(*sync);
        runtime.PrepareSourceBindings(sources);
        const auto token = runtime.BeginSubmission();
        runtime.BeginCommandBuffer();
        const auto formats = runtime.PublishSources(sources);
#ifdef VQM_STAGE_TIMING
        runtime.OpenStage(Stage::VmafV1Luma);
#endif
        const FramePlanes feature_planes{
            std::array{layout.adm.reference[0], layout.chroma_source[0],
                       layout.chroma_source[1]},
            std::array{layout.adm.distorted[0], layout.chroma_source[2],
                       layout.chroma_source[3]}};
        RecordInputPlanes(
            formats, layout.fit ? layout.fit->source : feature_planes
        );
        if (layout.fit) RecordFit(feature_planes);
#ifdef VQM_STAGE_TIMING
        runtime.CloseStage();
        runtime.OpenStage(Stage::VmafV1Adm);
#endif
        adm->Record(layout.feature_size, layout.adm, config.model);
#ifdef VQM_STAGE_TIMING
        runtime.CloseStage();
        runtime.OpenStage(Stage::VmafV1Speed);
#endif
        if (speed_prescaler)
            speed_prescaler->Record(
                layout.chroma_size, layout.speed_size, layout.chroma_source,
                layout.speed
            );
        speed->Record(layout.speed_size, layout.speed);
#ifdef VQM_STAGE_TIMING
        runtime.CloseStage();
        runtime.OpenStage(Stage::VmafV1Cambi);
#endif
        const auto encoded_depth = config.encoded_bit_depth != 0
                                       ? config.encoded_bit_depth
                                       : layout.depth;
        const auto display_depth = layout.fit ? 16u : layout.depth;
        cambi->Record(
            layout.feature_size, display_depth, encoded_depth, layout.cambi
        );
#ifdef VQM_STAGE_TIMING
        runtime.CloseStage();
        runtime.OpenStage(Stage::VmafV1Motion);
#endif
        motion->Record(layout.feature_size, layout.motion, HasMotionHistory());
#ifdef VQM_STAGE_TIMING
        runtime.CloseStage();
#endif
        if (runtime.arena_visible())
            runtime.ComputeToHostBarrier();
        else
            runtime.RecordReadback(
                layout.adm.results, layout.readback, ReadbackLayout::kWords
            );
        runtime.EndCommandBuffer();
        if (sync)
            runtime.QueueSubmit(*sync);
        else
            runtime.QueueSubmit();
        runtime.FinishSubmission(token);
        const float* values = runtime.arena_visible()
                                  ? runtime.arena_map() + layout.adm.results
                                  : runtime.staging_map() + layout.readback;
        const float speed_status = values[ReadbackLayout::kSpeedStatus];
        if (speed_status >= 2.0f || !std::isfinite(speed_status))
            throw std::runtime_error(
                "vmaf v1: SpEED covariance eigensystem failed"
            );
        const VmafV1Spatial spatial{values[ReadbackLayout::kCambi],
                                    values[ReadbackLayout::kSpeedScore],
                                    values[ReadbackLayout::kAdmScore]};
        const double transition = values[ReadbackLayout::kMotion];
        if (!std::isfinite(spatial.cambi) ||
            !std::isfinite(spatial.speed_chroma_uv) ||
            !std::isfinite(spatial.adm3) || !std::isfinite(transition) ||
            transition < 0.0 || (!HasMotionHistory() && transition != 0.0)) {
            throw std::runtime_error("vmaf v1: invalid GPU feature result");
        }
        VQM_TIMED(VmafV1Score);
        const auto frame = sequence.Push(spatial, transition);
        std::swap(layout.motion.previous, layout.motion.blurred);
        if (config.hfr) std::swap(layout.motion.previous, layout.older_motion);
        if (!HasMotionHistory()) ++motion_history;
        return PublicFrame(frame);
    }
    [[nodiscard]] bool HasMotionHistory() const {
        return motion_history >= (config.hfr ? 2u : 1u);
    }

    VmafV1Config config;
    Layout layout;
    std::shared_ptr<VulkanContext> context;
    VulkanRuntime runtime;
    std::optional<VmafLumaConverter> luma;
    std::optional<VmafSdrConverter> sdr;
    std::optional<VmafHdrConverter> hdr;
    std::optional<VmafResizer> resizer;
    std::optional<VmafAdm3> adm;
    std::optional<VmafSpeedPrescaler> speed_prescaler;
    std::optional<VmafSpeed> speed;
    std::optional<VmafCambi> cambi;
    std::optional<VmafMotion> motion;
    ResidentTable resident;
    VmafV1Sequence sequence;
    unsigned motion_history = 0;
    bool finished = false;
};
VmafV1Pipeline::VmafV1Pipeline(VmafV1Config config)
    : impl_(std::make_unique<Impl>(config, nullptr)) {}
VmafV1Pipeline::VmafV1Pipeline(VmafV1Config config, const VulkanDevice& device)
    : impl_(std::make_unique<Impl>(config, &device)) {}
VmafV1Pipeline::~VmafV1Pipeline() = default;
VmafV1Pipeline::VmafV1Pipeline(
    VmafV1Config config, const VulkanDevice* device,
    std::shared_ptr<internal::SharedMetricStorage> scratch
)
    : impl_(std::make_unique<Impl>(config, device, std::move(scratch))) {}
VmafV1Pipeline::VmafV1Pipeline(VmafV1Pipeline&&) noexcept = default;
VmafV1Pipeline& VmafV1Pipeline::operator=(VmafV1Pipeline&&) noexcept = default;
PipelineMemoryUsage VmafV1Pipeline::GetMemoryUsage() const {
    return impl_->runtime.GetMemoryUsage();
}
UploadedImage VmafV1Pipeline::Upload(const Image& image) {
    return impl_->Upload(image);
}
void VmafV1Pipeline::Release(UploadedImage image) { impl_->Release(image); }
std::optional<::vqm::VmafV1Frame> VmafV1Pipeline::Push(
    const Image& a, const Image& b
) {
    impl_->RequireOpen();
    impl_->RequireStreaming();
    impl_->ValidateFormat(a.format);
    impl_->ValidateFormat(b.format);
    impl_->runtime.UploadStreaming(a, b);
    return impl_->Run({});
}
std::optional<::vqm::VmafV1Frame> VmafV1Pipeline::Push(
    const ImageView& a, const ImageView& b
) {
    impl_->RequireOpen();
    impl_->RequireStreaming();
    impl_->ValidateFormat(a.format);
    impl_->ValidateFormat(b.format);
    impl_->runtime.UploadStreaming(a, b);
    return impl_->Run({});
}
std::optional<::vqm::VmafV1Frame> VmafV1Pipeline::Push(
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
std::optional<::vqm::VmafV1Frame> VmafV1Pipeline::Push(
    const VulkanBufferView& a, const VulkanBufferView& b
) {
    impl_->RequireOpen();
    impl_->ValidateFormat(a.format);
    impl_->ValidateFormat(b.format);
    return impl_->Run(impl_->runtime.ResolveExternalInputs(a, b));
}
std::optional<::vqm::VmafV1Frame> VmafV1Pipeline::Push(
    const VulkanBufferView& a, const VulkanBufferView& b,
    const VulkanSubmitSync& sync
) {
    impl_->RequireOpen();
    impl_->ValidateFormat(a.format);
    impl_->ValidateFormat(b.format);
    return impl_->Run(impl_->runtime.ResolveExternalInputs(a, b), &sync);
}
std::optional<::vqm::VmafV1Frame> VmafV1Pipeline::Flush() {
    VQM_TIMED(VmafV1Score);
    auto result = impl_->sequence.Flush();
    impl_->finished = true;
    return PublicFrame(result);
}
std::optional<double> VmafV1Pipeline::PooledScore() const {
    return impl_->sequence.PooledScore();
}
void VmafV1Pipeline::Reset() {
    impl_->sequence.Reset();
    impl_->motion_history = 0;
    impl_->finished = false;
}
}  // namespace vqm
