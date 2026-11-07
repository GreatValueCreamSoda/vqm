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
#include "metrics/vmaf/vmaf_memory.h"
#include "metrics/vmaf/speed/vmaf_speed.h"
#include "metrics/vmaf/v1/vmaf_v1_internal.h"
#include "vqm_views.h"
#include "vqm_vulkan.h"
#include "vulkan/vulkan_runtime.h"
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
    if (frames_ == 0) {
        if (transition != 0.0)
            throw std::invalid_argument(
                "vmaf motion3: first transition must be zero"
            );
        frames_ = 1;
        return std::nullopt;
    }
    if (frames_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("vmaf motion3: too many frames");
    const auto current = std::min(transition, 18.0);
    const VmafMotionFrame result{
        frames_ - 1,
        frames_ == 1 ? current : std::min(previous_transition_, current)};
    previous_transition_ = current;
    ++frames_;
    return result;
}
std::optional<VmafMotionFrame> VmafV1MotionTimeline::Flush() {
    if (finished_) return std::nullopt;
    finished_ = true;
    if (frames_ == 0) return std::nullopt;
    return VmafMotionFrame{frames_ - 1,
                           frames_ == 1 ? 0.0 : previous_transition_};
}
void VmafV1MotionTimeline::Reset() noexcept {
    frames_ = 0;
    previous_transition_ = 0.0;
    finished_ = false;
}

}  // namespace vqm::internal

// VMAF: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

namespace vqm::internal {

VmafV1Frame VmafV1Sequence::FinishFrame(
    std::uint64_t index, const VmafV1Spatial& spatial, double motion
) {
    const VmafV1Features features{spatial.cambi, spatial.speed_chroma_uv,
                                  spatial.adm3, motion};
    const double score = VmafV1Predict(features, clip_);
    score_sum_ += score;
    ++emitted_;
    return {index, features, score};
}
std::optional<VmafV1Frame> VmafV1Sequence::Push(
    const VmafV1Spatial& spatial, double transition
) {
    if (finished_) throw std::logic_error("vmaf v1: sequence is finished");
    if (!std::isfinite(spatial.cambi) ||
        !std::isfinite(spatial.speed_chroma_uv) || !std::isfinite(spatial.adm3))
        throw std::invalid_argument("vmaf v1: spatial features must be finite");
    const auto ready = motion_.Push(transition);
    if (!ready) {
        pending_ = spatial;
        return std::nullopt;
    }
    if (!pending_) throw std::logic_error("vmaf v1: missing pending frame");
    const auto result = FinishFrame(ready->index, *pending_, ready->motion2);
    pending_ = spatial;
    return result;
}
std::optional<VmafV1Frame> VmafV1Sequence::Flush() {
    if (finished_) return std::nullopt;
    const auto ready = motion_.Flush();
    finished_ = true;
    if (!ready) return std::nullopt;
    if (!pending_) throw std::logic_error("vmaf v1: missing pending frame");
    const auto result = FinishFrame(ready->index, *pending_, ready->motion2);
    pending_.reset();
    return result;
}
std::optional<double> VmafV1Sequence::PooledScore() const {
    if (!finished_)
        throw std::logic_error("vmaf v1: flush sequence before pooling");
    if (emitted_ == 0) return std::nullopt;
    return score_sum_ / static_cast<double>(emitted_);
}
void VmafV1Sequence::Reset() noexcept {
    motion_.Reset();
    pending_.reset();
    score_sum_ = 0;
    emitted_ = 0;
    finished_ = false;
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
         VmafMotion::PushBytes(), VmafLumaConverter::PushBytes(),
         VmafSdrConverter::PushBytes()}
    );
}

void ValidateDispatchLimits(
    Size size, Size chroma_size, const VmafCambiPlan& cambi,
    const VkPhysicalDeviceLimits& limits
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
        std::max<std::size_t>({Groups(size.width, 16), 2500, speed_partials,
                               cambi_partials});
    const auto groups_y =
        std::max<std::uint32_t>({Groups(size.height, 8),
                                 Groups(cambi.max_scale_size.height, 8), 5});
    // The 256x1 and 16x16 shaders are always used. The optional 512-wide
    // SpEED eigen shader is selected only when the device supports it.
    // CAMBI contrast's fixed shared arrays occupy 16,004 bytes; see its
    // shader declaration. No other unconditional VMAF v1 shader uses more.
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

struct Layout {
    std::array<std::uint32_t, 2> source{}, desc{};
    std::vector<std::uint32_t> resident;
    VmafAdmBuffers adm{};
    VmafSpeedBuffers speed{};
    VmafCambiBuffers cambi{};
    VmafMotionBuffers motion{};
    Size base_size{};
    Size chroma_size{};
    VmafCambiPlan cambi_plan{};
    std::uint32_t depth = 0;
    bool native_input = false;
    bool chroma420 = false;
    std::uint32_t total = 0;
    StagingSources staging{};
    std::size_t readback = 0;

    explicit Layout(const VmafV1Config& config) {
        const auto size = config.size;
        base_size = size;
        const auto& fmt = config.input_format;
        if (config.chroma_sampling != VmafChromaSampling::Native &&
            config.chroma_sampling != VmafChromaSampling::Yuv420 &&
            config.chroma_sampling != VmafChromaSampling::Yuv444)
            throw std::invalid_argument("vmaf v1: invalid chroma sampling");
        const bool native_mode =
            config.chroma_sampling == VmafChromaSampling::Native;
        if (native_mode) {
            ValidateVmafSdrFormat(fmt);
            ValidateImage(fmt, size, "vmaf v1 input format");
        }
        if (!std::isfinite(config.background) || config.background < 0.0f ||
            config.background > 1.0f)
            throw std::invalid_argument("vmaf v1: invalid alpha background");
        if (config.encoded_bit_depth > 16 ||
            (config.encoded_bit_depth > 0 && config.encoded_bit_depth < 6))
            throw std::invalid_argument("vmaf v1: invalid encoded bit depth");
        if (size.width < 17 || size.height < 17 ||
            size.width > std::numeric_limits<std::uint32_t>::max() / size.height)
            throw std::invalid_argument("vmaf v1: invalid image geometry");
        if (native_mode &&
            (fmt.subsample_w > 2 || fmt.subsample_h > 2 ||
             size.width % (std::size_t{1} << fmt.subsample_w) != 0 ||
             size.height % (std::size_t{1} << fmt.subsample_h) != 0))
            throw std::invalid_argument("vmaf v1: invalid subsampling geometry");
        if (config.gpu_resident_images > kResidentIndexMask)
            throw std::invalid_argument("vmaf v1: too many resident images");
        const std::size_t pixels = size.width * size.height;
        if (pixels > std::numeric_limits<std::uint32_t>::max() / 64u)
            throw std::invalid_argument(
                "vmaf v1: geometry exceeds arena addressing"
            );
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
        // Generic 4:2:0 includes the last chroma sample at odd edges.
        chroma_size =
            !native_input && chroma420
                ? Size{size.width / 2 + size.width % 2,
                       size.height / 2 + size.height % 2}
                : Size{size.width >> subsample_w, size.height >> subsample_h};
        const std::size_t chroma = chroma_size.width * chroma_size.height;
        const std::size_t speed_scratch =
            VmafSpeedFilter::ScratchWords(chroma_size);
        const std::size_t speed_output =
            VmafSpeedFilter::OutputPixels(chroma_size);
        const Size filtered{chroma_size.width / 16, chroma_size.height / 16};
        const auto speed_partials = VmafSpeedStats::PartialCount(filtered);
        cambi_plan = VmafCambi::Plan(size);
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
            source[0] = add(4 * pixels);
            source[1] = add(4 * pixels);
        }
        desc[0] = add(kSourceDescWords);
        desc[1] = add(kSourceDescWords);
        resident.reserve(config.gpu_resident_images);
        for (std::size_t i = 0; i < config.gpu_resident_images; ++i)
            resident.push_back(add(4 * pixels));
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
            speed.reference[i] = add(chroma);
            speed.distorted[i] = add(chroma);
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
        motion.partials = add(VmafMotion::PartialCount(size));
        total = static_cast<std::uint32_t>(cursor);
        staging = PlanStagingSources(
            pixels, config.streaming_inputs, config.gpu_resident_images
        );
        readback = staging.reserved_floats + 2 * kSourceDescWords;
    }

    [[nodiscard]] BufferMemoryPlan MemoryPlan(
        const VulkanContext& context
    ) const {
        ValidateDispatchLimits(
            base_size, chroma_size, cambi_plan, context.limits
        );
        const auto arena_bytes = Bytes(total);
        if (arena_bytes > context.limits.maxStorageBufferRange) {
            throw std::invalid_argument(
                "vmaf v1: image exceeds device storage buffer range"
            );
        }
        return {arena_bytes, Bytes(readback + ReadbackLayout::kWords)};
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
    explicit Impl(VmafV1Config pipeline_config, const VulkanDevice* device)
        : config(pipeline_config),
          layout(pipeline_config),
          context(RequireVulkanContext(device)),
          runtime(context, RuntimeConfig()),
          sequence(config.clip) {
        const auto memory = layout.MemoryPlan(*context);
        runtime.CreateArena(memory.arena_bytes);
        if (!runtime.arena_visible())
            runtime.CreateStaging(
                static_cast<std::size_t>(memory.fallback_staging_bytes)
            );
        runtime.CreateCommandObjects();
        runtime.CreateDescriptors(0, memory.arena_bytes);
        luma.emplace(runtime);
        if (!layout.native_input) sdr.emplace(runtime);
        adm.emplace(runtime);
        speed.emplace(runtime);
        cambi.emplace(runtime);
        motion.emplace(runtime);
        resident.Resize(config.gpu_resident_images);
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
        out.push_constant_size = RequiredPushBytes();
        return out;
    }
    void RequireOpen() const {
        if (finished)
            throw std::logic_error("vmaf v1: sequence finished; call Reset");
    }
    void ValidateFormat(const ImageFormat& format) const {
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
        if (layout.native_input) {
            luma->Record(
                0, layout.desc[0], layout.adm.reference[0], config.size,
                *formats[0]
            );
            luma->Record(
                1, layout.desc[1], layout.adm.distorted[0], config.size,
                *formats[1]
            );
            for (std::uint32_t plane = 1; plane <= 2; ++plane) {
                luma->RecordPlane(
                    0, layout.desc[0], layout.speed.reference[plane - 1],
                    config.size, *formats[0], plane
                );
                luma->RecordPlane(
                    1, layout.desc[1], layout.speed.distorted[plane - 1],
                    config.size, *formats[1], plane
                );
            }
        } else {
            sdr->Record(
                0, layout.desc[0], layout.adm.reference[0],
                layout.speed.reference[0], layout.speed.reference[1],
                config.size, *formats[0], layout.chroma420, true,
                config.background
            );
            sdr->Record(
                1, layout.desc[1], layout.adm.distorted[0],
                layout.speed.distorted[0], layout.speed.distorted[1],
                config.size, *formats[1], layout.chroma420, true,
                config.background
            );
        }
#ifdef VQM_STAGE_TIMING
        runtime.CloseStage();
        runtime.OpenStage(Stage::VmafV1Adm);
#endif
        adm->Record(config.size, layout.adm);
#ifdef VQM_STAGE_TIMING
        runtime.CloseStage();
        runtime.OpenStage(Stage::VmafV1Speed);
#endif
        speed->Record(layout.chroma_size, layout.speed);
#ifdef VQM_STAGE_TIMING
        runtime.CloseStage();
        runtime.OpenStage(Stage::VmafV1Cambi);
#endif
        cambi->Record(
            config.size, layout.depth, config.encoded_bit_depth, layout.cambi
        );
#ifdef VQM_STAGE_TIMING
        runtime.CloseStage();
        runtime.OpenStage(Stage::VmafV1Motion);
#endif
        motion->Record(config.size, layout.motion, previous);
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
            transition < 0.0 || (!previous && transition != 0.0)) {
            throw std::runtime_error("vmaf v1: invalid GPU feature result");
        }
        std::swap(layout.motion.previous, layout.motion.blurred);
        previous = true;
        VQM_TIMED(VmafV1Score);
        return PublicFrame(sequence.Push(spatial, transition));
    }
    VmafV1Config config;
    Layout layout;
    std::shared_ptr<VulkanContext> context;
    VulkanRuntime runtime;
    std::optional<VmafLumaConverter> luma;
    std::optional<VmafSdrConverter> sdr;
    std::optional<VmafAdm3> adm;
    std::optional<VmafSpeed> speed;
    std::optional<VmafCambi> cambi;
    std::optional<VmafMotion> motion;
    ResidentTable resident;
    VmafV1Sequence sequence;
    bool previous = false;
    bool finished = false;
};
VmafV1Pipeline::VmafV1Pipeline(VmafV1Config config)
    : impl_(std::make_unique<Impl>(config, nullptr)) {}
VmafV1Pipeline::VmafV1Pipeline(VmafV1Config config, const VulkanDevice& device)
    : impl_(std::make_unique<Impl>(config, &device)) {}
VmafV1Pipeline::~VmafV1Pipeline() = default;
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
    impl_->previous = false;
    impl_->finished = false;
}
}  // namespace vqm
