// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// VMAF input scaling: Copyright (c) Netflix, Inc.
// VMAF motion: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

#include "vmaf_preprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include "vulkan/vulkan_runtime.h"
#include "vulkan/vulkan_support.h"

namespace vqm::internal {

// Native coded-plane conversion

namespace {

struct VmafLumaPush {
    std::uint32_t desc_base;
    std::uint32_t output;
    std::uint32_t width;
    std::uint32_t height;
    float scale;
    std::uint32_t plane;
    std::uint32_t step;
};
static_assert(sizeof(VmafLumaPush) == VmafLumaConverter::PushBytes());

// Storage support is shared by native-input detection and coded-plane
// validation. Their callers separately check the color family and alpha.
std::optional<std::uint32_t> SupportedCodedDepth(const ImageFormat& format) {
    const std::uint32_t depth =
        format.depth != 0 ? format.depth
                          : (format.type == PixelType::Byte ? 8u : 16u);
    if ((format.type == PixelType::Byte && depth == 8u) ||
        (format.type == PixelType::Word &&
         (depth == 10u || depth == 12u || depth == 16u)))
        return depth;
    return std::nullopt;
}

}  // namespace

std::uint32_t VmafLumaDepth(const ImageFormat& format) {
    if ((format.family != ColorFamily::Yuv &&
         format.family != ColorFamily::Grey) ||
        format.alpha != AlphaType::None) {
        throw std::invalid_argument(
            "vmaf: input must be opaque planar YUV or grey"
        );
    }
    if (const auto depth = SupportedCodedDepth(format)) return *depth;
    throw std::invalid_argument(
        "vmaf: expected 8-bit bytes or 10/12/16-bit words"
    );
}

VmafLumaConverter::VmafLumaConverter(VulkanRuntime& runtime)
    : runtime_(runtime), pipeline_(runtime.CreatePipeline("vmaf_luma")) {}

VmafLumaConverter::~VmafLumaConverter() {
    runtime_.DestroyPipelines(std::array{pipeline_});
}

void VmafLumaConverter::Record(
    std::size_t image, std::uint32_t desc_base, std::uint32_t output, Size size,
    const ImageFormat& format
) {
    RecordPlane(image, desc_base, output, size, format, 0);
}

Size VmafLumaConverter::PlaneSize(
    Size size, const ImageFormat& format, std::uint32_t plane
) {
    (void)VmafLumaDepth(format);
    if (plane > 2 || (plane != 0 && format.family != ColorFamily::Yuv) ||
        format.subsample_w > 2 || format.subsample_h > 2)
        throw std::invalid_argument("vmaf: invalid coded plane");
    if (plane != 0) {
        size.width >>= format.subsample_w;
        size.height >>= format.subsample_h;
    }
    if (size.width == 0 || size.height == 0)
        throw std::invalid_argument("vmaf: coded plane has zero size");
    return size;
}

void VmafLumaConverter::RecordPlane(
    std::size_t image, std::uint32_t desc_base, std::uint32_t output, Size size,
    const ImageFormat& format, std::uint32_t plane
) {
    const std::uint32_t depth = VmafLumaDepth(format);
    size = PlaneSize(size, format, plane);
    if (image >= 2 || size.width == 0 || size.height == 0 ||
        size.width > std::numeric_limits<std::uint32_t>::max() ||
        size.height > std::numeric_limits<std::uint32_t>::max() ||
        size.width > std::numeric_limits<std::uint32_t>::max() / size.height ||
        size.width * size.height >
            std::uint64_t{std::numeric_limits<std::uint32_t>::max()} - output +
                1u ||
        Groups(size.width, 16) >
            runtime_.context().limits.maxComputeWorkGroupCount[0] ||
        Groups(size.height, 16) >
            runtime_.context().limits.maxComputeWorkGroupCount[1]) {
        throw std::invalid_argument(
            "vmaf: invalid luma geometry or source index"
        );
    }
    // Netflix libvmaf's picture_copy.c, pinned 4611e47, lines 21-61:
    // integer sample / 2^(bitdepth - 8) - 128, with the division in float.
    // This preserves low-order 10/12/16-bit detail; the image range and RGB
    // transfer metadata are deliberately not applied to the coded Y' values.
    const VmafLumaPush push{desc_base,
                            output,
                            static_cast<std::uint32_t>(size.width),
                            static_cast<std::uint32_t>(size.height),
                            std::ldexp(1.0f, 8 - static_cast<int>(depth)),
                            plane,
                            depth == 8 ? 4u : 2u};
    runtime_.BindSource(image);
    runtime_.Dispatch(
        pipeline_, &push, sizeof(push), Groups(size.width, 16 * push.step),
        Groups(size.height, 16), 1
    );
}

// SDR color conversion

namespace {

struct VmafSdrPush {
    std::uint32_t desc_base;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t out_y;
    std::uint32_t out_u;
    std::uint32_t out_v;
    std::uint32_t flags;
    float background;
};
static_assert(sizeof(VmafSdrPush) == VmafSdrConverter::PushBytes());

std::string SdrShaderName(SourceDecodeVariant variant) {
    std::string name = "vmaf_sdr_c" + std::to_string(variant.chroma) + "_t" +
                       std::to_string(variant.transfer);
    if (variant.chroma != 0) name += "_f" + std::to_string(variant.filter);
    return name;
}

}  // namespace

void ValidateVmafSdrFormat(const ImageFormat& format) {
    if (format.transfer == TransferCharacteristics::St2084 ||
        format.transfer == TransferCharacteristics::AribB67)
        throw std::invalid_argument("vmaf: HDR transfer is not supported");
    (void)SelectSourceDecodeVariant(format, ChromaFilter::Bilinear);
}

bool IsVmafNativeInput(const ImageFormat& format) {
    if ((format.family != ColorFamily::Yuv &&
         format.family != ColorFamily::Grey) ||
        format.alpha != AlphaType::None)
        return false;
    return SupportedCodedDepth(format).has_value();
}

VmafSdrConverter::VmafSdrConverter(VulkanRuntime& runtime) : runtime_(runtime) {
    try {
        for (std::size_t i = 0; i < kSourceDecodeVariants.size(); ++i) {
            // VMAF currently uses bilinear source chroma reconstruction.
            if (kSourceDecodeVariants[i].filter == 0)
                pipelines_[i] = runtime_.CreatePipeline(
                    SdrShaderName(kSourceDecodeVariants[i])
                );
        }
    } catch (...) {
        runtime_.DestroyPipelines(pipelines_);
        throw;
    }
}

VmafSdrConverter::~VmafSdrConverter() { runtime_.DestroyPipelines(pipelines_); }

void VmafSdrConverter::Record(
    std::size_t image, std::uint32_t desc_base, std::uint32_t out_y,
    std::uint32_t out_u, std::uint32_t out_v, Size source_size,
    const ImageFormat& format, bool chroma420, bool emit_chroma, float background
) {
    if (image >= 2 || source_size.width == 0 || source_size.height == 0 ||
        source_size.width > std::numeric_limits<std::uint32_t>::max() ||
        source_size.height > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("vmaf: invalid SDR conversion geometry");
    }
    const auto blocks_x = source_size.width / 2u + source_size.width % 2u;
    const auto blocks_y = source_size.height / 2u + source_size.height % 2u;
    if (Groups(blocks_x, 16) >
            runtime_.context().limits.maxComputeWorkGroupCount[0] ||
        Groups(blocks_y, 16) >
            runtime_.context().limits.maxComputeWorkGroupCount[1])
        throw std::invalid_argument(
            "vmaf: SDR conversion exceeds device limits"
        );

    const std::uint32_t flags = (chroma420 ? 1u : 0u) | (emit_chroma ? 2u : 0u);
    const VmafSdrPush push{desc_base,
                           static_cast<std::uint32_t>(source_size.width),
                           static_cast<std::uint32_t>(source_size.height),
                           out_y,
                           out_u,
                           out_v,
                           flags,
                           background};
    const auto variant =
        SelectSourceDecodeVariant(format, ChromaFilter::Bilinear);
    runtime_.BindSource(image);
    runtime_.Dispatch(
        pipelines_[SourceDecodeVariantIndex(variant)], &push, sizeof(push),
        Groups(blocks_x, 16), Groups(blocks_y, 16), 1
    );
}

// Fit-and-pad resampling

namespace {

struct VmafResizePush {
    std::uint32_t source;
    std::uint32_t destination;
    std::uint32_t source_width;
    std::uint32_t source_height;
    std::uint32_t out_width;
    std::uint32_t out_height;
    std::uint32_t content_x;
    std::uint32_t content_y;
    std::uint32_t content_width;
    std::uint32_t content_height;
    std::uint32_t resample_filter;
    std::uint32_t vertical;
    float pad;
    std::uint32_t quantize_16bit;
};
static_assert(sizeof(VmafResizePush) == VmafResizer::PushBytes());

void CheckFitSize(Size size) {
    constexpr auto max_dimension = std::numeric_limits<std::uint32_t>::max();
    if (size.width == 0 || size.height == 0 || size.width > max_dimension ||
        size.height > max_dimension)
        throw std::invalid_argument("vmaf resize: invalid fit geometry");
}

void CheckResizePlane(std::uint32_t offset, Size size) {
    if (size.width == 0 || size.height == 0 ||
        size.width > std::numeric_limits<std::uint32_t>::max() ||
        size.height > std::numeric_limits<std::uint32_t>::max() ||
        size.width > std::numeric_limits<std::uint32_t>::max() / size.height)
        throw std::invalid_argument("vmaf resize: invalid plane geometry");
    const std::uint64_t pixels =
        static_cast<std::uint64_t>(size.width) * size.height;
    if (pixels >
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) -
            offset + 1u)
        throw std::invalid_argument("vmaf resize: plane range overflows");
}

}  // namespace

VmafFitGeometry FitVmafInto(Size source, Size target) {
    CheckFitSize(source);
    CheckFitSize(target);

    const std::uint64_t sw = source.width, sh = source.height;
    const std::uint64_t tw = target.width, th = target.height;
    std::uint64_t width = 0, height = 0;
    if (sw * th >= sh * tw) {
        width = tw;
        height = (sh * tw + sw / 2u) / sw;
    } else {
        height = th;
        width = (sw * th + sh / 2u) / sh;
    }
    width = std::clamp<std::uint64_t>(width, 1, tw);
    height = std::clamp<std::uint64_t>(height, 1, th);
    return {{static_cast<std::size_t>(width), static_cast<std::size_t>(height)},
            static_cast<std::uint32_t>((tw - width) / 2u),
            static_cast<std::uint32_t>((th - height) / 2u)};
}

VmafFrameGeometry PlanVmafGeometry(Size source, Size model, VmafFitMode mode) {
    switch (mode) {
        case VmafFitMode::Native:
            CheckFitSize(source);
            return {source, {source, 0, 0}};
        case VmafFitMode::Fit: {
            const Size content = FitVmafInto(source, model).content;
            return {content, {content, 0, 0}};
        }
        case VmafFitMode::FitAndPad:
            return {model, FitVmafInto(source, model)};
        case VmafFitMode::Pad: {
            CheckFitSize(source);
            const auto multiple = std::max(
                (static_cast<std::uint64_t>(source.width) + 15u) / 16u,
                (static_cast<std::uint64_t>(source.height) + 8u) / 9u
            );
            if (multiple > std::numeric_limits<std::uint32_t>::max() / 16u)
                throw std::invalid_argument(
                    "vmaf resize: 16:9 canvas overflows"
                );
            const Size canvas{static_cast<std::size_t>(multiple * 16u),
                              static_cast<std::size_t>(multiple * 9u)};
            return {
                canvas,
                {source,
                 static_cast<std::uint32_t>((canvas.width - source.width) / 2),
                 static_cast<std::uint32_t>(
                     (canvas.height - source.height) / 2
                 )}};
        }
    }
    throw std::invalid_argument("vmaf: invalid fit mode");
}

VmafResizer::VmafResizer(VulkanRuntime& runtime)
    : runtime_(runtime), pipeline_(runtime.CreatePipeline("vmaf_resize")) {}

VmafResizer::~VmafResizer() { runtime_.DestroyPipelines(std::array{pipeline_}); }

void VmafResizer::RecordHorizontal(
    std::uint32_t source, std::uint32_t destination, Size source_size,
    Size content_size
) {
    if (content_size.height != source_size.height ||
        content_size.width == source_size.width)
        throw std::invalid_argument("vmaf resize: invalid horizontal geometry");
    CheckResizePlane(source, source_size);
    CheckResizePlane(destination, content_size);
    const std::uint32_t filter =
        content_size.width > source_size.width ? 1u : 2u;
    const VmafResizePush push{source,
                              destination,
                              static_cast<std::uint32_t>(source_size.width),
                              static_cast<std::uint32_t>(source_size.height),
                              static_cast<std::uint32_t>(content_size.width),
                              static_cast<std::uint32_t>(content_size.height),
                              0,
                              0,
                              static_cast<std::uint32_t>(content_size.width),
                              static_cast<std::uint32_t>(content_size.height),
                              filter,
                              0,
                              0.0f,
                              0};
    runtime_.Dispatch(
        pipeline_, &push, sizeof(push), Groups(content_size.width, 16),
        Groups(source_size.height, 16), 1
    );
}

void VmafResizer::RecordVertical(
    std::uint32_t source, std::uint32_t destination, Size source_size,
    VmafFitGeometry geometry, Size target_size, float pad, bool quantize_16bit
) {
    if (geometry.content.width != source_size.width ||
        geometry.content.height == 0 || geometry.content.width == 0)
        throw std::invalid_argument("vmaf resize: invalid vertical geometry");
    CheckResizePlane(source, source_size);
    CheckResizePlane(destination, target_size);
    const std::uint32_t filter =
        geometry.content.height == source_size.height
            ? 0u
            : (geometry.content.height > source_size.height ? 1u : 2u);
    const VmafResizePush push{
        source,
        destination,
        static_cast<std::uint32_t>(source_size.width),
        static_cast<std::uint32_t>(source_size.height),
        static_cast<std::uint32_t>(target_size.width),
        static_cast<std::uint32_t>(target_size.height),
        geometry.x,
        geometry.y,
        static_cast<std::uint32_t>(geometry.content.width),
        static_cast<std::uint32_t>(geometry.content.height),
        filter,
        1,
        pad,
        quantize_16bit ? 1u : 0u};
    runtime_.Dispatch(
        pipeline_, &push, sizeof(push), Groups(target_size.width, 16),
        Groups(target_size.height, 16), 1
    );
}

// Reference motion and temporal weighting

// Temporal semantics follow Netflix libvmaf float_motion.c:230-248,333-399
// at revision 4611e47a7330cd342498dcc130e1008efbeab6e4.

namespace {

struct MotionPush {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t luma;
    std::uint32_t previous;
    std::uint32_t blurred;
    std::uint32_t scratch;
    std::uint32_t partials;
    std::uint32_t score;
    std::uint32_t has_previous;
    std::uint32_t groups;
};
static_assert(sizeof(MotionPush) == VmafMotion::PushBytes());

void CheckMotionGeometry(Size size) {
    // A radius-two, one-reflection mirror needs dimensions of at least three.
    // Keep both doubled coordinates and arena word indices representable.
    constexpr std::size_t kMaxDimension =
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() / 2);
    if (size.width < 3 || size.height < 3 || size.width > kMaxDimension ||
        size.height > kMaxDimension ||
        size.width > std::numeric_limits<std::uint32_t>::max() / size.height) {
        throw std::invalid_argument("vmaf motion: invalid image geometry");
    }
}

void CheckMotionRange(std::uint32_t offset, std::size_t count) {
    if (count > std::numeric_limits<std::uint32_t>::max() - offset) {
        throw std::invalid_argument("vmaf motion: arena offset overflow");
    }
}

}  // namespace

VmafMotion::VmafMotion(VulkanRuntime& runtime) : runtime_(&runtime) {
    constexpr std::array names{"vmaf_motion_horizontal", "vmaf_motion_reduce"};
    pipelines_ = runtime_->CreatePipelines(names);
}

VmafMotion::~VmafMotion() { runtime_->DestroyPipelines(pipelines_); }

std::size_t VmafMotion::PartialCount(Size size) {
    CheckMotionGeometry(size);
    const std::size_t groups_x = (size.width + 31) / 32;
    const std::size_t groups_y = (size.height + 7) / 8;
    const std::size_t count = groups_x * groups_y;
    // The reduction loop advances by 256 in uint arithmetic.
    if (count > std::numeric_limits<std::uint32_t>::max() - 255u) {
        throw std::invalid_argument("vmaf motion: too many workgroups");
    }
    return count;
}

void VmafMotion::Record(
    Size size, const VmafMotionBuffers& buffers, bool has_previous
) {
    const std::size_t groups = PartialCount(size);
    const std::uint32_t gx = Groups(size.width, 32);
    const std::uint32_t gy = Groups(size.height, 8);
    const auto& limits = runtime_->context().limits;
    if (gx > limits.maxComputeWorkGroupCount[0] ||
        gy > limits.maxComputeWorkGroupCount[1]) {
        throw std::invalid_argument(
            "vmaf motion: geometry exceeds device limits"
        );
    }
    const std::size_t pixels = size.width * size.height;
    CheckMotionRange(buffers.luma, pixels);
    if (has_previous) CheckMotionRange(buffers.previous, pixels);
    CheckMotionRange(buffers.blurred, pixels);
    CheckMotionRange(buffers.partials, groups);
    CheckMotionRange(buffers.score, 1);
    const MotionPush push{static_cast<std::uint32_t>(size.width),
                          static_cast<std::uint32_t>(size.height),
                          buffers.luma,
                          buffers.previous,
                          buffers.blurred,
                          0u,  // Reserved to preserve push-constant offsets.
                          buffers.partials,
                          buffers.score,
                          has_previous ? 1u : 0u,
                          static_cast<std::uint32_t>(groups)};
    runtime_->ComputeBarrier();
    runtime_->Dispatch(pipelines_[0], &push, sizeof(push), gx, gy, 1);
    runtime_->ComputeBarrier();
    runtime_->Dispatch(pipelines_[1], &push, sizeof(push), 1, 1, 1);
    runtime_->ComputeBarrier();
}

std::optional<VmafMotionFrame> VmafMotionTimeline::Push(double transition) {
    if (finished_) throw std::logic_error("vmaf motion: sequence is finished");
    if (!std::isfinite(transition) || transition < 0.0) {
        throw std::invalid_argument("vmaf motion: invalid transition score");
    }
    if (frames_ == 0) {
        if (transition != 0.0) {
            throw std::invalid_argument(
                "vmaf motion: first transition must be zero"
            );
        }
        frames_ = 1;
        return VmafMotionFrame{0, 0.0};
    }
    if (frames_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("vmaf motion: too many frames");
    }
    std::optional<VmafMotionFrame> ready;
    if (frames_ > 1) {
        // Default motion_fps_weight=1 and motion_max_val=10000.
        ready = VmafMotionFrame{
            frames_ - 1, std::min({last_transition_, transition, 10000.0})};
    }
    last_transition_ = transition;
    ++frames_;
    return ready;
}

std::optional<VmafMotionFrame> VmafMotionTimeline::Flush() {
    if (finished_) return std::nullopt;
    finished_ = true;
    if (frames_ < 2) return std::nullopt;
    return VmafMotionFrame{frames_ - 1, std::min(last_transition_, 10000.0)};
}

void VmafMotionTimeline::Reset() noexcept {
    frames_ = 0;
    last_transition_ = 0.0;
    finished_ = false;
}

}  // namespace vqm::internal
