// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// VMAF input scaling: Copyright (c) Netflix, Inc.
// VMAF motion: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include <vulkan/vulkan.h>

#include "vqm_image.h"

namespace vqm::internal {

class VulkanRuntime;

// Native coded-plane conversion

// Validate the initial VMAF input contract and return its bit depth. VMAF's
// float features read coded luma, not linear RGB. Supported storage is 8-bit
// bytes or 10/12/16-bit words, with no alpha; callers still validate dimensions
// and the backing storage through the chosen VulkanRuntime input path.
[[nodiscard]] std::uint32_t VmafLumaDepth(const ImageFormat& format);

// Records conversion of one source bound by VulkanRuntime::BindSource to a
// tightly packed float plane in the runtime's arena. The caller publishes the
// descriptor at desc_base before dispatch and keeps runtime alive until after
// this object is destroyed. Offsets are arena words. The caller synchronizes
// the output before a subsequent shader reads it.
class VmafLumaConverter {
public:
    explicit VmafLumaConverter(VulkanRuntime& runtime);
    ~VmafLumaConverter();
    VmafLumaConverter(const VmafLumaConverter&) = delete;
    VmafLumaConverter& operator=(const VmafLumaConverter&) = delete;

    [[nodiscard]] static constexpr std::size_t PushBytes() noexcept {
        return 7 * sizeof(std::uint32_t);
    }

    // Size of the coded input plane (0 = Y, 1 = U, 2 = V). Chroma
    // follows the source's native subsampling and is never upsampled.
    [[nodiscard]] static Size PlaneSize(
        Size image_size, const ImageFormat& format, std::uint32_t plane
    );

    void Record(
        std::size_t image, std::uint32_t desc_base, std::uint32_t output,
        Size size, const ImageFormat& format
    );

    // The caller publishes the source descriptor and allocates the result
    // for PlaneSize(size, format, plane) tightly packed floats.
    void RecordPlane(
        std::size_t image, std::uint32_t desc_base, std::uint32_t output,
        Size size, const ImageFormat& format, std::uint32_t plane
    );

private:
    VulkanRuntime& runtime_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

// SDR color conversion

// VMAF's SDR model must not silently score PQ or HLG as SDR. The caller also
// performs the usual image geometry and storage validation.
void ValidateVmafSdrFormat(const ImageFormat& format);

// Whether a source can be consumed by the original coded-plane fast path.
// The caller separately checks whether its native subsampling is suitable.
[[nodiscard]] bool IsVmafNativeInput(const ImageFormat& format);

// Converts a described SDR source to limited-range BT.709, quantized to
// 16-bit code values and stored in the FP32 domain used by VMAF's features.
// A caller that already has the desired coded YUV should use
// VmafLumaConverter instead, to preserve its original samples and precision.
class VmafSdrConverter {
public:
    explicit VmafSdrConverter(VulkanRuntime& runtime);
    ~VmafSdrConverter();
    VmafSdrConverter(const VmafSdrConverter&) = delete;
    VmafSdrConverter& operator=(const VmafSdrConverter&) = delete;

    [[nodiscard]] static constexpr std::size_t PushBytes() noexcept {
        return 8 * sizeof(std::uint32_t);
    }

    // Output offsets are arena words. Y has source_size pixels; U and V have
    // source_size pixels for 4:4:4 or ceil(width/2) * ceil(height/2) for
    // 4:2:0. When emit_chroma is false the U/V offsets are unused. The caller
    // publishes the source descriptor before recording and allocates planes.
    // The caller also synchronizes prior writes to source storage and waits
    // for the output writes before reading or overwriting these planes. Calls
    // with disjoint outputs may be recorded together before one dependency.
    void Record(
        std::size_t image, std::uint32_t desc_base, std::uint32_t out_y,
        std::uint32_t out_u, std::uint32_t out_v, Size source_size,
        const ImageFormat& format, bool chroma420, bool emit_chroma,
        float background = 0.5f
    );

private:
    VulkanRuntime& runtime_;
    std::array<VkPipeline, 12> pipelines_{};
};

// Fit-and-pad resampling

// A source scaled to fit a target canvas, before the remaining canvas is
// padded. `x` and `y` are the top-left content position in the target.
struct VmafFitGeometry {
    Size content{};
    std::uint32_t x = 0;
    std::uint32_t y = 0;
};

// Scale `source` uniformly so it is contained by `target`, preserving its
// aspect ratio, and center the result. The content is never cropped.
[[nodiscard]] VmafFitGeometry FitVmafInto(Size source, Size target);

// Separable luma resampler for the FitAndPad path. It reads and writes coded
// luma in VmafLumaConverter's FP32 code domain. Horizontal output has the
// content width and the source height; vertical output has the target canvas
// and writes `pad` outside the content rectangle. Upscaling uses Catmull-Rom
// bicubic and downscaling uses Lanczos3 with a widened antialiasing support.
class VmafResizer {
public:
    explicit VmafResizer(VulkanRuntime& runtime);
    ~VmafResizer();
    VmafResizer(const VmafResizer&) = delete;
    VmafResizer& operator=(const VmafResizer&) = delete;

    [[nodiscard]] static constexpr std::size_t PushBytes() noexcept {
        return 13 * sizeof(std::uint32_t);
    }

    // Output is tightly packed at content.width x source.height.
    void RecordHorizontal(
        std::uint32_t source, std::uint32_t destination, Size source_size,
        Size content_size
    );

    // `source_size` is content.width x original source height. Destination is
    // the target canvas and is tightly packed at target_size.width.
    void RecordVertical(
        std::uint32_t source, std::uint32_t destination, Size source_size,
        VmafFitGeometry geometry, Size target_size, float pad
    );

private:
    VulkanRuntime& runtime_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

// Reference motion and temporal weighting

// Absolute arena offsets, in 32-bit words. The caller allocates full-resolution
// luma and blurred planes, PartialCount(size) words of partials, and
// one score word. Planes must not overlap while the dispatches run, except that
// `previous` may be ignored on the first frame. The input is the reference Y'
// plane in libvmaf's FP32 domain: code / 2^(bits-8) - 128; it is not RGB and
// must not undergo range expansion or transfer conversion.
struct VmafMotionBuffers {
    std::uint32_t luma = 0;
    std::uint32_t previous = 0;
    std::uint32_t blurred = 0;
    std::uint32_t partials = 0;
    std::uint32_t score = 0;
};

// Record the default five-tap FP32 motion feature for one frame. A previous
// blurred reference is retained in the caller's arena; current blur goes to a
// different plane. The scalar score is the mean absolute difference of those
// planes, or zero for the first frame. The runtime must have a push constant
// range of at least PushBytes() and must outlive this object. The caller owns
// command-buffer recording, queue submission, arena bounds, and score readback.
class VmafMotion {
public:
    // Disabling subgroups keeps the portable shared reduction available for
    // direct comparison on devices that support the subgroup variant.
    explicit VmafMotion(VulkanRuntime& runtime);
    ~VmafMotion();
    VmafMotion(const VmafMotion&) = delete;
    VmafMotion& operator=(const VmafMotion&) = delete;

    [[nodiscard]] static std::size_t PartialCount(Size size);
    [[nodiscard]] static constexpr std::size_t PushBytes() noexcept {
        return 10 * sizeof(std::uint32_t);
    }
    void Record(Size size, const VmafMotionBuffers& buffers, bool has_previous);

private:
    VulkanRuntime* runtime_;
    std::array<VkPipeline, 2> pipelines_{};
};

struct VmafMotionFrame {
    std::uint64_t index = 0;
    double motion2 = 0.0;
};

// Map consecutive transition MADs to the default v0.6.1 motion2 sequence.
// Push(0) for the first reference emits frame 0. Later pushes receive the MAD
// against the preceding reference; frame i>0 needs the next transition and is
// emitted when it arrives. Flush emits the last frame, using its preceding
// transition. Empty and single-frame sequences produce no further frame at
// flush. Flush is idempotent; Reset begins a new independent sequence.
class VmafMotionTimeline {
public:
    [[nodiscard]] std::optional<VmafMotionFrame> Push(double transition);
    [[nodiscard]] std::optional<VmafMotionFrame> Flush();
    void Reset() noexcept;

private:
    std::uint64_t frames_ = 0;
    double last_transition_ = 0.0;
    bool finished_ = false;
};

}  // namespace vqm::internal
