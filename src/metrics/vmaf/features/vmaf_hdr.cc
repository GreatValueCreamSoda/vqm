// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
#include "vmaf_hdr.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include "metrics/vmaf/features/vmaf_preprocess.h"
#include "vulkan/vulkan_runtime.h"

namespace vqm::internal {
namespace {

struct HdrPush {
    std::uint32_t desc_base, width, height;
    std::array<std::uint32_t, 3> planes;
    std::uint32_t plane;
    float background;
};

struct ResizePush {
    std::uint32_t source, destination, width, height, out_width, out_height;
    std::uint32_t vertical, flags;
    float shift;
};
static_assert(sizeof(ResizePush) == VmafHdrConverter::PushBytes());
static_assert(sizeof(HdrPush) <= VmafHdrConverter::PushBytes());

Size ChromaSize(Size size, const ImageFormat& format) {
    return {size.width >> format.subsample_w, size.height >> format.subsample_h};
}

std::array<float, 2> ChromaOrigin(const ImageFormat& format) {
    const float x = 0.5f - 0.5f / static_cast<float>(1u << format.subsample_w);
    const float y = 0.5f - 0.5f / static_cast<float>(1u << format.subsample_h);
    const auto location = format.chroma_location;
    const bool left = location == ChromaLocation::Left ||
                      location == ChromaLocation::TopLeft ||
                      location == ChromaLocation::BottomLeft;
    const bool top =
        location == ChromaLocation::Top || location == ChromaLocation::TopLeft;
    const bool bottom = location == ChromaLocation::Bottom ||
                        location == ChromaLocation::BottomLeft;
    return {left ? x : 0.0f, top ? y : bottom ? -y : 0.0f};
}

std::array<float, 2> ResizeShift(
    Size from, Size to, std::array<float, 2> source, std::array<float, 2> target
) {
    return {source[0] - target[0] * (static_cast<float>(from.width) /
                                     static_cast<float>(to.width)),
            source[1] - target[1] * (static_cast<float>(from.height) /
                                     static_cast<float>(to.height))};
}

}  // namespace

void ValidateVmafHdrFormat(const ImageFormat& format) {
    if (format.family != ColorFamily::Yuv || format.alpha != AlphaType::None ||
        format.primaries != ColorPrimaries::Bt2020 ||
        format.transfer != TransferCharacteristics::St2084 ||
        (format.matrix != MatrixCoefficients::Bt2020Ncl &&
         format.matrix != MatrixCoefficients::Bt2100ICtCp) ||
        format.subsample_w > 1 || format.subsample_h > format.subsample_w)
        throw std::invalid_argument(
            "vmaf v1 HDR: expected PQ BT.2020 YUV (BT.2020 NCL or ICtCp, "
            "4:2:0, 4:2:2 or 4:4:4)"
        );
    (void)VmafLumaDepth(format);
}

// libvmaf conversion_policy.c at 4611e47 scales the long side to the
// nearest even size whenever the short side changes.
Size VmafHdrGeometry(Size source, std::size_t short_side) {
    const auto small = std::min(source.width, source.height);
    const auto large = std::max(source.width, source.height);
    if (small == 0 || short_side == 0)
        throw std::invalid_argument("vmaf v1 HDR: invalid display geometry");
    const auto scaled =
        small == short_side
            ? large
            : (static_cast<std::uint64_t>(large) * short_side + small) /
                  (2u * static_cast<std::uint64_t>(small)) * 2u;
    if (scaled > 4 * short_side || scaled % 2 != 0)
        throw std::invalid_argument(
            "vmaf v1 HDR: target 4:2:0 needs even dimensions and aspect <= 4:1"
        );
    const auto long_side = static_cast<std::size_t>(scaled);
    return source.width >= source.height ? Size{long_side, short_side}
                                         : Size{short_side, long_side};
}

VmafHdrConverter::VmafHdrConverter(VulkanRuntime& runtime)
    : runtime_(runtime),
      pipelines_(
          runtime.CreatePipelines(std::array{"vmaf_hdr", "vmaf_hdr_resize"})
      ) {}

VmafHdrConverter::~VmafHdrConverter() { runtime_.DestroyPipelines(pipelines_); }

void VmafHdrConverter::Resize(
    std::uint32_t source, std::uint32_t destination, std::uint32_t scratch,
    Size from, Size to, std::array<float, 2> shift, unsigned plane, bool quantize
) {
    const auto dispatch = [&](Size input, Size output, bool vertical,
                              std::uint32_t src, std::uint32_t dst) {
        const unsigned flags =
            (plane != 0 ? 1u : 0u) | (vertical && quantize ? 2u : 0u);
        const ResizePush push{src,
                              dst,
                              static_cast<std::uint32_t>(input.width),
                              static_cast<std::uint32_t>(input.height),
                              static_cast<std::uint32_t>(output.width),
                              static_cast<std::uint32_t>(output.height),
                              vertical ? 1u : 0u,
                              flags,
                              shift[vertical ? 1 : 0]};
        runtime_.Dispatch(
            pipelines_[1], &push, sizeof(push), Groups(output.width, 16),
            Groups(output.height, 16), 1
        );
        runtime_.ComputeBarrier();
    };
    const Size horizontal{to.width, from.height};
    if (from.width != to.width || shift[0] != 0.0f) {
        dispatch(from, horizontal, false, source, scratch);
        source = scratch;
    }
    dispatch(horizontal, to, true, source, destination);
}

void VmafHdrConverter::Record(
    std::size_t image, std::uint32_t desc, const ImageFormat& format,
    Size source, Size target, const VmafHdrBuffers& buffers,
    const std::array<std::uint32_t, 3>& output
) {
    const bool ictcp = format.matrix == MatrixCoefficients::Bt2100ICtCp;
    const Size working{std::min(source.width, target.width),
                       std::min(source.height, target.height)};
    const Size target_chroma{target.width / 2, target.height / 2};
    constexpr std::array<float, 2> kLeft420{0.25f, 0.0f};
    const auto dispatch = [&](Size size, const auto& planes, unsigned plane) {
        const HdrPush push{desc,
                           static_cast<std::uint32_t>(size.width),
                           static_cast<std::uint32_t>(size.height),
                           planes,
                           plane,
                           0.0f};
        runtime_.Dispatch(
            pipelines_[0], &push, sizeof(push), Groups(size.width, 16),
            Groups(size.height, 16), 1
        );
        runtime_.ComputeBarrier();
    };

    runtime_.BindSource(image);
    for (unsigned plane = 0; plane < 3; ++plane) {
        const Size from = plane == 0 ? source : ChromaSize(source, format);
        const Size to = ictcp ? (plane == 0 ? target : target_chroma) : working;
        const auto shift = plane == 0
                               ? std::array{0.0f, 0.0f}
                               : ResizeShift(
                                     from, to, ChromaOrigin(format),
                                     ictcp ? kLeft420 : std::array{0.0f, 0.0f}
                                 );
        dispatch(from, std::array{buffers.source, 0u, 0u}, plane);
        Resize(
            buffers.source, ictcp ? output[plane] : buffers.working[plane],
            buffers.horizontal, from, to, shift, plane, ictcp
        );
    }
    if (ictcp) return;

    // zimg graphbuilder.cpp at f819b14 converts color at the smaller luma
    // dimensions. Chroma reaches 4:4:4 for the matrix before target resampling.
    dispatch(working, buffers.working, 3);
    for (unsigned plane = 0; plane < 3; ++plane) {
        const auto to = plane == 0 ? target : target_chroma;
        const auto shift =
            plane == 0 ? std::array{0.0f, 0.0f}
                       : ResizeShift(working, to, {0.0f, 0.0f}, kLeft420);
        Resize(
            buffers.working[plane], output[plane], buffers.horizontal, working,
            to, shift, plane, true
        );
    }
}

}  // namespace vqm::internal
