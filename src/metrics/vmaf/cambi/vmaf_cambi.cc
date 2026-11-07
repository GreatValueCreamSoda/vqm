// VMAF CAMBI: Copyright 2016-2020 Netflix, Inc.
// CAMBI pooling: Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

#include "vmaf_cambi.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "vulkan/vulkan_runtime.h"

namespace vqm::internal {

// ---- Preparation and texture mask ----

// VMAF CAMBI: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Same-size conversion, anti_dithering_filter and spatial mask from
// libvmaf/src/feature/cambi.c, pinned revision
// 4611e47a7330cd342498dcc130e1008efbeab6e4.

namespace {
struct CambiPreparePush {
    std::uint32_t width, height, input, processed, derivative, mask, depth,
        threshold, op;
};
static_assert(sizeof(CambiPreparePush) == VmafCambiPrepare::PushBytes());
}  // namespace

VmafCambiPrepare::VmafCambiPrepare(VulkanRuntime& runtime)
    : runtime_(&runtime),
      pipeline_(runtime.CreatePipeline("vmaf_cambi_prepare")) {}
VmafCambiPrepare::~VmafCambiPrepare() {
    runtime_->DestroyPipelines(std::array{pipeline_});
}
std::size_t VmafCambiPrepare::Pixels(Size size) {
    constexpr auto max = std::numeric_limits<std::uint32_t>::max();
    if (!size.width || !size.height || size.width > max || size.height > max ||
        size.width > max / size.height)
        throw std::invalid_argument("vmaf cambi: invalid geometry");
    return size.width * size.height;
}
std::uint32_t VmafCambiPrepare::MaskThreshold(Size size) {
    (void)Pixels(size);
    const auto product =
        static_cast<std::uint32_t>((size.width >> 6) * (size.height >> 6));
    const int ceil_log2 =
        product ? static_cast<int>(std::bit_width(product - 1)) : 0;
    return static_cast<std::uint32_t>((49 + 3 * (ceil_log2 - 11) - 1) >> 1);
}
void VmafCambiPrepare::Record(
    Size size, std::uint32_t input_depth, std::uint32_t encoded_depth,
    const VmafCambiPrepareBuffers& b
) {
    const auto pixels = Pixels(size);
    if ((input_depth != 8 && input_depth != 10 && input_depth != 12 &&
         input_depth != 16) ||
        encoded_depth < 1 || encoded_depth > 16)
        throw std::invalid_argument("vmaf cambi: unsupported bit depth");
    const std::array offsets{b.input, b.processed, b.derivative, b.mask};
    for (auto offset : offsets)
        if (pixels > std::numeric_limits<std::uint32_t>::max() - offset)
            throw std::invalid_argument("vmaf cambi: arena offset overflow");
    const auto& limits = runtime_->context().limits;
    if (Groups(size.width, 16) > limits.maxComputeWorkGroupCount[0] ||
        Groups(size.height, 16) > limits.maxComputeWorkGroupCount[1])
        throw std::invalid_argument(
            "vmaf cambi: geometry exceeds device limits"
        );
    // The tiled conversion/derivative dispatch pays for a shared-memory halo.
    // Interleaved resident measurements improved at 4K but slowed at 1440p.
    const bool fused = pixels >= 3840u * 2160u;
    runtime_->ComputeBarrier();
    for (std::uint32_t op = 0; op < 3; ++op) {
        if (fused && op == 1) continue;
        const CambiPreparePush push{
            static_cast<std::uint32_t>(size.width),
            static_cast<std::uint32_t>(size.height),
            b.input,
            b.processed,
            b.derivative,
            b.mask,
            input_depth,
            MaskThreshold(size),
            op | (encoded_depth < 10 ? 4u : 0u) | (fused ? 8u : 0u)};
        runtime_->Dispatch(
            pipeline_, &push, sizeof(push), Groups(size.width, 16),
            Groups(size.height, 16), 1
        );
        runtime_->ComputeBarrier();
    }
}

// ---- Scale decimation ----

// VMAF CAMBI: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Scale dimensions and decimation from libvmaf/src/feature/cambi.c
// at Netflix/vmaf revision 4611e47a7330cd342498dcc130e1008efbeab6e4.

namespace {
struct CambiDecimatePush {
    std::uint32_t width, height, pixels, mask, pixels_out, mask_out;
};
static_assert(sizeof(CambiDecimatePush) == VmafCambiDecimate::PushBytes());
void CheckCambiDecimateRange(std::uint32_t offset, std::size_t size) {
    if (size > std::numeric_limits<std::uint32_t>::max() - offset)
        throw std::invalid_argument("cambi decimate: arena offset overflow");
}
}  // namespace
VmafCambiDecimate::VmafCambiDecimate(VulkanRuntime& runtime)
    : runtime_(&runtime),
      pipeline_(runtime.CreatePipeline("vmaf_cambi_decimate")) {}
VmafCambiDecimate::~VmafCambiDecimate() {
    runtime_->DestroyPipelines(std::array{pipeline_});
}
Size VmafCambiDecimate::OutputSize(Size size) {
    const auto max = std::numeric_limits<std::uint32_t>::max();
    if (!size.width || !size.height || size.width >= max || size.height >= max ||
        size.width > max / size.height)
        throw std::invalid_argument("cambi decimate: invalid geometry");
    return {(size.width + 1) / 2, (size.height + 1) / 2};
}
void VmafCambiDecimate::Record(Size size, const VmafCambiDecimateBuffers& b) {
    const auto output = OutputSize(size);
    CheckCambiDecimateRange(b.pixels, size.width * size.height);
    CheckCambiDecimateRange(b.mask, size.width * size.height);
    CheckCambiDecimateRange(b.pixels_out, output.width * output.height);
    CheckCambiDecimateRange(b.mask_out, output.width * output.height);
    const auto& limits = runtime_->context().limits;
    if (Groups(output.width, 16) > limits.maxComputeWorkGroupCount[0] ||
        Groups(output.height, 16) > limits.maxComputeWorkGroupCount[1])
        throw std::invalid_argument("cambi decimate: exceeds device limits");
    const CambiDecimatePush push{static_cast<std::uint32_t>(size.width),
                                 static_cast<std::uint32_t>(size.height),
                                 b.pixels,
                                 b.mask,
                                 b.pixels_out,
                                 b.mask_out};
    runtime_->ComputeBarrier();
    runtime_->Dispatch(
        pipeline_, &push, sizeof(push), Groups(output.width, 16),
        Groups(output.height, 16), 1
    );
    runtime_->ComputeBarrier();
}

// ---- Mode filtering and local contrast ----

// VMAF CAMBI: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Mode filter and contrast: libvmaf/src/feature/cambi.c, pinned revision
// 4611e47a7330cd342498dcc130e1008efbeab6e4.

namespace {
bool SupportsCambiSubgroupRows(VulkanRuntime& runtime) {
    VkPhysicalDeviceSubgroupProperties subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(
        runtime.context().physical_device, &properties
    );
    // BASIC subgroup operations are core Vulkan 1.1, without an enabled-device
    // feature. Float16 storage remains independently negotiated below.
    return VmafCambiContrast::SupportsSubgroupRows(subgroup);
}
struct CambiContrastPush {
    std::uint32_t width, height, input, mask, horizontal, filtered, contrast,
        radius, phase, row_base;
};
static_assert(sizeof(CambiContrastPush) == VmafCambiContrast::PushBytes());
}  // namespace

VmafCambiContrast::VmafCambiContrast(
    VulkanRuntime& runtime, bool allow_half, bool allow_subgroups
)
    : runtime_(&runtime),
      // Float16 workgroup storage needs the enabled shaderFloat16 feature,
      // independently of buffer storage or the physical device's support.
      half_source_(allow_half && runtime.context().shader_float16),
      subgroup_pipeline_(allow_subgroups && SupportsCambiSubgroupRows(runtime)),
      pipeline_(runtime.CreatePipeline(
          subgroup_pipeline_
              ? (half_source_ ? "vmaf_cambi_contrast_subgroup_half"
                              : "vmaf_cambi_contrast_subgroup")
              : (half_source_ ? "vmaf_cambi_contrast_half"
                              : "vmaf_cambi_contrast")
      )) {}
VmafCambiContrast::~VmafCambiContrast() {
    runtime_->DestroyPipelines(std::array{pipeline_});
}
std::size_t VmafCambiContrast::Pixels(Size size, unsigned window) {
    if (!size.width || !size.height || size.width > 1920 || size.height > 1080 ||
        !window || window > 33 || (window & 1u) == 0 || size.width < window ||
        size.height < window)
        throw std::invalid_argument(
            "vmaf cambi: unsupported contrast geometry or window"
        );
    return size.width * size.height;
}
void VmafCambiContrast::Record(
    Size size, unsigned window, const VmafCambiContrastBuffers& b
) {
    const auto pixels = Pixels(size, window);
    const std::array offsets{b.input, b.mask, b.horizontal, b.filtered,
                             b.contrast};
    for (auto offset : offsets)
        if (pixels > std::numeric_limits<std::uint32_t>::max() - offset)
            throw std::invalid_argument("vmaf cambi: arena offset overflow");
    const auto& limits = runtime_->context().limits;
    if (Groups(size.width, 16) > limits.maxComputeWorkGroupCount[0] ||
        Groups(size.height, 8) > limits.maxComputeWorkGroupCount[1])
        throw std::invalid_argument(
            "vmaf cambi: geometry exceeds device limits"
        );
    CambiContrastPush push{static_cast<std::uint32_t>(size.width),
                           static_cast<std::uint32_t>(size.height),
                           b.input,
                           b.mask,
                           b.horizontal,
                           b.filtered,
                           b.contrast,
                           window / 2,
                           0,
                           0};
    runtime_->ComputeBarrier();
    for (std::uint32_t phase = 0; phase < 2; ++phase) {
        push.phase = phase;
        runtime_->Dispatch(
            pipeline_, &push, sizeof(push), Groups(size.width, 16),
            Groups(size.height, 16), 1
        );
        runtime_->ComputeBarrier();
    }
    // Sliding histograms bound each workgroup to 16x8 outputs and reuse
    // neighboring windows. Distinct groups write distinct output pixels.
    push.phase = 2;
    runtime_->Dispatch(
        pipeline_, &push, sizeof(push), Groups(size.width, 16),
        Groups(size.height, 8), 1
    );
    runtime_->ComputeBarrier();
}

// ---- Exact top-k pooling ----

// CAMBI: Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Pooling semantics follow libvmaf cambi.c:1197-1239 at revision
// 4611e47a7330cd342498dcc130e1008efbeab6e4. The radix selector and FP32
// parallel reduction are VQM's implementation of the same selected set.

namespace {
struct CambiPoolPush {
    std::uint32_t values, histogram, state, partials, result;
    std::uint32_t count, selected, groups, pass;
    std::array<std::uint32_t, 5> counts{}, selected_counts{};
};
static_assert(sizeof(CambiPoolPush) == VmafCambiPool::PushBytes());
void CheckCambiPoolRange(std::uint32_t offset, std::size_t count) {
    if (count > std::numeric_limits<std::uint32_t>::max() - offset)
        throw std::invalid_argument("vmaf cambi pool: arena offset overflow");
}
}  // namespace
VmafCambiPool::VmafCambiPool(VulkanRuntime& runtime) : runtime_(&runtime) {
    constexpr std::array names{
        "vmaf_cambi_pool_clear", "vmaf_cambi_pool_histogram",
        "vmaf_cambi_pool_select", "vmaf_cambi_pool_partial",
        "vmaf_cambi_pool_finish"};
    pipelines_ = runtime_->CreatePipelines(names);
}
VmafCambiPool::~VmafCambiPool() { runtime_->DestroyPipelines(pipelines_); }
std::size_t VmafCambiPool::PartialCount(std::size_t count) {
    if (count == 0 || count > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("vmaf cambi pool: invalid count");
    return count / 256u + (count % 256u != 0);
}
std::uint32_t VmafCambiPool::SelectedCount(std::size_t count, double ratio) {
    (void)PartialCount(count);
    if (!std::isfinite(ratio) || ratio <= 0.0 || ratio > 1.0)
        throw std::invalid_argument("vmaf cambi pool: invalid top-k ratio");
    const double selected = std::max(
        1.0,
        std::min(static_cast<double>(count), ratio * static_cast<double>(count))
    );
    return static_cast<std::uint32_t>(selected);
}
void VmafCambiPool::Record(
    std::size_t count, const VmafCambiPoolBuffers& buffers, double ratio
) {
    const auto selected = SelectedCount(count, ratio);
    const auto groups = PartialCount(count);
    if (groups > runtime_->context().limits.maxComputeWorkGroupCount[0])
        throw std::invalid_argument("vmaf cambi pool: too many workgroups");
    CheckCambiPoolRange(buffers.values, count);
    CheckCambiPoolRange(buffers.histogram, 256);
    CheckCambiPoolRange(buffers.state, 2);
    CheckCambiPoolRange(buffers.partials, groups);
    CheckCambiPoolRange(buffers.result, 1);
    CambiPoolPush push{buffers.values,
                       buffers.histogram,
                       buffers.state,
                       buffers.partials,
                       buffers.result,
                       static_cast<std::uint32_t>(count),
                       selected,
                       static_cast<std::uint32_t>(groups),
                       0,
                       {},
                       {}};
    runtime_->ComputeBarrier();
    push.pass = 3;
    runtime_->Dispatch(pipelines_[0], &push, sizeof(push), 1, 1, 1);
    runtime_->ComputeBarrier();
    for (int pass = 3; pass >= 0; --pass) {
        push.pass = static_cast<std::uint32_t>(pass);
        runtime_->Dispatch(
            pipelines_[1], &push, sizeof(push), push.groups, 1, 1
        );
        runtime_->ComputeBarrier();
        runtime_->Dispatch(pipelines_[2], &push, sizeof(push), 1, 1, 1);
        runtime_->ComputeBarrier();
    }
    runtime_->Dispatch(pipelines_[3], &push, sizeof(push), push.groups, 1, 1);
    runtime_->ComputeBarrier();
    runtime_->Dispatch(pipelines_[4], &push, sizeof(push), 1, 1, 1);
    runtime_->ComputeBarrier();
}

void VmafCambiPool::RecordBatch(
    const std::array<std::size_t, 5>& counts,
    const VmafCambiPoolBuffers& buffers, double ratio, bool allow_packed
) {
    std::size_t total = 0, partials = 0, max_groups = 0;
    CambiPoolPush push{buffers.values,
                       buffers.histogram,
                       buffers.state,
                       buffers.partials,
                       buffers.result,
                       0,
                       0,
                       0,
                       0,
                       {},
                       {}};
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const auto groups = PartialCount(counts[i]);
        if (counts[i] > std::numeric_limits<std::uint32_t>::max() - total ||
            groups > std::numeric_limits<std::uint32_t>::max() - partials)
            throw std::invalid_argument("vmaf cambi pool: batch overflow");
        total += counts[i];
        partials += groups;
        max_groups = std::max(max_groups, groups);
        push.counts[i] = static_cast<std::uint32_t>(counts[i]);
        push.selected_counts[i] = SelectedCount(counts[i], ratio);
    }
    const auto& limits = runtime_->context().limits;
    if (max_groups > limits.maxComputeWorkGroupCount[0] ||
        5 > limits.maxComputeWorkGroupCount[1])
        throw std::invalid_argument(
            "vmaf cambi pool: batch exceeds device limits"
        );
    CheckCambiPoolRange(buffers.values, total);
    CheckCambiPoolRange(buffers.histogram, 5 * 256);
    CheckCambiPoolRange(buffers.state, 5 * 2);
    CheckCambiPoolRange(buffers.partials, partials);
    CheckCambiPoolRange(buffers.result, 5);
    // Pack the variable-length scales into the x grid to avoid launching four
    // almost-empty grids the size of the largest scale. If the packed x grid
    // exceeds this device's limit, retain the rectangular fallback.
    const bool packed =
        allow_packed && partials <= limits.maxComputeWorkGroupCount[0];
    push.groups = static_cast<std::uint32_t>(packed ? partials : max_groups);
    const std::uint32_t groups_y = packed ? 1 : 5;
    const std::uint32_t packed_bit = packed ? 256 : 0;
    runtime_->ComputeBarrier();
    push.pass = 3;
    runtime_->Dispatch(pipelines_[0], &push, sizeof(push), 1, 5, 1);
    runtime_->ComputeBarrier();
    for (int pass = 3; pass >= 0; --pass) {
        push.pass = static_cast<std::uint32_t>(pass) | packed_bit;
        runtime_->Dispatch(
            pipelines_[1], &push, sizeof(push), push.groups, groups_y, 1
        );
        runtime_->ComputeBarrier();
        push.pass = static_cast<std::uint32_t>(pass);
        runtime_->Dispatch(pipelines_[2], &push, sizeof(push), 1, 5, 1);
        runtime_->ComputeBarrier();
    }
    push.pass |= packed_bit;
    runtime_->Dispatch(
        pipelines_[3], &push, sizeof(push), push.groups, groups_y, 1
    );
    push.pass &= 255u;
    runtime_->ComputeBarrier();
    runtime_->Dispatch(pipelines_[4], &push, sizeof(push), 1, 5, 1);
    runtime_->ComputeBarrier();
}

// ---- Multiscale feature composition ----

// VMAF CAMBI: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Scale ordering, high-resolution threshold, window scaling, and weights
// follow libvmaf/src/feature/cambi.c at Netflix/vmaf revision
// 4611e47a7330cd342498dcc130e1008efbeab6e4.

namespace {
struct CambiCombinePush {
    std::uint32_t scales, result, window;
};
void CheckCambiFeatureRange(std::uint32_t offset, std::size_t count) {
    if (count > std::numeric_limits<std::uint32_t>::max() - offset)
        throw std::invalid_argument("vmaf cambi: arena offset overflow");
}
}  // namespace
VmafCambi::VmafCambi(VulkanRuntime& runtime)
    : runtime_(&runtime),
      prepare_(runtime),
      decimate_(runtime),
      contrast_(runtime),
      pool_(runtime),
      combine_(runtime.CreatePipeline("vmaf_cambi_combine")) {}
VmafCambi::~VmafCambi() { runtime_->DestroyPipelines(std::array{combine_}); }
VmafCambiPlan VmafCambi::Plan(Size size) {
    const auto pixels = VmafCambiPrepare::Pixels(size);
    if (size.width < 216 && size.height < 216)
        throw std::invalid_argument(
            "vmaf cambi: both input dimensions below 216"
        );
    const bool speedup = pixels >= 1920u * 1080u;
    // Upstream adjust_window_size(65, width, height, speedup): integer
    // truncation occurs before the high-resolution halving and odd rounding.
    auto window =
        static_cast<unsigned>(((65u * (size.width + size.height)) / 375u) >> 4u);
    if (speedup) window = (window + 1u) >> 1u;
    window |= 1u;
    auto current = speedup ? VmafCambiDecimate::OutputSize(size) : size;
    VmafCambiPlan plan{};
    plan.max_scale_size = current;
    plan.max_scale_pixels = current.width * current.height;
    plan.window = window;
    plan.high_res_speedup = speedup;
    for (unsigned scale = 0; scale < 5; scale++) {
        if (scale > 0) current = VmafCambiDecimate::OutputSize(current);
        (void)VmafCambiContrast::Pixels(current, window);
        const auto count = current.width * current.height;
        plan.scale_pixels[scale] = count;
        if (count >
            std::numeric_limits<std::uint32_t>::max() - plan.total_scale_pixels)
            throw std::invalid_argument("vmaf cambi: scale storage overflow");
        plan.total_scale_pixels += count;
        plan.total_partials += VmafCambiPool::PartialCount(count);
    }
    return plan;
}
void VmafCambi::Record(
    Size size, std::uint32_t input_depth, std::uint32_t encoded_depth,
    const VmafCambiBuffers& b
) {
    const auto plan = Plan(size);
    if (encoded_depth == 0) encoded_depth = input_depth;
    if ((input_depth != 8 && input_depth != 10 && input_depth != 12 &&
         input_depth != 16) ||
        encoded_depth < 6 || encoded_depth > 16)
        throw std::invalid_argument(
            "vmaf cambi: unsupported input or encoded bit depth"
        );
    const auto pixels = size.width * size.height, sp = plan.max_scale_pixels;
    for (auto offset : std::array{b.input, b.prepared, b.derivative, b.mask})
        CheckCambiFeatureRange(offset, pixels);
    for (auto offset :
         std::array{b.scale_input, b.scale_mask[0], b.scale_mask[1],
                    b.contrast_horizontal, b.filtered})
        CheckCambiFeatureRange(offset, sp);
    CheckCambiFeatureRange(b.contrast, plan.total_scale_pixels);
    CheckCambiFeatureRange(b.pool_histogram, 5 * 256);
    CheckCambiFeatureRange(b.pool_state, 5 * 2);
    CheckCambiFeatureRange(b.pool_partials, plan.total_partials);
    CheckCambiFeatureRange(b.scale_scores, 5);
    CheckCambiFeatureRange(b.result, 1);
    prepare_.Record(
        size, input_depth, encoded_depth,
        {b.input, b.prepared, b.derivative, b.mask}
    );
    auto current = size;
    std::uint32_t pixels_src = b.prepared, mask_src = b.mask, mask_bank = 0;
    std::uint32_t contrast_offset = b.contrast;
    for (std::uint32_t scale = 0; scale < 5; scale++) {
        if (scale > 0 || plan.high_res_speedup) {
            decimate_.Record(
                current,
                {pixels_src, mask_src, b.scale_input, b.scale_mask[mask_bank]}
            );
            current = VmafCambiDecimate::OutputSize(current);
            pixels_src = b.scale_input;
            mask_src = b.scale_mask[mask_bank];
            mask_bank ^= 1u;
        }
        contrast_.Record(
            current, plan.window,
            {pixels_src, mask_src, b.contrast_horizontal, b.filtered,
             contrast_offset}
        );
        contrast_offset += static_cast<std::uint32_t>(plan.scale_pixels[scale]);
        pixels_src = b.filtered;
    }
    pool_.RecordBatch(
        plan.scale_pixels, {b.contrast, b.pool_histogram, b.pool_state,
                            b.pool_partials, b.scale_scores}
    );
    const CambiCombinePush push{b.scale_scores, b.result, plan.window};
    runtime_->Dispatch(combine_, &push, sizeof(push), 1, 1, 1);
    runtime_->ComputeBarrier();
}

}  // namespace vqm::internal
