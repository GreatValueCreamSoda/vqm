// VMAF SpEED: Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

#include "vmaf_speed.h"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "vulkan/vulkan_runtime.h"

namespace vqm::internal {

// ---- Chroma filtering and downsampling ----

// VMAF SpEED: Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// filter_and_downscale in libvmaf/src/feature/speed.c and
// vif_filter1d_dec16_scalar_s in vif_tools.c at revision
// 4611e47a7330cd342498dcc130e1008efbeab6e4.

namespace {
struct SpeedFilterPush {
    std::uint32_t width, height, op;
    std::array<std::uint32_t, 4> input, scratch, down, result;
};
static_assert(sizeof(SpeedFilterPush) == VmafSpeedFilter::PushBytes());

void CheckSpeedFilterGeometry(Size size) {
    constexpr auto maxdim =
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() / 2);
    if (size.width < 80 || size.height < 80 || size.width > maxdim ||
        size.height > maxdim ||
        size.width > std::numeric_limits<std::uint32_t>::max() / size.height) {
        throw std::invalid_argument("vmaf speed: invalid chroma geometry");
    }
}
void CheckSpeedFilterRange(std::uint32_t offset, std::size_t size) {
    if (size > std::numeric_limits<std::uint32_t>::max() - offset)
        throw std::invalid_argument("vmaf speed: arena offset overflow");
}
}  // namespace

VmafSpeedFilter::VmafSpeedFilter(VulkanRuntime& runtime) : runtime_(&runtime) {
    constexpr std::array names{"vmaf_speed_filter_vertical",
                               "vmaf_speed_filter_horizontal"};
    pipelines_ = runtime_->CreatePipelines(names);
}
VmafSpeedFilter::~VmafSpeedFilter() { runtime_->DestroyPipelines(pipelines_); }
std::size_t VmafSpeedFilter::ScratchWords(Size size) {
    CheckSpeedFilterGeometry(size);
    return size.width * (size.height / 16);
}
std::size_t VmafSpeedFilter::OutputPixels(Size size) {
    CheckSpeedFilterGeometry(size);
    return (size.width / 16) * (size.height / 16);
}
void VmafSpeedFilter::Record(Size size, const VmafSpeedFilterBuffers& b) {
    RecordMany(size, std::span(&b, 1));
}
void VmafSpeedFilter::RecordBatch(
    Size size, const std::array<VmafSpeedFilterBuffers, 4>& b
) {
    RecordMany(size, b);
}
void VmafSpeedFilter::RecordMany(
    Size size, std::span<const VmafSpeedFilterBuffers> buffers
) {
    const auto scratch = ScratchWords(size), output = OutputPixels(size);
    for (const auto& b : buffers) {
        CheckSpeedFilterRange(b.input, size.width * size.height);
        CheckSpeedFilterRange(b.scratch, scratch);
        CheckSpeedFilterRange(b.down, output);
        CheckSpeedFilterRange(b.result, output);
    }
    const auto& limits = runtime_->context().limits;
    if (Groups(size.width, 16) > limits.maxComputeWorkGroupCount[0] ||
        Groups(size.height / 16, 16) > limits.maxComputeWorkGroupCount[1] ||
        buffers.size() > limits.maxComputeWorkGroupCount[2])
        throw std::invalid_argument(
            "vmaf speed: geometry exceeds device limits"
        );
    const auto width = static_cast<std::uint32_t>(size.width),
               height = static_cast<std::uint32_t>(size.height);
    SpeedFilterPush push{};
    push.width = width;
    push.height = height;
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        push.input[i] = buffers[i].input;
        push.scratch[i] = buffers[i].scratch;
        push.down[i] = buffers[i].down;
        push.result[i] = buffers[i].result;
    }
    runtime_->ComputeBarrier();
    for (std::uint32_t op = 0; op < 2; ++op) {
        push.op = op;
        runtime_->Dispatch(
            pipelines_[0], &push, sizeof(push),
            Groups(op == 0 ? 9u * (width / 16u) : width / 16u, 16),
            Groups(height / 16, 16), static_cast<std::uint32_t>(buffers.size())
        );
        runtime_->ComputeBarrier();
        runtime_->Dispatch(
            pipelines_[1], &push, sizeof(push), Groups(width / 16, 16),
            Groups(height / 16, 16), static_cast<std::uint32_t>(buffers.size())
        );
        runtime_->ComputeBarrier();
    }
}

// ---- Covariance statistics and pooling ----

// VMAF SpEED: Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Follows compute_covariance_matrix, est_params and get_speed_score in
// libvmaf/src/feature/speed.c at 4611e47a7330cd342498dcc130e1008efbeab6e4.

namespace {
struct SpeedStatsPush {
    std::uint32_t width, height, reference, distorted, scratch, partials, result,
        groups, op;
};
static_assert(sizeof(SpeedStatsPush) == VmafSpeedStats::PushBytes());
void CheckSpeedStatsGeometry(Size size) {
    constexpr auto max = std::numeric_limits<std::uint32_t>::max();
    if (size.width < 5 || size.height < 5 || size.width > max / size.height)
        throw std::invalid_argument("vmaf speed statistics: invalid geometry");
}
void CheckSpeedStatsRange(std::uint32_t offset, std::size_t size) {
    if (size > std::numeric_limits<std::uint32_t>::max() - offset)
        throw std::invalid_argument("vmaf speed statistics: offset overflow");
}
}  // namespace
VmafSpeedStats::VmafSpeedStats(VulkanRuntime& runtime, bool allow_wide)
    : runtime_(&runtime) {
    const auto& limits = runtime.context().limits;
    const char* eigen_name =
        allow_wide && limits.maxComputeWorkGroupInvocations >= 512 &&
                limits.maxComputeWorkGroupSize[0] >= 512
            ? "vmaf_speed_stats_eigen_wide"
            : "vmaf_speed_stats_eigen";
    const std::array names{"vmaf_speed_stats_cov", eigen_name,
                           "vmaf_speed_stats_pool", "vmaf_speed_stats_reduce"};
    pipelines_ = runtime_->CreatePipelines(names);
}
VmafSpeedStats::~VmafSpeedStats() { runtime_->DestroyPipelines(pipelines_); }
std::size_t VmafSpeedStats::PartialCount(Size size) {
    CheckSpeedStatsGeometry(size);
    const auto count = (size.width / 5) * (size.height / 5);
    return (count + 63) / 64;
}
void VmafSpeedStats::Record(
    Size size, const VmafSpeedStatsBuffers& b, std::uint32_t channels
) {
    const auto groups = PartialCount(size);
    if (channels == 0 || channels > 2)
        throw std::invalid_argument(
            "vmaf speed statistics: invalid channel count"
        );
    const auto pixels = size.width * size.height;
    if (pixels > std::numeric_limits<std::uint32_t>::max() / channels)
        throw std::invalid_argument("vmaf speed statistics: offset overflow");
    CheckSpeedStatsRange(b.reference, channels * pixels);
    CheckSpeedStatsRange(b.distorted, channels * pixels);
    CheckSpeedStatsRange(b.scratch, channels * ScratchWords());
    CheckSpeedStatsRange(b.partials, channels * groups);
    CheckSpeedStatsRange(b.result, 2 * channels);
    if (groups > runtime_->context().limits.maxComputeWorkGroupCount[0])
        throw std::invalid_argument(
            "vmaf speed statistics: exceeds device limits"
        );
    SpeedStatsPush pc{static_cast<std::uint32_t>(size.width),
                      static_cast<std::uint32_t>(size.height),
                      b.reference,
                      b.distorted,
                      b.scratch,
                      b.partials,
                      b.result,
                      static_cast<std::uint32_t>(groups),
                      0};
    runtime_->ComputeBarrier();
    runtime_->Dispatch(pipelines_[0], &pc, sizeof(pc), 50 * channels, 1, 1);
    runtime_->ComputeBarrier();
    pc.op = 1;
    // Each 25x25 symmetric matrix has 325 independent covariance entries,
    // each reduced by a 64-lane workgroup.
    runtime_->Dispatch(pipelines_[0], &pc, sizeof(pc), 650 * channels, 1, 1);
    runtime_->ComputeBarrier();
    runtime_->Dispatch(pipelines_[1], &pc, sizeof(pc), 2 * channels, 2, 1);
    runtime_->ComputeBarrier();
    runtime_->Dispatch(
        pipelines_[2], &pc, sizeof(pc), static_cast<std::uint32_t>(groups),
        channels, 1
    );
    runtime_->ComputeBarrier();
    runtime_->Dispatch(pipelines_[3], &pc, sizeof(pc), channels, 1, 1);
    runtime_->ComputeBarrier();
}

// ---- UV feature composition ----

// VMAF SpEED: Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Standard feature options and channel combination are from libvmaf
// feature/speed.c and model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json at
// revision 4611e47a7330cd342498dcc130e1008efbeab6e4.

namespace {
struct SpeedCombinePush {
    std::uint32_t channels, result;
};
struct SpeedPrescalePush {
    std::uint32_t input_width, input_height, output_width, output_height;
    std::array<std::uint32_t, 4> source, destination;
};
static_assert(sizeof(SpeedPrescalePush) == VmafSpeedPrescaler::PushBytes());

void CheckSpeedFeatureRange(std::uint32_t offset, std::size_t count) {
    if (count > std::numeric_limits<std::uint32_t>::max() - offset)
        throw std::invalid_argument("vmaf speed: arena offset overflow");
}
}  // namespace

VmafSpeedPrescaler::VmafSpeedPrescaler(VulkanRuntime& runtime)
    : runtime_(runtime),
      pipeline_(runtime.CreatePipeline("vmaf_speed_prescale")) {}

VmafSpeedPrescaler::~VmafSpeedPrescaler() {
    runtime_.DestroyPipelines(std::array{pipeline_});
}

void VmafSpeedPrescaler::Record(
    Size input, Size output, const std::array<std::uint32_t, 4>& source,
    const VmafSpeedBuffers& buffers
) {
    CheckSpeedFilterGeometry(input);
    CheckSpeedFilterGeometry(output);
    if (output.width > input.width || output.height > input.height)
        throw std::invalid_argument("vmaf speed: invalid chroma prescale");
    const SpeedPrescalePush push{static_cast<std::uint32_t>(input.width),
                                 static_cast<std::uint32_t>(input.height),
                                 static_cast<std::uint32_t>(output.width),
                                 static_cast<std::uint32_t>(output.height),
                                 source,
                                 {buffers.reference[0], buffers.reference[1],
                                  buffers.distorted[0], buffers.distorted[1]}};
    for (auto offset : push.source)
        CheckSpeedFeatureRange(offset, input.width * input.height);
    for (auto offset : push.destination)
        CheckSpeedFeatureRange(offset, output.width * output.height);
    runtime_.ComputeBarrier();
    runtime_.Dispatch(
        pipeline_, &push, sizeof(push), Groups(output.width, 16),
        Groups(output.height, 16), 4
    );
    runtime_.ComputeBarrier();
}

VmafSpeed::VmafSpeed(VulkanRuntime& runtime)
    : runtime_(&runtime),
      filter_(runtime),
      stats_(runtime),
      combine_(runtime.CreatePipeline("vmaf_speed_combine")) {}
VmafSpeed::~VmafSpeed() { runtime_->DestroyPipelines(std::array{combine_}); }
void VmafSpeed::Record(Size size, const VmafSpeedBuffers& b) {
    // Validation precedes recording; no partial channel is dispatched on bad
    // geometry or range, even if only the second channel is invalid.
    const auto scratch = VmafSpeedFilter::ScratchWords(size);
    const auto output = VmafSpeedFilter::OutputPixels(size);
    const Size filtered{size.width / 16, size.height / 16};
    const auto partials = VmafSpeedStats::PartialCount(filtered);
    for (auto offset : b.reference)
        CheckSpeedFeatureRange(offset, size.width * size.height);
    for (auto offset : b.distorted)
        CheckSpeedFeatureRange(offset, size.width * size.height);
    CheckSpeedFeatureRange(b.filter_scratch, 4 * scratch);
    CheckSpeedFeatureRange(b.down, 4 * output);
    CheckSpeedFeatureRange(b.reference_filtered, 2 * output);
    CheckSpeedFeatureRange(b.distorted_filtered, 2 * output);
    CheckSpeedFeatureRange(b.stats_scratch, 2 * VmafSpeedStats::ScratchWords());
    CheckSpeedFeatureRange(b.partials, 2 * partials);
    CheckSpeedFeatureRange(b.channel_results, 4);
    CheckSpeedFeatureRange(b.result, 2);
    std::array<VmafSpeedFilterBuffers, 4> planes{};
    for (std::uint32_t channel = 0; channel < 2; ++channel) {
        const auto bank = static_cast<std::uint32_t>(channel * output);
        for (std::uint32_t distorted = 0; distorted < 2; ++distorted) {
            const auto i = 2 * channel + distorted;
            planes[i] = {
                distorted ? b.distorted[channel] : b.reference[channel],
                b.filter_scratch + static_cast<std::uint32_t>(i * scratch),
                b.down + static_cast<std::uint32_t>(i * output),
                (distorted ? b.distorted_filtered : b.reference_filtered) +
                    bank};
        }
    }
    // Reference/distortion U/V are independent; batch corresponding passes
    // while keeping each plane's source, scratch, and output disjoint.
    filter_.RecordBatch(size, planes);
    // The four independent U/V reference/distortion matrices share one
    // dispatch, rather than serializing two eigensolver critical paths.
    stats_.Record(
        filtered,
        {b.reference_filtered, b.distorted_filtered, b.stats_scratch, b.partials,
         b.channel_results},
        2
    );
    const SpeedCombinePush push{b.channel_results, b.result};
    runtime_->Dispatch(combine_, &push, sizeof(push), 1, 1, 1);
    runtime_->ComputeBarrier();
}

}  // namespace vqm::internal
