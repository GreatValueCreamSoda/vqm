// VMAF ADM: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// The Daubechies-2 transform follows libvmaf's adm_tools.c:1017-1153 at
// revision 4611e47a7330cd342498dcc130e1008efbeab6e4.

#include "vmaf_adm.h"

#include <array>
#include <limits>
#include <stdexcept>

#include "vulkan/vulkan_runtime.h"

namespace vqm::internal {
namespace {

struct WaveletPush {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t source;
    std::uint32_t low;
    std::uint32_t horizontal;
    std::uint32_t vertical;
    std::uint32_t diagonal;
};
static_assert(sizeof(WaveletPush) == VmafAdmWavelet::PushBytes());

void CheckAdmRange(std::uint32_t offset, std::size_t count) {
    if (count > std::numeric_limits<std::uint32_t>::max() - offset) {
        throw std::invalid_argument("vmaf adm: arena offset overflow");
    }
}

}  // namespace

// Shared wavelet transform

VmafAdmWavelet::VmafAdmWavelet(VulkanRuntime& runtime) : runtime_(&runtime) {
    pipeline_ = runtime_->CreatePipeline("vmaf_adm_dwt");
}

VmafAdmWavelet::~VmafAdmWavelet() {
    runtime_->DestroyPipelines(std::array{pipeline_});
}

Size VmafAdmWavelet::OutputSize(Size input) {
    // Reflection uses 2 * coordinate and a single reflection. The source
    // therefore needs at least two samples in each dimension.
    constexpr std::size_t kMaxDimension =
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() / 2);
    if (input.width < 2 || input.height < 2 || input.width > kMaxDimension ||
        input.height > kMaxDimension ||
        input.width > std::numeric_limits<std::uint32_t>::max() / input.height) {
        throw std::invalid_argument("vmaf adm: invalid wavelet geometry");
    }
    return {(input.width + 1) / 2, (input.height + 1) / 2};
}

void VmafAdmWavelet::Record(Size input, const VmafAdmWaveletBuffers& buffers) {
    const Size output = OutputSize(input);
    const std::size_t output_pixels = output.width * output.height;
    CheckAdmRange(buffers.source, input.width * input.height);
    for (const auto offset :
         {buffers.low, buffers.horizontal, buffers.vertical, buffers.diagonal}) {
        CheckAdmRange(offset, output_pixels);
    }
    const std::uint32_t gx = Groups(output.width, 16);
    const std::uint32_t gy = Groups(output.height, 16);
    const auto& limits = runtime_->context().limits;
    if (gx > limits.maxComputeWorkGroupCount[0] ||
        gy > limits.maxComputeWorkGroupCount[1]) {
        throw std::invalid_argument("vmaf adm: geometry exceeds device limits");
    }
    const WaveletPush push{static_cast<std::uint32_t>(input.width),
                           static_cast<std::uint32_t>(input.height),
                           buffers.source,
                           buffers.low,
                           buffers.horizontal,
                           buffers.vertical,
                           buffers.diagonal};
    runtime_->ComputeBarrier();
    runtime_->Dispatch(pipeline_, &push, sizeof(push), gx, gy, 1);
    runtime_->ComputeBarrier();
}

namespace {
struct AdmPush {
    std::uint32_t width, height, reference, distorted, restored, additive;
    std::uint32_t stride, partials, results, scale, groups, original_pixels;
    float axial, diagonal;
};
static_assert(sizeof(AdmPush) == VmafAdm::PushBytes());
static_assert(sizeof(AdmPush) == VmafAdm3::PushBytes());

struct WaveletDecouplePush {
    std::uint32_t width, height, reference, distorted, reference_low,
        distorted_low, reference_details, restored, additive, stride;
};
static_assert(sizeof(WaveletDecouplePush) <= VmafAdm::PushBytes());
static_assert(sizeof(WaveletDecouplePush) <= VmafAdm3::PushBytes());

constexpr std::array<std::array<float, 2>, 4> kAdm2Csf{{
    {0x1.1cc774p-6f, 0x1.820d54p-8f},
    {0x1.06050cp-5f, 0x1.d48d46p-7f},
    {0x1.634f1p-5f, 0x1.8fb80cp-6f},
    {0x1.762816p-5f, 0x1.008392p-5f},
}};

struct AdmBufferRequirements {
    std::size_t partial_values_per_group;
    std::size_t result_words;
    const char* overflow_message;
};

// Both models use the same pyramid and detail planes. Only partial/result
// widths differ; retain each model's overflow diagnostic at the call site.
std::size_t ValidateAdmBuffers(
    VulkanRuntime& runtime, Size input, const VmafAdmBuffers& buffers,
    const AdmBufferRequirements& requirements
) {
    const auto stride = VmafAdm::BandPixels(input);
    const auto partial_count = VmafAdm::PartialCount(input);
    const auto check_range = [&](std::uint32_t offset, std::size_t count) {
        if (count > std::numeric_limits<std::uint32_t>::max() - offset)
            throw std::invalid_argument(requirements.overflow_message);
    };
    for (auto offset :
         {buffers.reference_details, buffers.restored, buffers.additive})
        check_range(offset, 3 * stride);
    check_range(
        buffers.partials, requirements.partial_values_per_group * partial_count
    );
    check_range(buffers.results, requirements.result_words);
    Size size = input;
    for (std::size_t i = 0; i < 5; ++i) {
        const std::size_t pixels = size.width * size.height;
        check_range(buffers.reference[i], pixels);
        check_range(buffers.distorted[i], pixels);
        if (i != 4) size = VmafAdmWavelet::OutputSize(size);
    }
    const auto& limits = runtime.context().limits;
    const auto band = VmafAdmWavelet::OutputSize(input);
    if (Groups(band.width, 16) > limits.maxComputeWorkGroupCount[0] ||
        Groups(band.height, 16) > limits.maxComputeWorkGroupCount[1])
        throw std::invalid_argument("vmaf adm: geometry exceeds device limits");
    return stride;
}

// The four-scale dispatch schedule is shared; callers retain each model's
// buffer validation, shader selection, and exact CSF coefficient table.
void RecordAdmPyramid(
    VulkanRuntime& runtime, Size input, const VmafAdmBuffers& buffers,
    std::size_t stride, const std::array<VkPipeline, 4>& pipelines,
    const std::array<std::array<float, 2>, 4>& csf
) {
    Size size = input;
    AdmPush push{};
    push.reference = buffers.reference_details;
    // Fused wavelet/decouple consumes distorted detail bands in registers.
    // Keep the unused field to preserve the shared shader push-constant layout.
    push.distorted = 0;
    push.restored = buffers.restored;
    push.additive = buffers.additive;
    push.stride = static_cast<std::uint32_t>(stride);
    push.partials = buffers.partials;
    push.results = buffers.results;
    push.original_pixels =
        static_cast<std::uint32_t>(input.width * input.height);
    runtime.ComputeBarrier();
    for (std::uint32_t scale = 0; scale < 4; ++scale) {
        const WaveletDecouplePush wavelet{
            static_cast<std::uint32_t>(size.width),
            static_cast<std::uint32_t>(size.height),
            buffers.reference[scale],
            buffers.distorted[scale],
            buffers.reference[scale + 1],
            buffers.distorted[scale + 1],
            buffers.reference_details,
            buffers.restored,
            buffers.additive,
            static_cast<std::uint32_t>(stride)};
        size = VmafAdmWavelet::OutputSize(size);
        const auto gx = Groups(size.width, 16), gy = Groups(size.height, 16);
        push.width = static_cast<std::uint32_t>(size.width);
        push.height = static_cast<std::uint32_t>(size.height);
        push.scale = scale;
        push.groups = gx * gy;
        push.axial = csf[scale][0];
        push.diagonal = csf[scale][1];
        // The prior accumulator completed before its reduction began, so this
        // wavelet can reuse the bands while that reduction reads only partials.
        // The following barrier also orders that read before partials are
        // reused.
        runtime.Dispatch(pipelines[0], &wavelet, sizeof(wavelet), gx, gy, 1);
        runtime.ComputeBarrier();
        runtime.Dispatch(pipelines[1], &push, sizeof(push), gx, gy, 1);
        runtime.ComputeBarrier();
        runtime.Dispatch(pipelines[2], &push, sizeof(push), 1, 1, 1);
    }
    // The finish pass consumes every scale's reduced result.
    runtime.ComputeBarrier();
    runtime.Dispatch(pipelines[3], &push, sizeof(push), 1, 1, 1);
    runtime.ComputeBarrier();
}
}  // namespace

// ADM2 (v0.6.1)

VmafAdm::VmafAdm(VulkanRuntime& runtime, bool negative) : runtime_(&runtime) {
    const std::array names{negative ? "vmaf_adm_dwt_decouple_tiled_neg"
                                    : "vmaf_adm_dwt_decouple_tiled",
                           "vmaf_adm_accumulate", "vmaf_adm_reduce",
                           "vmaf_adm_finish"};
    pipelines_ = runtime_->CreatePipelines(names);
}

VmafAdm::~VmafAdm() { runtime_->DestroyPipelines(pipelines_); }

std::size_t VmafAdm::BandPixels(Size input) {
    if (input.width < 17 || input.height < 17)
        throw std::invalid_argument("vmaf adm: minimum geometry is 17x17");
    const auto band = VmafAdmWavelet::OutputSize(input);
    return band.width * band.height;
}

std::size_t VmafAdm::PartialCount(Size input) {
    (void)BandPixels(input);
    const auto band = VmafAdmWavelet::OutputSize(input);
    return static_cast<std::size_t>(Groups(band.width, 16)) *
           Groups(band.height, 16);
}

void VmafAdm::Record(Size input, const VmafAdmBuffers& buffers) {
    const auto stride = ValidateAdmBuffers(
        *runtime_, input, buffers,
        {.partial_values_per_group = 6,
         .result_words = 9,
         .overflow_message = "vmaf adm: arena offset overflow"}
    );
    RecordAdmPyramid(*runtime_, input, buffers, stride, pipelines_, kAdm2Csf);
}

// ADM3 (v1.0.16 3H)

namespace {
// Fixed blend coefficients for the pinned v1.0.16 3H model:
// libvmaf barten_csf_tools.h:130-133, revision 4611e47.
constexpr std::array<std::array<float, 2>, 4> kAdm3Csf{{
    {0.01183f, 0.004302f},
    {0.025026f, 0.011778f},
    {0.04295f, 0.023918f},
    {0.058621f, 0.035901f},
}};

}  // namespace

VmafAdm3::VmafAdm3(VulkanRuntime& runtime) : runtime_(&runtime) {
    constexpr std::array names{"vmaf_adm3_dwt_decouple", "vmaf_adm3_accumulate",
                               "vmaf_adm3_reduce", "vmaf_adm3_finish"};
    pipelines_ = runtime_->CreatePipelines(names);
}

VmafAdm3::~VmafAdm3() { runtime_->DestroyPipelines(pipelines_); }

void VmafAdm3::Record(Size input, const VmafAdmBuffers& buffers) {
    const auto stride = ValidateAdmBuffers(
        *runtime_, input, buffers,
        {.partial_values_per_group = 9,
         .result_words = 13,
         .overflow_message = "vmaf adm3: arena offset overflow"}
    );
    RecordAdmPyramid(*runtime_, input, buffers, stride, pipelines_, kAdm3Csf);
}

}  // namespace vqm::internal
