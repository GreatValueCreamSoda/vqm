// ColorVideoVDP: Copyright (c) 2024, Graphics and Displays group.
// MIT; see licenses/COLORVIDEOVDP-LICENSE. VQM tests: BSD-3-Clause.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "support.h"
#include "cvvdp.h"
#include "metrics/cvvdp/cvvdp_internal.h"
#include "vulkan/vulkan_runtime.h"

namespace {
using namespace vqm;
using namespace vqm::internal;

// Coefficients come from the pinned original CPU implementation. The probe
// bypasses display/spatial processing; expected FIR values use chronological
// samples rather than reproducing the shader's modulo ring addressing.
#include "cvvdp_reference.inc"

constexpr Size kProbeSize{7, 3};
constexpr std::uint32_t kPixels = 21;
enum class Waveform { Impulse, Step, Ramp };

float SignalSample(
    Waveform waveform, std::size_t frame, std::size_t taps, std::size_t side,
    std::size_t channel, std::size_t pixel
) {
    const float baseline = (side == 0 ? .4f : -.7f) +
                           .03f * static_cast<float>(channel) +
                           .005f * static_cast<float>(pixel);
    const float amplitude =
        (side == 0 ? .9f : -1.1f) * static_cast<float>(channel + 1) +
        .01f * static_cast<float>(pixel);
    float signal = 0;
    if (waveform == Waveform::Impulse)
        signal = frame == (side == 0 ? 0u : 2u) || frame == taps + 1 ? 1 : 0;
    if (waveform == Waveform::Step)
        signal = frame >= (side == 0 ? 1u : 2u) ? 1 : 0;
    if (waveform == Waveform::Ramp)
        signal = static_cast<float>(frame + 1) / static_cast<float>(taps);
    return baseline + amplitude * signal;
}

struct TemporalPush {
    std::uint32_t pixels, taps, current, available;
    std::uint32_t filters, reference, distorted;
};

struct ProbeLayout {
    std::uint32_t filters = 6 * kPixels;
    std::uint32_t reference, distorted, words;

    explicit ProbeLayout(std::size_t taps) {
        reference = filters + static_cast<std::uint32_t>(4 * taps);
        distorted = reference + 4 * kPixels;
        words = distorted + 4 * kPixels;
    }
};

class TemporalProbe {
public:
    TemporalProbe(std::span<const double> coefficients, MemoryPolicy policy)
        : taps_(coefficients.size() / 4),
          layout_(taps_),
          runtime_(AcquireVulkanContext(), Config(taps_, policy)),
          values_(layout_.words) {
        std::copy(
            coefficients.begin(), coefficients.end(),
            values_.begin() + layout_.filters
        );
        runtime_.CreateArena(Bytes(layout_.words));
        runtime_.CreateAuxiliaryStorage();
        if (!runtime_.arena_visible())
            runtime_.CreateStaging(Bytes(2 * layout_.words));
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, Bytes(layout_.words));
        pipeline_ = runtime_.CreatePipeline("cvvdp_temporal");
    }
    ~TemporalProbe() { runtime_.DestroyPipelines(std::array{pipeline_}); }

    std::vector<float> Push(Waveform waveform, std::size_t frame) {
        FillInputs(waveform, frame);
        float* upload = runtime_.arena_visible() ? runtime_.arena_map()
                                                 : runtime_.staging_map();
        std::copy(values_.begin(), values_.end(), upload);
        runtime_.ResetCommandPool();
        runtime_.BeginCommandBuffer();
        runtime_.BindDescriptorSet();
        PublishInputs();
        StoreCurrent(frame);
        const TemporalPush push{
            kPixels,
            static_cast<std::uint32_t>(taps_),
            static_cast<std::uint32_t>(frame % taps_),
            static_cast<std::uint32_t>(std::min(frame + 1, taps_)),
            layout_.filters,
            layout_.reference,
            layout_.distorted};
        runtime_.Dispatch(pipeline_, &push, sizeof(push), 1, 4, 1);
        if (runtime_.arena_visible())
            runtime_.ComputeToHostBarrier();
        else
            runtime_.RecordReadback(
                layout_.reference, layout_.words, 8 * kPixels
            );
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        const float* output = runtime_.arena_visible()
                                  ? runtime_.arena_map() + layout_.reference
                                  : runtime_.staging_map() + layout_.words;
        return {output, output + 8 * kPixels};
    }

private:
    static VulkanRuntime::Config Config(std::size_t taps, MemoryPolicy policy) {
        VulkanRuntime::Config config{};
        config.base_size = kProbeSize;
        config.base_pixels = kPixels;
        config.streaming = false;
        config.memory_policy = policy;
        config.push_constant_size = sizeof(TemporalPush);
        config.auxiliary_bytes.assign(2, Bytes(3 * kPixels * taps));
        return config;
    }
    void FillInputs(Waveform waveform, std::size_t frame) {
        for (std::size_t side = 0; side < 2; ++side) {
            for (std::size_t index = 0; index < 3 * kPixels; ++index)
                values_[side * 3 * kPixels + index] = SignalSample(
                    waveform, frame, taps_, side, index / kPixels,
                    index % kPixels
                );
        }
    }
    void PublishInputs() {
        if (runtime_.arena_visible()) {
            runtime_.HostToComputeBarrier();
            runtime_.Barrier(
                VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
            );
            return;
        }
        const std::array copies{VulkanRuntime::CopyRange{0, 0, values_.size()}};
        runtime_.RecordStagedCopies(copies);
        runtime_.Barrier(
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
        );
    }
    void StoreCurrent(std::size_t frame) {
        for (std::size_t side = 0; side < 2; ++side) {
            const VkBufferCopy copy{Bytes(side * 3 * kPixels),
                                    Bytes(frame % taps_ * 3 * kPixels),
                                    Bytes(3 * kPixels)};
            vkCmdCopyBuffer(
                runtime_.command_buffer(), runtime_.arena_buffer(),
                runtime_.auxiliary_buffer(side), 1, &copy
            );
        }
        runtime_.Barrier(
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT
        );
    }

    std::size_t taps_;
    ProbeLayout layout_;
    VulkanRuntime runtime_;
    std::vector<float> values_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

double ExpectedSample(
    std::span<const double> coefficients, Waveform waveform, std::size_t frame,
    std::size_t index
) {
    const std::size_t taps = coefficients.size() / 4;
    const std::size_t side = index / (4 * kPixels);
    const std::size_t channel = index / kPixels % 4;
    const std::size_t source_channel = channel == 3 ? 0 : channel;
    double expected = 0;
    for (std::size_t lag = 0; lag < taps; ++lag) {
        const std::size_t source_frame = frame > lag ? frame - lag : 0;
        expected += coefficients[channel * taps + lag] *
                    SignalSample(
                        waveform, source_frame, taps, side, source_channel,
                        index % kPixels
                    );
    }
    return expected;
}

void CheckWaveform(
    std::span<const double> coefficients, MemoryPolicy policy, Waveform waveform
) {
    TemporalProbe probe(coefficients, policy);
    const std::size_t taps = coefficients.size() / 4;
    // Every startup frame, both ring transitions, and the frames immediately
    // around each transition are checked, rather than only pooled clip JOD.
    for (std::size_t frame = 0; frame <= 2 * taps + 1; ++frame) {
        const auto actual = probe.Push(waveform, frame);
        for (std::size_t index = 0; index < actual.size(); ++index) {
            const double expected =
                ExpectedSample(coefficients, waveform, frame, index);
            CHECK_NEAR(
                actual[index], expected,
                std::max(2e-6, std::abs(expected) * 2e-6)
            );
        }
    }
}

void CheckFilters(std::span<const double> coefficients) {
    const auto requirements = GetMemoryRequirements(CvvdpConfig{.size = {8, 8}});
    for (const auto policy : {MemoryPolicy::Direct, MemoryPolicy::Staged}) {
        const auto& path = policy == MemoryPolicy::Direct ? requirements.direct
                                                          : requirements.staged;
        if (!path.supported) continue;
        for (const auto waveform :
             {Waveform::Impulse, Waveform::Step, Waveform::Ramp})
            CheckWaveform(coefficients, policy, waveform);
    }
}

Image ClipFrame(std::size_t frame, bool distorted) {
    auto image = vqm::test::MakeGradient(0);
    for (std::size_t index = 0; index < image.data.size(); ++index) {
        const std::size_t shift =
            (distorted ? 23 : 11) * frame + (distorted ? index % 13 : 0);
        image.data[index] = static_cast<std::uint8_t>(image.data[index] + shift);
    }
    return image;
}

CvvdpScore RunClip(
    CvvdpPipeline& pipeline, std::size_t frames, std::size_t offset = 0
) {
    for (std::size_t frame = 0; frame < frames; ++frame)
        pipeline.Push(
            ClipFrame(frame + offset, false), ClipFrame(frame + offset, true)
        );
    const auto result = pipeline.Finalize();
    CHECK(result.has_value());
    return result.value_or(CvvdpScore{});
}

}  // namespace

VQM_TEST(TestCvvdpTemporalFirStartupAndRing) {
    if (!vqm::test::GpuAvailable()) return;
    CheckFilters(kCvvdpFilters30_1);
    CheckFilters(kCvvdpFilters60000_1001);
}

VQM_TEST(TestCvvdpChangingTemporalIdentityIsExact) {
    if (!vqm::test::GpuAvailable()) return;
    CvvdpConfig config{.size = {32, 32}};
    CvvdpPipeline pipeline(config);
    const std::size_t frames = 2 * CvvdpTemporalTaps(config) + 3;
    for (std::size_t frame = 0; frame < frames; ++frame) {
        const auto image = ClipFrame(frame, false);
        pipeline.Push(image, image);
    }
    const auto score = pipeline.Finalize();
    CHECK(score.has_value());
    if (score) {
        CHECK(score->frames == frames);
        CHECK(score->score == 10.0);
    }
}

VQM_TEST(TestCvvdpResetDiscardsPendingAndWrappedHistory) {
    if (!vqm::test::GpuAvailable()) return;
    const CvvdpConfig config{.size = {32, 32}};
    CvvdpPipeline pipeline(config);
    const auto old_clip = RunClip(pipeline, 2 * CvvdpTemporalTaps(config) + 3);
    CHECK(old_clip.score < 10.0);
    pipeline.Reset();
    const auto actual = RunClip(pipeline, 3, 7);
    CvvdpPipeline fresh(config);
    const auto expected = RunClip(fresh, 3, 7);
    CHECK(actual.frames == expected.frames);
    CHECK_NEAR(actual.score, expected.score, 1e-7);
    pipeline.Reset();
    pipeline.Push(ClipFrame(1, false), ClipFrame(1, true));
    pipeline.Reset();
    const auto pending_reset = RunClip(pipeline, 3, 7);
    CHECK_NEAR(pending_reset.score, expected.score, 1e-7);
}

VQM_TEST(TestCvvdpResidentReleaseAndReusePreservesHistory) {
    if (!vqm::test::GpuAvailable()) return;
    CvvdpConfig config{.size = {32, 32}};
    config.streaming_inputs = false;
    CvvdpPipeline resident(config);
    const std::size_t frames = 2 * CvvdpTemporalTaps(config) + 3;
    for (std::size_t frame = 0; frame < frames; ++frame) {
        const auto original = resident.Upload(ClipFrame(frame, false));
        const auto distorted = resident.Upload(ClipFrame(frame, true));
        resident.Push(original, distorted);
        resident.Release(original);
        resident.Release(distorted);
    }
    const auto actual = resident.Finalize();
    config.streaming_inputs = true;
    CvvdpPipeline streamed(config);
    const auto expected = RunClip(streamed, frames);
    CHECK(actual.has_value());
    if (actual) {
        CHECK(actual->frames == expected.frames);
        CHECK_NEAR(actual->score, expected.score, 1e-7);
    }
}
