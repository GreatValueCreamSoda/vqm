# Unified metrics API

`vqm::Metrics` configures the inputs and selected metrics once. It returns the
selected scores together, supports still images and ordered sequences, and can
consume owning images, strided host views, tightly packed byte spans or
caller-owned Vulkan buffers.

```cpp
#include <vqm_metrics.h>

// original and distorted are decoded vqm::Image objects of the same size.
vqm::MetricsConfig config;
config.size = original.size;
config.original_format = original.format;
config.distorted_format = distorted.format;
config.ssimulacra2 = vqm::Ssimulacra2Config{};
config.butteraugli = vqm::ButteraugliConfig{};

// VMAF can be enabled too. Its explicit chroma mode allows different SDR
// source formats for reference and distorted images.
vqm::VmafV1Config vmaf;
vmaf.chroma_sampling = vqm::VmafChromaSampling::Yuv420;
config.vmaf_v1 = vmaf;

vqm::Metrics metrics(config);
auto result = metrics.ComputeImage(original, distorted);
// result.ssimulacra2, result.butteraugli and result.vmaf_v1 hold scores.
// Fields for metrics that were not selected are empty.
```

`ComputeImage` treats the pair as an independent one-frame sequence. It
performs VMAF's final flush and resets automatically. It rejects a call during
an active sequence; call `Reset` before switching to still-image processing.

## Video sequences

Call `Compute` for each frame pair in order. It returns a result once every
selected frame metric has completed that frame. Each result carries
`frame_index`. VMAF can defer results for motion lookahead. Call `Finalize`
after the final pair to collect any pending frame and the CVVDP clip score,
then call `Reset` before starting another sequence.

```cpp
for (const auto& pair : frames) {
    if (auto result = metrics.Compute(pair.original, pair.distorted)) {
        consume(result->frame_index, *result);
    }
}
const auto final = metrics.Finalize();
if (final.last_frame) {
    consume(final.last_frame->frame_index, *final.last_frame);
}
// final.cvvdp holds the clip score when ColorVideoVDP is enabled.
metrics.Reset();
```

The geometry and the two source formats are fixed by the configuration.
`Compute` checks each input against its side's format. For tightly packed input,
pass `std::span<const std::byte>` for each side: its size must equal
`RequiredBytes(format, size)`. No separate format descriptor is needed on the
call. For strided planes use `ImageView`; for GPU-resident inputs use
`VulkanBufferView` and construct `Metrics` with the caller's `VulkanDevice`.
The normal buffer-lifetime, queue ownership and synchronization requirements
in [vulkan-synchronization.md](vulkan-synchronization.md) apply.

The `VulkanSubmitSync` overload consumes its binary wait semaphores once on
the first metric and submits signals with the last. A failure after work has
started can leave waits consumed and signals unscheduled; the caller must not
assume a completion signal will be submitted on failure. A signal indicates GPU
completion, not successful score conversion. If a call fails during metric
processing, including backend validation of a borrowed buffer, `Reset` is
required before retrying. Errors caught by the initial input checks leave the
sequence unchanged.

## Configuration and memory

The top-level size, source formats, `streaming_inputs` and `memory_policy`
apply to all selected metrics. Alpha `background` applies where supported;
PSNR-HVS requires opaque inputs. The underlying metric
configurations retain their metric-specific settings, such as Butteraugli's
intensity target and VMAF's chroma selection. Their own geometry, streaming,
memory-policy, resident-slot and input-pool fields are overridden: this API
accepts caller images/views or external buffers rather than per-pipeline
resident handles. VMAF's background and v1 input format are also taken from
the top-level input configuration. `Native` VMAF v1 requires the two configured
formats to match; explicit 4:2:0 or 4:4:4 supports different source formats.

With both SSIMULACRA 2 and Butteraugli enabled, their temporary arena is shared
and the metrics run sequentially. VMAF v0, VMAF v1, CVVDP and PSNR-HVS retain
their own allocations; their scratch is not part of that shared arena.
PSNR-HVS has no temporal history. A single selected metric gets only its own
pipeline. A `Metrics` object
is one worker and is not thread-safe.

Host inputs are uploaded separately for each selected metric. To reuse an
already-uploaded pair without those copies, provide caller-owned
`VulkanBufferView` inputs on a borrowed device.

`GetMemoryRequirements(config)` from `vqm_metrics.h` queries the selected
spatial, VMAF, CVVDP and PSNR-HVS allocation requirements before construction;
`metrics.GetMemoryUsage()` reports the actual corresponding allocations. The
spatial entry counts shared scratch once. `MaxWorkersForBudgetWorstCase` accepts
these requirements and per-heap budgets, counting each selected component.
Leave room for external buffers and driver resources.

## PSNR-HVS frame scores

Set `config.psnr_hvs = vqm::PsnrHvsConfig{}` to populate `result.psnr_hvs`
with the combined score in dB. Both configured inputs must have matching opaque
native YUV420, YUV422 or YUV444 formats at 8, 10 or 12 bits. The constructor
rejects unsupported or mismatched formats before creating the metric pipelines.
Raw code values are preserved; there is no RGB conversion or range expansion.

PSNR-HVS returns each frame immediately unless another selected metric needs
lookahead. Its result stays aligned with the corresponding VMAF frame, and
`Flush` has no additional PSNR-HVS computation. Positive infinity is a valid
zero-error score. The `psnr_hvs` memory entry counts its independent pipeline;
resident-slot and input-pool settings are overridden by the top-level API as
for the other spatial metrics. See [PSNR-HVS](psnr-hvs.md) for the standalone
API, geometry requirements and current qualification status.

## ColorVideoVDP clip scores

Set `config.cvvdp = vqm::CvvdpConfig{}` and configure its physical display and
rational frame rate. Independent `ComputeImage` populates `result.cvvdp`.
Sequence `Compute` leaves that field empty because a frame JOD does not
represent clip quality. Call `auto final = metrics.Finalize()` after the last
pair: `final.last_frame` contains any remaining VMAF result, `final.cvvdp` the
whole-clip JOD/frame count, and `final.frames` the consumed pair count. Repeated
finalization repeats only the cached clip summary. Existing `Flush` remains
usable; a following `Finalize` retrieves the clip summary. A CVVDP-only sequence
emits frame-index metadata from `Compute`. See [CVVDP](cvvdp.md) for image/video
selection, supported inputs and temporal memory limits.

CVVDP retains private histories rather than sharing the spatial workspace.
Both memory APIs expose its arena, staging and individual auxiliary allocations;
worker budget helpers count each allocation against its own heap.
