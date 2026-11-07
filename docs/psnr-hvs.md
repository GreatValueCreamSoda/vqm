# PSNR-HVS

`PsnrHvsPipeline` computes an FP32 adaptation of libvmaf/Xiph PSNR-HVS on
Vulkan. Its combined and Y/Cb/Cr scores are in dB; larger values mean better quality.
This variant includes contrast masking and differs from the authors' original
PSNR-HVS definition. The implementation follows the pinned libvmaf reference
`4611e47a7330cd342498dcc130e1008efbeab6e4` and retains Xiph's license notice.

## Native input contract

Inputs are opaque planar YUV420, YUV422 or YUV444. Eight-bit samples use
`PixelType::Byte`; 10-bit and 12-bit samples use `PixelType::Word`. Reference
and distortion must have matching dimensions and declared image formats.
Samples must fit their declared depth. RGB, grey, alpha, floating-point inputs
and other depths are unsupported.

The metric unpacks native samples into exact FP32 code values. It does not
expand limited range, apply a transfer function, change color primaries or
resample chroma.
Color metadata still participates in format validation and pair matching.
Normalization uses the declared depth's maximum code value, including when
the samples use limited range.

Each scored plane must be at least 8x8. Minimum image sizes are 16x16 for
4:2:0, 16x8 for 4:2:2 and 8x8 for 4:4:4. Dimensions must be divisible by their
chroma subsampling; they need not be multiples of 8 or 7. Complete 8x8 windows
are visited with stride 7. Pixels beyond the final complete window are ignored.

## Standalone API

```cpp
#include <psnr_hvs.h>

// original and distorted contain matching native integer YUV planes.
vqm::PsnrHvsPipeline pipeline({.size = original.size});
vqm::PsnrHvsScore quality = pipeline.Compute(original, distorted);
double combined_db = quality.score;
double luma_db = quality.y;
```

`PsnrHvsScore` also exposes `cb`, `cr` and `normalized_error`, whose entries
are the normalized Y, Cb and Cr errors. The combined error is
`0.8 * Y + 0.1 * (Cb + Cr)` before conversion to dB. A zero error produces
positive infinity. This can occur for identical images, fully masked
differences or differences confined to uncovered border pixels. Individual
channel scores can be infinite while the combined score is finite. Scores
are not capped at 100 and can be negative.

The pipeline supports owning images, strided `ImageView` inputs, resident
uploads, caller-filled `ImageBuffer` allocations and borrowed
`VulkanBufferView` inputs. `Compute` is `Submit` followed by `Finish`.
There is one submission in flight per pipeline; finish it before uploading,
mapping, allocating or releasing input storage. The pipeline is not
thread-safe. Borrowed devices, buffers and synchronization follow
[the Vulkan interoperation contract](vulkan-synchronization.md).

`PsnrHvsConfig` controls geometry, resident slots, caller-input pool size,
streaming source storage and `MemoryPolicy`. `GetMemoryRequirements` uses
the construction layout, and `GetMemoryUsage` reports the actual retained
allocations. Prepared native planes and reduction scratch are private to the
pipeline. No temporal history is retained.

All metric arithmetic uses FP32, including the separable orthonormal 8x8 DCT,
variances, masking, weighted errors, reductions, normalization, plane mixture
and logarithms. Input storage does not select another processing precision.
The public `double` score and error fields hold promoted finished FP32 values;
they do not select FP64 computation. Integer operations describe storage,
sample unpacking, geometry and indexing.

## Combined metrics and video

Set `MetricsConfig::psnr_hvs` to include the combined score in
`MetricsResult::psnr_hvs`. The top-level input formats must meet the native
contract above. Top-level geometry, streaming and memory settings override
the leaf configuration; resident slots and the caller-input pool are disabled.
PSNR-HVS owns an independent allocation, reported by the `psnr_hvs` memory
entry. Its frame-local score remains associated with its input frame when
another selected metric needs lookahead. See [the unified API](metrics.md).

The normal CLI accepts native Y4M images and progressive fixed-rate clips:

```sh
./build/vqm --metric psnr-hvs original.y4m distorted.y4m
./build/vqm --metric psnr-hvs --json original.y4m distorted.y4m
./build/vqm --metric psnr-hvs --json --pairs-file clips.tsv
```

Every frame is evaluated. As for the other spatial metrics, the clip summary
is the arithmetic mean of the frame dB scores. One infinite frame therefore
makes the clip summary infinite. JSON represents positive infinity as the
string `"inf"`; ordinary finite scores remain numbers. Clip geometry, frame
counts and rational timing must match. The metric adds no temporal filtering
or clip-finalization state.

## Reference relationship

The implementation retains the reference's CSF tables, masking, block sampling
and plane mixture. It uses an orthonormal FP32 DCT instead of libvmaf's rounded
integer transform, parallel reductions instead of its serial FP32 fold, and
FP32 throughout the remaining metric arithmetic. Final scores need not be
identical.
Future qualification will measure direct ranking agreement against the
pinned reference on frozen input domains. Ranking qualification and validation
tool integration have not yet been performed; no corpus-wide ranking agreement
is claimed by this addition.

The original native-input and FP32 requirements are recorded in the
[implementation plan](design/psnr-hvs-implementation-plan.md).

The local `vqm_test` suite includes independent closed-form DC and
single-frequency masking checks, all supported depth/subsampling combinations,
identity, staged and resident inputs, reduction, frame alignment and lifecycle
contracts in [the PSNR-HVS tests](../tests/metrics/psnr_hvs/pipeline_test.cc).
These component checks do not establish corpus ranking agreement.

The [resident throughput report](../bench/notes/psnr_hvs/resident-throughput-20261106.md)
records five-run 1080p/4K timing, spread, stages and allocation provenance.
The subsequent [optimization report](../bench/notes/psnr_hvs/optimization-20261106.md)
records the interleaved timings, the matched SSIMULACRA 2 comparison and the
qualification limits of that experiment.
