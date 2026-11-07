# Platform and motion FP16 review

> Historical research note. The [FP16 practicality report](../vmaf-fp16.md) records the final
> measured result and retained implementation. Current source may differ from
> the reviewed snapshots. Relative experiment files and commands below refer
> to the archived research directory `artifacts/reports/vmaf-fp16/motion/`.

No GPU jobs were run by this review agent. Production files were not edited.

## Feature and storage contract

The existing owned context (`src/vulkan/vulkan_support.cc`, Inspect/CreateContext)
queries and enables `shaderFloat16` separately from `storageBuffer16BitAccess`.
Its Vulkan 1.1 instance explicitly enables `VK_KHR_shader_float16_int8` when
using Float16. Borrowed contexts validate physical support only when the caller
sets the corresponding enabled-feature promise; both promises default false.
Physical support cannot be used to infer which features an existing VkDevice
enabled. Current fallback behavior is correct and must remain.

Khronos distinguishes [FP16 shader operations/shared/function storage](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceShaderFloat16Int8Features.html)
from [16-bit interface storage](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDevice16BitStorageFeatures.html).
These motion prototypes need only the former: shared arrays and function-local
values are half, while the SSBO stays float32. A real compact arena would need
16-bit interface-storage support and layout changes, or explicit packed pairs
in uint32 words. Such a layout must update offset/stride planning, memory
requirements, producer/consumer shaders and ownership of shared packed words.
Casting a value to half then storing it into `arena[]` still uses four bytes.

No maintained code currently requests floating-point execution modes or queries
float-control properties. Subnormal preservation and explicit rounding modes
are optional, precision-specific [float-control properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceFloatControlsProperties.html).
A portable half path cannot assume FP16 denormals survive merely because
shaderFloat16 is enabled. Adding prescribed modes requires checking support
and enabling the relevant core/extension contract, including borrowed-device
capabilities. `precise` prevents certain contractions; it does not provide
extra mantissa bits, a larger exponent range or a denormal guarantee.

## Input ranges and conversion ordering

Native feature luma is `sample * 2^(8-depth) - 128`:

| native depth | feature interval | code step | exactly half-representable after centering? |
| --- | --- | --- | --- |
| 8 | [-128,127] | 1 | yes |
| 10 | [-128,127.75] | 1/4 | yes |
| 12 | [-128,127.9375] | 1/16 | yes |
| 16 | [-128,127.99609375] | 1/256 | no |

The exactness claim concerns already-centered values. Converting raw12-bit
4095 to half first rounds to4096; scaling and subtracting128 then produces128
rather than127.9375. Centering in integer or FP32 before the half cast avoids
that loss. A raw16-bit endpoint65535 also exceeds FP16's finite maximum65504.

At magnitudes in [64,128), half spacing is1/16. Native16-bit data has sixteen
coded steps per half ULP and can lose eight code steps to rounding. Near zero
resolution is finer; loss is content/range dependent. Canonical SDR is *also*
affected: `vmaf_sdr.comp` emits roundEven(value*256)/256 -128, retaining16-bit
fractional detail, not merely10-bit codes. It must not inherit the native<=12
exactness claim. Fit/pad and filtered planes are likewise not restricted to
exactly representable native-code grids.

## Motion variants prepared

All patches live under this directory and select an experimental shader only
when `runtime.context().shader_float16` is true. No public API change. The
portable default shader is byte-identical to the current FP32 module.

1. `input-shared-half/candidate.patch`: only the input tile is FP16. All blur,
   difference and reduction arithmetic is FP32. Shared storage3904→3040B.
   It can preserve native<=12 inputs exactly but loses canonical/16-bit detail.
2. `all-shared-half/candidate.patch`: input and vertical-result tiles FP16;
   all arithmetic remains FP32. Shared3904→2464B. Vertical quantization affects
   every source depth, even where the initial input cast is exact.
3. `fp16-blur-fp32-reduce/candidate.patch`: both five-tap passes multiply and
   accumulate in genuine FP16, with FP32 difference, partials, final reduction
   and float32 arena/history output. Shared3904→2464B. SPIR-V contains16-bit
   FAdd/FMul; the other two variants have none. `codegen.json` records this.

4. `fp16-products-fp32-sums/candidate.patch`: rounded half tap products are
   promoted before local FP32 sums; all reductions stay FP32. Codegen has two
   16-bit FMul sites and no16-bit FAdd. Shared storage2464B. This does not imply
   a tensor-core/mixed-precision fused multiply-accumulate instruction.

All four shaders compiled and passed spirv-val for Vulkan1.1. This is static
validation, not a claim that score or performance gates pass. The existing
source offsets, halo mirror, barrier schedule and group layout are preserved.
No arena allocation or upload traffic is reduced by these prototypes.

## Precision risks and targeted fixtures

FP32 final accumulation does not restore blur/input information discarded
beforehand. Illustrative sequential NumPy FP16 arithmetic in
`cpu-precision-examples.json` shows:

- Native16 constant frames127→127+1/256 become the same half input. Every half
  variant returns zero transition instead of0.00390625, even input-storage-only.
- Native12 constant127→127+1/16 produced transition0.125 with half blur versus
  FP32 0.0625. Native8 constant126→127 produced0.9375 versus~0.999985.
- Half-only vertical storage on random translated native8/10/12 arrays gives
  nonzero motion drift despite exact initial input storage.

These are CPU illustrations, not GPU-oracle measurements: GPU FMA contraction
can change the particular values. With half products but FP32 tap sums, the
same12-bit constant fixture gives0.0703125 versus0.0625: FP32 accumulation
cannot recover information lost by rounding products/coefficient values. They identify high-value moving-frame cases
for the root's GPU tests. A repeated resident pair has zero motion after the
first call and cannot establish temporal precision.

The converted motion weights sum to0.99981689453125 in FP64, so even the
coefficient set loses the original near-unity sum. FP16 arithmetic or vertical
storage can move both motion2 and motion3; min-neighbor selection and v1's
RBF feature-distance/polynomial predictor can amplify temporal differences or
change which neighbor supplies the minimum. First/last-frame rules remain model-specific.

Pure FP16 totals are not viable without rescaling/redesign. At4K a total
absolute difference exceeds65504 whenever mean motion exceeds about0.0079.
Even a256-pixel group can exceed the range for native16-bit extreme transitions.
Casting width*height to half also overflows at modest image sizes. FP32
reductions are mandatory for the current formulation, but do not by themselves
make the preceding FP16 blur accurate.

## Recommended gates

Compare each variant independently against the frozen baseline. Use actual
moving sequences, native8/10/12/16 and canonical RGB, tiny/odd tile boundaries,
nearblack/highlight one-code changes, and constant extremes. Compare feature
motion and final model scores, not only MCOS or repeated-frame checksums.
Keep existing tolerances unchanged. Run fallback/no-shaderFloat16 borrowed
contexts, synchronization validation and cross-device float-control checks.

For performance, separate shared-input-only from vertical quantization and
true ALU changes. Preserve explicit 5-run interleaving and inspect full frame
and motion-stage medians. The original3904B shared block is already small;
less shared storage need not change occupancy. Conversion/shuffle/packing costs
can erase ALU savings. No throughput factor can be inferred solely from the
presence of16-bit SPIR-V operations.
