# VMAF v1 precision review

> Historical research note. The [FP16 practicality report](../vmaf-fp16.md) records the final
> measured result and retained implementation. Current source may differ from
> the reviewed snapshots. Relative experiment files and commands below refer
> to the archived research directory `artifacts/reports/vmaf-fp16/`.

This review prepared isolated shader files; it did not modify maintained
production sources, run GPU jobs, or change test oracles. Numerical witnesses
in `v1-fp16-numeric-witnesses.json` are CPU IEEE round-to-nearest simulations,
not measurements of the Vulkan driver's default floating-point modes.

## Experimental arms ready for root's gates

- `cambi-half-shared/candidate.patch`: only CAMBI's shared source symbol tile
  changes from uint to float16_t. Exact symbols are 0..497; every read converts
  explicitly back to uint. Integer histograms/atomics and FP32 contrast/maps/
  reductions stay unchanged. Language-layout shared allocation falls from
  16,004 to 12,164 bytes, saving 3,840 bytes per group. No arena storage changes.
- `speed-half-products/candidate.patch`: both separable filter kernels use
  explicit FP16 coefficients, samples and products, promoted before each FP32
  running sum. High-pass subtraction stays FP32. All four logical passes are
  affected; there is no arena compression.
- `speed-half-covariance-products/candidate.patch`: means and centered
  subtractions remain FP32. Deviations/products become FP16; local and shared
  accumulation, matrix outputs, solvers and entropy remain FP32.

Each arm includes the baseline source, compiled SPIR-V and disassembly. All
compile with glslc -O for Vulkan 1.1 and pass spirv-val. Arithmetic arms have
OpFMul on the 16-bit type and no OpFAdd on that type: they perform actual FP16
arithmetic, rather than only storing a half and promoting it before arithmetic.
This does not prove which native GPU instructions the driver ultimately emits.

## Why pure FP16 is unsuitable for the existing numerical contract

SpEED's covariance eigensystem classifies a matrix as singular below 1e-6.
Householder/QR and the guarded bisection path also use relative 1e-6 tests,
1e-37 Sturm pivots, and a 1e-6 + 1e-3*spectral_bound fallback band. 1e-37 becomes
zero in FP16, and 1e-6 is subnormal. Cholesky subtracts nearly equal terms;
its factors feed a forward solve and entropy estimate. Keeping only the final
sum FP32 cannot recover detail already lost in products, matrix storage or
factorization.

Concrete witnesses:

- Matrix [[1,.9999],[.9999,1]] has a smallest FP32 eigenvalue near 1.00017e-4;
  rounding its entries to FP16 makes all four entries one and the minimum zero.
- A deviation .0009994999 squares to 9.990001e-7 in FP32, below the singular
  cutoff. Its FP16 product promoted to FP32 is 1.013279e-6, above the cutoff.
- The existing tests deliberately include eigenvalues just above and below
  1e-6, spectra spanning twelve orders, dense nearly singular matrices, exact
  rank deficiency, overflow/nonfinite inputs, and low-amplitude image cases.

CAMBI is predominantly an integer algorithm. A 33x33 window has at most 1,089
samples, so individual small local counts fit FP16 exactly, but its packed
integer atomic updates are already exact and compact. Global histogram counts,
selection counts and pooled sums need their existing integer/FP32 types. With
p0=544, pother=545 and d=4, the contrast numerator is 1,185,920: a literal
FP16 product overflows although the final contrast is only 1,088.999. Reordering
normalization could avoid overflow but changes rounding and does not accelerate
the dominant integer histogram work by itself.

An original 16-bit luma code 6367 produces FP32 centered luma -103.12890625,
then CAMBI's correct 10-bit code 99. Rounding that luma to FP16 first gives
-103.125, reconstructs original code 6368, and changes the 10-bit result to 100.
That can change mode/equality/mask/bin decisions before any accumulation.
Current VIF's bounded native <=12-bit shared-half path does not make this error.

## Filter approximation and possible mitigation

The nine antialias coefficients sum to approximately one in the reference;
their independently rounded FP16 values sum to .9998779297. The mean kernel's
rounded sum is 1.0002441406. A constant -128 input run through the four passes
with FP16 products and FP32 sums produces a simulated high-pass residue .046875,
versus about -1.53e-5 using separate FP32 multiply/add operations. A final flat
image can still be imputed singular; this witness concerns the raw signal, not
a claim about the final flat-image model score. For nearly flat content, bias
and quantization can affect covariance and the raw-vector variance solve.

If this arm's error is dominated by DC bias, a centered filter formulation
anchor + sum(weight*(sample-anchor)) could preserve constant inputs exactly,
while retaining FP32 accumulation. It adds subtraction/conversion work, still
quantizes differences/products and needs its own tests/performance gate; it is
not implemented or claimed faster here. Simply fixing FP32 accumulation alone
does not solve coefficient or product quantization.

## Portability and performance limits

All experimental Float16 shaders require the enabled shaderFloat16 feature.
That covers Function/Private/non-Block Workgroup use, not StorageBuffer
interfaces; whole-arena 16-bit storage needs separate storage features or
explicit packed-word addressing. Preserve the public borrowed-device promise
and choose an FP32 fallback when Float16 is not enabled.

Float-controls properties separately describe half denorm preservation/flush
and RTE/RTZ support. Do not treat the CPU numerical witnesses as a cross-driver
default-mode guarantee. Relevant primary documentation:

- https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceShaderFloat16Int8Features.html
- https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceFloatControlsProperties.html

The preceding final benchmark put all SpEED work at about 5% of the native 4K
frame and 11.5% at 1080p. Even a hypothetical 2x acceleration of every SpEED
operation would reduce those frame times by only about 2.5%/5.8%; actual mixed
filter/covariance arms affect a smaller fraction. CAMBI shared compression
instead targets a larger stage without numerical approximation, but occupancy,
conversion costs and native layout must be measured rather than inferred.

Covariance is mathematically a matrix product. A future cooperative-matrix
FP16-input/FP32-accumulator implementation would require separately enumerated
matrix types/dimensions and enabled extension/features; scalar float16_t code
does not automatically promise tensor-core execution. Packing, a 25-row matrix,
conditioning checks and rerunning sensitive cases in FP32 complicate the cost
model. No implementation or hardware-support claim is made here.

Primary API: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html
