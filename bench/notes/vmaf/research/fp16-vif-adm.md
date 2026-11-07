# VIF and ADM FP16 practicality review

> Historical research note. The [FP16 practicality report](../vmaf-fp16.md) records the final
> measured result and retained implementation. Current source may differ from
> the reviewed snapshots. Relative experiment files and commands below refer
> to the archived research directory `artifacts/reports/vmaf-fp16/experiments/`.

This is numerical/source analysis and CPU/compiler evidence. Root owns GPU
measurements. No production source was edited by this review.

## Experiments supplied

- `vif-half-products/candidate.patch`: actual FP16 sample squares/cross products,
  promoted before FP32 weighted filtering, variance/covariance, logarithms and
  reductions. Covers fused scale0, fused1–3 and two-pass vertical fallback.
- `adm-half-products/candidate.patch`: scalar FP16 coefficient/sample products
  with FP32 dot sums, output arena, decoupling, masking, cubes and reductions.
- `adm-paired-fp32/candidate.patch` and `adm-paired-fp16/candidate.patch`: reference
  and distortion are paired in two independent vector lanes. The FP32 control
  isolates scheduling/vectorization from the precision change. The FP16 version
  has six `OpFMul %v2half` sites; the control has none. All four source/gain
  specializations compile and pass `spirv-val` for each.
- `vif-adm-numerics.py/.json`: reproducible CPU range and rounding diagnostics.
- `../inputs/constant-vif10-{ref,dist}.y4m` and the corresponding `plateaus-vif10`
  pair: 192×160 native 10-bit inputs with neutral chroma and Y codes 914/913. The
  latter adds an opposite-sign 8×8 patch. Manifest includes input hashes.

The SPIR-V proves these candidates contain FP16 arithmetic, not merely casts at
FP32 operation boundaries. It does not prove packed native ISA throughput or
Tensor Core use. Candidate arena/shared layouts stay FP32 except the already
existing exact scale0 source tile; no memory-allocation or bandwidth saving is
claimed from arithmetic-only experiments.

## VIF: FP32 accumulation alone is insufficient

Native coded luma is `code / 2^(depth-8) - 128`. Every 8/10/12-bit centered sample
is exactly representable in binary16. Native 16-bit source values can lose 0.03125,
eight source codes, before any filtering. The current implementation correctly
uses its FP16 source tile only for eligible native inputs, then promotes values
before moment arithmetic.

Variance is `E[x²] - E[x]²`. At luma 100.5, the squared value is 10100.25, where
FP16 spacing is 8. Constant ref 100.5/dist 100.25 correspond to valid native 10-bit
Y codes 914/913:

| product | exact | binary16 |
| --- | ---: | ---: |
| reference square |10100.25|10104|
| distortion square |10050.0625|10048|
| cross product |10075.125|10072|

The CPU diagnostic retains actual FP32 17-tap filter weights, separate FP32
weighted adds and the horizontal pass. Ordinary FP32 yields VIF≈0.99999994.
FP16 products plus FP32 accumulation yield reference variance 3.75098 (> the 2.0
branch threshold), clamped distortion variance 0 and covariance −3.12402; VIF
becomes 0. This is product rounding before accumulation, so a wide accumulator
cannot repair it. GPU contraction can alter low FP32 bits, but the gap in this
counterexample is much larger than those bits.

`1e-10` becomes zero in binary16. The current early low-variance return avoids
some old zero-denominator cases, so that fact alone is not a proof of NaN in
current VIF. It nevertheless changes the residual clamp and zero-variance
protection semantics. Gain/residual/logarithms and their safeguards should stay
FP32. With gain up to100, `gain² * s1` may also overflow before a wider reduction.
Full-image numerator/denominator sums require FP32 independently.

## Recentering

Variance/covariance are shift-invariant in exact arithmetic when every sample
in a filtered window uses one constant per image. Subtracting the center in FP32
and dividing residuals by 2 bounds the residual square to 16384; variances and
covariance must be multiplied by 4 in FP32 before the existing thresholds and
noise constants. A power-of-two scale does not fix relative precision.

Per-tile centering fixes uniform fields but offers no general protection for
heterogeneous tiles. A tile containing opposite-sign values can have center 0
with first-sample or min/max centering, while distant output windows see only
the constant 100.5/100.25 plateau. Those windows reproduce VIF 0 above, including
after residual/2 scaling. One tile center cannot keep every local plateau near
zero. Source residual casts can also lose low input bits at the large end.

Per-window or per-column-window recentering is a more substantial alternative:
compute FP32 deltas and delta means, use FP16 products for small residuals, then
reconstruct original moments with FP32 cross terms. It adds FP32 work and
changes rounding. Column-dependent centers cannot be fed unchanged into the
current horizontal moment filter. A fallback guard would need to account for
product-error bounds, the variance 2 branch, near-zero covariance/residual and
feature/model sensitivity. This is a new algorithmic experiment, not a safe
one-line cast or a proven speedup.

## ADM: keep more than the final accumulator wide

The four-tap wavelet's local products can be evaluated in FP16 without immediate
range overflow for normal coded inputs; they still introduce coefficient and
cancellation error. On the constant 100.5 CPU field, coarsest detail magnitude
moves from≈0.00004316 to≈0.08838 with FP16 products and FP32 dot sums. This is a
fabricated edge, even though the accumulator is FP32.

Keep signed band decoupling in FP32: its 1e-30 guard becomes 0 in FP16; zero
reference/distorted detail can then evaluate 0/0. Dot products, squared magnitude
comparisons and the angular threshold are sensitive too. The threshold
0.9996954135 rounds to0.99951171875 in binary16, changing the enhancement-gain
branch. ADM2 gain 100 and NEG/ADM3 gain 1 must retain their distinct behavior.

FP16 cubes can overflow before accumulation. A valid 256×256 native 10-bit image
of 16-pixel stripes between −128 and127.75 produces a coarsest CSF-weighted
magnitude≈81.1483, whose cube is≈534366. A 16-pixel checkerboard gives≈89.6187,
cube≈719774. Both exceed 65504; a FP32 accumulator cannot recover an already
infinite FP16 cube. Mask sums, norm products, partial reductions and final
ratios should remain FP32. `vif-adm-numerics.json` retains all four scales and
FP32, mixed-product and all-FP16 CPU results.

The prior `vmaf-v0-optimization.md` pass already rejected FP16 ADM intermediate
planes as neutral/slower. The supplied experiments change actual arithmetic;
they intentionally do not repeat that storage-only candidate.

## GPU and test requirements

All arithmetic candidates require enabled `shaderFloat16`. That is distinct
from 16-bit buffer storage. Replacing existing blobs is appropriate only in an
isolated experiment with a supported owned device. Production would require
capability-selected shaders and an unchanged portable FP32 fallback, including
borrowed-device feature promises. Packed arena formats would additionally need
layout/indexing, memory-query and ownership work; these patches do not supply it.

Accuracy screens must include `TestVmafVifAgainstLibvmaf` (8/10/12-bit lowbits),
`TestVmafVifOddGeometryAndLowVariance`, near-black/high-contrast and fused/fallback
checks; ADM oracle, identity, odd-size, NEG and ADM3 high-bit/moving cases; and
full model/corpus scoring. A correlation pass alone cannot hide the constant
VIF failure or a changed singularity/status branch. Keep existing oracle
thresholds and report approximation separately if the experimental path fails.
