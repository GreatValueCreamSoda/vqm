# VMAF feature mapping to FP16-input, FP32-accumulating matrices

Source and CPU numerical investigation, 2026-11-05. This note covers both
VMAF versions outside the v1 SpEED covariance prototype. It makes no new GPU
timing claim and does not qualify a replacement implementation. The
[earlier FP16 report](../vmaf-fp16.md) distinguishes scalar half-rounded
products from a wide matrix product; that distinction is preserved here.

Reproduce the CPU diagnostics from any working directory:

```sh
python3 /path/to/repository/bench/experiments/cooperative_matrix/feature_numerics.py
```

The script supports `--out PATH` and otherwise writes
`artifacts/reports/vmaf-cooperative-matrix/feature-mapping/numerics.json`.
[Compact results](../../measurements/vmaf-cooperative-feature-numerics.json)
retain coefficient/source hashes, numerical examples and historical stage
shares. NumPy 2.5.4, seed 20261105 and the maintained shader coefficients were
used. The script contains no timers and does not use the GPU.

## What the wider product fixes

Two finite binary16 values have a product representable in FP32 before the
FP32 sum is rounded: their product needs at most 22 significant bits and lies
within the FP32 exponent range. In this numerical model,
`100.5 * 100.5 = 10100.25`, rather than the half-rounded `10104`. Actual
cooperative-matrix feature negotiation and implementation precision belong to
the separate platform investigation. The CPU script assumes full products and
explicit FP32 additions; it does not emulate a driver's matrix summation order.

Wide accumulation cannot reconstruct information discarded while forming the
operands. Exhaustive native-code enumeration gives a useful refinement:

| Centered native coded input | Maximum FP16 conversion error |
| --- | ---: |
| 8 bits, integer steps | 0 |
| 10 bits, quarter steps | 0 |
| 12 bits, sixteenth steps | 0 |
| 16 bits, 1/256 steps | 0.03125 |

These inputs are specifically `code * 2^(8-depth) - 128`, matching
`vmaf_luma.comp`, not arbitrary RGB floats. The negative endpoint is exactly
-128 and the largest positive value remains below 128, where half spacing is
1/16. Thus native 12-bit source values also fit exactly. Filter outputs,
filtered pyramid levels, moments and de-meaned values generally do not.

All 65,536 centered native 16-bit values reconstruct exactly as
`H + L / 4096`, where `H = half(x)` and
`L = half((x - H) * 4096)`. This is a representation result, not a statement
that two matrix products reproduce a complete filter exactly.

## Best order of investigation

| Candidate | Practical assessment |
| --- | --- |
| v1 SpEED covariance | First candidate: many samples feed a dense 25×25 Gram matrix. Keep means, covariance output, singular guards, solver and pooling wide. It has a separate numerical/prototype investigation. |
| v0 VIF filtering/moments | Largest v0 opportunity, but continuous weights and cancellation-sensitive moments require an actual packing and precision design. A direct half conversion of five precomputed statistic planes fails even with wide products. |
| v0/v1 ADM DWT | Short 4-tap low/high filters; first native operands can be exact, coefficients and subsequent stages cannot. Splitting improves numerical examples substantially, at several matrix products per pass plus packing. |
| v0/v1 motion | Short 5-tap filter with full-resolution history. Small temporal differences make quantization visible; keep history and differences FP32. |
| v1 SpEED filtering | Sparse antialias sampling and a short mean filter map less naturally than its Gram matrix. A wide product does not remove filter coefficient/operand rounding. |
| v1 CAMBI | Histogram, mode, rank/tie, mask and contrast logic dominate. Dense matrix multiplication does not match the responsibility; exact packed storage is the relevant separate opportunity. |
| v0/v1 model predictor | Only 4 or 6 input dimensions and one feature vector per completed frame. RBF exponential, double-precision decisions, transform and synchronization remain. Keep the existing host predictor. |

There is no counterpart of SpEED covariance in v0. A successful v1 covariance
backend therefore does not automatically accelerate v0.

## VIF needs more than an exact native source

The four filters in `vmaf_vif_common.glsl` have 17, 9, 5 and 3 taps.
`vmaf_vif_fused0.comp` and `vmaf_vif_fused.comp` form five vertical planes
(two means and three second moments), then filter horizontally before computing
variance and covariance. They already share source/halo loads and two adjacent
horizontal outputs. The scale-down passes add further filtering. A matrix
design must replace useful work in this schedule, rather than inserting a
packing/readback phase around otherwise unchanged scalar computation.

Converting the five precomputed planes into half matrix operands discards
the exact products the wide operation was intended to preserve. The CPU
illustration for a constant centered value of 100.5 computes variance
0.0048828125 with the original coefficients and explicit FP32 ordering, but
2.59375 after two separable convolutions with half operands and full FP32
products. This crosses the VIF branch at 2. These figures describe that
specific conversion and summation mapping; they are not GPU oracle values or
a proof against every matrix formulation.

A direct weighted Gram avoids precomputing `x²`: form columns
`sqrt(w) * reference`, `sqrt(w) * distorted` and `sqrt(w)`. It then has three
problems:

- The square-root-weighted operands are continuous even for exact native input.
- One independent 3×3 Gram uses only 9 of a 16×8 result tile's 128 entries,
  before considering zero padding or symmetry. Packing independent windows
  together needs either redundant off-diagonal work or block-diagonal zeros.
- Rounding the weighting column changes its normalization and therefore the
  cancellation in `E[x²] - E[x]²`. The 17×17 checker example with values
  `100.5 ± 1.5` has variance 2.234375 with FP32 weighted operands and
  1.072265625 with half operands in the same explicit sum order. The branch
  flips despite wide products. Dividing by the quantized weight norm changes
  the latter variance to 2.3076171875, demonstrating why normalization is a
  separate algorithm choice rather than an automatic precision fix.

Tile recentering/rescaling can improve conditioning but must be valid for
every output window represented by the tile, including heterogeneous areas.
Per-window recentering or a central-moment reconstruction changes the cost and
requires independent feature tests. Normalizing quantized weights to sum to
one only fixes their DC sum; it does not restore the original taps or preserve
near-branch behavior on nonconstant inputs.

## Dense filter utilization is a design constraint

The following are arithmetic counts for illustrative full tiles using shapes
advertised by the earlier local capability query. They are not performance
predictions or upper bounds on every possible packing. They exclude edge
waste, gathers, conversions, shared stores/barriers and result redistribution.
An MMA term means one multiply-add term in the dense matrix operation.

For a unit-stride filter with eight consecutive outputs, put 16 independent
columns/feature streams in matrix rows and eight output positions in columns.
The common source span is `taps + 7`, padded to a multiple of 16. Only `taps`
entries in each coefficient column are nonzero.

| Mapping | Useful terms / dense issued terms |
| --- | ---: |
| 17-tap VIF, K=32 | 17/32 = 53.125% |
| 9-tap VIF, K=16 | 9/16 = 56.25% |
| 5-tap VIF or motion, K=16 | 5/16 = 31.25% |
| 3-tap VIF or mean filter, K=16 | 3/16 = 18.75% |
| ADM: four decimated positions × low/high, K=16 | 4/16 = 25% |

Filling VIF's 16 rows with five quantities across three neighboring columns
uses only 15 rows, multiplying the corresponding fractions by 15/16. ADM's
eight output columns share ten input positions (`2 * (4-1) + 4`), padded to
16. Reference/distortion and neighboring columns can fill its independent rows.

SpEED's current antialias stage is different: it selects nine taps around
centers separated by 16 source samples. Its packed scratch discards source
columns the following horizontal stage never reads. A simple dense mapping
of independent 9-tap windows with one coefficient vector and a 16×8×16 tile
uses only `9/(16*8) = 7.03125%` of issued terms; the unit-stride 9-tap figure
does not describe that workload. Other packings would need an explicit cost
model and prototype.

## Splitting both operands

For a filter dot product, represent the coefficient and source as
`w ≈ Wh + Wl/s` and `x ≈ Xh + Xl/s`, with half-valued components and
`s = 4096`. Three matrix products reconstruct

```text
Wh*Xh + (Wh*Xl + Wl*Xh)/s
```

and a fourth adds `Wl*Xl/(s*s)`. If the source is exactly half, only the
coefficient's two terms are needed for this representation. Once a preceding
filter creates continuous values, both operand splits are relevant again.
Unlike a symmetric Gram matrix, an arbitrary filter product does not recover
both cross terms by transposing one product.

The powers of two avoid rescaling roundoff when representable and keep tiny
residual coefficients away from half underflow. A production implementation
would still need finite/range/subnormal checks, and possibly power-of-two
scaling per tile. A fixed two-part expansion is not an exact representation of
every FP32 value; the output's FP32 addition order also changes.

In the deterministic four-level ADM illustration on 65×67 native 10-bit
noise, the maximum final-level detail error falls from 0.2786712646484375
with single half operands and wide products to 0.00006103515625 with either
three or four split products. This is encouraging representation evidence,
not metric-score qualification. Earlier levels and native 16-bit inputs are
in the compact results. The four-term version is not uniformly closer to the
original FP32 order because reassociation and rounding remain.

This proposes changing the DWT only. ADM's subsequent restoration division,
`1e-30` guard, angular test around `0.9996954135`, CSF masking, cubic terms
and reductions still need their current wider arithmetic. Small transform
errors can also change those downstream decisions; DWT band agreement is not
equivalent to ADM2/ADM3 feature or predictor agreement.

Motion illustrates the remaining distinction clearly. For constant frames
`127 → 127 + 1/256`, original FP32 filtering preserves the 0.00390625
transition; one-half-operand filtering reports zero, while the split variants
preserve it in this example. For the 12-bit step `127 → 127 + 1/16`, the
source itself is exact half, but intermediate/coefficients and changed ordering
still move the measured filtered difference from 0.0625 to 0.0624847412109375.
Keep temporal sequence tests separate from still-image corpus checks.

## Historical frame-time ceilings

These are shares of the earlier FP16 study's retained B-arm native resident
frame medians, not measurements of the cooperative prototype. They derive
from the earlier five interleaved rounds; the source result hash and derived
medians are retained in the compact data, with full runs in the raw archive.
Stage timestamps and whole-frame times have different scopes, so shares are
useful scale estimates rather than a causal decomposition of every cycle.

| Stage | 1080p frame share | 4K frame share |
| --- | ---: | ---: |
| v0 VIF | 41.84% | 40.88% |
| v0 ADM | 20.96% | 28.15% |
| v0 motion | 15.96% | 17.81% |
| v1 ADM3 | 15.88% | 21.22% |
| v1 complete SpEED | 11.95% | 5.22% |
| v1 CAMBI | 42.62% | 48.05% |
| v1 motion | 10.63% | 11.41% |

Even deleting all SpEED work cannot remove more than its historical stage
duration on those workloads; covariance is only part of SpEED. V0/V1 predictor
medians are about 0.05–0.26% of the corresponding frame time, including their
host score stage; a single-frame GPU predictor has little headroom before
transfer/synchronization costs.

The practical next step remains an isolated, feature-gated covariance backend,
with FP32 fallback and explicit numerical tests. For v0, a coefficient/source
split VIF filter is the next meaningful matrix experiment only if its packing
and multiple-product cost can beat the existing fused implementation. Neither
a peak matrix throughput figure nor successful full-product arithmetic proves
that result.

## Verification and limits

The CPU probe ran successfully from the repository root and from `/tmp` with
an explicit output override; outputs matched. Importing it produces no output
or files. Native-code checks exhaust all supported code values; other examples
are bounded deterministic diagnostics rather than exhaustive error proofs.
The script preserves source hashes and explicit arithmetic boundaries. No
production file, public API, model table, reference oracle or tolerance changed
for this analysis, and no GPU/performance result is claimed here.
