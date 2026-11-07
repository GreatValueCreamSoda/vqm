# SpEED covariance with FP16 operands and wide accumulation

This 2026-11-05 numerical investigation evaluates the operands of a matrix
implementation of SpEED covariance. It uses deterministic CPU diagnostics,
not GPU timings, hardware matrix-unit emulation, a libvmaf oracle, or an
accuracy qualification. The production shaders were not changed by this
review. The associated GPU implementation must be measured separately.

Reproduce from the repository root with Python and NumPy:

```sh
python3 bench/experiments/cooperative_matrix/speed_numerics.py
```

The default output is
`artifacts/reports/vmaf-cooperative-matrix/numerics/speed.json`; `--out PATH`
overrides it and creates its parent directories. The compact, versioned
[result snapshot](../../measurements/vmaf-cooperative-speed-numerics.json)
includes NumPy version, seed, source hashes, sample hashes, all 20 cases and
the per-variant error metrics. The original source arithmetic is in
[`vmaf_speed_stats_cov.comp`](../../../../src/metrics/vmaf/speed/shaders/vmaf_speed_stats_cov.comp),
with filtering coefficients beside it in
[`vmaf_speed_filter_common.glsl`](../../../../src/metrics/vmaf/speed/shaders/vmaf_speed_filter_common.glsl).

## The matrix being computed

For a filtered plane of width `w` and height `h`, SpEED first crops to
`tw = floor(w/5)*5`, `th = floor(h/5)*5`. It enumerates all overlapping 5×5
patches within that crop. There are `N = (tw−4)*(th−4)` patches, each containing
25 components in row-major order. Each component has its own mean over those
same N patches. Means are computed in FP32 before quantizing any operand.

Let X contain the mean-subtracted patch vectors in its N rows. Covariance is
`G = XᵀX/N`, not the uncentered Gram matrix. For example, a filtered 37×29
plane uses 31×21 = 651 patches; the unused right and bottom pixels do not enter
the means or covariance. A 5×5 filtered plane has one sample and exactly zero
centered covariance. K padding for a matrix instruction must contain zeros and
must not change the denominator. Padding the 25 feature components to 32 also
requires zero rows/columns, while the solver still consumes 25 components.

The existing shader performs 64 strided per-lane accumulations followed by a
halving reduction tree. The diagnostic preserves that sequence to separate
operand changes from a wholly different reduction order. The Gram helper uses
binary64 product/add temporaries, then rounds each accumulation into FP32.
This models full products of finite FP16 operands in the normal FP32 range;
it does not round the product to FP16. For general FP32 inputs, its binary64
temporary is a diagnostic approximation to a fused GPU operation. Actual
cooperative-matrix addition order and subnormal handling can differ.
The diagnostic normalizes each component Gram by N before merging residual
terms; a GPU implementation that merges raw partials before its final division
has another rounding-order difference and must be checked directly.

The filter diagnostic keeps the maintained tap order, separate FP32
multiply/adds and mirroring. Production shader contraction can differ. Results
therefore compare the stated CPU constructions, not byte-for-byte GPU results.

## Operand representations

The plain candidate rounds the already centered X to FP16, multiplies those
operands with a full product and accumulates FP32. This removes the prior
experiment's explicit FP16 product rounding, but still loses operand bits.

A per-matrix power-of-two scale S places the largest magnitude in [128, 256).
Scaling protects small operands from FP16 underflow and can be undone by
dividing covariance by S². It does not add significand bits to FP16. Within
the normal FP16 range, scaling usually leaves the represented operand error
unchanged. This diagnostic also evaluates a scaled residual:

```text
H = half(S*X)
L = half((S*X − float(H))*2048)
S*X ≈ H + L/2048
```

Two Gram products are enough for the approximation
`HᵀH + (HᵀL + (HᵀL)ᵀ)/2048`. A third product adds `LᵀL/2048²`.
The symmetry saving requires retaining the cross matrix and reading its
transpose. A shader that computes `HL` and `LH` directly instead uses three
products without the low-low term, or four products with it. The two forms
are mathematically equivalent; their FP32 accumulation and merge rounding can
differ. This CPU script uses the two-/three-product symmetric form.

The residual itself still has an FP16 significand. Two halves do not generally
represent every FP32 value exactly. The 320×320 filtered test case reconstructs
4,945 of 6,400 centered values exactly, with maximum reconstruction error
1.90735e-6. Dropping the low-low term adds another error of order the squared
operand-rounding error. Merely multiplying the residual by a power of two
does not remove either limitation.

## Observed errors

These are maximum absolute covariance-entry differences from the stated
strided FP32 baseline. They are observed values, not error bounds.

| Diagnostic | Plain FP16 operands, wide products | Two-product residual Gram | Three-product residual Gram |
| --- | ---: | ---: | ---: |
| Filtered 320×320 test pattern, 256 samples | 0.00591278 | 0.0000152588 | 0.0000152588 |
| Filtered 593×465 test pattern, 651 samples | 0.00382996 | 0.0000152588 | 0.0000152588 |
| Filtered high-contrast stripes, 651 samples | 0.770996 | 0.000976562 | 0.000976562 |

The corresponding relative spectral errors of the two-product candidate are
3.82683e-8, 3.49990e-8 and 3.88220e-8. Those small normwise errors do not imply
small relative changes in a tiny eigenvalue. In the stripe case, the baseline
minimum eigenvalue is 0.0113895, while the two-product candidate gives 0.0109721
and the three-product candidate gives 0.0110659.

The cases include flat coded planes, low-bit ripples, smooth chroma, high
contrast, the 81×81 minimum-filter input, odd cropped geometry, near-low-rank
planes, tiny variations, threshold-scaled patch planes and synthetic matrices.
The unscaled plain-half candidate loses all of the tiny-variation matrix;
power-of-two scaling prevents that loss. No singular-classification changes
were found in the patch-derived cases in this set. This is not a universal
result for image patches.

## Strict singular decisions remain sensitive

SpEED treats an eigenvalue below 1e-6 as singular. Six additional direct-matrix
cases have controlled eigenvalues near that boundary. They are conditioning
stress matrices, not asserted to be realizable 5×5 overlapping image patches.
Their ideal target spectra differ slightly from the strided FP32 baseline.

Plain FP16-input wide accumulation changes the baseline singular decision in
three cases. The residual construction changes it in one case even when the
low-low term is included:

| Direct-matrix stress | Minimum eigenvalue |
| --- | ---: |
| Strided FP32 baseline | 1.0955203e-6 |
| Plain half operands, wide accumulation | 1.0037031e-6 |
| Two-product residual Gram | 7.1808990e-7 |
| Three-product residual Gram | 9.2492217e-7 |

Changing the summation order is sufficient to matter at this boundary; adding
the omitted low-low term does not establish equivalence. This is why an
accuracy claim cannot be inferred from lower average covariance error.

## What a guard would need to do

The existing solver's QR fallback re-evaluates the same supplied covariance
matrix. It cannot recover information lost while constructing that matrix.
A precision guard for a matrix backend must be able to recompute covariance
from the original centered FP32 operands before running the final solver and
pooling. Any downstream means or block samples remain FP32 too.

For an exact quantized operand Q with D = X−Q, a conservative real-arithmetic
bound is `||XᵀX−QᵀQ||₂/N <= (2||Q||F||D||F + ||D||F²)/N`. For the
two-product residual construction, add a bound on the omitted low-low Gram.
The result JSON records these operand-only bounds. They exclude accumulation,
reduction, rescaling and final-add rounding and therefore cannot alone certify
the actual GPU result.

An eigenvalue decision is safe from a known total perturbation bound E only
when its distance from the threshold exceeds E. Cholesky positivity and the
variance solve also need conditioning checks. Since pooling uses both a
factorization and the spectrum, preserving the singular flag alone does not
bound the final SpEED feature or VMAF score. The conservative operand bounds
can themselves reject useful inputs: the stripe two-product bound is about
0.01658, above that case's minimum eigenvalue despite its small normwise
observed error. A practical guard must balance the cost of statistics,
conservative fallback frequency and measured final-feature accuracy.

The residual approach is a better accuracy candidate than a direct half-input
replacement. It still needs actual cooperative-matrix feature/oracle tests,
near-singular and temporal coverage, full-corpus drift checks, device precision
contracts, and the repository's interleaved speed gate before adoption.
