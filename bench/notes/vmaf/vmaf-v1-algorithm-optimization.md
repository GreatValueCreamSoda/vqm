# VMAF v1 algorithm follow-up

This follows the [histogram and Jacobi pass](vmaf-v1-pipeline-optimization.md),
comparing `44e7e3e` with the numerical implementation in `778a02e` and the
benchmark cleanup in `24bad1a`. Measurements use an RTX 5070, driver 615.78.08,
on 2026-11-02. The standard SDR v1.0.16 3H model is used at both input sizes.

## The Netflix comparison

Netflix's published `ducks_take_off` table shows v1 beating v0 at every CPU
thread count, including 556 versus 349.7 fps at 1080p with 16 threads. Netflix
attributes the lower cost to removing VIF, optimizing CAMBI and measuring
chroma at a lower scale. See [the announcement](https://netflixtechblog.com/vmaf-v1-good-is-not-good-enough-60d7e4244ea8).

Our earlier GPU reversal exposed implementation costs: SpEED serialized U/V,
carried every eigenvector through Jacobi rotations, and used those vectors for
variance. The CPU reference computes eigenvalues separately and solves a linear
system for variance. CAMBI also performed histogram setup and sliding-window
barriers even for wholly ineligible blocks.

The CPU ratio does not specify the attainable GPU ratio. Our v0 uses the float
model, and the screenshot does not specify its exact v0 feature mode, input bit
depth, distortion or viewing model. In particular, our 4K measurements use the
1080p viewing model; Netflix's separate 4K 3H model sets SpEED prescale to 0.5.
That is a potential 4K workload difference, not an explanation of the 1080p gap.
V1's native-chroma `/16` dimensions match the standard model.

## Combined measurements

Five interleaved runs per arm: resident uses `--iterations 20 --warmup 3
--resident 2`; streaming uses `--iterations 8 --warmup 2`. Both images are fully
prepared every call. Decode is outside timing; resident uploads are also
outside timing. Pixels repeat and motion is zero after the first frame, so
resident approximates unique-frame work in hot GPU memory. The primary pairs
are the same 10-bit YUV420 fixtures as the preceding pass.

| workload | before ms/pair, median (range) | after ms/pair, median (range) | pairs/s before → after |
| --- | ---: | ---: | ---: |
| 1080p resident | 1.219366 (1.208262–1.220441) | 0.975430 (0.973681–0.981895) | 820 → 1025 |
| 4K resident | 3.869865 (3.847846–3.895395) | 3.456441 (3.449806–3.488911) | 258 → 289 |
| 1080p streaming | 1.771323 (1.763958–1.772688) | 1.535776 (1.528956–1.557196) | 565 → 651 |
| 4K streaming | 6.043592 (6.026907–6.079341) | 5.651378 (5.642408–5.664503) | 165 → 177 |

Resident SpEED moved 0.338214 → 0.156386 ms at 1080p and
0.395694 → 0.229859 ms at 4K. CAMBI moved 0.514586 → 0.464043 ms and
2.170667 → 1.934461 ms. Preparation, ADM3 and motion were effectively unchanged.
Resident score/checksum changed 95.973480 / 2207.390043 →
95.973463 / 2207.389650 and 96.065012 / 2209.495268 →
96.065004 / 2209.495096. Streaming checksums changed
959.734801 → 959.734630 and 960.650117 → 960.650042. Every arm was stable
across all five runs; the new numerical solver accounts for the score movement.

## Controlled ducks frame check

We fetched the first 8-bit YUV420 frame from [Xiph's source sequence](https://media.xiph.org/video/derf/y4m/ducks_take_off_1080p50.y4m),
using byte range 0–3110441, then encoded it with libx264 medium CRF 28 and
decoded one frame. This is a controlled still-frame pair, not a reproduction
of Netflix's full video/distortion workload. Both arms consume the same bytes.

Five interleaved before/after runs measured resident
1.071699 (1.062488–1.076348) → 0.828112 (0.818267–0.829852) ms and streaming
1.341519 (1.335704–1.359829) → 1.092428 (1.089593–1.101808) ms. Score changed
75.839108 → 75.839081; resident checksum 1744.299495 → 1744.298868 and streaming
758.391085 → 758.390812.

A separate interleaved comparison of the final v0 and v1 binaries measured:

| mode | v0 ms/pair, median (range) | v1 ms/pair, median (range) | v0 / v1 pairs/s |
| --- | ---: | ---: | ---: |
| resident | 0.565016 (0.563796–0.567281) | 0.819213 (0.818682–0.820263) | 1770 / 1221 |
| streaming | 0.838357 (0.835472–0.846542) | 1.090668 (1.088633–1.097283) | 1193 / 917 |

V1 still loses to v0 on this pair. The reversal relative to Netflix's CPU
table remains; these improvements do not establish that GPU v1 is fully
optimized. CAMBI remains the largest stage.

## Retained changes and tradeoffs

- **Batch U/V statistics (`88c0550`).** Keep two small filtered/statistics banks
  and dispatch four independent reference/distortion matrices together.
  Independent resident SpEED medians were 0.336827 → 0.216536 ms at 1080p and
  0.395256 → 0.269419 ms at 4K, with identical scores/checksums. Extra arena
  storage is 26,256 / 74,748 bytes. Push constants and shader count are unchanged.
- **Eigenvalues plus a variance solve (`778a02e`).** Cooperative Householder
  reduction and tridiagonal QR replace eigenvector rotations. Cholesky forward
  solves evaluate `yᵀK⁻¹y` without the eigenvector matrix. Shared memory drops
  from 10,504 to 5,408 bytes. An algebraically trace-preserving Givens update
  fixed spectral rounding drift; no test tolerance was relaxed. On top of
  batching and the CAMBI shortcut, independent resident SpEED medians improved
  0.217021 → 0.156290 ms and 0.269394 → 0.229986 ms. The 512-invocation variant
  remains selected where supported, with the existing 128 fallback.
- **Skip empty CAMBI groups (`efdb244`).** Probe 64 output centers first;
  wholly masked or out-of-band groups write zero and return before halo loads,
  histogram clearing and sliding barriers. Shared storage stays 15,108 bytes.
  Independent CAMBI medians improved 0.514107 → 0.465536 ms at 1080p,
  2.169174 → 1.935757 ms at 4K, and 0.388216 → 0.296930 ms on the ducks frame.
  A fully active constant-Y stress pair regressed 0.683088 → 0.695475 ms
  (+1.81%) and 2.962816 → 3.015595 ms (+1.78%). That content gets the added
  precheck barrier without the empty-group benefit. Scores/checksums matched.

Each numerical optimization passed the 2% targeted-stage gate at both primary
sizes, including the solver's incremental comparison after batching. A final
128-versus-512 tuning arm failed and was reverted: SpEED changed
0.156341 → 0.158522 ms at 1080p, 0.229445 → 0.231666 ms at 4K, and
0.177936 → 0.180445 ms on ducks, with identical scores/checksums.

Temporary diagnostic timestamps in a separate benchmark build apportioned the
original 1080p/4K costs to CAMBI contrast 0.279426/1.392309 ms, radix selection
0.116979/0.143682 ms, preparation 0.060750/0.466294 ms, and SpEED eigen work
0.240526/0.228904 ms. The extra timestamps perturb totals; these figures guide
attribution, while performance claims above use the unmodified outer stages.
The diagnostic code was not retained.

## Validation and cleanup

All 126 GPU tests passed, including both eigensolver workgroup variants,
near-singular and moving-sequence reference checks, known spectra, factor
reconstruction, variance solves, batched/separate channel equivalence, and
sentinel checks for every empty-map output. The ASAN/UBSAN GPU suite and a full
ASAN/UBSAN 4,292-pair corpus run passed. CTest passed 10/10, formatting and
SPIR-V validation passed, and SSIMULACRA 2 / Butteraugli identities remained
100.00000000 / 0.00000000.

Full CID22 validation verified all input hashes. Versus `44e7e3e`, maximum,
mean and p99 score drift were 0.00016136 / 0.000023908 / 0.00009515. Against
pinned libvmaf, maximum/mean absolute error was 0.004638 / 0.001675.
KRCC/SRCC/PCC were 0.56395083 / 0.76184472 / 0.71779541 versus reference
0.56396690 / 0.76186042 / 0.71780187. These are independent still pairs with
zero motion; moving-frame checks are separate.

One ordinary corpus invocation exited with SIGSEGV (-11). A debugger run was
interrupted with SIGINT while main waited inside `vkDestroyDevice` during
process exit and an NVIDIA compiler worker was active; it did not capture
the original SIGSEGV. An earlier [reference run](vmaf-reference.md) recorded
an intermittent driver-worker failure during device destruction.

The benchmark now releases its retained device after its automatic pipelines
are destroyed and before process-exit handlers (`24bad1a`). The private helper
rejects outstanding shared owners and performs normal destruction; it neither
leaks the device nor suppresses failures. Five ordinary buffered full-corpus
runs then exited normally with byte-identical scores. An unmodified baseline
also exited normally once, so the earlier crash's exact cause is unproven.
Production library teardown retains its existing fallback.

Raw logs, stage spreads, input provenance, failed-run output, debugger stacks,
exit checks and final corpus reports are under `artifacts/reports/vmaf-v1-chart/`.
All temporary worktrees and diagnostic instrumentation were removed.
