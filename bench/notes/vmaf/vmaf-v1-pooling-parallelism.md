# VMAF v1 pooling and parallelism

This follows [the bottleneck investigation](vmaf-v1-bottlenecks.md). It
compares baseline `01735d6` with numerical implementation `764188f` on an
RTX 5070, NVIDIA driver 615.78.08, on 2026-11-02. Both use the standard SDR
v1.0.16 3H model. The 1080p and 4K inputs are 10-bit YUV420; the ducks input
is one 8-bit 1080p frame distorted with x264 CRF 28, not Netflix's whole-video
benchmark. Input hashes and preparation are recorded in
[`vmaf-v1-algorithm-optimization.md`](vmaf-v1-algorithm-optimization.md).

Each arm uses its own build directory. Five runs per arm were interleaved in
one sitting, with `--iterations 20 --warmup 3 --resident 2` for resident and
`--iterations 8 --warmup 2` for streaming. Resident fully prepares both images
each call and excludes uploads; it reuses pixels in hot memory and motion is
zero after the first call. Both paths exclude decode. Reported spreads are the
minimum and maximum of five per-run averages, not per-frame percentiles.

| workload | baseline ms/pair, median (range) | candidate ms/pair, median (range) | pairs/s baseline → candidate |
| --- | ---: | ---: | ---: |
| 1080p resident | 0.986054 (0.974695–0.988984) | 0.829417 (0.819057–0.834942) | 1014 → 1206 |
| 4K resident | 3.456492 (3.441486–3.536615) | 3.299519 (3.279939–3.319759) | 289 → 303 |
| ducks 1080p resident | 0.820732 (0.820362–0.830057) | 0.641615 (0.640695–0.652285) | 1218 → 1559 |
| 1080p streaming | 1.537421 (1.524171–1.539421) | 1.379999 (1.370463–1.380558) | 650 → 725 |
| 4K streaming | 5.681476 (5.624283–5.751840) | 5.498884 (5.459670–5.521009) | 176 → 182 |
| ducks 1080p streaming | 1.101037 (1.088763–1.106503) | 0.914610 (0.908571–0.924991) | 908 → 1093 |

Resident CAMBI stages moved 0.464664 → 0.371387 ms at 1080p and 1.936496 →
1.850389 ms at 4K; SpEED moved 0.156312 → 0.111250 ms and 0.230254 →
0.178453 ms. On ducks CAMBI moved 0.293758 → 0.201142 ms and SpEED
0.177806 → 0.108861 ms. Scores/checksums were 95.973463 / 2207.389650 →
95.973463 / 2207.389651 at 1080p, 96.065004 / 2209.495096 → 96.065006 /
2209.495149 at 4K, and 75.839081 / 1744.298868 → 75.839083 / 1744.298908
on ducks. Streaming checksums were 959.734630 → 959.734631,
960.650042 → 960.650065, and 758.390812 → 758.390830 respectively.
The eigensolver accounts for the small numerical changes.

For comparison, a separate same-binary, same-sitting interleaved v0/v1 run
on the ducks pair gave resident 0.566422 (0.564811–0.570906) / 0.642260
(0.641685–0.648975) ms, or 1765 / 1557 pairs/s. Streaming was 0.848957
(0.838002–0.850337) / 0.922366 (0.909446–0.927050) ms, or 1178 / 1084
pairs/s. The v0 float and v1 SDR models are different metrics; their respective
scores were 66.947328 and 75.839083. V1 remains slower here, and this still
frame does not reproduce Netflix's CPU or temporal workload.

## Changes

- **Batch CAMBI pooling across scales (`93dcb1b`).** Keep five disjoint,
  packed contrast maps and per-scale histogram, state, partial and result
  regions, then dispatch each of the 11 pooling steps across all five scales.
  The former implementation serialized 11 steps per scale, or 55 dispatches.
  Variable scale sizes are mapped to a packed workgroup grid; a rectangular
  fallback handles devices where the packed X dimension exceeds the limit.
  Per-scale selection and reduction order are unchanged. Extra arena storage
  is 695,624 bytes at 1080p and 2,769,132 bytes at 4K. Tests cover packed and
  forced-rectangular grids, independent early completion, ties, signed zero,
  tiny values, scratch reuse, nonmonotonic counts and odd-size maps.
- **Concurrent factorizations (`82e6135`).** Run the four eigenvalue systems
  and four independent Cholesky factorizations as separate workgroups in one
  dispatch, with separate status words and the former eigen-status precedence.
  An independent interleaved resident comparison moved SpEED 0.156291 →
  0.141968 ms at 1080p and 0.229821 → 0.214672 ms at 4K, with matching
  printed scores/checksums. This adds two FP32 scratch words per U/V bank.
- **Parallel eigenvalues with guarded QR (`764188f`).** Independent FP32
  Sturm bisections replace the serial tridiagonal QR path when the estimated
  minimum eigenvalue is sufficiently far from the singular threshold. The
  25 eigenvalues run in parallel within each eigen workgroup; this does not
  increase the number of independent matrices. If bisection fails or the
  minimum falls within `1e-6 + 1e-3 * spectral_bound`, the shader reruns the
  previous QR solver and preserves its status precedence. The safety band is
  an engineering heuristic, **not** a proven FP32 error bound. With pooling
  and concurrent factorization already applied, independent interleaved
  resident SpEED medians moved 0.141990 → 0.111250 ms at 1080p and 0.215440 →
  0.178341 ms at 4K. GPU reference tolerances were not relaxed.

## Precision and validation

Full CID22 validation scored 4,292 independent still pairs with verified input
hashes; the temporal motion feature is zero for this experiment. Against the
baseline, max/mean/p99 absolute score changes were 0.00000923 / 0.000001632 /
0.00000647. Against pinned libvmaf v1, max/mean errors were 0.0046381 /
0.0016748. KRCC/SRCC/PCC against MCOS were 0.56395105 / 0.76184506 /
0.71779543 versus libvmaf's 0.56396690 / 0.76186042 / 0.71780187. Moving-frame
and near-singular reference cases are covered separately in the GPU tests.

A deterministic stress harness evaluated 10,000 deliberately threshold-focused
FP32 tridiagonal positive-definite matrices on this GPU. Unguarded bisection
changed the existing QR's status in 3,509 cases; the final guarded solver
matched QR status in all 10,000. These are targeted synthetic matrices, not an
estimate of how often real images are affected. The guard preserves legacy
behavior in these cases; neither result is always correct against a
higher-precision oracle, and a change of singular status can produce a large
individual model-score change. The guard does not prove equivalence outside
the tested set. A diagnostic-only standalone guarded build reported fallback
for 34 of 4,292 CID22 frames (41 of 17,168 matrices); its printed scores
matched the ordinary guarded build. It reported no fallback on the three
benchmark fixtures. Numerical behavior has been measured on this NVIDIA GPU;
AMD and Intel implementations have not been tested.

The final GPU and ASAN/UBSAN GPU suites each passed 129 tests; CTest passed
10/10. Formatting, changed-shader `spirv-val`, SSIMULACRA 2 identity
100.00000000 and Butteraugli identity 0.00000000 passed. Input hashes, source
patches, raw per-run stage tables, stress dataset, scripts and outputs are in
the ignored local directory `artifacts/reports/vmaf-v1-pooling-parallelism/`.

## Alternatives measured and not retained

- A two-pass 16-bit radix selector reduced the dispatch count but required
  512 KiB of histogram scratch. In an independent interleaved run, CAMBI
  regressed 0.464638 → 0.561299 ms at 1080p and 1.933086 → 2.016722 ms at
  4K. Scores/checksums matched. It was discarded.
- A three-pass 11/11/10-bit radix selector used about 23 KiB of extra scratch
  and improved over the original, but lost a direct interleaved comparison to
  packed batching: 0.418958 versus 0.371275 ms at 1080p and 1.883888 versus
  1.850698 ms at 4K. It was not retained.
- Rectangular five-scale batching improved 1080p but regressed CAMBI at 4K
  from 1.934971 to 2.001021 ms. Flattening the workgroup grid removed that
  regression; the rectangular mapping remains only as a device-limit fallback.
- Unguarded Sturm bisection passed CID22 MOS and score gates but changed many
  statuses in the synthetic threshold stress set described above. It was not
  retained without the QR guard.
