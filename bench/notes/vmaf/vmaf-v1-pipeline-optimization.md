# VMAF v1 histogram and eigensolver optimizations

For the subsequent batched statistics and eigenvalue-only implementation, see
[`vmaf-v1-algorithm-optimization.md`](vmaf-v1-algorithm-optimization.md).

Measured on 2026-11-02 with an RTX 5070, driver 615.78.08, comparing commit
`5509ac2` against `9ab9c92`. These follow the [first optimization
pass](vmaf-v1-optimization.md). The model is standard SDR v1.0.16 3H, including
when the input is 4K; this does not benchmark the separate 4K viewing model.

## Combined result

Five interleaved runs per arm using `ssimulacra2_bench`: resident uses
20 iterations, 3 warm-ups and `--resident 2`; streaming uses 8 iterations and
2 warm-ups. Both images undergo full preparation every call. Resident inputs
reuse pixels, approximate a unique-frame stream in hot GPU memory, and exclude
uploads. Both modes exclude decoding. The repeated reference produces zero
motion after the first frame.

| workload | before ms/pair, median (range) | after ms/pair, median (range) | pairs/s before → after |
| --- | ---: | ---: | ---: |
| 1080p resident | 2.093698 (2.089169–2.105603) | 1.213097 (1.210267–1.219867) | 478 → 824 |
| 4K resident | 8.063964 (7.999755–8.072384) | 3.874002 (3.868442–3.930551) | 124 → 258 |
| 1080p streaming | 2.639916 (2.633965–2.658556) | 1.773248 (1.759614–1.913936) | 379 → 564 |
| 4K streaming | 10.262941 (10.219252–10.400685) | 6.064529 (6.030835–6.092299) | 97 → 165 |

Resident throughput improves 1.73× at 1080p and 2.08× at 4K. Stage medians:

| stage | 1080p before → after, ms | 4K before → after, ms |
| --- | ---: | ---: |
| coded-plane preparation | 0.074664 → 0.075126 | 0.283216 → 0.283542 |
| ADM3 | 0.119030 → 0.119021 | 0.573011 → 0.572976 |
| SpEED | 0.903398 → 0.338090 | 0.955270 → 0.395885 |
| CAMBI | 0.789352 → 0.514672 | 5.693400 → 2.173448 |
| motion | 0.073082 → 0.073059 | 0.297557 → 0.297206 |
| host model | 0.001492 → 0.001079 | 0.002036 → 0.001539 |

The resident score/checksum changes were 95.973465 / 2207.389685 →
95.973480 / 2207.390043 at 1080p and 96.065015 / 2209.495343 →
96.065012 / 2209.495268 at 4K. Streaming checksums were
959.734646 → 959.734801 and 960.650149 → 960.650117 respectively.
All runs had stable per-arm scores and checksums. The changed Jacobi rotation
order accounts for the small score movement.

V0 regression checks in the same sitting had overlapping timing ranges and
identical scores/checksums. Resident medians were 0.566786 → 0.566572 ms at
1080p and 2.368155 → 2.366799 ms at 4K. Streaming was
1.128929 → 1.117413 ms and 4.555951 → 4.554172 ms. V1 remains slower than
V0 on these workloads. An additional interleaved 1080p photographic check,
upscaling CID22's `1025469` original and mozjpeg q40 distortion with Lanczos,
also measured v0/v1 resident medians of 0.567936/1.224122 ms and streaming
1.127338/1.783528 ms; changing content did not close the gap on that pair.

## Accepted changes

- **Independent Jacobi rotations (`68a0229`).** A circle schedule performs
  12 disjoint rotations per round, visiting all 300 pairs in 25 rounds. Shared
  double buffers avoid cross-pivot read/write races and preserve symmetric
  covariance entries. The singularity, nonfinite and convergence guards remain.
  Shared storage is 10,504 bytes. Independent five-run resident comparisons
  reduced SpEED 0.903366 (0.902746–0.903590) →
  0.442029 (0.440877–0.442170) ms at 1080p and
  0.954819 (0.952688–0.955034) → 0.494018 (0.493624–0.495061) ms at 4K.
- **Wider eigensolver (`9ab9c92`).** Use 512 invocations when both the device's
  total invocation limit and X dimension limit allow it, retaining the portable
  128-invocation path. Against 128, five interleaved runs reduced SpEED
  0.442030 → 0.338136 ms at 1080p and 0.494683 → 0.396334 ms at 4K.
  Both variants produce identical scores/checksums and have direct matrix
  reconstruction, orthogonality, singularity and feature coverage.
- **Sliding CAMBI histograms (`1829702`, test extension `c13f3ea`).** Four
  output rows maintain exact integer histograms across 16 output columns,
  updating only entering/leaving samples. This follows Netflix's histogram
  reuse mechanism, with our cooperative workgroup layout and shared tile.
  A full contrast map now needs one dispatch, replacing the row slices. Shared
  storage is 15,108 bytes, below Vulkan's 16 KiB minimum. Independent resident
  CAMBI medians were 0.789446 (0.789301–0.789819) →
  0.669952 (0.669314–0.670154) ms at 1080p and
  5.691562 (5.685251–5.694248) → 2.409986 (2.407013–2.412187) ms at 4K.
- **Avoid redundant radix pooling (`c43b699`).** Derive the first zero-bin
  count without contending on an atomic for every zero, stop scans once an
  exact selected bucket is complete, and clear the next histogram in the
  selection dispatch. Pooling drops from 14 to 11 dispatches per scale with
  the same scratch allocation. Tiny positive values, signed zero and ties
  retain exact selection semantics. Independent resident CAMBI medians were
  0.789512 (0.789248–0.790234) → 0.636014 (0.635710–0.636301) ms at 1080p
  and 5.692464 (5.690333–5.695283) →
  5.456328 (5.454005–5.457669) ms at 4K.

Each retained optimization passed the 2% targeted-stage gate at both sizes.
The independent comparisons isolate changes; the combined table measures
their interaction. Shared reductions and tiling reuse patterns from our
existing VIF, motion and ADM implementation. No Vship code was copied.

## Tested alternatives not retained

These arms passed correctness checks; all comparisons used five interleaved
resident runs at both sizes, with scores/checksums checked in every log.

| alternative | targeted stage, 1080p before → after ms | 4K before → after ms | disposition |
| --- | ---: | ---: | --- |
| 32-invocation Jacobi workgroup | 0.441861 → 1.128549 | 0.494432 → 1.143296 | reverted |
| 64-invocation Jacobi workgroup | 0.441995 → 0.664226 | 0.494357 → 0.704363 | reverted |
| 256-invocation Jacobi workgroup | 0.441922 → 0.364922 | 0.494442 → 0.420528 | superseded by 512 |
| 32-column CAMBI tile with packed counters | 0.669712 → 0.671083 | 2.411405 → 2.576502 | reverted |
| fused horizontal/vertical CAMBI mode filter | 0.669749 → 0.670341 | 2.411981 → 2.429054 | reverted |

No cause for these regressions is established by stage timings alone.

## Accuracy and build checks

All 125 GPU tests and the ASAN/UBSAN GPU suite passed with no skipped tests.
Direct-window contrast oracles cover masks, odd image edges, windows 1/17/33,
useful-luma boundaries and dense adjacent histogram bins. Matrix tests cover
repeated eigenvalues, a small regular eigenvalue, exact rank deficiency and
nonfinite input; the existing nearly singular chroma and moving-sequence
reference checks also passed. Formatting and SPIR-V validation passed.
CTest passed 10/10. SSIMULACRA 2 and Butteraugli identity scores were exactly
100.00000000 and 0.00000000. The 300-pair SSIMULACRA 2 MCOS correlation gate
passed; its separately reported raw-score parity difference is the existing
reference divergence documented in the repository.

Full CID22 validation used the same 4,292 still pairs and verified input
payload hashes. Versus `5509ac2`, maximum/mean/p99 score drift was
0.00007618 / 0.000008006 / 0.00003364. Against pinned libvmaf,
maximum/mean absolute error was 0.004660 / 0.001698.
KRCC/SRCC/PCC were 0.56395213 / 0.76184520 / 0.71779577, versus the
reference's 0.56396690 / 0.76186042 / 0.71780187. These still-pair results
have zero motion. Numerical behavior has been validated on this NVIDIA driver;
AMD/Intel hardware was not available for this run.

The build race fixed by `54daa57` gave shader embedding a single producer
target. Fresh and incremental parallel Make builds each embedded once; Ninja
also rebuilt successfully. The shader count is now 92, including both
eigensolver variants.

Individual logs, stage tables, spreads, score CSVs and the reference-validation
manifest are under `artifacts/reports/vmaf-v1-opt2/`. The 1080p input hashes are in
[`vmaf-notes.md`](vmaf-notes.md). The 4K reference/distortion Y4M hashes are
`2584ec50663849149448e60a79148bfb0f201c0ca7e4a6390c726afffbd19003` and
`4600bae9899fac01bcad9e797101a034ddaa6989db920aac713d3ce5959bf719`.
