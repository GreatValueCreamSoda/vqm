# SpEED parallelism and 1080p VMAF measurements

Investigated on 2026-11-05, RTX 5070, 48 SMs, driver 615.78.08, GCC 16.2.1.
The current default SpEED path still has underfilled solver and pooling phases.
The old four-group solver description is stale: the current dispatch runs four
spectral systems and four independent Cholesky systems, eight groups combined.
A tested research backend distributes bisection across 100 groups and reduces
SpEED time by about 6–11% on the primary workloads. This is available parallel
work, not a measured SM-occupancy percentage. Its full-frame benefit is smaller.

The final current-production CTest run passes all 12 groups, recursive formatting
checks 211 maintained files, and peer review verified the numeric tables and
compatibility evidence. Production runtime, shaders, layouts, public headers and tests remain unchanged
in this investigation. The matrix problems are small, and the winning prototype
still needs portable capability selection and borrowed-device fallback before
it could become a production default. The maintained source and all reproduction
commands are in [`bench/experiments/speed_parallelism/`](../../experiments/speed_parallelism/README.md).

## What remains underfilled

The stage schedule was read from the current code, with all counts reflecting
both U/V channels and both reference/distorted matrices where applicable:

| Processing | Filtered chroma plane | Means | Covariance | Solver prepare/single dispatch | Pooling |
| --- | --- | ---: | ---: | ---: | ---: |
| Native 420, 1920×1080 | 60×33 | 100 | 1300 | 8 | 4 |
| Native 420, 3840×2160 | 120×67 | 100 | 1300 | 8 | 10 |
| Explicit 444, 1920×1080 | 120×67 | 100 | 1300 | 8 | 10 |
| Explicit 444, 3840×2160 | 240×135 | 100 | 1300 | 8 | 42 |

Covariance already has ample groups. Each small 25×25 system stays inside one
spectral workgroup, and only 25 invocations perform the existing independent
eigenvalue searches. Cholesky also uses at most 25 useful rows. On this GPU,
eight groups cannot keep 48 SMs busy on their own, and the four spectral groups
form a narrow tail. This is a scheduling bound; it does not establish the
average occupancy of SpEED or the complete VMAF pipeline.

Pooling assigns one invocation to each non-overlapping 5×5 block and groups
64 blocks together. Thus its 72 blocks/channel at native 1080p produce only
four groups in total; the 312 blocks/channel at 4K produce ten. Its forward
solves contain independent block work, but parallelizing it adds communication
and data-loading costs. See the [source review](research/speed-parallelism-review.md)
and [dispatch snapshot](../measurements/vmaf-speed-dispatch-counts.json).

## Profiling evidence and attribution limits

A new Nsight Systems capture of the current 4K v1 resident path records the
48-SM GPU, 503 command-buffer workloads, and 32,695 pipeline binds: 65 dispatches
per call. It includes GPU metric samples and Vulkan API/workload events. The
supported kernel-level Nsight Compute handoff cannot operate on this report
because it contains no CUDA kernel activity table. Recipe lookup also reports
a local skill-pack `SKILL.md` content-hash mismatch. No private-cache analysis,
ad hoc sampled-counter joins, per-shader utilization percentage or stall cause
was substituted for that missing supported analysis.

An isolated diagnostic `vqm_bench` adds seven GPU timestamp brackets inside
SpEED. These extra queries perturb execution, so the values below are for
attribution, not ordinary performance comparisons. Each is the median of five
per-run stage reports, in microseconds:

| Substage | Native 1080p | Native 4K | Ducks 1080p |
| --- | ---: | ---: | ---: |
| Filtering | 12.11 | 41.95 | 11.33 |
| Means | 3.16 | 6.46 | 3.17 |
| Covariance | 8.16 | 27.37 | 8.15 |
| Solver | 48.45 | 48.69 | 46.05 |
| Pooling | 5.63 | 5.67 | 5.64 |
| Reduction | 1.43 | 1.57 | 1.43 |
| Combine | 1.40 | 1.25 | 1.40 |

The solver is the larger narrow phase. The public feature probe also finds
nonzero chroma SpEED on the main pairs; these are not all trivially skipped
pooling workloads. Public features alone do not identify every channel's
singular status or whether its solver used the QR fallback.

## Tested changes

All ordinary timing comes from `vqm_bench`. The full current working source was
captured before experimentation and built independently; the already-modified
working tree is not described by `HEAD` alone. Each comparison used five
interleaved A,B rounds in one sitting, 20 calls/3 warmups/resident 2, with
streaming separately measured at 8 calls/2 warmups. Scores, checksums, stages,
commands, source/input/module/binary provenance and run spreads are retained in
[compact results](../measurements/vmaf-speed-parallelism-results.json).
The data includes 29 comparison workloads and 290 valid interleaved runs,
plus the final 30-run 1080p sitting below. Negative changes mean faster.

| Resident control | SpEED 1080p | SpEED 4K | SpEED ducks | Paired scores/checksums |
| --- | ---: | ---: | ---: | --- |
| Certified prefix, diagonal gate | +0.33% | +0.07% | -0.02% | Exact |
| Certified prefix, Sturm gate | +0.88% | +0.47% | +0.78% | Exact |
| 100-group split bisection | -11.47% | -6.33% | -9.90% | Exact |
| Two subgroups per pooling group | +2.72% | +2.87% | +2.97% | Small changes |
| 32-lane solver groups | +19.52% | +11.82% | +20.18% | Exact |
| 64-lane solver groups | +8.01% | +4.82% | +8.17% | Exact |
| Subgroup ranks, original grid | -2.29% | -1.27% | -0.91% | Exact |

The pooling prototype exposes 72 groups at native 1080p and 312 at 4K by using
one full 32-lane subgroup per block. It reuses dead means/covariance storage
only when the score bank fits before the live factor region, and reconstructs
the old two-level sum graph. It passes the existing checks, but measured small
score changes and slower execution. More available work alone did not make it
a useful optimization.

The 32/64-lane controls preserve the eight-group grid and reported scores.
They are slower. Earlier 128-versus-512 tuning was already documented; the
current 32/64 trials specifically tested the later Sturm/Cholesky solver.

The certified-prefix variants try to establish that the existing QR fallback
is inevitable before computing eigenvalues that would be discarded. The
certificate follows an exact prefix of the existing rank-zero midpoint search;
a Rayleigh or Sturm predicate only decides whether to try it. These variants
pass correctness but provide no speedup on the measured pairs. A separate
Rayleigh-only heuristic was compiler/CPU checked, not adopted or GPU benchmarked.

## Strongest result: split the bisection work

The phase prototype preserves the existing Householder, Cholesky, singular
guard and QR arithmetic. It records:

1. Eight prepare workgroups, producing four tridiagonal systems and factors.
2. 100 bisection workgroups, one per eigenvalue rank and matrix, each with a
   complete 32-lane subgroup.
3. Four finishing groups, applying the existing strict threshold, safety band
   and QR fallback before pooling.

Each subgroup evaluates the 31 internal nodes of a five-level binary trial
tree. Repeated FP32 midpoint operations reconstruct the exact node endpoints;
five subgroup decisions then follow the same path as five original bisection
steps. Equally spaced interpolation would not preserve that path. Invalid
unused speculative nodes do not affect the selected path. This increases total
arithmetic while reducing dependency latency through more independent groups.
The original prepare and finishing phases remain narrow.

The backend adds two dispatches and 56 auxiliary FP32 words per matrix,
896 bytes for all four matrices. Host barriers separate publication, search,
finish and pooling; original covariance reads are not overwritten while
Cholesky is in flight. The [solver review](research/speed-solver-parallelism.md)
and [CPU tree snapshot](../measurements/vmaf-speed-solver-tree.json) explain
the control-flow and lifetime checks.

Resident SpEED reductions were 11.47% at native 1080p, 6.33% at native 4K,
9.90% on ducks, 11.28%/6.07% on the RGB 1080p/4K pairs and 10.47% on the
photo-labelled 1080p pair. Streaming reductions were 11.51%, 6.35% and 9.54%
on the primary three pairs. Explicit 444 processing of the native sources
improved SpEED by 6.85% at 1080p and 2.77% at 4K. All these paired reported
scores/checksums are exact. Full-frame changes are much smaller and have their
own run spreads; they must not be inferred from SpEED percentages alone.

A lower-complexity single-dispatch subgroup variant keeps the old grid and
scratch, assigning a subgroup to each rank inside its workgroup. It improves
SpEED only 2.29% at 1080p, 1.27% at 4K and 0.91% on ducks. Its SM coverage
ceiling is unchanged; it does not meet the stage speed threshold consistently.

## Correctness and remaining integration work

All seven tested algorithm/control variants pass the 204-test required-GPU
suite with Vulkan synchronization validation and no validation errors. The
phase tree additionally reroutes the direct factorization, adversarial-spectra
and legacy-classification tests through all three phases; the independent
forced-legacy QR shader remains unchanged. Assertions and tolerances were not
loosened.

The historical seed-113 dataset was regenerated byte for byte. Across all
10,000 near-threshold matrices, both split and single-dispatch subgroup solvers
match the current original solver's classifications and all 25 FP32 eigenvalue
bit patterns. FP64 truth near the cutoff is approximate; this is a compatibility
check, not a claim of exact mathematical classification. The 268-case CPU tree
traversal check and 270-case prefix diagnostic are additional numerical evidence.

All 288 unclipped synthetic frames, covering both VMAF versions at
8/10/12/16 bits, match baseline scores and every recorded feature exactly for
the split solver. All 4,292 CID22 RGB pairs processed through explicit GPU 420
conversion have identical reported eight-decimal scores and MCOS correlations.
That comparison retains normal clipping and is an optimization-drift check,
not a fresh libvmaf CPU-reference parity claim.

The prototypes currently assume the measured device's complete 32-lane
compute subgroups and shuffle/size-control support. The narrow-prepare option
still uses subgroup bisection; it is not a portable fallback test. Production
integration would need feature/property negotiation, borrowed-device promises
or a genuine original-kernel fallback, and sanitizer/software-driver and
multi-device qualification. Those paths were not added to the default library.
The source review also found a stale conservative 2500-group covariance limit
in the v1 planner, versus the current 1300 actual groups; it has no impact on
this GPU's minimum-supported dispatch capacity and was not mixed into the study.

## Requested 1080p VMAF benchmarks

These are a fresh final sitting, five rounds interleaving the six arms. One
worker compares a fixed repeated 1920×1080 10-bit YUV420 pair. Decode and
construction are excluded; upload-inclusive includes per-call upload. Resident
uploads both images before timing and still performs the VMAF feature work.
Reported rate is `1000 / median_ms`, not multi-worker aggregate throughput.
The score is stable within each arm. See the
[full 1080p run snapshot](../measurements/vmaf-1080-2026-11-05.json).

| Implementation | Path | Median ms | Five-run range ms | Comparisons/s |
| --- | --- | ---: | ---: | ---: |
| VMAF v0.6.1 float | Resident | 0.376935 | 0.376505–0.380034 | 2653 |
| VMAF v1.0.16 | Resident | 0.654920 | 0.653985–0.666750 | 1527 |
| VMAF v0.6.1 float | Upload-inclusive | 0.752134 | 0.742684–0.767394 | 1330 |
| VMAF v1.0.16 | Upload-inclusive | 1.212462 | 1.208422–1.216497 | 825 |
| VMAF v1 phase prototype | Resident | 0.648380 | 0.646715–0.655836 | 1542 |
| VMAF v1 phase prototype | Upload-inclusive | 1.208058 | 1.197603–1.211927 | 828 |

The default library is the current implementation in the table. The phase rows
are the isolated research backend, not an enabled production switch. In this
sitting the prototype moves v1's resident median from 0.654920 to 0.648380 ms
(about 1.00%) and upload-inclusive from 1.212462 to 1.208058 ms (about 0.36%).
The ranges overlap; the per-stage interleaved trials establish the SpEED gain
more clearly than whole-frame latency. V0 does not use SpEED.

Raw report files, logs, binaries, generated shader variants, stress CSVs,
source snapshots and temporary-build provenance are kept under ignored
`artifacts/reports/vmaf-speed-parallelism/` and `artifacts/snapshots/`. Shared
conclusions, compact evidence and reusable probes live in maintained source
paths. Temporary source/build copies can be discarded after preservation.
