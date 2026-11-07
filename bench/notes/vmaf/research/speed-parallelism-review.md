# Current SpEED work distribution

Source review on 2026-11-05, before any new parallelism experiment. This note
derives launch counts and identifies candidates; it supplies no new GPU timing,
average occupancy, bandwidth or stall measurement. The
[compact dispatch snapshot](../../measurements/vmaf-speed-dispatch-counts.json)
records the reviewed source hashes, dimensions, formulas and candidate counts.

## What remains underdistributed

The older [bottleneck report](../vmaf-v1-bottlenecks.md) described four solver
workgroups. The later [pooling and parallelism changes](../vmaf-v1-pooling-parallelism.md)
separated eigenvalue extraction and Cholesky factorization, so the current
`VmafSpeedStats::Record` launches eight groups: four matrices times two kinds
of work. Four groups run the spectrum path and four run Cholesky. Even with
the optional 512-invocation shader, the grid contains only eight groups.

On the 48-SM device identified by the earlier report, at most eight SMs can
execute that dispatch's groups concurrently, falling to at most four while
only spectral groups remain. This is a grid-size ceiling, not a measurement of
average SM occupancy or utilization over the entire frame. More invocations in
one group do not distribute that group over more SMs.

Pooling is also narrow. Its current 64-invocation group assigns one independent
5×5 block to each invocation. Each invocation performs two sequential
25-component triangular solves and two entropy sums. At native 4:2:0 1080p,
there are only 72 blocks per chroma channel and four groups in the entire pool
dispatch. Native 4:2:0 4K has 312 blocks per channel and ten pool groups.

The current launch counts below include both U/V channels and reference /
distorted planes. Filter dimensions use integer division by 16, pooling crops
each dimension by division by five, and each pool group covers 64 blocks.

| Input and chroma mode | Filtered plane | Nine-tap vertical | Each other filter pass | Means | Covariance | Solver/factor | Pool | Final reduce/combine |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1920×1080, 4:2:0 | 60×33 | 408 | 48 | 100 | 1300 | 8 | 4 | 2 / 1 |
| 3840×2160, 4:2:0 | 120×67 | 1360 | 160 | 100 | 1300 | 8 | 10 | 2 / 1 |
| 1920×1080, 4:4:4 | 120×67 | 1360 | 160 | 100 | 1300 | 8 | 10 | 2 / 1 |
| 3840×2160, 4:4:4 | 240×135 | 4860 | 540 | 100 | 1300 | 8 | 42 | 2 / 1 |

Thus a whole-stage statement such as “SpEED uses only four SMs” would be
incorrect: covariance already exposes 1300 independent groups, and most filter
passes have dozens or hundreds. The solver and small pool dispatches still
have structural underutilization. Timings must establish their share of the
critical path before choosing a redesign.

## First candidate: cooperate on independent pooling blocks

A subgroup can solve one block while preserving the row-wise subtraction
sequence. Let lane r own the input residual for row r. For k = 0..24:

1. Lane k divides its residual by `L[k,k]`, obtaining `z[k]`.
2. Publish `z[k]` to the participating lanes.
3. Each lane r > k updates its residual with `residual[r] -= L[r,k]*z[k]`.

Each row sees exactly the same preceding k values in ascending order as the
existing serial forward solve. A designated lane can retain the original
ascending norm and entropy sums. Reference and distorted inputs keep separate
state and the final score expression stays unchanged. This is an
arithmetic-preserving algorithmic schedule; compiler contraction and generated
code still require actual oracle and exact-output comparisons.

One block per group would expose 144 groups at 4:2:0 1080p and 624 at 4K.
Two subgroup-owned blocks per group give 72 and 312 groups; four give 36 and
156. The two-block mapping is a sensible first tradeoff: it can cover the
48-SM device without the maximum repetition of factor loads.

The repeated loading is a real cost. The current group stages two 25×25
factors and two 25-component spectra once for as many as 64 blocks. Smaller
groups stage them for fewer blocks. A wider grid alone does not establish a
speedup. The current dynamically indexed per-invocation `solution[25]` also
deserves code-generation investigation, but this source pattern alone does
not prove register spilling or a particular stall.

The original partial-sum tree should remain independently visible. Producing
one score per block and then applying the same 64-block halving tree preserves
the existing grouping, at the cost of another dispatch and a small score bank:
576 bytes at 4:2:0 1080p and 2496 at 4K. Changing the block count per reduction
instead alters floating-point addition order and needs its own drift gate.
Do not hide that arithmetic change inside a work-distribution claim.

A subgroup implementation must negotiate the operations and subgroup-size
requirements it actually uses and retain a portable implementation. Padding
lanes and tail blocks must participate in required publication and
synchronization; invalid blocks produce zero without reading out of bounds.
Singular/imputed channels retain their existing status precedence. These are
part of the contract, not optional benchmark shortcuts.

## Other options and their limits

- **Split tridiagonalization from eigenvalue bisection.** The four Householder
  reductions remain matrix-dependent, but the 100 eigenvalues can then be
  assigned to more groups. This requires publication, another dispatch and
  care with QR fallback. Only 100 independent scalar Sturm sequences exist;
  spreading one per group can lose SIMD efficiency while increasing the SM
  count. A larger occupied-SM number is not sufficient evidence of useful
  throughput.
- **Narrow Cholesky, keep wide spectral groups.** Only the first 25 lanes do
  useful Cholesky row work, yet the current shared shader can launch 512 lanes
  for those groups. Separate local sizes may reduce overhead, but require
  either a new dispatch/pipeline or another explicit schedule. They do not
  increase the count of independent matrices, and cannot improve the critical
  path when factorization already finishes before spectral extraction.
- **Shorter scalar pooling groups.** This is a simpler distribution control,
  though it repeats factor loads, retains each solve's serial dependency chain
  and normally changes the partial-sum grouping. It is useful as an isolated
  experiment, not proof that subgroup cooperation pays.
- **Batch independent videos.** More independent sequences supply more
  matrices, but change the workload and ownership schedule. Aggregate wall
  time over a fixed pair count is the relevant metric. This does not by itself
  improve one sequence's latency.

Do not alias tridiagonal scratch over covariance while independently launched
Cholesky groups can still read covariance. There is no implicit inter-group
barrier. New stage boundaries, scratch lifetime and status publication must be
explicit. The existing QR guard protects singular classification; a redesign
must preserve it rather than relying only on final clipped scores.

## Prior work to avoid repeating blindly

The [algorithm optimization report](../vmaf-v1-algorithm-optimization.md)
already retested 128 versus 512 after the eigenvalue-only / Cholesky redesign;
the tuning arm failed and the 512 choice remained. Earlier Jacobi local-size
experiments are in the
[pipeline optimization report](../vmaf-v1-pipeline-optimization.md). The
[later speed investigation](../vmaf-speed-investigation.md) retained shared
pooling factors and 325-entry cooperative covariance; the older serial-entry
symmetry rejection does not describe the current 64-lane reductions.

The [cooperative-matrix study](../vmaf-cooperative-matrix.md) also showed why
more groups or faster covariance arithmetic need not improve the full frame.
Its FP16 experiments are distinct from the FP32 work-distribution candidates
above. Current performance claims must come from fresh interleaved
`vqm_bench` runs, with each score, checksum and stage table inspected.

One minor stale source description remains outside this review's write scope:
the v1 memory planner's dispatch-limit comment and conservative X-grid constant
still describe 1250 covariance groups per channel / 2500 total, while the
current triangular dispatch has 650 / 1300. Update that description, limit
check and boundary tests together if changing the dispatch contract.

Only this research note and its compact source-derived snapshot were added by
this review. No GPU work or production edits were performed.
