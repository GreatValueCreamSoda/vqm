# Current SpEED solver parallelism review

Source investigation on 2026-11-05. This note reviews the maintained FP32 solver
and proposes controlled experiments. It contains no new GPU timings or measured
occupancy claim. The current implementation is
[`vmaf_speed_stats_eigen.comp`](../../../../src/metrics/vmaf/speed/shaders/vmaf_speed_stats_eigen.comp),
recorded by
[`VmafSpeedStats::Record`](../../../../src/metrics/vmaf/speed/vmaf_speed.cc).

## The structural utilization limit is still present

The standard two-channel SpEED call records `Dispatch(4, 2, 1)` for the solver.
The X coordinate selects reference/distorted U/V; Y selects spectral work or
Cholesky. There are eight workgroups in total: four spectral and four factor
groups. On the study's 48-SM GPU, those groups can occupy at most eight SMs at
once, and the spectral-only tail at most four. These are grid-size ceilings,
not measured average occupancy or activity percentages.

The current wide path allocates 512 invocations per workgroup where device
limits allow it, with a 128-invocation portable variant. Useful parallelism is
uneven within each group:

| Phase | Useful work |
| --- | --- |
| Matrix load and Householder trailing update | Up to 625 independent element assignments; extra lanes shorten element loops. |
| Householder norms/vector setup and scalar correction | One invocation retains ordered sums and decisions. |
| Householder matrix-vector product | At most 24 independent rows, falling as the trailing matrix shrinks. |
| Sturm eigenvalue bisection | 25 independent invocations; each count has a sequential 25-pivot recurrence. |
| Guarded QR fallback | One invocation per spectral matrix. |
| Cholesky | One diagonal invocation and at most 24 independent rows, falling each column. |

The normal spectral path executes 99 workgroup barriers, including 92 in 23
Householder steps. The Cholesky path executes 51. These counts follow source
control flow, not a profiler's hardware barrier counter. Different workgroup
sizes do not change their count or the number of independent matrices.

Concurrency and parallel bisection have already improved this code considerably.
The [earlier parallelism report](../vmaf-v1-pooling-parallelism.md) records
four concurrent Cholesky groups and guarded parallel eigenvalues replacing
serial QR in regular cases. The old description of only four solver workgroups
predates concurrent factorization; current code has eight.

## Low-risk control: tune the current kernel's lane count

[`solver_lanes.py`](../../../experiments/speed_parallelism/solver_lanes.py)
changes only the wide shader declaration in a fresh isolated source copy:

```sh
python3 bench/experiments/speed_parallelism/solver_lanes.py \
  /absolute/isolated/source --lanes 32
```

Choices are 32, 64, 128, 256 and 512. The installer refuses this maintained
checkout and its ancestors/descendants. It leaves the shader expressions,
scratch layout, eight-workgroup dispatch and portable 128-lane blob unchanged.
The caller must configure/build the isolated copy normally and use the real
`vqm_bench` under the interleaved protocol. No feature extension or public API
change is required for these workgroup sizes on the study device.

The motivation is reduced idle-warp and synchronization cost, not wider SM
coverage. Matrix element loops get longer with fewer lanes, so a speedup is not
assumed. Per-entry arithmetic is intentionally unchanged: independent matrix
assignments are redistributed, while norms, row products, bisection pivots and
Cholesky row updates retain their original expression and iteration order.
Compiler contraction and device behavior still require actual correctness
checks; source reasoning is not a bit-identity guarantee.

There is an important historical limit on this proposal. The original
[512-lane improvement](../vmaf-v1-pipeline-optimization.md) used an earlier
Jacobi/eigenvector algorithm. However, the later
[Householder/QR investigation](../vmaf-v1-algorithm-optimization.md) explicitly
retried 128 versus 512 and rejected it: SpEED regressed approximately 1.0–1.4%
with matching scores. Repeating that exact historical arm is not new evidence.
The present parallel Sturm phase and independently dispatched Cholesky branch
came later; 32/64 lanes on this current algorithm are different experiments.
128 can serve as a current control, with its prior rejection reported.

All five lane variants compile and pass `spirv-val --target-env vulkan1.1`.
Commands and module hashes are retained under ignored
`artifacts/reports/vmaf-speed-parallelism/solver/compiler-checks.json`.
No GPU execution or performance measurements were performed by this review.

## Separate narrow Cholesky is a control, not a multi-SM solution

[`solver_cholesky.comp`](../../../experiments/speed_parallelism/solver_cholesky.comp)
extracts the existing Cholesky branch into a 32-lane workgroup. Its diagonal,
row-subtraction and final-write expressions retain the maintained order. It
compiles and passes SPIR-V validation, but is not installed into production or
GPU-qualified.

To test it, a caller would compile this extra blob, run the spectral solver with
Y size one, then dispatch `2*channels` Cholesky groups before pooling. Both
passes need completion before consumers. A separate command adds launch cost;
it does not create more independent factorization work, and ordinary separate
dispatches may serialize previously concurrent branches. It is therefore a
lower-priority control unless a profile specifically identifies the fat
Cholesky workgroup as costly. The isolated all-phase lane-count control above
is simpler and avoids another launch.

## Increasing actual solver SM coverage needs a different decomposition

Simply adding workgroups that repeat the same matrix work increases activity
without removing latency. Giving one scalar eigenvalue to each workgroup also
wastes 31 lanes and multiplies issued SIMD instructions, so a larger grid alone
does not establish useful utilization.

A possible more ambitious design is:

1. Four groups perform the unchanged Householder reduction while four factor
   groups finish Cholesky, then publish each tridiagonal system and bounds.
2. Assign one 32-lane group to each of the 25 eigenvalues, producing 100 groups
   for four matrices. Its lanes evaluate the 31 internal trial midpoints of a
   depth-five binary bisection tree in parallel. Select the same five decisions
   the baseline would make; preserve each midpoint's FP32 operation order,
   early adjacent-endpoint convergence and invalid-pivot handling. This trades
   roughly six times as many Sturm evaluations for more SM-level work and fewer
   dependent iterations on each group.
3. A final per-matrix pass applies the existing scale-relative guard and reruns
   the unchanged serial QR when required. It publishes the original status
   precedence only after all eigenvalue outputs are complete.

The implemented research variant of this sketch is documented below; this
source review performed no GPU execution. The design needs auxiliary
tridiagonal/status storage and extra dispatches, and may lose because each
matrix has only 25 components. Padding, speculative trials, barriers and launch
cost count against any reduction in dependency-chain latency. Original
Householder state must be retained for fallback, and independent factorization
must not race a reuse of covariance storage. The 10,000 threshold-focused
tridiagonal stress cases from the previous study and all metric/oracle/corpus
gates remain necessary; removing the QR guard is not a valid shortcut.

For throughput across independent videos, batching more independent matrices
could fill additional SMs without duplicating arithmetic. That requires an
explicit batched API/runtime and independent temporal histories; it is a
different workload from reducing one sequence's latency and must be measured
as aggregate wall clock over a fixed frame count.

## Implemented three-phase research variant

[`solver_phases.py`](../../../experiments/speed_parallelism/solver_phases.py)
now installs the split solver into a fresh isolated source copy. The earlier
sketch's prepare/final shaders are generated from the maintained source, retaining
its Householder, Cholesky, guard and QR expressions rather than carrying a
second independently edited implementation. The separate maintained
[`solver_phase_bisect.comp`](../../../experiments/speed_parallelism/solver_phase_bisect.comp)
implements the speculative binary tree. Nothing is installed into the maintained
core by these tools.

```sh
python3 bench/experiments/speed_parallelism/solver_phases.py \
  /absolute/fresh/isolated/source
```

The integration contract is:

- Existing two-matrix bank offsets retain their 2,604-word stride. Auxiliary
  state starts at `pc.scratch + 2604*channels`; each of the `2*channels` matrices
  owns 56 words. `ScratchWords()` becomes `2604 + 2*56` per U/V pair, adding
  896 bytes in total for the ordinary two-channel call. `pc.op` temporarily
  carries the channel count for these three phases, then returns to 1 before
  pooling.
- Prepare keeps the existing eight-workgroup grid, including concurrent
  Cholesky. It writes 25 diagonal and 24 off-diagonal values, lower/upper bounds,
  the preparation failure state and a separate invalid-input flag. The retained
  covariance is not overwritten while Cholesky might still read it.
- Bisection uses `Dispatch(25, 2*channels, 1)`, with one complete 32-lane
  subgroup per workgroup. Lanes 0–30 represent breadth-first tree nodes 1–31.
  Every lane independently reconstructs its node's bounds with repeated FP32
  midpoint operations. Uniform subgroup shuffles then select exactly one path
  of at most five baseline decisions. Equally spaced interpolation is not used.
  An unused node's invalid count is ignored; an invalid selected count requests
  fallback. The original 320-test convergence limit is retained.
- A nonconverged or invalid rank writes a NaN marker into its eigenvalue slot.
  The finish phase gathers these markers before any pool consumer and forces
  the original QR path, which overwrites the spectrum. The marker replaces no
  externally accepted eigenvalue; it carries internal failure state between
  phases. Early invalid covariance is handled separately without reading an
  uninitialized tridiagonal system.
- Finish uses four workgroups, recomputes the original strict-threshold Sturm
  count, applies the original minimum-eigenvalue/scale guard, and reruns the
  extracted serial QR when required. It preserves the original spectral versus
  Cholesky status precedence. Compute barriers separate all three phases and
  the subsequent pooling dispatch.

The installer requests `VK_EXT_subgroup_size_control`, enables
`subgroupSizeControl` and `computeFullSubgroups`, and sets required subgroup
size 32 plus `REQUIRE_FULL_SUBGROUPS` on the bisection pipeline only. It also
explicitly enables the existing FP16 extension while chaining these private
features, matching the local-device pooling prototype. The selected study
GPU's subgroup shuffle support and these features must be checked first. This
is an owned-default-device research promise, not production portability or a
borrowed-device feature contract.

Full installer smoke checks succeeded. Generated prepare variants with 128/512
invocations, bisection with 32 and finish with 32 compile and pass SPIR-V
validation. The original solver blobs remain in the build as independent test
oracles. All fragile baseline anchors are checked before any source is written.
No GPU timing or correctness claim follows from compilation.

### Direct GPU spectral qualification must target the new phases

The original `EigenDevice` fixture directly names the old full solver blobs.
Merely installing the new metric path does not route its matrix reconstruction,
adversarial spectra and legacy classification comparisons through the new
algorithm. Apply the separate research test installer before using those tests
as phase-solver evidence:

```sh
python3 bench/experiments/speed_parallelism/solver_phase_tests.py \
  /absolute/already/phase-installed/source
cmake --build /absolute/already/phase-installed/source/build --target vqm_test -j8
```

The helper changes only that isolated test source. Ordinary `EigenDevice`
instances allocate 2,716 words, set `pc.op=1`, and execute prepare `(2,2,1)`,
bisection `(25,2,1)` and finish `(2,1,1)` with barriers. The original factor,
spectrum and status readback offsets and every assertion/tolerance remain
unchanged. Devices requested with `force_legacy_qr=true` continue to run only
the original forced-QR blob, providing an independent comparison. The destructor
cleans all three optional pipeline handles.

The installer refuses the maintained checkout, non-phase sources and a second
application. Its fragile test anchors are checked before writing. Applying it
changes no benchmark/library source or binary; the benchmark hash was checked
unchanged on the study tree. The root owner schedules the actual GPU run.

### Independent CPU interval traversal check

Run:

```sh
python3 bench/experiments/speed_parallelism/solver_tree_check.py
```

The diagnostic compares sequential NumPy FP32 midpoint decisions against a
breadth-first tree using separately rounded scalar binary32 operations. All
268 cases passed, including 256 deterministic random finite intervals,
adjacent normal/subnormal endpoints, opposite-sign near-maximum endpoints,
signed zeros, equal endpoints, and invalid counts on selected versus unused
nodes. It compares the bit patterns of both final endpoints and midpoint,
convergence/invalid flags and exact number of midpoint tests. Source/input bits,
seed and per-case outcomes are retained in ignored
`artifacts/reports/vmaf-speed-parallelism/solver/tree-checks.json`.

This checks the tree traversal responsibility, not real Sturm conditioning,
GPU contraction, denormal handling or end-to-end metric parity. The numerical
and reference gates remain required for the actual device execution.

## Single-dispatch subgroup-per-rank control

[`subgroup_solver.py`](../../../experiments/speed_parallelism/subgroup_solver.py)
installs a lower-complexity comparison in a fresh isolated source copy:

```sh
python3 bench/experiments/speed_parallelism/subgroup_solver.py \
  /absolute/fresh/isolated/source
```

It keeps the original eight-workgroup grid, one solver dispatch, 2,604-word
scratch-bank layout and host recording sequence. During bisection, each
complete 32-lane subgroup owns one rank and runs the same 31-node trial tree
as the phase variant. The 512-lane workgroup has 16 subgroups and processes
25 ranks in two batches; the 128-lane workgroup has four subgroups and needs
seven batches. There are no workgroup barriers inside a rank's subgroup search.
The original whole-workgroup barrier still completes every rank before the
minimum-eigenvalue guard and QR fallback.

The original main function before bisection, including Householder and
Cholesky, remains byte-identical in the generated source. Its main-function
suffix, including the guard and QR fallback, also remains byte-identical.
Eigenvalue/status statements are copied with only the rank identifier changed;
subgroup leaders publish them, and only rank zero publishes the minimum and
strict-threshold count. The total source workgroup-barrier count is unchanged.
The trial-tree helper is extracted from `solver_phase_bisect.comp` so the
speculative midpoint traversal has one maintained implementation.

Only the ordinary `vmaf_speed_stats_eigen` and `_wide` CMake blob rules are
redirected to the generated `solver_subgroup.comp`. The original shader file
and forced-legacy-QR blob stay unchanged. Consequently the existing direct
`EigenDevice` normal/wide tests automatically exercise this new algorithm,
while forced-legacy comparisons keep their independent baseline oracle. The
phase-specific test installer must not be applied to this variant.

The owned research device enables subgroup size control/full subgroups; required
size 32 and the full-subgroup stage flag apply only to the two new algorithm
blob names. Other shader contracts are unchanged. Native compute shuffle and
these size-control features remain explicit local-device requirements, not a
portable production promise.

This control uses more of the existing workgroup's warps, but still caps the
spectral grid at four SMs. It is a latency experiment, not wider matrix-level
SM coverage. It can also lose: speculative counts multiply arithmetic, shared
arrays still feed four matrices, and the two/seven rank batches lengthen the
critical path. The 128/512 generated modules compile and pass SPIR-V validation;
source-boundary preservation, installer guards, original-source retention and
reapply rejection passed. No GPU run was made by this source review. Compiler
and source checks are under ignored
`artifacts/reports/vmaf-speed-parallelism/solver/subgroup-*-checks.json`.
