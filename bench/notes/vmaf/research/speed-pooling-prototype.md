# SpEED pooling with one subgroup per block

Source and compiler investigation, 2026-11-05. This note accompanies the
experimental [`pool_subgroup.comp`](../../../experiments/speed_parallelism/pool_subgroup.comp)
and [`pool_merge.comp`](../../../experiments/speed_parallelism/pool_merge.comp).
It records a candidate work distribution, not a measured speed improvement.
No GPU work was run for this review, and no production file was edited.

## Why pooling is a candidate

The current `vmaf_speed_stats_pool.comp` assigns one 5×5 patch to each invocation
in a 64-invocation workgroup. Every invocation performs two 25-row triangular
solves, two 25-term entropy sums and the final block score. Its workgroups are
therefore relatively long, but there are few of them after the SpEED filter's
16× downsampling.

The following counts describe two pooled channels, U and V. They use the
current integer geometry and omit any fit/pad conversion that changes it:

| Input and processing | Filtered chroma | 5×5 blocks per channel | Current pooling workgroups | Four-subgroup candidate workgroups |
| --- | --- | ---: | ---: | ---: |
| 1920×1080, 4:2:0 | 60×33 | 72 | 4 | 36 |
| 3840×2160, 4:2:0 | 120×67 | 312 | 10 | 156 |
| 1920×1080, 4:4:4 | 120×67 | 312 | 10 | 156 |
| 3840×2160, 4:4:4 | 240×135 | 1296 | 42 | 648 |

For a 48-SM GPU, four workgroups cannot populate more than four SMs at once.
This is a structural work-distribution limit for that dispatch, not a measured
whole-frame utilization percentage. More workgroups alone do not establish a
speedup: the candidate duplicates factor staging more often and introduces
shuffle dependencies.

## Per-block arithmetic

One full 32-lane subgroup owns one patch. Lane r owns triangular-solve row r;
the seven lanes beyond row 24 participate in every shuffle but own no row.
The input is the original filtered patch, matching the production pooling
code. It is not centered again using covariance means.

Each row starts from its filtered value. For k=0..24, lane k divides its
current value by L[k,k] and publishes z[k] with `subgroupShuffle`. Every later
row r subtracts `L[r,k]*z[k]` from its own value. Thus row r retains the original
subtraction sequence k=0..r−1. Lane zero accumulates z[k]² in the original
k=0..24 order. No dot-product or sum reassociation is introduced intentionally.

After the variance is published, the 25 entropy terms are computed concurrently
by their row lanes. Lane zero receives each term in the original order using
25 shuffles, preserving the scalar entropy add sequence. Only lane zero
computes the final threshold and score. Both image components use the same
existing factor, eigenvalue and singular-status banks.

These are logical arithmetic-order properties. Compiler contraction,
instruction selection and transcendental lowering still require same-device
comparison; compilation alone does not prove byte-identical feature values.

## Output and synchronization modes

`SPEED_POOL_SUBGROUPS` accepts 1, 2 or 4 in the compile-checked variants. The
local size is 32 times that count. Each workgroup stages the two 25×25 factors
and two spectra once. Every edge invocation participates before the workgroup
barrier. The valid-block and regular-status conditions are subgroup-uniform;
all 32 lanes of a live subgroup execute every shuffle.

`SPEED_POOL_WRITE_SCORES` selects the following layouts:

| Value | Output | Required host work |
| ---: | --- | --- |
| 0 | One reduced partial per workgroup in the existing `pc.partials` channel layout | Allocate and dispatch `ceil(blocks/subgroups)` groups per channel, and set `pc.groups` accordingly. This changes the global score-add tree from the original grouping of 64. |
| 1 | One score per block at `pc.partials + channel*blocks + block` | Allocate `channels*(blocks+old_groups)` words. Dispatch the supplied merge shader with `old_groups=ceil(blocks/64)`, `pc.groups=old_groups`, `pc.op=channels`; it writes old-tree partials after the raw-score bank. Before the unchanged final reduce, shift `pc.partials` by `channels*blocks` and restore `pc.op`. |
| 2 | One score per block at `bank(2*channel)+kMeans+block` | Reuse dead means/covariance storage only after all eigensolver and factorization work has completed. Require `blocks <= kCholesky` (1300), retain the factor/spectra/status regions, and merge using a compatible final-reduce shader. |

Mode 1's merge reconstructs each original 64-score halving tree, including
zero-filled edge lanes. The unchanged final reduction then retains its original
order over those group partials. Mode 2 enables the same reconstruction without
an extra raw-score allocation if the final reduction is adapted; that combined
reduce is outside these two prototype files.

The scratch alias in mode 2 is valid only because pooling reads factors,
eigenvalues, source patches and status, while means/covariance have no further
consumer. The entire preceding eigensolver dispatch must complete before any
pooling workgroup overwrites those banks. A compute dependency is required
between raw-score publication and its reducer. The 1300-word region ends
immediately before the Cholesky bank; exceeding it would corrupt live factors.

The four-subgroup shader uses 5200 bytes for shared factors and spectra, plus
16 bytes of score partials in mode 0. The original kernel uses the same 5200
bytes plus 256 bytes of score partials. This source layout does not establish
the driver's physical allocation or occupancy.

## Subgroup capability contract

The shader requests `GL_KHR_shader_subgroup_basic` and
`GL_KHR_shader_subgroup_shuffle`. Before selecting it, the host must establish
compute-stage shuffle support and full 32-lane subgroups. Extension presence
or a workgroup size divisible by 32 is not sufficient evidence that all source
lanes used by shuffles are active. The current local-device experiment can
require explicit subgroup-size/full-subgroup pipeline controls; a production
backend needs capability and enabled-feature handling, including borrowed
devices, plus the original portable shader as fallback.

The [GLSL subgroup extension](https://github.com/KhronosGroup/GLSL/blob/main/extensions/khr/GL_KHR_shader_subgroup.txt)
defines shuffle semantics, and the
[Vulkan subgroup rules](https://docs.vulkan.org/spec/latest/chapters/shaders.html#shaders-subgroup)
define subgroup size and active-invocation behavior. Unlike cooperative
matrices, this arithmetic uses ordinary FP32 shader operations and subgroup
value exchange; it does not require a matrix feature or FP16 arithmetic.

## Compiler checks and next gate

All nine combinations of subgroup count 1/2/4 and output mode 0/1/2 compile
with `glslc -O --target-env=vulkan1.1` and pass
`spirv-val --target-env vulkan1.1`. The separate old-tree merge shader passes
the same checks. Compiler commands, module hashes and disassembly are under
`artifacts/reports/vmaf-speed-parallelism/pool-compiler/`; generated modules are under
`build/speed-parallelism-probes/`.

For example, compile the four-subgroup scratch-alias arm from the repository
root after creating its disposable output directory:

```sh
mkdir -p build/speed-parallelism-probes
glslc -O --target-env=vulkan1.1 \
  -DSPEED_POOL_SUBGROUPS=4 -DSPEED_POOL_WRITE_SCORES=2 \
  -I src/vulkan/shaders -I src/metrics/vmaf/speed/shaders \
  bench/experiments/speed_parallelism/pool_subgroup.comp \
  -o build/speed-parallelism-probes/pool-4-2.spv
spirv-val --target-env vulkan1.1 build/speed-parallelism-probes/pool-4-2.spv
```

The next gate is an isolated same-device comparison against the existing
pooling path: original numerical oracles, edge/singular cases, batched channels,
memory and dispatch-limit contracts, validation layers, unclipped temporal
sequences and interleaved `vqm_bench` stage measurements. The one/two/four-subgroup
arms should be compared separately. A higher workgroup count should not be
reported as a performance gain without that measurement.
