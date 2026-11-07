# Exact CAMBI pooling headroom

Review date: 2026-11-05. This review considers the current five-scale batched
CAMBI radix pool and preserves its floating-point reduction order and canonical
threshold/rank state. No production source was changed and no GPU execution or
performance measurement was performed for this candidate. The retained
[compact evidence](../../measurements/vmaf-v1-pool-zero-sum.json) records the
compiler versions, source hashes and CPU checks. The candidate is an experiment,
not an established speed improvement.

## Current cost and previously tested approaches

`src/metrics/vmaf/cambi/vmaf_cambi.cc` records one initial clear, four histogram
scans interleaved with four selectors, a partial sum, and a final sum: eleven
dispatches across all five independent scales. Packed workgroup coordinates
avoid launching a full rectangular grid for each unequal-size scale. Histograms,
selection state, partials and results remain disjoint between scales.

These earlier changes already address the obvious dispatch and histogram ideas:

- [Pooling and parallelism](../vmaf-v1-pooling-parallelism.md) reduced 55
  dispatches to eleven. Two-pass 16-bit radix required 512 KiB of scratch and
  regressed CAMBI; three-pass 11/11/10-bit radix lost to packed batching.
- [CAMBI optimization](../vmaf-v1-cambi-optimization.md) retained a cooperative
  sixteen-lane exact selector. Its historical diagnostic radix measurements were
  0.022589 ms at 1080p and 0.058870 ms at 4K after that change. These are dated
  substage measurements, not a current bound or a fresh baseline.
- [Pipeline optimization](../vmaf-v1-pipeline-optimization.md) already recognizes
  zero first-byte buckets and completed selections, and clears histograms in
  selectors. Completed scales skip subsequent value scans, but their last
  partial-sum dispatch still reads every map element.
- [Algorithm optimization](../vmaf-v1-algorithm-optimization.md) records other
  changes to exact pooling. Shared histogram aggregation was measured and
  rejected in [the optimization review](../vmaf-v1-optimization.md).

There is no basis here for claiming a large general gain from pooling. The new
candidate removes one scan only when the selector can prove it sums zero;
it retains all eleven dispatches.

## New candidate: skip a provably zero final scan

The existing final scan adds values whose unsigned magnitude key is strictly
greater than the selected threshold. The finish shader adds the remaining
threshold ties. Two exact facts can prove that the scan contributes zero:

1. On the first byte, the histogram counts every nonzero high byte and the
   state prefix ORs every nonzero key whose high byte is zero. A zero count
   outside bin zero **and** a zero OR prefix prove every input magnitude is
   exactly zero. The OR condition is essential: tiny normal and subnormal
   values can have a zero high byte.
2. On the last byte, the threshold key is exact. The residual rank equals the
   requested selected count minus the number of keys strictly greater than
   that threshold. If residual rank equals the selected count, that number is
   zero. Every selected value is a threshold tie.

After clearing the histogram, the modified selector stores this Boolean proof
in histogram word zero. Active intermediate passes leave it zero. An all-zero
map completes on the first byte; later histogram workgroups already return
before reading or writing the completed scale, and later completed selectors
preserve word zero. A last-byte proof has no subsequent histogram scan. The
next pooling operation's initial clear resets the word.

The modified partial shader checks the proof uniformly within its workgroup.
When set, invocation zero writes positive floating zero to that group's partial
and every invocation returns before reading values or entering reduction
barriers. Otherwise, the existing scan and reduction run unchanged.

This preserves the original result bit for bit for supported finite maps:
the omitted scan initializes all leaves to positive zero, contributes no
values, and its fixed addition tree therefore produces positive zero. Finish
uses the original partial-sum order, threshold, residual rank, multiplication
and division. It does not replace the pooled answer with a differently ordered
mean. The threshold/rank state is unchanged: the independent pool tests inspect
both raw words, so hiding a sentinel bit in the threshold would violate an
observed internal contract.

The proof is deliberately incomplete. Constant maps whose selection completes
before the last byte may still take the original scan. Mixed zero/positive
maps never receive the all-zero proof. A missed proof changes no result and
only leaves the original work in place. This adds no host branch, allocation,
layout field, shader extension, extra dispatch or public API behavior.

## Reproduction and checks

The installer refuses this maintained checkout, its ancestors and descendants.
Use a fresh isolated copy of the current working source; a checkout of `HEAD`
alone may omit the current uncommitted source reorganization.

```sh
python3 bench/experiments/v1_headroom/pool_zero_sum.py /path/to/isolated-source
python3 bench/experiments/v1_headroom/pool_zero_sum_check.py
python3 bench/experiments/v1_headroom/pool_zero_sum_compile.py
```

The [installer](../../../experiments/v1_headroom/pool_zero_sum.py) validates all
source anchors before writing either file. The
[CPU probe](../../../experiments/v1_headroom/pool_zero_sum_check.py) models the
existing radix decisions and checks the proof against independently sorted
integer magnitude keys. It covers lengths 1–8193, zeros, tiny values, threshold
ties, mixed maps, scattered finite keys, four selection ratios and 600 seeded
random cases. All 880 canonical threshold/rank states matched. Of these,
154 produced the new proof; every one independently had zero values above its
threshold. These synthetic counts are coverage, not estimates of occurrence in
real CAMBI maps. Seed: 20261105; NumPy: 2.5.4.

The [compiler probe](../../../experiments/v1_headroom/pool_zero_sum_compile.py)
copies the current pool shaders into a temporary isolated source tree, installs
the candidate, formats the two changed files with clang-format 23.1.4, compiles
them with glslc 2026.4 targeting Vulkan 1.1, and runs `spirv-val`. Both modules
passed. Reapplying the installer and targeting the maintained checkout were
both rejected. All three scripts parse and import without running their main
operation; the CPU probe also reproduced identical JSON from another working
directory with a space-containing output path.

Raw outputs default to
`artifacts/reports/vmaf-v1-headroom/pool-zero-sum/`. Compilation validates syntax
and SPIR-V structure; the CPU model checks integer decisions. Neither establishes
GPU visibility, execution equivalence or performance. The next required gates
are the unchanged pool oracle/state tests, full GPU test suite, reference
feature/sequence checks, formatting, and five interleaved benchmark runs per
arm. Include a zero map, a dense active map, tiny values, ties, scratch reuse,
both packed and forced-rectangular dispatch layouts, 1080p and 4K. The relevant
stage must improve by at least 2% under the repository protocol before this is
described as an optimization. Inspect scores and checksums for every run.

## Other exact avenues reviewed

| Avenue | Reason not implemented here |
| --- | --- |
| GPU indirect dispatch for completed scales | Can suppress some remaining launch work, but requires an indirect-capable arena, argument buffers, indirect-read barriers and active-scale compaction. Empty launches remain in the current candidate. Profile their cost before adding this cross-cutting machinery. |
| Fuse initial clear with a many-workgroup histogram | A normal compute dispatch supplies no inter-workgroup barrier ensuring all bins were cleared before another group increments them. Separate ordered dispatches supply the required boundary. |
| Fuse final selection with every partial group | Duplicating exact selection per group repeats histogram reads and control work; a single selector workgroup cannot globally synchronize independent partial groups inside the same dispatch. No safe, low-cost formulation was found. |
| Cross-scale partial aggregation | Current five scales already dispatch together, and each has its own selected threshold. Combining their arithmetic would change the established reduction order and lifetime boundaries. |
| Selected-count-one maximum shortcut | Potentially useful for tiny synthetic maps; the default production top-k fraction selects many pixels even at the smallest normal scale. It also needs to retain canonical threshold/rank behavior, not only the final value. |
| Selected-count-equals-count mean shortcut | A plain full-map sum reassociates the existing above-threshold sum and threshold-tie contribution. It would need an explicit numerical qualification and does not apply to the ordinary selection fraction. |
| Copy histograms to the CPU for selection | Introduces a host visibility wait and another GPU submission in a pipeline that currently resolves selection entirely on the GPU. It offers no demonstrated advantage and worsens traceability of the synchronization contract. |

The best small exact experiment is therefore the zero-above-threshold scan
shortcut. Its practical value depends on how often real scales meet the proof
and how much of current CAMBI time belongs to that final scan. Those remain GPU
measurement questions; no performance conclusion follows from the CPU results.
