# PSNR-HVS parallel reduction and lookup-table optimization

2026-11-06. Investigated why the initial resident PSNR-HVS path was slower
than expected. The adopted changes reduce fixed-count wall time by 44% at
1080p and 51% at 4K in interleaved baseline/candidate runs. A separate matched
comparison on the same input pairs places the optimized path ahead of
SSIMULACRA 2. All signal processing remains FP32. Test integration and corpus
ranking qualification remain deferred at the user's request.

## Baseline, method and provenance

The baseline is the entire uncommitted implementation immediately before this
optimization. HEAD `a949b6177efb88890dd44860cc30dff45486a25d` alone lacks
PSNR-HVS. A detached worktree at that commit received a copy of the current
maintained source, including untracked files, and was configured and built
separately with the same GNU 16.2.1 Release/warnings-as-errors configuration.
Its executable SHA-256 exactly matched the initial implementation binary:
`c2dd56d79652c9ae916bab6690e872d8938c44d31aa411b355719657abc962e9`.

Raw results, candidate shaders/patches, source fingerprints and build metadata
are under `artifacts/reports/psnr-hvs-optimization-20261106/`. Before removing
the temporary worktree, source hashes were checked and its maintained build
inputs archived as `artifacts/snapshots/psnr-hvs-baseline-20261106.tar.gz`.
The original executable is retained in the raw report's `baseline/` directory.
Historical commands retain their original temporary worktree paths.

Hardware was NVIDIA GeForce RTX 5070, driver 615.78.08. Inputs are unchanged
first frames from the synthetic 1080p and 4K YUV420 8-bit clips documented in
the [original resident measurement](resident-throughput-20261106.md).
Full input hashes were rechecked after measurement. The distortion quantizes
luma to multiples of eight; chroma is unchanged. Raw input storage is 8-bit;
all DCT, variance, masking, pooling and score arithmetic is FP32.

Each five-run series alternated baseline/candidate within each resolution,
using `--resident 2 --workers 1 --iterations 20 --warmup 3`. Both images are
prepared every call. Construction, decode and upload are outside the timing.
Each pair is reused, so results describe a hot-memory workload. Wall throughput
uses the total fixed-count duration; CSV call latency is the median of the
20 individual calls. Stage values are medians of five per-run stage means.
All runs and their spreads were retained. These are unprofiled timings from
the instrumented benchmark; no hardware stall or physical spill counters were
collected.

## Mechanisms investigated

The original block pass delegated the whole variance and masking calculation
to two lanes and its coefficient-error sum to one lane. The original global
reduction dispatched one workgroup per plane, serially walking many thousands
of block values. SPIR-V inspection additionally found three function-private
192-float arrays for CSF lookup and two 64-float arrays for the DCT basis:
2,816 logical bytes of indexed function arrays per invocation. This establishes
the intermediate representation, not physical local-memory spilling.

Adopted changes:

- Hierarchical reduction uses multiple 128-lane groups targeting 512 inputs
  each, then one group reduces the compact group sums. Small inputs retain a
  single pass. Original block counts still supply the score denominator.
- Eight lanes compute independent quadrant means and centered variances;
  sixteen lanes compute short masking rows; eight lanes compute error rows.
  Short final sums combine those partials. The unchanged FP32 matrix DCT and
  centered variance formulas remain explicit. The shader uses nine barriers
  and 2,000 bytes of shared data.
- The 192 CSF and 64 DCT constants are published once as read-only arena
  metadata. All 256 FP32 bit patterns match their previous GLSL values.
  Binding 2 reads metadata; raw pixels continue to use binding 3. Direct and
  staged initialization are both accounted for in memory planning.

The final block SPIR-V has no function variables, eliminating those indexed
private arrays. Its floating-point types remain exclusively 32-bit.

## Intermediate and rejected approaches

An all-lane shared tree for quadrant statistics, masking and block error raised
the barrier count from five to 28. In five interleaved runs, its block stage
changed 0.301398 to 0.310634 ms at 1080p and 1.196994 to 1.241347 ms at 4K.
It was rejected and the original shader restored before the next experiment.

Hierarchical plane reduction alone passed the speed gate: median reduction
time fell 0.041958 to 0.003635 ms at 1080p and 0.324434 to 0.005182 ms at 4K.
Its resident throughput changed 2,409 to 2,593 pairs/s and 542 to 667 pairs/s,
respectively. The block stage remained essentially unchanged.

The short-row/quadrant block organization alone showed a modest preliminary
gain. Moving its lookup tables to arena metadata produced the larger block
gain measured below. A symmetric four-pair DCT candidate was prepared but not
executed or adopted; the final arithmetic retains the original eight-term
matrix DCT.

## Final interleaved baseline comparison

| Resolution | Baseline wall pairs/s, median (range) | Optimized wall pairs/s, median (range) | Median call ms, baseline to optimized |
| --- | ---: | ---: | ---: |
| 1920x1080 | 2429.697 (2246.999-2442.268) | 4334.279 (4072.819-4474.313) | 0.409720 to 0.223302 |
| 3840x2160 | 531.704 (495.547-547.791) | 1074.714 (1051.410-1104.795) | 1.835870 to 0.905887 |

The median fixed-count wall times were 0.411574 to 0.230719 ms at 1080p
(-43.94%) and 1.880747 to 0.930481 ms at 4K (-50.53%). Throughput factors
were 1.784x and 2.021x. Every optimized run was faster than every baseline
run at its resolution. The first 4K baseline run was slower than later runs;
it remains included in the reported range.

| Stage, ms/call | 1080p baseline | 1080p optimized | 4K baseline | 4K optimized |
| --- | ---: | ---: | ---: | ---: |
| Prepare | 0.034685 | 0.034528 | 0.255584 | 0.255886 |
| DCT/masking | 0.301496 | 0.154613 | 1.223405 | 0.605045 |
| Reduction | 0.041957 | 0.003605 | 0.333453 | 0.004968 |
| Host score | 0.001076 | 0.000964 | 0.001226 | 0.001142 |

The targeted block stage improved 48.72%/50.54%; reduction improved
91.41%/98.51%, clearing the repository's 2% target-stage speed gate. The
optimized owned allocations increased by 2,176 bytes at 1080p and 5,056 bytes
at 4K, to 116,630,720 and 466,517,312 bytes. The planner includes all table
and intermediate reduction storage.

Scores/checksums were stable within every arm. 1080p changed from
`40.20762634` to `40.20761871` dB; 4K from `40.35023117` to `40.35021210` dB.
The maximum observed change here is about 0.0000191 dB. Reassociated reductions
cause this expected numerical difference; the unchanged constant-table bit
patterns were verified separately. All checksums match 23 calls including
warmup. No identical-score acceptance requirement was introduced.

## Matched SSIMULACRA 2 comparison

A separate five-run series alternated optimized PSNR-HVS and SSIMULACRA 2
using the same executable, native YUV pairs and resident flags, with half
planes disabled. These are new measurements, not comparisons to historical
RGB results. Both metrics prepare both images every call. SSIMULACRA 2
legitimately includes color conversion, chroma reconstruction and its own
resident-slot policy, while PSNR-HVS evaluates coded planes.

| Resolution | Optimized PSNR-HVS pairs/s, median (range) | SSIMULACRA 2 pairs/s, median (range) | Call latency ms, PSNR-HVS / SSIMULACRA 2 |
| --- | ---: | ---: | ---: |
| 1920x1080 | 4248.064 (4214.976-4442.056) | 2254.7 (2124.0-2286.4) | 0.225437 / 0.437240 |
| 3840x2160 | 1097.579 (1073.151-1112.925) | 587.7 (574.2-591.3) | 0.900763 / 1.687642 |

SSIMULACRA 2's console throughput has one decimal place; CSV call latencies
retain six decimals. The table preserves those reporting precisions.

## Correctness checks and limits

Release build, maintained formatting (257 files) and clang-tidy (52 production
translation units) passed. The existing manual public API probe was rebuilt
against the optimized core and passed direct/staged identity, memory-query
agreement, streamed/resident/allocated inputs and ownership/token guards.
Its direct/staged arenas were both 25,152 bytes; staging was 0/9,584 bytes.
The changed-image score remained `30.9334907532` dB in both policies.

Three-frame 420/8-bit, 422/10-bit and 444/12-bit CLI checks passed for differing
and identical inputs. Their finite clip scores matched the initial smoke
values; all identities retained infinite scores with exactly zero errors.
No maintained test, corpus policy or validation fixture was added or modified.
No ranking corpus, sanitizer or maintained GPU test suite was run in this
optimization task because test integration remains deferred.

Compact per-run measurements, stage ranges, hashes and manual-check results
are retained in [`optimization-20261106.json`](optimization-20261106.json).
The unchanged formulas and manual checks do not establish corpus-wide ranking
agreement. Results cover this GPU, these synthetic pairs and one worker;
streaming throughput, other devices and unseen-content behavior require
separate evidence.

Conclusion: large private lookup arrays and serialized reductions were
avoidable implementation costs. The adopted FP32 organization approximately
doubles 4K resident throughput and outperforms SSIMULACRA 2 on the matched
native-input workload without adding test integration.
