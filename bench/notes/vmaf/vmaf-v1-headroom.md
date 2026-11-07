# VMAF v1 remaining performance headroom

2026-11-05, RTX 5070 (48 SMs), NVIDIA driver 615.78.08.

There is still useful exact-arithmetic headroom, but this pass found an
incremental gain: the retained row-local CAMBI synchronization reduces the
primary 1080p/4K resident frame medians by **1.86% / 2.09%**, with CAMBI
**5.03% / 3.92%** faster. The current implementation is close to the easy
local cleanup gains tested here; these results do not establish a hardware or
algorithmic ceiling. Most seemingly cheaper memory/layout variants were
neutral, and some made execution substantially slower.

The [SpEED study](vmaf-speed-parallelism.md) separately found a portable-
qualification-pending solver prototype worth about 1% of the 1080p frame.
It remains research. Its gain must not be added to this change without an
integrated comparison.

## Method and provenance

The frozen baseline includes the entire pre-change working source, including
uncommitted reorganization, rather than just HEAD `a949b61`. Its benchmark
SHA-256 is `72b11a0251fbeba4fe3a85191b0e14738c79f13b40e92f8fcdf14425eaec1f8a`.
The baseline and candidate have separate Release builds with warnings as
errors. All performance numbers come from ordinary `vqm_bench` stage timestamps
and frame timers. Extra diagnostic timestamps are used only for attribution.

Each comparison interleaves A,B for five rounds: resident `--resident 2
--iterations 20 --warmup 3`; streaming separately uses `--iterations 8
--warmup 2`. One worker, fixed pairs repeatedly evaluated, decode excluded,
both inputs prepared each call; temporal motion becomes zero after the first
call. Ranges are min–max of five run averages, not per-frame percentiles.
The target-stage gate is at least 2% faster. No profiler or heavy build ran
alongside ordinary timed comparisons.

The 1080p/4K primary pairs are 10-bit YUV420. Ducks is a single 8-bit YUV420
1080p compressed frame. Dense stress pairs have constant luma 100 versus 101
and chroma 512, and are not a typical image workload. Extra RGB/dense resident
arms explicitly request 4:2:0; the original native pairs retain native format.
Input hashes, every score/checksum/stage, binary/source fingerprints and all
spreads are preserved in [compact evidence](../measurements/vmaf-v1-headroom-results.json).
There are **15 configurations, 51 A/B workloads and 510 ordinary benchmark
runs**, plus 20 diagnostic runs. Every ordinary comparison's score/checksum
matched. Raw logs/builds are ignored under `artifacts/reports/vmaf-v1-headroom/`.
The exact adopted patch and [reproduction tools](../../experiments/v1_headroom/README.md)
allow reconstructing the pre-change baseline in an isolated copy.

## Current CAMBI attribution

Five diagnostic runs per workload replace the enclosing CAMBI bracket with
six non-nested categories. The 1080p/4K command uses 44 timestamp queries;
production retains its existing aggregate bracket and no extra timers.
Extra query boundaries perturb execution, so these totals explain the search
and are not the retained speed gate.

| diagnostic category | 1080p µs | 4K µs | ducks 1080p µs | dense 1080p µs |
| --- | ---: | ---: | ---: | ---: |
| prepare | 58.91 | 321.65 | 63.12 | 58.19 |
| decimate, five scales | 14.05 | 69.14 | 14.08 | 13.98 |
| two mode filters, five scales | 22.02 | 49.49 | 22.18 | 21.88 |
| contrast, five scales | 152.81 | 670.07 | 55.48 | 258.46 |
| batched pooling | 37.04 | 102.61 | 37.02 | 37.35 |
| combine | 1.23 | 1.25 | 1.24 | 1.24 |

CAMBI has thousands of workgroups at the primary geometries; its distribution
is different from SpEED's four-group spectral tail. The adopted change keeps
the CAMBI grid unchanged. No new utilization counters were captured here, so
the timing gain does not establish a particular stall reason or occupancy
change.

## Retained change

`vmaf_cambi_contrast.comp` still constructs its shared tile and exact integer
histograms with workgroup-wide publication and clone barriers. After construction,
each logical row exclusively owns one histogram and four FP32 contrast terms.
The two barriers in each sliding-column iteration synchronize only that row
when a complete 32-lane subgroup owns it.

The subgroup shader maps the cooperative index to
`gl_SubgroupID * 32 + gl_SubgroupInvocationID`; it does not assume physical
local-index order. A workgroup-uniform check requires subgroup size 32 and
eight subgroups for 256 live invocations. Other packing uses the original
local-index mapping and workgroup barriers. The standalone probe actually
observed **32 lanes, eight subgroups, every logical slot 0..255 exactly once**
on the measured GPU, so the row path was exercised.

Two optional shader IDs preserve the existing portable uint/half IDs. The host
selects them only for default subgroup size 32 with compute BASIC operations.
BASIC requires no newly enabled logical-device feature under the existing
Vulkan 1.1 contract. Float16 shared storage remains independently guarded by
the enabled `shaderFloat16` promise, including borrowed devices. There are no
FP16 subgroup arithmetic operations, new public API fields, arena changes,
extra dispatches or temporal-history changes. Original portable uint/half
SPIR-V is **byte-for-byte identical** to the frozen baseline. The active shader
manifest grows from 114 to 116 variants.

Direct tests can disable subgroup selection and independently disable half
storage. New tests cover eligibility properties and exact filtered/contrast
maps across six portable/automatic/borrowed combinations. Existing window,
mask, histogram, boundary and libvmaf-oracle cases remain unchanged.
Independent review found no blocking correctness issue; see the
[storage/synchronization proof](research/v1-storage-dispatch-headroom.md).
The eight-lane software device selects the portable module. The dynamic
nonmatching-packing branch within the subgroup module was reviewed but was
not observed executing on these devices.

| workload | baseline ms/frame, median (range) | retained ms/frame, median (range) | frame change | CAMBI change |
| --- | ---: | ---: | ---: | ---: |
| dense1080 resident | 0.864286 (0.863677–0.866127) | 0.838042 (0.829867–0.838487) | -3.04% | -6.92% |
| dense4k resident | 3.531466 (3.485697–3.545835) | 3.409323 (3.402154–3.466722) | -3.46% | -4.57% |
| rgb1080 resident | 0.679664 (0.676380–0.684195) | 0.663215 (0.661395–0.673520) | -2.42% | -5.34% |
| tenbit1080 resident | 0.655814 (0.655325–0.660020) | 0.643600 (0.641546–0.645285) | -1.86% | -5.03% |
| tenbit4k resident | 2.501632 (2.486247–2.515916) | 2.449403 (2.438873–2.468082) | -2.09% | -3.92% |
| ducks1080 resident | 0.560311 (0.557482–0.562476) | 0.555586 (0.555396–0.556246) | -0.84% | -1.86% |
| tenbit1080 streaming | 1.214492 (1.206302–1.217646) | 1.195707 (1.191872–1.201071) | -1.55% | -5.11% |
| tenbit4k streaming | 4.697593 (4.687468–4.719333) | 4.655114 (4.631989–4.681509) | -0.90% | -3.94% |
| ducks1080 streaming | 0.843062 (0.833697–0.846442) | 0.840832 (0.837702–0.843687) | -0.26% | -2.05% |

The final ducks resident CAMBI result is −1.86%, slightly below the stage gate
on that workload, with no frame regression. The primary 1080p/4K resident and
streaming arms pass the targeted-stage gate; RGB and dense arms improve too.
Different rounds/sittings must not be compared as interchangeable absolutes.

## Independent rejected/control arms

Percentages below are CAMBI median changes; negative is faster. Each row has
its own interleaved baseline. The final capability-selected result above was
remeasured independently from the earlier fixed-device subgroup prototype.

| arm | 10-bit 1080p | 10-bit 4K | ducks 1080p |
| --- | ---: | ---: | ---: |
| barriers | +0.02% | +0.05% | +0.05% |
| conditional128 | -2.40% | -0.09% | -1.84% |
| mode | -0.17% | -0.05% | -0.14% |
| packed | -0.29% | -0.20% | -0.06% |
| pool | +0.16% | +0.14% | +0.22% |
| radius | +0.07% | +0.08% | -0.03% |
| radius-tail | -0.31% | -0.40% | -0.16% |
| rows4 | +15.78% | +19.95% | +2.01% |
| rows4-128 | -2.47% | +4.12% | -1.85% |
| rows4-128-small | -2.61% | +3.24% | -1.91% |
| subgroup | -5.06% | -3.91% | -2.24% |
| subgroup-skip | -2.05% | -1.28% | -1.26% |
| tail | -0.11% | -0.48% | -0.33% |
| wide | +1.15% | +32.89% | +9.45% |

- `tail`: reuse unused histogram words for terms/active mask. Exact, but misses
  the stage gate. `radius` reduces source-tile capacity for windows ≤17;
  `radius-tail` combines both. Neither passes.
- `mode`: compile a separate mode-only pipeline without shared allocations;
  neutral. `wide`: fully buffered 32-column half tile, distinct from the
  historical ring/refill tile; it regresses substantially at 4K.
- `pool`: skip a provably zero sum-above-threshold scan; exact but neutral on
  these inputs. `packed`: exact uint16 global horizontal/filtered banks with
  unchanged reservations; neutral. `barriers`: remove duplicate component
  entry barriers while retaining producer barriers; no GPU-stage gain.
- `rows4`: four output rows with 256 invocations; slower. `rows4-128` and
  `rows4-128-small` use 128 invocations and optionally compact capacity/tail
  reuse. They improve 1080p but regress 4K. Their unconditional planner also
  fails the existing synthetic dispatch-boundary test by conservatively
  rejecting a device whose original integer fallback fits. These three are
  failed research controls, not retained changes.
- `conditional128` preserves the original planner and chooses the smaller
  half tile only for small windows when the actual grid fits. Its 204-test
  suite passes, but the benefit is smaller than the retained row-sync path.
- `subgroup-skip` additionally returns inactive row subgroups. It is correct
  on the GPU suite but slower than row synchronization alone; not retained.

All eight initial arms passed 204 GPU tests each. The two fixed-device subgroup
arms and conditional128 also passed 204 each. The three unconditional four-row
controls each had the one planner-boundary failure described above; their
metric checks passed. No test or numerical tolerance was weakened.

Supporting reviews preserve [contrast variants and CPU ownership proofs](research/v1-contrast-headroom.md),
[independent tail/radius review](research/v1-contrast-independent-review.md),
[pooling proof](research/v1-pooling-headroom.md), and
[remaining luma/ADM/motion opportunities](research/v1-other-stages-headroom.md).
The latter identifies modest controls, not established speedups, and documents
prior failed packing, halo, preweighting and reduction experiments.

## Final qualification

The retained production build passes 206 GPU tests, ASAN/UBSAN GPU checks,
12 CTest groups, 213-file formatting, 41 compiled-production-file lint checks,
and hardware/software synchronization validation. The software Vulkan device
also passes all 206 tests through the portable selection. Both validation runs
report zero validation errors. SSIMULACRA 2 / Butteraugli identities remain
`100.00000000` / `0.00000000`. All 288 unclipped sequence rows match in every
reported feature and score across both implementations, six scenes and
8/10/12/16-bit inputs.

All 4,292 original CID22 RGB still-pair scores match the frozen baseline at
eight printed decimals with explicit 4:2:0 conversion. This is a drift check
against the current GPU implementation, not a new libvmaf CPU-reference
comparison. Direct libvmaf feature/model oracles remain in the GPU suite.
Moving frames and unclipped sequence features are checked separately.

## Fresh final 1080p measurements

After qualification, five rounds interleaved the four retained-production arms
below on the same 10-bit YUV420 pair. One worker, native format, decode excluded,
resident 20/3 and streaming 8/2 as above. These are absolute current timings;
use the earlier A/B table for the measured optimization gain.

| metric and path | median ms/frame (five-run range) | pairs/s |
| --- | ---: | ---: |
| vmaf-v0-resident | 0.377609 (0.377030–0.397754) | 2648 |
| vmaf-v1-resident | 0.641675 (0.640296–0.641995) | 1558 |
| vmaf-v0-streaming | 0.746899 (0.743149–0.749749) | 1339 |
| vmaf-v1-streaming | 1.193138 (1.189918–1.200757) | 838 |

The repeated still pair does not measure video decoding, unique-frame memory
working sets or moving-frame throughput. The sequence checks establish temporal
correctness separately. Remaining opportunities in the supporting reviews need
their own numerical/portability qualification and interleaved speed gate; they
are not promised additional speedups.
