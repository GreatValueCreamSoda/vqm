# VMAF v1 storage and dispatch headroom

Investigated 2026-11-05 against the current uncommitted VMAF implementation.
This review owns two independent, reproducible installers; it changes no
production source and performs no GPU work. It does not establish a speedup.
The parent investigation owns interleaved GPU measurements and the final
retention decision. Earlier RTX 5070 measurements below are historical evidence,
not new measurements from this review.

## Questions and priorities

Can CAMBI move fewer exact integer codes through global memory? Can its
composition avoid recording the same full compute dependency twice? Both are
smaller, more isolated candidates than another mode-filter fusion or changing
all eight integer-valued banks at once.

The highest-confidence correctness change is removing immediately repeated
entry barriers inside the CAMBI composite. The larger traffic experiment is
packing only the horizontal and filtered mode banks as `uint16_t`, while
preserving FP32 arithmetic and existing arena reservations. Neither candidate
should be described as faster until the applicable stage improves at least 2%
under the repository's interleaved protocol.

## Two exact packed mode banks

The preparation shader reconstructs 8/10/12/16-bit coded samples and rounds
them to 10-bit codes. A 16-bit maximum can round to 1024. Mode filtering selects
an existing value, so every horizontal and filtered output remains an integer
in 0..1024. These values round-trip exactly through unsigned 16-bit storage.
This is integer compression; it introduces no FP16 arithmetic or approximation.

The isolated installer is
[`storage_mode_banks.py`](../../../experiments/v1_headroom/storage_mode_banks.py).
It generates separate packed contrast and decimation shaders from the current
sources, retaining provenance and leaving the original shader modules intact.
The default standalone contrast/decimation constructors and buffer contracts
remain FP32. Only `VmafCambi` enables packed mode banks when the Vulkan context
reports the already-enabled `storageBuffer16BitAccess` feature. Existing
`shaderFloat16` negotiation independently selects the retained half shared tile.

The data flow is deliberately narrow:

1. Every contrast input remains FP32. Scale zero consumes preparation or its
   initial high-resolution decimation.
2. Horizontal mode output is stored as uint16. The vertical mode pass reads
   those codes, performs the unchanged `mode3` operation in FP32 and stores its
   filtered result as uint16. Top/bottom rows still use the original input.
3. Local contrast reads the filtered codes as uint16, converts them to FP32/
   integer exactly, and keeps its current integer histograms and FP32 contrast.
4. Later decimation reads that filtered bank as uint16, then writes the existing
   FP32 scale-input bank. Its mask input/output and odd-edge coordinates do not
   change. The next contrast therefore again starts with FP32 input.

Bindings 0 and 2 both describe the arena at byte zero
(`src/vulkan/vulkan_runtime.cc:479–488` in the reviewed source). Packed shaders
use a typed uint16 array at binding 2. A bank whose existing offset is `o`
32-bit words places code `i` at uint16 index `2*o+i`, or byte offset `4*o+2*i`.
The original FP32-capacity reservation remains disjoint from every other bank;
its second half is unused. Each invocation writes a distinct 16-bit element.
This avoids the read/modify/write races that naive two-codes-per-uint stores
would introduce on neighboring outputs and odd tails. V1's existing storage
buffer byte-range check also bounds doubled word offsets within uint32 indexing.

No installed API, public memory query, arena allocation, synchronization point,
push-constant layout, histogram or pooling expression changes in this arm.
Typed accesses of two banks and three shader consumers are sufficient. If a
measured traffic win warrants retention, a separate follow-up can compact the
two allocations and update memory-layout tests; that is intentionally excluded
from the first comparison.

### Analytical sizes, not bandwidth measurements

The high-resolution speedup begins at 1920×1080 input, so the largest contrast
map is 960×540 at 1080p and 1920×1080 at 4K. These two banks therefore differ
substantially from the earlier all-eight-bank estimate in
[`vmaf-fp16.md`](../vmaf-fp16.md).

| quantity | 1920×1080 input | 3840×2160 input |
| --- | ---: | ---: |
| contrast window | 17 | 33 |
| five scale pixels in total | 690,600 | 2,762,160 |
| arena saving in this unchanged-reservation arm | 0 B | 0 B |
| possible later two-bank allocation saving | 2,073,600 B | 8,294,400 B |
| logical mode-pass bytes removed by smaller H/F loads/stores | 6,883,680 B | 27,576,960 B |
| filtered source bytes removed, all-active grid upper bound | 8,366,592 B | 83,047,680 B |

The last row counts source-tile probes for every contrast workgroup, including
out-of-bounds halo calls that return before a memory read. Sparse center masks
also return before loading the halo. It is therefore an upper bound on logical
load bytes, not a VRAM traffic count or a throughput prediction. Caches,
transaction widths, conversion instructions, histogram contention and barrier
cost can erase any benefit. Mask loads remain FP32 in this small arm.

Packing all three full-resolution preparation banks and five scale banks could
save the previously estimated 17,625,600 B/70,502,400 B at 1080p/4K. That requires
updating preparation, mask generation, every decimator, mode filtering and
contrast together, plus layout/memory tests and fallback coverage. The two-bank
arm should price the simpler benefit before taking that larger scope.

## Repeated component-entry dependencies

`VulkanRuntime::ComputeBarrier` uses the same full compute-shader write to
compute-shader read/write dependency each time
(`src/vulkan/vulkan_runtime.cc:619–624`). CAMBI's standalone components retain
both an entry barrier and a trailing barrier. In `VmafCambi::Record` the
producer's trailing barrier and the next consumer's entry barrier consequently
have no intervening GPU dispatch. CPU validation, push preparation and function
calls introduce no GPU access between them.

The independent installer is
[`storage_component_barriers.py`](../../../experiments/v1_headroom/storage_component_barriers.py).
It adds `synchronize_input = true` defaults to decimation, contrast and batched
pool recording. Only the composite passes false. Standalone calls retain their
existing entry/exit contracts, and every producer's trailing dependency remains.

Reviewed source anchors in `src/metrics/vmaf/cambi/vmaf_cambi.cc`:

| retained producer barrier | repeated consumer entry | occurrences at 1080p/4K |
| --- | --- | ---: |
| preparation's final dispatch, line 95 | decimator entry, line 145 | 1 |
| prior scale's contrast tail, line 229 | next decimator entry, line 145 | 4 |
| decimator tail, line 150 | contrast entry, line 213 | 5 |
| final contrast tail, line 229 | batch-pool entry, line 363 | 1 |
| **total duplicate entries removed** | | **11** |

Below the high-resolution threshold scale zero does not decimate: the total
becomes 10 (four decimator, five contrast and one pool entries). Preparation's
entry barrier, every preparation/mode/histogram/reduction dependency, every
component's trailing barrier and the final combine barrier all remain. There
is still a dependency for every write→read edge and for reuse of filtered,
horizontal, scale-input and alternating mask banks. This does not rely on host
recording order to make GPU writes visible.

The driver may coalesce consecutive dependencies already. Removing eleven API
calls therefore does not prove eleven stalls disappear. This arm may matter
more to command recording at 1080p than to a long 4K contrast dispatch; only
measurement can establish its value. No queue, submission count, public sequence
semantics or number of GPU dispatches changes.

## Internal timing probe

[`cambi_stages.py`](../../../experiments/v1_headroom/cambi_stages.py) replaces
the isolated copy's aggregate CAMBI bracket with six accumulated categories:
`vmaf_v1_cambi_prepare`, `decimate`, `mode`, `contrast`, `pool` and `combine`.
The composite brackets preparation, each decimation, pooling and combination;
`VmafCambiContrast::Record` separately brackets its two mode dispatches and its
contrast dispatch. The V1 pipeline closes SpEED before CAMBI's first internal
range and opens motion only after CAMBI's last range closes. There are no
nested ranges and no overlapping aggregate CAMBI stage to double-count.

At 1080p/4K this records 18 CAMBI ranges (1+5+5+5+1+1), plus the four ordinary
luma/ADM/SpEED/motion ranges: 44 timestamp queries in total. Smaller inputs
without the initial high-resolution decimation use 42. The installer verifies
the existing capacity is sufficient and leaves it unchanged; the reviewed
runtime had 256 queries. Both budgets fit even a 64-query pool. Production
sources, dispatches, barriers, feature arithmetic and public APIs remain
unchanged. Added timestamps can perturb these small stages, so ordinary
uninstrumented arms remain the source for final speed claims.

The probe passed GCC syntax checks for `timing.cc`, CAMBI host code, the V1
pipeline and Vulkan runtime, each with and without `VQM_STAGE_TIMING`, using
the project's C++20 warning flags and `-Werror` (eight checks). It is installed
with `python3 bench/experiments/v1_headroom/cambi_stages.py /tmp/vmaf-stage-copy`.
Raw commands are in the ignored storage report directory's `stages-syntax.txt`.

## Row-local subgroup synchronization probe

[`subgroup_rows.py`](../../../experiments/v1_headroom/subgroup_rows.py) is a
separate research arm for the dominant contrast dispatch. It keeps one contrast
routine and the ordinary shader blob names; both the current uint and half
shared-tile variants compile it. The installer refuses the maintained checkout.
It does not enable any new device feature or alter installed interfaces.

The isolated host constructor queries subgroup properties before creating a
pipeline and requires default subgroup size 32, COMPUTE-stage support and BASIC
subgroup operations. The shader additionally evaluates the workgroup-uniform
condition `gl_SubgroupSize == 32 && gl_NumSubgroups == 8`. Only that path uses
subgroup row IDs/lanes and subgroup barriers. Otherwise the same routine uses
its original local-index mapping and workgroup barriers. This avoids assuming
that a physical subgroup size alone guarantees complete active subgroups.
The [Khronos shader execution rules](https://docs.vulkan.org/spec/latest/chapters/shaders.html#shaders-subgroup-size)
describe command-uniform compute subgroup size and the separate full-subgroup
guarantees. No physical-device query was executed by this review.

The dependency proof is specific to the final 16-column sliding loop:

- Phase 2 enters `contrast_map` before main's pixel-bound early returns. All
  256 workgroup invocations therefore reach the subgroup-shape condition.
  Eight groups of 32 available slots contain all 256 real invocations, proving
  complete row groups. Logical index `row*32+lane` covers 0..255 exactly even
  if subgroup builtin order differs from physical local invocation order.
- Every cooperative initializer, center probe, tile load, dense histogram
  build and cross-row clone uses that logical index. All existing workgroup
  barriers before the sliding loop remain, including dense cross-row clones
  and the sparse-path publication barrier. The shared source tile is immutable
  after those barriers.
- In the final loop row `r` reads/updates only histogram words
  `[r*256, r*256+248]`, writes/reads terms `[r*4, r*4+3]` and writes its own
  16 output pixels. Other rows use disjoint mutable regions. Shared tile reads
  can overlap across rows because no invocation writes the tile anymore.
- The first final-loop barrier completes that row's histogram reads and term
  writes before updates and lane-zero combination. The second completes that
  row's updates before its next column's reads. Neither orders a cross-row
  dependency. Each becomes `subgroupBarrier()` only on the complete-row path;
  the uniform fallback retains `barrier()` at the same sites.
- The loop bound and final-column break are uniform within each subgroup.
  All per-lane branches reconverge before its barriers; `row_active` is uniform
  within a subgroup. The four FP32 term expressions and lane-zero maximum
  order remain unchanged, and no shuffle or floating-point reduction is added.

`--skip-inactive-rows` is an independent variant built on that row mapping.
After all cross-row publication barriers, an inactive row has sixteen zero
center symbols and consequently sixteen exactly zero contrast results. Its
first 16 lanes publish bounded zero outputs and the entire subgroup returns.
The return is gated by `subgroup_rows && !row_active`; the workgroup fallback
cannot return individual rows while other rows still use workgroup barriers.
The all-zero-tile return from the original shader remains workgroup-uniform.

Both subgroup variants' uint and half shaders compiled for Vulkan 1.1 and
passed SPIR-V validation (four combinations). Capabilities are Shader and
GroupNonUniform, plus Float16 only for the existing half variant. The two
new barriers emit subgroup execution/memory scope with memory semantics 3400,
which includes WorkgroupMemory; original/fallback barriers retain workgroup
scope. Both isolated host versions passed the project warning flags with
`-Werror` under `-fsyntax-only`.

[`subgroup_rows_check.py`](../../../experiments/v1_headroom/subgroup_rows_check.py)
passed CPU coverage/ownership checks for 100 arbitrary physical local-index
permutations, every one of the 256 active-row masks, and radii 0/8/16. It checks
unique cooperative coverage, disjoint row-owned writes and bounds for every
center/entering/leaving tile index. It is an access-domain proof, not execution
of the GLSL or a GPU numerical oracle. Raw compiler evidence and its output are
in `artifacts/reports/vmaf-v1-headroom/subgroup-rows/`.

```sh
python3 bench/experiments/v1_headroom/subgroup_rows.py /tmp/vmaf-row-copy
python3 bench/experiments/v1_headroom/subgroup_rows.py /tmp/vmaf-row-skip-copy --skip-inactive-rows
python3 bench/experiments/v1_headroom/subgroup_rows_check.py
```

The fallback's actual use on a given driver has not been observed here. Root's
GPU checks must distinguish a row-path result from a fallback-only result
before drawing a conclusion about row-local synchronization. Replacing a
workgroup dependency with a subgroup dependency can still lose to branch or
code-generation overhead. This probe has no measured correctness or speed
claim from this review.

## Earlier measurements constrain the remaining search

The review read the retained reports before choosing these arms:

| alternative | existing evidence and implication |
| --- | --- |
| Fuse horizontal/vertical mode filtering | [`vmaf-v1-pipeline-optimization.md`](../vmaf-v1-pipeline-optimization.md): CAMBI 0.669749→0.670341 ms at 1080p and 2.411981→2.429054 ms at 4K. Do not repeat unchanged. |
| Fuse decimation/horizontal mode filtering | [`vmaf-v1-cambi-optimization.md`](../vmaf-v1-cambi-optimization.md): −1.28% at 1080p, +0.16% at 4K, misses the primary gate. Exactness alone did not justify it. |
| Always fuse conversion/derivative | Same report: regresses intermediate sizes. Current 4K-only policy already reflects this. |
| Balance histogram clone dependencies | [`vmaf-speed-investigation.md`](../vmaf-speed-investigation.md): fewer barriers increased row updates and regressed CAMBI 0.8%/1.9%. This differs from deleting identical consecutive command-buffer barriers. |
| Combine six native Y/U/V conversions into two dispatches | Same report: exact outputs, preparation +7.2%/+1.1%. Dispatch count alone did not predict speed. |
| Force staged streaming | Same report: whole-frame time +19–36%; preparation timestamps alone hid upload cost. Keep default memory policy. |
| Compress contrast shared symbols | [`vmaf-fp16.md`](../vmaf-fp16.md): exact 0..497 source tile is already retained; do not confuse it with global-bank compression. |

The current V1 pipeline records ADM, SpEED, CAMBI and motion into one command
buffer and performs one final readback/submission. CAMBI introduces no separate
host wait per scale. Batching submissions or exposing multiple frames in flight
would be a wider scheduling/API investigation, especially with private temporal
history; it is not a local fix for these redundant barriers. Streaming and
resident paths must be measured separately because upload can dominate work
outside the CAMBI timestamp.

## Checks and reproduction

Compiler checks used the local GCC 16 compile-database arguments with C++20,
all maintained warnings and `-Werror`; both isolated modified host files passed
`-fsyntax-only`. Generated GLSL was compiled with `glslc -O
--target-env=vulkan1.1` and passed `spirv-val --target-env vulkan1.1` for all
three variants. SPIR-V capabilities were:

| shader | capabilities |
| --- | --- |
| packed contrast, uint shared tile | Shader, StorageBuffer16BitAccess |
| packed contrast, existing half shared tile | Shader, Float16, StorageBuffer16BitAccess |
| packed-source decimator | Shader, StorageBuffer16BitAccess |

The storage change therefore does not newly require 16-bit arithmetic. Both
installers are import-safe, reject the maintained checkout and require fresh
copies; they validate every expected source anchor before writing. A probe
installed on an older source version must fail rather than silently replacing
the wrong expression. Keep arms independent when initially measuring them.

```sh
python3 bench/experiments/v1_headroom/storage_mode_banks.py /tmp/vmaf-packed-copy
python3 bench/experiments/v1_headroom/storage_component_barriers.py /tmp/vmaf-barrier-copy
python3 bench/experiments/v1_headroom/storage_mode_banks_check.py
```

The CPU checker passed all 1,025 reachable code round-trips and 48 deterministic
mode/edge/address/decimation cases, including one-row/column images, odd sizes,
random codes, ties, extremes and untouched guard/tail bytes. It also reproduces
the analytical size table. It is an independent CPU storage-contract check,
not a GPU oracle or a test of GLSL execution. Raw compiler commands, SPIR-V and
CPU output were written under
`artifacts/reports/vmaf-v1-headroom/storage/`; the note and reusable probes are
the versioned reproduction material.

No GPU correctness, synchronization validation, corpus, sanitizer, correctness
gate or speed gate was run by this review. The parent must freshly configure
and build each complete source copy, check exact score/checksum agreement on
native depths, converted SDR and odd sizes, retain the independent CAMBI and
sequence oracles, run synchronization validation, and compare resident/upload
arms under the standard protocol. A passing compile or reduced byte count is
insufficient to retain either candidate as a performance improvement.

## Parent screening and capability-selected follow-up

The parent subsequently measured these isolated arms on 2026-11-05, using five
interleaved A/B resident rounds, 20 iterations, three warm-ups and two resident
inputs. The selected baseline is the separately built working-tree snapshot
at `/tmp/vqm-v1-headroom/baseline`; source hashes are recorded in
`artifacts/reports/vmaf-v1-headroom/baseline-source-sha256.json`. Per-arm
`resident/results.json` files retain binary/input hashes, all commands, scores,
checksums, timing rounds and spreads. Those are parent measurements, not GPU
work performed by this review.

| research arm | 10-bit 1080p CAMBI change | 10-bit 4K CAMBI change | ducks 1080p CAMBI change | decision |
| --- | ---: | ---: | ---: | --- |
| packed two mode banks | −0.29% | −0.20% | −0.06% | Misses 2% gate; exclude. |
| remove duplicate component-entry barriers | +0.02% | +0.05% | +0.05% | Misses gate; exclude. |
| subgroup row barriers | −5.06% | −3.91% | −2.24% | Qualify a portable capability-selected implementation. |
| subgroup row barriers plus inactive-row return | −2.05% | −1.28% | −1.26% | Slower than the simpler subgroup arm; exclude extra return. |

Scores/checksums matched the baseline for all listed content arms. The
subgroup arm's CAMBI median/range in milliseconds was 0.281707
(0.281251–0.282309) → 0.267461 (0.267264–0.267573) at 1080p; 1.208530
(1.208194–1.208771) → 1.161315 (1.161181–1.162142) at 4K; and 0.189138
(0.188333–0.189366) → 0.184896 (0.184693–0.185019) for ducks. These results
justify a shipping-form experiment, not automatic adoption of a research-only
device guard.

[`subgroup_pipeline.py`](../../../experiments/v1_headroom/subgroup_pipeline.py)
installs that follow-up in a fresh source copy. The existing portable uint/half
blob names remain unchanged. Two additional modules define
`VQM_CAMBI_SUBGROUP_ROWS`; only those modules declare the subgroup extension,
logical mapping and row barriers. Both portable SPIR-V binaries recompiled
byte-for-byte identically to the current baseline and have no GroupNonUniform
capability. All four modules passed SPIR-V validation. The same source routine
and the subgroup-shape fallback described above remain shared across modules.

The internal constructor gains `allow_subgroups = true` after the existing
`allow_half` argument. It selects subgroup modules only when physical subgroup
properties report size 32, COMPUTE support and BASIC operations; unsupported
devices select the original portable modules without an error. This adds no
enabled-device feature or installed API promise. Half selection still depends
on the logical device's separately enabled Float16 feature, including borrowed
contexts. `UsesSubgroupPipeline` exposes the selected module for tests.

The follow-up adds a pure eligibility table for unsupported sizes/stages/
operations, plus exact bit comparisons of portable/subgroup × uint/half and
conservative borrowed-device modules on all eight edge/dense/masked fixtures.
Existing direct-window/libvmaf-oracle tests use the automatic default. Both
the host implementation and expanded test file passed project-flags syntax
checks, and the pure eligibility table was compiled and executed successfully
without any Vulkan call. Evidence lives in the ignored
`artifacts/reports/vmaf-v1-headroom/subgroup-pipeline/` directory. The parent
must rerun GPU correctness and the performance protocol against this final
macro-gated module selection before adoption; the earlier research timings do
not establish its final code-generation performance.

Conclusion: the two storage/dispatch candidates did not pay for their added
complexity. Row-local synchronization provides measured headroom in the
research arm and merits final portable qualification. Existing evidence does
not support another unchanged mode fusion or a blanket FP16 pipeline, and it
does not establish a hard VMAF v1 speed limit.
