# VMAF v1 CAMBI contrast headroom

Source and CPU/compiler investigation on 2026-11-05. These are controlled
experiments, not production changes or GPU performance claims. The maintained
working tree was already modified; source SHA-256 fingerprints and compact
checks are preserved in [v1-contrast-headroom-checks.json](v1-contrast-headroom-checks.json).
GPU timing, oracle and corpus results must be reported by the measuring arm.

The current [contrast shader](../../../../src/metrics/vmaf/cambi/shaders/vmaf_cambi_contrast.comp)
uses one pipeline for two 16×16 mode-filter passes and a 16×8 contrast tile.
Contrast source symbols are exact integers 0–497 stored in FP16 when supported;
packed histogram counts stay integer and four contrast terms stay FP32. The
mode-filter passes never access the contrast workgroup arrays, but those arrays
remain statically present in the shared pipeline's SPIR-V.

## Independent experiments

The [isolated installer](../../../experiments/v1_headroom/contrast_variants.py)
provides four independent arms, plus a lower-priority four-row control. Each
requires a fresh source copy outside the maintained checkout. It refuses the
maintained checkout, its ancestors and descendants, and validates all source
anchors before writing. Existing shader IDs and public headers remain stable;
experimental pipeline IDs are additional internal IDs only.

```sh
python3 bench/experiments/v1_headroom/contrast_variants.py /tmp/v1-mode --mode-only
python3 bench/experiments/v1_headroom/contrast_variants.py /tmp/v1-radius --radius8
python3 bench/experiments/v1_headroom/contrast_variants.py /tmp/v1-wide --tile-width 32
python3 bench/experiments/v1_headroom/contrast_variants.py /tmp/v1-tail --reuse-tail
```

Configure and build each isolated copy normally, then apply the interleaved
protocol in [AGENTS.md](../../../../AGENTS.md). Do not install these commands
successively into the same copy. Flags can be combined deliberately after
independent measurements; standalone results identify each contribution.

1. **Mode-only pipeline:** compile the existing source with the phase-two call
   unreachable, allowing `glslc -O` to remove all contrast workgroup arrays.
   Route only phases zero and one through that new pipeline. Source mode ties,
   boundary policies, arithmetic, output banks, push layout, synchronization,
   workgroup shape and dispatch count remain unchanged. Compiler inspection
   confirms zero Workgroup variables and no Float16 capability in the new mode
   module. Separate compiled code may improve occupancy or reduce resources;
   SPIR-V alone does not establish actual driver allocation or a speed gain.
2. **Radius-eight capacity:** compile additional uint and half variants with
   a maximum source capacity for radius eight. `Record` selects this compact
   pipeline only for windows ≤17, including the standard 1080p CAMBI plan's
   17-pixel window. The original capacity remains available for windows 19–33,
   including independent direct tests at window 33 and `allow_half=false`.
   Runtime indexing, histogram operations, contrast expressions, workgroup
   geometry and dispatch count are identical. This reduces unused shared
   capacity; it does not remove halo loads or floating-point calculations.
3. **Fully buffered 32×8 outputs:** double horizontal outputs and load a full
   64×40 half source tile for maximum radius 16. This halves horizontal group
   count and reduces duplicated halo/setup, while doubling the sequential
   horizontal loop and increasing shared capacity. The portable uint fallback
   keeps its original 16×8 geometry. Full buffering avoids rolling-ring address
   arithmetic and mid-loop refills, making this distinct from the rejected
   historical 32-column stripe below. More serial work per group can still
   erase any savings; adoption requires measured evidence.

4. **Reuse dead histogram tail:** bins 0–496 address only packed words 0–248
   of each 256-word histogram row. Words 249–252 instead carry the four FP32
   contrast terms through `floatBitsToUint`/`uintBitsToFloat`; no arithmetic is
   changed. The final row's word 255 also holds the active-row flags during
   probing, removing the separate flag word. Each invocation caches this flag
   after the existing probe barrier. Initial clearing deliberately skips that
   single flag word: clearing it immediately would race other invocations'
   cached reads. The existing halo barrier finishes all cached reads before
   dense 256-word clones can overwrite that dead flag slot. Later decisions
   use cached masks. Sparse rows keep that unused word until the next probe
   resets it. Term publication occurs only after clone construction; subsequent
   updates/readers never address the tail. This independent arm changes only
   the shader, preserving the host, CMake and pipeline ownership byte for byte.
   Combine `--radius8 --reuse-tail` to test the compact half tile at exactly
   9,728 shared bytes. Allocation granularity might make 132 bytes matter;
   actual occupancy and speed remain measurement questions.

`--tile-rows 4` tests a 16×4 tile; combine it with `--tile-width 32` for 32×4.
These controls keep physical workgroups at 16×16 with 256 invocations; the
separate 128-invocation option below changes that shape. Four-row contrast
teams use 64 lanes per row, and the contrast dispatch/planner cover the larger
Y group count. The installer retains mode passes' 16×16 grid and validates both
mode and contrast X group counts, even when the contrast grid is narrower.
All offered half variants fit below the existing 16,004-byte portable shared
capacity; no new device-limit requirement is introduced.

| Compiled contrast geometry/capacity | uint shared bytes | half shared bytes |
| --- | ---: | ---: |
| Original 16×8, maximum radius 16 | 16,004 | 12,164 |
| Compact 16×8, maximum radius 8 | 11,396 | 9,860 |
| Buffered 32×8, maximum radius 16 | original 16×8 fallback | 13,444 |
| 16×4, maximum radius 16 | original 16×8 fallback | 7,620 |
| 32×4, maximum radius 16 | original 16×8 fallback | 8,772 |
| Buffered 32×8, maximum radius 8 | compact 16×8 fallback | 10,628 |
| Original 16×8, reused tail | 15,872 | 12,032 |
| Compact 16×8, radius 8, reused tail | 11,264 | 9,728 |
| Mode-only entry point | 0 | no half capability needed |

These are sums of statically declared SPIR-V Workgroup variable sizes, validated
from the compiled modules. They are not hardware occupancy measurements. The
mode-only source preserves the entire active mode body byte for byte; this was
checked automatically before compilation. Creation of multiple pipelines is
transactional through `VulkanRuntime::CreatePipelines`; failure cleans up prior
handles, and the contrast owner destroys all created handles together.

## Correctness contracts and CPU/compiler evidence

Run the reproducible checks without a GPU:

```sh
python3 bench/experiments/v1_headroom/contrast_contracts.py
```

The tool requires Python 3, a C++20 compiler, `glslc`, `spirv-val` and
`spirv-dis`. This run used glslc 2026.4, target Vulkan 1.1, optimized SPIR-V,
and the current source fingerprints in the compact evidence. All twelve installed
host combinations passed C++ syntax checking. All 40 resulting shader modules
compiled and passed `spirv-val --target-env vulkan1.1`; extracted shared sizes
matched the table. Full generated modules and checks live in ignored
`artifacts/reports/vmaf-v1-headroom/contrast-contracts/`.

The CPU simulation used deterministic seed `0xCA8B1`, odd 35×37 and 65×41
images, radii 0/8/16 and dense, sparse, empty and second-half-only eligible
centers. It covered four offered output geometries, including the 128-invocation
versions of both four-row widths. Its independent reference
counts every nonzero packed symbol in each clipped window, then compares the
sliding implementation's exact integer histogram at every eligible center.
Results: 144 cases, 107,526 exact center histograms, 5,228 dense-row clones,
179,547 horizontal slides, 16,752 cooperative coverage loops, and 285,120 output
coordinates owned exactly once.
It checks center/halo address bounds, partial tiles, inactive rows and groups,
leaving-sample presence, and the ≤1,089 maximum per-bin count. The latter keeps
both halves of each packed 16-bit histogram word independent of carries or
borrows. No subgroup-size assumption is introduced.

The tail-lifetime proof additionally explores 800 randomized interleavings of
256 invocations' cached flag reads and histogram clears, with four/eight rows
and empty, sparse, alternating and dense masks. It verifies 19,200 FP32 bit
roundtrips, disjoint bin/term/flag slots, preserved histogram words, and full
clone overwrite only after the flag lifetime ends. These are CPU scheduling
models; actual shader memory behavior still requires GPU oracle/readback gates.

This simulation validates indexing and integer transactions; it does not model
GPU atomic/barrier behavior or certify FP32 device results. The arithmetic
expressions and their local order are preserved in source. Required GPU gates
remain the independent CAMBI mode/contrast oracle tests, window 33 and portable
uint tests, temporal/full VMAF checks, full `vqm_test --require-gpu`, CTest,
quality/lint checks for production adoption, and relevant pinned-reference
comparisons. Score/checksum agreement must accompany each interleaved timing
arm. Both resident and streaming modes and 1080p/4K inputs are needed before
making a wider performance claim.

## Historical limits and rejected approaches

The fully buffered geometry must not be described as the first 32-column trial.
The earlier rolling stripe in ignored
`artifacts/reports/vmaf-more/stripe-hist.patch` retained a 48-wide source ring,
refilled retired columns and wrapped shared indices while producing 32 outputs.
Its 1080p CAMBI time regressed 0.318698→0.357678 ms (+12.2%); ducks regressed
0.188766→0.224478 ms (+18.9%), with matching scores/checksums. That experiment
predated the current half tile. The new arm removes its ring/refill work, but
the older failure is evidence against assuming width alone supplies a gain.

The [VMAF speed investigation](../vmaf-speed-investigation.md) rejected balanced
histogram clones: six barriers instead of fourteen required twelve row updates
instead of seven and regressed CAMBI 0.8%/1.9%. The serial clone chain is retained
by all these arms. A later two-seed clone trial also missed the gate; repeating
those clone algorithms is lower priority than changing unused resource capacity.

The [CAMBI optimization report](../vmaf-v1-cambi-optimization.md) rejected
combining decimation with horizontal mode filtering: −1.28% CAMBI at 1080p and
+0.16% at 4K. The mode-only arm here keeps the existing dispatch count and
focuses on static shared resource separation; it does not repeat that fusion.
Thresholds, mask policy, tie handling, histograms, FP32 contrast and top-k
pooling are unchanged by the current contrast proposals.

The initial priority was mode-only and compact radius capacity first, then
tail reuse and fully buffered geometry. The follow-up screens below found
those capacity-only controls neutral. These remain experiments until the measuring arm
records the target-stage ≥2% median gain under the repository protocol and
passes the correctness gates. A smaller shader allocation by itself does not
establish a performance improvement.


## Follow-up: 128-invocation four-row tiles and geometry fallback

The parent GPU investigation's initial screens found mode-only, radius-eight,
tail-only and radius-plus-tail controls neutral within about 0.5% CAMBI time.
Those are initial screen observations, not independent timing performed by this
CPU/compiler review. Lowering the invocation count changes the experiment more
substantially: four rows retain 32 lanes each, rather than the 64 lanes per row
used by the original four-row/256-invocation control.

```sh
python3 bench/experiments/v1_headroom/contrast_variants.py /tmp/v1-128 \
  --mode-only --tile-rows 4 --tile-invocations 128
```

The half contrast blob uses 16×8 physical invocations, covers 16×4 output
pixels, and strides every cooperative load/clear/seed loop by 128. Each
invocation copies two of a row's 256 histogram words before the existing clone
barrier. Mode filtering remains a separate 16×16/256-invocation blob, and the
uint fallback retains 256 invocations and eight output rows. Shared capacity
is unchanged from the four-row table: 7,620 bytes, or 5,376 with radius-eight and
tail reuse. All exact histogram and cooperative-coverage CPU checks cover this
scheduling change.

Initial parent measurements of the unconditional 128/four-row arm reported
CAMBI −2.47% at 1080p, +4.12% at 4K, and −1.85% on ducks. Adding radius-eight
and tail reuse gave −2.61%, +3.24% and −1.91%. Neither broad geometry arm is an
unconditional improvement. The original four-row planner also failed
`TestVmafV1DispatchLimitBoundaries`: its conservative four-row Y-count check
rejected a synthetic device with disabled Float16 even though that device uses
the unchanged eight-row uint fallback. This failed gate is retained as a
reason to reject raising the baseline dispatch requirement.

The conditional experimental installer addresses both issues:

```sh
python3 bench/experiments/v1_headroom/contrast_variants.py /tmp/v1-small \
  --mode-only --tile-rows 4 --tile-invocations 128 --small-window-only
```

It leaves the existing `vmaf_cambi_contrast_half` at its original 256-invocation,
eight-row geometry and adds `vmaf_cambi_contrast_small_half`. The new blob is
selected only when Float16 is enabled for the owner, the requested window is
≤17, and the actual four-row Y dispatch fits the device limit. Window 33,
disabled half support and devices whose Y limit fits only eight-row geometry
use the original pipeline. Width/Y limit validation follows the selected
dispatch. The v1 planner remains byte-identical to the baseline; capability
selection cannot raise its API minimum. If half is disabled, the unused small
pipeline slot transactionally owns another uint pipeline rather than attempting
to create a Float16 pipeline.

A 72-case CPU selector check covers disabled/enabled half, windows 1/17/19/33,
odd/1080p heights, and Y limits that allow only the baseline or both geometries.
It proves every selected grid fits a baseline-compatible device and all
fallback cases keep eight rows. Compiler checks explicitly verify SPIR-V local
sizes: 16×8 for the small blob and 16×16 for the original/uint/mode blobs.
`--radius8` specializes only the additional small blob in this conditional arm;
`--reuse-tail` remains optional. This conditional arm still needs its own GPU
correctness and content/size performance gates before production adoption.


## Observe the subgroup-row guard directly

The research-only [subgroup probe host](../../../experiments/v1_headroom/subgroup_rows_probe.cc)
and [shader](../../../experiments/v1_headroom/subgroup_rows_probe.comp) observe an
ordinary 16×16 compute pipeline using the same default runtime context and
pipeline creation flags as the library. They request no particular subgroup
size and do not set require-full-subgroup or varying-size flags. The shader
records a four-word header from local invocation zero: `gl_SubgroupSize`,
`gl_NumSubgroups`, the 256-invocation group size, and the exact size-32/eight-group
condition used by the research row path. Another 256 words publish each local
invocation's `subgroupID*size+subgroupInvocationID`; the host checks complete,
unique coverage of logical slots 0–255. There are no reductions or timestamps.

The host prints only the selected device, its physical default subgroup size
and compute BASIC support, then actual shader values and coverage. An
unsupported BASIC path is reported as an error before dispatch. This probe
observes its own pipeline with the matching workgroup/creation policy; CAMBI's
actual arithmetic and row barriers still need their own GPU correctness gates.
The compiler-only review passed C++20 linking against `build/libvqm_core.a`,
`spirv-val --target-env vulkan1.1` and clang-format 23. Compiled SPIR-V declares
Shader/GroupNonUniform capabilities and local size16×16×1. No GPU execution was
performed by this review; subsequent observations belong in the measured
headroom report.

```sh
glslc -O --target-env=vulkan1.1 \
  bench/experiments/v1_headroom/subgroup_rows_probe.comp \
  -o /tmp/v1-subgroup-rows-probe.spv
spirv-val --target-env vulkan1.1 /tmp/v1-subgroup-rows-probe.spv
c++ -std=c++20 -O2 -I include -I src \
  bench/experiments/v1_headroom/subgroup_rows_probe.cc \
  build/libvqm_core.a -lvulkan -pthread -o /tmp/v1-subgroup-rows-probe
/tmp/v1-subgroup-rows-probe /tmp/v1-subgroup-rows-probe.spv
```


## Independent review of the shipping subgroup candidate

Reviewed the five-file candidate under `/tmp/vqm-v1-headroom/candidate` against
its frozen baseline after the parent GPU checks. This review made no production
edits or GPU calls and found no blocking defect. Candidate file/module hashes
are in the compact checks JSON. Its changes are the optional subgroup module,
physical-capability selection, logical row mapping, two row-private barriers,
and independent portable/subgroup equivalence tests; none of the unsuccessful
geometry/capacity arms is part of that candidate.

The documented borrowed-instance contract already requires Vulkan 1.1 or later.
The physical compute/BASIC gate matches the
[Khronos GroupNonUniform capability contract](https://docs.vulkan.org/refpages/latest/refpages/source/VkSubgroupFeatureFlagBits.html).
The new subgroup operations require no newly enabled optional logical-device
feature. Half-source selection remains separately gated by the owner's promised
enabled `shaderFloat16` feature. The runtime does not request varying subgroup
sizes or a required size; these are distinct
[pipeline-stage flags](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineShaderStageCreateFlagBits.html).

The size32/eight-subgroup condition is evaluated before any pixel-bound return.
Every cooperative clear/load/clone uses the same bijective subgroup logical
index. Cross-row source/histogram construction retains workgroup barriers.
After construction, a row's histogram, four FP32 terms and horizontal updates
are accessed by one complete subgroup, so the subgroup barrier orders every
producer/consumer of those row-private locations. The last column exits without
later writes to its terms. Integer atomic counts and FP32 contrast expressions
are unchanged. Other actual packing uses the original quotient/remainder local
mapping and full workgroup barriers; the decision is uniform for the ordinary
pipeline policy. The host also rejects the optimized module on non32 physical
defaults, retaining the portable module.

All six compared modules (baseline uint/half and candidate uint/half/subgroup/
subgroup-half) independently compiled and passed Vulkan 1.1 SPIR-V validation.
The candidate's original uint and half modules are byte-identical to baseline.
Subgroup modules add only GroupNonUniform, plus the independently gated Float16
capability in the half module. Their subgroup control barriers have subgroup
execution/memory scope and acquire/release shared-memory semantics. This review
did not GPU-execute the non32 actual-packing branch; synthetic capability checks,
portable byte identity, source reasoning and the parent's device-specific tests
cover different parts of that portability argument.
