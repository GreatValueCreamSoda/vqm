# CVVDP integration architecture review

This review records the design as of 2026-11-05. See the [CVVDP guide](../cvvdp.md)
for the implemented API and [Contributing](../../CONTRIBUTING.md) for current
local checks. The review used source inspection without building or running
either implementation.

## Question and conclusion

What must change to make CVVDP a supported metric while retaining the current
Vulkan input, allocation, synchronization, testing and packaging contracts?

The metric can reuse the Vulkan runtime and image ownership machinery. It
cannot safely inherit either the existing fixed display-referred color decode
contract or VMAF's one-frame sequence policy. Physical display interpretation,
spatial/angular geometry, frame timing, temporal boundary handling and final
pooling are metric inputs, not incidental CLI options.

Reference review confirms that the default video path is causal, retains
three DKL components per input and derives a fourth transient output channel;
the final clip JOD is produced after clip-wide pooling. A true one-frame image
uses a different three-channel path and an image integration correction. The
architecture must expose these differences. This review only checks the
integration consequences of those upstream contracts; the full equations,
parameter assets, tolerances and licensing remain owned by the companion
[reference review](cvvdp-matlab-review.md) and implementation plan. Names below are
proposed repository names, not finalized public interfaces.

## Provenance and method

- HEAD at inspection: `a949b6177efb88890dd44860cc30dff45486a25d`.
- The working tree already contained extensive staged/unstaged/untracked
  reorganization and implementation work. The review uses the actual working
  files, including `include/ssimulacra2.h` and metric-local source trees; HEAD
  alone does not reproduce this state. Existing work was not changed.
- Read `AGENTS.md`, the public image/metric/memory/interop APIs, shared Vulkan
  runtime, source descriptors and color GLSL, VMAF sequence/pipeline structure,
  top-level CMake, CLI/benchmark drivers, reference tooling and CI.
- Commands were read-only `git status`, `git diff`, `git rev-parse`, `rg`,
  `cat` and `sed`. No build or hardware was used, so there are no measured
  results, speed claims or GPU-correctness claims in this note.
- After the initial repository pass, inspected the pinned original
  `gfxdisp/ColorVideoVDP` v0.5.7, commit
  `2a268bce8d56e2f3abde46df3927d8a633707a24`, through the read-only local
  checkout `/tmp/cvvdp-original-review-20261105`. In
  `pycvvdp/cvvdp_metric.py`, lines 334–344 distinguish one-frame images,
  456–568 implement the input history/filtering, 615–649 pool the clip and
  apply image integration, and 1030–1070 generate the temporal filters.
  `pycvvdp/vvdp_data/cvvdp_parameters.json:10` sets the default time/channel/
  band exponents to 2/4/4, respectively.
  Inspecting source is not an executed numerical parity result.

Source anchors below refer to this working-tree inspection and may move as
the ongoing reorganization settles.

## Exact integration touchpoints

| Area | Current owner / anchor | CVVDP work |
| --- | --- | --- |
| Public API | `include/ssimulacra2.h`, `include/vmaf.h`, `include/vmaf_v1.h` | Add `include/cvvdp.h` with complete metric configuration, result semantics and sequence contracts. Follow PImpl/move-only ownership and decoded-input APIs. |
| Umbrella/package | `include/vqm.h`; `CMakeLists.txt:299` | Include/install the new header and add direct-header and installed-consumer coverage. Include any required upstream license/asset notices after reference review. |
| Unified selection | `include/vqm_metrics.h:32`; `src/metrics/metrics.cc:26` | Add optional CVVDP configuration only after defining how video results and still-image results map to this façade. Extend selection validation and override only shared resource fields. |
| Unified results | `include/vqm_metrics.h:46`; `src/metrics/metrics.cc:246` | Add CVVDP to still-image results, but expose the video result through a distinct final sequence-summary type. Do not put a fabricated frame JOD in `MetricsResult` or imply clip JOD is a mean of frame JODs. |
| Unified alignment/drain | `src/metrics/metrics.cc:254`, `:266`, `:427` | Preserve VMAF frame alignment and failure/reset handling. Current `Flush` emits at most one result and rejects leftover frames; default causal CVVDP does not require a filter tail, but final clip pooling requires a sequence-summary operation separate from this frame-only contract. |
| Unified semaphore transaction | `src/metrics/metrics.cc` `Run` / `next_sync` | Count CVVDP submissions in wait-on-first/signal-on-last ordering. If a push can defer all GPU work, input-consumption signals still need a precise meaning. Preserve the existing partial-failure contract. |
| Memory query API | `include/vqm_memory.h`; `src/vulkan/vulkan_memory_query.cc` | Add default-device/borrowed-device overloads using a metric-local planner shared with construction. Reuse `QueryMetric`, direct/staged scenarios, heap budgets and worker arithmetic. |
| Unified memory | `include/vqm_metrics.h:57`, `:66`; `src/metrics/metrics.cc:119`, budget helper and `GetMemoryUsage` | Add CVVDP as an independent component, charged once per worker. Temporal history is private metric state. |
| Shared spatial arena | `include/vqm_shared.h`; `src/metrics/shared_workspace.cc` | No initial CVVDP membership. Current workspace deliberately serves serial SSIMULACRA 2/Butteraugli scratch, without temporal history or resident handles. |
| Source storage / validation | `include/vqm_image.h`; `src/input.{h,cc}`; `src/image_view.cc` | Reuse planar storage, exact byte sizing, format validation, padded host views and enum membership. Add only format restrictions required by the selected CVVDP path. |
| Source descriptor / decoding | `src/vulkan/vulkan_source.cc:77`; `src/vulkan/vulkan_support.h:101`; `src/vulkan/shaders/color.glsl` | Reuse sample unpacking, depth/range, matrix and chroma reconstruction only where reference contracts match. The current high-level decoder ends in linear Rec.709 with a fixed luminance convention. |
| Device/runtime | `src/vulkan/vulkan_runtime.{h,cc}`, `vulkan_runtime_inputs.cc`, `vulkan_support.{h,cc}` | Reuse context borrowing, allocation paths, descriptor sets, sources, barriers, submit/fence lifetime, stage queries, transactional pipeline creation and readback. |
| Shader build/embedding | `CMakeLists.txt:63`, `:89`, `:207`; `cmake/embed_spirv.cmake` | Register metric-local GLSL and stable `cvvdp_*` blob names. Reuse `vqm_add_source_shaders` only if its 12 source classes and shader contract remain appropriate. |
| Core/timed build | `CMakeLists.txt:233`, `vqm_configure_core`, benchmark target sources | Add host sources to the common list so core and timed builds use the same arithmetic. Keep reference/Python dependencies outside the installed core. |
| Timing | `src/timing.h:18`; `src/timing.cc:13` | Add stages based on actual CVVDP responsibilities, including host pooling when applicable. Keep the timer-free core and GPU timestamp/host-clock distinction. |
| CLI | `cli/main.cc` | Add `--metric cvvdp`, validated metric options and score presentation. Current CLI supports only SSIMULACRA 2 and Butteraugli; VMAF's benchmark support is not CLI sequence support. |
| File decoding | `tools/image_io/io.{h,cc}` | Current `LoadImage` reads only the first Y4M frame and ignores PNG ICC/gAMA/cHRM/sBIT metadata. Real video needs an ordered reader or explicit frame-list workflow plus timing metadata. |
| Benchmark dispatch | `bench/driver/bench.h`, `options.cc`, `main.cc` | Add enum, parser/validation and a dedicated driver. Do not let generic spatial defaults silently choose CVVDP modes. |
| Benchmark data/metadata | `bench/driver/common.cc`, `bench/references.json`, `bench/benchmark.py`, `bench/run_inmemory.py` | Current CSV records pair paths, geometry, decode/core time and scalar score. Add enough resolved configuration and temporal provenance to compare like-for-like runs. Existing comparison defaults select only SSIMULACRA 2/Butteraugli. |
| References | `tools/references.lock`, `fetch_references.py`, `build_references.py` | Pin upstream revision and assets/dependencies; keep `.reference/` read-only and stage runnable reference copies/environments under the selected build tree. Extend explicit step/implementation maps. |
| Validation | `validation/references.json`, `run_corpus.py`, `validation/vmaf_cid22.py` | Add still-image correlation only after matching reference display/input assumptions. Add independent temporal sequences; CID22 alone cannot validate time processing. |
| Local tests | `tests/support.h`, `tests/metrics/vmaf/`, `tests/core/metrics_test.cc`, `CMakeLists.txt` | Add metric-local independent oracles and explicit test sources. Keep clean core builds offline and retain synchronization, sanitizer and package-consumer checks. |
| Docs/quality/profiling | `docs/README.md`, metric docs, `README.md`, `bench/README.md`, `tools/check_quality.py`, `bench/profile*.sh` | Document supported scope and reference provenance, register new profile choices, and retain recursive C++/GLSL quality checks. Do not attach old metrics' score or speed gates to CVVDP. |

## Public API and sequence decisions

Use one authoritative `CvvdpConfig` and resolved configuration representation
for public construction, CLI setup, memory planning and reference manifests.
Avoid duplicating preset resolution or physical-unit conversion in tools.
Likely categories, subject to upstream confirmation, are source geometry,
display/photometry, viewing geometry or angular sampling, frame rate/timebase,
metric/model version, resource policy and optional diagnostics. Validate finite
values and enums before allocation or state mutation, with error messages that
name the metric and offending field.

Keep display configuration distinct from source `ImageFormat`:

- `ImageFormat` describes stored samples, transfer, primaries, range, chroma
  and alpha. It has no display peak/black level, ambient/reflection,
  screen geometry, viewing distance, angular sampling or frame time.
- Metric configuration describes the viewing/display model and time semantics.
  Named presets, if supported, must resolve to recorded values. An upstream
  preset name alone is insufficient provenance if upstream can change it.
- Fix display/time/model configuration for a sequence, or explicitly reset
  every affected history/cache when it changes. A shared image handle stores
  raw pixels; derived display-conditioned planes have additional cache keys.

Proposed names are `CvvdpPipeline`, `CvvdpConfig` and `CvvdpScore`. Recommended
operation contracts, with exact signatures left to API review:

1. `ComputeImage` evaluates the true image path: three channels with no video
   FIR, then the reference's image integration factor at pooled Q. It must not
   silently evaluate a one-frame causal video at a default fps.
2. `BeginSequence` / ordered `Push` / `Finalize` evaluate a clip. Default
   replicate padding is causal: N-1 historical samples plus the current frame,
   with no future-frame lookahead. Seed the first frame by replicating its
   three DKL components into history and actually apply the FIR; do not bypass
   the first video frame. `Finalize` maps the accumulated whole-clip quantity
   to one JOD. A push may expose typed feature diagnostics but must not promise
   an independently valid final frame JOD.
3. Upstream decides `is_image` from total frame count, including in pooling.
   A streaming API cannot infer that at its first push. Require a declared
   expected frame count in `BeginSequence`, or defer processing the first
   frame until a second push or finalization distinguishes the two paths.
   Validate declared length. Keep any explicitly different one-frame-video
   API semantics named and documented instead of silently claiming parity.
4. Decide fixed-rate versus timestamped input from reference research. If only
   fixed-rate is supported initially, reject missing/zero/nonfinite fps and
   state that variable-rate streams require an external resampling policy.
5. Document empty/single/short sequence behavior, temporal padding at both
   ends, flush idempotence, push-after-end, reset-after-failure, length overflow
   and whether temporal data must be retained through the whole clip.
6. Preserve host memory lifetime through the upload-consuming call and Vulkan
   buffer lifetime through the GPU input-consumption boundary. Copy retained
   temporal history into owned storage before returning to a caller allowed to
   overwrite its buffer. Do not retain borrowed pointers in a temporal ring.

The current unified façade has useful ordered frame storage but a hard limit:
`Flush()` calls both VMAF flushes once, takes one ready result, marks the
sequence finished and rejects any leftover pending frames. Repeated calls
return empty. `ComputeImage()` assumes at most one result from `Run` plus
`Flush`. Recommended façade addition: a typed sequence summary containing
`optional<CvvdpScore>`, returned by an explicit finalization operation. Keep
video CVVDP absent from the frame-completeness predicate; frame-aligned
diagnostics, if enabled, receive a different type/field name. The finalization
API must also deliver the existing pending VMAF frame, either explicitly in
its return type or through the established flush step. Update façade tests to
retain existing behavior. Deferring unified video CVVDP until that small API
is designed is preferable to pretending a clip result is a frame result.

Upstream `do_pooling_and_jods` applies weighted L4 pooling across bands, L4
across channels, then a normalized L2 across time for the default model; only
then does it convert to JOD. For images the integration factor is applied to
pooled Q. Do not derive clip output by averaging per-frame JODs. A score-only
streaming implementation can accumulate the required temporal power sum and
count rather than retaining every frame's full features; reproduce the exact
upstream norm/`safe_pow` arithmetic and reduction ordering within the agreed
tolerances.

Initially leave `Ssimulacra2Pool`/`ButteraugliPool` and
`SharedMetricWorkspace` unchanged. Their scheduling assumes independent pairs.
Workers for a temporal metric own independent sequences and histories;
successive frames of one sequence cannot be round-robined among them.

## Color, HDR and display boundaries

Current public source decoding is explicitly display-referred with zimg-style
defaults and peak luminance fixed to 100 cd/m² (`vqm_image.h`). In
`src/input.cc:664`, `SelectTransfer` supplies PQ scale `10000/100` and HLG
scale `1000/100`. `BuildSourceDesc` creates the source-primaries-to-Rec.709
matrix. `block_linear_rgb` in `color.glsl:447` combines raw decode, chroma
upsampling, matrix conversion, alpha, transfer decode and the Rec.709 matrix.
It is not a neutral raw RGB reader or configurable physical display model.

Consequently, PQ/HLG enum support alone does not establish CVVDP HDR support.
Verify reference requirements for absolute luminance, display EOTF/OOTF,
black/peak/ambient behavior, gamut conversion, negative/out-of-gamut values,
clamping, alpha and chromatic adaptation. Record which side each operation is
applied to and its units. These details may change values before any CVVDP
feature arithmetic runs.

Preferred extension sequence:

1. Begin with an explicit, oracle-backed source/display contract and a
   metric-local display transform. If an initial supported path requires
   opaque inputs or preconverted linear values, validate and document it.
2. Reuse current source unpacking/range/chroma helpers when they match that
   contract exactly. If the fused `block_linear_rgb` boundary is unsuitable,
   extract one real shared decode phase with regression tests preserving
   existing metrics' arithmetic and source bindings. Do not copy all of
   `color.glsl` into CVVDP or globally change existing decode semantics.
3. Keep CVVDP-specific opponent channels, display response, adaptation and
   perceptual arithmetic under `src/metrics/cvvdp/`. Promote only a proven
   second use of an identical contract into metric-independent code.

VMAF's preprocess converters are not substitutes: its SDR validation rejects
PQ/HLG and its features intentionally read coded luma or quantized limited-range
BT.709 values. Its motion timeline, resizer and pooling also encode VMAF
policies. Similar shader operations do not prove interchangeable boundaries,
edge extension or coefficients.

`MetricsConfig::background` currently forces one encoded RGB scalar onto
selected metrics. Do not automatically overwrite CVVDP alpha/display policy
with that value. Either define a matching contract, keep a per-metric setting,
or reject unsupported alpha inputs before processing.

## Allocation, lifetime and dispatch plan

Keep a metric-local layout planner used unchanged by pipeline construction and
`GetMemoryRequirements`. Inputs include every option affecting spatial scales,
temporal support, history, diagnostics and precision. If fps affects the
filter-support length, it affects planned memory; if clip length matters,
require a bounded configuration or document a distinct offline interface.

For the pinned default video FIR,
`N = 2 * ceil(0.250 * fps / 2) + 1`. A direct FP32 ring with N slots for two
inputs and three DKL components per input costs
`2 * 3 * N * 4 * width * height` bytes before any spatial working set,
source storage, descriptor tables, temporally filtered outputs or readback.
The transient channel is derived from the achromatic history; storing four
history channels would add unnecessary memory and misstate the input contract.

At 60 fps, N=17:

| Geometry | Two-input three-component FP32 N-slot ring only |
| --- | --- |
| 1920×1080 | 806.84 MiB |
| 3840×2160 | 3.15 GiB |

These are arithmetic capacity estimates, not allocations or measurements.
Retained past history alone is N-1 slots; the N-slot figure includes current
samples and is a useful direct design baseline. More compact buffering needs a
documented lifetime proof and numerical validation. Treat these figures as
an early feasibility gate: a single 4K worker may approach a storage-buffer
range limit before adding the pyramid, while two workers double private
history. `GetMemoryRequirements`, actual `GetMemoryUsage` and worst-case
direct/staged worker budgeting must include it. Score-only mode should remain
bounded by history plus working scratch and pooling accumulators, independent
of clip duration; optional saved diagnostics/heatmaps need separate budgeting.

The plan should expose named regions for source/descriptor storage, constants,
retained temporal samples, per-level working sets, partials and readback.
Build a lifetime table before aliasing scratch: resources retained across
pushes cannot share storage with upload regions or temporary reductions.
Separate cache identity from scratch reuse. Disabling streaming or resident
uploads must remove only their source regions, never temporal history.

Reuse `VulkanRuntime::Config`, `PlanStagingSources`, `BufferMemoryPlan`,
direct/staged allocation and runtime source resolution. Preserve:

- Source bytes use binding 3; descriptors/tables use binding 2. Each source
  conversion dispatch calls `BindSource` for its own image. Internal inputs
  use absolute arena offsets, external buffers use binding-relative offsets.
- One submission in flight per runtime/pipeline; all pipelines and buffers
  outlive dispatches that reference them. A temporal sequence owns history,
  even though each push may wait for its GPU submission.
- Full overflow/alignment/storage-range/dispatch-count/shared-memory/push-
  constant checks in planning and construction. Large temporal working sets
  may hit the single arena's `maxStorageBufferRange`; measure required memory
  rather than assuming existing 4K configurations imply CVVDP fits.
- Transactional creation/destruction of shader pipelines and safe readback on
  staged devices. One compact result readback per push/end is preferable to
  hidden per-feature waits, subject to exact algorithm dependencies.

No need has been demonstrated for new queues, a new memory allocator, generic
graph scheduling, tensor runtime, runtime shader compilation or a core Python/
PyTorch dependency. Do not introduce them speculatively.

## CLI, benchmark and provenance requirements

The image tool can first expose the verified still mode, but that is not full
video support. An ordered frame-list reader with explicit timebase is a small,
reproducible path to sequence testing without placing codec libraries in the
core. A true Y4M sequence reader belongs to `tools/image_io/` and must retain
header frame-rate information, validate matching sequence lengths/geometry
and define truncation errors. Generic compressed video decoding can remain
external until deliberately scoped.

Add `bench/driver/cvvdp.cc` rather than changing `RunVmaf` into an ambiguous
temporal-metric template. VMAF's driver repeats one decoded pair for warm-up
and measured pushes, then flushes outside the timed interval. CVVDP needs
separate documented workloads:

- Independent still pairs, resetting the still path as required.
- A real ordered clip, with warm-up that does not contaminate the measured
  sequence history, and all mandatory temporal tail processing included in
  end-to-end clip time.
- Optional repeated/static-frame, resident-input and borrowed-buffer probes,
  labeled as synthetic workloads with their history/reset policy.

Report fixed-count wall time, frames accepted/emitted, final score and a
well-defined checksum of the appropriate outputs. Do not sum arbitrary
per-frame diagnostic scores as though that verifies the sequence's final
pooling. Publish upload and resident paths separately. Reuse stage timer
plumbing, but assign new stages only after the feature decomposition is known.

Each run/reference fixture needs resolved model/version/assets, source format
on both sides, color overrides/conversion, display preset and numeric values,
viewing/angular geometry, fps/timebase, frame order/count, temporal padding,
precision, pooling, diagnostics, GPU/driver/build and input hashes. Store raw
logs under `artifacts/reports/`, generated sequence caches under `.cache/`, and
compact evidence plus reproduction commands in versioned notes.

## Proposed colocation and file boundaries

Start with responsibilities, then split at meaningful boundaries if an
implementation approaches 1,000 lines. Approximate implementation sizes below
are design budgets, not estimates of completed code or quotas.

```text
include/cvvdp.h                         public config/results/pipeline
src/metrics/cvvdp/
  pipeline.cc                          public ownership and input dispatch
  cvvdp_internal.h                     internal plans/results/contracts
  layout.h, layout.cc                  validation, resource/lifetime plan
  display/
    display.h, display.cc              preset resolution and display constants
    shaders/...                        conversion/display kernels
  temporal/
    temporal.h, temporal.cc            history, boundary state and filters
    shaders/...                        temporal kernels
  spatial/
    spatial.h, spatial.cc              decomposition and GPU recording
    shaders/...                        decomposition kernels
  perception/
    perception.h, perception.cc        exact perceptual stages/parameters
    shaders/...                        perceptual kernels
  pooling/
    pooling.h, pooling.cc              reductions and final score conversion
    shaders/...                        partial/reduce kernels
```

This full-metric target keeps phase host code with its shaders and leaves
ownership/layout at the metric root, mirroring VMAF's feature organization.
Create a phase directory when its implementation lands; do not prepopulate
empty scaffolding. Aim for 500–1,000 lines per substantial implementation,
allowing focused small headers/kernels and combining two closely related
small phases when justified. Split a large phase at a real responsibility,
not into pass-through wrappers. Keep generated model/parameter tables next to
the phase that consumes them with the generator/provenance.

If reference research reveals a simpler phase boundary, use that boundary;
the tree above is not an assertion about the CVVDP algorithm. Avoid a second
all-purpose `vulkan.cc` that duplicates feature sequencing already expressed
in phase recorders. `pipeline.cc` should show validate/resolve sources,
publish, record meaningful phases, submit/readback and advance history.
Phase classes borrow one runtime and document their arena units/lifetimes.

Mirror this ownership under `tests/metrics/cvvdp/` with independent CPU/reference
oracles, compact fixtures and phase-level checks. Share storage/packing test
helpers; retain independent metric arithmetic and reference implementations.

## Implementation phases and gates

1. **Freeze reference contracts.** Pin upstream revision, parameter/model
   assets, software environment and redistribution terms. Record still/video
   API behavior, source/display units, angular geometry, timebase, padding,
   spatial/temporal pooling and default settings. Generate compact independent
   intermediates for tiny stills and short ordered clips. Exit: numerical and
   API acceptance criteria can be written without guessing.
2. **Add configuration/planning and reference workflow.** Define public API,
   metric validation, memory planner and immutable resolved configuration.
   Add fetch/copy/reference runner and fixture generation outside the core.
   Add CPU tests for invalid values, overflow, layout/resource lifetime and
   empty/single/short sequence behavior. Exit: planner and reference inputs are
   reproducible; no production metric placeholder returns a fabricated score.
3. **Implement verified image path in phases.** Match display transform and
   each algorithm intermediate against independent fixtures before final-score
   parity. Integrate host views, resident/raw buffers and borrowed Vulkan
   inputs using existing runtime contracts. Exit: phase tolerances, final
   reference parity and identity behavior match the selected reference path.
4. **Implement temporal/sequence path.** Test lengths shorter/equal/longer than
   filter support, start/end padding, fps changes via reconstruction/reset,
   cuts/static/dynamic content, ordered output/drain, reset/failure and
   nontrivial pooling. Exit: full-sequence reference parity, not just repeated
   still-frame parity. Include all necessary tail work in benchmarks.
5. **Integrate supported surfaces.** Extend memory façade, resolved metadata,
   CLI/sequence reader, dedicated benchmark and unified façade if its output
   contract can represent the metric. Add package-consumer and parser tests.
   Exit: documented still/video scope and clean offline core configure/build.
6. **Validate portability and optimize only afterward.** Release CTest,
   `vqm_test --require-gpu`, synchronization validation, ASan/UBSan,
   `tools/check_quality.py --lint`, pinned-reference comparison and selected
   SDR/HDR/display/time workloads. Existing metrics retain all their golden
   scores/format pins. Benchmark performance changes with an explicit baseline
   and interleaved arms, separate resident/upload paths and score checks. Exit:
   measured correctness/memory/performance limits recorded in versioned notes.

Run the small checked-in oracle fixtures locally on software Vulkan without
network access or an external reference installation. Use the optional
reference workflow for pinned corpus regeneration. Run the required-GPU tests
on a discrete device when qualifying its correctness or performance.

## Open dependencies and rejected shortcuts

Remaining design questions are upstream/reference questions: supported
source/display domains and presets, exact display parameters and parameter
distribution, nondefault temporal padding/filter modes, diagnostic semantics,
numerical tolerances and licensing. The pinned default causal schedule and
clip-wide pooling above already rule out the VMAF sequence/mean-score shortcut.
No empirical performance bottleneck has been identified here.

Rejected for this plan: treating tagged PQ/HLG as sufficient HDR support,
silently using 100-nit Rec.709 preprocessing, reusing VMAF's motion/mean pooling,
discarding multi-frame flush tails, using CID22 alone for video correctness,
round-robin processing one clip across independent workers, timing temporal
tail work outside the clip benchmark, importing a Python framework into the
core, or broad shared-runtime refactoring before an exact reuse requirement is
demonstrated.

Only this note was authored by this review. Build, test, sanitizer, quality,
reference, corpus and GPU gates were not run because the work was restricted
to architecture research; they remain implementation-phase requirements.
