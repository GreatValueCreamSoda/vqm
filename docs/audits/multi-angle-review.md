# Detailed code, structure and correctness review

This archived review describes the working tree at the time of the audit.
See [Contributing](../../CONTRIBUTING.md) for current local verification.

This review covers the maintained working tree after the initial audit and
VMAF host consolidation. Seven independent review agents inspected different
responsibilities; the primary agent corroborated findings, implemented fixes
and ran the gates. Generated build outputs, downloaded corpora and external
reference checkouts are not maintained project source. Reference contracts,
model evaluation and provenance were checked where the project relies on them.

This is evidence from a detailed review and testing, not a guarantee that every
possible bug has been eliminated. The [initial audit](audit-correctness-readability.md)
records the earlier P1 repairs and their separate measurements. The
[VMAF consolidation](../../bench/notes/system/vmaf-host-consolidation.md) records that change's baseline.

## Review coverage

| Angle | Inspection | Result |
| --- | --- | --- |
| Vulkan runtime | Allocation cleanup, descriptors and source bindings, fences/tokens, semaphore contracts, staging, shared arenas, owned/borrowed device features | No additional confirmed P1/P2; validation and synchronization gates below |
| Metric numerics | SSIMULACRA 2 weights/score/pyramid, shared color conversion, Butteraugli levels/reduction, VMAF features/models/temporal state | Confirmed odd-size Butteraugli subsample bug; fixed |
| Public API | Installed headers, configuration validation, handle ownership, pools, shared/unified metrics, Reset/Flush, memory planning and exports | No additional confirmed supported-use bug |
| Input/security | PNG/JPEG ownership and error paths, Y4M parsing/layout, CLI numbers, reference download/extraction/build commands | Confirmed truncated JPEG scan recovery was accepted; fixed |
| Tests/benchmarks | Optional skips, CPU oracles, accuracy gates, alpha checksums, resident caching, worker timers, serial/shared/unified measurements and provenance | Throughput provenance and consistency gates repaired |
| Architecture | Whole maintained tree, responsibility boundaries, names, includes, source/shader placement, tests/oracles, tools, documentation and editor discovery | Current structure is coherent; editor exclusion and shared I/O boundary improved |
| Build/distribution | CMake options, dependencies, generated shaders, installed consumer, toolchains, CI, packaging and license/provenance rules | Confirmed sanitizer consumer link failure; fixed |

Reviews distinguished existing design limits from bugs. For example, VMAF owns
private temporal history, while the shared workspace intentionally serves the
two spatial metrics. Moved-from owners remain destructible/move-assignable;
calling undocumented scoring methods on them was not classified as a new API
contract violation. Historical benchmark observations were not treated as
current performance promises.

## Confirmed findings and fixes

### P2: Butteraugli doubled odd subsample edges twice

`src/metrics/butteraugli/shaders/butter_opsin.comp` populated its shared tile
with repeated-edge mirroring, summed four samples, then multiplied by two
again for each odd edge. Mirroring already supplied the missing sample. The
final subsample row/column therefore doubled, and the corner quadrupled.
Comparisons with a folded second level were affected even though identity
checks still returned zero.

The four-tap average now uses the mirrored samples without a second correction.
Its source provenance explains the equivalence to libjxl's in-bounds sum and
edge correction. The new CPU-oracle test covers 32x16, 33x16, 32x17 and 33x17
constant-color comparisons. Before the fix all three odd cases failed both
score and distance_p3 checks. For the 33x17 case, the old score was
16.28422546 versus the oracle's 20.34281183. All cases pass after the fix.

The test's byte-to-linear helper also now honors four-byte plane alignment;
its previous direct `pixels`/`2*pixels` offsets could not serve as a correct
oracle for odd-area byte images. Existing even-size golden constants did not
change. Twelve opsin shader variants changed; the other 101 active variants
remain byte-identical to the pre-cleanup snapshot.

### P2: JPEG decoding accepted missing compressed samples

`tools/image_io/io.cc` replaced libjpeg's fatal handler but retained warning
recovery. A valid 64x64 JPEG truncated immediately after its scan header or
midway through entropy data could return an image with synthesized gray pixels.
A CPU-only probe disabled Vulkan device discovery and showed both files
reaching device creation, rather than reporting a load error. Direct libjpeg
inspection found `JWRN_JPEG_EOF` followed by `JWRN_HIT_MARKER`; the default
emitter printed only the first warning.

The decoder now rejects `JWRN_HIT_MARKER` through its established heap-owned
error/jump state. It delegates other messages to the saved default emitter.
This deliberately preserves missing-end-marker-only recovery and benign
metadata warnings when no compressed samples are missing. Tests generate
zero/partial entropy failures and complete/EOI-only/JFIF-version controls with
GPU creation disabled. Both malformed cases fail before the fix and pass the
new rejection checks after it. Independent follow-up review checked the
longjmp/destructor and callback lifetimes.

### P2: Installed sanitizer consumer dropped the parent toolchain flags

`tests/package_consumer.cmake` configured a fresh consumer without the parent
compiler and sanitizer flags. Linking the instrumented static core consequently
failed on `__asan_*` and `__ubsan_*` symbols. This was reproduced with the
existing ASAN build's `package_consumer` CTest.

A test-only initial-cache file now carries compiler/toolchain, generator,
sysroot/dependency paths, and configuration-specific compile/link settings to
installed and embedded consumers. Bracket quoting preserves spaces, semicolons,
dollars and bracket sequences; a round-trip check exercised those cases.
The installed prefix is prepended without losing inherited dependency paths.
The settings helper/cache are not installed with the library. Release and
sanitizer consumer tests pass, and CI now tests the sanitizer consumer as well.
Native consumers still execute; cross consumers build without directly running
a foreign executable.

### P3: Throughput reports did not fingerprint the measured inputs

`bench/run_inmemory.py` captured no inputs in throughput-only mode and reused
latency input hashes in combined mode. Throughput actually reads separate
`synthetic-<size>.txt` lists. Reports now independently record those selected
list paths, list hashes and image hashes. Latency retains its own provenance.
Tests cover throughput-only and combined mode using distinct inputs.

### P3: Fixed-window throughput lacked a score-consistency gate

The window parser checked checksum validity but did not compare per-call
scores across worker counts/repetitions. The fake integration even supplied
the same checksum for different call counts. This was a missing benchmark
gate, not evidence that the current worker implementation computed bad scores.

Synthetic window workloads now require one pair, and each implementation/size
must keep `checksum / calls` consistent across its arms, allowing printed
rounding and floating summation error. Different implementations are not
required to produce identical scores. Tests cover worker/repetition drift,
rounding, supported infinite reference results, multiple-pair rejection and
input provenance. All 34 focused benchmark-tool tests pass. Aggregate agreement
cannot prove oracle accuracy or exclude compensating individual worker errors.

### P3: Editor search excluded maintained build scripts

`.vscode/settings.json` used `**/build*`, hiding `tools/build_references.py` as
well as generated builds. Search exclusions now target `**/build/**` and
`**/build-*/**`, keeping maintained build tooling discoverable.

## File structure and readability assessment

The current [architecture guide](../architecture.md) is the ownership
map. The maintained boundaries are:

```text
include/                  installed image types, metric APIs and umbrella
src/
  input.*, image_view.*   format/storage validation and host views
  handles.*, pool.*       identity, ownership and batch scheduling
  metrics/
    metrics.cc            alignment and combined evaluation
    shared_workspace.*    shared spatial scratch ownership
    ssimulacra2/          facade, layout, GPU recording, score and shaders
    butteraugli/          facade, layout, GPU recording, tables and shaders
    vmaf/
      features/           preprocessing, ADM2/ADM3, VIF and their shaders
      cambi/, speed/      complete v1 feature passes and their shaders
      v0/, v1/            temporal pipelines and models beside generated tables
  vulkan/                 shared device/memory/source/dispatch machinery
    shaders/              common GLSL bindings and decoding
cli/                      argument parsing, presentation and scoring policy
tools/image_io/           decoder and shared color-option tables
tests/                    common/input/metric contracts, fixtures and oracles
bench/                    timed drivers, reference harnesses and measurement notes
validation/               corpus fetching, scoring and correlation
cmake/                    build-generation and test configuration helpers
docs/                     API, architecture, usage and metric guides
```

VMAF remains consolidated at 9 implementation files and 10 headers, with
generated model tables and individual shaders kept separate. In this review's
final source snapshot, production `.cc` implementations across library, CLI,
image I/O and timed driver total 41 files / 14,861 lines; median 297, maximum 884. None exceeds
1,000 lines. Small model evaluators and focused headers have real ownership
boundaries and should not be padded or merged merely to meet a quota.
Larger test/oracle/reference/report files have cohesive purposes; their size
alone did not justify another split. VMAF tests' cross-version `model/` and
`pipeline/` grouping remains understandable alongside source version folders.

The independent include scan found no cycles and no installed-header references
into `src`. Maintained local Markdown file links resolve, and current code,
build files and tools have no stale former source/shader/test paths. Markdown
anchors and external URLs were outside that link check.

Additional readability changes in this pass:

- Exact image-format comparison has one internal definition, retaining the
  distinction between implicit and explicit depth declarations.
- ADM2/ADM3 share the four-scale dispatch phase; model-specific validation,
  pipelines, coefficient tables and result sizes remain explicit.
- Native coded-depth support has one definition for input detection/validation,
  preserving their distinct false/throw behavior and error precedence.
- Spatial facades retain only persistent geometry/backend/handle state.
  Constructor-only scale plans/kernel data are local; backends copy retained
  data. Constructors consistently use `make_unique`.
- Pool failure draining is separate from normal scheduling, with original
  exception precedence, fence waits and poisoning behavior preserved.
- Unified metric execution names its spatial dispatch phases, leaving frame
  alignment and semaphore sequencing visible in the main operation.
- Butteraugli benchmark dark/bright calls share input-selection logic without
  moving timers. CLI/benchmark help uses readable raw literals and remains
  byte-identical at runtime.
- Shared decoder/table helpers use `vqm::image_io` and the common image header;
  `vqm::cli` owns command-line behavior. Public installed headers are unchanged
  by this follow-up. Documentation now correctly distinguishes the two-metric
  CLI from the four-metric library and benchmark.

## Validation scope and remaining limits

Release and ASAN/UBSan builds, both complete CTest suites, 196 hardware Vulkan
tests, formatting and production C++ lint pass. The installed sanitizer
consumer passes with inherited flags. Hardware and Mesa software Vulkan pass
the 196-test validation suite, both with and without synchronization validation.
No VUIDs, synchronization hazards or GPU-dependent skips were reported.

| Final gate | Result |
| --- | --- |
| Release build, warnings as errors | PASS |
| Hardware `vqm_test --require-gpu` | 196 tests, zero failed checks |
| Release CTest | 12/12 passed |
| ASAN/UBSan `vqm_test --require-gpu` | 196 tests, zero failed checks |
| Complete ASAN/UBSan CTest | 12/12 passed, including installed consumer and CLI decoding |
| Hardware/software Vulkan validation | 196/196 tests passed on each device |
| Hardware/software synchronization validation | 196/196 tests passed on each device |
| Recursive formatting / production nesting lint | PASS; 199 maintained files formatted, 41 compiled C++ files linted |
| Focused benchmark-tool unit tests | 34 passed |
| CLI / benchmark help output | Byte-identical to pre-cleanup output |
| SSIMULACRA 2 / Butteraugli identity | `100.00000000` / `0.00000000` |
| CID22, 300 pairs, MCOS tolerance 0.005 | `accuracy vs MCOS: PASS`, overall `PASS` |

The separate absolute-reference parity diagnostic still reports the existing
SSIMULACRA 2 divergence: maximum 0.524941 and mean 0.202585 against Cloudinary.
Correlations `(KRCC, SRCC, PCC)` are `(0.8145, 0.9554, 0.9433)` versus
`(0.8144, 0.9554, 0.9436)`, satisfying the required correlation gate.
This was not relabeled as strict numerical-reference parity.

CI now enables synchronization validation and rejects synchronization-hazard
diagnostics. Full local synchronization runs used the supported legacy enable
alias; CI uses `VK_LAYER_VALIDATE_SYNC=1`, which also passed a startup probe.
The installed layer's configuration metadata and
[Khronos' 1.3.275 configuration code](https://raw.githubusercontent.com/KhronosGroup/Vulkan-ValidationLayers/v1.3.275/layers/layer_options.cpp)
confirm the mapping to the same `validate_sync` setting.

The quality command covers maintained C++/headers/GLSL under its documented
roots and 41 compiled production C++ files. It is not a universal formatter or
analyzer for Python, shell, CMake, C or every optional reference driver.
Python compilation and relevant unit/integration suites were also exercised.

Remaining test/review limits include driver-error fault injection, another
physical GPU memory topology (forced Staged paths are tested), Windows/MSVC,
multi-configuration generators and compiling the embedded consumer beyond its
configuration check. External reference internals were not exhaustively
re-audited; their inputs/versions/provenance remain pinned. Other libjpeg
recovery behavior remains intentional. No additional confirmed P1/P2 issues
remain from these review scopes. CI workflow changes were checked locally;
the remote CI service was not executed from this session.

Raw before-fix failures, after-fix logs, source manifests, help-output comparisons
and performance records are retained under `artifacts/reports/source-readability/`.

## Interleaved performance check

The baseline is a complete frozen snapshot of the working tree immediately
before this cleanup, including the preceding audit/consolidation changes. A
comparison with Git HEAD alone would include those earlier changes. Both arms
were built independently with Release, GNU C++ 16.2.1, matching build options,
and the RTX 5070 / driver 615.78.08. The baseline source archive and executable,
source/hash manifests and measurement binary hashes are retained with the logs.

Each workload used five A/B alternations in one sitting. Resident runs use
20 iterations / 3 warm-ups / resident 2; streaming runs use 8 / 2. Half planes
are separate arms. Two-worker runs use 2,000 timed calls total and 3 warm-ups
per worker. Unified metric-pair mode measures both spatial metrics serially,
using 20 / 3 with preuploaded Vulkan buffers and 8 / 2 for streaming. No
profilers or other test jobs ran concurrently. There is no claimed optimization
or required speedup for this correctness/readability change.

The 3840x2160 RGB input pair has these SHA-256 hashes:

```text
original   756b05ccfb73363a0baf14c6909e8da9fd8938fb68847858cbf34f9892d563f8
distorted  aa1b297ab18ed2c4bce9b6c6a920ae2218f02811d389f45c57a9545cd7ad6d28
```

Per-call wall-clock medians and full five-run ranges are below in milliseconds.
The unified rows are medians of fixed-count average serial-pair wall times;
the ordinary single-worker rows use the driver's per-call median. Each arm
retains its own identical measurement definition.

| Workload | A median [min, max] | B median [min, max] | Change |
| --- | --- | --- | --- |
| SSIMULACRA 2 resident | 1.163638 [1.160669, 1.167893] | 1.168228 [1.159684, 1.169543] | +0.39% |
| SSIMULACRA 2 streaming | 3.383332 [3.375852, 3.392437] | 3.384057 [3.376387, 3.400297] | +0.02% |
| Butteraugli resident | 9.233619 [9.227474, 9.263094] | 9.229939 [9.227429, 9.247694] | -0.04% |
| Butteraugli streaming | 14.930709 [14.904194, 14.962498] | 14.915354 [14.910769, 14.928323] | -0.10% |
| VMAF v0 resident | 1.929912 [1.909502, 1.939283] | 1.928148 [1.916458, 1.969602] | -0.09% |
| VMAF v0 streaming | 4.113662 [4.111337, 4.133652] | 4.120838 [4.112391, 4.142561] | +0.17% |
| VMAF v1 resident | 2.768866 [2.767291, 2.770156] | 2.755126 [2.740676, 2.772656] | -0.50% |
| VMAF v1 streaming | 4.951631 [4.941956, 4.975180] | 4.967180 [4.958865, 4.990880] | +0.31% |
| SSIMULACRA 2 half resident | 1.098689 [1.091795, 1.098885] | 1.094720 [1.091614, 1.097049] | -0.36% |
| SSIMULACRA 2 half streaming | 3.306783 [3.300103, 3.317683] | 3.309462 [3.303498, 3.317933] | +0.08% |
| Unified, Vulkan buffers | 13.961803 [13.920515, 13.995064] | 13.943395 [13.922466, 13.969475] | -0.13% |
| Unified, streaming | 18.374210 [18.333106, 18.415508] | 18.386967 [18.360294, 18.429841] | +0.07% |

Two-worker fixed-count wall-clock totals are
`2.189850 [2.184812, 2.190180]` seconds for A and
`2.189928 [2.188462, 2.190408]` seconds for B: +0.004% for the same 2,000 calls.
This is an aggregate wall-clock comparison, not a sum of worker/process rates.

Every corresponding reported score and checksum matches exactly across all 130 runs,
including separate half-plane and unified arms. Stage tables were checked and
are retained in full. The changed Butteraugli opsin stage's resident median
is 1.005955 ms for A and 1.005306 ms for B. ADM2 medians are 0.497128 / 0.496845
ms; ADM3 medians are 0.578045 / 0.578301 ms. All whole-pair median changes are
within 0.5%, with overlapping observed ranges, so this check resolves no material
regression. It makes no speedup claim or general promise for other images or
devices. The odd-size correctness repair is established by the independent
oracle tests; it is not inferred from the even-size timing pair.

`performance/results.json` and `performance/unified-results.json` contain every
run, commands, binary hashes and per-stage summaries. Standard workload records
also include process-bracketing clock telemetry. No results from the preceding
audit/consolidation sittings were combined with this sitting.
