# Correctness and organization audit

This archived audit describes the 2026-11-05 working tree. See
[Contributing](../../CONTRIBUTING.md) for the current local checks.

Measured on 2026-11-05 UTC against commit `a949b61`. The changed tree preserves
the metric arithmetic and public entry points while fixing invalid input,
shader access, capability negotiation and state publication bugs. No golden
score constants changed.

## Ownership and traceability

The ownership map is in [the architecture guide](../architecture.md).
All metric implementations now live under `src/metrics/`: `ssimulacra2/`,
`butteraugli/` and `vmaf/`. Their layout planners, command recording and shaders
live with their implementations. VMAF feature shaders live with `features/`,
`cambi/` or `speed/`; only common bindings and input decoding remain under
`src/vulkan/shaders/`. Model tables remain beside their models. Tests mirror
metric ownership under `tests/metrics/`.

The Vulkan runtime owns resources and synchronization. It no longer owns
metric-specific implementations or their layout declarations. Cross-component
includes use a single `src` root. Shared image types are in `vqm_image.h`,
SSIMULACRA 2 declarations are in `ssimulacra2.h`, and `vqm.h` remains the public
umbrella. Leaf metric headers do not depend on that umbrella.
Consumers that used SSIMULACRA 2 through another leaf header's incidental
transitive includes should include `ssimulacra2.h` or `vqm.h` explicitly.

Internal interfaces and timing stages now identify the metric they implement;
shared input decoding and benchmark pipeline traits have generic names.
Benchmark selection uses one enum instead of overlapping metric booleans.
The driver is split by metric and workload without changing its measurement
boundaries. `vqm`, `vqm_bench` and `vqm_test` are the canonical executables;
the `ssimulacra2*` compatibility commands and build targets remain available.

Generated shader identifiers remain stable. The configured manifest replaces
the stale directory-glob shader count: 112 production variants, plus one
test-only SpEED numerical oracle when tests are enabled. Production SpEED
shader bytes are unchanged by the oracle support.

## Correctness repairs

- SSIMULACRA 2 prepare: inactive lanes no longer index beyond a shared pyramid
  level. All lanes still participate in the required barriers. The level-two
  64x64 path previously allowed an access at index 768 of a 768-element array.
- VMAF v0 input selection: staging/upload masks use the same native-versus-SDR
  decision as dispatch, including pair coding and retained sequence domain.
  A 64x48 mixed BT.709/BT.2020 reproduction previously returned
  `98.518423632` through streaming and `93.138920061` through resident input;
  both now return `93.138920061`. Tests cover host views, resident input and
  automatic/staged memory paths.
- Allocated input validation: malformed image formats are rejected before
  pool reservation. A malformed subsampled RGB float allocation could make a
  comparison read a neighboring allocation. Rejected allocations leave the
  pool usable. Enum validation also prevents an invalid range/depth pair from
  reaching an undefined negative shift.
- Vulkan capabilities: VIF's half-source path requires enabled `shaderFloat16`,
  separately from 16-bit storage access. Owned devices negotiate the feature;
  borrowed devices declare it through the default-false `shader_float16`
  field. Unsupported paths use FP32. This follows the
  [Vulkan feature contract](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceShaderFloat16Int8Features.html).
- Score conversion: nonfinite inputs and overflowing intermediate results
  raise errors instead of becoming a perfect score. Finite arithmetic retains
  its evaluation order. Half-float subnormal conversion in the input oracle
  also has independent boundary coverage.
- State publication: resident slot state is published after successful upload;
  Butteraugli reference cache keys are published after successful submission.
  Failed recording/submission cannot leave a reusable key for incomplete work.
- ADM: tiled padding loads are clamped before single-reflection addressing.
  ADM3 no longer reserves an unused distorted-detail buffer, saving
  24,883,200 arena bytes at 3840x2160.
- Decoding and CLI: Y4M numeric parsing is overflow-checked, frame markers are
  exact, PNG/JPEG decoder state survives their longjmp error paths safely,
  and CLI exceptions are caught across loading and metric execution.
- Timing and test lifecycle: timestamp subtraction respects queue counter
  width. The test harness releases its default Vulkan context before process
  teardown; validation-clean execution now includes successful exit.

Malformed decoder fixtures, invalid ingestion, overflow, state contracts,
capability fallback and benchmark option replacement have regression coverage.
Optional dependency imports are separated from imports under test so broken
modules cannot silently turn failures into skips. CI installs the image/video
test dependencies, runs recursive formatting and production C++ nesting lint,
and rejects Vulkan validation diagnostics.

## Correctness and quality gates

The final reorganized tree passed:

| Gate | Result |
| --- | --- |
| Release build, warnings as errors | PASS |
| `vqm_test --require-gpu`, RTX 5070 | 195 tests, zero failed checks, no skips |
| `ctest --test-dir build --output-on-failure` | 12/12 passed |
| ASan/UBSan `vqm_test --require-gpu` | 195 tests, zero failed checks |
| Khronos validation, RTX 5070 | 195 tests, zero failed checks, no VUID/errors/skips, exit 0 |
| Khronos validation, Mesa software Vulkan | 195 tests, zero failed checks, no VUID/errors/skips, exit 0 |
| Recursive clang-format 23 and production clang-tidy | PASS; 54 compiled C++ files linted |
| Standalone library-only build, tests/CLI/bench disabled | PASS; 112 active shader variants |
| SSIMULACRA 2 identical gradient | `100.00000000` |
| Butteraugli identical gradient | `0.00000000` |
| CID22, 300 pairs, MCOS correlation tolerance 0.005 | `accuracy vs MCOS: PASS` and overall `PASS` |
| Same CID22 scores versus prechange binary | All 300 exactly equal; maximum and mean absolute delta 0 |

CID22 correlations `(KRCC, SRCC, PCC)` are `(0.8145, 0.9554, 0.9433)` for
this implementation and `(0.8144, 0.9554, 0.9436)` for Cloudinary. The separate
strict absolute-reference parity diagnostic remains `FAIL` (maximum absolute
difference 0.524941, mean 0.202585). That is existing metric divergence, not
the required correlation gate or a change introduced here; see
[divergence](../divergence.md).

Software Vulkan exposed pre-existing test portability assumptions. Shared
workspace tests now compare against standalone pipelines on the same device.
The public identity test retains the documented CPU-device band and exact
hardware identity. SpEED tests retain hardware pins and use matrix residuals
and a same-device legacy QR oracle for ill-conditioned software-driver cases.
No production numerical algorithm changed to accommodate those tests.

Local gate logs are under `artifacts/reports/audit-gates/`; the corpus report,
per-pair results and baseline scores are under `artifacts/reports/audit-validation/`.

## Interleaved regression measurements

Both arms used Release builds, GNU C++ 16.2.1 and the same RTX 5070 with driver
615.78.08. Arm A was a separate worktree at `a949b61`; arm B was the changed
tree. Each workload ran five A/B alternations in one sitting, without a
profiler. Resident runs used `--iterations 20 --warmup 3 --resident 2`;
streaming runs used `--iterations 8 --warmup 2`. Half planes were separate
arms. The two-worker workload used 2,000 timed calls total and three warm-up
calls per worker; its wall clock begins before the first timed submission.

The 3840x2160 RGB pair has these SHA-256 digests:

```text
original   756b05ccfb73363a0baf14c6909e8da9fd8938fb68847858cbf34f9892d563f8
distorted  aa1b297ab18ed2c4bce9b6c6a920ae2218f02811d389f45c57a9545cd7ad6d28
```

The final sitting's per-call wall-clock medians and full five-run ranges are
below, in milliseconds. Negative changes are faster. These are regression
observations for this pair, not general throughput promises.

| Workload | A median [min, max] | B median [min, max] | Change |
| --- | --- | --- | --- |
| SSIMULACRA 2 resident | 1.205808 [1.201738, 1.210508] | 1.184508 [1.174979, 1.392360] | -1.77% |
| SSIMULACRA 2 streaming | 3.595554 [3.466436, 4.240581] | 3.609580 [3.431367, 3.690633] | +0.39% |
| Butteraugli resident | 10.036265 [9.595612, 11.158275] | 10.089514 [9.253180, 11.278567] | +0.53% |
| Butteraugli streaming | 16.231824 [15.750026, 17.489446] | 16.431881 [14.968211, 17.145031] | +1.23% |
| VMAF v0 resident | 2.015867 [1.915288, 2.641628] | 1.992427 [1.927824, 2.707302] | -1.16% |
| VMAF v0 streaming | 4.128202 [4.120708, 4.177217] | 4.163217 [4.101528, 4.227101] | +0.85% |
| VMAF v1 resident | 3.246959 [2.831376, 3.451391] | 2.971893 [2.752832, 3.524796] | -8.47% |
| VMAF v1 streaming | 4.951501 [4.943416, 5.126568] | 4.954516 [4.941431, 5.029375] | +0.06% |
| SSIMULACRA 2 half resident | 1.129809 [1.126854, 1.138859] | 1.091910 [1.090475, 1.101390] | -3.35% |
| SSIMULACRA 2 half streaming | 3.677374 [3.386032, 4.105383] | 3.371978 [3.352173, 3.413682] | -8.30% |

SSIMULACRA 2 prepare, the shader stage affected by the shared-memory bounds
fix, is `0.428347 [0.427770, 0.428552]` ms for A and
`0.395648 [0.395218, 0.396563]` ms for B: **7.63% faster**, exceeding the
2% stage speed gate. Scale medians are 0.739248 and 0.738854 ms. The largest
whole-pair median regression is 1.23%, within the observed spread; there is no
resolved performance regression in these measurements. The broad VMAF v1 and
half-streaming ranges do not support a precise speedup claim for those paths.

For two workers, the fixed 2,000-call wall-clock totals are
`2.617008 [2.611960, 2.774948]` seconds for A and
`2.510786 [2.504122, 2.593420]` seconds for B, a 4.06% median reduction.
This compares the same two-worker workload across arms; it does not claim a
worker-scaling benefit from individual pipeline rates.

Every run's score, checksum and stage table was checked. Corresponding A/B
scores and checksums match exactly, including the separate half-plane arms.
SSIMULACRA 2's normal per-call score is 92.437150; half planes produce
92.236057 and are not mixed with normal results. Warm-ups are included in
checksums, not timings.

Raw commands, output, CSV stage tables, binary hashes and before/after GPU
clock telemetry are preserved in `artifacts/reports/audit-performance/`. An earlier
final-layout sitting had severe variation in both arms (the unchanged baseline
resident SSIMULACRA 2 ranged from 1.20 to 2.84 ms); its complete data is retained
in `artifacts/reports/audit-performance-noisy-layout/`. That uncertainty prompted
the telemetry repeat above. A separate pre-layout sitting is retained in
`artifacts/reports/audit-performance-before-layout/`. No medians are combined or
compared across sittings. Telemetry samples bracket processes and do not prove
the clock or competing graphics load during individual dispatches.

## Build workspace and research retention, 2026-11-05

The default `build/` tree mixed required generated outputs with research notes,
experimental source snapshots and about 14 GiB of disposable corpus conversions.
Its authored knowledge is now indexed in the [benchmark archive](../../bench/notes/README.md). All
24 Markdown reports were preserved (five supporting reviews and 19 generated
summaries), with original bodies and historical context. Compact VMAF speed and
FP16 result exports live in the [measurement archive](../../bench/notes/measurements/README.md), and
three reusable probes with current recipes live in
the [experiment directory](../../bench/experiments/README.md).

Raw reports and historical experiment snapshots moved to ignored
`artifacts/reports/`; a hash manifest is in
`artifacts/build-cleanup-2026-11-05.json`. Recovery copies of previously built
reference tools are in `artifacts/snapshots/`. The original CID22 corpus and
read-only `.reference/` checkouts were preserved. The fssimu2 checkout already
contained an untracked `zig-pkg/` entry, so rebuilding every optional reference
was not claimed. Pinned VMAF tools were rebuilt successfully from their copied
source tree. Reference binaries are optional workflow inputs, not prerequisites
for configuring or building the core project.

Report defaults now use `artifacts/reports/`, profiler output has an independent
`--out-dir`, and generated benchmark/corpus caches use `.cache/`. The retained
benchmark pair lists were updated to the relocated cache. `AGENTS.md` documents
ownership, retention, cleanup, work preservation and the applicable gates.

Verification started with the previous `build/` directory removed. Release
configuration and a clean rebuild passed; the resulting tree contains about
109 MiB of disposable outputs rather than the previous 15 GiB. The required-GPU
suite passed all 204 tests with zero failed checks and no skipped tests, and all
12 CTest groups passed. CLI identities were exactly `100.00000000` for
SSIMULACRA 2 and `0.00000000` for Butteraugli. Formatting checked 200 maintained
files; production lint checked 41 C++ files. Both CPU probes reproduced their
original substantive JSON, and the compiled GPU sequence probe produced 288
frame rows with no errors. The root compile database was regenerated. Current
cleanup logs are under `artifacts/reports/workspace-cleanup/`. These storage and
workflow changes introduce no new performance claim.
