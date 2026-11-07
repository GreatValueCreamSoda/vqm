# Shared contracts and DRY refactor

This pass consolidates repeated contracts in the module that owns them. It adds
no implementation files and removes a net 122 production C++/header lines
across 30 existing files, including the new shared interfaces. Public headers,
shader sources, generated shader names and all 113 active shader binaries are
unchanged from the frozen pre-refactor working tree.

## Shared responsibilities

| Repeated responsibility | Single owner | Preserved contract |
| --- | --- | --- |
| Resident upload/release across all four metrics | `src/handles.h`, `ResidentTable` | Validation/idle checks stay at metric entry points; publish after backend success; stale release is idempotent |
| Default/borrowed context selection | `src/vulkan/vulkan_support.*` | Nullable acquisition for probing/factories, required acquisition for constructors/queries, original borrowed-device errors |
| Ordered pipeline-array creation, rollback and cleanup | `VulkanRuntime::CreatePipelines` / `DestroyPipelines` | Creation order, prior-handle rollback, explicit caller ownership and wait-before-destroy lifetime |
| Shared-workspace activation | `SharedMetricWorkspace::Impl::Activate` | Both idle checks precede the cached-state check; clear before publication; commit active metric only after success |
| ADM2/ADM3 buffer validation | `ValidateAdmBuffers` in `features/vmaf_adm.cc` | Original order, geometry checks, 6/9 partial terms, 9/13 result words and distinct diagnostics |
| Unified input configuration | `SelectConfigs` in `src/metrics/metrics.cc` | Shared geometry/streaming/memory policy, no resident pools; metric-specific fields stay explicit |
| CLI/benchmark color flags and overrides | `tools/image_io`, `InputColorOptions` | Same names, accepted values, defaults, parse order, diagnostics and metadata application |
| Serial benchmark decode policy | `bench/driver`, `PairDecoder` | Same ordered prefetch, thread defaults and synchronous attempts; original timed decode bodies unchanged |
| Butteraugli CSV rows/output | Existing `Row` / `WritePairsCsv` | Same header, quoting, six-decimal formatting and write-failure handling |

Six fixed VMAF feature constructors share transactional pipeline creation.
Conditional VIF and SDR selection remains explicit, with common destruction.
Spatial backends also reuse destruction while clearing their own handles and
retaining their fence waits. Runtime/context lifetimes remain visible in the
owning classes; no resource-owner hierarchy was introduced.

Dead resident entries now clear their device index consistently. VMAF previously
left that private value behind; dead entries are rejected by lookup and are
overwritten on reuse, so no handle/cache/temporal behavior changes.

`InputColorOptions` keeps tables private and exposes typed flag identification,
parsing and application. Each frontend still owns argument consumption and
error presentation. Its existing floating-parameter grammar is preserved.
`PairDecoder` stores decode policy and its optional prefetch queue; metric
construction, scoring and timing remain in the metric drivers. CSV path-string
ownership/copying stays outside scoring timers.

Metric arithmetic, temporal policies, typed public overloads and independent
CPU reference oracles retain their separate responsibilities. The repository
organization guidance now states this ownership approach to sharing contracts.
The [architecture guide](../../../docs/architecture.md) identifies the owners.

## Correctness evidence

Three new tests cover publication and rollback rather than implementation
syntax:

- Upload callbacks that return no slot or throw leave the host entry free;
  a full table does not invoke another backend upload.
- Invalid/foreign/stale releases do not invoke the backend, a failed release
  keeps the handle live, and successful release/reuse remains generation-safe.
- Pipeline batch creation fails on a missing second shader after creating a
  real first pipeline; rollback frees it, and a subsequent valid batch succeeds.
  Vulkan validation at device teardown checks for leaked objects.

Independent cross-review checked callback ordering, activation failure state,
resource lifetimes, parser/prefetch behavior and ADM validation equivalence.
The new resident tests are CPU-only and require no driver fault injection.

All 54 baseline/candidate frontend probes match exit status, stdout and stderr
exactly. They cover help, unknown flags and each of the five shared color flags
with missing, invalid, valid and repeated values. Existing integration tests
cover scoring, CSV output, borrowed views/buffers and workers. The benchmark
`LoadOnce` and `MeasureDecode` bodies are unchanged.

Local logs and source/hash manifests are in `artifacts/reports/dry-refactor/`.

## Final gates

| Gate | Result |
| --- | --- |
| Release build, warnings as errors | PASS |
| `vqm_test --require-gpu` | 199 tests, zero failed checks |
| Release CTest | 12/12 passed |
| ASAN/UBSan `vqm_test --require-gpu` | 199 tests, zero failed checks |
| Complete ASAN/UBSan CTest | 12/12 passed |
| Vulkan + synchronization validation, RTX 5070 | 199 tests, zero errors/hazards/skips, exit 0 |
| Vulkan + synchronization validation, Mesa software device | 199 tests, zero errors/hazards/skips, exit 0 |
| Recursive clang-format 23 | PASS; 199 maintained files |
| Production clang-tidy | PASS; 41 compiled C++ files |
| Frontend baseline/candidate probes | 54 exact exit/stdout/stderr matches |
| CSV comparison, paths containing comma/quote | Same metadata, escaping, six-decimal precision and score |
| SSIMULACRA 2 / Butteraugli identity | `100.00000000` / `0.00000000` |
| CID22, 300 pairs, MCOS tolerance 0.005 | `accuracy vs MCOS: PASS`, overall `PASS` |
| Shader sources / active binaries / installed public headers | Byte-identical to baseline |

The corpus runner's separate absolute-reference parity diagnostic retains the
existing Cloudinary divergence documented in the preceding audit. Correlation
triples `(KRCC, SRCC, PCC)` remain `(0.8145, 0.9554, 0.9433)` for this library
versus `(0.8144, 0.9554, 0.9436)` for Cloudinary. No score constants or metric
arithmetic changed in this refactor.

## Interleaved regression measurement

The baseline is a complete snapshot of the pre-DRY working tree, including the
preceding audit fixes. Both arms were built separately with Release, GNU C++
16.2.1 and matching options on the RTX 5070 / driver 615.78.08. Comparing Git
HEAD alone would incorrectly include earlier uncommitted changes. Baseline
source/archive/binary hashes and all measured commands are retained locally.

Each of 15 workloads ran five A/B alternations without profiling or concurrent
test jobs. Resident runs use 20 timed iterations, 3 warm-ups and resident 2;
streaming runs use 8 / 2. Half planes are separate arms. The two-worker arm
uses 2,000 timed calls total and 3 warm-ups per worker. Shared and unified
serial two-metric arms use 20 / 3 with preuploaded Vulkan buffers and 8 / 2 for
streaming. Each arm keeps the same timer definition and input preparation.

The 3840x2160 RGB pair hashes are:

```text
original   756b05ccfb73363a0baf14c6909e8da9fd8938fb68847858cbf34f9892d563f8
distorted  aa1b297ab18ed2c4bce9b6c6a920ae2218f02811d389f45c57a9545cd7ad6d28
```

Per-call wall-clock medians and full five-run ranges, in milliseconds:

| Workload | A median [min, max] | B median [min, max] | Change |
| --- | --- | --- | --- |
| SSIMULACRA 2 resident | 1.168269 [1.159014, 1.172139] | 1.169214 [1.168374, 1.169759] | +0.08% |
| SSIMULACRA 2 streaming | 3.382428 [3.375189, 3.387214] | 3.381658 [3.380864, 3.399028] | -0.02% |
| Butteraugli resident | 9.232603 [9.218773, 9.246863] | 9.247543 [9.224528, 9.263203] | +0.16% |
| Butteraugli streaming | 14.914276 [14.900245, 14.964365] | 14.914755 [14.903955, 14.917330] | +0.003% |
| VMAF v0 resident | 1.906229 [1.903404, 1.927844] | 1.919393 [1.911788, 1.938148] | +0.69% |
| VMAF v0 streaming | 4.114593 [4.106943, 4.128024] | 4.117038 [4.104089, 4.121249] | +0.06% |
| VMAF v1 resident | 2.759047 [2.736333, 2.769122] | 2.767792 [2.736912, 2.769702] | +0.32% |
| VMAF v1 streaming | 4.942212 [4.940437, 4.971577] | 4.953342 [4.940817, 5.001192] | +0.23% |
| SSIMULACRA 2 half resident | 1.089905 [1.089700, 1.100010] | 1.098154 [1.094200, 1.107290] | +0.76% |
| SSIMULACRA 2 half streaming | 3.313720 [3.299680, 3.320650] | 3.311524 [3.307219, 3.313060] | -0.07% |
| Unified, Vulkan buffers | 13.976143 [13.952408, 14.005703] | 13.980769 [13.964838, 14.008474] | +0.03% |
| Unified, streaming | 18.389711 [18.338783, 18.424775] | 18.357963 [18.339226, 18.392175] | -0.17% |
| Shared, Vulkan buffers | 13.923470 [13.914246, 13.969231] | 13.948406 [13.922090, 13.994410] | +0.18% |
| Shared, streaming | 18.374310 [18.334140, 18.395360] | 18.374082 [18.350297, 18.857690] | -0.001% |

Single-worker metric rows use the driver's per-call median; shared/unified
rows use the median of five fixed-count average serial-pair wall times.
The two-worker fixed-count wall totals are
`2.190362 [2.184604, 2.190718]` seconds for A and
`2.189670 [2.189438, 2.190734]` seconds for B, a -0.032% median change.
These compare the same 2,000 calls, rather than summing worker/process rates.

All corresponding reported scores/checksums match exactly across 150 runs;
every stage table was checked and retained. Whole-pair median changes stay
within 0.8%, with overlapping observed ranges. This resolves no material
regression for the measured pair and makes no speedup claim. No medians from
earlier audit/consolidation sittings were combined with this sitting.

`performance/results.json` and `performance/metric-pair-results.json` include
all runs and per-stage summaries. The standard workload records also retain
process-bracketing GPU clock telemetry. Timings remain specific to this device,
driver, pair and workloads.
