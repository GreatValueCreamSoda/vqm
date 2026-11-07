# VMAF host implementation consolidation

This follow-up to the organization audit consolidates small host files by
feature responsibility. VMAF now has **9 implementation files and 10 internal
headers**, down from 22 and 21. Public headers, generated model tables and
individual shader files are unchanged.

| Owner | Consolidation | Implementation lines |
| --- | --- | --- |
| `features/vmaf_preprocess` | Native luma, SDR conversion, fit/pad resizing, reference motion | 519 |
| `features/vmaf_adm` | Shared wavelet recording, ADM2 and ADM3 | 316 |
| `features/vmaf_vif` | VIF, retained as a separate feature | 260 |
| `cambi/vmaf_cambi` | Preparation, decimation, contrast, pooling and feature orchestration | 517 |
| `speed/vmaf_speed` | Filtering, statistics and feature orchestration | 297 |
| `v0/vmaf_pipeline` | GPU pipeline and ordered sequence state | 673 |
| `v1/vmaf_v1_pipeline` | GPU pipeline, motion3 timeline and ordered sequence state | 674 |
| `v0/vmaf_model`, `v1/vmaf_v1_model` | Model evaluation beside the generated tables | 102, 67 |

Each version has one internal header for temporal declarations. The small
model implementations and focused headers remain separate because they own
pure model evaluation and generated-data provenance. Shader files remain with
their features and retain independent compilation/debugging boundaries.

Existing stage classes and their buffer contracts remain visible in the
consolidated headers for focused tests. Named sections follow the feature's
stage order. Private push structs and validation helpers have stage-specific
names where merging anonymous namespaces would otherwise cause collisions.
Identical ADM push structs are shared, retaining both variants' layout asserts.
Arithmetic, validation order, error text, dispatch order, barriers, member
construction/destruction order and provenance are preserved.

CMake and all consumers use the consolidated files. Independent mechanical
comparisons verified moved definitions/declarations against the pre-change
snapshot, allowing only whitespace, private identifier changes and the verified
struct deduplication. All 41 VMAF shader source files, all 113 active shader
binaries and all 14 installed public headers are byte-identical to that snapshot.

## Correctness and quality

| Gate | Result |
| --- | --- |
| Release build with warnings as errors | PASS |
| `vqm_test --require-gpu` | 195 tests, zero failed checks, no skips |
| CTest | 12/12 passed |
| ASan/UBSan `vqm_test --require-gpu` | 195 tests, zero failed checks |
| Khronos validation on RTX 5070 | 195 tests, no VUID/errors/skips, exit 0 |
| Recursive clang-format 23 | 199 maintained files passed |
| Production clang-tidy | 41 compiled C++ files passed |
| SSIMULACRA 2 / Butteraugli identity | `100.00000000` / `0.00000000` |
| CID22, 300 pairs, accuracy tolerance 0.005 | `accuracy vs MCOS: PASS`, overall `PASS` |

CID22's separate absolute-reference parity diagnostic retains the existing
failure documented in the [audit report](../../../docs/audits/audit-correctness-readability.md).
This consolidation changes neither the metric nor its reference drift.

Logs, source/shader hashes and benchmark records are preserved locally under
`artifacts/reports/vmaf-consolidation/`.

## Regression measurement protocol

The baseline is a frozen source snapshot taken immediately before consolidation,
including the previous audit's uncommitted fixes. It was built independently in
`/tmp/vqm-consolidation-base/build`; comparing against Git HEAD alone would
incorrectly include those earlier fixes. Both arms use Release, GNU C++ 16.2.1,
the same build options and the RTX 5070 with driver 615.78.08. The baseline
source hash manifest and both binary hashes are retained with the records.
The pre-consolidation source archive and baseline benchmark executable are also
retained there; the temporary source/build directory is removed after checking.

Each VMAF version runs five A/B alternations for resident input
(`--iterations 20 --warmup 3 --resident 2`) and separately for streaming
(`--iterations 8 --warmup 2`). No profiler or other GPU gates run concurrently.
Before/after process telemetry is recorded. The 3840x2160 RGB pair is the same
as the audit, with SHA-256 digests:

```text
original   756b05ccfb73363a0baf14c6909e8da9fd8938fb68847858cbf34f9892d563f8
distorted  aa1b297ab18ed2c4bce9b6c6a920ae2218f02811d389f45c57a9545cd7ad6d28
```

The final standard sitting produced these per-call wall-clock medians and
five-run ranges, in milliseconds. Negative changes are faster.

| Workload | A median [min, max] | B median [min, max] | Change |
| --- | --- | --- | --- |
| v0 resident | 1.975072 [1.915568, 2.383961] | 2.058326 [1.926597, 2.166829] | +4.22% |
| v0 streaming | 4.264494 [4.195740, 4.346023] | 4.238774 [4.180715, 4.531535] | -0.60% |
| v1 resident | 3.145905 [2.879449, 3.374522] | 2.898594 [2.880553, 3.268598] | -7.86% |
| v1 streaming | 5.159996 [5.078472, 5.423752] | 5.173156 [5.041853, 5.322589] | +0.26% |

Short resident measurements are noisy. In the preceding isolated sitting,
resident medians instead moved +1.78% for v0 and +4.14% for v1, with overlapping
ranges in both arms. Those complete records are retained in
`performance-variable/`; the final standard sitting is in `performance/`.
These observations cannot support a precise resident speedup/regression claim.

To investigate, a separate sustained resident workload used **2,000 timed
calls per run**, still five A/B alternations, warm-up 3 and resident 2. Its
per-call medians/ranges are:

| Workload | A median [min, max] | B median [min, max] | Change |
| --- | --- | --- | --- |
| v0 sustained resident | 2.068145 [1.970421, 2.082539] | 2.065670 [1.993279, 2.146889] | -0.12% |
| v1 sustained resident | 2.906462 [2.850475, 2.999615] | 2.939755 [2.916686, 2.968615] | +1.15% |

The longer workload does not resolve a material regression: medians are within
1.15% and the ranges overlap. It supplements the required short protocol; the
two workloads and separate sittings are not combined. No speedup is claimed.
Process-bracketing telemetry does not determine the clocks or competing
graphics load during individual dispatches.

All score/checksum and stage-table checks pass in every sitting. Corresponding
A/B per-call scores and checksums match exactly: v0 scores `88.35260140`, v1
scores `92.11456420`. Sustained checksums are `176970.260598` and
`184505.472095`, respectively, covering 2,003 calls including warm-up. Raw
commands, CSV/stage tables, output, telemetry and binary hashes are retained in
`performance-long-resident/`. An initial sitting in
`performance-overlapped-start/` is excluded because two identity fixture checks
could have overlapped its beginning; its data and exclusion reason are retained.
