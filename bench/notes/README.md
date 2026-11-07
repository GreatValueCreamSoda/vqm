# Benchmark and research archive

To measure the current build, start with the [benchmark guide](../README.md).
It covers the hosted video suite, still-image peer comparisons and direct
`vqm_bench` runs. Use the [validation guide](../../docs/validation.md) for
reference agreement and the [contributing guide](../../CONTRIBUTING.md) for
local build and test commands.

The reports here describe the builds, hardware, inputs and policies recorded in
each study. Use the [interleaved protocol](../../AGENTS.md#protocol) when
comparing a current baseline and candidate. Historical commands, output paths,
test counts and references to earlier development workflows are part of the
record.

| Topic | Research records |
| --- | --- |
| Benchmark workflow and qualification | [Fresh clone video benchmark](system/fresh-video-benchmark.md), [five-metric reference correlation](system/all-metric-reference-correlation.md), [ranking policy transition](../../docs/design/ranking-default-gates.md) |
| Historical NVIDIA video input | [NVDEC CLI qualification and frame-transfer trace](system/nvdec-cli-qualification.md) |
| SSIMULACRA 2 and shared Vulkan infrastructure | [Vship comparison](system/vship-notes.md), [resident ring workload](system/resident-ring.md), [shared memory](system/shared-memory.md), [other system studies](system/) |
| Butteraugli | [Reference drift and JPEG ICC investigation](system/butteraugli-drift.md), [memory layout](system/butter-memory.md) |
| VMAF | [Speed investigation](vmaf/vmaf-speed-investigation.md), [FP16](vmaf/vmaf-fp16.md), [cooperative matrices](vmaf/vmaf-cooperative-matrix.md), [SpEED parallelism](vmaf/vmaf-speed-parallelism.md), [CAMBI headroom](vmaf/vmaf-v1-headroom.md), [supporting reviews](vmaf/research/README.md) |
| ColorVideoVDP | [Implementation and initial benchmarks](cvvdp/cvvdp-implementation.md), [reference qualification](cvvdp/cvvdp-reference-qualification.md), [Vship still comparison](cvvdp/cvvdp-vship-still-comparison.md) |
| PSNR-HVS | [Implementation](psnr_hvs/psnr-hvs-implementation.md), [initial resident results](psnr_hvs/resident-throughput-20261106.md), [interleaved optimization study](psnr_hvs/optimization-20261106.md) |
| Publication and local verification | [2026-11-06 GitHub readiness audit](../../docs/audits/github-readiness-audit-20261106.md), including the PSNR-HVS tests added after its initial measurements |

The [measurement archive](measurements/README.md) indexes compact data and
preserved generated summaries. The [experiment guide](../experiments/README.md)
indexes reusable probes and identifies the source snapshots needed for older
variants. The original integration and validation plans are in the
[design archive](../../docs/design/).

On 2026-11-05, authored reviews and generated summaries were moved from
`build/reports/` into this tree. The ignored `artifacts/reports/` path in
historical records refers to local raw runs, experiment patches, snapshots,
and binaries. New runs create reports there as needed. Regenerable datasets
go under `.cache/`.

Record new research with its question, method, build, input and hardware
provenance, results, correctness limits and rejected alternatives. Keep compact
evidence with the note and preserve useful probes under `bench/experiments/`.
See the [retention rules](../../AGENTS.md#workspace-files-and-retention).
