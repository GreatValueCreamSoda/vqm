# Measurement archive

These snapshots preserve the outputs and provenance of specific studies. For a
new run, follow the [benchmark guide](../../README.md) or the
[validation guide](../../../docs/validation.md). Each archived result describes
its measured build, hardware, workload and acceptance policy.

The 2026-11-05 archive moved generated summaries out of `build/reports/`.
The [snapshot index](snapshot-index.json) records their original locations
and hashes. Later reports add compact evidence alongside that collection.
Original commands retain their original paths. Raw runs, binaries and source
snapshots are local evidence under ignored `artifacts/reports/` and
`artifacts/snapshots/`.

## Benchmark and qualification records

| Workload | Evidence |
| --- | --- |
| Hosted video benchmark reproduction | [Setup and five-round run](fresh-video-benchmark.json), [method and results](../system/fresh-video-benchmark.md) |
| Five-metric hosted reference correlation | [Aggregate results](all-metric-reference-correlation.json), [per-pair scores](all-metric-reference-correlation-scores.csv), [method and domain limits](../system/all-metric-reference-correlation.md) |
| Ranking policy audits | [Balanced CID22](ranking-still-results.json), [hosted video results](ranking-temporal-results.json), [hosted video scores](ranking-temporal-scores.csv), [policy transition](../../../docs/design/ranking-default-gates.md) |
| Butteraugli drift and production qualification | [Cumulative corrections](butteraugli-drift-results.json), [ICC isolation](butteraugli-drift-icc.csv), [qualification and 40 timing runs](butteraugli-production-results.json), [investigation](../system/butteraugli-drift.md) |
| ColorVideoVDP implementation and comparison | [Implementation report](../cvvdp/cvvdp-implementation.md), [benchmark rounds](cvvdp-implementation-results.json), [reference qualification](cvvdp-qualification-results.json), [CID22 scores](cvvdp-cid22-results.csv), [Vship still comparison](cvvdp-vship-still-results.json) |
| PSNR-HVS resident measurements | [Initial throughput](../psnr_hvs/resident-throughput-20261106.json), [interleaved optimization study](../psnr_hvs/optimization-20261106.md), [optimization results](../psnr_hvs/optimization-20261106.json) |

## VMAF experiments

| Study | Evidence |
| --- | --- |
| FP16 precision | [Eleven precision and control arms, sequence and corpus comparisons](vmaf-fp16-results.json) |
| Speed investigation | [Final 24-workload comparison and corpus drift](vmaf-speed-results.json) |
| Cooperative matrices | [18 platform correctness cases](vmaf-cooperative-platform.json), [20 SpEED numerical examples](vmaf-cooperative-speed-numerics.json), [feature precision examples](vmaf-cooperative-feature-numerics.json), [integrated benchmark and sequence comparisons](vmaf-cooperative-results.json) |
| SpEED dispatch and solver parallelism | [Dispatch counts for the studied revision](vmaf-speed-dispatch-counts.json), [interleaved variants and compatibility checks](vmaf-speed-parallelism-results.json), [268 midpoint checks](vmaf-speed-solver-tree.json), [270 early QR cases](vmaf-speed-earlyqr-numerics.json) |
| VMAF 1080p, 2026-11-05 | [Five rounds for both defaults and the isolated v1 phase prototype](vmaf-1080-2026-11-05.json) |
| CAMBI headroom | [15 configurations and 510 timing runs](vmaf-v1-headroom-results.json), [zero-sum selection examples](vmaf-v1-pool-zero-sum.json) |

See the [VMAF research index](../vmaf/research/README.md) for authored
conclusions and reproduction notes.

## Earlier generated summaries

The in-memory performance tables use their original timing and resident-input
contracts. For example, older results that reused a prepared SSIMULACRA 2
reference describe a different workload from the current resident benchmark.
See [benchmark timing and caveats](../../../docs/performance.md).

- [Performance smoke](perf-smoke--summary.md)
- [In-memory latency](performance--summary.md)
- [Fixed-duration throughput](performance--throughput.md)

These still-image summaries preserve the earlier CID22 correlation and
regression runs. The former `tools/refresh_results.py` combined still-image
validation and benchmark runs and remains available in earlier Git history.
The current guides describe those workflows separately.

- [Audit validation](audit-validation--ssimulacra2-summary.md)
- [Butteraugli memory change, Butteraugli CID22](butter-memory-cid22--butteraugli-summary.md)
- [Butteraugli memory change, SSIMULACRA 2 regression](butter-memory-ssimu-gate--ssimulacra2-summary.md)
- [Cleanup regression](cleanup-final--ssimu2-cid22--ssimulacra2-summary.md)
- [Shared helper refactor](dry-refactor--corpus--ssimulacra2-summary.md)
- [Memory validation](memory-validation--ssimu2--ssimulacra2-summary.md)
- [Readability regression](readability-cid22--ssimulacra2-summary.md)
- [Source organization regression](source-readability--corpus--ssimulacra2-summary.md)
- [Unified metrics regression](unified-metrics-cid22--ssimulacra2-summary.md)
- [Butteraugli validation](validation--butteraugli-summary.md)
- [SSIMULACRA 2 validation](validation--ssimulacra2-summary.md)
- [SSIMULACRA 2 correlation gate](validation-gate--ssimulacra2-summary.md)
- [VMAF consolidation regression](vmaf-consolidation--corpus--ssimulacra2-summary.md)
- [VMAF SDR regression](vmaf-sdr-ssimu-gate--ssimulacra2-summary.md)
- [VMAF speed investigation regression](vmaf-speed-investigation--ssimulacra2-corpus--ssimulacra2-summary.md)
- [VMAF v1 optimization regression](vmaf-v1-opt2--ssimu-regression--ssimulacra2-summary.md)
