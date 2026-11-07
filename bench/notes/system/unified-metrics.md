# Unified metrics regression check

Measured on an NVIDIA GeForce RTX 5070, driver 615.78.08. Baseline
`97766b3` scores the pair with `SharedMetricWorkspace`; candidate uses
`Metrics`. Both calculate SSIMULACRA 2 and Butteraugli serially, with
a single shared arena and the same 8-bit YUV420 input. Five runs per arm
were interleaved. Values are median milliseconds per pair with min–max spread.

| Geometry | Input | Baseline ms (min–max) | Unified ms (min–max) |
| --- | --- | ---: | ---: |
| 1080 | vulkan-buffers | 4.1489 (4.1141–4.3063) | 4.1495 (4.1352–4.1897) |
| 1080 | streaming | 5.2109 (5.2016–5.2202) | 5.2178 (5.2056–5.2837) |
| 4k | vulkan-buffers | 17.3187 (17.3055–17.3280) | 17.2997 (17.2545–17.3373) |
| 4k | streaming | 21.7612 (21.6900–21.8061) | 21.7050 (21.6715–21.7775) |

For Vulkan buffers, two pre-uploaded buffers are reused with `--iterations
20 --warmup 3`; streaming uses `--iterations 8 --warmup 2`. The wall timer
covers both metrics and workspace switches. Scores and checksums matched
across all runs. These measurements show no material throughput change and
do not measure a multi-metric VMAF workload. [All rounds, stage timings and
input hashes](unified-metrics-results.json) are recorded alongside this report.
