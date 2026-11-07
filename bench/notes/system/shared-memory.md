# Shared workspace: memory and performance

This is a historical measurement. Later Butteraugli memory reductions and
updated allocations are documented in [butter-memory.md](butter-memory.md).

Measured on an NVIDIA GeForce RTX 5070, driver 615.78.08. The baseline was
`643c1c5`; the candidate was `6ee1bfc`. The GPU has host-visible device-local
memory, so these plans use the direct path without staging. Buffer allocation
sizes below are the Vulkan-reported requirements, in decimal MB, for one
worker. They exclude externally owned images and driver-managed resources.

All inputs were 8-bit YUV420; VMAF v1 was configured for that format. Each
standalone configuration used zero resident slots. `External` disables the
streaming source regions; `Streaming` retains them. The shared arena is one
allocation for both SSIMULACRA 2 and Butteraugli.

| Geometry | Input | SSIMULACRA 2 | Butteraugli | Separate sum | Shared | VMAF v0 | VMAF v1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1080p | External | 66.5 MB | 365.1 MB | 431.6 MB | 365.1 MB | 79.5 MB | 110.6 MB |
| 1080p | Streaming | 132.9 MB | 431.4 MB | 564.3 MB | 431.4 MB | 145.9 MB | 176.9 MB |
| 4K | External | 266.1 MB | 1,460.3 MB | 1,726.5 MB | 1,460.3 MB | 318.1 MB | 442.3 MB |
| 4K | Streaming | 531.6 MB | 1,725.8 MB | 2,257.3 MB | 1,725.8 MB | 583.5 MB | 707.7 MB |

A workspace owns one arena per concurrent worker; its metrics run serially.
These figures depend on configuration, format, device limits and memory
policy. Query your device instead of using this table as a budget.

## Serial metric-pair timing

Each cycle evaluates SSIMULACRA 2 and Butteraugli once. Both arms use zero
resident slots; the independent arm owns separate pipelines, and the shared
arm alternates through one workspace. Five A/B runs were interleaved for each
workload. External uses two pre-uploaded Vulkan buffers and
`--iterations 20 --warmup 3`; streaming uses `--iterations 8 --warmup 2`.
The pair rate is for *both* metric calls together. Reused pixels make external
a hot-memory workload. Wall time includes table republishing on switches.

| Geometry | Input | Independent ms (min–max) | Shared ms (min–max) |
| --- | --- | ---: | ---: |
| 1080 | resident-external | 4.3565 (4.3464–4.4196) | 4.3973 (4.3480–4.4163) |
| 1080 | streaming | 5.4652 (5.4536–5.4869) | 5.4681 (5.4353–5.5816) |
| 4k | resident-external | 18.1768 (18.0818–18.2475) | 18.1533 (18.1294–18.1938) |
| 4k | streaming | 22.6475 (22.6011–22.6725) | 22.6809 (22.6304–22.7308) |

Scores and checksums matched across every paired run. The observed timings
do not show a material throughput change; the benefit here is lower retained
GPU buffer memory. The separate baseline-vs-candidate regression runs for all
four metrics also matched scores and checksums, with no material timing
regression in these measurements. [Detailed stage medians, spreads, input
hashes and workload data](shared-memory-results.json) are retained alongside
this report. Those standalone runs used five interleaved runs per arm at
each geometry, `--resident 2 --iterations 20 --warmup 3` for resident input
and `--iterations 8 --warmup 2` for streaming. SSIMULACRA 2 prepares both
images on each resident call; Butteraugli caches its reference in that arm.

## Accuracy

The 300-pair CID22 SSIMULACRA 2 correlation gate passed. Its existing
pairwise Cloudinary differences remain (mean 0.202585, maximum 0.524941);
strict parity is not the correlation gate. On the same 300 CID22 pairs, the
shared workspace and independent pipelines matched exactly at the reported
precision for SSIMULACRA 2, Butteraugli and Butteraugli distance_p3.

Both VMAF models matched their previous VQM scores across all 4,292 CID22
still-image pairs to reported precision. V0 KRCC/SRCC/PCC against MCOS were
0.595376/0.794819/0.755195; v1 were 0.563951/0.761845/0.717795. These are
independent one-frame sequences and do not validate temporal correlation.
Forced-staging multi-frame VMAF tests separately cover retained motion state.
