# Vship and Vship-fast against this implementation

Nsight Systems profile of this metric, dispatch inventory of both GPU peers,
normalized comparison, and measured candidates. Protocol and gates are in
`AGENTS.md`. Peer sources are read-only checkouts under `.reference/`, MIT with
a non-AI clause: ideas are reimplemented here, no source text is copied, and
comments identify the peer file and line where a borrowed mechanism applies.

**Benchmark audit, 2026-11-02:** Earlier multi-worker SSIMULACRA 2 latency
timings started after the first submissions, and Butteraugli latency ignored
`--workers`. Re-run in-process worker comparisons with the corrected binary
before relying on them. Earlier Butteraugli and Vship checksums omitted warm-up
calls, and alpha stage tables could include an untimed bright-background
verification call. Historical checksum values below retain the convention of
the binary that produced them; compare per-call scores across versions.

## Vship SSIMULACRA 2: upload and GPU work

On 2026-11-02 we compared one pipeline/handler per process on an RTX 5070,
driver 615.78.08, using the same decoded 8-bit RGB `testsrc2` images (a
contrast/brightness-adjusted distortion). Five unprofiled rounds alternated
arm order within each geometry. The resident arm ran 20 measured calls after
3 warm-ups; every streaming arm ran 8 after 2. The table gives the median of
the per-run median call times, with the full range in milliseconds. Decoding,
pipeline creation, and the resident uploads are outside these times.

|  | Ours, resident | Ours, streaming | Vship CUDA, streaming | Vship Vulkan, streaming |
| --- | ---: | ---: | ---: | ---: |
| 1920x1080 | 0.3279 (0.3235–0.3363) | 0.8768 (0.8731–0.8845) | 1.6188 (1.6147–1.6234) | 2.3715 (2.3430–2.3849) |
| 3840x2160 | 1.1934 (1.1917–1.1960) | 3.3989 (3.3860–3.4140) | 6.2558 (6.2199–6.2714) | 10.2475 (10.1864–10.3802) |

Each of our resident calls prepared **both** inputs; the pair's bytes were
already on the GPU. Vship's public `Vship_ComputeHandler` and
`Vship_ComputeSSIMU2` accept host plane pointers and strides, but expose no
resident input handles or device-buffer entry point. Every Vship row therefore
includes the upload; there is no measured Vship resident arm in this table.
The implementations are not numerically identical. Scores were stable across
all five runs: ours 91.716994 / 92.437150, Vship CUDA 92.531456 / 93.417641,
and Vship Vulkan 91.576729 / 92.284729, at 1080p / 4K respectively. Checksums
were ours resident 2109.490866 / 2126.054439, ours streaming 917.169942 /
924.371495, Vship CUDA 925.314560 / 934.176407 and Vship Vulkan 915.767288 /
922.847290; they include each arm's warm-ups.

Our resident GPU timestamp medians for prepare and scale were 0.1001 / 0.1448
ms at 1080p and 0.4210 / 0.6812 ms at 4K; streaming medians were 0.0977 /
0.1447 and 0.4310 / 0.6798 ms respectively. These are our timestamps, not
Vship kernel measurements.

Separately, Nsight Systems captured Vship CUDA for 100 measured calls plus 3
warm-ups using `--sample=none -t cuda,osrt`; the public API driver and image
pair were unchanged. Gateway `kernel_summary` and `memcpy_summary` reported:

| Profiled CUDA operations, sums divided by 103 calls | 1080p | 4K |
| --- | ---: | ---: |
| Kernel durations | 1.1653 ms | 4.1787 ms |
| Kernel durations on the busier one of two streams | 0.5868 ms | 2.2276 ms |
| Kernel durations on the other stream | 0.5785 ms | 1.9511 ms |
| Host-to-device copy durations | 0.4580 ms | 2.4207 ms |
| Device-to-host copy durations | 0.00058 ms | 0.00074 ms |
| Kernel launches per call | 41 | 43 |
| Host-to-device payload per image pair | 12.44 MB | 49.77 MB |

Every kernel's launch count was a multiple of 103; the capture also contains
one 140-byte host-to-device setup copy, in addition to six image-plane copies
per call. This supports attributing the kernel rows to the 103 scoring calls,
but the capture includes warm-ups and was not restricted to a measured-only
range. A bounded gateway query grouped the kernel rows by CUDA stream to
produce the single-stream figures; it does not establish overlap. The GPU
duration figures are **sums of profiled operations**, which can overlap across
streams; they are neither critical-path elapsed time nor a measured Vship
resident throughput. They show substantial GPU kernel work as well as
transfers. They do not establish that the PCIe link is saturated, or allow a
precise compute-only speed ratio against our timestamps.
Vship Vulkan's capture reports a GPU workload per submission but does not
isolate its copy and compute command buffers, so this capture cannot provide a
Vulkan resident-equivalent or a per-stage compute time. Timing comparisons use
the unprofiled table, not profiler-instrumented calls.

The Vship arms use the pinned upstream library through the public C API only:
`python3 tools/build_references.py --only vship-cuda,vship-vulkan`. The
unprofiled commands are, for each geometry and in alternating order:

```sh
./build/ssimulacra2_bench --pair "$ORIG" "$DIST" --iterations 20 --warmup 3 --resident 2
./build/ssimulacra2_bench --pair "$ORIG" "$DIST" --iterations 8 --warmup 2
./build/vship-build-cuda/vship_bench --metric ssimu2 --pair "$ORIG" "$DIST" --iterations 8 --warmup 2
./build/vship-build-vulkan/vship_bench --metric ssimu2 --pair "$ORIG" "$DIST" --iterations 8 --warmup 2
```

The CUDA captures used `nsys profile --sample=none -f true -t cuda,osrt`
against the CUDA Vship command with `--iterations 100 --warmup 3`. The report
gateway's `report-context`, `report-fact --intent kernel_summary --max-rows 50`
and `--intent memcpy_summary --max-rows 50` produced the attribution above.
The stream breakdown used `SELECT streamId, COUNT(*), SUM("end" - start)
FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY streamId` through `report-query`.
The original captures are `/tmp/ss2-vship-gpu-audit/profiles/{1080p,4k}-{cuda,vulkan}.nsys-rep`
and the clean unprofiled logs and CSVs are under `/tmp/ss2-vship-clean-gpu-audit/`;
both directories are temporary.
The installed Nsight skill-pack doctor reported an existing `SKILL.md` content
hash mismatch; the live capture and report-fact commands completed.

The current SSIMULACRA 2 resident benchmark forces both images' preparation on
every call. Results below that used a cached reference describe a different
workload and must be remeasured before comparing with the current benchmark.
The current benchmark still reuses pixel contents and can benefit from hot GPU
memory caches.

## Handle-safety regression check

Five interleaved 1080p resident rounds on the same RTX 5070 compared baseline
`dc03459` and handle-safety candidate `597a7db`, both with the same dormant
host-view benchmark harness. Each arm used `--resident 2 --iterations 2000
--warmup 3`; iterations are the total calls across pipelines and warm-ups are
per pipeline. The inputs reused synthetic RGB pixels and every run scored
91.716994 with the expected checksum. Median fixed-count wall-clock rates and
their full ranges were:

| Pipelines | Baseline pairs/s | Candidate pairs/s |
| --- | ---: | ---: |
| 1 | 3015.0 (3010.6–3019.1) | 3016.0 (3014.9–3023.8) |
| 2 | 3756.1 (3753.5–3758.7) | 3749.9 (3749.3–3760.7) |
| 4 | 3753.6 (3750.8–3756.4) | 3752.7 (3750.6–3756.2) |

This is a regression check, not a performance-improvement claim. The raw
logs and stage tables were written under `/tmp/ss2-handle-overhead/` and are
temporary.

## Vulkan buffer input regression check

On 2026-11-02, the borrowed-buffer input added a fourth storage binding for
raw samples. Five alternating baseline/candidate runs on the RTX 5070 used a
3840x2160 RGB8 `testsrc2` pair, the full-preparation resident protocol
(`--iterations 20 --warmup 3 --resident 2`) and streaming separately
(`--iterations 8 --warmup 2`). These numbers measure the existing input paths;
they do not benchmark external buffers. Values are median ms/pair (range):

| Metric and mode | Baseline | Candidate |
| --- | ---: | ---: |
| SSIMULACRA 2 resident | 1.1906 (1.1847–1.1951) | 1.1940 (1.1866–1.1983) |
| SSIMULACRA 2 streaming | 3.3963 (3.3877–3.4016) | 3.4023 (3.3979–3.4126) |
| Butteraugli resident | 9.2118 (9.1526–9.2578) | 9.2342 (9.2300–9.4813) |
| Butteraugli streaming | 14.9027 (14.8671–14.9083) | 14.9086 (14.8960–14.9265) |

SSIMULACRA 2 resident prepare was 0.4183 (0.4170–0.4185) versus 0.4210
(0.4201–0.4213) ms; scale was 0.6815 versus 0.6816 ms. Butteraugli resident
opsin was 0.9866 (0.9857–0.9873) versus 0.9927 (0.9914–1.0944) ms. The first
candidate Butteraugli run includes an outlier. Scores and per-call checksums
matched across arms. There is a small prepare/opsin increase in this sitting;
the median end-to-end differences were below 0.3%, with overlapping ranges.

## Full-preparation resident SSIMULACRA 2 at 1080p

On 2026-11-02, the RTX 5070 with driver 615.78.08 scored an RGB24 1920x1080
`testsrc2` image against a contrast/brightness-adjusted version. The bench
pre-uploaded three images per pipeline (two copies of the original in different
slots and one distorted image), then rebuilt both pyramids for every call.
Each of five interleaved rounds timed 2000 pairs after three warm-ups per
pipeline, using elapsed wall time around the fixed-count loop. The order of
the 1/2/3/4-pipeline arms reversed in alternate rounds.

| Pipelines | Median pairs/s | Range |
| --- | ---: | ---: |
| 1 | 3017 | 3008–3028 |
| 2 | 3769 | 3763–3779 |
| 3 | 3762 | 3759–3770 |
| 4 | 3760 | 3757–3770 |

Every run scored 91.716994 with the expected checksum. One-pipeline median
prepare and scale times were 0.0995 and 0.1454 ms/call. An Nsight capture of
four measured calls and one warm-up counted 45 pipeline binds, nine per call,
confirming two prepare dispatches each time. Timing claims above come from
unprofiled runs. The same pixel contents and a small number of addresses repeat;
the measurement does not capture the cache behavior of a diverse sequence.

## Historical profiles

Reports: `resident.nsys-rep` (400 timed iterations, 20 warm-up, `--resident 2`)
and `streaming.nsys-rep` (50 timed, 5 warm-up, upload path). The original captures
were stored in `/tmp/gpu-prof/`, so the queries are written out here. Both used an
RTX 5070 (GB205, 48 SMs, driver 615.78), a 3840x2160 pair, and
`-t vulkan,osrt -s process-tree --vulkan-gpu-workload=individual`. The gateway
JSON was also in `/tmp/gpu-prof/`. The capture predates channel pruning.
[Post-pruning timing](#the-largest-lever-left-is-not-a-shader)

## Profile of this implementation

One command buffer per pair, one `vkQueueSubmit`, one fence wait. The layer
records 420 `vkBeginCommandBuffer`, 420 `vkQueueSubmit`, 420 `vkWaitForFences`
and 420 GPU workload rows for the 420 pairs, and 3780 `vkCmdBindPipeline` rows:
exactly 9.00 binds per command buffer, minimum 9 and maximum 9 across all 420.
`Dispatch()` binds once per dispatch (`src/metrics/ssimulacra2/vulkan.cc:1055-1068`), so the
trace confirms the nine dispatches per pair. The layer does not enumerate
`vkCmdDispatch`, and it reports one `vkCmdPipelineBarrier` row per command
buffer although the code issues five (`src/metrics/ssimulacra2/vulkan.cc:1236`, `:1300`,
`:1179`, `:1191`, `:1312`), so the trace's barrier count is not ours to quote.

Per-pair GPU time from the 420 workload rows: mean 2.2254 ms, median 2.1616 ms,
minimum 2.1292 ms, p95 2.6907 ms, maximum 3.3427 ms. GPU busy 934.683 ms inside
a 990.879 ms span, a 94.3% duty cycle. The 419 gaps between consecutive
submissions have median 0.1085 ms, mean 0.1341 ms, maximum 0.5981 ms, 56.196 ms
in total: 5.7% of the session is GPU idle between pairs. Median workload plus
median gap is 2.2701 ms, which is the 2.28 ms per pair the bench printed under
the profiler.

Per-dispatch allocation, from the stage timestamps of the profiled run
(prepare 0.61 ms for slots 0-1; scale 1.61 ms for slots 2-8, of which the reduce
is the residual below 0.02 ms) and each dispatch's grid at 4K. The workgroup
counts are exact (they follow from the level table and the 32x32 tile:
the dispatch and `Record`'s grid for it); the per-slot split is an allocation
by workgroup count, not a measurement, because the layer's GPU workload row
spans the whole command buffer. Prepare's grid is
`Groups((w+1)/2, 16) x Groups((h+1)/2, 16)`, so at 4K 120 x 68 workgroups, each
owning 32 x 32 base pixels (`src/metrics/ssimulacra2/vulkan.cc:1295-1296`).

| slot | dispatch | grid at 4K | workgroups | GPU ms/pair | share | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `prepare[0]` | 120x68 | 8160 | 0.305 | 13.7% | half of the measured 0.61 ms; both prepare dispatches run the same shader on the same grid, differing only in source |
| 1 | `prepare[1]` | 120x68 | 8160 | 0.305 | 13.7% | same |
| 2 | `scale[0]` | 120x68x3 | 24480 | 1.198 | 53.9% | base level; 74.9% of the scale stage's workgroups |
| 3 | `scale[1]` | 60x34x3 | 6120 | 0.300 | 13.5% | |
| 4 | `scale[2]` | 30x17x3 | 1530 | 0.0749 | 3.4% | |
| 5 | `scale[3]` | 15x9x3 | 405 | 0.0198 | 0.9% | |
| 6 | `scale[4]` | 8x5x3 | 120 | 0.0059 | 0.3% | thin grid: 120 workgroups on 48 SMs |
| 7 | `scale[5]` | 4x3x3 | 36 | 0.0018 | 0.1% | thinner still, so its real cost is above this share |
| 8 | `reduce` | 18x6 | 108 | <0.02 | <0.9% | 18 norms per level, 10 897 partials per norm; 108 workgroups of 256 |

Host side per pair, from `graphics_api_summary`: `vkWaitForFences` 2.2736 ms and
is the wall clock; `vkQueueSubmit` 4.559 us, `vkBeginCommandBuffer` 1.563 us,
`vkCmdPipelineBarrier` 0.621 us, `vkEndCommandBuffer` 0.429 us, and nine
`vkCmdBindPipeline` rows at 0.126 us, about 8.3 us of traced host recording per
pair, 0.4% of the pair. Recording is not the cost; the 0.11 ms of idle per pair
is the submit-and-wait turnaround, and the wait dominates everything the host
does.

Streaming (`--iterations 50 --warmup 5`): 55 command buffers, GPU workload mean
2.2649 ms, median 2.1692 ms, 124.567 ms busy in a 250.283 ms span, a 49.8% duty
cycle. The 54 inter-submission gaps have median 2.3187 ms and mean 2.3281 ms,
125.716 ms idle in total, which is the PCIe write of both images: the untimed
stage the bench reported in the same run was 2.25 ms per iteration. The upload
path is half idle and the GPU work itself is the same as in the resident run.

Where the resident time goes, by size: one pair measures 0.12 ms at 512x512,
0.22 at 1024x1024, 0.64 at 2048x2048 and 1.21 at 3840x2160, i.e. about 0.08 ms
of per-pair cost (nine dispatches, a command buffer, a fence) plus roughly
0.13 ms per megapixel. The per-megapixel rate is flat from two megapixels up --
0.210, 0.153, 0.146 -- which is what a per-pixel work bound looks like: a
bandwidth roofline would turn up at the top of that range, and it does not.

### Occupancy probe

GPU counters are blocked, so occupancy cannot be read; it can be stepped. Pad the
workgroup's shared block by a compile-time constant that the shader still
touches, and the driver drops one resident block each time the SM's budget is
crossed, which shows up as a step in the stage time (4K, pad in floats of 4 B,
median of three runs at `--iterations 20 --warmup 3 --resident 2`):

| block | pair | scale |
| --- | --- | --- |
| 21 120 B | 835.8 pairs/s | 0.79 ms |
| 25 216 B | 837.2 | 0.75 |
| 29 312 B | 730.0 | 0.93 |
| 33 408 B | 729.0 | 0.98 |
| 37 504 B | 593.0 | 1.32 |
| 41 600 B | 588.0 | 1.26 |

Two steps: between 25 216 and 29 312 B (-13%) and between 33 408 and 37 504 B
(-19%), flat elsewhere. Each step is one resident block, so the block count at
21 120 B is the number of steps still below the budget: at least three. Timing
alone cannot separate three blocks with about 68 KB of shared per SM from five
with about 105 KB, which is why the counters below were needed: they show 66% of
the SM's warp capacity filled with register allocation allowing 94%, so shared
memory caps it and the 25% that the 48 KB per-workgroup guarantee suggested was
wrong twice over.

Two consequences. The first step is inside 8 KB of where the block is, so any
candidate that grows it past about 25 KB pays 13-19% before it earns anything --
which is what the double-buffered `mid` of rank 4 (26 752 B) does, and why it
was ranked as a likely loss. And the earlier reading of the 43.6 KB fusion -- two
workgroups per SM at 21 120 B, one at 43.6 KB -- undercounted the small end by at
least one block and puts 43.6 KB one step too high: it is past both measured
steps, so it loses two blocks, 26-38% of the stage, before its own extra traffic
is counted. The rejection stands; the attribution is now the two steps plus that
traffic.

### Measured utilization

The follow-up capture had GPU counter access with the NVIDIA driver's
`NVreg_RestrictProfilingToAdminUsers` set to `0`. Nsight offered `gb20x`
(25 metrics) and `gb20x-top` (280 metrics) on the GB205. These sets had no
direct stall, barrier or eligible-warp counters. The `gb20x-top` results are
medians of 3 580 samples at 4K and 783 at 1080p, using `--resident 2` and
windowing to samples with `Sync Compute in Flight` at 100%.

| counter | 4K | 1080p |
| --- | --- | --- |
| SM Issue Stage | 48% | 57% |
| SM Throughput | 48% | 57% |
| L1TEX (LSU data stage) | 40% | 45% |
| VRAM total bandwidth | 34% | 12% |
| SM FMA pipe | 23% | 24% |
| SM ALU pipe | 12% | 20% |
| SM XU pipe (`pow`) | 2% | 8% |

Occupancy, from the same window: `Sync CS SM Warps` reports 63 warps on average
and 66% of capacity at 4K (65% at 1080p, p95 98%), and 63 / 0.66 puts that
capacity at 96 warp slots where Ada's SM has 64, which is the one number here
that would be worth confirming on another device. `CS Register Allocation` is
94%, so registers are not the cap. `Active Thread Groups in SM` reports 8 CTAs,
which cannot be concurrent -- eight blocks of 21 120 B is 169 KB against the step
map's three to five, and eight blocks of eight warps would fill those slots -- so
this file does not use that column. `Predicated-On Active Threads Per Warp`
decodes to 31 of 32 lanes: 97% lane utilisation, so the idle issue slots are
latency and not divergence.

Four readings. No unit is above 45%: the issue stage at 48-57% is the closest
thing to a limiter, and it is at half capacity, so the stage is neither FMA
bound, nor DRAM bound, nor bound by any one pipe. The XU pipe at 2% says the
`pow` calls cost nothing once they are latency-hidden -- which is why
Vship-fast's Newton cube root lost 9% (it moved work off an idle pipe onto a
busy one) and why stubbing the transfer function out was worth 2%. VRAM at 34%
of peak at 4K, 12% at 1080p, is the same message for bytes, and why
`--half-planes` is worth 7%. What is left is eligibility: 48% issue at 66%
warp occupancy means a third of the warp slots are empty and, with two thirds
of them filled, half the issue slots still have nothing eligible to run. The
barrier chain is the likely cause, and the double-buffered `mid` that would
attack it costs a block (21 120 -> 26 752 B lands on the first step), which is
the trade the 43.6 KB variant already lost. Every 100 us sample in either
capture had some issue activity. The capture could not identify individual
stalls. A later Nsight Compute attempt reported "No kernels were profiled"
for the Vulkan workload.

    Occupancy therefore looks like the lever,
    and the tile table says it is not
    : forcing the 16 x 32 variant(13 440 B, so more resident blocks) is 7.6 %
            *slower * on the pair and
        16 % slower on the stage(scale 0.70->0.81 ms),
    because that tile reads 2.4 halo values per output against 1.9 and
        its ten barriers per quantity cover half as many outputs.Occupancy that
            helps has to come with the same instruction stream,
    and nothing measured does that.

        Profiler overhead : the GPU -
        side numbers are unchanged(
            workload median 2.1616 ms against a 0.60 + 1.56 -
            1.64 ms unprofiled stage split
        ),
    and the per - pair wall time moves from 2.24 -
        2.25 ms unprofiled to 2.28 ms under the layer,
    1.3 - 1.8 %,
    which is the extra traced calls plus fence turnaround
            .Timing claims below come from unprofiled runs.

        ##What the peers dispatch per pair

        | implementation,
    backend | dispatches per 4K pair | barriers | reduction | readback | | -- -|
        -- -| -- -| -- -| -- -| | this one,
    Vulkan | 9 : 2 prepare, 6 scale,
    1 reduce | 5 on the resizable - BAR path(
                                        host to compute, after prepare,
                                        after the six scales, after the reduce,
                                        publish
                                    );
6 when the arena is not host - visible, where the upload needs two | 1 dispatch,
    18 norms x 6 levels | 108 floats | | Vship,
    Vulkan | 37 : 10 downsample(5 per image, strictly serial),
    11 planescale_map(18 minus 7 pruned at 4K), 16 sumreduce; plus one converter dispatch per image | 37, one per dispatch, `ALL_COMMANDS` on both sides | 16 sumreduce launches of 1024 threads and 24 KiB shared | 432 B copy of the 108 norms, score on the host |
| Vship, HIP | 37 on two streams: 10, 11, 16 | none (stream ordering), 4 events per pair | 16 sumreduce launches | 432 B plus a stream sync |
| Vship-fast, Vulkan | 18: 8 conversion, 2 yuvToPyramid, 8 score (4 blurRows, 4 blurColumnsAndScore) | 8, one per blur pass, plus the sums barrier | none: the host sums 6 floats per workgroup | the whole per-workgroup block |
| Vship-fast, HIP | 55: 10, 18, 27 (upstream's backend, unchanged) | stream ordering only | 27 sumreduce launches | 432 B plus a stream sync |

Structure: upstream Vship builds the pyramid outside the scoring pass
(`.reference/Vship/src/Vulkan/ssimu2/main.hpp:56-71`), ten dispatches deep, each
followed by a device-wide barrier, and writes the XYB form of every level back
over it (`shaders/downsampleSSIMU2.slang:44-70`) because the scoring dispatch is
a separate pass. Its `planescale_map` is ours in spirit -- four 17-tap blurs in
shared memory, both error maps and six norm partials per block
(`shaders/scoreSSIMU2.slang:116-226`) -- and it prunes whole (plane, scale)
pairs the same way (`score.hpp:122-131`). Vship-fast replaces the pyramid with
one dispatch per image (`shaders/downsampleSSIMU2.slang:198`, 32x32 base pixels,
12 288 B shared, our prepare's shape) and the scoring pass with libjxl's
recursive filter split across threads and signals
(`shaders/scoreSSIMU2.slang:131,258`), then drops the reduce dispatch and sums
on the host (`score.hpp:207-245`). Neither peer has a resident-image path: they
upload per frame, on a transfer-family queue in the fork's case
(`main.hpp:137-151`).

## Normalized throughput

| implementation | GPU, driver | clip, threads | ms per pair | ms per MPx per pair | pairs/s |
| --- | --- | --- | --- | --- | --- |
| this one, resident | RTX 5070, 615.78 | 3840x2160, 1 pair | 1.21 | 0.146 | 823 |
| this one, resident, before the pruning | RTX 5070, 615.78 | same | 2.25 | 0.271 | 445 |
| this one, streaming | RTX 5070, 615.78 | same | 3.6 | 0.434 | 278 |
| Vship-fast, Vulkan | RTX 3070, 616.92, Windows 11 | HoneyBee 4K, 1 | 13.11 | 1.580 | 76.3 |
| Vship-fast, Vulkan | same | 3 | 9.03 | 1.088 | 110.8 |
| Vship, Vulkan | same | 1 | 13.46 | 1.623 | 74.3 |
| Vship, Vulkan | same | 3 | 13.75 | 1.658 | 72.7 |

The peer rows are the forks' own tables (`.reference/Vship-fast/README.md:20-30`),
converted from frames per second: 8.2944 Mpx per 4K frame, and the caveat is
theirs -- those numbers include VapourSynth frame handling, YUV 4:2:0 input, the
upload and the readback, where ours are one resident pair with decoding
excluded, on a different card (a 5070 is roughly 1.4-1.6x a 3070 on
float-heavy compute; that is an estimate, not a measurement). The HIP backends
publish no frames per second in either repository, only whole-run seconds
against other tools, so they get no normalized row. Vship-fast's own table says
what the fork's blur bought: SSIMULACRA2 is 1.03x upstream at 4K on one thread
and 1.52x at three, because matching libjxl's recursive filter cost 2.8-4.4x
before their concurrency work (`README.md:31-32`).

## Where each is faster

### They are

- **Pruning what the score cannot see.** The peer drops a map when that map's
  1-norm and 4-norm weights are both at most 0.01
  (`.reference/Vship/src/Vulkan/ssimu2/score.hpp:4` is the threshold,
  `:125-131` decides the three maps separately), leaves the whole
  (plane, scale) dispatch out only when all three go (`:181`), and suppresses
  the downsample store on the same conjunction (`main.hpp:32-42`). At 4K that
  removes seven of the 18 channel-level pairs; our weight table
  (`src/metrics/ssimulacra2/score.cc`) gives the same seven at six levels. Our workgroup
  produces all six norms of a pair at once, so the pair is the unit we can
  drop. Adopted, rank 1 below: -57% on the scale stage.
- **One blur instead of two for the variance sum.** `s11 + s22` is recovered as
  `blur((a+b)^2) - 2*blur(a*b)` (`shaders/scoreSSIMU2.slang:145-166`), which
  holds for any linear blur and therefore applies to our 13-tap kernel without
  touching its arithmetic. Rank 2 below.
- **Recording the dispatch graph once.** Vship records all 37 dispatches at
  init and resubmits the buffer (`main.hpp:151-159`, `176-186`), and carries the
  buffer addresses in the push block so there is no descriptor set at all. We
  re-record 9 dispatches and 5 barriers per pair (`src/metrics/ssimulacra2/vulkan.cc:1294`).
  Rank 6.
- **A shared-memory pitch that does not collide.** Vship-fast pads the input
  and output pitches to 27 and 17 floats "so that lanes reading one column each
  of their rows do not conflict" (`shaders/scoreSSIMU2.slang:80-83`). Our tile
  pitch is 44 floats, so the four-row window LoadTile feeds four ways into
  eight banks. Rank 3.
- **Packing the coarse levels into one dispatch.** The fork scores scale 0 on
  its own and levels 1-5 together in `grid.y`. (`score.hpp:190-204`, early
  return at `scoreSSIMU2.slang:136`). Rank 5; our own packed-grid attempt
  measured 2.30 ms against 1.63 (`src/metrics/ssimulacra2/vulkan.cc:1152-1158`), so this
  is only worth it for levels 1-5, not for the base level.
- **Loading the next chunk before the barrier.** The fork issues the next
  chunk's global loads behind the barrier and blurs the current one
  (`scoreSSIMU2.slang:150-170`). Ranks 4.

### We are

- **Dispatch and barrier economy.** 9 dispatches per pair against 37
  (Vship, either backend), 18 (Vship-fast Vulkan) and 55 (Vship-fast HIP), and
  five barriers against 37: our six scale dispatches are independent -- each
  reads only what prepare wrote -- so no barrier separates them
  (`src/metrics/ssimulacra2/vulkan.cc:1151-1179`). The peers cannot overlap the same work
  because their barrier stage masks are `ALL_COMMANDS`
  (`.reference/Vship/src/Vulkan/vulkanHelper.hpp:750-752`).
- **Shared-load reuse.** Both of our passes load 16 values to feed 52 taps
  (`src/metrics/ssimulacra2/shaders/scale.comp:141-200` and `:208-245`), four loads per output per
  quantity; the peer's one-output-per-thread tile loads its own 17 taps in each
  pass, 34 per output per quantity (`shaders/scoreSSIMU2.slang:54-56`,
  `:86-88`). That is why our scale stage is 0.74 ms at 0.33 GB of traffic where
  theirs is shared-instruction bound.
- **The pyramid.** Two dispatches and two barriers per pair, with the linear
  plane never reaching VRAM (`src/metrics/ssimulacra2/shaders/prepare.comp:12-15`), against ten
  serialized downsample dispatches, ten `ALL_COMMANDS` barriers and an in-place
  XYB write-back of every level.
- **Reduction and readback.** One dispatch reduces all six levels
  (`src/metrics/ssimulacra2/shaders/reduce.comp`) and the host reads 108 floats; the peers launch
  16 or 27 reductions and either copy 432 B or hand the whole per-workgroup
  block to the host. Our partial block is 0.78 MB at 4K, so their host-sum
  design would cost us a readback 7 000 times larger than our reduce dispatch.
- **Geometry in push constants.** Our taps and level geometry ride the 120-byte
  push block (`src/metrics/ssimulacra2/shaders/ssimu2_common.glsl:14-41`), where every peer block starts
  with a dependent device-memory load of its kernel table
  (`Vship/src/Vulkan/ssimu2/shaders/scoreSSIMU2.slang:61-68`).
- **No per-pair allocation or host synchronisation.** One arena, one fence per
  submission, resident slots; the HIP peer allocates and frees twice per pair
  and synchronises the host inside every pair
  (`Vship/src/HIP/ssimu2/main.hpp:137-141`, `score.hpp:449-450`).
- **Throughput.** 0.146 ms per megapixel per pair against 1.088-1.658 for the
  peers' published 4K numbers, with the caveats in the table above.

## Candidates

Ranked by expected stage delta; ties between equal deltas would go to the
row that leaves the arithmetic alone. Only rank 1 was implemented.

| rank | id | mechanism | peer evidence | our anchor | expected delta | changes arithmetic | patch sketch | gate result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | prune-channels | skip the (channel, level) pairs whose six weights are all <= 0.01: no plane store, no scale workgroup, norm forced to zero | `Vship/src/Vulkan/ssimu2/score.hpp:4,125-131` (per-map decision), `:181` (pair skipped when all three maps go); `main.hpp:32-42` (store suppression) | weight table `src/metrics/ssimulacra2/score.cc`; `scale.comp` `main`; `prepare.comp` `StoreXyb`; `reduce.comp` `main`; `metrics/ssimulacra2/vulkan.cc` `ScaleStage`/`Record` | scale 1.72 -> 0.74 ms (-57%), prepare 0.61 -> 0.43, pair 2.25 -> 1.21 ms (-46%) | yes | implemented: `uint channel_mask` after `groups_x` in `common.glsl` and `PushParams` (116 -> 120 B); `Ssimulacra2ChannelPruned(channel, level, levels_)` in `metric.cc` builds a per-geometry mask; `scale.comp` returns on a cleared bit, `reduce.comp` writes 0 for it, `StoreXyb` masks its three stores | see *Deviations*: all gates pass |
| 2 | one-blur-variance | recover `s11+s22` as `blur((a+b)^2) - 2*blur(a*b)`, five blurs down to four | `Vship/src/Vulkan/ssimu2/shaders/scoreSSIMU2.slang:145-166` | `scale.comp` `QuantitySource`, `AccumulateMaps`, `main`'s five `FilterQuantity` calls | measured: scale 0.69 -> 0.65 ms (-5.8%), pair 1.19 -> 1.16 ms (-2.5%) | yes | `QuantitySource`: add `q == 5u` returning `(a+b)^2`; `AccumulateMaps` takes `s11_s22`; `main` drops the `a^2` and `b^2` filters and passes `b2 - 2.0f*b3` | rejected: the small win is not worth losing exact identity (99.846 instead of 100 on identical input) and a 1e-9 arithmetic pin; see below |
| 3 | bank-pad | pad the shared row pitch so a warp's four tile rows land on four different bank residues | `Vship-fast/src/Vulkan/ssimu2/shaders/scoreSSIMU2.slang:80-83` | `scale.comp` `kSrcW`/`tile_index`/`LoadTile`; `kScaleVariants` in `metrics/ssimulacra2/vulkan.cc` | a few percent of the scale stage, up to more if the window loads vectorize; 21 472 B stays under the first occupancy step (25 216 B is flat, 29 312 B is not) | no | `kSrcW = kOutW + 2*kRadius + 1` with the pad column skipped in `LoadTile`; `kScaleVariants[0].shared_bytes` 21120 -> 21472 and the 16x32 entry recomputed from its own shape | rejected: scale 0.72 -> 0.79 ms. The pitch of 44 is a multiple of four, so the four floats of a window are 16-byte aligned and the compiler merges them into one 128-bit shared load; an odd pitch breaks that and four 32-bit loads cost more than the conflicts they remove |
| 4 | prologue-overlap | two mid buffers, so the next quantity's horizontal pass runs while the previous vertical pass reads | `Vship-fast/src/Vulkan/ssimu2/shaders/scoreSSIMU2.slang:150-170` | `scale.comp` `FilterQuantity` (10 barriers per dispatch) | low single-digit percent of the scale stage, against a certain loss: 26 752 B lands inside the 25 216-29 312 B occupancy step, so it drops one resident block for 13-19% | no | split into `FilterHorizontal(q, base, tid)` and `FilterVertical(base, tid)` with no barrier inside either; alternate `kMidOffset` and `kMidOffset2`; `shared_bytes` 21120 -> 26752 | not implemented |
| 5 | pack-small-levels | one dispatch for levels 1-5 behind level 0, workgroups outside a level's grid returning | `Vship-fast/src/Vulkan/ssimu2/score.hpp:190-204`, `scoreSSIMU2.slang:136` | `metrics/ssimulacra2/vulkan.cc` `ScaleStage`, level table in `common.glsl` | 0.3-0.8% of the pair | no | `PushParams += first_level`; `ScaleStage` dispatches level 0 as today then `Dispatch(scale_, params, Groups(sizes_[1].width, cols), levels_-1, 3)`; `scale.comp` reads each level's geometry from the level table and returns when `gl_WorkGroupID.x/y` exceeds its grid | not implemented |
| 6 | prerecord-graph | record the pair's dispatches once and resubmit the command buffer | `Vship/src/Vulkan/ssimu2/main.hpp:151-159,176-186` | `metrics/ssimulacra2/vulkan.cc` `Record` | <=2% of the pair, inside the noise floor | no | `Record` returns before its first barrier when a recorded key (both resident slot ids, background, geometry) matches; `.h` gains `recorded_key_`/`has_recorded_`, invalidated by `Upload`/`Release` | not implemented |
| 7 | vendor-prepare-shape | ship two prepare workgroup shapes and pick one by PCI vendor | `Vship-fast/src/Vulkan/ssimu2/shaders/downsampleSSIMU2.slang:135-139`, `downsample.hpp:25-26` | `prepare.comp` `local_size` and the register-resident level 1 | 0.5-1.6% of the pair, unknown for our kernel | no | `open choice: how to keep prepare's register-resident level 1 when an invocation owns 4x4 base pixels instead of 2x2, and which duplicate pipeline needs a new CMakeLists entry` | not implemented |
| 8 | transfer-queue-upload | upload on the transfer family in its own command buffer, joined by a semaphore | `Vship-fast/src/Vulkan/ssimu2/main.hpp:67-70,137-151`; `vulkanHelper.hpp:579-581,902` | `metrics/ssimulacra2/vulkan.cc` `Record`'s streaming path | 0 on the resident arm; up to the whole 2.3 ms upload on the streaming arm | no | `.h`: add `upload_command_buffer_`, `upload_done_`, `RecordUpload(...)`; `.cc`: move the four `CopyUpload` calls out of `Record`, submit twice with `VkSubmitInfo2` | not implemented: needs a new queue family and semaphore |
| 9 | exact-divide | correctly rounded `divide()` and un-fused products | `Vship-fast/src/Vulkan/ssimu2/shaders/scoreSSIMU2.slang:19-40,340-355` | `scale.comp` `AccumulateMaps`'s two divisions; `prepare.comp` `cbrt_` | 0 speed, slightly negative | yes | add `divide(a, b)` = reciprocal plus one Newton step; replace the two `/` expressions in `AccumulateMaps` | not implemented: buys nothing, and our accuracy target is the libjxl reference, not a CPU tier |
| 10 | host-sum | drop the reduce dispatch and sum the partials on the host | `Vship-fast/src/Vulkan/ssimu2/score.hpp:159-245` | `reduce.comp` and its caller | negative: 0.78 MB of readback on the critical path | yes | delete the reduce dispatch and widen the readback from 108 floats to the partial block | not implemented: the fork can afford it at 60 workgroups per column group, we have 32 691 |
| 11 | recursive-blur | replace the 13-tap sampled Gaussian with libjxl's recursive one | `Vship-fast/src/Vulkan/ssimu2/shaders/scoreSSIMU2.slang:131-232,258-360`; `gaussianblur.hpp:34-136` | `scale.comp` `FilterQuantity`, the tap order contract | changes the metric; their result is also 1.088 ms/MPx/pair against our 0.146 | yes | `open choice: whether the metric may move to libjxl's recursive blur at all -- it re-baselines every golden and the CID22 correlation, and needs a different operator (three-term IIR) plus a coefficient table rather than an edit to FilterQuantity` | not implemented |
| 12 | seventeen-tap | match the peer's blur: 17 taps, sigma 1.5, truncation-renormalized borders | `Vship/src/Vulkan/ssimu2/shaders/scoreSSIMU2.slang:53-57,85-89`; `gaussianblur.hpp:29-40` | `scale.comp` `LoadTile` (zero-padded halo) and the fixed 13 taps | changes the metric; more taps, wider halo | yes | `open choice: whether the blur definition may change to 17 taps with a renormalized border -- kRadius 8 makes the tile 21 120 B into 30 976 B and loses the second resident workgroup, and every golden must be re-derived` | not implemented |

Ranks 7, 11 and 12 stop at `open choice:`; rank 8 needs a new queue family and
rank 9 buys nothing, so none of them would have been taken by the selection rule
before rank 1 either.

### The largest lever left is not a shader

This section records the measurements before the 2026-11-02 reference-cache
and mixed-input work below; its figures are historical, not a current baseline.

Post-pruning resident pairs took 1.19–1.20 ms at 4K (835–840 pairs/s), with
0.42 ms for prepare, 0.74 ms for scale and 0.01–0.05 ms untimed. At 1080p they
took 0.33 ms (about 3000 pairs/s), with 0.10 ms for prepare, 0.15 ms for scale
and 0.08 ms untimed. The original benchmark protocol also recorded 2.26 and
2.63 ms per resident 4K pair for the same build in sittings minutes apart.

Two pairs in flight would hide what the shader cannot touch. The size sweep
above puts 0.08 ms per pair in dispatch and fence cost, and the profile puts
0.109 ms of median GPU idle between submissions: 5.7% of a 4K pair and 25% of a
1080p one, all of it host turnaround rather than GPU work. Vship-fast's own
table shows what concurrency buys it -- 76.3 to 110.8 fps at 4K by running
three frames in flight (`Vship-fast/README.md:20-30`) -- and it can do that
because each thread owns its upload buffer and its temporaries.

That lever is bounded, and the bound is the point -- and the bound was measured
after the split submission API existed, so this is a result and not an estimate.
Interleaved wall clock, 2000 resident 4K pairs per run, three pairs of arms: one
pipeline 667, 670, 671 pairs/s, two pipelines in one process 695, 694, 694, so
+3.5% -- which is the same figure the in-loop measurement gives (+3.2%, 836.6 to
863.3). Two separate *processes* give none at all: 6.00 s for 4000 pairs against
5.89 s for one, each process dropping to 363 pairs/s. Two queues on one device were worse still:
789.6 pairs/s against 835.1 for one queue. So the GPU has no second stream's
worth of headroom: the +6% is draining the dispatches and the fence turnaround,
which is what this section predicted, and the "two pairs would double it" reading
that a short two-process run suggests is an artifact -- with 200 iterations each
process finishes before the other competes. The traced 0.109 ms gap
flatters it, because the Vulkan layer's own host overhead is inside that number;
the unprofiled stage shares say what is really recoverable: at 4K the untimed
row is 0.01-0.05 ms of a 1.20 ms pair, 1-4%, so quote about 3%, and at 1080p it
is 0.08 of 0.33 ms, a quarter. Two pairs in flight reaches the 4K bound; three
adds memory and nothing else.

The streaming arm is where the same reasoning is worth a multiple: it runs at
49.8% GPU duty because 2.25 ms of host upload sits in front of 1.29 ms of
compute, so overlapping the next pair's upload with the current pair's compute
would take the pair to the PCIe wall, 3.5 -> ~2.3 ms (about 1.5x). The arena
holds exactly two source regions (`source[2]`, metrics/ssimulacra2/vulkan.cc:462-467); the
eight slots are the staging buffer's, counted by `kUploadSlots * base_pixels_`,
so a second pair in flight needs a second source pair: 265 MB at 4K, 66 MB at
1080p. The descriptor block needs no duplicate, because WriteSourceDesc runs
inside Record(), after the wait.

`open choice: how to overlap two pairs without doubling the writable state --
the XYB pyramid is one region per resident image, so two calls on the same pair
would race on it. On the resident path the writable side is the 265 MB plane set
plus the 0.8 MB partial block, about 1.07 GB at 4K rather than a doubled arena,
since the resident slots are shared read-only; on the streaming path it is a
second source pair, 265 MB at 4K. Options are a second command buffer, fence and
norm block plus a pipeline of two in `Run`/`Submit` (metrics/ssimulacra2/vulkan.cc) with a
`Flush()` on the API, a `Submit`/`Finish` split that hides only the host upload
(the streaming case, no second pyramid), a same-image-only overlap for the
bench, or per-call plane regions; the API shape (does Compute still block?) and
the memory budget are the decisions, so this is not a statement level patch.`

### Measured after the first pass

Five further ideas were measured after that ranking; one was adopted.

- **Adopted: the dead sign branch in the cube root.** `cbrt_` carried
  `v < 0.0 ? -pow(-v, 1.0/3.0) : pow(v, 1.0/3.0)` while its one caller clamps to
  at least zero. The branch was not free: `glslc -O -S --target-env=vulkan1.1`
  emits 61 cube-root instructions and 148 selects for the module with it, and
  55 and 142 without. Removing it takes a resident 4K pair from 824.8 to 835.1
  pairs/s (median of seven
  interleaved runs of 30 iterations, one 730 pairs/s outlier in the candidate
  arm) and prepare from 0.43 to 0.42 ms, with the five-pair capture
  bit-identical.
- **Rejected: Vship-fast's `cubeRootAndAdd`.** The fork replaces the cube root
  with an estimate from the exponent bits refined by four Newton steps
  (`shaders/makeXYBSSIMU2.slang:30-46`). Reimplemented here it is *slower*:
  prepare 0.43 -> 0.47 ms (+9%), pair 1.21 -> 1.25 ms, because three dependent
  Newton chains on the FP32 pipe cost more than the two SFU instructions of
  `pow`, and the counters say why: `SM XU Pipe` sits at 2% while `SM FMA Pipe`
  is at 23%, so the change moved work off an idle pipe onto a busy one. The SFU work is latency-hidden, which is the same
  reason the README already records for adding one Newton step.
- **Rejected: the bank-conflict pitch pad** (rank 3). Implemented exactly as its
  sketch describes (pitch 45, pad column skipped, 21 472 B) it costs the scale
  stage 10%: 0.72 -> 0.79 ms, pair 1.21 -> 1.26 ms, at identical scores. The
  predicted four-way conflict is real, but at a pitch of 44 floats a window's
  four values are 16-byte aligned, so the compiler emits one 128-bit shared load
  which already takes four bank cycles; the odd pitch splits it into four
  32-bit loads and the instruction count outweighs the conflict.
- **Rejected: the one-blur variance** (rank 2), implemented in full. It is
  real but small -- scale 0.69 -> 0.65 ms (-5.8%), resident pair 1.19 -> 1.16 ms
  (-2.5%) -- because `(a+b)^2` reads both tiles, so the horizontal pass's shared
  loads are unchanged (40 -> 36 per output) and only the vertical pass, the mid
  store and two barriers go away. What it costs is the exactness the five-blur
  form has: `denom_s = (s11 - mu1^2) + (s22 - mu2^2)` is bitwise the same
  expression as `num_s = 2*(s12 - mu1*mu2)` when the images are equal, so `d`
  cancels to zero and identical input scores exactly 100; recovering the sum
  makes the two different expressions, the compiler contracts each differently,
  and `d` lands at 1e-7 -- 99.846 instead of 100 on the identity pair, up to
  0.12 at 4K, and `tests/input/input_format_chroma_test.cc:253` (a 1e-9 pin) fails by 9.5e-5.
  Both associations of the denominator (`- mu1^2 - mu2^2` and
  `- (mu1^2 + mu2^2)`) were tried; neither restores the cancellation, because
  the fused multiply-adds of the two shapes round differently.
- **Rejected: the smaller tile for occupancy.** Forcing `kScaleVariants[1]`
  (16 x 32, 13 440 B) to buy more resident blocks costs 7.6% on the pair and 16%
  on the scale stage, for the halo and barrier reasons above. The historical
  benchmark notes also recorded a 7% scale-stage increase for the 32 x 64 tile.
- **Rejected by ablation: the vertical pass as vec4 across columns.** Making the
  vertical pass four times cheaper was the one restructure with an instruction
  argument left. Cutting it to a single tap -- 12 of its 16 shared loads and 48
  of its 52 FMAs, with the five quantities still distinct so the compiler cannot
  deduplicate them -- takes the stage from 0.73 to 0.67 ms, 8%. So the entire
  vertical filter stream is worth about a tenth of the stage and the 32 x 64 tile
  such a restructure needs (38 016 B, past the second occupancy step) could never
  pay for itself. The counters move with that ablation -- `SM FMA Pipe` 23 -> 19%,
  issue stage 48 -> 43%, L1TEX 40 -> 36%, pair 824.8 -> 865 pairs/s -- which is
  what makes it a measurement and not a reshuffle. With the one-blur result (one
  filter of five = 5.8%) the two agree that the filter arithmetic is a quarter to
  a third of the stage; tile loads, barriers, the maps, the norm ladder and the
  latency behind them are the rest.

  A variant that replaced four of the five quantities with a repeat of the first
  priced nothing: 2% on the pair and `SM FMA Pipe` 23 -> 22%, because `q` only
  selects *which* value a pass filters, so five passes with different `q` run the
  same loads and the same taps and the substitution removes one multiply per
  source value. Recorded so the next ablation does not repeat it.
- **Ceiling measured, not pursued: an sRGB transfer LUT.** Stubbing the whole
  transfer function out of `srgb_eotf` -- three `pow` per pixel per image --
  moves prepare from 0.42 to 0.41 ms at 4K. A 256-entry table for byte input
  would therefore buy at most about 2% of prepare and less than 1% of the pair,
  which does not pay for a new arena region, a second decode path and the
  arithmetic risk.

## Deviations
Gateway: `doctor` passes all seven CLI checks and the report dependencies
(`duckdb 2.0.0`, `pandas 3.1.0`, `pyarrow 26.0.0` on the bundled Python 3.12.12,
installed with `pip install --user` because `/opt` is not writable), but fails
the bundled skill pack on `content hash mismatch: SKILL.md`, so `search-docs`
and `check-claims` refuse to run. Substitution: the numbers in the profile
section above were re-derived mechanically from the saved gateway JSON in
`/tmp/gpu-prof/evidence/` and all 36 matched. Commands that did run:
`report-context`, `report-fact` with `report_inventory`, `activity_summary`,
`graphics_api_summary`, `graphics_api_distribution`, `graphics_api_timeline`,
and `report-query`.

`kernel_summary`, `timeline_summary` and `kernel_variance` fail with
`CUPTI_ACTIVITY_KIND_KERNEL is not present in this report`: the capture has no
CUDA activity, only Vulkan. Their Vulkan equivalents were used instead, and no
`nsys stats` fallback was needed.

Per-dispatch GPU durations are not separable in this capture.
`--vulkan-gpu-workload=individual` produces one GPU workload row per command
buffer (420 rows for 420 pairs), and the layer records no `vkCmdDispatch` rows,
so the dispatch count comes from `vkCmdBindPipeline` (one per `Dispatch()`,
exactly 9.00 per command buffer) and the per-slot split in section 1 is an
allocation from the stage timestamps and each dispatch's grid, labelled as such
in its notes column.

`-f` takes a value in this nsys version, so the captures used `-f true`.

No privilege was raised: `kernel.perf_event_paranoid` is `2`, so the report has
no CPU sampling (`has_cpu_sampling: false`), and
`/sys/module/nvidia/parameters/NVreg_RestrictProfilingToAdminUsers` is `1`, so
there are no GPU counters and no `gpu_utilization` fact. Occupancy is therefore
stepped rather than sampled (see *Occupancy probe* above), and what that probe
gives is a step map and a range, not a utilization percentage: 3-4 resident
blocks at 21 120 B, one block lost at 26-29 KB and another at 34-38 KB.

The profiled checksums differ from the unprofiled 1975.890968 only through the
iteration count: the bench sums every call's score, warm-up included, so the
resident capture printed 36081.487239 = 420 x 85.908303 and the streaming capture
4724.956662 = 55 x 85.908303. The per-call score 85.908303 and the five-pair
capture are unchanged, which is the invariant the check was for.

### Channel-pruning validation

Phase 6 took rank 1, `prune-channels`, and implemented its patch sketch as
written except for the mask itself. The sketch carried a fixed six-level table
(`{0b010, 0b011, 0b011, 0b111, 0b111, 0b000}`), which is wrong for any smaller
pyramid: `Msssim::Score` indexes `kScaleWeights` as one flat run of
`scales.size()` rows per channel (`src/metrics/ssimulacra2/pipeline.cc:334-340`), so at four scales --
the 32x32 golden pair -- channel B's level 0 reads the row labelled `Y, 1:4`
with a weight of 225, and a fixed table would have pruned a term that dominates
that pair and broken `TestGoldenScores`. The mask is therefore derived per
pipeline from `Ssimulacra2ChannelPruned(channel, level, levels_)`: it reproduces the
sketch's table at six levels and the correct set at every other scale count.
That puts the helper in `src/metrics/ssimulacra2/ssimulacra2_internal.h`/`src/metrics/ssimulacra2/pipeline.cc`, outside the Phase 6
rule's enumerated file set (`src/metrics/**/shaders/* and src/vulkan/shaders/*`, `src/metrics/ssimulacra2/vulkan.cc`,
`src/metrics/ssimulacra2/vulkan.h`). Pinning the seven pairs as a constant in the pipeline
was the alternative, and it would duplicate a fact derived from the weight
table -- one that goes stale silently the next time those weights are retuned,
as a score change rather than a failure. The derivation was kept; both files
already existed and neither gained a dependency.

Checks recorded for the pruning experiment follow.

- `./build/ssimulacra2_test` prints `22 test(s), 0 failed check(s)`;
  `ctest --test-dir build` prints `100% tests passed out of 1`; the ASAN build
  is clean with `detect_leaks=0`.
- Identity: `build/fixtures/gradient.png` against itself prints
  `100.00000000`, exactly, in both plane modes, so `TestIdenticalImages` needed
  no edit and no README exactness claim needed qualifying.
- The five-pair capture moved by +1.7e-4 to +3.8e-4, with no `nan` and no
  negative norm: gradient 93.17481251 -> 93.17505696, alpha -85.86776077 ->
  -85.86737644, 33x17 93.02238381 -> 93.02261281, 1919x1079 85.06003368 ->
  85.06020124, 3840x2160 85.90830295 -> 85.90848933. The bench checksum moved
  with them, 1975.890968 -> 1975.895255, as intended.
- CID22, 300 pairs, the HEAD build as the reference arm: max |diff| 8.07e-4,
  mean 2.67e-4, p99 7.32e-4, RMSE 2.97e-4, no pair over 0.001;
  `accuracy vs MCOS: PASS` with KRCC 0.8145, SRCC 0.9554 and PCC 0.9433 in both
  arms, and `PASS`.
- CID22, all 4292 pairs, against a scalar build of `.reference/ssimulacra2`:
  KRCC 0.7046, SRCC 0.8885, PCC 0.8765 for this implementation against 0.7072,
  0.8901, 0.8784 for the reference binary -- the README's table to four
  decimals, so the pruning moved the fit to MCOS by less than the last digit
  the harness prints (table under *Validation* in the README).
- Speed, five interleaved runs per arm: scale 1.72 -> 0.74 ms (median, -57%),
  prepare 0.61 -> 0.43 ms (-30%), resident pair 2.25 -> 1.21 ms (-46%, 445 ->
  823 pairs/s). The baseline arm measured 2.24, 2.78, 2.25, 2.25, 2.25 ms and
  the candidate 1.22, 1.22, 1.21, 1.21, 1.21 ms.
- `clang-format --dry-run --Werror` passed for the source files listed by the
  repository guide at the time.

`TestGoldenScores`' constants were re-pinned to the measured 93.17505696 and
92.12553218 (tolerance `5e-3` unchanged), and the README's Speed, Precision,
half-planes and deliberate-differences sections now carry the measurements
above.

## Butteraugli against the peers

Butteraugli's oracle is libjxl (`lib/jxl/butteraugli/butteraugli.cc`,
BSD-3-Clause), not a peer: the number this repository produces is compared with a
line-cited double-precision CPU port of the comparator (tests/metrics/butteraugli/butter_ref.cc),
never with Vship's score, because Vship's Butteraugli is not bit-identical to
libjxl (its own README notes ~1.2% by leaving UHF at "15 patterns").

What the peers changed in the design, and what was not taken:

- **Two levels, not a pyramid.** libjxl's comparator folds exactly one
  subsampled level (`Diffmap`, :1829-1875);
the deeper `sub_` chain `Make` builds is never consulted.Per patch the first
        design assumed a nine -
    level pyramid at 4K(1.33x a level); the source says 1.25x and only two levels of
  diff work, so the backend runs two. This is the single largest difference
  from the plan and it came from re-reading the oracle.
- **Fused blur epilogues** (Vship-fast's "five post-op modes in the blur
  epilogue"): adopted as one blur shader with a runtime op that writes the
  range-transformed plane and the residual in the same pass, so the separation
  never round-trips the transformed values through memory.
- **Chains walked by workgroup z** (Vship's kernel-per-plane port): adopted as
  `gl_WorkGroupID.z` over the three channels of a chain, with per-band Malta
  prefactors passed per z (band 0 in `p0..p2`, band 1 in `q0..q2`), so a level's
  LF, HF, UHF and three Malta pairs are one dispatch each instead of one per
  channel.
- **Malta's exact tap parenthesization and duplicated lines**: libjxl's UHF
  unit repeats four of its sixteen line terms (:882-946). The shader keeps them
  (dropping them under-weights UHF energy); the CPU oracle keeps them too, and
  the two agree to -0.65%/-0.014% on the fixtures.
- **Not taken: libjxl's own kernel default.** `MakeGaussian` (SSIMULACRA 2,
  3.5 sigma, 13 taps) is deliberately not shared with Butteraugli's
  `ComputeKernel` (2.25 sigma, radii 2..16): sharing would change the metric.
  The shared part is the arena, descriptors, submission and upload paths in
  `src/vulkan/vulkan_runtime.{
    h, cc}`.
- **Not taken: fp64 arithmetic** (Vship-fast's double-float exactness module).
  The measured fp32 residual against the double oracle is 0.65% at worst, well
  inside the 5% budget, so the branch-math escalation was unnecessary.

Measured on this machine (RTX 5070, driver 615.78), resident 4K pair with the
reference chain cached: **11.1 ms/pair (1.34 ms/MPx)**; one 20-iteration run
splits it opsin 2.01 / separate 2.53 / malta 2.86 / diff 1.50 / mask 0.90 /
combine 0.70 / reduce 0.23 / untimed 0.35 ms, and 1080p is 2.46 ms/pair
(1.19 ms/MPx). `diff` now carries the mask's precompute, which used to be its
own dispatch.

The first correct build was 63.9 ms/pair, and five measured steps got it here:
the reduce gained a second GPU stage so the host reads 16 bytes instead of
32 400 partials through the BAR mapping (49 ms of PCIe round trips), and the
blur taps moved from arena loads inside both passes to a register array at
shader start -- 62.5 -> 77.2 pairs/s, `separate` 6.16 -> 3.56 ms, checksum
bit-identical, both interleaved A/B against the previous binary.

Then the two passes took scale.comp's shape, four outputs sharing one window of
source values (`separate` 3.65 -> 2.73 ms, `mask` 1.74 -> 1.58, pair 12.89 ->
11.37, medians of seven 20-iteration runs, score identical), and the L2 terms
fused with the mask's precompute into one pass so the post-UHF planes are read
once (diff + mask 2.68 -> 2.42 ms, pair 11.96 -> 11.14). The pattern is the one
run_bench's `scale.comp` established: 16 shared loads feeding 52 taps instead
of 33 loads per output per pass. Vship-fast
publishes 1.088 ms/MPx on a 3070, which at the ~1.5x 3070->5070 allowance is
~0.73 ms/MPx, so this backend is now ~1.8x their number and still above the
0.35 ms/MPx design target. What is left is Malta's window traffic: a 32x32
output tile reads a 40x40 window (1.56x halo amplification) and the whole Malta
unit is only 8% of that stage, so a wider tile is the only lever there.
Dispatch count per pair (cache hit) is 24 binds, 32 on the streaming path
(measured through bench/profile_nsys.sh: six submissions of the resident arm
are 152 binds, i.e. 24 per pair plus the reference chain's 8), against Vship's
whole-metric kernel list.

Three ablations inside `malta` priced what is left there, and all were
rejected:
hoisting the four duplicated line sums (36 of the unit's 132 shared loads
removed, bit-identical) measured malta 2.84 against 2.80 ms in an interleaved
A/B of five runs each -- 1.4%, under the 2% bar, so it was reverted rather than
kept as folklore, and replacing the two divides with one reciprocal measured the
same nothing (2.82-3.17 against 2.97-3.08) and is not bit-identical, so it was
reverted too. The stage is therefore bound by the latency of its own sixteen
independent of its loads and divisions. Removing the whole unit costs 0.23 ms of
2.82 (8%), so splitting its dependency chain across lanes has a ceiling below 1%
of the pair -- it was skipped rather than bought with a golden re-pin.

### 2026-11-02: repeated references and Butteraugli passes

Unprofiled interleaved A/B on the RTX 5070, driver 615.78. The baseline is
`aca666e` (including the cache correctness fix); the candidate includes the
opsin loader, erosion/combine fusion, resident-reference pyramid cache and
mixed-input API. The 3840x2160 pair is ffmpeg `testsrc` against `testsrc2`.
Each arm ran five times, alternating baseline then candidate. Resident arms
used `--iterations 20 --warmup 3 --resident 2`; streaming arms used
`--iterations 8 --warmup 2`. The mixed arm compares baseline streaming with
`--resident-reference` on the candidate: the original is uploaded once before
timing and each distortion is still uploaded in the timed call. It represents
a reused reference, not a first-pair or all-unique-reference measurement.

Each cell is median ms per pair with the min–max spread from the five runs;
stage values are printed to 0.01 ms by the benchmark. `mask + combine` is
included because fusion moves the erosion dispatch across that timing boundary.

| metric | arm | stage | baseline median (range) | candidate median (range) |
| --- | --- | --- | --- | --- |
| ssimulacra2 | resident | prepare | 0.42 (0.42–0.42) | 0.21 (0.21–0.21) |
| ssimulacra2 | resident | scale | 0.68 (0.68–0.68) | 0.68 (0.68–0.68) |
| ssimulacra2 | resident | score | 0.00 (0.00–0.00) | 0.00 (0.00–0.00) |
| ssimulacra2 | resident | untimed | 0.09 (0.08–0.10) | 0.09 (0.08–0.10) |
| ssimulacra2 | stream | prepare | 0.43 (0.43–0.43) | 0.43 (0.43–0.43) |
| ssimulacra2 | stream | scale | 0.68 (0.68–0.68) | 0.68 (0.68–0.68) |
| ssimulacra2 | stream | score | 0.00 (0.00–0.00) | 0.00 (0.00–0.00) |
| ssimulacra2 | stream | untimed | 2.30 (2.29–2.30) | 2.29 (2.28–2.30) |
| ssimulacra2 | mixed | prepare | 0.43 (0.43–0.43) | 0.22 (0.22–0.22) |
| ssimulacra2 | mixed | scale | 0.68 (0.68–0.69) | 0.68 (0.68–0.68) |
| ssimulacra2 | mixed | score | 0.00 (0.00–0.00) | 0.00 (0.00–0.00) |
| ssimulacra2 | mixed | untimed | 2.29 (2.28–2.30) | 1.18 (1.18–1.20) |
| butteraugli | resident | opsin | 1.98 (1.98–1.98) | 0.99 (0.99–0.99) |
| butteraugli | resident | separate | 2.47 (2.47–2.47) | 2.47 (2.47–2.47) |
| butteraugli | resident | malta | 2.68 (2.68–2.68) | 2.68 (2.68–2.68) |
| butteraugli | resident | diff | 1.42 (1.42–1.42) | 1.42 (1.42–1.42) |
| butteraugli | resident | mask | 0.85 (0.85–0.85) | 0.39 (0.39–0.39) |
| butteraugli | resident | combine | 0.69 (0.69–0.69) | 0.78 (0.78–0.78) |
| butteraugli | resident | reduce | 0.18 (0.18–0.18) | 0.18 (0.18–0.18) |
| butteraugli | resident | untimed | 0.32 (0.31–0.33) | 0.33 (0.30–0.34) |
| butteraugli | resident | mask + combine | 1.54 (1.54–1.54) | 1.17 (1.17–1.17) |
| butteraugli | stream | opsin | 3.97 (3.97–3.97) | 1.99 (1.99–1.99) |
| butteraugli | stream | separate | 4.94 (4.94–4.94) | 4.94 (4.94–4.94) |
| butteraugli | stream | malta | 2.68 (2.68–2.68) | 2.68 (2.68–2.68) |
| butteraugli | stream | diff | 1.42 (1.42–1.42) | 1.42 (1.42–1.42) |
| butteraugli | stream | mask | 0.85 (0.85–0.85) | 0.39 (0.39–0.39) |
| butteraugli | stream | combine | 0.69 (0.69–0.69) | 0.78 (0.78–0.78) |
| butteraugli | stream | reduce | 0.18 (0.18–0.18) | 0.18 (0.18–0.18) |
| butteraugli | stream | untimed | 2.53 (2.51–2.54) | 2.53 (2.48–2.54) |
| butteraugli | stream | mask + combine | 1.54 (1.54–1.54) | 1.17 (1.17–1.17) |
| butteraugli | mixed | opsin | 3.97 (3.97–3.97) | 0.98 (0.98–0.98) |
| butteraugli | mixed | separate | 4.94 (4.94–4.94) | 2.47 (2.47–2.47) |
| butteraugli | mixed | malta | 2.68 (2.68–2.68) | 2.68 (2.68–2.68) |
| butteraugli | mixed | diff | 1.42 (1.42–1.42) | 1.42 (1.42–1.42) |
| butteraugli | mixed | mask | 0.85 (0.85–0.85) | 0.39 (0.39–0.39) |
| butteraugli | mixed | combine | 0.69 (0.69–0.69) | 0.78 (0.78–0.78) |
| butteraugli | mixed | reduce | 0.18 (0.18–0.18) | 0.18 (0.18–0.18) |
| butteraugli | mixed | untimed | 2.51 (2.50–2.55) | 1.38 (1.38–1.51) |
| butteraugli | mixed | mask + combine | 1.54 (1.54–1.54) | 1.17 (1.17–1.17) |

The same protocol on a 1920x1080 `testsrc` / `testsrc2` pair gives the
following stage medians and ranges (ms). The benchmark prints two decimals;
small changes near a rounding boundary should not be read as precise deltas.

| metric | arm | stage | baseline median (range) | candidate median (range) |
| --- | --- | --- | --- | --- |
| ssimulacra2 | resident | prepare | 0.10 (0.10–0.10) | 0.05 (0.05–0.05) |
| ssimulacra2 | resident | scale | 0.14 (0.14–0.14) | 0.15 (0.15–0.15) |
| ssimulacra2 | resident | score | 0.00 (0.00–0.00) | 0.00 (0.00–0.00) |
| ssimulacra2 | resident | untimed | 0.08 (0.08–0.08) | 0.08 (0.08–0.08) |
| ssimulacra2 | stream | prepare | 0.10 (0.10–0.10) | 0.10 (0.10–0.10) |
| ssimulacra2 | stream | scale | 0.14 (0.14–0.14) | 0.14 (0.14–0.14) |
| ssimulacra2 | stream | score | 0.00 (0.00–0.00) | 0.00 (0.00–0.00) |
| ssimulacra2 | stream | untimed | 0.64 (0.63–0.64) | 0.64 (0.63–0.64) |
| ssimulacra2 | mixed | prepare | 0.10 (0.10–0.10) | 0.05 (0.05–0.05) |
| ssimulacra2 | mixed | scale | 0.14 (0.14–0.14) | 0.15 (0.15–0.15) |
| ssimulacra2 | mixed | score | 0.00 (0.00–0.00) | 0.00 (0.00–0.00) |
| ssimulacra2 | mixed | untimed | 0.63 (0.63–0.64) | 0.37 (0.36–0.38) |
| butteraugli | resident | opsin | 0.52 (0.52–0.52) | 0.26 (0.26–0.26) |
| butteraugli | resident | separate | 0.56 (0.56–0.57) | 0.57 (0.56–0.57) |
| butteraugli | resident | malta | 0.58 (0.57–0.58) | 0.57 (0.57–0.58) |
| butteraugli | resident | diff | 0.26 (0.26–0.26) | 0.26 (0.26–0.26) |
| butteraugli | resident | mask | 0.18 (0.17–0.18) | 0.09 (0.09–0.09) |
| butteraugli | resident | combine | 0.14 (0.14–0.14) | 0.18 (0.18–0.18) |
| butteraugli | resident | reduce | 0.04 (0.04–0.04) | 0.04 (0.04–0.04) |
| butteraugli | resident | untimed | 0.09 (0.08–0.12) | 0.08 (0.07–0.09) |
| butteraugli | resident | mask + combine | 0.32 (0.31–0.32) | 0.27 (0.27–0.27) |
| butteraugli | stream | opsin | 1.03 (1.03–1.03) | 0.51 (0.51–0.51) |
| butteraugli | stream | separate | 1.13 (1.13–1.13) | 1.13 (1.13–1.13) |
| butteraugli | stream | malta | 0.58 (0.57–0.58) | 0.57 (0.57–0.58) |
| butteraugli | stream | diff | 0.26 (0.26–0.26) | 0.26 (0.26–0.26) |
| butteraugli | stream | mask | 0.17 (0.17–0.18) | 0.09 (0.09–0.09) |
| butteraugli | stream | combine | 0.14 (0.14–0.14) | 0.18 (0.18–0.18) |
| butteraugli | stream | reduce | 0.04 (0.04–0.04) | 0.04 (0.04–0.04) |
| butteraugli | stream | untimed | 0.66 (0.65–0.69) | 0.66 (0.65–0.68) |
| butteraugli | stream | mask + combine | 0.31 (0.31–0.32) | 0.27 (0.27–0.27) |
| butteraugli | mixed | opsin | 1.03 (1.03–1.03) | 0.24 (0.24–0.24) |
| butteraugli | mixed | separate | 1.13 (1.13–1.13) | 0.56 (0.56–0.57) |
| butteraugli | mixed | malta | 0.57 (0.57–0.58) | 0.57 (0.57–0.57) |
| butteraugli | mixed | diff | 0.26 (0.26–0.26) | 0.26 (0.26–0.26) |
| butteraugli | mixed | mask | 0.17 (0.17–0.18) | 0.09 (0.09–0.09) |
| butteraugli | mixed | combine | 0.14 (0.14–0.14) | 0.18 (0.18–0.18) |
| butteraugli | mixed | reduce | 0.04 (0.04–0.04) | 0.04 (0.04–0.04) |
| butteraugli | mixed | untimed | 0.67 (0.67–0.84) | 0.37 (0.36–0.38) |
| butteraugli | mixed | mask + combine | 0.31 (0.31–0.32) | 0.27 (0.27–0.27) |

All 1080p scores and checksums matched their corresponding baseline arms.

The two individual Butteraugli experiments also cleared the stage gate against
`aca666e`: the opsin tile loader reduced resident opsin 1.98 (1.97–1.98) ->
0.99 (0.99–0.99) ms and streaming opsin 3.96 (3.96–3.96) -> 1.99
(1.99–1.99) ms; erosion/combine fusion reduced mask + combine 1.54
(1.54–1.54) -> 1.17 (1.17–1.17) ms in both arms. The separate SSIMULACRA
resident-reference prototype reduced prepare 0.42 (0.42–0.42) -> 0.21
(0.21–0.21) ms. The original-upload-once prototype reduced SSIMULACRA's
streaming untimed row 2.30 (2.28–2.33) -> 1.19 (1.19–1.20) ms; its
one-time upload was outside the loop. All individual and combined run scores
and checksums matched their baseline arms. On 300 CID22 pairs, each Butteraugli
shader prototype and both mixed-input metrics matched baseline scores exactly
at the bench's printed precision. The final combined implementation also
matched all 300 baseline scores for both metrics in streaming and mixed modes.
The cache tests additionally cover background changes, reference replacement
and mutable mapped buffers.

### Vship (CUDA) against this backend, measured

This peer comparison predates the 2026-11-02 optimizations and has not been
repeated against the current build.

`bench/vship_bench.cc` has the same shape as `ssimulacra2_bench`: it decodes
with this repository's io library, hands Vship's public C API
(`src/VshipAPI.h`) the same full-range, identity-matrix, sRGB planes the CLI
pins for PNG (with the image's own sample depth), and times only the metric
call. `python3 tools/build_references.py --only vship-cuda,vship-vulkan` builds
it -- it copies `.reference/Vship` under `build/`, runs `make BACKEND=Cuda` or
`BACKEND=Vulkan` (nvcc 13.4, `-arch=native`) and links the tool against the
resulting `libvship.so`. Vship is never part of this project's build and only
its C API is called; no Vship code is copied.

Interleaved runs on this machine (RTX 5070, driver 615.78), 20 iterations and
3 warm-up pairs per round, 5 rounds at 4K and 3 at 1080p, medians:

| arm | 4K ms/pair | 1080p ms/pair |
|---|---|---|
| Vship 5.1.2, CUDA backend | 29.98 | 5.95 |
| this backend, resident (reference cached) | 10.99 | 2.38 |
| this backend, streaming (upload + rebuild) | 18.1 | -- |
| Vship 5.1.2, ssimu2 | 6.32 | -- |
| this backend, ssimu2 resident | 1.20 | -- |

So Butteraugli costs 2.7x less per resident pair than Vship's CUDA build, and
1.66x less when the reference is rebuilt every call -- the fair comparison,
since Vship is handed fresh planes each time. SSIMULACRA 2 is 5.3x (1.86x
streaming). Vship's Butteraugli is its straight kernel-per-pass port, not the
faster fork: Vship-fast publishes 1.088 ms/MPx, about 8.7 ms/pair at 4K before
any card allowance, so the plain-Vship table above is not the strong
comparison.

Scores are not comparable across tools. On the 4K testsrc/testsrc2 pair Vship's
Butteraugli reports norminf 504.404236 against this backend's 504.458984 (0.01%
apart, both being the diffmap maximum), but on the 32x32 gradient fixture it
reports 3.639969 against libjxl's 3.775699854 (-3.6%, and this backend's
3.751055). The throughput table is the only claim here; accuracy is what the
CPU oracle gates.
