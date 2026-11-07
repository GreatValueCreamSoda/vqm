# Butteraugli workspace memory

These changes reduce retained Butteraugli buffer memory by reusing regions
whose lifetimes do not overlap. They do not change metric arithmetic. The
objective was memory reduction; the timing comparisons below check for a
regression, rather than establish a speedup.

Measurements used an RTX 5070 with driver 615.78.08 and 8-bit YUV420 input.
The original baseline was `111e596`. The three experiments were applied in
order, each compared with the immediately preceding version:

1. **Scratch reuse:** put both three-plane scratch regions in the distorted
   chain, whose corresponding planes are unused at those stages. The reference
   chain remains intact for caching. This removes 248,832,000 bytes at 4K.
2. **RGB subsample reuse:** store the two sequentially consumed subsampled RGB
   inputs in level-0 AC storage before that storage is cleared for the diff.
   This removes 49,766,400 bytes at 4K.
3. **Coarse-level reuse:** share AC storage between levels and put the coarse
   diffmap in a dead distorted-chain plane. Barriers order the AC clears after
   its earlier compute uses. This removes 33,177,600 bytes at 4K.

Those savings total 331,776,000 bytes at 4K and 82,944,000 bytes at 1080p.
Odd sizes include allocation alignment, and a single-level geometry does not
receive the coarse-level saving.

## Allocations

These are actual Vulkan allocation sizes in bytes on this device, verified
for one worker with zero resident slots. External input disables streaming
source regions; streaming enables both. The device uses direct host-visible
device-local memory, so these configurations have no separate staging
allocation. External input buffers, resident slots if configured, other workers,
Vulkan driver resources and unrelated GPU usage are excluded.

| Geometry | Input | Baseline bytes | New bytes | Saved |
| --- | --- | ---: | ---: | ---: |
| 1080p | External | 365,084,096 | 282,140,096 | 22.7% |
| 1080p | Streaming | 431,439,296 | 348,495,296 | 19.2% |
| 4K | External | 1,460,333,696 | 1,128,557,696 | 22.7% |
| 4K | Streaming | 1,725,754,496 | 1,393,978,496 | 19.2% |

The same reduction applies to the shared SSIMULACRA 2 and Butteraugli workspace
when Butteraugli determines the arena size. Query memory requirements on the
target device for a particular configuration; allocation sizes and staging
policy can differ between devices.

## Performance

Each arm used five interleaved runs. Resident tests used `--resident 2
--iterations 20 --warmup 3`; Butteraugli caches the reference chain in this
workload, so the distorted image is rebuilt each call. Streaming tests used
`--iterations 8 --warmup 2` and rebuild both chains. Resident pixels are
pre-uploaded and reused. Values are pair time in milliseconds: median
followed by the minimum and maximum of the five runs.

| Experiment, 4K | Input | Before ms (min–max) | After ms (min–max) |
| --- | --- | ---: | ---: |
| Scratch reuse | Resident | 11.0042 (10.9417–11.0152) | 11.0138 (10.9993–11.0263) |
| Scratch reuse | Streaming | 18.4814 (18.4496–18.4980) | 18.4921 (18.4839–18.5016) |
| RGB subsample reuse | Resident | 10.9951 (10.9737–11.0149) | 11.0269 (11.0088–11.0330) |
| RGB subsample reuse | Streaming | 18.4928 (18.4471–18.5039) | 18.5054 (18.4586–18.5147) |
| Coarse-level reuse | Resident | 11.0064 (10.9840–11.0293) | 11.0044 (10.9725–11.0276) |
| Coarse-level reuse | Streaming | 18.5024 (18.4580–18.5167) | 18.4991 (18.4758–18.5049) |

The combined change was also compared directly against `111e596` in a fresh
interleaved set:

| Geometry | Input | Baseline ms (min–max) | New ms (min–max) |
| --- | --- | ---: | ---: |
| 1080p | Resident | 2.5431 (2.5400–2.5516) | 2.5463 (2.5433–2.5523) |
| 1080p | Streaming | 4.3838 (4.3768–4.5624) | 4.3819 (4.3760–4.3913) |
| 4K | Resident | 11.0114 (10.9875–11.0292) | 10.9974 (10.9302–11.0345) |
| 4K | Streaming | 18.4929 (18.4435–18.5128) | 18.5041 (18.4820–18.6289) |

The median changes in this final comparison range from -0.13% to +0.13%;
these measurements do not establish a throughput improvement. Scores and
checksums matched between arms.

The shared-workspace regression comparison evaluates SSIMULACRA 2 and
Butteraugli serially per pair, using either pre-uploaded Vulkan buffers or
streamed input. It uses five interleaved runs per arm, with the same iteration
and warmup counts as above:

| Geometry | Input | Baseline ms (min–max) | New ms (min–max) |
| --- | --- | ---: | ---: |
| 1080p | Vulkan buffers | 4.3721 (4.3585–4.4125) | 4.4045 (4.3680–4.4088) |
| 1080p | Streaming | 5.5042 (5.4579–5.6389) | 5.4691 (5.4385–5.5033) |
| 4K | Vulkan buffers | 18.1387 (18.0983–18.2310) | 18.1701 (18.1360–18.1811) |
| 4K | Streaming | 22.6383 (22.6196–22.6584) | 22.6380 (22.6195–22.7026) |

Shared-workspace scores and checksums matched. Detailed measurements and
provenance are in [butter-memory-results.json](butter-memory-results.json).

## Accuracy

All 4,292 CID22 pairs produced the same Butteraugli scores as the previous
implementation to eight decimal places. Against MCOS, our KRCC/SRCC/PCC were
0.5445/0.7335/0.6701; libjxl's were 0.5511/0.7413/0.6843. The 0.02
correlation gate passed. Strict cross-implementation score parity fails
historically; unchanged scores against the previous implementation and the
correlation gate measure different properties.

The 300-pair SSIMULACRA 2 CID22 correlation gate also passed at its 0.005
tolerance. The release and ASan/UBSan GPU suites passed 153 tests, and all 12
CTest checks passed.
