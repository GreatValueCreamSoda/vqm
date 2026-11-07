# PSNR-HVS resident throughput

2026-11-06. Measured the new FP32 PSNR-HVS resident path on NVIDIA GeForce
RTX 5070, driver 615.78.08. Five runs per resolution alternated 1080p then
4K in one sitting. Each process used one pipeline, two resident inputs,
20 timed calls and three warmups. Construction, decode and upload are outside
the measurement; both images are prepared on every timed call. The same pair
is reused, so this is a hot-memory workload rather than a unique-image stream.

## Inputs and build

The inputs are the first frames of the existing synthetic YUV420 8-bit clips
under `.cache/bench/cvvdp/`, at 1920x1080 and 3840x2160. Original pixels come
from FFmpeg `testsrc2`; distortion quantizes luma in steps of eight with
`lutyuv=y='floor(val/8)*8'`, leaving chroma unchanged. Generation commands and
the original cache provenance are in the
[CVVDP implementation record](../cvvdp/cvvdp-implementation.md).
The benchmark decodes one frame from each clip. All PSNR-HVS processing is
FP32, independent of that 8-bit storage.

Used the fresh GNU 16.2.1 Release build with warnings as errors described in
the [implementation record](psnr-hvs-implementation.md). No implementation
changes were made during this measurement. Executable and full input hashes,
all five values per series and compact stage data are in
[`resident-throughput-20261106.json`](resident-throughput-20261106.json).
Raw per-run commands, logs and CSVs remain in
`artifacts/reports/psnr-hvs-throughput-20261106/`.

## Results

Throughput is the reciprocal of wall time over the fixed 20 timed calls.
Latency is the median of each run's 20 individual Compute calls. The table
reports the median across the five runs; ranges retain every run.

| Resolution | Pairs/s, median (range) | Wall ms/pair, median (range) | Median call latency, ms | Owned buffers, MB |
| --- | ---: | ---: | ---: | ---: |
| 1920x1080 | 2424.484 (2236.964-2439.129) | 0.412459 (0.409982-0.447034) | 0.410289 | 116.628544 |
| 3840x2160 | 541.781 (490.022-557.896) | 1.845765 (1.792449-2.040726) | 1.844018 | 466.512256 |

Buffer values count the library's arena/staging/auxiliary allocations rather
than all driver VRAM. The resident path disables streaming storage; two raw
resident slots are included.

Stage medians below are medians of five per-run stage means, in ms/call:

| Resolution | Prepare | DCT and masking | Reduction | Host score |
| --- | ---: | ---: | ---: | ---: |
| 1920x1080 | 0.034989 | 0.301472 | 0.041970 | 0.001097 |
| 3840x2160 | 0.255379 | 1.192651 | 0.358350 | 0.001219 |

The per-run summed GPU-stage total had median 0.378476 ms at 1080p and
1.806151 ms at 4K. A sum of independent stage medians need not equal the
median total. Detailed stage ranges remain in the compact JSON.

All five runs per resolution returned the same combined score and checksum:
1080p `40.20762634` dB / `924.77540588`; 4K `40.35023117` dB /
`928.05531693`. Each checksum matches 23 calls: three warmups and 20 timed.
The two unchanged chroma planes score infinity, while combined/luma scores
remain finite. Stability alone does not establish reference agreement.

## Reproduction and limits

For each of five rounds, run the following command first with `SIZE=1080`
and then `SIZE=4k`, giving each round's CSVs distinct paths:

```sh
./build-psnr-hvs/vqm_bench --metric psnr-hvs \
  --pair .cache/bench/cvvdp/${SIZE}-original.y4m \
         .cache/bench/cvvdp/${SIZE}-distorted.y4m \
  --resident 2 --workers 1 --iterations 20 --warmup 3 \
  --csv artifacts/reports/psnr-hvs-throughput-20261106/run-pairs.csv \
  --stages-csv artifacts/reports/psnr-hvs-throughput-20261106/run-stages.csv
```

4K wall time fell from 2.040726 to 1.800261 ms/pair during the five-run series,
with a minimum of 1.792449 ms. The full spread is material and was retained;
clock behavior was not instrumented. No result was discarded or selected as
the claimed sustained rate. These measurements describe one synthetic pair
per size and one worker. No alternatives or before/after builds were compared,
so there is no speedup claim. Upload/streaming throughput and corpus ranking
qualification remain outside this resident measurement. Test/qualification
integration remains deferred as requested.

Conclusion: median resident throughput on this workload is about 2,424 pairs/s
at 1080p and 542 pairs/s at 4K, with the recorded run ranges above. DCT/masking
is the largest measured GPU stage; no optimization was attempted here.

These are the initial implementation's measurements. The later
[optimization report](optimization-20261106.md) preserves a fresh interleaved
baseline comparison and updated throughput; the historical values above are
unchanged.
