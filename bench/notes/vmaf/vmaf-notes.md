# VMAF 1080p measurement

These are historical measurements from commit `c96b5f7`, before the later
v0/v1 optimizations. See [`vmaf-v0-optimization.md`](vmaf-v0-optimization.md)
and [`vmaf-v1-optimization.md`](vmaf-v1-optimization.md) for interleaved
before-and-after measurements. The latest v1 results are in
[`vmaf-v1-pipeline-optimization.md`](vmaf-v1-pipeline-optimization.md), followed by
[`vmaf-v1-algorithm-optimization.md`](vmaf-v1-algorithm-optimization.md).

On an NVIDIA GeForce RTX 5070 with driver 615.78.08, five interleaved
resident and streaming runs of both pipelines produced:

| input, model and workload | median ms/pair | range ms/pair | pairs/s from median |
| --- | ---: | ---: | ---: |
| 4:4:4 v0 resident | 0.7406 | 0.7379–0.7411 | 1,350 |
| 4:4:4 v0 streaming | 1.8410 | 1.8395–1.8499 | 543 |
| 4:4:4 v1 resident | 7.6432 | 7.6176–7.6564 | 131 |
| 4:4:4 v1 streaming | 8.7159 | 8.6802–8.8410 | 115 |
| 4:2:0 v0 resident | 0.7409 | 0.7387–0.7455 | 1,350 |
| 4:2:0 v0 streaming | 1.2906 | 1.2875–1.2931 | 775 |
| 4:2:0 v1 resident | 7.4102 | 7.3493–7.4449 | 135 |
| 4:2:0 v1 streaming | 7.9742 | 7.9117–8.0904 | 125 |

The 1920×1080 inputs were two pre-existing 4K PNG fixtures scaled with
FFmpeg to planar YUV444P10LE and YUV420P10LE and wrapped as one-frame Y4M.
Resident arms used 20 timed calls and 3 warm-ups; streaming arms used 8 and 2.
The measured 4:4:4 Y4M
SHA-256 values were `637584b93245a1a3bfc2cbbae88b866973daa771e71c862ac2543367780e130a`
and `b32a0153817b1ac8d4a8eabf7ba68dc76c8e5ade40bde010fd362a71c87cfc5f`.
The 4:2:0 files were `62e7eb9841117655e242944b272dd6df32d660ed723d868e42562307c8497e20`
and `3fe644c0c252cdb8bfb689a116c2dfd8df3b4fe735e2e1911f892db7a25ab4fd`.
Both images are fully prepared for each call. The resident pair reuses pixels,
so it is a hot-memory approximation to unique inputs, not a measurement of
an unlimited stream. Each iteration advances a sequence with a repeated
reference, so reference-frame motion after the first frame is zero. Decoder
time and resident uploads are outside the timed loop. On 4:4:4, VQM scored
92.424365 for v0 and 96.093930 for v1; pinned libvmaf produced 92.424163 and
96.092752 respectively on the same raw planes. V0 checksums were 2125.760403
(23 calls) and 924.243654 (10 calls); v1 checksums were 2210.160383 and
960.939297. On 4:2:0 the scores were 92.424365 and 95.973505; the respective
checksums were 2125.760403 / 924.243654 and 2207.390606 / 959.735046.

For 4:4:4, resident median v0 GPU stages were luma 0.0504 ms, VIF 0.4371 ms, ADM
0.1209 ms and motion 0.0794 ms. V1 stages were coded-plane preparation
0.1465 ms, ADM3 0.1568 ms, chroma SpEED 6.1162 ms, CAMBI 0.8893 ms and
motion 0.0874 ms. The host model calculation took 0.0010 ms for v0 and
0.0019 ms for v1; the remaining time includes submission, synchronization and
other host work. These are measurements of this build, device and input, not
general throughput guarantees. For 4:2:0, median v1 SpEED time was 5.9885 ms
resident. The 4:4:4 format has four times the chroma samples of 4:2:0 and is
important when comparing inputs.

For any same-size 10-bit Y4M pair, reproduce the workload with:

```sh
./build/ssimulacra2_bench --metric vmaf-v0 --pair REF.y4m DIST.y4m \
  --iterations 20 --warmup 3 --resident 2
./build/ssimulacra2_bench --metric vmaf-v0 --pair REF.y4m DIST.y4m \
  --iterations 8 --warmup 2
./build/ssimulacra2_bench --metric vmaf-v1 --pair REF.y4m DIST.y4m \
  --iterations 20 --warmup 3 --resident 2
./build/ssimulacra2_bench --metric vmaf-v1 --pair REF.y4m DIST.y4m \
  --iterations 8 --warmup 2
```

Take five runs of each, alternating arms in one sitting, and retain scores,
checksums and stage tables. An ordinary non-timing build has no timers.
The v1 result uses the standard SDR v1.0.16 3H model, not a 4K, phone, HFR or
HDR variant.
