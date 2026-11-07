# Further VMAF v1 CAMBI optimization

For subsequent histogram and filter batching measurements, see
[the follow-up report](vmaf-v1-histogram-filter-batching.md).

This follows [the pooling and parallelism changes](vmaf-v1-pooling-parallelism.md).
Measurements compare baseline `2941f74` and implementation `c31ba05` on an
RTX 5070, driver 615.78.08, on 2026-11-02. Both use the standard SDR v1.0.16
3H model. Five runs per arm were interleaved in one sitting: resident uses
`--iterations 20 --warmup 3 --resident 2`, streaming uses `--iterations 8
--warmup 2`. Both images are prepared every call. Resident preuploads inputs
and repeats their pixels, so it is a hot-memory approximation; both modes
exclude decoding, and motion becomes zero after the first call. Ranges are
min–max of five run averages, not per-frame percentiles.

The primary 1080p and 4K pairs are 10-bit YUV420, and ducks is one 8-bit
1080p frame distorted with x264 CRF 28, not Netflix's full video workload.
The 4K 8-bit pair was made from the 10-bit 4K pair with `ffmpeg -frames:v 1
-pix_fmt yuv420p`. The 1440p and crossover pairs are scaled from the same 4K
pair with `ffmpeg -frames:v 1 -vf scale=WIDTH:HEIGHT -pix_fmt yuv420p10le
-strict -1`. The dense 10-bit YUV420 stress pairs have constant luma 100 versus
101 and chroma 512, and exercise fully active contrast tiles; they are not
representative of a general image corpus. Their paths, SHA-256s and benchmark
binary hashes are in `artifacts/reports/cambi-next/`. The earlier inputs are identified in
[`vmaf-v1-algorithm-optimization.md`](vmaf-v1-algorithm-optimization.md).

| workload | baseline ms/pair, median (range) | candidate ms/pair, median (range) | pairs/s before → after | CAMBI ms before → after |
| --- | ---: | ---: | ---: | ---: |
| 1080p resident | 0.822323 (0.819587–0.830567) | 0.768803 (0.759888–0.781098) | 1216 → 1301 | 0.372370 → 0.314126 |
| 1440p resident | 1.336000 (1.325819–1.338344) | 1.197656 (1.196096–1.205607) | 749 → 835 | 0.629723 → 0.501123 |
| 4K resident | 3.340018 (3.312234–3.373333) | 2.921944 (2.912350–2.929150) | 299 → 342 | 1.853550 → 1.473757 |
| 4K 8-bit resident | 3.297103 (3.274454–3.301183) | 2.937050 (2.888725–2.947128) | 303 → 340 | 1.861618 → 1.500979 |
| ducks resident | 0.642315 (0.641800–0.642845) | 0.626646 (0.624371–0.631356) | 1557 → 1596 | 0.202168 → 0.187546 |
| dense 1080p resident | 1.017880 (1.007334–1.021599) | 0.863772 (0.856662–0.865852) | 982 → 1158 | 0.603043 → 0.453557 |
| dense 4K resident | 4.380757 (4.340232–4.412176) | 3.631718 (3.628344–3.656093) | 228 → 275 | 2.936819 → 2.224294 |
| 1080p streaming | 1.380204 (1.379119–1.381754) | 1.321605 (1.307840–1.331499) | 725 → 757 | 0.372436 → 0.314388 |
| 4K streaming | 5.499729 (5.491944–5.515689) | 5.115226 (5.098221–5.147615) | 182 → 196 | 1.851284 → 1.472892 |
| ducks streaming | 0.927676 (0.924031–0.949860) | 0.907846 (0.897211–0.916346) | 1078 → 1102 | 0.202248 → 0.187588 |

Scores and checksums matched at printed precision for every run and every
resident/streaming arm. The 1080p score/checksum was 95.973463 / 2207.389651,
4K was 96.065006 / 2209.495149, and ducks was 75.839083 / 1744.298908.
Corresponding streaming checksums were 959.734631, 960.650065 and 758.390830.
All per-run tables, additional streaming rows, timestamps and input/binary
hashes are in `artifacts/reports/cambi-next/`.

A separate same-sitting v0/v1 comparison on the ducks frame gave resident
0.569976 (0.568121–0.570416) / 0.628320 (0.625991–0.638665) ms, or
1755 / 1592 pairs/s; streaming was 0.843896 (0.839602–0.849627) / 0.901581
(0.898706–0.905756) ms, or 1185 / 1109 pairs/s. The models and scores differ;
this still-frame workload is not Netflix's multithreaded video benchmark.

## Changes and independent stage gates

- **Parallel contrast distances (`8cc097e`).** Four lanes per output row
  independently calculate the four contrast terms, using the existing
  read/update barrier to publish them. An independent interleaved comparison
  moved CAMBI 0.371323 → 0.344024 ms at 1080p and 1.849688 → 1.726938 ms at
  4K, with identical scores/checksums. Dense 4K CAMBI moved 2.930426 →
  2.483261 ms. On ducks it moved 0.201085 → 0.201656 ms (+0.28%); the frame
  has relatively few active contrast groups.
- **Shared 7×7 mask filter (`3e02947`).** A 16×16 tile computes the horizontal
  sums for its 3-row halo in shared memory and then the vertical mask, removing
  a full-image pass. Only exactly representable integer sums are combined.
  Independent CAMBI medians moved 0.371386 → 0.365851 ms at 1080p and
  1.851246 → 1.755950 ms at 4K, with identical scores/checksums. Against the
  paired diagnostic baseline, targeted preparation moved 0.061355
  (0.061259–0.061376) → 0.056398 (0.056203–0.056405) ms at 1080p and
  0.465886 (0.465810–0.466038) → 0.370298 (0.370173–0.370387) ms at 4K.
  The diagnostic uses separate builds and its extra timestamps perturb
  timing. Direct CPU-oracle tests cover image and tile
  boundaries, four input depths and encoded depths 6–16.
- **Cooperative radix selection (`5618f78`).** Sixteen lanes sum independent
  histogram blocks, then one selects the exact bin in the chosen block. Zero,
  ties, already-complete selections and ordering retain their former rules.
  GPU-timestamp diagnostics bracketed the radix substage in separate builds:
  0.031739 (0.031717–0.031746) → 0.022589 (0.022586–0.022600) ms at
  1080p and 0.068101 (0.068010–0.068104) → 0.058870
  (0.058827–0.058891) ms at 4K. An independent uninstrumented comparison
  measured whole CAMBI 0.371181 → 0.362214 ms and 1.849829 → 1.841112 ms.
  Scores/checksums matched; direct selection tests include ties and scattered
  keys.
- **Eight-row packed contrast tile (`31fd14c`).** Reuse the source halo across
  eight rows with 16-bit packed histogram counters, then compute the same four
  terms in parallel. The maximum window contains 33×33 samples, safely below
  the 16-bit limit. Shared storage is 16,004 bytes, below Vulkan's 16 KiB
  minimum. Directly against the four-lane implementation in the same sitting,
  CAMBI moved 0.344114 → 0.326416 ms at 1080p, 1.728283 → 1.635261 ms at
  4K, and 2.485006 → 2.382680 ms on dense 4K. Scores/checksums matched.
- **Fused 4K conversion and derivative (`c31ba05`).** Reuse a 17×17 tile to
  convert and form the derivative together, and remove the unused full-size
  horizontal scratch plane, saving 4 bytes per input pixel: 8,294,400 bytes
  at 1080p and 33,177,600 bytes at 4K. Compared directly with mask fusion alone,
  CAMBI moved 1.756274 → 1.698934 ms at 4K 10-bit and 1.766536 → 1.723154 ms
  at 4K 8-bit, scores/checksums identical. The new shared shader/layout moved
  1080p CAMBI 0.367355 → 0.368362 ms and 1440p 0.614782 → 0.616315 ms
  (+0.27% and +0.25%). The conversion fusion is therefore enabled only for at
  least 3840×2160 pixels; because the current CAMBI contrast geometry caps each
  dimension, this effectively selects exactly 3840×2160 in the full pipeline.
  An unconditional version regressed CAMBI at 2880×1620 (0.807813 → 0.817634
  ms) and 3200×1800 (0.994219 → 1.017933 ms). This cutoff is empirical for this
  GPU, not a guarantee for other devices. The preparation oracle exercises odd
  3841×2161 geometry directly, including 8-bit anti-dithering and 10-bit input.

Updated temporary diagnostic timestamps on the baseline put 1080p/4K CAMBI
contrast at 0.231461/1.158590 ms, preparation at 0.061230/0.465949 ms and
radix selection at 0.031709/0.067888 ms. These are diagnostic substage totals,
not independently additive claims about the optimized build.

## Validation and rejected work

The full 4,292-pair CID22 validation verified matching input hashes and had
**zero score drift at the benchmark's printed precision** versus `2941f74`.
KRCC/SRCC/PCC against MCOS were 0.56395105 / 0.76184506 / 0.71779543, versus
pinned libvmaf v1's 0.56396690 / 0.76186042 / 0.71780187. These are still
pairs with zero motion; moving-frame checks are in the GPU suite. GPU and
ASAN/UBSAN GPU suites passed 131 tests, CTest passed 10/10, formatting and
changed SPIR-V passed, and SSIMULACRA 2 / Butteraugli identities remained
100.00000000 / 0.00000000. The final shader count remains 92.

Fusing decimation with horizontal mode filtering did not meet the primary
CAMBI stage gate: 0.371360 → 0.366590 ms at 1080p (−1.28%) and 1.851190 →
1.854109 ms at 4K (+0.16%), although ducks improved 2.28%. Scores/checksums
matched. It was discarded. The eight-row packed tile alone passed but was
slower than the combined eight-row/four-lane tile on the primary workloads;
only the measured combination was retained. Unconditional conversion/derivative
fusion also regressed on the intermediate resolutions above, so it was not
retained as a universal path.
