# VMAF v1 histogram reuse and batched filters

This follows [the CAMBI optimization pass](vmaf-v1-cambi-optimization.md).
The baseline is `20a035e`; the combined implementation is `3a7c159`. Tests
ran on an RTX 5070 with NVIDIA driver 615.78.08 on 2026-11-02, using the
standard SDR v1.0.16 3H model. The 1080p and 4K primary inputs are 10-bit
YUV420. Ducks is one 8-bit frame with an x264 CRF 28 distortion, not Netflix's
full video workload. The 8-bit 4K, 1440p and constant-luma dense stress inputs
are described, with hashes, in [the previous report](vmaf-v1-cambi-optimization.md).

Each arm used its own build directory. Five runs per arm were interleaved in
one sitting. Resident uses `--iterations 20 --warmup 3 --resident 2`; streaming
uses `--iterations 8 --warmup 2`. Both images are fully prepared every call.
Resident inputs are preuploaded and repeat, so the result approximates a stream
of unique pairs in hot GPU memory; motion becomes zero after the first call.
Both modes exclude decoding. Ranges below are min–max of the five run averages,
not per-frame percentiles.

| workload | baseline ms/pair, median (range) | candidate ms/pair, median (range) | pairs/s before → after |
| --- | ---: | ---: | ---: |
| 1080p resident | 0.764289 (0.762914–0.772209) | 0.736785 (0.732564–0.741115) | 1308 → 1357 |
| 1440p resident | 1.193938 (1.193448–1.207323) | 1.149794 (1.141943–1.150724) | 838 → 870 |
| 4K resident | 2.906403 (2.891468–3.003726) | 2.736791 (2.708831–2.747535) | 344 → 365 |
| 4K 8-bit resident | 2.941533 (2.932487–2.944763) | 2.766095 (2.739236–2.769876) | 340 → 362 |
| ducks resident | 0.630611 (0.628691–0.630926) | 0.604851 (0.604381–0.606532) | 1586 → 1653 |
| dense 1080p resident | 0.862063 (0.860393–0.869588) | 0.823853 (0.823089–0.833643) | 1160 → 1214 |
| dense 4K resident | 3.635202 (3.630518–3.639188) | 3.289648 (3.284013–3.345787) | 275 → 304 |
| 1080p streaming | 1.324556 (1.322041–1.328586) | 1.294126 (1.289547–1.296886) | 755 → 773 |
| 4K streaming | 5.116117 (5.102596–5.140371) | 4.924144 (4.917334–4.937904) | 195 → 203 |
| ducks streaming | 0.905517 (0.897962–0.909547) | 0.882552 (0.876862–0.886447) | 1104 → 1133 |

All resident and streaming scores/checksums matched at printed precision. At
1080p the resident score/checksum was 95.973463 / 2207.389651, at 4K
96.065006 / 2209.495149, and on ducks 75.839083 / 1744.298908. Corresponding
streaming checksums were 959.734631, 960.650065 and 758.390830. Per-run logs,
all stage spreads, input/binary hashes and experimental patches are in the
ignored local directory `artifacts/reports/vmaf-more/`.

## Retained

- **Batch SpEED filter passes (`706f257`).** The four reference/distorted U/V
  filter chains use disjoint scratch and output banks and dispatch each of the
  four passes across the planes together. The first downsampling, second
  filtering and arithmetic order per plane are unchanged. Independent
  interleaved resident SpEED stage medians moved 0.111437
  (0.111370–0.111582) → 0.088922 (0.088707–0.089082) ms at 1080p and
  0.179701 (0.179258–0.180560) → 0.159966 (0.159762–0.160242) ms at 4K.
  Scores/checksums matched in all arms. For 4:2:0 input it adds 403,920 bytes
  of scratch at 1080p and 1,640,160 bytes at 4K. An exact GPU test compares
  four batched planes with independent calls at odd geometries, including a
  flat plane; the pinned-libvmaf, low-bit and near-singular tests also pass.
- **Reuse CAMBI histograms across rows (`3a7c159`).** When all eight output
  rows in a tile are active, build one histogram and derive subsequent rows by
  copying it and adding/removing the entering/leaving samples. Partially active
  tiles retain independent construction. This is an independent GPU mapping
  of the same vertical sliding principle used by pinned libvmaf; output bins
  remain exact integer counts. Direct-window GPU oracles cover 17/33 windows,
  odd edges, masks, mixed adjacent bins and fully active rows. Independent
  interleaved resident CAMBI medians moved 0.315259 → 0.309864 ms at 1080p and
  1.476667 → 1.300888 ms at 4K, with identical scores/checksums. To test the
  targeted contrast substage, separate builds with the same diagnostic GPU
  timestamps measured 0.187965 (0.187682–0.188118) → 0.182373
  (0.182339–0.182659) ms at 1080p and 0.948920 (0.946472–0.949426) →
  0.772589 (0.772112–0.773392) ms at 4K. The additional timestamps perturb
  execution, so overall throughput claims use the uninstrumented runs.

On the ducks pair the final CAMBI stage rose from 0.189062 to 0.192570 ms
(+1.86%), while SpEED fell from 0.109194 to 0.086670 ms and the overall pair
became faster. The isolated filter-batching comparison also showed the CAMBI
increase, but does not establish its cause; the arena layout changes with the
new scratch banks. An alignment-padding experiment reduced the ducks CAMBI
median 0.192530 → 0.190944 ms (−0.82%), but missed the 2% stage gate and was
not retained.

## Validation and rejected work

Full CID22 validation scored all 4,292 independent still pairs, verifying input
hashes. There was zero score drift at the benchmark's printed precision versus
`20a035e`. KRCC/SRCC/PCC against MCOS were 0.56395105 / 0.76184506 /
0.71779543, compared with pinned libvmaf v1's 0.56396690 / 0.76186042 /
0.71780187. The still-pair motion feature is zero; moving-frame, low-bit,
external-buffer and near-singular checks are covered separately. The release
and ASAN/UBSAN GPU suites each passed 132 tests; CTest passed 10/10.

The following were built in separate worktrees, tested, benchmarked, and not
retained. Their patches and raw logs are saved under `artifacts/reports/vmaf-more/`.

- **32-column rolling stripe.** Despite reusing histogram state without more
  shared memory, CAMBI regressed from 0.318698 to 0.357678 ms at 1080p and
  from 0.188766 to 0.224478 ms on ducks. The 4K measurements had wide spreads;
  the stable 1080p regressions already miss the gate. Scores/checksums matched.
- **Shared derivative halo for mask preparation.** GPU-timestamp diagnostics
  measured preparation 0.058283 (0.058218–0.058333) → 0.060533
  (0.060490–0.060557) ms at 1080p and 0.313488 (0.313304–0.313642) →
  0.341989 (0.341771–0.342067) ms at 4K. Fewer source reads did not translate
  to a speedup. Scores/checksums matched; the hardware cause was not measured.
- **Two-seed vertical reuse.** Seeding rows 0 and 4 in parallel failed to beat
  the retained single-seed version: CAMBI was 0.311670 → 0.309859 ms at 1080p,
  1.323770 → 1.341749 ms at 4K, and 1.908326 → 1.956134 ms on dense 4K.
  Scores/checksums matched. A separate streaming comparison also missed the gate.
- **Arena alignment padding.** Padding the extra filter scratch to preserve
  the old 256-byte relative placement improved CAMBI by less than 1% on the
  tested inputs, below the 2% targeted-stage gate; it was discarded.
