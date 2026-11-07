# VMAF v0 optimization measurements

Measured on an RTX 5070, driver 615.78.08. Each arm uses five interleaved
runs and the benchmark's per-stage timings; scores and checksums were read on
every run. Resident runs used 20 iterations, 3 warm-ups and `--resident 2`;
streaming used 8 iterations and 2 warm-ups. Both input images are prepared
on every call. Inputs are 10-bit YUV420; resident mode reuses their pixels.

In one initial 1080p comparison of the same input pair, SSIMULACRA 2 took
0.4898 ms/pair (range 0.4876–0.4905) and VMAF v0 took 0.7394 ms/pair
(0.7380–0.7409); the latter's VIF stage took 0.4369 ms. These metrics perform
different work, including VMAF's temporal reference processing.

## Combined result

After the changes below, five interleaved runs of the original commit
`c96b5f7` and optimized build, on the same 10-bit YUV420 pair, measured:

| workload | baseline ms/pair | optimized ms/pair | change |
| --- | ---: | ---: | ---: |
| 1080p resident | 0.739624 (0.736514–0.753383) | 0.566752 (0.565387–0.575917) | −23.4% time; +30.5% pairs/s |
| 4K resident | 3.387789 (3.367035–3.440488) | 2.356950 (2.346770–2.370469) | −30.4% time; +43.7% pairs/s |
| 1080p streaming | 1.290426 (1.287471–1.300455) | 1.122439 (1.118588–1.130887) | −13.0% time; +15.0% pairs/s |

The same sitting measured SSIMULACRA 2 at 0.490043 ms/pair
(0.488218–0.498148) on the 1080p resident pair. VMAF still took 15.7% more
time on this workload. At 1080p VIF moved 0.436896 → 0.308318 ms, ADM
0.120851 → 0.099485 ms and motion 0.079501 → 0.062238 ms. V0's resident
score/checksum changed 92.424365 / 2125.760403 → 92.424390 / 2125.760981;
at 4K it was 92.309458 / 2123.117545 → 92.309458 / 2123.117532.

V1 shares the motion pass. In the same sitting its 1080p resident median was
7.419243 → 7.419368 ms, with matching score/checksum. Its motion stage improved
0.087330 → 0.073088 ms. Timings are GPU- and content-specific.

The complete optimized build was also scored on all 4,292 CID22 pairs using
the exact same converted input hashes as the original build. Pairwise score
drift was maximum 0.00064234, mean 0.00002073 and p99 0.00017896. Against
pinned float libvmaf, maximum/mean absolute error was 0.017094 / 0.000195;
KRCC/SRCC/PCC were 0.59537624 / 0.79481890 / 0.75519545 versus the reference's
0.59537254 / 0.79481730 / 0.75519364. CID22 is an independent still-image
workload with zero temporal motion, not a general video error bound.

## Filter only consumed downsample rows

Only even rows of the vertical downsample result are used by the horizontal
pass. Computing those rows alone preserves the existing per-pixel arithmetic
and leaves scratch addresses unchanged. Interleaved baseline and candidate:

| workload | pair ms baseline → candidate | VIF ms baseline → candidate |
| --- | ---: | ---: |
| 1080p resident | 0.7389 (0.7378–0.7441) → 0.7087 (0.7071–0.7163) | 0.4369 → 0.4080 |
| 4K resident | 3.3909 (3.3615–3.4223) → 3.2665 (3.2360–3.2738) | 2.0694 → 1.9451 |
| 1080p streaming | 1.3019 (1.2923–1.3051) → 1.2705 (1.2606–1.2971) | 0.4364 → 0.4071 |

Scores and checksums were identical to the displayed precision in all arms;
the 118-check GPU suite also passed. The 1080p resident score/checksum was
92.424365 / 2125.760403; the 4K resident values were 92.309458 / 2123.117545.
The 1080p files have the hashes in [`vmaf-notes.md`](vmaf-notes.md); 4K inputs
were made by converting the local `/tmp/uhd_orig.png` and `/tmp/uhd_dist.png`
to YUV420P10LE without rescaling.

## Specialize VIF kernels by scale

Compile the four fixed filter widths and coefficient tables into separate
vertical and horizontal shader variants. Their arithmetic and accumulation
order are unchanged in source; shader compilation changes FP32 rounding. On
top of the downsample-row change, five interleaved runs gave:

| workload | pair ms baseline → specialized | VIF ms baseline → specialized |
| --- | ---: | ---: |
| 1080p resident | 0.7083 (0.7080–0.7096) → 0.6695 (0.6690–0.6720) | 0.408154 (0.407954–0.408192) → 0.368517 (0.368381–0.368878) |
| 4K resident | 3.3120 (3.2408–3.3315) → 3.1305 (3.0969–3.1430) | 1.945362 (1.942829–1.946933) → 1.808795 (1.808282–1.809922) |
| 1080p streaming | 1.2722 → 1.2268 | 0.407200 → 0.367800 |
| 4K streaming | 5.4600 → 5.3224 | 1.946628 → 1.807032 |

The 1080p resident score/checksum changed from 92.424365 / 2125.760403 to
92.424575 / 2125.765219; at 4K they changed from 92.309458 / 2123.117545 to
92.309525 / 2123.119083. All 118 GPU tests passed. Across all 4,292 CID22
pairs, score drift versus the prior VQM build was max 0.002237, mean 0.0000973
and p99 0.000608. Versus pinned float libvmaf, max/mean absolute error was
0.017228 / 0.0001874; KRCC/SRCC/PCC were 0.59537211 / 0.79481782 /
0.75519609 versus 0.59537254 / 0.79481730 / 0.75519364 for libvmaf.

The full specialization tested without the downsample-row change was slower
at 4K: VIF 2.069309 (2.068984–2.070554) → 2.120546 ms
(2.119970–2.121270), though it reduced 1080p 0.436818 → 0.412272 ms.
The interaction matters: specialization was adopted only with the preceding
downsample-row change. Specializing only scale 0 on that baseline gave
0.386614 ms at 1080p and 1.885310 ms at 4K with identical printed scores;
specializing scales 1–3 gave 0.390296 and 1.869392 ms and accounted for the
score shift.

Replacing libvmaf's polynomial logarithm with the GPU's native `log2` did not
meet the speed gate: VIF was 0.436818 → 0.435270 ms at 1080p and
2.069309 → 2.064178 ms at 4K. These results were measured separately before
the downsample-row change and the candidate was reverted.

## Share input rows in the VIF vertical statistics pass

Each workgroup stages its reference and distorted input rows and halo once in
shared memory. Downsample filtering retains its even-row path. Five
interleaved runs on top of specialization gave:

| workload | pair ms baseline → tiled | VIF ms baseline → tiled |
| --- | ---: | ---: |
| 1080p resident | 0.671810 (0.668560–0.676380) → 0.651635 (0.650315–0.662230) | 0.368786 (0.368496–0.369170) → 0.349040 (0.349019–0.349435) |
| 4K resident | 3.129049 (3.112703–3.154347) → 2.852258 (2.843687–2.862192) | 1.806992 (1.806781–1.808250) → 1.539872 (1.539242–1.540414) |
| 1080p streaming | 1.227727 → 1.211862 | 0.367544 → 0.347856 |
| 4K streaming | 5.309856 → 5.039735 | 1.806452 → 1.538848 |

At 1080p, score/checksum changed from 92.424575 / 2125.765219 to 92.424579 /
2125.765313; at 4K both reported 92.309525 / 2123.119083. All 118 GPU tests
passed. Across all 4,292 CID22 pairs, max/mean/p99 score drift versus the
specialized baseline was 0.0003022 / 0.00002144 / 0.0000857; max/mean absolute
error against libvmaf was 0.017237 / 0.0001884. KRCC/SRCC/PCC were
0.59537189 / 0.79481770 / 0.75519640.

A separate fused scale-0 vertical and horizontal pass computed five moments
in a shared 16×32 tile, but performed redundant vertical filtering for each
horizontal halo. It was rejected: on the specialized baseline, VIF at 1080p
was 0.368816 ms (0.368130–0.368952) versus 0.501120 ms
(0.501027–0.501234), and at 4K 1.808765 ms (1.807659–1.810458) versus
2.319502 ms (2.318115–2.319928). Scores and checksums matched to printed
precision in all five runs: 92.424575 / 2125.765219 at 1080p and 92.309525 /
2123.119083 at 4K. The GPU test suite passed.

Staging a full 32×32 source tile and its halo removes those redundant global
loads. The fused 16×16 output tile uses 20 KiB of shared memory; devices with
less use the specialized two-pass implementation. Five interleaved runs on
the current main tree gave:

| workload | pair ms baseline → fused | VIF ms baseline → fused |
| --- | ---: | ---: |
| 1080p resident | 0.610111 (0.607580–0.611416) → 0.592410 (0.591431–0.595841) | 0.348421 (0.348072–0.348720) → 0.331290 (0.331160–0.331328) |
| 4K resident | 2.623700 (2.620080–2.630095) → 2.478473 (2.452313–2.491187) | 1.539658 (1.538819–1.540310) → 1.394736 (1.394520–1.395534) |
| 1080p streaming | 1.158473 → 1.144047 | 0.347744 → 0.331700 |
| 4K streaming | 4.812372 → 4.670248 | 1.538332 → 1.393684 |

All 120 GPU tests passed, including the forced fallback and both sides of the
20 KiB capability boundary. Scores/checksums were identical at printed
precision: 92.424579 / 2125.765313 at 1080p resident and 92.309525 /
2123.119083 at 4K resident. Across all 4,292 CID22 pairs the fused and prior
scores matched to eight decimal places; max/mean absolute error against
libvmaf was 0.017237 / 0.0001884.

Staging only horizontal filtered intermediates in a 16×32 shared tile on the
vertical-tiled baseline did not meet the 2% gate: VIF was 0.349155 ms
(0.349098–0.349261) → 0.346922 ms (0.346798–0.347128) at 1080p and
1.540173 ms (1.539690–1.540618) → 1.530762 ms (1.530085–1.531034) at 4K.
The 1080p score/checksum changed from 92.424579 / 2125.765313 to 92.424545 /
2125.764544; at 4K they changed from 92.309525 / 2123.119083 to 92.309477 /
2123.117980. The candidate was reverted.

Changing the fused output from 16×16 to 32×8 reduced its shared allocation
but did not meet the speed gate: VIF at 1080p was 0.331222 ms
(0.329882–0.331286) → 0.328670 ms (0.328587–0.328803), while 4K worsened
from 1.394141 ms (1.393701–1.394770) to 1.443030 ms
(1.442886–1.443710). The 1080p score/checksum was unchanged at 92.424579 /
2125.765313; 4K score was 92.309525, checksum 2123.119083 → 2123.119082.
All 120 GPU tests passed, but the candidate was reverted.

## Fuse the motion blur

A shared-memory tile evaluates the existing vertical and horizontal five-tap
filters in one dispatch and reduces motion differences in the same workgroup.
The separate vertical output is no longer written or read. Five interleaved
runs, baseline → fused:

| workload | pair ms | motion ms |
| --- | ---: | ---: |
| 1080p resident | 0.7451 → 0.7358 | 0.07950 (0.07934–0.07966) → 0.06904 (0.06898–0.06905) |
| 4K resident | 3.3812 → 3.2973 | 0.39126 (0.39097–0.39150) → 0.29733 (0.29695–0.29747) |
| 1080p streaming | 1.3013 → 1.2880 | 0.07948 → 0.06907 |
| 4K streaming | 5.5921 → 5.4950 | 0.39146 → 0.29712 |

Scores and checksums matched to printed precision. The GPU suite passed,
including v0/v1 sequence oracles and added small, odd and tile-edge geometries.

## Fuse ADM wavelet and decoupling

Compute the reference and distorted wavelets and their decoupling in one
dispatch per scale, retaining the reference detail bands and avoiding the
distorted detail round trip. With the row-skipping and fused-motion changes
already present, five interleaved runs gave:

| workload | pair ms baseline → fused | ADM ms baseline → fused |
| --- | ---: | ---: |
| 1080p resident | 0.697554 (0.696895–0.698584) → 0.665490 (0.664755–0.666790) | 0.119219 (0.119208–0.119411) → 0.098406 (0.098258–0.098669) |
| 4K resident | 3.174082 (3.149593–3.231007) → 3.030725 (2.999745–3.086923) | 0.646715 (0.646222–0.647019) → 0.512541 (0.512269–0.512664) |
| 1080p streaming | 1.260846 (1.252046–1.269176) → 1.227992 (1.221652–1.242597) | 0.119564 → 0.098620 |

Scores and checksums matched to printed precision; 119 GPU tests passed.

## Rejected ADM wavelet variants

Combining the two images' wavelet passes into one dispatch improved 1080p
resident ADM from 0.120850 ms (0.120738–0.120859) to 0.113018 ms
(0.112886–0.113102), but 4K moved only 0.646611 ms (0.646224–0.646870) to
0.639682 ms (0.639168–0.639851), below the 2% gate. A 34×34 shared-memory
tile with that paired dispatch measured 1080p 0.120862 → 0.112966 ms, but 4K
0.646738 → 0.650733 ms. Both variants were reverted; their scores and
checksums matched baseline at printed precision. All values are five-run
interleaved medians; 1080p and 4K resident scores/checksums were respectively
92.424365 / 2125.760403 and 92.309458 / 2123.117545.

## Rejected symmetric motion tap pairing

Pairing symmetric neighbors changes arithmetic order but saved less than 0.3%
of the fused motion stage: five-run 1080p resident 0.069067 ms
(0.068987–0.069075) → 0.068893 ms (0.068832–0.068922), and 4K resident
0.297211 ms (0.297030–0.297429) → 0.296605 ms (0.296542–0.296718).
1080p pair time was 0.698110 → 0.698134 ms. Scores and checksums matched to
printed precision and 119 GPU checks passed, but the candidate missed the
2% gate and was reverted without a broader drift study.

## Fuse VIF downsampling

For scales 1–3, a workgroup loads a shared source tile, performs the vertical
filter, then the horizontal filter and 2× decimation without writing an
intermediate plane. The 16×8 output tile uses at most 10,240 bytes of shared
memory. The capability check is independent of scale 0's larger tile; it
retains the prior two-pass downsample as a fallback.

Five interleaved runs against the already-fused scale-0 baseline gave:

| workload | pair ms baseline → fused | VIF ms baseline → fused |
| --- | ---: | ---: |
| 1080p resident | 0.592961 (0.590602–0.596526) → 0.565216 (0.564382–0.566951) | 0.331216 (0.330134–0.331355) → 0.308272 (0.308173–0.308333) |
| 4K resident | 2.467408 (2.448773–2.479202) → 2.338334 (2.332615–2.350449) | 1.394944 (1.394110–1.395584) → 1.280669 (1.280101–1.281301) |
| 1080p streaming | 1.142842 (1.142043–1.149828) → 1.125368 (1.116698–1.126543) | 0.331640 → 0.308456 |

1080p resident score/checksum shifted from 92.424579 / 2125.765313 to
92.424390 / 2125.760981; 4K shifted from 92.309525 / 2123.119083 to
92.309458 / 2123.117532. Across all 4,292 CID22 pairs, absolute score drift
versus the prior VQM build was max 0.002211, mean 0.0000945, p99 0.000576;
versus pinned float libvmaf it was max 0.017094 and mean 0.0001948. The
GPU suite passed 120 checks, including odd sizes and both the all-fallback
and 16 KiB limit (scale-0 fallback with fused downsampling) paths.

A 16×16 downsample variant required 17,920 bytes of shared memory and a
hardware-gated fallback; it measured VIF 0.307254 ms at 1080p and 1.280376 ms
at 4K versus 0.308272 and 1.280669 ms for the 16×8 variant. The smaller tile
was selected for its 10 KiB footprint and practically equal performance.

## Remove unused full-resolution buffers

The fused motion pass does not need its former scratch plane. Removing it
saves 8,294,400 bytes per pipeline at 1080p and 33,177,600 bytes at 4K for
both v0 and v1. Five interleaved runs of this allocation-only change gave:

| workload | pair ms baseline → reduced arena |
| --- | ---: |
| v0 1080p resident | 0.697480 (0.697104–0.698500) → 0.697715 (0.697179–0.705379) |
| v0 4K resident | 3.196933 (3.150088–3.228796) → 3.171218 (3.165378–3.196962) |
| v0 1080p streaming | 1.252617 (1.250541–1.270991) → 1.254277 (1.250822–1.263742) |
| v0 4K streaming | 5.364265 (5.352726–5.401099) → 5.381970 (5.348326–5.411700) |
| v1 1080p resident | 7.373540 (7.335306–7.421435) → 7.381505 (7.319275–7.405479) |
| v1 1080p streaming | 7.902132 (7.863818–7.927107) → 7.927122 (7.913882–7.929977) |

The fused ADM2 wavelet and decoupling pass also no longer writes or reads the
distorted detail planes. Removing those three planes saves a further 6,220,800
bytes at 1080p and 24,883,200 bytes at 4K for v0; v1 ADM3 still needs its
distorted detail planes. Five interleaved runs gave:

| v0 workload | pair ms baseline → reduced arena |
| --- | ---: |
| 1080p resident | 0.628761 (0.626101–0.632650) → 0.631935 (0.626630–0.640425) |
| 4K resident | 2.892050 (2.863800–2.898945) → 2.890016 (2.874381–2.901165) |
| 1080p streaming | 1.179432 (1.174692–1.191461) → 1.186972 (1.174707–1.190482) |
| 4K streaming | 5.070187 (5.059377–5.101362) → 5.077021 (5.067952–5.093331) |

The 1080p resident ADM stage shifted from 0.098402 to 0.099339 ms and 4K
from 0.512426 to 0.513128 ms. These allocation changes were measured as
regression checks rather than speed claims; observed pair timings overlapped
in all arms. Scores and checksums matched to printed precision, and the GPU
suite passed 119 tests for each candidate.

The fused scale-zero VIF path also needs fewer full-resolution scratch planes.
Sizing the arena for two planes on devices that support the fused shader,
while retaining five on the fallback path, saves a further 24,883,200 bytes
at 1080p or 99,532,800 at 4K. Five interleaved no-regression runs after
downsample fusion measured 1080p resident 0.566241 → 0.566032 ms and 4K
2.366200 → 2.368379 ms; scores/checksums matched. Exact-size canaries and
both the fused and forced-fallback paths passed the GPU suite.

## Rejected symmetric VIF taps

Pairing equal-weight symmetric samples halves some multiplications but changes
the floating-point accumulation order. For the vertical statistics pass
before the scale-zero fusion, five interleaved runs produced:

| workload | pair ms baseline → paired | VIF ms baseline → paired |
| --- | ---: | ---: |
| 1080p resident | 0.607486 (0.606180–0.609125) → 0.598881 (0.597841–0.602606) | 0.348659 (0.348301–0.348894) → 0.338984 (0.338915–0.339346) |
| 4K resident | 2.618876 (2.606495–2.630955) → 2.599885 (2.584035–2.611824) | 1.540080 (1.538394–1.540592) → 1.518002 (1.517904–1.523891) |
| 1080p streaming | 1.165917 (1.159162–1.172267) → 1.163123 (1.158973–1.187872) | 0.348104 (0.347364–0.348168) → 0.338908 (0.338812–0.339400) |

1080p resident score/checksum changed from 92.424579 / 2125.765313 to
92.424603 / 2125.765864; at 4K it changed from 92.309525 / 2123.119083 to
92.309577 / 2123.120260. Streaming values changed from 92.424579 / 924.245788
to 92.424603 / 924.246028. Although all 119 GPU tests passed, 4K VIF improved
only 1.43%, below the 2% gate, so the candidate was reverted.

Pairing both the vertical and horizontal taps inside the fused scale-zero
shader also failed the speed gate:

| workload | pair ms baseline → paired | VIF ms baseline → paired |
| --- | ---: | ---: |
| 1080p resident | 0.591421 (0.590476–0.593896) → 0.591411 (0.589801–0.605246) | 0.331274 (0.331192–0.331358) → 0.330074 (0.330005–0.330302) |
| 4K resident | 2.464348 (2.444283–2.487398) → 2.476047 (2.445588–2.479538) | 1.394702 (1.393939–1.394987) → 1.387629 (1.387432–1.388190) |
| 1080p streaming | 1.144897 (1.141638–1.153487) → 1.153827 (1.142378–1.155507) | 0.331540 (0.331188–0.331672) → 0.330332 (0.330304–0.330492) |

1080p resident score/checksum changed from 92.424579 / 2125.765313 to
92.424580 / 2125.765330; at 4K it changed from 92.309525 / 2123.119083 to
92.309527 / 2123.119126. Streaming values changed from 92.424579 / 924.245788
to 92.424580 / 924.245796. All 120 GPU tests passed, including low-variance and
fallback checks, but 4K VIF improved only 0.51%. The candidate was reverted
without a broader corpus run. All timings above are five-run interleaved
medians on the same RTX 5070 and YUV420P10LE inputs described in this document.

## Copy only luma for native streaming inputs

Native v0 consumes coded luma only, but the streaming path copied every source
plane. A per-image plane mask now restricts the host fill and staged GPU copy to
plane zero when the sequence uses native coding; explicit conversion still copies
all planes. Five interleaved streaming runs used 8 iterations and 2 warm-ups:

| workload | baseline ms/pair | luma-only ms/pair | change |
| --- | ---: | ---: | ---: |
| 1080p 4:2:0 | 1.123031 | 0.940458 | -16.3% |
| 4K 4:2:0 | 4.565604 | 3.843888 | -15.8% |
| 1080p 4:4:4 | 1.673397 | 0.894452 | -46.5% |

The 4:2:0 saves one third of the source bytes; 4:4:4 saves two thirds.
Scores and checksums matched exactly in every arm: 92.42439050 / 924.243905
at 1080p and 92.30945792 / 923.094579 at 4K. This does not change resident
work, and the full GPU suite passed.

## Widen the fused scale-zero VIF tile

The fused scale-zero output tile changed from 16x16 to 32x16. Its source tile
grows from 32x32 to 48x32, reducing global source samples per output from four
to three. Shared storage rises from 20,480 to 29,696 bytes; the 16x16 fallback
remains available below that limit. The vertical moment pass writes all 48
source columns, then each thread evaluates two output columns. Five interleaved
resident runs on top of the luma-only change gave:

| workload | baseline pair ms | candidate pair ms | VIF baseline | VIF candidate |
| --- | ---: | ---: | ---: | ---: |
| 1080p | 0.568734 | 0.531239 | 0.309875 | 0.269517 |
| 4K | 2.373711 | 2.257256 | 1.285558 | 1.167744 |

Pair time fell 6.6% at 1080p and 4.9% at 4K; VIF fell 13.0% and 9.2%.
Scores and checksums matched exactly for the two benchmark pairs, and all 173
GPU checks passed, including odd geometry and the forced fused/fallback paths.
Across all 4,292 CID22 still pairs, drift from the pre-change tree was maximum
3.6e-7 and mean 4.7e-8; parity against pinned libvmaf stayed maximum 0.017094
and mean 0.0001948.

Combined with the streaming-copy change, the original baseline versus the
retained tree measured 1080p resident 0.567157 -> 0.527463 ms (-7.0%), 4K
resident 2.374293 -> 2.231800 ms (-6.0%), 1080p streaming 1.119210 ->
0.895687 ms (-20.0%), and 4K streaming 4.554558 -> 3.697295 ms (-18.8%).
All scores and checksums remained identical.

## Fuse VIF statistics for scales 1-3

The two-pass statistics path wrote five full scratch planes from the vertical
pass and then read them again in the horizontal pass. A scale-specialized fused
shader now stages both source planes and computes the five vertical moments in
shared memory. Its 16x16 output tile uses a 24x24 source tile at scale 1, 20x20
at scale 2 and 18x18 at scale 3, for 13,312, 10,624 and 9,376 bytes of shared
storage. The existing two-pass path remains the fallback.

Five interleaved runs against commit 81fd48d gave:

| workload | baseline pair ms | fused pair ms | VIF baseline | VIF fused |
| --- | ---: | ---: | ---: | ---: |
| 1080p resident | 0.535253 | 0.512263 | 0.269426 | 0.250638 |
| 4K resident | 2.381317 | 2.141425 | 1.211384 | 1.111490 |
| 1080p streaming | 0.907262 | 0.880598 | | |
| 4K streaming | 4.008978 | 3.746372 | | |

Resident pair time fell 4.3% at 1080p and 10.1% at 4K; VIF fell 7.0% and
8.2%. Streaming pair time fell 2.9% and 6.6%; a 1080p 4:4:4 streaming check
fell 0.902407 -> 0.874043 ms (-3.1%). The resident 1080p score/checksum changed
92.42439050 / 2125.760981 -> 92.42436537 / 2125.760403; 4K changed
92.30945792 / 2123.117532 -> 92.30945849 / 2123.117545.

Across all 4,292 CID22 still pairs, drift from 81fd48d was maximum 0.000642,
mean 0.0000207 and p99 0.0001784. Against pinned libvmaf, maximum error stayed
0.017094 and mean moved 0.0001948 -> 0.0001957. KRCC/SRCC/PCC were
0.59537645 / 0.79481995 / 0.75519605 versus libvmaf's 0.59537254 / 0.79481730
/ 0.75519364. All 173 GPU checks, including the two-pass and mixed fallback
paths, passed.

## Disable temporal weighting

`VmafConfig::temporal_weighting = false`, exposed by
`--vmaf-no-temporal`, forces motion2 to zero, skips the motion blur and
reduction dispatches, and removes the one-frame temporal look-ahead. Five
interleaved resident runs (`--iterations 20 --warmup 3 --resident 2`) on native
10-bit YUV420 inputs measured:

| workload | temporal pairs/s | no-temporal pairs/s | change |
| --- | ---: | ---: | ---: |
| 1080p | 1979.3 | 2281.3 | +15.3% |
| 4K | 466.7 | 545.1 | +16.8% |

The motion stage was 11.6% of the 1080p pair and 13.9% of the 4K pair in the
same sitting. Both benchmark inputs are still pairs, so their reference
transition and motion2 score are already zero and no score change is expected;
the sequence tests cover the changed prediction when nonzero motion is
supplied. This mode is a derived VQM extension, not libvmaf behavior, and does
not claim parity with the upstream score.

## Wider VIF windows, packed luma and tiled ADM

Five interleaved resident runs against commit `7eb305e` on native 10-bit
YUV420 measured the combined raw-compute changes:

| workload | baseline pairs/s | optimized pairs/s | change |
| --- | ---: | ---: | ---: |
| 1080p temporal | 1979.6 | 2499.0 | +26.2% |
| 4K temporal | 472.8 | 589.2 | +24.6% |

At 4K, the stage medians moved from luma 0.19, VIF 1.08, ADM 0.51 and motion
0.29 ms to luma 0.15, VIF 0.70, ADM 0.49 and motion 0.29 ms. With temporal
weighting disabled, the optimized build measured 3005.4 pairs/s at 1080p and
715.1 pairs/s at 4K.

The retained changes are:

- VIF scale 0 computes two adjacent outputs per thread and loads their
  18-sample horizontal union once. Native 8-12-bit luma stores the source tile
  as packed FP16, which is exact for those input depths; 16-bit, SDR-converted
  and fit/pad paths use the FP32 source-tile variant.
- VIF scales 1-3 use the same two-adjacent-output schedule with a 32x16 tile.
  Scale 1's fused statistics block grows to 22,528 bytes and keeps the 16x16
  two-pass fallback below that limit.
- Luma conversion processes four 8-bit or two 16-bit samples per thread.
- ADM2's v0 decouple stages a 34x34 source tile per image, cutting repeated
  global reads. ADM3 retains the untiled shared shader.

The 4K score/checksum changed from 92.30945849 / 2123.117545 to
92.30947114 / 2123.117836; all 185 GPU checks and the reference gates passed.

Rejected during this pass: direct VIF source loads without the shared tile,
four adjacent VIF outputs per thread, a fused ADM decouple/contrast-mask pass,
staging ADM's additive neighborhood, FP16 ADM intermediate planes, and fusing
native luma conversion into VIF scale 0. Each was neutral or slower. Suppressing
all VIF reductions saved only 0.02 ms, so reduction fusion is not a path to the
750 fps target. The exact native-4K model remains below 750 fps: 715.1 fps with
temporal weighting disabled. The fit/pad 1080 predictor reaches 1038.9 fps on the
same 4K pair, but it is the 1080p model and produces a different score.

## Rejected reduction epilogue

Deferring the four VIF reductions into one dispatch was rejected in both forms.
One workgroup looping over all four scales measured pair time 0.528943 ->
0.531418 ms at 1080p and 2.246250 -> 2.338309 ms at 4K. Four workgroups in one
dispatch measured 0.527423 -> 0.527443 ms at 1080p and 2.235620 -> 2.310029 ms
at 4K. Per-scale partials were removed with the experiment; scores and checksums
matched. Running motion before VIF and ADM measured 0.527568 -> 0.523398 ms at
1080p and 2.233975 -> 2.245424 ms at 4K, so it was also reverted.
