# VMAF SDR input conversion

Measured on an NVIDIA GeForce RTX 5070, driver 615.78.08. Baseline was
`bf5e5cd`. Five runs per arm were interleaved. Resident uses `--resident 2
--iterations 20 --warmup 3`; streaming uses `--iterations 8 --warmup 2`.
Resident inputs are pre-uploaded and the same pixels are reused, with both
images converted on each call. Numbers are pair time in milliseconds,
median with minimum–maximum across runs.

## RGB conversion and chroma selection

These comparisons use the same PNG pair per geometry and explicitly select
the processing chroma. Both arms include conversion to 16-bit BT.709 limited
range. The different chroma sampling is a different scoring workload; the
resulting scores need not match.

| Geometry | Input | 4:2:0 ms (min–max) | 4:4:4 ms (min–max) |
| --- | --- | ---: | ---: |
| 1080 | resident | 0.7500 (0.7497–0.7582) | 0.8701 (0.8687–0.8782) |
| 1080 | streaming | 1.3119 (1.3113–1.3193) | 1.4324 (1.4299–1.4530) |
| 4k | resident | 2.7655 (2.7326–2.7707) | 3.2164 (3.2120–3.2224) |
| 4k | streaming | 4.9367 (4.9335–4.9457) | 5.4200 (5.3820–5.4446) |

## Native coded-YUV regression check

The same 8-bit YUV420 pair was scored in Native mode by the baseline and
candidate. Scores and checksums matched for every paired run. SSIMULACRA 2
and Butteraugli also use the corrected shared color decoder; their source
chroma is now loaded only with valid subsampled coordinates.

| Metric | Geometry | Input | Baseline ms (min–max) | Candidate ms (min–max) |
| --- | --- | --- | ---: | ---: |
| ssimulacra2 | 1080 | resident | 0.4896 (0.4883–0.4927) | 0.4570 (0.4547–0.4796) |
| ssimulacra2 | 1080 | streaming | 1.0399 (1.0367–1.0433) | 1.0081 (1.0065–1.0126) |
| ssimulacra2 | 4k | resident | 1.8724 (1.8583–1.8765) | 1.7704 (1.7688–1.7793) |
| ssimulacra2 | 4k | streaming | 4.0615 (4.0527–4.0709) | 3.9796 (3.9775–4.0169) |
| butteraugli | 1080 | resident | 2.5438 (2.5356–2.5463) | 2.4365 (2.4067–2.4413) |
| butteraugli | 1080 | streaming | 4.3850 (4.3729–4.4123) | 4.1740 (4.1630–4.1871) |
| butteraugli | 4k | resident | 10.9901 (10.9861–11.0004) | 10.5746 (10.5643–10.5834) |
| butteraugli | 4k | streaming | 18.4597 (18.4530–18.5456) | 17.6403 (17.5988–17.7202) |
| vmaf-v0 | 1080 | resident | 0.5707 (0.5687–0.5725) | 0.5700 (0.5682–0.5756) |
| vmaf-v0 | 1080 | streaming | 1.1304 (1.1273–1.1325) | 1.1322 (1.1304–1.1347) |
| vmaf-v0 | 4k | resident | 2.3630 (2.3425–2.3724) | 2.3579 (2.3433–2.3758) |
| vmaf-v0 | 4k | streaming | 4.5576 (4.5498–4.5759) | 4.5573 (4.5526–4.5594) |
| vmaf-v1 | 1080 | resident | 0.7293 (0.7267–0.7294) | 0.7299 (0.7277–0.7323) |
| vmaf-v1 | 1080 | streaming | 1.2896 (1.2884–1.2962) | 1.2896 (1.2884–1.2915) |
| vmaf-v1 | 4k | resident | 2.7341 (2.7121–2.7408) | 2.7336 (2.7052–2.7395) |
| vmaf-v1 | 4k | streaming | 4.9117 (4.9071–4.9230) | 4.9173 (4.9067–4.9327) |

The VMAF native path shows no material regression in these runs. For the YUV
workloads, removing the unnecessary chroma reads reduced the 4K resident
SSIMULACRA 2 prepare median from 1.0795 to 0.9905 ms and Butteraugli opsin
from 2.7785 to 2.3514 ms. These figures apply to this input and device.
[All rounds, stages, scores, checksums and input hashes](vmaf-sdr-input-results.json)
are recorded alongside this report.


## Accuracy on CID22

All 4,292 pairs were scored from their original RGB sources as independent
still-image sequences. The old corpus conversion used encoded sRGB directly
with a 10-bit BT.709 matrix and truncated RGB16 PNGs to 8 bits; it is a
different input convention, not a same-input parity reference. The new path
applies the documented display transfer and retains 16-bit input samples.

| VQM path | KRCC | SRCC | PCC |
| --- | ---: | ---: | ---: |
| Historical v0, YUV444 | 0.5954 | 0.7948 | 0.7552 |
| New v0, RGB to YUV444 | 0.5811 | 0.7779 | 0.7401 |
| Historical v1, YUV444 | 0.5640 | 0.7618 | 0.7178 |
| New v1, RGB to YUV444 | 0.5703 | 0.7694 | 0.7322 |
| New v1, RGB to YUV420 | 0.5927 | 0.7874 | 0.7553 |

V1 420 vs 444 changed the score by a mean absolute 2.7042 and maximum
17.2834 on this corpus. The choice changes the chroma supplied to SpEED and
should not be treated as a precision-only switch. The v0 feature set uses
luma only. These correlations are against CID22 MCOS, collected for
SSIMULACRA 2, and do not validate temporal VMAF.

For a same-conversion reference check, the first 300 pairs were independently
converted on the CPU to 16-bit BT.709 YUV444, preserving 8- and 16-bit source
PNG precision, then scored with pinned libvmaf. Mean/max absolute score
differences were 0.000224/0.008688 for v0 and 0.001520/0.002638 for v1.
Corresponding VQM and libvmaf KRCC/SRCC/PCC were 0.63254/0.83062/0.79771
for both v0 implementations and 0.61234/0.80305/0.80511 vs
0.61238/0.80312/0.80513 for v1. This sample does not establish a general
error bound. CPU uses float64 and GPU uses FP32 for conversion; their rounding
can differ at code boundaries.

Reproduce with the pinned reference build and CID22 data:

```sh
python3 validation/vmaf_sdr_cid22.py --jobs 8
```

The script uses FFmpeg to decode 16-bit PNGs, verifies corpus/score provenance,
and stores individual scores and hashes in the report directory.
