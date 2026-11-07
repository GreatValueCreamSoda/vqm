# Organization regression check

Measured on an NVIDIA GeForce RTX 5070, driver 615.78.08. The baseline was
commit `b98a9eb`; the candidate was `6055742` for SSIMULACRA 2 and VMAF v1,
and `6de8ebc` for VMAF v0 and Butteraugli (before the final file moves and
comment-only formatting). These are regression checks, not speedup claims.

Each arm used the same predecoded YUV420 8-bit Y4M pair for its geometry. The
[data file](organization-regression-data.json) records input hashes, every
stage median and min/max, scores and checksums. Five runs per arm were
interleaved within each workload. Resident used `--iterations 20 --warmup 3
--resident 2`; streaming used `--iterations 8 --warmup 2`. Resident reuses
pixels in hot memory and Butteraugli caches the reference, as documented in
[the benchmark protocol](../../../AGENTS.md).

| Metric | Geometry | Mode | Baseline ms (min–max) | Candidate ms (min–max) |
| --- | --- | --- | ---: | ---: |
| ssimu2 | 1080 | resident | 0.4869 (0.4856–0.4901) | 0.4893 (0.4865–0.4924) |
| ssimu2 | 1080 | streaming | 1.0441 (1.0353–1.0485) | 1.0367 (1.0353–1.0452) |
| ssimu2 | 4k | resident | 1.8745 (1.8528–1.8812) | 1.8637 (1.8606–1.8748) |
| ssimu2 | 4k | streaming | 4.0662 (4.0607–4.0687) | 4.0633 (4.0621–4.0671) |
| butter | 1080 | resident | 2.5276 (2.5150–2.5426) | 2.5195 (2.5140–2.5406) |
| butter | 1080 | streaming | 4.3669 (4.3575–4.3787) | 4.3643 (4.3465–4.3807) |
| butter | 4k | resident | 11.0096 (10.9967–11.0354) | 11.0201 (10.9475–11.0310) |
| butter | 4k | streaming | 18.4881 (18.4533–18.4957) | 18.4657 (18.4351–18.5033) |
| vmaf-v0 | 1080 | resident | 0.5674 (0.5668–0.5683) | 0.5685 (0.5666–0.5708) |
| vmaf-v0 | 1080 | streaming | 1.1283 (1.1190–1.1310) | 1.1284 (1.1165–1.1306) |
| vmaf-v0 | 4k | resident | 2.3621 (2.3513–2.3668) | 2.3634 (2.3527–2.3751) |
| vmaf-v0 | 4k | streaming | 4.5598 (4.5404–4.5827) | 4.5572 (4.5533–4.5600) |
| vmaf-v1 | 1080 | resident | 0.7303 (0.7268–0.7435) | 0.7287 (0.7270–0.7419) |
| vmaf-v1 | 1080 | streaming | 1.2868 (1.2776–1.2911) | 1.2893 (1.2861–1.3017) |
| vmaf-v1 | 4k | resident | 2.7354 (2.7203–2.7364) | 2.7344 (2.6958–2.7408) |
| vmaf-v1 | 4k | streaming | 4.9154 (4.9113–4.9428) | 4.9245 (4.9107–4.9257) |

All paired scores and checksums matched at the output precision. The per-stage
measurements and observed spreads do not indicate a material performance
regression from the organization changes.

The reorganized SSIMULACRA 2 implementation matched the baseline to the printed
precision on 300 CID22 pairs. Its KRCC/SRCC/PCC against MCOS were
0.8145/0.9554/0.9433, within 0.005 of the Cloudinary arm. Its pairwise scores
still differ from Cloudinary (mean absolute difference 0.202585, maximum
0.524941); the correlation gate passes, while the optional strict parity gate
would not.

For VMAF v1, all 4,292 independent CID22 still pairs matched the previous VQM
scores to the reported precision. VQM's KRCC/SRCC/PCC were
0.563951/0.761845/0.717795 versus 0.563967/0.761860/0.717802 for libvmaf.
VMAF v0 also matched the previous VQM scores on all 4,292 pairs, with
KRCC/SRCC/PCC of 0.595376/0.794819/0.755195 versus
0.595373/0.794817/0.755194 for libvmaf. These still-image comparisons have zero
motion and do not validate temporal video behavior.
