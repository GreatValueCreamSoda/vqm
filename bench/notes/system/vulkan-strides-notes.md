# Vulkan planar buffer strides: regression check

The per-plane Vulkan buffer layout is an input-compatibility feature, not an
optimization. Default packed inputs also use the new stride field, so both
metrics' existing paths were measured for regressions before accepting it.

On 2026-11-02, an RTX 5070 with driver 615.78.08 scored the same 3840x2160
RGB8 `testsrc2` pair (original against a contrast/brightness-adjusted copy).
The baseline was commit `72fcdea` in `/tmp/ss2-stride-baseline`; the candidate
was the separate worktree containing the stride implementation. Each arm used
its own release build of `ssimulacra2_bench`. Five runs alternated baseline and
candidate in one sitting. Resident arms used `--resident 2 --iterations 20
--warmup 3` and prepare both images on every call; streaming arms used
`--iterations 8 --warmup 2`.

Median milliseconds per pair, with the five-run range:

| Metric and mode | Baseline | Candidate |
| --- | ---: | ---: |
| SSIMULACRA 2 resident | 1.198262 (1.188322–1.200181) | 1.193196 (1.192847–1.202482) |
| SSIMULACRA 2 streaming | 3.401808 (3.394752–3.410798) | 3.397127 (3.392879–3.418582) |
| Butteraugli resident | 9.204512 (9.160761–9.271329) | 9.241501 (9.189445–14.033501) |
| Butteraugli streaming | 14.893304 (14.867558–14.919954) | 14.945746 (14.916738–14.958047) |

Stage medians and ranges, milliseconds per call:

| Metric, mode and stage | Baseline | Candidate |
| --- | ---: | ---: |
| SSIMULACRA 2 resident prepare | 0.421037 (0.420794–0.421576) | 0.425448 (0.424877–0.425552) |
| SSIMULACRA 2 resident scale | 0.681598 (0.681312–0.681850) | 0.682067 (0.681325–0.682541) |
| SSIMULACRA 2 streaming prepare | 0.427236 (0.427128–0.431256) | 0.431200 (0.430924–0.435360) |
| SSIMULACRA 2 streaming scale | 0.679608 (0.679124–0.680752) | 0.680096 (0.678788–0.680128) |
| Butteraugli resident opsin | 0.992512 (0.992274–0.992822) | 1.003938 (1.003872–1.168184) |
| Butteraugli streaming opsin | 1.998016 (1.997308–1.998256) | 2.019968 (2.019464–2.020468) |

Prepare and opsin rose roughly 1.0–1.2% in this sitting; median total pair
changes were less than 0.5%. The first candidate Butteraugli resident run was
an outlier (14.03 ms per pair and elevated times across several stages); it is
included in the ranges. No throughput improvement is claimed.

Every SSIMULACRA 2 run scored `92.437150`, with checksum `2126.054439` resident
and `924.371495` streaming. Every Butteraugli run scored `7.279592`, with
checksum `167.430606` resident and `72.795916` streaming. Raw stage tables,
CSVs and commands are under `/tmp/ss2-plane-comparison` and disappear on
reboot. Both metrics' padded-plane paths passed the Vulkan GPU integration
checks against equivalent packed inputs; those checks cover RGB8, RGB16,
float RGB and 10-bit YUV420 with alpha, including odd dimensions where valid.
They are correctness checks, not measurements of strided-buffer performance.

After integrating host views, the pipeline pool, the external-buffer
benchmark and semaphore support, the same five-run interleaved comparison
against `72fcdea` was repeated. Median core milliseconds per pair (full
range):

| Metric and mode | Baseline | Integrated |
| --- | ---: | ---: |
| SSIMULACRA 2 resident | 1.1962 (1.1893–1.2020) | 1.1952 (1.1916–1.2049) |
| SSIMULACRA 2 streaming | 3.4065 (3.3967–3.4133) | 3.4092 (3.3999–3.4126) |
| Butteraugli resident | 9.2472 (9.2254–9.2538) | 9.2518 (9.2365–9.2809) |
| Butteraugli streaming | 14.9246 (14.8904–14.9469) | 14.9442 (14.9184–14.9645) |

Resident SSIMULACRA 2 prepare medians were 0.4222 vs 0.4255 ms and
Butteraugli opsin medians 0.9934 vs 1.0049 ms. Scores and checksums matched
the earlier table. This is an existing-path regression check, not a claim of
speed improvement. Raw logs and stage tables are in
`/tmp/ss2-integrated-regression/` until reboot.
