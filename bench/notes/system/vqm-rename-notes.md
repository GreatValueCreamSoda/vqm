# VQM rename: regression check

On 2026-11-02, an RTX 5070 with driver 615.78.08 scored the same synthetic
3840x2160 RGB8 pair used in [the earlier regression check](publication-regression-notes.md).
The baseline was commit `8c405c5` in a separate release worktree; the candidate
was the VQM rename. Five runs per arm alternated in one sitting. Resident runs
used `--resident 2 --iterations 20 --warmup 3`; streaming runs used
`--iterations 8 --warmup 2`. Times are milliseconds per pair, shown as median
(minimum–maximum).

| Metric and mode | Baseline | VQM |
| --- | ---: | ---: |
| SSIMULACRA 2 resident | 1.1935 (1.1902–1.1961) | 1.1928 (1.1874–1.1995) |
| SSIMULACRA 2 streaming | 3.4138 (3.4005–3.4266) | 3.4113 (3.3983–3.4150) |
| Butteraugli resident | 9.2381 (9.2355–9.2524) | 9.2265 (9.2149–9.2623) |
| Butteraugli streaming | 14.9333 (14.8862–14.9535) | 14.9230 (14.9071–14.9448) |

Median GPU stages in milliseconds per call were prepare 0.4240 vs 0.4243 and
scale 0.6790 vs 0.6794 for SSIMULACRA 2 resident; prepare 0.4344 vs 0.4343
and scale 0.6791 vs 0.6786 for streaming. Butteraugli opsin was 1.0029 vs
1.0031 resident and 2.0174 vs 2.0174 streaming. No speedup is claimed.

Every SSIMULACRA 2 run scored `92.437150`, with checksum `2126.054439`
resident and `924.371495` streaming. Every Butteraugli run scored `7.279592`,
with checksum `167.430606` resident and `72.795916` streaming. Checksums include
warmups. All per-run logs and stage CSVs are under `/tmp/vqm-rename-regression`
until reboot.
