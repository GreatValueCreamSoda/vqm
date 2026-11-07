# Scoring and benchmark tooling: regression check

On 2026-11-02, an RTX 5070 with driver 615.78.08 scored the same synthetic
3840x2160 RGB8 pair as in [the Vulkan stride regression check](vulkan-strides-notes.md).
The baseline was commit `ec49b04` in a separate release worktree; the candidate
included the scoring fixes and the resident-ring benchmark. Five runs per arm
were interleaved in one sitting. Resident runs used `--resident 2 --iterations
20 --warmup 3`; streaming runs used `--iterations 8 --warmup 2`. Times below are
milliseconds per pair, reported as median (minimum–maximum).

| Metric and mode | Baseline | Candidate |
| --- | ---: | ---: |
| SSIMULACRA 2 resident | 1.1980 (1.1892–1.2008) | 1.1979 (1.1851–1.1999) |
| SSIMULACRA 2 streaming | 3.4103 (3.4018–3.4299) | 3.4092 (3.3970–3.4149) |
| Butteraugli resident | 9.2458 (9.2034–9.2649) | 9.2384 (9.1850–9.4199) |
| Butteraugli streaming | 14.9263 (14.9197–14.9606) | 14.9385 (14.9163–14.9703) |

Median GPU stages, in milliseconds per call, were prepare 0.4243 vs 0.4242
and scale 0.6799 vs 0.6791 for SSIMULACRA 2 resident; prepare 0.4348 vs
0.4349 and scale 0.6810 vs 0.6801 for SSIMULACRA 2 streaming. Butteraugli
resident opsin was 1.0034 vs 1.0035 and streaming opsin 2.0190 vs 2.0181.
All per-stage results are in the logs; no speedup is claimed.

Every SSIMULACRA 2 run scored `92.437150` with checksum `2126.054439`
resident and `924.371495` streaming. Every Butteraugli run scored `7.279592`
with checksum `167.430606` resident and `72.795916` streaming. The checksum
includes warmups. Raw logs and stage CSVs are in `/tmp/ss2-integrated-benchmark`
and disappear on reboot. The different mixed-alpha behavior is tested
separately; this opaque workload checks the existing paths for regression.
