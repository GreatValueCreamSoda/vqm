# Host-plane upload measurements

On 2026-11-02, an RTX 5070 with driver 615.78.08 scored a 3840x2160 RGB8
`testsrc2` image against a contrast/brightness-adjusted copy. Each implementation
decoded once, then used borrowed host-plane views. The baseline packed each view
into an owning `Image` before the ordinary upload; the candidate copied each view
directly into the mapped upload or staging region. This comparison covers the
host-view path, not the existing `Image` path or GPU-resident input.

The baseline was commit `72fcdea` with the same `--host-views` benchmark
harness as the candidate. The two arms were run alternately, five times each,
using `--iterations 8 --warmup 2`; GPU calls, image decoding and stage timers
were otherwise identical. `core` is the median measured call time. `untimed`
is the per-call total elapsed time less the reported GPU/score stages; its
value below was calculated for each run from the six-decimal stage CSV before
taking the median. Times are milliseconds per pair; parentheses show ranges.

| Metric | Stage | Baseline | Direct views |
| --- | --- | ---: | ---: |
| SSIMULACRA 2 | core | 8.0191 (7.9563–8.6327) | 3.4012 (3.3919–3.4070) |
| SSIMULACRA 2 | untimed | 6.9336 (6.8742–7.5288) | 2.2929 (2.2864–2.3014) |
| SSIMULACRA 2 | prepare | 0.4272 (0.4270–0.4311) | 0.4270 (0.4269–0.4310) |
| SSIMULACRA 2 | scale | 0.6799 (0.6780–0.6803) | 0.6801 (0.6798–0.6804) |
| Butteraugli | core | 18.7034 (18.6416–18.7344) | 14.9158 (14.8688–14.9252) |
| Butteraugli | untimed | 6.3572 (6.2824–6.4363) | 2.5288 (2.4909–2.5549) |
| Butteraugli | opsin | 1.9955 (1.9932–1.9970) | 1.9971 (1.9958–1.9978) |
| Butteraugli | separate | 4.9311 (4.9301–4.9380) | 4.9379 (4.9363–4.9386) |

The target untimed stage improved 66.9% for SSIMULACRA 2 and 60.2% for
Butteraugli; median core time improved 57.6% and 20.3%, respectively. The
SSIMULACRA 2 score was 92.437150 with checksum 924.371495, and Butteraugli's
score was 7.279592 with checksum 72.795916 in every arm. The pair count was
two warm-ups plus eight measured calls. The GPU stage values are reported as
checks, not as improvements; this change removes host copying rather than shader
work. Data are one machine and one synthetic pair, not a universal rate.

The test pair can be regenerated with:

```sh
ffmpeg -hide_banner -loglevel error -f lavfi \
  -i 'testsrc2=size=3840x2160:rate=1' -frames:v 1 -threads 1 \
  -y /tmp/uhd_orig.png
ffmpeg -hide_banner -loglevel error -i /tmp/uhd_orig.png \
  -vf 'eq=contrast=0.97:brightness=0.01' -frames:v 1 -threads 1 \
  -y /tmp/uhd_dist.png
```

Use a separate baseline worktree and the same benchmark harness in both arms:

```sh
for i in 1 2 3 4 5; do
  for metric in ssimulacra2 butteraugli; do
    /tmp/bench-view-base/build/ssimulacra2_bench --metric "$metric" \
      --host-views --pair /tmp/uhd_orig.png /tmp/uhd_dist.png \
      --iterations 8 --warmup 2
    ./build/ssimulacra2_bench --metric "$metric" \
      --host-views --pair /tmp/uhd_orig.png /tmp/uhd_dist.png \
      --iterations 8 --warmup 2
  done
done
```

## Ordinary Image regression check

The same baseline and candidate were also compared using ordinary `Image`
inputs, which do not use the host-view overload. Five runs per arm were
interleaved on the same 4K pair using the resident and streaming protocols in
`AGENTS.md`. The median per-pair core times in milliseconds (ranges in
parentheses) were:

| Metric and mode | Baseline | Candidate |
| --- | ---: | ---: |
| SSIMULACRA 2 resident | 1.1978 (1.1899–1.2199) | 1.1941 (1.1850–1.1991) |
| SSIMULACRA 2 streaming | 3.4034 (3.3931–3.4110) | 3.4086 (3.3906–3.4109) |
| Butteraugli resident | 9.2286 (9.2146–9.2466) | 9.2222 (9.1563–9.2500) |
| Butteraugli streaming | 14.8896 (14.8782–14.9339) | 14.9129 (14.8881–14.9341) |

Scores and checksums matched for every arm; the measured ranges overlap in all
four cases. The comparison checks for a regression in the ordinary upload and
resident paths rather than claiming that those paths became faster.
