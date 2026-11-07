# Resident working-set benchmark

`vqm_bench --resident-ring N` measures a fixed number of SSIMULACRA 2
calls while cycling through multiple pre-uploaded image pairs. Decode, device
allocation and uploads happen before the timed calls, so the result excludes
PCIe uploads. The device still converts and prepares both images on every call.

Supply at least two and at most `N` pairs of the same geometry, with distinct
images if that is the workload you want to study. The bound `N` is a maximum
number of pairs, not an allocation by itself. Use `--pair` repeatedly or pass
a tab-separated list with `--pairs-file`:

```sh
./build/vqm_bench --resident-ring 8 --pairs-file pairs.txt \
  --iterations 200 --warmup 8
```

This uses one pipeline and two resident slots for each listed pair. Consecutive
calls use different original slots, forcing the original's preparation to be
rebuilt rather than reusing a cached reference. The output reports the number
of pairs and slots, geometry, resident input allocation, per-pair scores,
fixed-count wall-clock throughput, per-stage timings and a checksum. The byte
count is the input-slot allocation (`16 * width * height` per slot); the
pipeline's working buffers and any staging memory are additional. Device
limits may prevent a large ring from being allocated.

The pair list determines the actual image content. The tool does not check that
files have different decoded pixels. A larger ring changes the working set and
may reduce hot-cache effects, but it still cycles through the supplied images;
it does not establish a universal theoretical limit. Compare rings on the same
hardware in one sitting and alternate runs to reduce clock and thermal bias.
For an even per-pair contribution, choose iteration and warmup counts divisible
by both ring sizes. `--iterations` must cover every listed pair.

Every pair is scored once before warmup to print its score. If either image
has alpha, the printed score is the worse of dark and bright backgrounds; the
timed calls use the dark background only, matching the standard latency arm.
The checksum sums the warmup and timed scores only. The timed sequence follows
the warmup sequence, so the first timed original differs from the last warmup
original. With no warmup it follows the last untimed per-pair score.

This mode currently accepts one worker and SSIMULACRA 2 only, and cannot be
combined with the pool, other input modes, fixed-duration throughput,
`--score-only`, or per-pair `--csv`. `--stages-csv` is supported. The ordinary
`--resident 2` arm remains useful for the repeated-pair hot-memory measurement;
the ring is a separate workload and its rate is not directly comparable to
Butteraugli's cached-reference resident mode.

## Example measurement

On an RTX 5070 with driver 615.78.08, five runs of each ring were alternated
in one sitting. Each run used 200 timed calls after eight warmups, with one
pipeline and 1920x1080 RGB8 inputs. The inputs were eight different frames of
FFmpeg's `testsrc2`, each compared with a bicubic downscaled/upscaled version;
the two-pair arm used the first two frames. Both images were fully prepared on
each call. Scores and checksums were stable in all five runs.

| Pairs | Input-slot bytes | Median pairs/s (range) | Median ms/pair (range) | Checksum |
| --- | ---: | ---: | ---: | ---: |
| 2 | 132,710,400 | 2987.3 (2970.3–2994.9) | 0.3347 (0.3339–0.3367) | 12010.599789 |
| 8 | 530,841,600 | 2927.0 (2922.7–2934.8) | 0.3416 (0.3407–0.3422) | 11763.405437 |

The eight-pair rate was about 2% lower in this sitting. The image sets differ,
so this does not isolate cache pressure from image content or prove a general
throughput limit. The pair scores were 57.69174455 and 57.79479188 for the
first two pairs; the remaining scores were 57.14893176, 57.14225081,
57.03339050, 55.38115449, 54.48271806, and 55.76368862. Both arms printed
prepare and scale stages of approximately 0.11 and 0.14 ms/pair respectively,
rounded to two decimal places. Measurements are machine-specific.

The images and per-run command logs are local build artifacts under
`.cache/bench/ring-workload/`. The commands below retain the original benchmark
binary name. Use `./build/vqm_bench` when rerunning with a current build.

```sh
mkdir -p .cache/bench/ring-workload
ffmpeg -hide_banner -loglevel error -threads 1 -filter_threads 1 \
  -f lavfi -i testsrc2=size=1920x1080:rate=1 -frames:v 8 -pix_fmt rgb24 \
  .cache/bench/ring-workload/original-%02d.png
ffmpeg -hide_banner -loglevel error -threads 1 -filter_threads 1 -framerate 1 \
  -i .cache/bench/ring-workload/original-%02d.png \
  -vf 'scale=iw/2:ih/2:flags=bicubic,scale=iw*2:ih*2:flags=bicubic' \
  -frames:v 8 -pix_fmt rgb24 .cache/bench/ring-workload/distorted-%02d.png
python3 - <<'PY'
from pathlib import Path
root = Path('.cache/bench/ring-workload').resolve()
rows = [f'{root / f"original-{n:02d}.png"}\t{root / f"distorted-{n:02d}.png"}\n'
        for n in range(1, 9)]
(root / 'pairs-2.txt').write_text(''.join(rows[:2]))
(root / 'pairs-8.txt').write_text(''.join(rows))
PY
for i in 1 2 3 4 5; do
  for count in 2 8; do
    ./build/ssimulacra2_bench --resident-ring "$count" \
      --pairs-file ".cache/bench/ring-workload/pairs-$count.txt" \
      --iterations 200 --warmup 8
  done
done
```
