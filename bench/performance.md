# Performance methodology

Use the benchmark guide to select a workload and build its dependencies.
[Benchmark guide](README.md)

## Protocol

`vqm_bench` and the in-memory reference drivers decode input outside
the timed metric calls. The CLI comparison in `bench/run_bench.py`
includes process startup and decoding. Whole-clip timing through
`vqm --json` reports core time, decode time and frames for the selected
complete iteration. Keep these timing boundaries separate when comparing results.

- Use the same inputs and flags for both builds. Alternate arms within one
  sitting and keep the raw timings, scores, checksums and per-stage spread.
  [Regression protocol](../AGENTS.md#protocol)
- `bench/run_inmemory.py --mode latency` runs interleaved rounds. Each driver
  runs its configured warmups, then records the median of the timed calls for
  each pair. The report groups by geometry and gives reciprocal pair time as
  `1000 / median_ms`. Use the raw CSV for each pair's decode time and score.
- SSIMULACRA 2 and Butteraugli latency drivers use four decode threads by
  default. `--decode-threads 1` performs decoding in the main thread and uses
  three loads for the reported median decode time. This flag belongs to the
  individual drivers.
- `--resident 2` uploads inputs before timing. SSIMULACRA 2 uses duplicate
  reference slots to prepare both images on each call. Butteraugli can reuse
  its prepared reference. Repeated pixels may stay hot in GPU memory.
- Vship's per-call API includes upload. Match input storage and preparation
  when interpreting differences. Keep SSIMULACRA 2's `--half-planes`
  results in a separate arm.
- The score difference in the latency report describes the measured pairs.
  Reference qualification has its own corpus and acceptance policies.
  [Validation](../validation/README.md)

For shared scratch, measure sequential SSIMULACRA 2 and Butteraugli on one
pair with the independent, shared workspace or unified API arm.

```sh
./build/vqm_bench --pair original.png distorted.png \
  --metric-pair independent --iterations 20 --warmup 3
./build/vqm_bench --pair original.png distorted.png \
  --metric-pair shared --iterations 20 --warmup 3
```

Add `--vulkan-buffers` to both arms for caller-owned buffers uploaded before
timing. The paired interval includes switching metrics and excludes image
decoding and construction. The stage table and CSV report both metrics'
stages per iteration.

## Throughput

Measure aggregate throughput with process wall time over the same fixed pair
count. For example, run these commands in alternating order and compare
elapsed time across runs.

```sh
time ./build/vqm_bench --pair original.png distorted.png \
  --workers 1 --iterations 1000 --warmup 0
time ./build/vqm_bench --pair original.png distorted.png \
  --workers 2 --iterations 1000 --warmup 0
```

The in-memory runner's `--mode throughput` uses exploratory fixed windows
for SSIMULACRA 2 and Butteraugli. Its output captures the following details.

- One decoded synthetic pair per geometry. Drivers report calls, elapsed
  seconds, FPS, cumulative metric time (`busy`) and checksum.
- One-worker and configured worker-count measurements alternate for each arm.
  The manifest currently requests three workers for GPU arms and sixteen for
  CPU references. The runner caps CPU workers using 60% of Linux
  `MemTotal` and an estimate of 256 bytes per pixel per worker.
- The reported window FPS is the median across `--throughput-runs`.
  The JSON retains every measurement and the effective thread count per size.
- The one-worker GPU stage rate uses stage timestamps and omits host and
  transfer costs. The call-rate column uses `busy / calls`, which includes
  host work and GPU waits. The CPU ideal column multiplies single-worker
  FPS by the effective worker count. These figures describe different
  timing boundaries.

## Corpus

Generate benchmark pairs before running the lower-level image runner.
The generator takes up to 100 CID22 pairs by default and creates synthetic
pairs at 512×512, 1920×1080 and 3840×2160. FFmpeg generates a test pattern and
a downscaled then upscaled version. `--synthetic-only` skips CID22.
Select built drivers with `--only` and inspect the manifest with `--list`.

```sh
.venv/bin/python bench/make_bench_pairs.py --synthetic-only
.venv/bin/python bench/run_inmemory.py --metric ssimu2 --mode latency \
  --only ours-resident,ours-streaming
```

`--metric all` selects SSIMULACRA 2 and Butteraugli in this runner.
Select CVVDP explicitly with `--metric cvvdp --mode latency`.

## Recorded results

The image runner writes `summary.md` and `summary.json` for latency and
`throughput.md` and `throughput.json` for windows under
`artifacts/reports/performance/` by default. Raw CSVs and logs sit alongside
them. Keep the build, inputs, and hardware with each run.

## Caveats

Alpha latency arms time the dark background and evaluate the bright
background outside the timer. libjxl's Butteraugli API handles alpha
internally, and score-only and throughput runs evaluate the complete alpha
score. fssimu2 reduces 16-bit inputs to 8 bits and drops alpha. Use the
measured workload, decoder, memory path and reference identity when comparing
a result.
