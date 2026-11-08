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

### Streaming versus resident scoring

Use **scoring wall time** for both sides of the comparison. Streaming starts
with decoded host pixels and includes uploading both inputs on every call.
Resident inputs are uploaded before timing. Both exclude decoding,
pipeline/device setup and warmups, and include dispatch, synchronization and
score readback. These are not whole-process or file-to-score CLI rates.

The VQM and Vship drivers accept `--throughput-iterations N` for a fixed
**total** count across workers. VQM uses `--workers`; peers use `--threads`.
Supply three warmups per worker and the same pair and total count to each arm.
The Vship drivers call the public host-input API, so only their streaming arm
is measured. Do not compare VQM resident FPS with peer streaming FPS to claim
an implementation speedup. The CVVDP peer path resets score and history for
every independent still; it is not a temporal-video comparison.

```sh
./build/vqm_bench --metric ssimulacra2 --pair original.png distorted.png \
  --workers 3 --warmup 3 --throughput-iterations 2400
./build/vqm_bench --metric ssimulacra2 --pair original.png distorted.png \
  --resident 2 --workers 3 --warmup 3 --throughput-iterations 2400
./build/vship-fast-build-vulkan/vship_bench --metric ssimu2 \
  --pair original.png distorted.png --threads 3 --warmup 3 \
  --throughput-iterations 2400
```

PSNR-HVS and VQM CVVDP use `--iterations N` and print a `fixed-count wall clock`
rate. For PSNR-HVS this is a shared submission/completion interval. CVVDP
stills accumulate timed scoring-call durations, not reciprocal median latency;
resets between independent stills are outside VQM's timing and inside the
peer's public API call. Use one CVVDP worker. These small API/bookkeeping
differences remain part of the comparison; scores can also differ by backend.

For temporal CVVDP, each iteration is a whole clip, not an independent still.
Frames stay ordered on one pipeline and history resets only between clips.
Both paths time `Push` and `Finalize`, excluding decoding and clip reset.
`--vulkan-buffers` preuploads the **entire clip** into benchmark-owned external
device-local buffers before warmups. This trades VRAM for avoiding transfers;
it is not a constant-memory streaming implementation. Separate buffers avoid
exceeding Vulkan's per-buffer storage range with a large resident arena.
The pipeline still owns its scratch and temporal history; the benchmark owns
only the input pixels through the public Vulkan-buffer API.

```sh
./build/vqm_bench --metric cvvdp --cvvdp-video \
  --pair original.y4m distorted.y4m --iterations 10 --warmup 3 \
  --cvvdp-ppd 75.402449345163049
./build/vqm_bench --metric cvvdp --cvvdp-video --vulkan-buffers \
  --pair original.y4m distorted.y4m --iterations 10 --warmup 3 \
  --cvvdp-ppd 75.402449345163049
```

Divide timed clip frame counts by accumulated scoring wall seconds, not by
median frame latency. Warmup clip frames are not in either timed count.
Check whole-clip JOD equality between streaming and resident paths. Keep video
and independent-image results separate. The normal CLI's serial scheduling
is unchanged by any benchmark worker or resident-clip mode.

### Resident compute throughput

For peak resident-input throughput, upload decoded pixels before timing, warm
every worker, then divide a **fixed completed evaluation count** by the shared
scoring interval. Do not include uploads, decoding, device/pipeline creation or
warmups in that interval. Host dispatch, GPU waits and score readback remain
included: this is achieved scoring throughput, not a shader-time ceiling.

```sh
./build/vqm_bench --metric ssimulacra2 --pair original.png distorted.png \
  --resident 2 --workers 3 --warmup 3 --throughput-iterations 12000
./build/vqm_bench --metric butteraugli --pair original.png distorted.png \
  --resident 2 --workers 3 --warmup 3 --throughput-iterations 1800
./build/vqm_bench --metric psnr-hvs --pair original.y4m distorted.y4m \
  --resident 2 --workers 3 --warmup 3 --iterations 18000
./build/vqm_bench --metric cvvdp --pair original.png distorted.png \
  --resident 2 --warmup 3 --iterations 2000 --cvvdp-ppd 75.402449345163049
```

The first two commands emit `throughput` with the total `calls`, shared
`seconds`, `fps` and timed-only `checksum`. The fixed count is **total across
workers**, not per worker; an uneven count is distributed without rounding it
up. Worker creation and warmups precede the shared start, and the interval ends
only after all workers finish. `--throughput-seconds` is a separate exploratory
mode and cannot be combined with `--throughput-iterations`.

PSNR-HVS and CVVDP use their existing fixed-count paths and print a
`fixed-count wall clock` rate. PSNR-HVS `--iterations` is also a total across
workers; its score-only mode is not a concurrent throughput measurement.
CVVDP here scores independent still images on one worker, not a temporal clip.
Their checksums include warmups. PSNR-HVS reads only the first Y4M frame.

Use matching inputs and interleave five rounds. Report medians and spread,
worker counts, cache behavior and resident working-set size. SSIMULACRA 2
alternates original slots to rebuild both images; Butteraugli may reuse its
prepared reference. This repeated-pair workload is not uncached distinct-frame
throughput. Keep half planes separate and do not compare these rates directly
against a peer's host-input/upload-inclusive API.

### Host-input process throughput

Measure aggregate throughput with process wall time over the same fixed pair
count. For example, run these commands in alternating order and compare
elapsed time across runs.

```sh
time ./build/vqm_bench --pair original.png distorted.png \
  --workers 1 --iterations 1000 --warmup 0
time ./build/vqm_bench --pair original.png distorted.png \
  --workers 2 --iterations 1000 --warmup 0
```

For SSIMULACRA 2 and Butteraugli batch comparisons, `--score-only` suppresses
the table; it does not disable `--iterations` or `--warmup` in either VQM or
the Vship harness. One output row can therefore represent many metric calls.
Set both options explicitly. The evaluated pair count is the number of pair-list
rows multiplied by `--iterations`, not the number of printed scores. Warmup
calls are additional work. When timing the whole process, include their cost in
the reported boundary and keep them equal between implementations.

To compare one and three workers at a fixed total count, use one pair-list row
with 1,200 iterations or three rows with 400 iterations each. Use the same
worker count, row count, repetitions and input files for each implementation.
Process wall time includes initialization, decoding, warmups and teardown;
reciprocal median call latency and GPU timestamps are different measurements.
Compare host-input arms against other host-input arms. Report VQM's resident
arm separately when a peer's measured API requires host inputs on every call.

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
