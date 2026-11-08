# Run benchmarks

Run the commands from the repository root.

| Workload | Entry point |
| --- | --- |
| Time GPU stages on one decoded image pair | [`vqm_bench`](#time-one-pair) |
| Score and time complete clips | [`vqm --json`](#score-a-complete-clip) |
| Compare complete hosted clips against pinned original scores | [`benchmark.py`](#hosted-video-benchmark) |
| Compare streaming/resident scoring with Vship and Vship-fast | [Scoring throughput comparison](#streaming-and-resident-scoring-comparison) |
| Compare still-image latency with a pinned peer | [`benchmark.py --suite images`](#still-image-peer-comparison) |
| Compare latency, exploratory windows or fixed-count workers | [Performance methodology](performance.md) |

## Streaming and resident scoring comparison

<!-- throughput-comparison:start -->
Measured `2026-11-07T19:37:39.121668-05:00` on **NVIDIA GeForce RTX 5070**.
Candidate core revision: `5c89f1317ac8074e5dcf64c1e659d398f4f63b02`;
the report preserves working-tree changes, input/binary/library hashes,
build fingerprints, compiler, CPU, driver inventory and every process receipt.

- `vship-cuda`: `91d03ec2f2e03e4817b64655a5ea58ab7ab3d3ba`.
- `vship-vulkan`: `91d03ec2f2e03e4817b64655a5ea58ab7ab3d3ba`.
- `vship-fast-vulkan`: `2a937ebaa1ae18718596a047c1e4947caf858c8e`.

5 rotating rounds, 3 warmups per worker (whole clips for video).
No benchmark processes overlap. Median pair evaluations/s (video frames/s),
with minimum–maximum rates in parentheses:

| Metric | Workers | 1080p streaming | 1080p resident | 4K streaming | 4K resident |
| --- | ---: | ---: | ---: | ---: | ---: |
| SSIMULACRA 2 | 3 | 1,816.4 (1,795.3–1,818.5) | 3,836.7 (3,832.0–3,866.4) | 455.2 (455.0–455.4) | 915.0 (913.7–915.5) |
| Butteraugli | 3 | 351.3 (351.2–352.0) | 498.4 (498.2–499.5) | 72.7 (72.7–72.7) | 109.0 (109.0–109.1) |
| PSNR-HVS | 3 | 3,453.8 (3,447.4–3,463.1) | 4,247.7 (4,246.8–4,253.6) | 899.5 (899.1–899.9) | 1,134.1 (1,132.7–1,135.6) |
| CVVDP images | 1 | 390.4 (388.3–390.9) | 500.5 (494.9–502.2) | 96.3 (96.0–96.5) | 122.3 (121.9–122.4) |
| CVVDP video | 1 | 256.2 (255.5–256.6) | 274.1 (272.6–275.0) | 63.1 (62.9–63.2) | 68.3 (68.2–68.4) |

Host-input peer scoring, including per-call uploads:

| Metric | Size | VQM Vulkan | Vship CUDA | Vship Vulkan | Vship-fast Vulkan |
| --- | --- | ---: | ---: | ---: | ---: |
| SSIMULACRA 2 | 1920×1080 | 1,816.4 (1,795.3–1,818.5) | 862.9 (857.0–864.4) | 470.8 (469.7–473.0) | 594.4 (594.3–595.9) |
| SSIMULACRA 2 | 3840×2160 | 455.2 (455.0–455.4) | 197.2 (195.8–199.2) | 103.9 (103.6–116.3) | 155.4 (155.4–155.4) |
| Butteraugli | 1920×1080 | 351.3 (351.2–352.0) | 170.5 (170.4–170.9) | 148.8 (148.7–149.0) | 318.1 (318.0–318.7) |
| Butteraugli | 3840×2160 | 72.7 (72.7–72.7) | 37.9 (37.8–37.9) | 34.0 (32.9–34.0) | 78.4 (78.4–78.4) |
| CVVDP images | 1920×1080 | 390.4 (388.3–390.9) | — | 215.0 (214.9–215.2) | 360.4 (360.0–362.0) |
| CVVDP images | 3840×2160 | 96.3 (96.0–96.5) | — | 54.6 (54.5–54.8) | 87.6 (87.3–88.2) |

Timed counts are **total across workers**, not per worker:

| Metric | 1920×1080 | 3840×2160 |
| --- | ---: | ---: |
| SSIMULACRA 2 | 2400 | 600 |
| Butteraugli | 600 | 120 |
| PSNR-HVS | 12000 | 2400 |
| CVVDP images | 120 | 30 |
| CVVDP video | 320 | 160 |

SSIMULACRA 2 rebuilds both images by alternating resident reference slots;
Butteraugli may reuse its prepared reference. RGB8 synthetic PNG pairs are
testsrc2 versus bicubic half-size/downscale-upscale distortion. PSNR-HVS
uses the first native YUV420p8 clip frame and always prepares both inputs.
Temporal CVVDP uses complete 24 fps YUV clips: 32 frames at 1080p, 16 at 4K,
ten timed clips each. Every frame stays on the same sequential pipeline.
The resident video arm preuploads all raw frames into separate external
device-local buffers; pipeline scratch and private temporal history remain
core-owned. This avoids Vulkan's per-buffer storage-range limit but uses
more VRAM than streaming and is not an unbounded-video throughput claim.
CVVDP uses fixed 75.402449345163049 PPD, standard_4k display, no resizing
or heatmaps. Independent stills reset between calls; temporal clips reset
only between clips and report finalized whole-clip JOD.

All rates exclude setup, decoding and warmups; only streaming includes
uploads. CVVDP stills sum scoring-call wall durations; other arms use a
shared worker interval or whole-clip scoring durations. VQM still resets
are outside the timer, peer still resets are inside its public API call.
Inputs can remain GPU-cache-hot. Device identity matches names, not UUIDs.
The timing-enabled core is unprofiled and uses FP32 planes. No PSNR-HVS,
temporal-video or resident-input peer arm is measured. No VMAF throughput
is claimed. These workloads do not measure normal CLI throughput.

Scores are stable across rounds; VQM streaming/resident scores match
within 1e-6. Peer score differences are not numerical qualification:

| Metric | Size | VQM | Vship CUDA | Vship Vulkan | Vship-fast Vulkan |
| --- | --- | ---: | ---: | ---: | ---: |
| SSIMULACRA 2 | 1920×1080 | 30.74238424 | 30.36651802 | 30.54326820 | 29.86768371 |
| SSIMULACRA 2 | 3840×2160 | 27.30450868 | 27.12780571 | 27.32060242 | 26.22902194 |
| Butteraugli | 1920×1080 | 22.25156975 | 22.12087059 | 22.12087250 | 22.25155067 |
| Butteraugli | 3840×2160 | 22.21629143 | 22.14696121 | 22.14696503 | 22.21628952 |
| CVVDP images | 1920×1080 | 9.58955901 | — | 9.58127880 | 9.58955860 |
| CVVDP images | 3840×2160 | 9.57244480 | — | 9.56727980 | 9.57244490 |

Raw evidence: `/home/creamsoda/Projects/vqm/artifacts/reports/throughput-script-2026-11-07`. The standalone `summary.md` and
`summary.json` preserve these results even after later README refreshes.
<!-- throughput-comparison:end -->

### Refresh the entire comparison

Run the complete maintained workflow from the repository root:

```sh
uv run --no-project --python 3.12 bench/benchmark.py --suite throughput
```

It fetches pinned Vship/Vship-fast sources, builds VQM plus CUDA/Vulkan peers,
generates and fingerprints both RGB image pairs and native YUV video clips,
matches devices by name, then runs five rotating rounds of every arm. It saves
all process exits, stdout/stderr, commands, timed counts, checksums, scores,
median/range tables and environment provenance in a new timestamped directory
under `artifacts/reports/`. Setup, decoding and warmups stay outside scoring
throughput; streaming uploads stay inside it. No benchmark processes overlap.

Add `--update-readme` to regenerate the managed tables in both READMEs **only
after** the full matrix succeeds and validates. Incomplete runs, process crashes,
changed binaries/inputs, inconsistent counts, unstable scores, and
streaming/resident score mismatches leave the README tables untouched. Smoke
profiles and reduced protocols cannot publish headline numbers.
The smoke profile runs every selected arm on a 640×360 pair and a four-frame
clip; it checks the harness, not meaningful performance.

```sh
uv run --no-project --python 3.12 bench/benchmark.py --suite throughput \
  --update-readme
uv run --no-project --python 3.12 bench/benchmark.py --suite throughput \
  --profile smoke --skip-build
uv run --no-project --python 3.12 bench/benchmark.py --suite throughput --list
```

Use `--build-dir`, `--input-dir` and `--out-dir` for alternate locations. Existing
output directories are rejected to preserve prior evidence. Existing generated
inputs are reused only when their recipe and hashes match. `--skip-build`
requires prebuilt current drivers and verified pinned checkouts. `--references`
can select fewer peers for local reports; README publication requires all three.
Source snapshots include untracked benchmark changes and CMake caches alongside
the HEAD-relative patch. Input, binary, shared-library and captured-source hashes
are checked again after timing to reject mixed-build results.
The default needs the CUDA toolchain for the CUDA peer; VQM itself remains
Vulkan-only. See `--help` for rounds, warmups, worker count and process timeout.
VMAF is not part of this spatial/CVVDP comparison; the separate default video
suite retains its existing qualification workload.

### Earlier resident-only measurement

Measured November 7, 2026 on an NVIDIA GeForce RTX 5070, driver 615.78.08,
AMD Ryzen 7 7700X. Desktop applications remained active. VQM Release core
revision was `5c89f1317ac8074e5dcf64c1e659d398f4f63b02`, with the benchmark-only
fixed-count throughput extension retained beside the results as a patch.
The timing-enabled benchmark core uses FP32 planes, without an external
profiler. No scoring implementation changed for these measurements.

All inputs are uploaded before timing. The interval excludes pipeline/device
construction, loading, uploads, worker creation and three warmups per worker.
It includes host dispatch, GPU work, synchronization, score conversion and
completion of the final call. Five rounds rotate arm order, with no competing
benchmark processes. Values are medians with the slowest–fastest run in
parentheses; these are achieved pair evaluations/s, not reciprocal median
latency or a shader-only estimate.

| Metric | Size | GPU workers | Resident evaluations/s |
| --- | --- | ---: | ---: |
| ssimulacra2 | 1920×1080 | 1 | 3,043.0 (3,026.4–3,050.9) |
| ssimulacra2 | 1920×1080 | 3 | 3,818.9 (3,811.5–3,836.1) |
| butteraugli | 1920×1080 | 1 | 479.4 (477.9–483.4) |
| butteraugli | 1920×1080 | 3 | 498.3 (497.0–498.5) |
| psnr-hvs | 1920×1080 | 1 | 4,105.8 (4,045.0–4,122.8) |
| psnr-hvs | 1920×1080 | 3 | 4,048.2 (4,047.6–4,058.4) |
| cvvdp | 1920×1080 | 1 | 492.6 (487.4–497.9) |
| ssimulacra2 | 3840×2160 | 1 | 845.7 (845.1–846.4) |
| ssimulacra2 | 3840×2160 | 3 | 911.4 (911.2–911.5) |
| butteraugli | 3840×2160 | 1 | 107.6 (107.3–107.8) |
| butteraugli | 3840×2160 | 3 | 109.0 (109.0–109.0) |
| psnr-hvs | 3840×2160 | 1 | 1,089.9 (1,087.1–1,097.9) |
| psnr-hvs | 3840×2160 | 3 | 1,130.2 (1,129.5–1,131.1) |
| cvvdp | 3840×2160 | 1 | 120.9 (120.5–121.6) |

SSIMULACRA 2, Butteraugli and CVVDP use the same opaque synthetic RGB8 PNG
pairs as the host-input comparison below. PSNR-HVS uses the first native
YUV420p8 frame of the generated testsrc2 clips described in the additional
measurements section. All scores agree with their corresponding host-input
runs within output precision, and checksums are stable across workers/rounds.
PSNR-HVS does not always benefit from extra resident workers (1080p is slightly
faster with one); the README consistently reports its three-worker result.

Timed counts per process are:

| Metric | 1920×1080 | 3840×2160 |
| --- | ---: | ---: |
| SSIMULACRA 2 | 12,000 | 2,400 |
| Butteraugli | 1,800 | 360 |
| PSNR-HVS | 18,000 | 3,600 |
| CVVDP images | 2,000 | 400 |

Counts are total across workers, not multiplied by the worker count.
Each SSIMULACRA 2 worker parks two original copies and one distorted image,
alternating original slots to force full preparation of both images. The other
workers park one pair each; Butteraugli can reuse its prepared reference.
These small repeated working sets can remain GPU-cache-hot. They are not a
claim about distinct uncached frames, fresh-frame video or the normal serial
CLI. CVVDP is explicitly an independent-image measurement with one worker and
fixed 75.402449345163049 PPD/standard_4k display, not temporal-video throughput.

Use the commands and exact timing boundaries in the
[resident protocol](performance.md#resident-compute-throughput). Select each
geometry's inputs and counts above, run both one and three spatial workers and
interleave five rounds. CVVDP uses one worker. Raw stdout, stderr, counts,
checksums, process receipts, binary/input hashes, device information and the
benchmark patch are in `artifacts/reports/resident-throughput-2026-11-07/`.

Peer drivers here accept host inputs and upload during scoring. No matching
resident peer path was measured. Keep the upload-inclusive peer comparisons
below separate; resident/host rate ratios are not implementation speedups.

## Time one pair

Build the CLI and benchmark with the prerequisites in the main README.
[Build setup](../README.md#build)

```sh
cmake --preset release
cmake --build --preset release --target vqm vqm_bench
./build/vqm_bench --pair original.png distorted.png --iterations 20 --warmup 3
./build/vqm_bench --pair original.png distorted.png --iterations 20 --warmup 3 --resident 2
```

The default arm streams the input pixels. The resident arm uploads before
timing. SSIMULACRA 2 prepares both images on each call in either arm.
Use `./build/vqm_bench --help` for metrics, input modes and CSV flags.
The performance methodology describes timing boundaries, worker scaling and
how to compare two builds in one sitting. [Performance methodology](performance.md)

## Score a complete clip

Use the normal CLI for whole-clip timing. The CLI supports all six metrics
on compatible progressive Y4M input, including PSNR-HVS on native YUV.

```sh
./build/vqm --metric vmaf-v0 --json original.y4m distorted.y4m \
  --iterations 5 --warmup 1
./build/vqm --metric cvvdp --json --pairs-file pairs.tsv \
  --iterations 5 --warmup 1
```

Each TSV row contains the sample ID, original path and distorted path.
Each iteration starts a new sequence and includes its finalization. JSON
reports the median-core-time iteration, its separate decode time and the
frame count. CVVDP reports whole-clip JOD. Other metrics include ordered
frame results. Use `vqm_bench --cvvdp-video` for CVVDP stage timing when
needed. The ordinary `vqm_bench` VMAF and PSNR-HVS arms use the first Y4M
frame as a repeated-pair workload. Use `--vmaf-scale` and `--vmaf-pad` in the
normal CLI to control VMAF geometry independently. The repeated-pair benchmark
uses `--vmaf-fit pad` for the combined operation.

Run `.venv/bin/python -m validation.temporal.run --profile smoke` for a
short reference-agreement check or use `--profile qualification` for the full
matrix. [Hosted video validation](../validation/temporal/README.md)

## Hosted video benchmark

The default suite fetches the pinned Cloudinary, libjxl, libvmaf and
ColorVideoVDP sources, builds the CLI and native references, and prepares the
locked CPU reference environment in `.venv/`. It fetches missing checksum-pinned
Xiph source prefixes and uses checked-in original scores for agreement checks.

```sh
uv run --no-project --python 3.12 bench/benchmark.py --dry-run
uv run --no-project --python 3.12 bench/benchmark.py
```

Both suites accept `--build-dir DIR` (default `build/`). Relative build paths
resolve from the repository root. The selected directory holds candidate
binaries, copied native references and build fingerprints; provenance reads
that same directory. The Python reference environment remains in `.venv/`.

The smoke profile covers 39 clip pairs per metric with 12 frames per clip.
The qualification profile covers 390 pairs per metric with 40 frames per clip.
The five metrics are SSIMULACRA 2, Butteraugli, VMAF v0, VMAF v1 and CVVDP.
The default is five rounds in rotating metric order, 20 timed iterations and
three warmups. `--metrics` selects a subset. `--setup-only` stops after
preparation. `--offline` requires cached sources and packages. The first run
needs network access. Video conversion caches can use tens of GB.

Use Git, `uv`, CMake 3.25+, Ninja, a C++20 compiler, Make, Meson, pkg-config
and a working Vulkan compute driver. On Debian or Ubuntu, install the
development dependencies with the following command.

```sh
sudo apt install git build-essential cmake ninja-build meson pkg-config \
  libvulkan-dev glslc libpng-dev libjpeg-dev libhwy-dev liblcms2-dev libbrotli-dev
```

Reports default to a new directory under `artifacts/reports/video-benchmark/`.
An explicit `--out-dir` must be empty. The summary keeps core time, decode
time, CLI process wall time, per-round spread and reference agreement separate.
The core timing includes metric finalization. Process wall time also includes
warmups, decoding and setup. The reference adapters can perform additional
conversion and serialization. Use the reports for the specific workload and
timing boundaries recorded there. [Performance methodology](performance.md)

## Still-image peer comparison

The image suite builds the selected references and compares decoded-image
latency. It defaults to a synthetic 1920x1080 pair, pinned Vship Vulkan,
and five interleaved rounds of 20 calls with three warmups.

```sh
uv run --no-project --python 3.12 bench/benchmark.py --suite images \
  --pair original.png distorted.png
uv run --no-project --python 3.12 bench/benchmark.py --suite images --help
```

Without `--pair`, FFmpeg generates a synthetic pair. Use even dimensions with
`--size WxH`. Select `--metric ssimu2|butteraugli|cvvdp` and
`--references NAMES` to change the workload. Cloudinary and fssimu2 cover
SSIMULACRA 2 only. The CVVDP image suite supports Vship and Vship-fast Vulkan. CUDA and
fssimu2 need their own toolchains. Reports default to
`artifacts/reports/quick-benchmark/`.

### Fixed-count GPU comparison

Measured on November 7, 2026 with an NVIDIA GeForce RTX 5070, driver 615.78.08,
and AMD Ryzen 7 7700X. Desktop applications remained active. VQM was Release
`5c89f1317ac8074e5dcf64c1e659d398f4f63b02`, using the timing-enabled benchmark
core without half planes. Vship was pinned
`91d03ec2f2e03e4817b64655a5ea58ab7ab3d3ba` (v5.1.2), built separately for
Vulkan and native CUDA. Device selection matched the NVIDIA device name;
Vship's API does not expose a UUID for stronger identity verification.

Each geometry uses one opaque 8-bit RGB synthetic PNG pair: FFmpeg `testsrc2`
against its bicubic downscaled/upscaled copy. Each 1080p process performs 1,200
timed evaluations; each 4K process performs 300. With one worker, the pair list
has one row and `--iterations` is 1,200 or 300. With three workers, it has three
identical rows and `--iterations` is 400 or 100 per row. Each worker also runs
three warmups. There are five rounds per arm, rotating implementation order
and alternating worker-count order; implementations never run concurrently.

**Rate is the timed evaluation count divided by total process wall time.**
It includes initialization, input loading, warmups, transfers, CSV/score output
and device teardown. Warmups are excluded from the numerator, not the measured
wall time. Loading occurs per pair-list row, not per scoring iteration. The
one-thread decoding path loads its pair three times; the three-worker path
loads three identical rows using three decode threads. Repeated input files
and GPU working sets can stay cached. This is not kernel-only FPS, a video
workload or a throughput ceiling.

Host-input rates below are medians, with the slowest–fastest run in parentheses:

| Metric | Size | Workers | VQM Vulkan | Vship CUDA | Vship Vulkan |
| --- | --- | ---: | ---: | ---: | ---: |
| SSIMULACRA 2 | 1920×1080 | 1 | 995.3 (925.3–1,000.6) | 535.5 (530.6–539.4) | 350.7 (348.9–351.4) |
| SSIMULACRA 2 | 1920×1080 | 3 | 1,413.6 (1,401.6–1,423.1) | 698.4 (689.2–711.1) | 390.6 (386.1–393.6) |
| SSIMULACRA 2 | 3840×2160 | 1 | 219.5 (216.3–226.5) | 124.0 (119.1–129.9) | 81.3 (72.2–82.0) |
| SSIMULACRA 2 | 3840×2160 | 3 | 278.1 (256.5–283.7) | 150.9 (141.9–156.1) | 80.0 (74.0–87.2) |
| Butteraugli | 1920×1080 | 1 | 274.1 (273.0–275.4) | 161.1 (160.8–161.4) | 136.9 (136.8–137.3) |
| Butteraugli | 1920×1080 | 3 | 328.4 (327.7–328.7) | 163.1 (162.4–163.5) | 138.5 (138.2–138.9) |
| Butteraugli | 3840×2160 | 1 | 61.4 (61.2–61.5) | 34.1 (34.0–34.2) | 30.3 (30.3–30.4) |
| Butteraugli | 3840×2160 | 3 | 61.9 (61.4–61.9) | 35.1 (35.0–35.1) | 30.8 (30.7–30.8) |

VQM's resident path uploads inputs before repeated calls. SSIMULACRA 2
alternates reference slots so both images are prepared on every call;
Butteraugli can reuse the prepared reference. The following rates still use
whole-process wall time, including initial uploads and setup. They are not
like-for-like comparisons with Vship's host-input API:

| Metric | Size | One worker | Three workers |
| --- | --- | ---: | ---: |
| SSIMULACRA 2 | 1920×1080 | 2,175.8 (2,132.1–2,184.4) | 2,449.9 (2,413.7–2,457.3) |
| SSIMULACRA 2 | 3840×2160 | 444.5 (421.6–462.3) | 421.4 (415.0–424.8) |
| Butteraugli | 1920×1080 | 452.0 (448.3–453.0) | 459.2 (457.9–459.8) |
| Butteraugli | 3840×2160 | 95.6 (95.5–96.0) | 87.9 (87.6–92.2) |

Scores were finite and stable across all rounds and worker counts, but they
are not identical between implementations. Butteraugli uses maximum distance
(`norminf` for Vship). These are peer comparisons, not numerical qualification:

| Metric | Size | VQM, both paths | Vship CUDA | Vship Vulkan |
| --- | --- | ---: | ---: | ---: |
| SSIMULACRA 2 | 1920×1080 | 30.74238424 | 30.36651802 | 30.54326820 |
| SSIMULACRA 2 | 3840×2160 | 27.30450868 | 27.12780571 | 27.32060242 |
| Butteraugli | 1920×1080 | 22.25156975 | 22.12087059 | 22.12087250 |
| Butteraugli | 3840×2160 | 22.21629143 | 22.14696121 | 22.14696503 |

An additional 1080p identity check produced SSIMULACRA 2 scores of 100.00000000
for VQM, 99.55310822 for Vship CUDA and 99.27654266 for Vship Vulkan.
Butteraugli identity was zero for all three. No temporal, all-metrics, normal
CLI, CPU reference or Vship-fast throughput is claimed by this comparison.

To reproduce the three-worker 1080p SSIMULACRA 2 arm, build the release benchmark
and the pinned peers, then time each process separately. Rotate the three
scoring commands across five rounds and calculate `1200 / median_wall_seconds`.
Do not count printed rows as evaluations: each row below runs 400 timed calls.

```sh
cmake --preset release -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build --preset release --parallel
.venv/bin/python tools/fetch_references.py --only Vship
.venv/bin/python tools/build_references.py --only vship-vulkan,vship-cuda
.venv/bin/python bench/make_bench_pairs.py --synthetic-only --sizes 1920x1080,3840x2160
out=artifacts/reports/fixed-count-peer
mkdir -p "$out"
awk 'NR == 1 { print; print; print; exit }' \
  .cache/bench/pairs/synthetic-1920x1080.txt > "$out/pairs.tsv"
time ./build/vqm_bench --metric ssimulacra2 --pairs-file "$out/pairs.tsv" \
  --score-only --threads 3 --decode-threads 3 --iterations 400 --warmup 3
time ./build/vship-build-cuda/vship_bench --metric ssimu2 --pairs-file "$out/pairs.tsv" \
  --score-only --threads 3 --decode-threads 3 --iterations 400 --warmup 3
time ./build/vship-build-vulkan/vship_bench --metric ssimu2 --pairs-file "$out/pairs.tsv" \
  --score-only --threads 3 --decode-threads 3 --iterations 400 --warmup 3
```

Use `--metric butteraugli` for all three drivers to measure Butteraugli. For 4K,
use its generated pair and 100 iterations per row. For one worker, use one row,
one thread and 1,200 or 300 iterations. Add `--resident 2` only to VQM for the
separate resident arm. Keep fresh CSVs, output, exit statuses, input hashes,
device information and source revisions with each rerun. The captured matrix,
all commands and raw outputs are under the local ignored directory
`artifacts/reports/corrected-throughput-2026-11-07/`.

Vship exposes device names and indices. The runner requires a unique name
match against VQM's selected device and records the available identities in
JSON. After checking the devices manually, use `--allow-unverified-devices`
to permit the explicitly marked device-zero fallback.

Reference checkouts must be clean at their pinned commits, including submodules.
The build tools work on copies and rebuild when the recorded recipe, source or
toolchain fingerprint changes. Fetch or build a named reference with
`tools/fetch_references.py --only NAME` or
`tools/build_references.py --only NAME`.
[Reference setup and validation](../validation/README.md)

For an alternate build, pass the same directory when building references and
running the in-memory comparison. Its pair list has two columns: original and
distorted paths separated by a tab. The corpus runner also accepts `--build-dir`.

```sh
python3 tools/build_references.py --only cloudinary --build-dir build-comparison
.venv/bin/python bench/run_inmemory.py --mode latency --metric ssimu2 \
  --only cloudinary --build-dir build-comparison --pairs-file image-pairs.tsv
python3 tools/vscode_compile_commands.py --build-dir build-comparison
```

### Additional host-input and temporal measurements

The same device, driver and VQM/Vship revisions were used in a second,
five-round rotating-arm matrix on November 7, 2026. These are **whole-process**
rates, including setup, warmups, loading and teardown, not resident compute
throughput. CVVDP image inputs match the RGB8 pairs above, with fixed
75.402449345163049 PPD and the standard_4k display (200 nit, contrast 1000,
ambient 250 lux), no resizing or heatmap. Each still call starts with reset
history and score. Each process runs 120 timed images at 1080p or 30 at 4K,
plus three warmups, with one worker on both implementations.

| Metric | Size | Workers | VQM Vulkan | Vship Vulkan |
| --- | --- | ---: | ---: | ---: |
| CVVDP images | 1920×1080 | 1 | 253.2 (249.0–256.5) | 108.6 (108.0–111.8) |
| CVVDP images | 3840×2160 | 1 | 47.2 (46.8–47.5) | 24.2 (23.6–24.4) |

CVVDP image JOD was 9.589559 / 9.581279 (VQM / Vship) at 1080p and
9.572445 / 9.567280 at 4K. All ten published peer processes exited successfully.
A preceding 600-call 1080p Vship attempt segfaulted after writing its CSV and
was excluded, along with that incomplete matrix. It is retained in
`artifacts/reports/metric-throughput-2026-11-07/rejected-600-call-attempt/`.
The successful shorter runs do not establish long-batch peer stability.

PSNR-HVS uses the first frame of a native YUV420p8 testsrc2 Y4M and its
bicubic half-size/downscale-upscale distortion. Every process performs 1,200
(1080p) or 300 (4K) timed evaluations **total**, distributed across workers,
plus three warmups per worker. Do not multiply these iterations by the worker
count or use score-only mode. Both raw inputs are prepared on every call.
No compatible PSNR-HVS peer was measured in this matrix.

| Metric | Size | One worker | Three workers |
| --- | --- | ---: | ---: |
| PSNR-HVS | 1920×1080 | 1,652.4 (1,565.4–1,653.8) | 2,527.4 (2,462.2–2,543.1) |
| PSNR-HVS | 3840×2160 | 403.7 (400.0–404.1) | 577.0 (568.8–584.4) |

PSNR-HVS scores were 41.914886 dB (1080p) and 44.195782 dB (4K), stable across
worker counts and rounds. The native YUV workload differs from the RGB image
workload; do not rank different metrics' efficiency from these rates.

The CVVDP video measurement consumes each complete 24 fps clip in order on one
worker and finalizes whole-clip JOD. Each process runs ten timed clips plus
three warmup clips: 32 frames/clip at 1080p, 16 at 4K. The numerator excludes
warmups (320 / 160 frames); process wall time includes them, reading every
frame, input transfers, initialization and teardown. History resets between
clips, never between frames within a clip. This uses the same YUV inputs and
display settings as above, not CVVDP's independent-still algorithm.

| Metric | Size | VQM frames/s | Clip JOD |
| --- | --- | ---: | ---: |
| CVVDP video | 1920×1080 | 169.4 (168.6–169.7) | 9.65536548 |
| CVVDP video | 3840×2160 | 42.1 (41.3–42.4) | 9.56438341 |

The Vship harness currently scores independent stills only; no temporal-video
peer speedup is claimed. No resident-video path is measured here. Raw commands,
counts, scores and process receipts for all 50 successful measurements are in
`artifacts/reports/metric-throughput-2026-11-07/`.

To reproduce the missing arms, create the inputs before timing (use a new
cache directory or explicitly replace previously generated files):

```sh
mkdir -p .cache/bench/metric-throughput
ffmpeg -f lavfi -i 'testsrc2=size=1920x1080:rate=24' -frames:v 32 \
  -pix_fmt yuv420p .cache/bench/metric-throughput/orig-1920x1080.y4m
ffmpeg -i .cache/bench/metric-throughput/orig-1920x1080.y4m \
  -vf 'scale=960:540:flags=bicubic,scale=1920:1080:flags=bicubic' \
  -pix_fmt yuv420p .cache/bench/metric-throughput/dist-1920x1080.y4m
./build/vqm_bench --metric psnr-hvs --workers 3 --iterations 1200 --warmup 3 \
  --pair .cache/bench/metric-throughput/orig-1920x1080.y4m \
  .cache/bench/metric-throughput/dist-1920x1080.y4m
./build/vqm_bench --metric cvvdp --iterations 120 --warmup 3 \
  --cvvdp-ppd 75.402449345163049 \
  --pair .cache/bench/pairs/synthetic-1920x1080/orig.png \
  .cache/bench/pairs/synthetic-1920x1080/dist.png
./build/vship-build-vulkan/vship_bench --metric cvvdp --threads 1 \
  --iterations 120 --warmup 3 \
  --pair .cache/bench/pairs/synthetic-1920x1080/orig.png \
  .cache/bench/pairs/synthetic-1920x1080/dist.png
./build/vqm_bench --metric cvvdp --cvvdp-video --iterations 10 --warmup 3 \
  --cvvdp-ppd 75.402449345163049 \
  --pair .cache/bench/metric-throughput/orig-1920x1080.y4m \
  .cache/bench/metric-throughput/dist-1920x1080.y4m
```

For 4K, generate 3840×2160 clips with 16 frames and a 1920×1080 intermediate;
use 300 PSNR-HVS iterations, 30 CVVDP still iterations and ten video iterations.
Interleave five rounds, save CSVs and process exit statuses, and measure each
whole process externally. Divide only timed evaluations/frames by process wall
time. For resident compute timings instead, use the
[resident protocol](performance.md#resident-compute-throughput).

## Profiling and specialist workflows

```sh
bash bench/profile.sh --build-dir build --out-dir artifacts/reports/profile \
  --tool stages --pair original.png distorted.png
bash bench/profile_nsys.sh --list
```

The generic profiler supports built-in stage timing, perf and Callgrind.
The Nsight script is a separate capture recipe for SSIMULACRA 2 and
Butteraugli. Its capture mode requires the local Nsight gateway and setup
described in the contributor instructions. [Nsight setup](../AGENTS.md#profiling-with-nsight-systems)

`bench/run_bench.py` measures the older CLI-to-CLI workload with process
startup and decoding inside the timed interval. It supports explicit
`--baseline NAME=PATH` entries and `--stages` for a separate stage profile.
Use `--jobs 1` for an uncontended process comparison. The in-memory runner
uses a different timing boundary. [Performance methodology](performance.md)

## Local reports and probes

Keep benchmark and test reports under `artifacts/reports/` and generated
inputs under `.cache/`. Earlier checked-in reports are available in Git at
`26bdffe` under `bench/notes/`. [Reusable probes](experiments/README.md)
