# Run benchmarks

Run the commands from the repository root.

| Workload | Entry point |
| --- | --- |
| Time GPU stages on one decoded image pair | [`vqm_bench`](#time-one-pair) |
| Score and time complete clips | [`vqm --json`](#score-a-complete-clip) |
| Compare complete hosted clips against pinned original scores | [`benchmark.py`](#hosted-video-benchmark) |
| Compare still-image latency with a pinned peer | [`benchmark.py --suite images`](#still-image-peer-comparison) |
| Compare latency, exploratory windows or fixed-count workers | [Performance methodology](../docs/performance.md) |

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
how to compare two builds in one sitting. [Performance methodology](../docs/performance.md)

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
matrix. [Hosted video validation](../docs/temporal-validation.md)

## Hosted video benchmark

The default suite fetches the pinned Cloudinary, libjxl, libvmaf and
ColorVideoVDP sources, builds the CLI and native references, and prepares the
locked CPU reference environment in `.venv/`. It fetches missing checksum-pinned
Xiph source prefixes and uses checked-in original scores for agreement checks.

```sh
uv run --no-project --python 3.12 bench/benchmark.py --dry-run
uv run --no-project --python 3.12 bench/benchmark.py
```

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
timing boundaries recorded there. [Performance methodology](../docs/performance.md)

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
SSIMULACRA 2 only. The CVVDP image suite supports Vship Vulkan. CUDA and
fssimu2 need their own toolchains. Reports default to
`artifacts/reports/quick-benchmark/`.

Vship exposes device names and indices. The runner requires a unique name
match against VQM's selected device and records the available identities in
JSON. After checking the devices manually, use `--allow-unverified-devices`
to permit the explicitly marked device-zero fallback.

Reference checkouts must be clean at their pinned commits, including submodules.
The build tools work on copies and rebuild when the recorded recipe, source or
toolchain fingerprint changes. Fetch or build a named reference with
`tools/fetch_references.py --only NAME` or
`tools/build_references.py --only NAME`.
[Reference setup and validation](../docs/validation.md)

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
uses a different timing boundary. [Performance methodology](../docs/performance.md)

## Archived results

Keep raw local runs under `artifacts/reports/` and generated inputs under
`.cache/`. The research index points to dated results, preserved command
lines and reusable probes. [Research and measurement archive](notes/README.md)
