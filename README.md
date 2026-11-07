# VQM — Vulkan quality metrics

VQM is a C++20 library and command-line tool for comparing images and video
with Vulkan compute. It implements SSIMULACRA 2.1, Butteraugli, floating-point VMAF
v0.6.1, SDR and HDR VMAF v1, ColorVideoVDP v0.5.7, and FP32
PSNR-HVS. The library accepts decoded planar images in a range of color
formats and can read caller-owned Vulkan buffers directly. The CLI reads PNG,
JPEG and progressive YUV4MPEG2 clips.

[Build](#build) · [Command line](#command-line) · [C++ library](#c-library) ·
[Benchmarks](#benchmark) · [Documentation](docs/README.md)

The [metric guide](#command-line) covers score direction, clip pooling and
viewing settings. See [validation](docs/validation.md) and
[reference differences](docs/divergence.md) for comparisons with upstream
implementations.

## Build

You need a C++20 compiler, CMake 3.25 or newer, Ninja, the Vulkan loader and
headers, `glslc`, libpng and libjpeg. The complete local test suite also uses
`uv`, Python, NumPy, Pillow and FFmpeg. A Vulkan 1.1 compute device and driver
are needed to run the metrics; a software Vulkan driver can run them without
a physical GPU.
Apple platforms are not supported. The bundled presets use Ninja; another
CMake generator can be used with a manual configuration.

On Ubuntu 24.04, for example, the build dependencies are available as:

```sh
sudo apt install build-essential cmake ninja-build libvulkan-dev glslc \
  libpng-dev libjpeg-dev ffmpeg
```

Install `uv` and a Vulkan driver for your system as well, then prepare the
lightweight Python test environment, build and test locally:

```sh
if [ ! -d .venv ]; then
  uv venv --no-project --python 3.12 .venv
fi
uv pip install --python .venv/bin/python -r tools/test-requirements.txt
cmake --preset release -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build --preset release -j
ctest --preset release --output-on-failure
./build/vqm_test --require-gpu
```

The final command requires a usable Vulkan device and fails if GPU-dependent
tests would be skipped. Presets for `debug` and `asan` are also available.
Testing is local; this project does not provide or support CI testing or
workflows. The [contributing guide](CONTRIBUTING.md) covers quality, sanitizer
and synchronization-validation checks.
Use `vqm_bench` for stage timings and `vqm_test` for the C++ test suite.
To build only the library, disable the CLI, benchmarks and tests with
`-DVQM_BUILD_CLI=OFF -DVQM_BUILD_BENCH=OFF -DVQM_BUILD_TESTS=OFF`; libpng and
libjpeg are then unnecessary.
The standalone core build needs no Python environment, external reference
checkout or downloaded corpus.

For end-to-end reference correlation across the five established metrics, use the
[hosted video validation command](#reference-correlation) after building.

## Command line

Pass the original first and the distorted rendition second. Use matching
dimensions and align video frames to the same content, frame rate and frame
count. The CLI reads PNG, JPEG and progressive Y4M, plus compressed clips with
[BestSource indexed hardware decoding](#bestsource-indexed-hardware-decoding).

```sh
./build/vqm original.png distorted.png
./build/vqm --metric ssimulacra2,butteraugli original.png distorted.png
./build/vqm --metric all --verbose original.y4m distorted.y4m
```

Use `--list-devices` to see compatible Vulkan compute devices with their
zero-based indexes, names, types, and UUIDs. Pass an index to `--device` to
select it for the whole invocation, including batch runs.

```sh
./build/vqm --list-devices
./build/vqm --device 0 --metric all original.y4m distorted.y4m
```

SSIMULACRA 2 is the default. Use a comma-separated list or repeat `--metric`
to run several metrics on the same pair. `all` runs the six metrics below
in table order. Each metric runs once in first-selected order. The pair must meet
every selected metric's input requirements.

| CLI metric | Better score | How the CLI scores a clip |
| --- | --- | --- |
| [`ssimulacra2`](#ssimulacra-2) | Higher, up to 100 | Mean of frame scores |
| [`butteraugli`](#butteraugli) | Lower, down to 0 | Mean of frame diffmap maxima |
| [`vmaf-v0`](#vmaf-v0) | Higher, up to 100 | Mean of model-predicted frame scores |
| [`vmaf-v1`](#vmaf-v1) | Higher, up to 100 or 110 by preset | Mean of model-predicted frame scores |
| [`cvvdp`](#colorvideovdp) | Higher, up to 10 JOD | Pools whole-clip distortion, then converts to JOD |
| [`psnr-hvs`](#psnr-hvs) | Higher dB, up to `inf` | Averages weighted frame errors, then converts to dB |

Compare renditions of the same original using the same metric, color
interpretation, model and viewing settings. The metrics use different scales,
so compare their rankings separately.

PNG and JPEG use sRGB with BT.709 primaries. Y4M defaults to limited range,
the BT.470 BG matrix, and BT.709 transfer and primaries. Set `--input-range`,
`--input-matrix`, `--input-transfer` and `--input-primaries` when the samples
use different colorimetry. These overrides apply to both inputs. Run
`./build/vqm --help` for accepted values.

### SSIMULACRA 2

```sh
./build/vqm --metric ssimulacra2 original.png distorted.png
```

Use SSIMULACRA 2 for visible image artifacts such as blur and ringing. Identity
scores 100 on a discrete GPU. As rough landmarks from the original calibration,
70 is good visual quality, 90 is close to visually lossless, and negative
scores indicate severe distortion. For images with alpha, the CLI scores dark
and bright backgrounds and keeps the worse result.
[Original calibration](https://github.com/cloudinary/ssimulacra2#usage) ·
[VQM implementation](docs/ssimulacra2.md)

### Butteraugli

```sh
./build/vqm --metric butteraugli original.png distorted.png
```

A frame's score is the maximum of its perceptual distance map. Identity scores
0, and a small area with a strong defect can drive the score up. For a clip,
**Frame max** is the worst frame score. Use `--raw` to see the per-frame
`distance_p3` as well, which pools distances across the image. The CLI keeps the
worse result across dark and bright backgrounds for images with alpha.
`--intensity-target` and `--hf-asymmetry` default to 80 nits and 1.0.
[Butteraugli score details](docs/butteraugli.md#score-scale)

### VMAF v0

```sh
./build/vqm --metric vmaf-v0 --vmaf-model 4k --vmaf-scale --vmaf-neg \
  original.y4m distorted.y4m
```

The CLI uses floating-point v0.6.1 SDR predictors. Select `--vmaf-model 1080`
for 1080p at 3H (the default) or `--vmaf-model 4k` for 4K at 1.5H. Higher means
better predicted quality on the 0–100 scale. The clip score averages the frame
predictions, which include reference motion.

Use `--vmaf-neg` to disable enhancement gain when comparing encoders.
`--vmaf-transform` applies the phone score transform calibrated on the 1080p
model. VQM also offers both options with the 4K predictor as derived variants.
The phone transform can saturate 4K scores at 100.

`--vmaf-scale` and `--vmaf-pad` apply to both selected SDR VMAF versions.
Scaling fits each metric's input within its selected model canvas while
preserving aspect ratio. Padding adds black bars to reach 16:9 at the current
scale. Together the flags fit and pad to each model's full canvas. By default
both versions score at the input dimensions.
[VMAF v0 model details](docs/vmaf-development.md#v061-models)

### VMAF v1

```sh
./build/vqm --metric vmaf-v1 --vmaf-v1-model 4k-1.5h --vmaf-scale \
  original.y4m distorted.y4m
```

VMAF v1 includes features for banding and chroma detail. Select a preset for
the viewing setup. The suffix `h` expresses viewing distance in screen heights.
Use the same preset when comparing renditions.

| `--vmaf-v1-model` | Viewing setup | Score range |
| --- | --- | --- |
| `1080p-3h` (default) | SDR 1080p at 3H | 0–100 |
| `4k-1.5h` | SDR 4K at 1.5H | 0–100 |
| `4k-3h` | SDR 4K at 3H | 0–110 |
| `phone-5h` | SDR phone at 5H | 0–100 |
| `hdr-4k-1.5h` | HDR 4K at 1.5H | 0–100 |

Add `--vmaf-v1-hfr` to any preset for roughly 50/60 fps content. It selects
five-frame motion and smoothing. The HDR preset automatically scales the short
side to 2160 pixels, keeping the aspect ratio. The example below declares the
colorimetry of limited-range HDR10 in Y4M.

```sh
./build/vqm --metric vmaf-v1 --vmaf-v1-model hdr-4k-1.5h \
  --input-matrix bt2020ncl --input-transfer st2084 --input-primaries bt2020 \
  --input-range limited original-hdr.y4m distorted-hdr.y4m
```

[VMAF v1 presets and reference details](docs/vmaf-development.md#v1016-sdr-models)

### ColorVideoVDP

```sh
./build/vqm --metric cvvdp --cvvdp-display compact --cvvdp-peak 200 \
  original.y4m distorted.y4m
```

ColorVideoVDP scores opaque SDR and HDR inputs in just-objectionable
differences (JOD) under a specified viewing setup. Identity scores 10 JOD and
lower scores indicate more objectionable distortion.
Negative scores are valid. Under the model's calibration, a 1 JOD advantage
corresponds to a predicted 75% preference in a paired comparison.
[ColorVideoVDP paper](https://www.cl.cam.ac.uk/~rkm38/pdfs/mantiuk2024_ColorVideoVDP.pdf)

The default `standard` is a 30-inch 16:9 display viewed from 0.7472 m. `compact` is
24 inches at 0.6 m. Override the aspect ratio, diagonal in inches or distance
in metres with `--cvvdp-aspect`, `--cvvdp-diagonal` and `--cvvdp-distance`.
`--cvvdp-peak` sets peak brightness in nits and defaults to 200. Keep these
conditions fixed when ranking encodes. Read **Aggregate** for the whole-clip
JOD and use the temporally filtered frame diagnostics to locate problem frames.
[CVVDP display and input options](docs/cvvdp.md#display-and-input-interpretation)

### PSNR-HVS

```sh
./build/vqm --metric psnr-hvs --json original.y4m distorted.y4m
```

Use matching opaque 8-, 10- or 12-bit planar YUV 4:2:0, 4:2:2 or 4:4:4 inputs.
The metric measures their native sample codes. Higher dB means lower weighted
error. A gain of about 3 dB corresponds to halving that error. `inf` means
zero measured error, which includes identical frames and can also occur when
masking or block coverage excludes the differences. A clip reports `inf` only
when every frame has zero measured error. JSON writes infinity as the string
`"inf"`. Use the per-frame Y, Cb and Cr results to see which planes contribute
to the distortion.
[PSNR-HVS inputs and error weighting](docs/psnr-hvs.md)

### Reading the report

Use `--verbose` (or `-v`) for an aligned table. This run scored a three-frame,
320×240 clip with a translucent box over the middle frame. Scoring FPS
depends on the input, device and selected metrics.

```text
Frames: 3
Scoring FPS: 746.57
Metric             Aggregate       Frame min       Frame max   Finite stddev  Perfect
ssimulacra2        65.688104      -2.9356868             100       48.524348        2
butteraugli        29.944338               0       89.833015        42.34769        2
vmaf-v0            97.460145       95.452462       99.500102       1.6525997        0
vmaf-v1            98.612725       95.838175             100       1.9619032        2
cvvdp              9.6586087       9.5719113              10      0.19466893        1
psnr-hvs           37.861903       33.090691             inf               0        2
```

- **Aggregate** uses the clip calculation for each metric. PSNR-HVS stays
  finite here because the altered frame contributes error to the clip.
- **Frame min / max** show the extremes. Read the minimum for higher-is-better
  metrics and the maximum for Butteraugli to find the worst frame score.
- **Finite stddev** measures the spread of finite frame scores as a population
  standard deviation. One finite score gives 0. With no finite scores it
  displays `n/a`.
- **Perfect** counts scores that equal the metric's best value, including
  `inf` for PSNR-HVS. Motion can affect a frame's VMAF prediction, and clipping
  can put distorted frames at the ceiling.
- **Scoring FPS** divides the shared frame count by the sum of the selected
  metrics' core scoring times. Decode time is measured separately.

Plain text prints a bare score for one metric and labeled scores for multiple
metrics. `--raw` prints additional metric-specific values for a single metric.
`--json` emits one object per pair and metric, with the clip `score`, ordered
`frame_results` with zero-based indices, `core_seconds` and `decode_seconds`.
Add `--verbose` for `scoring_fps`, `frame_stats` including the finite-frame
mean, and CVVDP's diagnostic frame JODs. Without it, CVVDP returns its clip score with an empty
`frame_results` array. Unavailable statistics are `null`.

Use `--pairs-file` with `--json` to process a batch. Each TSV row holds a
unique sample ID, original path and distorted path. `--iterations` repeats
the complete clip with reset history and reports the iteration with median
combined core time. `--warmup` runs untimed clips first.

```sh
./build/vqm --metric vmaf-v0 --json --pairs-file pairs.tsv \
  --iterations 5 --warmup 1
```

[Complete video and batch examples](docs/temporal-validation.md#score-a-complete-video)

### BestSource indexed hardware decoding

On Linux, build with `-DVQM_ENABLE_BESTSOURCE=ON` to enable compressed video
input. The CLI detects compressed video pairs by their contents and decodes
them with BestSource. CMake fetches pinned BestSource R23 and libp2p sources,
verifies their archive hashes, and builds both into static libraries linked
into the CLI. [BestSource R23 release](https://github.com/vapoursynth/bestsource/releases/tag/R23)

The build needs a Python 3 interpreter to embed the GPU shaders. FFmpeg
and xxHash are external dependencies. Make their development packages
visible to `pkg-config` and their shared libraries visible to the runtime
linker. Use an FFmpeg build with Vulkan Video, libavutil 61.6 or newer, and
the Vulkan UUID device selector. For a non-system prefix, add its
`lib/pkgconfig` directory to `PKG_CONFIG_PATH` before configuring.

```sh
cmake --preset release -B build-bestsource -DVQM_ENABLE_BESTSOURCE=ON
cmake --build build-bestsource --parallel 8
./build-bestsource/vqm --start-frame 174 --frame-count 24 \
  --metric vmaf-v1 --json original.mp4 distorted.mp4
```

`--start-frame` is zero-based and applies to both inputs. It defaults to zero.
`--frame-count` sets the exact number of frames to score and must fit in both
clips. With no count, both inputs must end together. Metric history starts at
the first selected frame, and per-frame output indexes are relative to that
frame. BestSource builds and caches the frame index as needed. It can
temporarily download candidate frames while verifying a seek. Once selected,
frames are packed and scored in Vulkan device memory.

The backend requires the same physical GPU for decoding and scoring, Vulkan
opaque FD memory and semaphore sharing, and hardware support for the input
codec. `--device` selects that GPU for both operations. The device list
checks baseline compute support, so a listed device may lack Vulkan Video
support for a given codec or format. The backend accepts local progressive,
constant-rate 8-, 10- and 12-bit YUV 4:2:0 and 4:2:2 clips, subject to each
metric's format requirements. The CLI uses decoded color metadata and accepts
the same `--input-*` overrides as Y4M.
This backend is CLI-only and requires hardware decoding at runtime.

## C++ library

Install the core and its CMake package into a prefix.

```sh
cmake --install build --prefix /path/to/prefix
```

Configure the consumer with `-DCMAKE_PREFIX_PATH=/path/to/prefix` if the
prefix is outside CMake's default search paths. In the consumer, find the
package and link its imported target.

```cmake
find_package(vqm CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE vqm::core)
```

With `add_subdirectory`, link the same target. The CLI, benchmark and tests
default to off in an embedded build. The core links the Vulkan loader without
requiring libpng or libjpeg. The installed core is a static C++ library that
can also link into a shared library. Use compatible C++ compiler and standard
library settings for the core and consumer.

Use `vqm::ListDevices()` from `<vqm_device.h>` to discover compatible
compute devices. Pass the chosen UUID to `vqm::SetDefaultDevice(uuid)`
before the first operation that uses the library-owned device. This choice
applies to default pipelines, pools, and memory queries in the process.
Applications managing their own Vulkan devices can select a device per
pipeline with the existing `VulkanDevice` overloads.

```cpp
#include <vqm_metrics.h>

// Populate these with decoded, planar pixels and their format descriptors.
vqm::Image original, distorted;
vqm::MetricsConfig config;
config.size = original.size;
config.original_format = original.format;
config.distorted_format = distorted.format;
config.ssimulacra2 = vqm::Ssimulacra2Config{};
config.butteraugli = vqm::ButteraugliConfig{};
vqm::Metrics metrics(config);
auto scores = metrics.ComputeImage(original, distorted);
double ssimu2 = *scores.ssimulacra2;
double butter = *scores.butteraugli;
```

The [unified metrics guide](docs/metrics.md) covers VMAF, video sequences,
external buffers, and memory planning. Lower-level pipelines remain available.
A pipeline has a fixed geometry and is used by one host thread. It also
supports uploaded images, borrowed host planes and caller-owned Vulkan buffers.
`ButteraugliPipeline` offers the corresponding API for Butteraugli. For batch
work, `Ssimulacra2Pool` keeps a bounded number of pipelines in flight. The core
API evaluates one configured alpha background; `ComputeWorstBackground` in
`vqm_alpha.h` implements the CLI's policy.
`VmafPipeline` in `vmaf.h` accepts ordered frame pairs and emits scores with
the motion look-ahead described in [VMAF development](docs/vmaf-development.md).
`VmafConfig::model` selects the 1080 or 4K predictor; `neg` and `transform`
toggle the feature-gain and phone-score options, and `fit`/`model_size` select
the evaluation geometry. `temporal_weighting = false` is a derived
mode for content where motion2 is misleading: it forces motion2 to zero, skips
the motion passes, and scores every frame immediately.
`VmafV1Pipeline` in `vmaf_v1.h` selects a viewing preset through
`VmafV1Config::model`. Its `fit` setting can scale or pad luma and chroma.
Set `VmafV1Config::chroma_sampling` to
`VmafChromaSampling::Yuv420` or `Yuv444` to convert SDR inputs to a common
BT.709 representation, including mixed source formats; the default `Native`
mode preserves compatible coded YUV. The benchmark exposes the same selection
as `--vmaf-chroma native|420|444`.

See the [API and implementation guide](docs/reference.md#core-api) for input
layout, lifetime and synchronization requirements, and
[Vulkan synchronization](docs/vulkan-synchronization.md) for interoperation
with other Vulkan work. [GPU memory planning](docs/memory.md) describes how to
query per-metric buffer requirements and estimate worker counts before allocating.
[Shared workspaces](docs/shared-workspace.md) let SSIMULACRA 2 and Butteraugli
reuse an arena when they run sequentially.

## Benchmark

Use `vqm_bench` for a local stage timing on an existing image pair.

```sh
./build/vqm_bench --pair original.png distorted.png --iterations 20 --warmup 3
```

The hosted video comparison fetches pinned inputs and prepares the reference
environment. Use `--dry-run` to inspect its commands before running it. The
still-image suite compares selected peer implementations on a single pair.

```sh
uv run --no-project --python 3.12 bench/benchmark.py --dry-run
uv run --no-project --python 3.12 bench/benchmark.py --suite images --help
```

See the benchmark guide for dependencies, workloads and report locations.
[Benchmark guide](bench/README.md)

## Reference correlation

Run hosted video qualification for the five established metrics against their
independent originals with one command:

```sh
.venv/bin/python -m validation.temporal.run --profile qualification
```

The default matrix includes SSIMULACRA 2, Butteraugli, VMAF v0, VMAF v1 and
CVVDP, with 390 complete clip pairs per metric. The runner downloads missing
pinned source prefixes automatically and scores through the normal `build/vqm`
CLI. Its `summary.json` reports each selected metric's Spearman/Kendall
correlation, ordering coverage, raw numerical errors and gate outcome. Normal
runs use checked-in original scores and do not require the optional original
implementations to be installed.
PSNR-HVS reference-correlation qualification is pending and is outside this
five-metric matrix.

`--profile smoke` selects a quick check with 39 pairs per metric. Ranking is
the default for these registered qualified profiles and the frozen
source-balanced CID22 still policies. Raw numerical errors remain diagnostic;
`--gate-mode both` additionally requires numerical parity. Unregistered or
changed policies retain the combined default. Identity and numerical component
tests remain required. The [default-gate transition](docs/design/ranking-default-gates.md)
records this scope. See the
[workflow guide](docs/temporal-validation.md) for the separately hashed spatial
and temporal representations, source notices and exact qualification policy,
and the [five-metric reference-correlation record](bench/notes/system/all-metric-reference-correlation.md)
for measured results. Historical three-metric and still-image reports retain
their original scope.

The full [source manifest](validation/temporal/sources.json) pins the first 40
frames of 30 hosted Xiph SD sequences, about 180 MB in total. It links the source
catalogue and its per-source copyright notices; source videos stay outside version
control.

## Documentation

- [API and implementation guide](docs/reference.md)
- [Metric details, provenance and documentation index](docs/README.md)
- [Validation and reference differences](docs/validation.md)
- [Performance methodology](docs/performance.md)
- [Contributing](CONTRIBUTING.md)

## License and provenance

VQM is derived from Cloudinary's SSIMULACRA 2 and includes Butteraugli
material from the JPEG XL Project, under BSD-3-Clause. See
[LICENSE](LICENSE), the [JPEG XL license](licenses/JPEG-XL-LICENSE), and
the [upstream JPEG XL patent grant](licenses/JPEG-XL-PATENTS).
Other metric material retains its upstream notices:

- [Netflix VMAF](licenses/NETFLIX-VMAF-LICENSE): BSD-2-Clause-Patent, with a
  separate [libsvm notice](licenses/LIBSVM-LICENSE).
- [ColorVideoVDP](licenses/COLORVIDEOVDP-LICENSE): MIT.
- [Xiph PSNR-HVS](licenses/XIPH-PSNR-HVS-LICENSE): permissive two-clause notice.

Fetched reference and peer implementations have their own licenses. The
[provenance notes](docs/README.md#provenance) explain source ownership and
attribution. Hosted videos are downloaded from their source catalogue and
remain outside version control.
