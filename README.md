# VQM — Vulkan quality metrics

VQM is a C++20 library and command-line tool for comparing images and video
with Vulkan compute. It implements SSIMULACRA 2.1, Butteraugli, floating-point VMAF
v0.6.1, standard SDR VMAF v1.0.16 3H, ColorVideoVDP v0.5.7, and FP32
PSNR-HVS. The library accepts decoded planar images in a range of color
formats and can read caller-owned Vulkan buffers directly. The CLI reads PNG,
JPEG and progressive YUV4MPEG2 clips.

[Build](#build) · [Command line](#command-line) · [C++ library](#c-library) ·
[Benchmarks](#benchmark) · [Documentation](docs/README.md)

- **SSIMULACRA 2:** higher scores are better; identical images score 100 on a
  discrete GPU. Scores may be negative.
- **Butteraugli:** lower scores are better; identical images score 0. The
  reported score is the maximum of the distance map. See the [Butteraugli
  guide](docs/butteraugli.md) for reference agreement and input-profile limits.
- **VMAF v0.6.1:** a sequence-aware perceptual score; higher is better. VQM
  implements the 211-support-vector 1080p/3H and 262-support-vector 4KTV/1.5H
  float predictors, with NEG and phone-transform options. Its SDR inputs can be
  converted on the GPU to coded YUV.
- **VMAF v1.0.16 3H:** the standard 1080p SDR sequence model. It accepts SDR
  planar inputs and supports selectable 4:2:0 or 4:4:4 processing. Other v1
  display, HFR and HDR models are not implemented.
- **ColorVideoVDP:** display-aware SDR/HDR image and video quality in JOD;
  higher is better, identity is 10 and negative scores are valid. The base
  v0.5.7 model supports explicit physical viewing conditions and whole-clip
  temporal pooling. See [CVVDP](docs/cvvdp.md).
- **PSNR-HVS:** an FP32 adaptation of libvmaf/Xiph's masked metric; higher
  dB scores are better and zero measured error scores positive infinity.
  Initial inputs are native opaque 8/10/12-bit YUV420/422/444. Reference
  ranking qualification is pending. See [PSNR-HVS](docs/psnr-hvs.md).

The implementations make documented choices that can produce different scores
from the upstream references. See [validation](docs/validation.md), [VMAF
development](docs/vmaf-development.md) and [reference differences](docs/divergence.md)
before comparing absolute scores.

The VMAF model matrix distinguishes the supported FP32 predictors, feature
variants, and derived extensions. [VMAF models](docs/vmaf-development.md#v061-models)

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

```sh
./build/vqm original.png distorted.png
./build/vqm --metric butteraugli original.png distorted.png
./build/vqm --metric cvvdp original.png distorted.png
./build/vqm --metric vmaf-v0 original.y4m distorted.y4m
./build/vqm --metric vmaf-v1 --json original.y4m distorted.y4m
./build/vqm --metric cvvdp original.y4m distorted.y4m
./build/vqm --metric psnr-hvs --json original.y4m distorted.y4m
```

The CLI supports PNG, JPEG and progressive YUV4MPEG2. All six metrics consume
every Y4M frame in order, with matching geometry, rational frame rates and frame
counts. SSIMULACRA 2, Butteraugli, VMAF and PSNR-HVS return the arithmetic mean of their
frame scores; CVVDP returns its whole-clip JOD after temporal distortion pooling.
PNG and JPEG are interpreted as sRGB with BT.709 primaries; the CLI also offers
input color overrides and a `--raw` mode. Run `./build/vqm --help` for the
options. For SSIMULACRA 2 and Butteraugli images with alpha, the CLI evaluates
dark and bright backgrounds and returns the worse score. CVVDP requires
opaque inputs; PSNR-HVS requires native opaque
YUV and encodes positive infinity as the JSON string `"inf"`.
[Run the three-frame Y4M example](docs/temporal-validation.md#score-a-complete-video).

`--json` prints one object per complete input pair, including `score`, `frames`,
ordered `frame_results`, `core_seconds` and `decode_seconds`. CVVDP has a clip
score and an empty `frame_results` array. Use `--pairs-file` with a TSV containing
sample ID, original path and distorted path to process several clips in one
process:

```sh
./build/vqm --metric vmaf-v0 --json --pairs-file pairs.tsv \
  --iterations 5 --warmup 1
```

Each iteration reads and scores the whole clip after resetting its temporal
history. The reported timing selects the iteration with median core time; decode
time is separate. `--json` and `--raw` are separate output modes.

### NVIDIA hardware decoding

On Linux, the build enables NVDEC automatically when it finds the CUDA Toolkit
(11.8 or newer) and FFmpeg development libraries (6.1 or newer). Use
`-DVQM_ENABLE_NVDEC=OFF` to build without the NVIDIA backend or its dependencies.
`-DVQM_ENABLE_NVDEC=ON` requires the backend at configure time. At runtime,
NVDEC needs a compatible NVIDIA driver, NVDEC-enabled FFmpeg, and a Turing or
newer GPU with NVDEC and Vulkan opaque FD interop.

```sh
cmake --preset release -B build-nvdec
cmake --build build-nvdec --parallel 8
./build-nvdec/vqm --metric vmaf-v1 --json original.mp4 distorted.mp4
```

Compressed video pairs select NVDEC automatically when built in. The
`--nvdec` option explicitly requires compressed video inputs. NVDEC works with
all six metrics and with `--pairs-file`, `--iterations` and `--warmup`. It
reads compressed clips in presentation order and requires
progressive, constant-rate inputs with matching dimensions, rational frame
rates and frame counts. Supported surfaces are 8-, 10- and 12-bit YUV 4:2:0,
4:2:2 and 4:4:4 where the codec, GPU and selected metric support them. The
CLI uses decoded color metadata and accepts the same `--input-*` overrides
as Y4M. Unsupported decoding and changes to geometry or format within a clip
are errors.

FFmpeg demuxes compressed packets on the CPU. NVDEC decodes pixels on the
GPU, a CUDA kernel packs them into shared device memory, and Vulkan reads
that memory to compute the metrics. The CLI reads back score data. Decoded
frames stay on the GPU, including during layout conversion. The core library
has no FFmpeg or CUDA dependency.

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
the native or fit-and-pad geometry. `temporal_weighting = false` is a derived
mode for content where motion2 is misleading: it forces motion2 to zero, skips
the motion passes, and scores every frame immediately.
`VmafV1Pipeline` in `vmaf_v1.h` provides the standard v1 model with its own
frame-boundary behavior. Set `VmafV1Config::chroma_sampling` to
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
