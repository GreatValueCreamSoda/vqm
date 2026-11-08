# VQM — Vulkan quality metrics

VQM is a C++20 library and CLI for GPU-accelerated image and video quality
assessment with Vulkan compute.

## Supported metrics

| Metric | Better score | Guide |
| --- | --- | --- |
| SSIMULACRA 2.1 | Higher, up to 100 | [SSIMULACRA 2](docs/ssimulacra2.md) |
| Butteraugli | Lower, down to 0 | [Butteraugli](docs/butteraugli.md) |
| VMAF v0.6.1 | Higher, up to 100 | [VMAF v0](docs/vmaf.md#v061-models) |
| VMAF v1 SDR/HDR | Higher, up to 100 or 110 by preset | [VMAF v1](docs/vmaf.md) |
| ColorVideoVDP v0.5.7 | Higher, up to 10 JOD | [CVVDP](docs/cvvdp.md) |
| FP32 PSNR-HVS | Higher dB, up to `inf` | [PSNR-HVS](docs/psnr-hvs.md) |

The CLI reads PNG, JPEG and progressive Y4M, with optional BestSource compressed
video decoding. The library accepts decoded planar images and caller-owned
Vulkan buffers. See [CLI usage and score interpretation](docs/cli.md),
[C++ quick start](docs/library.md), [the unified API](docs/metrics.md), and
[all documentation](docs/README.md).

## Performance

<!-- throughput-comparison:start -->
**Scoring throughput on NVIDIA GeForce RTX 5070**, measured November 7, 2026: medians
of 5 rotating, non-overlapping rounds. Rates are pair evaluations/s
for stills and frames/s for temporal CVVDP video. Setup, decoding and
warmups are excluded; dispatch, synchronization and score readback remain.

### Streaming versus resident inputs

Streaming uploads decoded host pixels for each call. Resident inputs
are uploaded before timing. These are repeated-input benchmark rates,
not file-to-score CLI FPS.

| Metric | Workers | 1080p streaming | 1080p resident | 4K streaming | 4K resident |
| --- | ---: | ---: | ---: | ---: | ---: |
| SSIMULACRA 2 | 3 | 1,816.4 | 3,836.7 | 455.2 | 915.0 |
| Butteraugli | 3 | 351.3 | 498.4 | 72.7 | 109.0 |
| PSNR-HVS | 3 | 3,453.8 | 4,247.7 | 899.5 | 1,134.1 |
| CVVDP images | 1 | 390.4 | 500.5 | 96.3 | 122.3 |
| CVVDP video | 1 | 256.2 | 274.1 | 63.1 | 68.3 |

### Competitor comparison: streaming inputs

All columns include per-call uploads and exclude setup/decoding. Spatial
metrics use three GPU workers; CVVDP images use one.

| Metric | Size | VQM Vulkan | Vship CUDA | Vship Vulkan | Vship-fast Vulkan |
| --- | --- | ---: | ---: | ---: | ---: |
| SSIMULACRA 2 | 1920×1080 | 1,816.4 | 862.9 | 470.8 | 594.4 |
| SSIMULACRA 2 | 3840×2160 | 455.2 | 197.2 | 103.9 | 155.4 |
| Butteraugli | 1920×1080 | 351.3 | 170.5 | 148.8 | 318.1 |
| Butteraugli | 3840×2160 | 72.7 | 37.9 | 34.0 | 78.4 |
| CVVDP images | 1920×1080 | 390.4 | — | 215.0 | 360.4 |
| CVVDP images | 3840×2160 | 96.3 | — | 54.6 | 87.6 |

CVVDP video preserves temporal history and frame order on one worker; its
resident arm preuploads a complete clip. CVVDP images are independent
stills, not video. Peer scores differ; these rates do not establish
numerical equivalence. `—` means not measured. VMAF throughput is not
measured in this matrix. Benchmark workers do not change normal CLI scheduling.

[Methodology, spread, scores and limitations](bench/README.md#streaming-and-resident-scoring-comparison) ·
[Run and refresh this entire comparison](bench/README.md#refresh-the-entire-comparison)
<!-- throughput-comparison:end -->

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
