# Benchmark probes

Run these standalone diagnostics from the repository root. Save generated
reports under `artifacts/reports/`. [Benchmark guide](../README.md)

## VMAF precision

The CPU examples require Python 3 and NumPy.

### CPU range and rounding examples

```sh
python3 bench/experiments/vmaf_precision_examples.py
python3 bench/experiments/vmaf_motion_precision.py
```

Each command creates its output directory and writes JSON there:

| Script | Default output | Question illustrated |
| --- | --- | --- |
| `vmaf_precision_examples.py` | `artifacts/reports/precision-examples/vmaf-precision.json` | Native-code representability, VIF moment cancellation and four-level ADM coefficient/cube ranges. |
| `vmaf_motion_precision.py` | `artifacts/reports/precision-examples/vmaf-motion-precision.json` | Motion changes under half input, intermediate, product and accumulator precision. |

Both scripts also print a compact summary to stdout and accept `--out PATH` to
choose another JSON destination. Their defaults resolve from the repository
location, so the scripts also work when invoked from another working directory.
Importing either module runs no diagnostics and writes no files.

The examples retain the original explicit FP32/FP16 casts and operation order.
VIF and DWT filter coefficients are read from the maintained metric shaders;
changes to those arrays can therefore change the example results. ADM CSF values
are fixed illustrative weights from the original range study. The motion filter
weights and deterministic seed are also retained. VIF uses Python `math.log2`,
and all examples use NumPy operations with separate additions. GPU contraction,
rounding, denormal handling and the production shader's arithmetic may differ.
These examples illustrate numerical failure modes; they are not independent
reference oracles or performance measurements.

### GPU sequence feature probe

`vmaf_precision_probe.cc` emits six-frame synthetic sequences for both VMAF
implementations at 8, 10, 12 and 16 bits. Its six scenes exercise flat low-bit
steps, near-white motion, low-bit texture, near-black inputs, chroma texture and
extreme motion. Every sequence gets a fresh pipeline, including its own temporal
history. The probe uses the current production shaders and configuration; it does
not enable or reconstruct any historical experimental FP16 variant.

Build the core library normally, then compile the standalone probe against it:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vqm_core -j8
c++ -std=c++20 -O2 -I include -I src \
  bench/experiments/vmaf_precision_probe.cc build/libvqm_core.a \
  -lvulkan -pthread -o build/vmaf_precision_probe
mkdir -p artifacts/reports/precision-examples
./build/vmaf_precision_probe \
  > artifacts/reports/precision-examples/sequences.csv \
  2> artifacts/reports/precision-examples/platform.log
```

The probe uses internal Vulkan context access only to print device FP16 capability
flags to stderr and to shut down the context after all pipelines have been
destroyed. It is a repository research tool, not an installed API example. The
build directory remains disposable, and no additional CMake target is required.

CSV columns include the scene, implementation (`v0` or `v1`), depth, frame,
score and that implementation's public features. Features not exposed by that
implementation have empty cells. A complete run has 288 frame rows plus the
header. Inspect the `error` column: sequence failures are reported as rows with
frame `-1`, and the process continues to the next sequence. Exit status 2 means
that no Vulkan context was available; exit status 0 alone does not prove every
sequence succeeded.

These rows contain no timers. Performance claims must use `build/vqm_bench` and
the interleaved protocol in [AGENTS.md](../../AGENTS.md).

Pass `--unclipped` to disable the final 0–100 score clipping in both pipelines.
Use that arm when comparing arithmetic: saturated scores can hide feature
errors. The default retains the original clipped probe behavior; compare
like-for-like settings.

## Cooperative matrices

The Vulkan probe checks FP16 operands with FP32 accumulators and results
for the three tested subgroup shapes. It requires a C++20 compiler,
Vulkan headers and loader, `glslc`, `spirv-val`, and `spirv-dis`.
Pass `--build-only` to compile and validate the shader modules without a
GPU, or `--validation` to run with the Khronos validation layer.
The NumPy examples isolate operand rounding and covariance error.

```sh
.venv/bin/python bench/experiments/cooperative_matrix/run_platform.py
.venv/bin/python bench/experiments/cooperative_matrix/speed_numerics.py
.venv/bin/python bench/experiments/cooperative_matrix/feature_numerics.py
```

## Butteraugli score drift

### Checker pair and independent oracle

Run from the repository root after the Release build. The generator uses
Pillow and writes a deterministic 64×64 checker pair with a tab-separated
list of absolute paths.

```sh
mkdir -p artifacts/reports
probe_dir=$(mktemp -d "$PWD/artifacts/reports/butteraugli-drift.XXXXXX")
.venv/bin/python bench/experiments/butteraugli_drift/make_checker.py \
  --out-dir "$probe_dir"
c++ -O3 -std=c++20 -Iinclude -Isrc -Itools/image_io \
  -Itests/metrics/butteraugli \
  bench/experiments/butteraugli_drift/oracle_probe.cc \
  tests/metrics/butteraugli/butter_ref.cc build/libvqm_io.a build/libvqm_core.a \
  -lpng -lz -ljpeg -lvulkan -pthread -o "$probe_dir/oracle_probe"
"$probe_dir/oracle_probe" --pairs-file "$probe_dir/checker.pairs"
./build/vqm_bench --metric butteraugli --pairs-file "$probe_dir/checker.pairs" \
  --score-only --iterations 1 --warmup 0
```

The oracle accepts opaque integer RGB or grayscale PNG/JPEG and prints both
the diffmap maximum and `distance_p3`. It uses the image decoder's declared
sRGB interpretation.

### JPEG ICC isolation

`strip_jpeg_icc.py` removes ICC APP2 segments before the first JPEG scan,
copies the scan bytes unchanged, and requires a new output path. Compare
the same original against the embedded-profile and stripped JPEGs, keeping
all other inputs and settings fixed. Verify that both JPEGs decode to the
same pixels before attributing a score change to the profile.

```sh
.venv/bin/python bench/experiments/butteraugli_drift/strip_jpeg_icc.py \
  embedded.jpg "$probe_dir/no-icc.jpg"
```

## CVVDP memory

`memory_probe.cc` compares queried Vulkan buffer requirements to actual
allocations at 1080p and 4K, each at 24/30/60/120 fps. It counts individual
temporal allocations, including their memory type and heap, and constructs only
one pipeline at a time. Device-limit rejection is reported separately from
allocation failures, which fail the process. A successful query does not promise
enough available VRAM. The 4K120 case can require roughly 7 GiB.

From the repository root, after a release build:

```sh
c++ -std=c++20 -O2 -Iinclude bench/experiments/cvvdp/memory_probe.cc \
  build/libvqm_core.a -lvulkan -o build/cvvdp_memory_probe
./build/cvvdp_memory_probe
```

Use a separate machine/device-specific result file under `artifacts/reports/`.
The reference guide describes the numerical oracles and compatibility limits.
[CVVDP reference validation](../../validation/cvvdp.md)
