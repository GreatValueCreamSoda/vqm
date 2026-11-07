# Reusable precision experiments

These small probes preserve the numerical examples behind the
[VMAF FP16 research](../notes/vmaf/research/README.md). Their source belongs in
version control; generated JSON, CSV and diagnostic logs belong in
`artifacts/reports/precision-examples/`. Run the commands below from the repository
root. The Python examples require Python 3 and NumPy.

## CPU range and rounding examples

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

## GPU sequence feature probe

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

## Provenance and historical reproduction

These helpers were promoted from the local `build/reports/vmaf-fp16/` study:

| Maintained helper | Original report-relative path |
| --- | --- |
| `vmaf_precision_probe.cc` | `sequence_probe.cc` |
| `vmaf_precision_examples.py` | `experiments/vif-adm-numerics.py` |
| `vmaf_motion_precision.py` | `motion/cpu_precision_examples.py` |

The promotion reorganized the code for readability and moved generated outputs
out of the source tree. Both CPU scripts reproduced the original substantive
JSON exactly when relocated. The associated research notes preserve the
historical hardware, build settings, measurements and rejected approaches.
Historical baseline executables, patched shader paths and raw commands in those
notes describe their original experiments; they are not prerequisites for these
current-tree recipes. Recreating an experimental comparison requires the source
revision and variant described in its research note.

The [cooperative-matrix tools](cooperative_matrix/README.md) extend this work to
FP16 operands with FP32 matrix accumulation, including a private-device probe,
CPU conditioning examples and isolated SpEED variants.

The [SpEED parallelism tools](speed_parallelism/README.md) preserve diagnostic
timestamp brackets, subgroup and phased solver trials, the historical matrix
stress generator, and compatibility comparisons.

The [VMAF v1 headroom tools](v1_headroom/README.md) reproduce CAMBI storage,
tile, pooling and synchronization trials, including the retained subgroup
pipeline and an exact patch for reconstructing its pre-change baseline.

The [Butteraugli drift tools](butteraugli_drift/README.md) preserve isolated
frequency/Malta correction patches, a deterministic checker pair generator,
an independent CPU oracle runner and the JPEG ICC isolation probe.

- [`cvvdp/`](cvvdp/README.md): compare planned and actual CVVDP auxiliary allocations across geometry/frame-rate limits.
