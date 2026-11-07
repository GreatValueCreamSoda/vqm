# Cooperative-matrix research tools

This standalone Vulkan probe checks FP16 matrix operands with FP32 accumulator
and result types. It has no performance timers and does not change the core
library. The shared findings and API limitations are recorded in
[`../../notes/vmaf/research/cooperative-platform.md`](../../notes/vmaf/research/cooperative-platform.md).

From the repository root:

```sh
python3 bench/experiments/cooperative_matrix/run_platform.py --validation
```

The runner needs a C++20 compiler, Vulkan headers/loader, `glslc`, `spirv-val`
and `spirv-dis`. Validation additionally requires the Khronos validation layer.
It writes compiled files into `build/cooperative-matrix-probe/` and results,
compiler commands, source/module/binary hashes, SPIR-V disassembly and logs into
ignored `artifacts/reports/vmaf-cooperative-matrix/platform/`. `--build-dir` and
`--out-dir` override those locations; `--build-only` compiles and validates
SPIR-V without using the GPU. Importing the runner performs no work.

The default build creates the three shapes observed on the study GPU:
16×16×16, 16×8×16 and 16×8×8. The executable checks advertised shapes and runs
only FP16 A/B, FP32 C/result, subgroup-scope, nonsaturating operations. It requires
a fixed subgroup size of 32; a different size fails explicitly. A device that
advertises an additional matching shape needs that variant compiled as well;
this focused probe does not implement a general matrix backend.

The shader receives FP32 storage-buffer inputs and converts them into shared
FP16 arrays before the cooperative load. Therefore this experiment measures
operand rounding as well as matrix arithmetic, without requiring 16-bit storage
buffers. The `platform.comp` shader lives beside its owning probe. It is compiled
for Vulkan 1.1 and SPIR-V 1.3 with the Vulkan memory model. The private device
explicitly enables cooperative matrices, FP16 arithmetic, the Vulkan memory
model and full compute subgroups; the pipeline requires full subgroups.

Six scenes per shape cover exact products, signed cancellation, products beyond
FP16 range, small residuals, irreversible operand quantization and mixed-sign
values. The five structured scenes require exact agreement with the known FP32
result of the FP16 operands. The mixed-sign scene compares against a double sum
of those operands with an explicit local acceptance bound. That bound is a
probe policy, not a Vulkan specification guarantee. JSON also records the
unrounded-input reference and FP16-rounded-product sum to distinguish errors.
`null` in the half-product comparison denotes a nonfinite value.

A passing platform probe establishes neither VMAF feature parity nor a speedup.
Integrated correctness needs the metric oracles and temporal/corpus tests;
performance must come from `build/vqm_bench` under the repository protocol.

## Numerical examples

```sh
python3 bench/experiments/cooperative_matrix/speed_numerics.py
python3 bench/experiments/cooperative_matrix/feature_numerics.py
```

These deterministic NumPy examples write into separate subdirectories of
`artifacts/reports/vmaf-cooperative-matrix/` and accept `--out PATH`. They run no
GPU work or timers. Their precision models and limitations are documented in
[SpEED numerics](../../notes/vmaf/research/cooperative-speed-numerics.md) and
[feature mapping](../../notes/vmaf/research/cooperative-feature-mapping.md).

## Integrated SpEED variants

The [measured investigation](../../notes/vmaf/vmaf-cooperative-matrix.md)
records why the current variants were retained as research rather than adopted.
`make_speed_variant.py` changes a fresh isolated source copy only. It rejects
this checkout and its ancestors/descendants, requires its expected baseline
anchors, and has no undo mode. Recreate the copy before applying another arm.
It is a local-device experiment: supported matrix shapes, 32-lane subgroups and
all requested features must already have been checked with the platform probe.
It does not implement production feature negotiation or borrowed-device promises.

For an uncommitted source state, capture both arms consistently without relying
on `HEAD`. This copies maintained files, excluding ignored artifacts/builds:

```sh
python3 - <<'PY'
from pathlib import Path
import shutil, subprocess
root = Path.cwd()
files = subprocess.check_output([
    'git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'
]).decode().split('\0')
for arm in ('baseline', 'candidate'):
    dest = Path('/tmp/vqm-coop-reproduce') / arm
    if dest.exists():
        raise SystemExit(f'refusing existing source copy: {dest}')
    for name in files:
        if not name or not (root / name).is_file():
            continue
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
PY
python3 bench/experiments/cooperative_matrix/make_speed_variant.py \
  /tmp/vqm-coop-reproduce/candidate --products 3 --split-k 128
cmake -S /tmp/vqm-coop-reproduce/baseline \
  -B /tmp/vqm-coop-reproduce/baseline/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/vqm-coop-reproduce/baseline/build --target vqm_bench vqm_test -j8
cmake -S /tmp/vqm-coop-reproduce/candidate \
  -B /tmp/vqm-coop-reproduce/candidate/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/vqm-coop-reproduce/candidate/build --target vqm_bench vqm_test -j8
```

The installed research shaders are copied next to the isolated SpEED source;
the maintained shader originals live beside this installer. Original FP32 means
and solver passes remain in place. `--products` counts cooperative MMAs per K
tile: 1 means HH only, 2 means HH and HL with a transposed cross term, 3 means
HH/HL/LH, and 4 adds LL. Residual L is scaled by 2048. The two-MMA arm keeps two
partial streams; the others keep one. `--split-k` controls partition count.
One/three/four-MMA arms normally omit unused lower tile 2. `--full-tiles` also
computes that tile, reproducing the early four-tile controls. The measured
parameter combinations were:

| Products | Split K | Extra flag | Measured arm |
| ---: | ---: | --- | --- |
| 1 | 8 | `--full-tiles` | `half8` |
| 3 | 8 | `--full-tiles` | `split3-8` |
| 2 | 32 | None | `split2-32` |
| 3 | 32 | None | `split3-32-tri` |
| 3 | 128 | None | `split3-128-tri` |

Run the actual gates without changing oracle tolerances. Several variants
intentionally fail numerical or memory-budget checks; read the study before
interpreting an exit code:

```sh
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LAYER_VALIDATE_SYNC=1 \
  /tmp/vqm-coop-reproduce/candidate/build/vqm_test --require-gpu
```

Use the comparison driver for five interleaved A,B rounds. It retains every
command, input/binary hash, score, checksum and stage CSV; all times come from
`vqm_bench`. `--streaming` is a separate 8-call/2-warmup arm; the default is
resident 2, 20 calls and 3 warmups. `--metric` defaults to vmaf-v1. Give each pair
a distinct case name. Paths below stand for an existing same-geometry image pair:

```sh
python3 bench/experiments/cooperative_matrix/compare_benchmarks.py reproduced \
  /tmp/vqm-coop-reproduce/baseline/build/vqm_bench \
  /tmp/vqm-coop-reproduce/candidate/build/vqm_bench \
  --pair 4k /absolute/reference.y4m /absolute/distorted.y4m
```

Results default to `artifacts/reports/vmaf-cooperative-matrix/reproduced/`;
`--out-dir DIR` overrides that destination. Run the two arms without other GPU
work. Summaries report per-arm medians and min–max spread; compare each arm only
with its interleaved baseline. No full-corpus accuracy qualification is implied
by these few benchmark pairs.

The existing [sequence probe](../README.md#gpu-sequence-feature-probe) can be
compiled against either arm. Pass its `--unclipped` option when checking score
drift so final saturation cannot hide feature changes. Preserve logs and useful
snapshots outside disposable builds, and write shared conclusions into
`bench/notes/`.
