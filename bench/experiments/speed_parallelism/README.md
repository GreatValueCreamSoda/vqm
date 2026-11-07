# SpEED work-distribution experiments

These tools reproduce the [SpEED parallelism investigation](../../notes/vmaf/vmaf-speed-parallelism.md).
They retain useful research source; generated binaries belong in ignored
`build/` directories and raw results in `artifacts/reports/vmaf-speed-parallelism/`.
The production implementation is unchanged. The
[source review](../../notes/vmaf/research/speed-parallelism-review.md) records
the original grids and their limits. Group counts establish available work,
not measured SM occupancy or a latency guarantee.

| Tool or source | Responsibility |
| --- | --- |
| `stage_probe.py` | Adds seven diagnostic GPU timestamp brackets: filter, means, covariance, solve, pool, reduce and combine. |
| `solver_lanes.py` | Changes only the normal wide-solver variant's local size; grid and arithmetic remain unchanged. |
| `install_pool.py` | Installs subgroup-per-block pooling and an exact-order reduction using dead means/covariance storage when the block count fits. |
| `solver_phases.py`, `solver_phase_bisect.comp` | Splits preparation, speculative-tree bisection and guarded QR completion into three dispatches. |
| `solver_phase_tests.py` | Routes direct spectral tests through those three phases, retaining the unchanged forced-QR oracle. |
| `subgroup_solver.py` | Uses the same speculative tree within the existing one-dispatch solver and standard blob names. |
| `early_qr.py` | Tries a short rank-zero prefix certificate before the full search; optional gates and a heuristic control are explicit. |
| `solver_tree_check.py`, `early_qr_numerics.py` | Deterministic CPU traversal/certificate diagnostics; not GPU arithmetic emulators or metric oracles. |
| `feature_probe.cc` | Decodes a single pair and prints public VMAF v1 features after Push/Flush, without timers. |
| `eigen_stress.cc`, `stress_dataset.py` | Reproduce and compare the seed-113, 10,000-matrix threshold stress dataset. |
| `pool_subgroup.comp`, `pool_reduce_alias.comp` | Shader sources copied next to the isolated metric by `install_pool.py`. |
| `pool_merge.comp`, `solver_cholesky.comp` | Optional standalone research controls; the current installers do not activate these files. |

## Capture the current source before experimenting

Run from the repository root. The example below captures maintained tracked and
untracked files, so an already-modified working tree is represented faithfully.
Ignored builds, raw artifacts, caches and external checkouts are excluded.
Existing destinations are refused rather than overwritten.

```sh
python3 - <<'PY'
from pathlib import Path
import shutil, subprocess
root = Path.cwd()
files = subprocess.check_output([
    'git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'
]).decode().split('\0')
for arm in ('baseline', 'candidate', 'diagnostic'):
    destination = Path('/tmp/vqm-speed-reproduce') / arm
    if destination.exists():
        raise SystemExit(f'refusing existing source copy: {destination}')
    for name in files:
        if not name or not (root / name).is_file():
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
        shutil.copymode(root / name, target)
PY
```

Every installer requires a fresh copy outside the maintained checkout; the
checkout, its ancestors and its descendants are refused. Apply one candidate
installer per fresh copy. The phase-test routing below is the intentional
follow-up exception. Installers are not undo tools and their anchors describe
the reviewed baseline. Preserve useful patches/results before replacing a copy.

When restoring an experiment, `copyfile` or explicitly refreshed modification
times matter: `copy2` can restore old timestamps and leave stale Ninja/SPIR-V
outputs. Use a fresh build or verify rebuilt module hashes before measuring.

## Three-phase solver and direct oracle checks

```sh
python3 bench/experiments/speed_parallelism/solver_phases.py \
  /tmp/vqm-speed-reproduce/candidate
python3 bench/experiments/speed_parallelism/solver_phase_tests.py \
  /tmp/vqm-speed-reproduce/candidate
cmake -S /tmp/vqm-speed-reproduce/baseline \
  -B /tmp/vqm-speed-reproduce/baseline/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DVQM_WARNINGS_AS_ERRORS=ON
cmake --build /tmp/vqm-speed-reproduce/baseline/build --target vqm_bench vqm_test -j8
cmake -S /tmp/vqm-speed-reproduce/candidate \
  -B /tmp/vqm-speed-reproduce/candidate/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DVQM_WARNINGS_AS_ERRORS=ON
cmake --build /tmp/vqm-speed-reproduce/candidate/build --target vqm_bench vqm_test -j8
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LAYER_VALIDATE_SYNC=1 \
  /tmp/vqm-speed-reproduce/candidate/build/vqm_test --require-gpu
```

The direct-test installer is important. Without it, integrated VMAF uses the
new phases while the direct `EigenDevice` tests still select the old blobs.
Routing changes only the isolated test harness; assertions, original output
offsets, oracle tolerances and the forced-legacy QR shader remain unchanged.
Both preparation widths exercise the new bisection, so the 128-width run is
not a portable fallback validation.

Subgroup prototypes assume local 32-lane compute subgroups, supported shuffle
operations and enabled subgroup-size control/full-subgroup features. Their
private runtimes request these features and relevant stages require full
32-lane groups. Portable device negotiation, borrowed-device promises and
production fallback selection are **not implemented**. A passing local test
does not establish AMD/Intel/software-Vulkan support or full-corpus parity.

## Other isolated controls

Choose one of these **instead of** the phase installer, using a new candidate
copy. Configure/build it using the same commands above.

```sh
python3 bench/experiments/speed_parallelism/solver_lanes.py \
  /tmp/vqm-speed-reproduce/candidate --lanes 32
python3 bench/experiments/speed_parallelism/install_pool.py \
  /tmp/vqm-speed-reproduce/candidate --subgroups 2
python3 bench/experiments/speed_parallelism/subgroup_solver.py \
  /tmp/vqm-speed-reproduce/candidate
python3 bench/experiments/speed_parallelism/early_qr.py \
  /tmp/vqm-speed-reproduce/candidate
python3 bench/experiments/speed_parallelism/early_qr.py \
  /tmp/vqm-speed-reproduce/candidate --gate sturm
```

These lines are alternatives, not a sequence to apply to one copy.
`solver_lanes.py` accepts 32/64/128/256/512. Pooling accepts one, two or four
subgroups and preserves the previous two-level sum tree; above 1300 blocks it
retains the original pooling shaders. The one-dispatch subgroup solver keeps
the normal/wide blob names, so direct spectral tests select it automatically;
do not apply the phase-test router to that arm.

Early QR defaults to `--mode prefix --gate diagonal --probe-iterations 16
--margin-scale 1e-6`. The Sturm gate changes whether the prefix is attempted;
the prefix itself follows unchanged rank-zero midpoint decisions before
selecting the existing QR fallback. `--gate always` is another explicit control.
`--mode rayleigh` is a heuristic without that prefix certificate and requires
the diagonal gate. It should not be interpreted as equivalent to the certified
prefix arm or as a production-safe shortcut.

CPU-only checks write their own JSON outputs and support `--out PATH`:

```sh
python3 bench/experiments/speed_parallelism/solver_tree_check.py
python3 bench/experiments/speed_parallelism/early_qr_numerics.py
python3 bench/experiments/speed_parallelism/early_qr_numerics.py --gate sturm \
  --out artifacts/reports/vmaf-speed-parallelism/earlyqr/numerics-sturm.json
```

## Diagnostic stage attribution and ordinary comparisons

Use the separate untouched diagnostic copy for extra timestamp boundaries:

```sh
python3 bench/experiments/speed_parallelism/stage_probe.py \
  /tmp/vqm-speed-reproduce/diagnostic
cmake -S /tmp/vqm-speed-reproduce/diagnostic \
  -B /tmp/vqm-speed-reproduce/diagnostic/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/vqm-speed-reproduce/diagnostic/build --target vqm_bench -j8
mkdir -p artifacts/reports/vmaf-speed-parallelism/reproduced-diagnostic
/tmp/vqm-speed-reproduce/diagnostic/build/vqm_bench --metric vmaf-v1 \
  --pair /absolute/reference.y4m /absolute/distorted.y4m \
  --iterations 20 --warmup 3 --resident 2 \
  --stages-csv artifacts/reports/vmaf-speed-parallelism/reproduced-diagnostic/stages.csv
```

Replace example input paths with existing equal-geometry images. Extra queries
perturb execution. Read the seven `speed_*` rows; the original aggregate SpEED
bracket was replaced. These diagnostic rows are not occupancy counters and
must not be used as a baseline for ordinary speed claims.

All optimization comparisons use ordinary `vqm_bench` binaries through the
shared [interleaved comparison driver](../cooperative_matrix/compare_benchmarks.py).
Override its cooperative-study output default so this work stays together:

```sh
python3 bench/experiments/cooperative_matrix/compare_benchmarks.py reproduced-resident \
  /tmp/vqm-speed-reproduce/baseline/build/vqm_bench \
  /tmp/vqm-speed-reproduce/candidate/build/vqm_bench \
  --pair sample /absolute/reference.y4m /absolute/distorted.y4m \
  --out-dir artifacts/reports/vmaf-speed-parallelism/reproduced-resident
python3 bench/experiments/cooperative_matrix/compare_benchmarks.py reproduced-streaming \
  /tmp/vqm-speed-reproduce/baseline/build/vqm_bench \
  /tmp/vqm-speed-reproduce/candidate/build/vqm_bench --streaming \
  --pair sample /absolute/reference.y4m /absolute/distorted.y4m \
  --out-dir artifacts/reports/vmaf-speed-parallelism/reproduced-streaming
```

The driver interleaves five A,B rounds: resident 2 uses 20 calls/3 warmups;
streaming separately uses 8/2. Every command, input/binary hash, score, checksum
and stage CSV is retained. `--vmaf-chroma native|420|444` selects the same mode
for both arms; 4:4:4 is a separate workload. Do not overlap other GPU work.
Compare complete stage/frame medians with their spreads, not group counts or
diagnostic timings. Final measured outcomes and remaining limits belong in
the [investigation report](../../notes/vmaf/vmaf-speed-parallelism.md).

## Public feature probe

This uses the benchmark's decoder and one public VMAF v1 Push/Flush. Link the
chosen build's core and image-I/O libraries; that core owns its device setup.
Only explicit context shutdown uses a private helper to avoid headless driver
exit-order problems. There are no timers or private per-channel status reads.

```sh
mkdir -p build
VQM_FEATURE_TREE=/tmp/vqm-speed-reproduce/baseline
c++ -std=c++20 -O2 -I "$VQM_FEATURE_TREE/include" -I "$VQM_FEATURE_TREE/src" \
  -I tools/image_io bench/experiments/speed_parallelism/feature_probe.cc \
  "$VQM_FEATURE_TREE/build/libvqm_core.a" "$VQM_FEATURE_TREE/build/libvqm_io.a" \
  -lvulkan -lpng -ljpeg -lz -pthread -o build/speed_feature_probe
./build/speed_feature_probe /absolute/reference.y4m /absolute/distorted.y4m --unclipped
```

`--chroma native|420|444` matches the pipeline selection. Output includes ADM3,
SpEED, CAMBI, motion and unclipped or clipped score. Zero public SpEED alone
does not identify singular status or QR fallback.

## Full threshold stress comparison

The generator preserves seed 113, 500 diagonal threshold cases and the original
four condition-range bands. On NumPy 2.5.4, both its 10,000-matrix binary and
truth CSV reproduce the archived study byte for byte. NumPy/LAPACK differences
may change the accepted dataset; the manifest records its hash/version and
whether it matches the historical binary hash. FP64 truth near the cutoff is
approximate, not a formal classification oracle.

Compile the same driver against each selected library. The example produces
one original/legacy driver and one phase driver; keep tests enabled in the
libraries so the unchanged forced-QR oracle blob exists.

```sh
python3 bench/experiments/speed_parallelism/stress_dataset.py generate
c++ -std=c++20 -O2 -I /tmp/vqm-speed-reproduce/baseline/include \
  -I /tmp/vqm-speed-reproduce/baseline/src \
  bench/experiments/speed_parallelism/eigen_stress.cc \
  /tmp/vqm-speed-reproduce/baseline/build/libvqm_core.a \
  -lvulkan -pthread -o build/speed_eigen_stress_baseline
c++ -std=c++20 -O2 -I /tmp/vqm-speed-reproduce/candidate/include \
  -I /tmp/vqm-speed-reproduce/candidate/src \
  bench/experiments/speed_parallelism/eigen_stress.cc \
  /tmp/vqm-speed-reproduce/candidate/build/libvqm_core.a \
  -lvulkan -pthread -o build/speed_eigen_stress_phases
VQM_STRESS_DATA=artifacts/reports/vmaf-speed-parallelism/stress/data/matrices.bin
./build/speed_eigen_stress_baseline --input "$VQM_STRESS_DATA" --solver original \
  --out artifacts/reports/vmaf-speed-parallelism/stress/original.csv
./build/speed_eigen_stress_baseline --input "$VQM_STRESS_DATA" --solver legacy \
  --out artifacts/reports/vmaf-speed-parallelism/stress/legacy.csv
./build/speed_eigen_stress_phases --input "$VQM_STRESS_DATA" --solver phases \
  --out artifacts/reports/vmaf-speed-parallelism/stress/phases.csv
python3 bench/experiments/speed_parallelism/stress_dataset.py compare \
  --csv original artifacts/reports/vmaf-speed-parallelism/stress/original.csv \
  --csv legacy artifacts/reports/vmaf-speed-parallelism/stress/legacy.csv \
  --csv phases artifacts/reports/vmaf-speed-parallelism/stress/phases.csv
```

`--width auto|128|512` selects the original or preparation width; phase bisection
still uses full 32-lane groups. Legacy QR uses its fixed 128-lane blob and
rejects a 512 request. The current 2604-word legacy layout is distinct from
the archived 2602-word legacy-guard driver. The phase driver reserves 2716
words per matrix pair and passes `pc.op=1`; an odd final case uses a padded
second matrix whose output is omitted.

Comparison verifies dataset SHA, every input ID/SHA/FNV, statuses and eigenvalue
bit formatting before reporting status and exact bit differences. Those
differences are compatibility evidence, not independent metric correctness.
`generate --out-dir`, `compare --data-dir` and `compare --out` override paths.
The same driver can test a one-dispatch subgroup library with `--solver original`
because that installer retains the standard blob names. Never compare CSVs
from different matrix hashes or claim throughput from this untimed driver.
