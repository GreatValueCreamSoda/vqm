# VMAF v1 headroom experiments

These tools install individual CAMBI experiments into isolated source copies.
They do not edit the maintained checkout and have no undo operation. Use a
fresh copy for each arm: installers validate exact source anchors, and arbitrary
installer combinations are not supported. CPU/compiler checks establish their
stated contracts; they do not establish GPU correctness or a speedup.

The capability-selected row-subgroup pipeline is retained in production and
has passed its correctness and performance gates. The
[retained five-file patch](retained-subgroup.patch) reconstructs its preceding
screening baseline without discarding the surrounding uncommitted work.

Measurements and retention decisions belong in the
[VMAF v1 headroom report](../../notes/vmaf/vmaf-v1-headroom.md). Supporting investigations cover
[contrast](../../notes/vmaf/research/v1-contrast-headroom.md),
[independent contrast review](../../notes/vmaf/research/v1-contrast-independent-review.md),
[pooling](../../notes/vmaf/research/v1-pooling-headroom.md),
[storage and dispatch](../../notes/vmaf/research/v1-storage-dispatch-headroom.md),
and [other stages](../../notes/vmaf/research/v1-other-stages-headroom.md).
Read the final report for measured outcomes; the recipes below make no
performance claim.

Run commands from the repository root. Python 3 is required; the pool numerical
probe additionally uses NumPy. Compiler checks need the repository's C++20
toolchain, Vulkan headers, `glslc`, `spirv-val`, `spirv-dis`, and clang-format 23
where used. GPU runs require the normal Vulkan build prerequisites and a usable
device. Keep raw logs, source manifests and results under
`artifacts/reports/vmaf-v1-headroom/`; keep build outputs in each copy's scratch
`build/`. Shared conclusions and compact evidence remain in `bench/notes/`.

## Capture both arms from the same source state

The investigation began with a large uncommitted source reorganization. A
worktree of `HEAD` alone is therefore not the intended baseline. Capture the
full current maintained working tree, including nonignored untracked files and
current edits, once. The current tree already contains the retained subgroup
change: reverse only that patch inside a newly created isolated baseline, then
copy the reconstructed baseline into the candidate. Keep source editing paused
while capturing the snapshot. For a
historical committed comparison, choose and record the intended baseline commit
explicitly, using the worktree protocol in [AGENTS.md](../../../AGENTS.md).

This recipe refuses existing destinations, excludes ignored builds/external
inputs, and records working-tree and baseline hashes together with `HEAD`,
status and the retained patch hash. Patch reversal is preflighted with
`git apply --reverse --check`; incompatible context fails instead of changing
an unexpected source body. It runs only in the new isolated baseline. All
other working-tree changes remain in both arms.
It uses `copyfile`, which gives copies fresh modification times. Do not replace
it with `copy2`: preserving old source timestamps can cause incremental build
outputs to appear newer than changed source when an experiment is reapplied or
recovered. Each arm below starts with an empty build directory.

```sh
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path.cwd()
destination = Path('/tmp/vqm-v1-headroom')
if destination.exists():
    raise SystemExit(f'refusing existing snapshot: {destination}')
patch = root / 'bench/experiments/v1_headroom/retained-subgroup.patch'
if not patch.is_file():
    raise SystemExit(f'missing retained-change reconstruction patch: {patch}')
report = root / 'artifacts/reports/vmaf-v1-headroom/reproduction'
report.mkdir(parents=True, exist_ok=True)
manifest = {
    'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'status': subprocess.check_output(['git', 'status', '--short'], text=True),
    'reversed_patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
    'working_tree_files': {},
    'baseline_files': {},
}
names = subprocess.check_output([
    'git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'
]).decode().split('\0')
captured = []
baseline_root = destination / 'baseline'
candidate_root = destination / 'candidate'
for name in sorted(set(names)):
    if not name or not (root / name).is_file():
        continue  # Tracked deletions are absent from the working-tree snapshot.
    baseline = baseline_root / name
    baseline.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / name, baseline)
    manifest['working_tree_files'][name] = hashlib.sha256(baseline.read_bytes()).hexdigest()
    captured.append(name)
for args in (['--reverse', '--check'], ['--reverse']):
    subprocess.run(['git', '-C', str(baseline_root), 'apply', *args, str(patch)], check=True)
for name in captured:
    baseline = baseline_root / name
    candidate = candidate_root / name
    candidate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(baseline, candidate)
    manifest['baseline_files'][name] = hashlib.sha256(baseline.read_bytes()).hexdigest()
(report / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(destination)
PY
```

Never reverse this patch in the maintained checkout or an existing worktree.
The reconstruction is specifically for the historical screening baseline;
ordinary builds of the retained production tree need no reversal. The copied
core build needs no `.reference/` checkout or corpus. Reference and
corpus gates have their own prerequisites in
[VMAF development](../../../docs/vmaf-development.md). Existing installer
anchors may intentionally reject a newer production source; use the recorded
source fingerprint rather than weakening an anchor to patch an unrelated body.

## Install one arm

The following examples each assume a fresh candidate snapshot. Choose one
command; they are alternatives, not a sequence to apply to the same directory.
These installers expect the reconstructed pre-adoption baseline. Invoking them
directly on a copy of the current retained tree can correctly reject its anchors.

| Tool or flags | Responsibility and constraints |
| --- | --- |
| `contrast_variants.py DIR --mode-only` | Separate mode-filter pipeline whose reachable shader body has no contrast shared arrays. Mode passes keep their 16×16 grid. |
| `contrast_variants.py DIR --radius8` | Add compact capacity for validated windows ≤17; larger windows keep radius-16 capacity. Includes integer and enabled-Float16 paths. |
| `contrast_variants.py DIR --reuse-tail` | Store terms and the temporary active mask in unused packed-histogram tail words. This independent arm changes the shader only. |
| `contrast_variants.py DIR --tile-width 32` | Test fully buffered 32-column output tiles on the enabled half-source path; the portable integer path keeps 16×8 outputs. |
| `contrast_variants.py DIR --tile-rows 4` | Test four-row output tiles. Nonconditional four-row arms change the planner and have a known synthetic dispatch-boundary test failure; retain them as research controls, not a production recommendation. |
| `contrast_variants.py DIR --mode-only --tile-rows 4 --tile-invocations 128` | Test 128 physical invocations for four output rows. The mode pipeline remains separate with its original geometry. This is also a nonconditional research control. |
| `contrast_variants.py DIR --mode-only --tile-rows 4 --tile-invocations 128 --small-window-only` | Conditional four-row/128-invocation half-source arm. Select it only with enabled half source, window ≤17 and a supported four-row Y grid. Otherwise keep original integer/half 16×8 contrast and the original planner. |
| `pool_zero_sum.py DIR` | Prove that the sum strictly above the exact selected threshold is zero, then skip the final value scan. Canonical threshold/rank, finish arithmetic and dispatch count stay unchanged. |
| `storage_mode_banks.py DIR` | Store horizontal/filtered mode banks as exact uint16 codes when enabled 16-bit buffer storage is available. Existing arena reservations remain unchanged; subsequent decimation restores FP32 input. |
| `storage_component_barriers.py DIR` | Remove repeated component-entry dependencies only inside the CAMBI composite, retaining standalone entry defaults and producer exit dependencies. |
| `subgroup_rows.py DIR` | Research-only row-private subgroup barriers after cross-row construction. Requires physical default subgroup size 32 with compute BASIC operations before pipeline creation. A uniform shader guard retains the workgroup-barrier path for other actual packing. |
| `subgroup_rows.py DIR --skip-inactive-rows` | Separate subgroup arm that returns complete inactive row subgroups only after the final cross-row barrier. |
| `subgroup_pipeline.py DIR` | Reconstruct the retained capability-selected implementation. Adds separate subgroup modules while portable modules keep their original capabilities; automatically selects portable modules when subgroup properties are unsuitable. Adds eligibility and exact portable/subgroup tests. Requires `subgroup_rows.py` beside the installer for shared source helpers. |

For example, reconstruct the retained implementation in the candidate:

```sh
python3 bench/experiments/v1_headroom/subgroup_pipeline.py \
  /tmp/vqm-v1-headroom/candidate
```

`--tile-invocations 128` requires both `--tile-rows 4` and `--mode-only`.
`--small-window-only` requires all three. The conditional arm accepts optional
`--radius8` to constrain only the extra small half-source module's capacity,
and optional `--reuse-tail`. The contrast installer supports its documented
flag combinations in a single invocation; it does not promise compatibility
with separately installed storage, subgroup or timing probes. Subgroup arms
remain device-specific research and do not add a public device-feature promise.
The retained `subgroup_pipeline.py` form adds portable capability selection;
its ordinary uint/half modules do not require subgroup operations. It excludes
the inactive-row return experiment.

## CPU and compiler checks

```sh
report_dir="$PWD/artifacts/reports/vmaf-v1-headroom/reproduction"
baseline=/tmp/vqm-v1-headroom/baseline/bench/experiments/v1_headroom
mkdir -p "$report_dir"
python3 "$baseline/contrast_contracts.py" --out-dir "$report_dir/contrast-contracts"
python3 "$baseline/pool_zero_sum_check.py" --out "$report_dir/pool-zero-sum.json"
python3 "$baseline/pool_zero_sum_compile.py" --out "$report_dir/pool-compiler"
python3 "$baseline/storage_mode_banks_check.py" > "$report_dir/storage-contracts.json"
python3 "$baseline/subgroup_rows_check.py" > "$report_dir/subgroup-contracts.json"
```

Run source-transforming contract checks from the reconstructed baseline copy:
their source-root discovery follows the script location, and the maintained
tree now contains the retained subgroup change. The independent numerical
probes themselves remain usable without source reversal.

`contrast_contracts.py` checks supported installed host combinations, compiled
shader capacities, histogram/indexing ownership, tail lifetimes and conditional
pipeline selection. Its `--out-dir DIR` selects the raw evidence directory.
The pool model's `--out PATH` selects JSON; its compiler runner's `--out DIR`
selects formatted shader copies, compiled modules and compiler evidence. The
storage-bank and subgroup access-domain checks print JSON to stdout and perform
no GPU work. These checks preserve independent mathematical/reference reasoning;
they do not replace the actual GPU oracles, especially for atomics, barriers,
subgroup packing and borrowed-device feature contracts.

## Build and run correctness gates

Use separate, empty build directories. For both arms:

```sh
set -eu
for arm in baseline candidate; do
  cmake -S "/tmp/vqm-v1-headroom/$arm" \
    -B "/tmp/vqm-v1-headroom/$arm/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DVQM_WARNINGS_AS_ERRORS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  cmake --build "/tmp/vqm-v1-headroom/$arm/build" \
    --target vqm vqm_bench vqm_test --parallel 8
  "/tmp/vqm-v1-headroom/$arm/build/vqm_test" --require-gpu
  ctest --test-dir "/tmp/vqm-v1-headroom/$arm/build" --output-on-failure
done
```

Run the quality/lint checks from a maintained development checkout containing
the proposed production patch and its configured compile database, as directed
by [AGENTS.md](../../../AGENTS.md). Do not weaken GPU oracle tolerances to make
an experiment pass. Production adoption also needs applicable sanitizer,
independent VMAF feature/sequence, pinned-reference and corpus checks. Report
unavailable prerequisites and known failing control arms explicitly.

## Interleave ordinary benchmark arms

Reuse the general
[comparison runner](../cooperative_matrix/compare_benchmarks.py). It performs
five A,B rounds with ordinary `vqm_bench`, preserving every command, input and
binary hash, score, checksum, stage CSV, median and min–max spread. Its default
is VMAF v1 resident 2, 20 calls and three warmups. `--streaming` uses eight calls
and two warmups as a separate upload-path workload. Always supply `--out-dir`:
the shared runner's default otherwise belongs to the cooperative-matrix study.

Use existing same-geometry pairs in the following command; placeholder paths
are not supplied fixtures. Give every pair a distinct case name, include both
1080p and 4K, and repeat resident and streaming separately without other GPU work.

```sh
python3 bench/experiments/cooperative_matrix/compare_benchmarks.py subgroup-pipeline \
  /tmp/vqm-v1-headroom/baseline/build/vqm_bench \
  /tmp/vqm-v1-headroom/candidate/build/vqm_bench \
  --pair 1080 /absolute/1080-reference.y4m /absolute/1080-distorted.y4m \
  --pair 4k /absolute/4k-reference.y4m /absolute/4k-distorted.y4m \
  --out-dir artifacts/reports/vmaf-v1-headroom/subgroup-pipeline-resident

python3 bench/experiments/cooperative_matrix/compare_benchmarks.py subgroup-pipeline \
  /tmp/vqm-v1-headroom/baseline/build/vqm_bench \
  /tmp/vqm-v1-headroom/candidate/build/vqm_bench \
  --streaming \
  --pair 1080 /absolute/1080-reference.y4m /absolute/1080-distorted.y4m \
  --pair 4k /absolute/4k-reference.y4m /absolute/4k-distorted.y4m \
  --out-dir artifacts/reports/vmaf-v1-headroom/subgroup-pipeline-streaming
```

`--vmaf-chroma native|420|444` selects the same explicit input arm for both
binaries; do not mix settings when comparing results. Read every score,
checksum and stage table. A passing subprocess or unchanged rounded score is
insufficient numerical qualification. Call a candidate an improvement only
after its targeted stage passes the ≥2% interleaved median speed gate and its
correctness gates. Resident reuse excludes upload and is a hot-memory workload,
not a claim about decoded video throughput or many independent frames.

## Observe the ordinary pipeline's subgroup packing

The standalone `subgroup_rows_probe.cc` and `subgroup_rows_probe.comp` use the
ordinary runtime/device and a 16×16 compute workgroup. They request no required
subgroup size, full-subgroup pipeline flag, new device feature or timestamp.
The host prints physical default subgroup/BASIC support, actual shader subgroup
size and subgroup count, the row-path guard, and whether subgroup-based logical
slots cover all 256 invocations exactly once. Unsupported compute BASIC
operations fail explicitly. This observes packing; it does not score images or
measure performance.

Build the current retained core, then compile and run the probe:

```sh
cmake --preset release
cmake --build --preset release --target vqm_core --parallel 8
mkdir -p build/v1-headroom artifacts/reports/vmaf-v1-headroom/subgroup-observation
glslc -O --target-env=vulkan1.1 \
  bench/experiments/v1_headroom/subgroup_rows_probe.comp \
  -o build/v1-headroom/subgroup-rows-probe.spv
spirv-val --target-env vulkan1.1 build/v1-headroom/subgroup-rows-probe.spv
c++ -std=c++20 -O2 -I include -I src \
  bench/experiments/v1_headroom/subgroup_rows_probe.cc build/libvqm_core.a \
  -lvulkan -pthread -o build/v1-headroom/subgroup-rows-probe
build/v1-headroom/subgroup-rows-probe build/v1-headroom/subgroup-rows-probe.spv \
  > artifacts/reports/vmaf-v1-headroom/subgroup-observation/result.txt
```

The study GPU was observed using size 32, eight subgroups and all 256 logical
slots exactly once with ordinary pipeline flags. This local observation is not
a cross-device guarantee. The runtime guard and portable modules remain
necessary; the final report retains the device/build provenance and observation.

## Diagnostic CAMBI stage split

`cambi_stages.py DIR` installs six nonnested accumulated timestamp categories:
prepare, decimate, mode, contrast, pool and combine. It replaces the aggregate
CAMBI bracket and inspects the existing query capacity without changing it.
At 1080p/4K the complete V1 submission uses 44 queries; inputs without initial
high-resolution decimation use 42. It is diagnostic instrumentation only.

Create a third fresh copy of the reconstructed pre-adoption baseline and
install this probe there, then build
its `vqm_bench` with the same release settings. Do not install it sequentially
with component-barrier or contrast installers: their exact anchors overlap and
need source-aware adaptation. Added timestamp calls can perturb these small
stages. Use diagnostic output to locate work and ordinary uninstrumented
baseline/candidate arms for measured performance conclusions. Preserve useful
findings and compact evidence in the shared report before deleting scratch
builds or ignored raw results.
