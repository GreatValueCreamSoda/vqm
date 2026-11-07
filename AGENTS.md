# VQM repository guidance

These instructions apply to anyone working on VQM, including AI coding
assistants. Paths are relative to the repository root. Shell examples use POSIX
syntax. Use the equivalent paths and commands for your environment.

## Working defaults

- Inspect `git status --short`, `git diff --stat`, and the relevant diffs before
  editing. Preserve existing staged, unstaged, and untracked work. Isolate
  experiments in a worktree or save your own patch before editing shared files.
- Create commits only when requested. Report the changes, the checks that ran,
  and any skipped checks or unresolved limitations.
- Run tests locally. Keep the CTest, GPU, sanitizer, formatting, and benchmark
  workflows working. This project excludes CI workflows, external test
  services, and automated status checks.
- Use the repository's documented tools and standard configuration discovery.
  Keep changes scoped to the task and follow existing style in the files you
  edit. Use Ruff to lint Python and YAPF to format it. See
  [Contributing](CONTRIBUTING.md#formatting-and-provenance).
- Treat `.reference/` as read-only pinned input. Use the fetch and build tools
  for the references the task needs. Build copies in the selected build
  directory.
- Select the checks that match the change. Report unavailable tools, device
  features, and reference inputs. CPU-only results do not establish GPU
  correctness. Report speedups only after the applicable measured speed gate.

## Repository organization

- `include/` contains the installed API. `vqm_image.h` owns shared image types,
  the metric headers define individual APIs, and `vqm.h` is the umbrella.
- `src/metrics/` owns each metric's host code, layout, dispatch, shaders, and
  model data. VMAF groups its feature and model pipelines inside its directory.
  `src/vulkan/` owns metric-independent device, memory, input, and dispatch
  machinery. Shared GLSL lives in `src/vulkan/shaders/`. Benchmark timing lives
  in `src/timing.{h,cc}`.
- `cli/` owns the command line and optional BestSource decoding.
  `tools/image_io/` supplies image and Y4M decoding for the CLI, benchmarks,
  and tests. The installed core remains independent of these decoders.
- `tests/` contains the local harness and integration tests. Keep numerical
  oracles and fixtures with their owning tests. Metric tests mirror the source
  ownership under `tests/metrics/`.
- `bench/driver/` owns the timed benchmark, `bench/inmemory/` the reference
  drivers, and `bench/experiments/` reusable research probes.
- `tools/` owns setup and reference builds. `validation/` owns CID22 and hosted
  video preparation and correlation. Metric guides and implementation
  documentation live in [docs](docs/README.md).

Keep files and directories organized by responsibility. Roughly 500–1,000
lines per implementation file and 5–10 files per directory are guidelines.
Headers, shaders, generated data, tests, documentation, and small tools can
be shorter. Split large files at meaningful boundaries. When moving code,
update CMake, scripts, documentation, and path references together. Keep
installed header paths and embedded shader identifiers stable unless a
breaking change is requested.

Use descriptive names, early validation, and a visible sequence of phases.
Share repeated contracts and resource transactions in the module that owns
them. Keep ownership, synchronization, cached state, and buffer lifetimes
close to the code that uses them. Preserve independent test and reference
oracles, metric arithmetic, and measured hot-path behavior during refactors.

## Documentation, tests, and experiments

- Keep `docs/` focused on how the metrics work, how VQM implements them, and
  where its behavior or scores diverge from the pinned references. Put active
  build, benchmark, and validation instructions with their workflows.
- Fold durable findings into the metric guides or the active validation workflow.
  Remove superseded plans, audits, development logs, and duplicate summaries
  once the current documentation covers their lasting conclusions. Use Git
  history for retired material.
- Keep tests that cover current behavior or independent reference contracts.
  Consolidate overlapping cases and remove tests that exercise retired
  prototypes. Preserve distinct coverage when merging tests.
- Keep benchmark probes that run against the current tree or reproduce
  maintained reference evidence. Keep `bench/` for runnable tools and concise
  usage guides. Store generated reports, test results, and experiment logs
  under ignored `artifacts/reports/`. Retire one-off prototype installers
  and use Git history for historical reproduction.

## Workspace files and retention

| Location | Retention |
| --- | --- |
| Source, headers, shaders, CMake files, editor defaults, docs, scripts, tests, and license notices | Keep in version control. Use ignored `CMakeUserPresets.json` for local CMake overrides. |
| Checked-in model `.inc` tables, oracle fixtures, and frozen validation policies | Keep as maintained inputs beside their owning tests or workflows. |
| `build/` and `build-*/` | Regenerable CMake outputs, binaries, copied reference build trees, and fixtures. Check for unique work before cleaning. |
| `.venv/` | Ignored managed Python environment. Preserve existing environments during setup and ordinary cleanup. |
| `.reference/`, `validation/data/`, and `validation/download/` | Read-only pinned checkouts and external corpus files. Keep during ordinary build cleanup. |
| `.cache/bench/` and `.cache/validation/` | Regenerable benchmark and corpus caches. |
| `artifacts/reports/` and `artifacts/snapshots/` | Ignored raw reports and local backups. Preserve unique evidence and recovery copies. |
| `compile_commands.json` at the repository root | Generated clangd database. Regenerate with `python3 tools/vscode_compile_commands.py` after cleaning or switching builds. |

A fresh core build must regenerate its outputs from maintained source. Write
raw runs to `artifacts/reports/` and generated datasets to `.cache/`. Use explicit
output paths when working out of tree. Keep copied reference build inputs
and binaries in the selected build directory.

Before cleaning, inspect builds for authored notes, patches, and one-off
experiments. Record the workload, device, driver, build, inputs, and
limitations with local benchmark results under `artifacts/reports/`.

## Build

Follow the [contributor setup](CONTRIBUTING.md#build-and-test) for the C++
dependencies and lightweight `uv`-managed Python environment. Select that
interpreter explicitly when configuring the full local suite.

```sh
cmake --preset release -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build --preset release --parallel
ctest --preset release
```

Use `vqm`, `vqm_bench`, and `vqm_test` in current workflows. A core build needs
no reference checkout or corpus. The default CLI build needs no BestSource
or FFmpeg libraries. See
[BestSource prerequisites](README.md#bestsource-indexed-hardware-decoding) for
the optional Linux backend. Use the [sanitizer preset](CONTRIBUTING.md#build-and-test)
on supported toolchains for memory-sensitive changes.

Run `python3 tools/check_quality.py` with clang-format 23 for maintained C++,
headers, CUDA source, and shaders. Add `--lint` for production C++ changes.
Use `--build-dir DIR` to select the compile database and `--format-tool` or
`--lint-tool` for versioned tool names. `--include-tests` extends lint to test
matrices and oracles for advisory review.

## Corpus

Generate small CLI fixtures with `.venv/bin/python tests/make_goldens.py`.
The default output is `build/fixtures/`. Use `--out-dir` for another build
directory. Select original and distorted images with matching dimensions
for performance runs. To generate a reproducible 4K pair with FFmpeg:

```sh
.venv/bin/python bench/make_bench_pairs.py --synthetic-only --sizes 3840x2160
```

The pair is in `.cache/bench/pairs/synthetic-3840x2160/` as `orig.png` and
`dist.png`. Use separate workloads for odd dimensions when validating geometry.

Fetch only the references and corpora needed for the change. For example,
the SSIMULACRA 2 CID22 comparison uses these commands.

```sh
python3 tools/fetch_references.py --only cid22,ssimulacra2
python3 tools/build_references.py --only cloudinary
```

`tools/fetch_references.py --check --only NAMES` verifies selected inputs
offline. For CVVDP reference setup and model regeneration, follow
[CVVDP reference validation](validation/cvvdp.md). After fetching the pinned
original and provisioning its environment, `tools/generate_cvvdp_tables.py`
regenerates the maintained tables and `--check` verifies the existing tables.

## Protocol

Use the [benchmark guide](bench/README.md) to select a workload.
`vqm_bench` uses the timing-enabled core and reports stages, scores, and
checksums. `vqm` and the installed core use the untimed library. Keep decode,
upload, core, stage, and whole-process timing boundaries distinct.
See [Performance methodology](bench/performance.md).

- Record the baseline commit before the change. Build it separately in a
  detached worktree. Use a unique worktree directory and a separate build
  directory for each arm. Check for unique work before removing an experiment.
- Use the same hardware, driver, build settings, and input pairs. Interleave
  five runs per arm in one sitting and report the median and spread. GPU clocks
  and thermals can change between sessions.
- Use `--iterations 20 --warmup 3 --resident 2` for the resident comparison.
  Measure upload separately with `--iterations 8 --warmup 2` and no resident
  flag. Keep `--half-planes` in its own comparison.
- SSIMULACRA 2 alternates pre-uploaded copies of the original to force full
  preparation of both images. Butteraugli can reuse the prepared reference.
  `--resident-ring N` bounds the repeated working set and is a separate workload.
  See [Performance methodology](bench/performance.md#protocol).
- Inspect scores, checksums, and stage tables for every run. Checksums include
  warmups and timed calls. Workers warm up separately before sharing the
  timed call count. Alpha latency arms time the dark background and check the
  bright background separately.
- Compare aggregate throughput by wall time over the same fixed pair count.
  Report device, driver, flags, geometry, build and input provenance, and the
  observed spread. Use unprofiled timings for performance claims.

Set `BASELINE_BUILD_DIR` to the baseline build directory and set `ORIGINAL`
and `DISTORTED` to a matching input pair. Run the resident arms in this order.

```sh
for round in 1 2 3 4 5; do
  "$BASELINE_BUILD_DIR/vqm_bench" --pair "$ORIGINAL" "$DISTORTED" \
    --iterations 20 --warmup 3 --resident 2
  ./build/vqm_bench --pair "$ORIGINAL" "$DISTORTED" \
    --iterations 20 --warmup 3 --resident 2
done
```

For a hosted video benchmark, the documented bootstrap is
`uv run --no-project --python 3.12 bench/benchmark.py`. It provisions the
pinned reference environment locally. Use `--suite images` for still-image
peer comparisons. Vship is exercised through its public C API. The tool
matches its device by name, which gives weaker identity evidence than a UUID.

## Gates

Apply the checks relevant to the change. Documentation changes need link,
command, and text checks. Changes that affect a hot path need an interleaved
regression comparison. A performance optimization must improve the target
stage median by at least 2% while meeting the correctness gates.

1. **Identity and component oracles.** SSIMULACRA 2 identity uses the
   `99.5 <= score <= 100.0` test band and should print `100.00000000` on a
   discrete GPU. Butteraugli identity is exactly `0.00000000`. Score the
   generated gradient fixture against itself through `vqm`. Preserve numerical
   feature, filter, model, and arithmetic pins, including `TestGoldenScores`
   and the chroma oracle. Butteraugli's gradient, spatial-frequency,
   chroma-suppression, and directional-Malta cases compare score and
   `distance_p3` to the CPU oracle within 0.1%. VMAF changes retain the
   independent oracles in `tests/metrics/vmaf/`.
2. **Local GPU and build checks.** Run `ctest --preset release` and
   `./build/vqm_test --require-gpu` for GPU behavior changes. Optional device
   features can be skipped when unavailable. Use the local synchronization,
   sanitizer, and SPIR-V checks in [Contributing](CONTRIBUTING.md#build-and-test)
   as appropriate.
3. **Still-image qualification.** Use the frozen
   [source-balanced CID22 policies](validation/fixtures/cid22-balanced/README.md)
   for SSIMULACRA 2 and Butteraugli. They require SRCC at least 0.999, Kendall
   tau-b at least 0.99, source-balanced clearly ordered errors at most 0.5%,
   family errors at most 1%, and worst-source errors at most 5%. Keep their
   coverage, provenance, and paired subjective noninferiority checks.
4. **File-based video qualification.** Keep file-based scoring in the
   normal `vqm` CLI. Use `--json`, `--pairs-file`, `--iterations`, and
   `--warmup` for whole-clip runs. After the Release build and Python setup,
   run the complete default matrix with the frozen original scores and policies.

   ```sh
   .venv/bin/python -m validation.temporal.run --profile qualification
   ```

   The runner fetches missing pinned clip inputs. It scores 390 complete
   pairs per metric for SSIMULACRA 2, Butteraugli, VMAF v0, VMAF v1, and CVVDP.
   Record each outcome with correlation, coverage, and numerical diagnostics.
   Partial runs retain their metric selection. Use the exact spatial and
   temporal representations in the [video validation guide](validation/temporal/README.md).
   Keep independent HDR, high-bit-depth, alternate-display, alternate-model,
   and extreme-rate oracles where applicable. Use direct API tests for temporal
   arithmetic, ownership, synchronization, and lifecycle contracts.
5. **Final-score changes.** Registered qualified input domains default to
   ranking with numerical diagnostics. Unregistered or modified policies use
   the combined `both` default. Use `--gate-mode both` to run both acceptance
   rules before retiring another numerical gate. Document recalibration,
   revised score interpretation, and measured results while preserving
   identity, component oracles, and existing thresholds. See
   [Qualified score gates](validation/README.md#qualified-score-gates).

## Invariants a change must not break

- Preserve exact cancellation for equal SSIMULACRA 2 planes. Its error maps
  divide the difference of numerator and denominator. Keep flat score-weight
  indexing and channel pruning consistent across scale counts.
- Source pixels use Vulkan binding 3 and metadata uses binding 2. Internal
  arena offsets are absolute. External input offsets are relative to their
  source binding. Bind each image's own source before preparing it.
- Each pipeline uses one queue and permits one submission in flight. Preserve
  token validity, handle ownership, and mutation guards. The public
  headers and [Vulkan synchronization guide](docs/vulkan-synchronization.md)
  describe the buffer and queue contracts.
- The SSIMULACRA 2 scale ladder requires a power-of-two invocation count and
  four outputs per invocation. Current build variants use 32×32 and 32×16
  tiles. Keep planner limits and shader variants in sync.
- Memory queries and construction share layout and dispatch checks. Count
  per-heap allocations, including CVVDP's private device-local history, in
  memory reporting and worker budgets. Preserve the independent temporal
  state of VMAF and CVVDP. The shared scratch workspace serves serial
  SSIMULACRA 2 and Butteraugli evaluation. See [Memory planning](docs/memory.md)
  and [Shared workspace](docs/shared-workspace.md).
- Use CVVDP's epsilon-adjusted norm. Report whole-clip JOD after finalization
  and keep frames of one clip on the same sequence worker.
  See [CVVDP contracts](docs/cvvdp.md).
- Keep BestSource optional and CLI-local. Selected decoded pixels stay on
  the same GPU through Vulkan packing and scoring. BestSource can temporarily
  read back candidates while verifying seeks. Preserve frame order, format,
  timing, and ownership across the handoff. Keep the existing Y4M path and the
  decoder-free installed core.

Respect upstream licenses and record provenance by pinned revision, file,
and mechanism where applicable. Keep Vship sources read-only and use its
public API for comparisons. Implement borrowed ideas independently and keep
peer source and tables out of VQM. Keep the SSIMULACRA 2 channel-pruning
provenance with its implementation in `src/metrics/ssimulacra2/score.cc`.

## Profiling with Nsight Systems

Start with the built-in stage timers from a compatible Vulkan device.
Set `ORIGINAL` and `DISTORTED` to a matching input pair, or use the generated
corpus pair above.

```sh
bash bench/profile.sh --tool stages --pair "$ORIGINAL" "$DISTORTED"
```

On supported Linux/NVIDIA setups, `bench/profile_nsys.sh --list` previews the
optional capture matrix without a profiler installation. For
capture, the wrapper expects the Nsight gateway at
`/opt/nvidia-nsight-systems-bin/skills/nsight-systems` and can install missing
gateway Python dependencies into the user site. Use this optional wrapper
only on an environment prepared for it. Pass an explicit `--pair` and `--out`
for the inputs and report destination. Counter collection depends on the
device, driver, and local permissions. Keep system policy changes outside
the repository workflow. Use the built-in stage timers when the gateway or
counters are unavailable.
