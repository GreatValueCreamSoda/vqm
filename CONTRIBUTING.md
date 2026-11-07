# Contributing

Thank you for helping improve the project. Please describe the problem and the
behavior of a change in your pull request, and include the checks you ran. For
larger changes, an issue or discussion first can help establish the intended
behavior.

For a bug report, include the smallest reproducible input, the command you ran,
and the expected and observed behavior. For Vulkan issues, include the device
and driver reported by `./build/vqm_bench --device-info` and the build preset.

Testing is local only. The project does not support CI testing, GitHub Actions
workflows, external test services or automated status checks. Include the local
checks and results in your pull request; keep the existing tests and independent
reference oracles working.

## Build and test

A C++20 compiler, CMake 3.25 or newer, Ninja, the Vulkan headers and loader,
`glslc`, libpng, libjpeg, `uv`, Python 3.11+, NumPy, Pillow, and FFmpeg are needed
for the full build and test suite. Use `uv` to provision Python 3.12 and the
pinned local test dependencies in `.venv/`. For example, on Ubuntu 24.04 with
`uv` already installed:

```sh
sudo apt-get install build-essential cmake ninja-build glslc libvulkan-dev \
  mesa-vulkan-drivers libpng-dev libjpeg-dev ffmpeg
if [ ! -d .venv ]; then
  uv venv --no-project --python 3.12 .venv
fi
uv pip install --python .venv/bin/python -r tools/test-requirements.txt
cmake --preset release -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build --preset release --parallel
ctest --preset release
```

This preserves an existing environment and its unrelated packages. NumPy and
Pillow are the lightweight local test dependencies; the core build and tests
need no reference checkout or external corpus. The reproducible hosted-video
benchmark provisions the larger pinned reference runtime and its dependencies:

```sh
uv run --no-project --python 3.12 bench/benchmark.py
```

See [the benchmark guide](bench/README.md) for system prerequisites, exact
workloads and output reports.

A working Vulkan device is required to exercise the metric. Mesa's software
Vulkan driver can run the correctness suite without a physical GPU; it is not
representative of GPU performance. On a GPU-equipped machine,
`./build/vqm_test --require-gpu` additionally fails if no
usable Vulkan device is detected. Run this when validating changes to the
backend. Some optional device features can still be skipped.

For local Vulkan API and synchronization checks, install `rg` and the Vulkan
validation layers (`ripgrep` and `vulkan-validationlayers` on Ubuntu), then run:

```sh
mkdir -p artifacts/reports/local-validation
validation_status=0
VK_LOADER_DEBUG=error,warn \
  VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LAYER_VALIDATE_SYNC=1 \
  ./build/vqm_test --require-gpu \
  > artifacts/reports/local-validation/vulkan-validation.log 2>&1 \
  || validation_status=$?
validation_scan_status=0
rg -n 'VUID-|Validation Error|SYNC-HAZARD|Vulkan Loader.*ERROR' \
  artifacts/reports/local-validation/vulkan-validation.log \
  || validation_scan_status=$?
test "$validation_status" -eq 0 && test "$validation_scan_status" -eq 1
```

The final command succeeds when the test process passes and `rg` returns 1 for
no matches. Inspect the captured log and resolve any matching validation
error or synchronization hazard. Vulkan validation can report an API violation
while metric tests still pass. Loader errors, including a missing requested
validation layer, also invalidate the check: the loader can continue without
that layer and the test process can still exit successfully.

After changing shader compilation or embedding, use `spirv-val` from SPIRV-Tools
to check the modules listed by the Release build.

```sh
(
  set -e
  while IFS= read -r shader; do
    spirv-val --target-env vulkan1.1 "build/shaders/$shader.spv"
  done < build/generated/vulkan_shader_manifest.txt
)
```

The `asan` preset builds with AddressSanitizer and UndefinedBehaviorSanitizer.
When supported by the platform, run the local sanitizer suite for memory-sensitive
changes:

```sh
cmake --preset asan -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build --preset asan --parallel
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 ctest --preset asan
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
  ./build-asan/vqm_test --require-gpu
```

CTest also exercises installed executable and shared-library consumers and
configures an embedded consumer. `AGENTS.md` documents the
full project gates, accuracy corpus, and benchmark protocol. Benchmark changes on
real hardware and report the device, driver, workload, and results; software Vulkan
timings are not comparable.

Follow the ownership and dependency boundaries in [the architecture guide](docs/architecture.md)
when adding a metric, shared helper or shader.

## Formatting and provenance

Format C++, headers, and shaders using the version specified in `.clang-format`;
older versions do not understand all of its options. Keep changes focused and
include tests for behavior changes. Respect upstream licenses when using
references: cite independent mechanisms and do not copy code from incompatible
sources. The reference checkouts under `.reference/` are for comparison and are
not part of the build or distribution.

Run `python3 tools/check_quality.py` to check every maintained C++, header and
shader file recursively with clang-format 23. After configuring a build
with a compilation database, `python3 tools/check_quality.py --lint` also checks
production C++ in the library, CLI, benchmark and image decoders against
`.clang-tidy`; `--build-dir` selects another build directory. This command
enforces the documented nesting limit. `--include-tests` extends the check to
test matrices and reference oracles for advisory review; those intentionally
retain some deeper data-generation loops and do not currently pass that limit.
Formatting always covers tests as well as production code.
`--lint-only` runs the production nesting check without the formatter, for local
environments that provision those tools separately.

Use Ruff for Python linting and YAPF for formatting. The repository's
[`ruff.toml`](ruff.toml) and [`.style.yapf`](.style.yapf) carry the project settings;
both tools discover them automatically. They use 80 columns, four-space
indentation, double quotes, type annotations, and PEP 257 docstrings. Write
docstring fields in Sphinx/reStructuredText style (`:param name:`, `:returns:`,
and `:raises ExceptionType:` as applicable). Functions are limited to complexity
7, 30 statements, and two nested control-flow blocks. YAPF packs arguments
that fit on a continuation line, including calls with trailing commas.
Ruff enforces the 80-column limit. Wrap comments, docstrings, and string literals
explicitly when YAPF cannot split them automatically.

Check all maintained Python scripts from the repository root:

```sh
uvx ruff check .
uvx yapf --diff --recursive bench tools validation tests
```

For scoped changes, use `uvx ruff check path/to/file.py` and
`uvx yapf --diff path/to/file.py`. Apply formatting with
`uvx yapf -i path/to/file.py`, then repeat both checks.
