# SpEED eigensolver stress provenance

> Historical research note. The [pooling and parallelism report](../vmaf-v1-pooling-parallelism.md) records the final
> measured result and retained implementation. Current source may differ from
> the reviewed snapshots. Relative experiment files and commands below refer
> to the archived research directory
> `artifacts/reports/vmaf-v1-pooling-parallelism/stress-provenance/`.

`matrices.bin` contains 10,000 tridiagonal SPD matrices, each rounded to FP32
before its independent reference calculation. `manifest.json` records the
seed, generator version, binary format and file SHA-256. `truth.csv` records
the SHA-256 of each 49-float input, its generation band and FP64 minimum and
maximum eigenvalues. FP64 eigenvalues are approximate near the threshold for
poorly conditioned matrices; `high-precision-audit.py` uses 100-digit Decimal
Sturm counts for additional evidence, without claiming a formal interval proof.

The snapshot shaders and drivers provide the exact tested programs. `qr` is
the QR shader at `82e6135`; `sturm` is a candidate applied to the same base;
`legacy-guard` is a candidate on `01735d6`; `final` is `764188f`.
The legacy guard uses 2602 scratch words per matrix pair and computes its
factorization inside the eigen workgroup; its `spectral_status` column is the
already combined status, and `factor_status` is `nan`. The other programs use
2604 words and dispatch factor and spectral work concurrently; `combined_status`
matches the production status merge. All report the exact 25 output eigenvalue
bit patterns as eight-digit hexadecimal FP32 words. Every row also contains the
input's 64-bit FNV-1a hash so comparison verifies it against the dataset.

The scripts expect `/tmp/vqm-eigen-stress-data`; copy `matrices.bin`,
`manifest.json`, and `truth.csv` there, or run `generate.py` with NumPy to
recreate them. Generation can vary across NumPy versions; verify the resulting
SHA-256 against this saved manifest. Results are CSVs in that directory.

To rebuild one program, create a detached worktree of its base commit, copy
`eigen_stress.cc` into the worktree root (or `eigen_stress_legacy_guard.cc`
for the legacy guard), replace `src/shaders/vmaf_speed_stats_eigen.comp` with
the corresponding `shader-*.comp`, and append this target to its CMakeLists:

```cmake
add_executable(vqm_eigen_stress eigen_stress.cc)
target_include_directories(vqm_eigen_stress PRIVATE src)
target_compile_features(vqm_eigen_stress PRIVATE cxx_std_20)
target_link_libraries(vqm_eigen_stress PRIVATE vqm_core Vulkan::Vulkan)
```

Configure with `-G Ninja -DCMAKE_BUILD_TYPE=Release -DVQM_BUILD_CLI=OFF
-DVQM_BUILD_BENCH=OFF -DVQM_BUILD_TESTS=OFF`, then build target
`vqm_eigen_stress`. Run the four binaries serially, passing the binary dataset
and output CSV path. For example, after building final:

```sh
./build/vqm_eigen_stress /tmp/vqm-eigen-stress-data/matrices.bin \
  /tmp/vqm-eigen-stress-data/final.csv
python3 compare.py qr final
python3 high-precision-audit.py
```

The comparison verifies every input ID and FNV against the binary and SHA
metadata before reporting status disagreements. Its FP64 expected status is
approximate around the cutoff. The high-precision audit examines cases where
available GPU variants disagree, or accepts explicit integer IDs. It writes
`high-precision-audit.csv`. Save the resulting CSVs and JSON alongside this
provenance after running; the original `qr`, `sturm`, and `guard` outputs are
included here.
