# Butteraugli score-drift diagnostics

These patches reproduce the algorithm comparisons in the
[Butteraugli drift investigation](../../notes/system/butteraugli-drift.md).
Production now contains all three corrections. To reproduce the historical
comparison, first reconstruct the pre-fix shaders in an isolated source copy,
then apply the three original diagnostic patches cumulatively. These commands
leave the maintained checkout unchanged. This is a correctness investigation;
the one-call scoring commands below do not establish a performance result.

| Arm | Patch applied after the preceding arm | Change |
| --- | --- | --- |
| `baseline` | `00-reconstruct-baseline.patch` | Reconstructed implementation from before the three corrections. |
| `residual` | `01-frequency-residual.patch` | Compute HF from the raw blurred MF before adjusting MF's range. |
| `suppression` | `02-suppress-x-by-y.patch` | Also suppress HF X using HF Y before separating HF/UHF. |
| `malta` | `03-malta-taps.patch` | Also correct four first-tap X signs in the UHF Malta unit. |

These are **cumulative arms**. Patch 00 starts from the fixed production source
and reconstructs the historical shaders. Patch 01 then applies to that baseline;
patch 02 expects patch 01. Patch 03 touches the separate Malta shader and also
applies independently to the historical baseline, but its result in this
sequence includes all three corrections. Patch 01 updates the epilogue's
comment to describe subtraction of the raw blur. Patch 02 performs suppression
while loading the UHF source tile, so the blur taps and unblurred center receive
the same transform without an extra dispatch. The final diagnostic arm has the
same shader arithmetic as production, whose additional comments describe the
immutable source-plane contract. Historical arms predate the corrected test
goldens; the commands below rebuild the scoring tools for their comparison.

The derivation uses the pinned BSD-3-Clause libjxl source at
`.reference/libjxl/lib/jxl/butteraugli/butteraugli.cc`: residual order at
449–464, `SuppressXByY` at 367–390 and 472, and Malta taps at 792, 800, 925 and
941. The starting shader Git blob hashes were
`079a1cdaaffa3696217f8e69f65e09bfb81c473a` for `butter_blur.comp` and
`33ad879ba78ff8128ddbef66c8fb489673a34c66` for `butter_malta.comp`.

## Capture and build the source

Run from the repository root in one shell. The snapshot includes current tracked
and untracked maintained files, preserving an already-modified working tree.
Ignored builds, reports, corpus data and reference checkouts are excluded. An
existing source destination is refused. Builds remain disposable under `build/`;
scores and logs go to `artifacts/reports/`.

```sh
set -eu
probe_repo=$PWD
probe_root="$probe_repo/build/butteraugli-drift"
probe_source="$probe_root/source"
probe_build="$probe_root/arm-build"
probe_reports="$probe_repo/artifacts/reports/butteraugli-drift/reproduction"
probe_patches="$probe_repo/bench/experiments/butteraugli_drift"
mkdir -p "$probe_reports"
python3 - <<'PY'
from pathlib import Path
import shutil
import subprocess

root = Path.cwd()
destination = root / "build/butteraugli-drift/source"
if destination.exists():
    raise SystemExit(f"refusing existing source copy: {destination}")
files = subprocess.check_output([
    "git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"
]).decode().split("\0")
for name in files:
    if not name or not (root / name).is_file():
        continue
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / name, target)
    shutil.copymode(root / name, target)
PY
patch -p1 -d "$probe_source" < "$probe_patches/00-reconstruct-baseline.patch"
cmake -S "$probe_source" -B "$probe_build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DVQM_WARNINGS_AS_ERRORS=ON
cmake --build "$probe_build" --target vqm_bench vqm_core -j8
```

The core build needs Vulkan, the shader compiler, and the project's usual
PNG/JPEG dependencies; it needs no external reference checkout. Record the
compiler, Vulkan device/driver, inputs and source state with any new result.

## Make a deterministic spatial counterexample

This opaque 64x64 sRGB pair exercises spatial detail that the two gradient
fixtures miss. It requires Pillow. The pair list contains absolute paths
separated by a tab, with no header, comments or blank lines, so both runners
below accept it. Replace the list with a larger matched corpus for aggregate
claims; this one pair demonstrates a failure rather than estimating its
frequency.

```sh
python3 bench/experiments/butteraugli_drift/make_checker.py \
  --out-dir "$probe_root/fixtures"
probe_pairs="$probe_root/fixtures/checker.pairs"
```

The generator's default output directory is
`artifacts/reports/butteraugli-drift/synthetic/`; `--out-dir` selects the build
fixture directory above. The published baseline scored this pair `2.42218304`
against pinned libjxl's `2.57976222`, a `6.11%` difference. The independent
double oracle gives `2.57974731586`.

## Score each cumulative arm

```sh
score_arm() {
  cmake --build "$probe_build" --target vqm_bench -j8 \
    > "$probe_reports/$1-build.log" 2>&1
  "$probe_build/vqm_bench" --metric butteraugli \
    --pairs-file "$probe_pairs" --score-only --iterations 1 --warmup 0 \
    --intensity-target 80 --hf-asymmetry 1 \
    --csv "$probe_reports/$1.csv" \
    > "$probe_reports/$1-scores.txt" 2> "$probe_reports/$1.log"
}
score_arm baseline
patch -p1 -d "$probe_source" < "$probe_patches/01-frequency-residual.patch"
score_arm residual
patch -p1 -d "$probe_source" < "$probe_patches/02-suppress-x-by-y.patch"
score_arm suppression
patch -p1 -d "$probe_source" < "$probe_patches/03-malta-taps.patch"
score_arm malta
```

The patches also use standard `a/src/...` and `b/src/...` paths for
`git apply -p1` in a standalone copied checkout. The commands above use `patch`
to address the copy explicitly when it lives underneath the maintained Git
checkout. Preserve each arm's CSV before rebuilding: the executable is replaced.
Inspect exit statuses and stderr as well as scores. Keep identical input lists,
decode declarations and parameters across arms. The CSV includes incidental
timings; use only its scores for this experiment.

## Compare the independent CPU oracle

`oracle_probe.cc` decodes the same images with `image_io`, inverts the sRGB
transfer, and calls the existing independent double-precision test oracle. It
prints `score,distance_p3,width,height,original,distorted` as CSV. It accepts
integer RGB/grayscale PNG/JPEG and rejects nonopaque alpha. It assumes the
decoder's documented color declarations, so embedded ICC differences are outside
this arithmetic comparison.

```sh
c++ -O3 -std=c++20 -I"$probe_source/include" -I"$probe_source/src" \
  -I"$probe_source/tools/image_io" \
  -I"$probe_source/tests/metrics/butteraugli" \
  "$probe_source/bench/experiments/butteraugli_drift/oracle_probe.cc" \
  "$probe_source/tests/metrics/butteraugli/butter_ref.cc" \
  "$probe_build/libvqm_io.a" "$probe_build/libvqm_core.a" \
  -lpng -lz -ljpeg -lvulkan -pthread -o "$probe_root/oracle_probe"
"$probe_root/oracle_probe" --pairs-file "$probe_pairs" \
  > "$probe_reports/oracle-double.csv"
```

For a precision-isolation control, copy the oracle sources and change their
scalar arithmetic to `float`; keep their algorithm unchanged:

```sh
python3 - <<'PY'
from pathlib import Path

root = Path("build/butteraugli-drift")
source = root / "source/tests/metrics/butteraugli"
destination = root / "oracle-float"
destination.mkdir(parents=True, exist_ok=True)
for name in ("butter_ref.cc", "butter_ref.h"):
    (destination / name).write_text((source / name).read_text().replace("double", "float"))
PY
c++ -O3 -std=c++20 -fsingle-precision-constant \
  -I"$probe_root/oracle-float" -I"$probe_source/include" \
  -I"$probe_source/src" -I"$probe_source/tools/image_io" \
  "$probe_source/bench/experiments/butteraugli_drift/oracle_probe.cc" \
  "$probe_root/oracle-float/butter_ref.cc" \
  "$probe_build/libvqm_io.a" "$probe_build/libvqm_core.a" \
  -lpng -lz -ljpeg -lvulkan -pthread -o "$probe_root/oracle_probe_float"
"$probe_root/oracle_probe_float" --pairs-file "$probe_pairs" \
  > "$probe_reports/oracle-float.csv"
```

This float control distinguishes algorithm differences from ordinary precision
effects. It does not emulate GPU FMA contraction, native `log2`, instruction
ordering or the reference decoder. The unchanged double oracle remains the
independent comparator; the float copy is generated research output, not a new
maintained oracle. Direct comparisons with pinned libjxl still need its fetch
and copied reference build, as described in the investigation note.

## Isolate the remaining JPEG ICC differences

The eight corrected full-corpus residuals larger than `0.01` were MozJPEG
inputs. Their [compact evidence CSV](../../notes/measurements/butteraugli-drift-icc.csv)
records the original score, corrected score, reference with/without ICC, removed
byte count and decoded pixel comparison. `strip_jpeg_icc.py` removes only ICC
APP2 segments before the first JPEG scan and copies the scan bytes verbatim. It
creates the destination's parent directories and refuses to overwrite a file.

For the largest remaining residual, run the following from the repository root
after the cumulative corrected arm above. The corpus and reference build are
separate prerequisites:

```sh
python3 tools/fetch_references.py --only cid22,libjxl
python3 tools/build_references.py --only libjxl --jobs 8
probe_original="$probe_repo/validation/data/CID22_validation_set/original/2887497.png"
probe_jpeg="$probe_repo/validation/data/CID22_validation_set/compressed/2887497/mozjpeg/q60.jpg"
probe_noicc="$probe_reports/2887497-q60-noicc.jpg"
python3 bench/experiments/butteraugli_drift/strip_jpeg_icc.py \
  "$probe_jpeg" "$probe_noicc"
python3 - "$probe_original" "$probe_jpeg" "$probe_noicc" \
  "$probe_reports/icc.pairs" <<'PY'
from pathlib import Path
import sys
from PIL import Image

original, embedded, stripped, output = sys.argv[1:]
with Image.open(embedded) as first, Image.open(stripped) as second:
    assert first.convert("RGB").tobytes() == second.convert("RGB").tobytes()
Path(output).write_text(f"{original}\t{embedded}\n{original}\t{stripped}\n")
PY
"$probe_repo/build/libjxl-build/tools/ss2_libjxl_bench" \
  --metric butteraugli --score-only --pairs-file "$probe_reports/icc.pairs" \
  --iterations 1 --warmup 0 --threads 1 --decode-threads 1 \
  > "$probe_reports/icc-reference.txt" 2> "$probe_reports/icc-reference.log"
"$probe_build/vqm_bench" --metric butteraugli --score-only \
  --pairs-file "$probe_reports/icc.pairs" --iterations 1 --warmup 0 \
  --csv "$probe_reports/icc-corrected.csv" \
  > "$probe_reports/icc-corrected.txt" 2> "$probe_reports/icc-corrected.log"
```

All eight probes removed 554 bytes and left the decoded pixels identical. Seven
then agreed with corrected GPU scores within `5.96e-6`; this worst example
retained `0.0002408` difference instead of `0.03281712`. The reference retains
the profile in `lib/extras/dec/jpg.cc:238–275` and uses CMS when converting to
linear sRGB in `lib/jxl/enc_butteraugli_comparator.cc:33,67`; our
`tools/image_io/io.cc:293–318` declares sRGB without retaining ICC. Removing the
profile isolates that input difference. It does not establish exact agreement
between the two JPEG decoding and arithmetic paths.
