# ColorVideoVDP reference validation

ColorVideoVDP's original implementation is the numerical oracle. The pinned
base model is **v0.5.7**, commit
`2a268bce8d56e2f3abde46df3927d8a633707a24`, from
[Graphics and Displays group](https://github.com/gfxdisp/ColorVideoVDP).
The original MATLAB interface invokes this Python implementation; it is not a
second, native MATLAB numerical oracle. CVVDP's JOD score is higher-is-better,
identity is 10, and negative scores are valid.

The installed core and ordinary tests use checked-in tables and compact
independent fixtures. They do not import Python, PyTorch, MATLAB, a video
decoder or an upstream source checkout. All commands below run optional
reference-generation workflows, outside the core build.

## Pinned reference environment

Use `uv` and the reference-build step to prepare the repository's managed
`.venv/` and keep the original checkout read-only:

```sh
uv run --no-project --python 3.12 tools/fetch_references.py --only ColorVideoVDP
uv run --no-project --python 3.12 tools/build_references.py --only cvvdp
```

The build step provisions CPU-only Python dependencies from the pinned
requirements, verifies the original revision and records runtime fingerprints.
It preserves an existing environment rather than deleting it. The
[single-command video benchmark](../bench/README.md) performs this setup
alongside the other four metrics' originals. No system-wide Python packages
are installed, and the original package is imported from its pinned source.

The initial oracle generation used the following manual environment recipe.
It is retained as provenance for the historical `build/cvvdp-reference-env/`
paths in measurements; current workflows use the managed `.venv/` above:

```sh
python3 tools/fetch_references.py --only ColorVideoVDP
python3 tools/fetch_references.py --check --only ColorVideoVDP
uv venv --python 3.12 build/cvvdp-reference-env
uv pip install --python build/cvvdp-reference-env/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match -r tools/cvvdp-reference-requirements.txt
```

The requirements file records the exact environment used for the checked-in
oracles: CPython 3.12.14, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu and NumPy
2.5.4. GPU execution is deliberately excluded from this oracle. The original
package is imported through its pinned checkout, rather than installing a
potentially newer `cvvdp` release. `sys.dont_write_bytecode` prevents cache files
from modifying that checkout. Reference scripts verify the pin and checkout
cleanliness before generation.

The generated C++ arrays contain the original calibration and CSF values, with
MIT attribution in [COLORVIDEOVDP-LICENSE](../licenses/COLORVIDEOVDP-LICENSE).
The generator checks SHA-256 for calibration, bundled CSF LUT, display presets
and source-color definitions. It evaluates logarithmic axes and composed
RGB→XYZ→LMS→DKL matrices in FP32 with the original functions, preserving their
rounding. The bundled LUT is a pinned asset; regenerating it with a newer
castleCSF fit would change the target model.

```sh
.venv/bin/python tools/generate_cvvdp_tables.py --check
.venv/bin/python tests/make_cvvdp_oracles.py
```

`tools/generate_cvvdp_tables.py` writes the maintained metric-local
`src/metrics/cvvdp/cvvdp_model_data.inc`; `--check` only compares it.
`tests/make_cvvdp_oracles.py` writes the maintained sequence and spatial
fixtures under `tests/metrics/cvvdp/`. Its diagnostic report goes to
`artifacts/reports/cvvdp/oracle-characterization.json`. Use `--output`,
`--spatial-output`, and `--report` to direct a separate run to local paths.

## Oracle inputs and precision

Every pair is a deterministic raw `torch.float32` tensor in `[B,C,F,H,W]`
order, with the source interpretation and physical display explicitly selected.
No image or video file loader participates. The formula in
`tests/make_cvvdp_oracles.py:sample` is reproduced independently in C++ tests.
Ordinary samples are normalized encoded RGB; linear samples are absolute nits.
The report includes the exact raw tensor SHA-256, resolved pixels per degree,
first-pixel DKL values, per-frame/channel/band latent distortions and temporal
filter coefficients. Each full score is evaluated three times with one CPU
thread and `torch.set_float32_matmul_precision("highest")`.

The initial matrix covers ordinary and identical images, near-zero error,
flat-field intensity changes, opponent color, small and mixed-parity sizes,
FHD/4K display presets, P3 primaries, PQ, HLG, absolute linear input, one-frame
video/image equivalence, short clips, ring wrap, flicker and negative JOD.
Filters cover 24/30/60/120 and 24000/1001, 30000/1001, 60000/1001 fps. The
original coefficients use inclusive frequency sampling and odd-length irFFT;
there is no Nyquist-only coefficient in an odd-length transform.
Whole-clip cases additionally exercise the supported 1 and 240 fps limits,
40-frame 120 fps history, repeated fractional-rate ring wraps and frame counts
just below, equal to and just above the 30 fps filter length.

All repeated scores in the retained CPU characterization are identical. This
establishes deterministic reference behavior in that environment, not CPU/GPU
bit identity. GPU parity gates must inspect latent features and complete JOD;
1e-3 absolute JOD was the implementation plan's initial target. Final GPU
bounds and measured outcomes belong with the qualification report.

Flat uniform black versus white scores 10 in this pinned model: its local and
baseband normalization removes that achromatic intensity change. The fixture
preserves observed reference behavior instead of assuming every visually
obvious intensity change must reduce JOD. Opponent-color and temporally
flickering fixtures exercise large distortions independently.

## Mixed-parity pyramid compatibility

The original `pycvvdp/lpyr_dec.py:206` tests height parity in the right-column
padding branch. This is numerically observable, not just a suspected typo.
For `arange(20)` reshaped to height 5, width 4, the original five-tap reduction
produces a right-column error of up to 4.300001 compared with a mathematically
symmetric reduction. Height 4, width 5 produces up to 3.387501. Matched-parity
sizes agree within 1.91e-6 from reduction order.

The v0.5.7 compatibility implementation preserves that branch. Correcting its
edge behavior silently would break the declared numerical target. A future
upstream fix needs a separate compatibility/version decision and regenerated
fixtures. The independent characterization script computes both paths without
editing the upstream checkout; the JSON retains both complete reduced planes.

## Corpus parity

`validation/cvvdp.py` compares the pinned original against one VQM benchmark
process over the pair list. Start with an evenly sampled 20-pair subset, then
run all 4,292 CID22 distortion pairs:

```sh
.venv/bin/python validation/cvvdp.py --limit 20 \
  --out-dir artifacts/reports/cvvdp/cid22-20
.venv/bin/python validation/cvvdp.py --limit 0 \
  --out-dir artifacts/reports/cvvdp/cid22-full
```

The default absolute JOD tolerance is fixed at 1e-3. Every source is decoded to
canonical opaque RGB8 PNG under `.cache/validation/cvvdp-rgb8/`; both arms read
those exact samples. This excludes differences between JPEG decoder settings.
The original arm takes those pixels through its raw-tensor API with explicit
sRGB source interpretation and `standard_4k` physical display. Each row retains
source and decoded-pixel hashes, dimensions and both scores. The candidate is
copied into the report directory before execution, preserving the measured
binary and SHA-256 even if the working build is replaced later.

The original score cache is keyed by decoded pixels, display, model pin,
Torch version and precision settings. `--reference-only` populates that cache
without running VQM and reports `REFERENCE ONLY`, never a passing comparison.
`--pairs-file` accepts the repository's tab-separated custom still-pair lists;
`--display standard_fhd` explicitly changes both arms. Use separate output
directories for independent qualification runs. These are correctness checks,
not benchmarks: input canonicalization and unpaired reference timings do not
establish a speed comparison. CID22 is a still-image SDR corpus; the separate
raw-tensor fixtures provide temporal, HDR and display-color coverage.
