# Hosted video reference correlation

The normal `vqm` CLI consumes complete progressive, fixed-frame-rate Y4M clips.
File-based qualification uses this CLI path for SSIMULACRA 2, Butteraugli,
VMAF v0, VMAF v1 and CVVDP. The installed core retains decoder-independent
image and sequence APIs, which the C++ unit tests exercise directly.

After the [local build and Python setup](../CONTRIBUTING.md#build-and-test),
run reference correlation for the five established metrics:

```sh
.venv/bin/python -m validation.temporal.run --profile qualification
```

The runner fetches any missing pinned source prefixes, prepares the matching
representations, scores complete clips through `vqm` and compares each metric
with checked-in independent original scores. `--profile smoke` runs the small
matrix. The default metric selection is SSIMULACRA 2, Butteraugli, VMAF v0,
VMAF v1 and CVVDP; `--metrics` selects an explicit subset. Each run writes a
`summary.json` with the selected metrics' coefficients, coverage and statuses.
The CLI also scores complete clips with [PSNR-HVS](psnr-hvs.md) in its native
YUV domain. That metric is outside this qualification matrix.
Use that run's evidence when describing its outcome. The
[five-metric reference-correlation record](../bench/notes/system/all-metric-reference-correlation.md)
documents the full matrix's settings, measured correlations and numerical
diagnostics. The historical native three-metric VMAF/CVVDP qualification
remains recorded under its original policies and report paths.

## Score a complete video

Generate a three-frame Y4M clip with FFmpeg and score it against itself.

```sh
sample_dir=$(mktemp -d)
ffmpeg -hide_banner -loglevel error -f lavfi \
  -i 'testsrc2=size=32x32:rate=30' -frames:v 3 -pix_fmt yuv420p \
  "$sample_dir/sample.y4m"
./build/vqm --metric ssimulacra2 --json \
  "$sample_dir/sample.y4m" "$sample_dir/sample.y4m"
rm -- "$sample_dir/sample.y4m"
rmdir -- "$sample_dir"
```

Expect a score of 100 for the clip and each of its three frames.

For your own clips, decode both inputs to progressive Y4M before scoring.
Align them to the same content frames, coded dimensions, and fixed frame rate.
Keep chroma siting consistent with the Y4M header. When the Y4M defaults
differ from the decoded samples, pass the matching matrix, transfer, primaries,
and range through the `--input-*` options. These options apply to both clips.
[Input formats and defaults](reference.md#input-formats)

```sh
./build/vqm --metric vmaf-v1 original.y4m distorted.y4m
./build/vqm --metric cvvdp --json original.y4m distorted.y4m
./build/vqm --metric ssimulacra2 --json original.y4m distorted.y4m
```

JSON contains `sample_id`, `metric`, `score`, `frames`, `frame_results`,
`core_seconds` and `decode_seconds`. VMAF frame results include its features
in emitted index order. SSIMULACRA 2, Butteraugli, and VMAF average their frame
scores. PSNR-HVS averages weighted frame errors before converting to dB and
represents a positive infinite score as the JSON string `"inf"`. CVVDP converts
pooled whole-clip distortion to JOD and returns an empty `frame_results` array.
The optional `--verbose` mode adds `frame_stats` and includes diagnostic CVVDP
frame JODs. The qualification runner uses the default JSON output.

`--pairs-file` accepts UTF-8 TSV rows `sample_id<TAB>original<TAB>distorted`
and requires `--json`. It runs all clips in one process, resetting sequence
history between clips. IDs must be nonempty and unique. A failed clip emits
no partial JSON object. Geometry, input kinds, rational timing and frame counts
must match; truncated, interlaced or unspecified-timing streams are rejected.
The sequence reader uses coded pixel geometry and does not resample the Y4M
sample aspect ratio. The hosted qualification preparation explicitly declares
square metric pixels; callers needing another display geometry must resample
their inputs before scoring.

```sh
./build/vqm --metric vmaf-v0 --json --pairs-file clips.tsv \
  --iterations 5 --warmup 1
```

Each iteration evaluates the complete clip from reset. JSON reports the
iteration with median core duration; core time covers Push/Compute and
finalization, while decode is measured separately. Pipeline creation/reset and
external preparation are outside core time. These are reproducible workloads,
not comparative speed claims. Use the repository's interleaved protocol for
performance claims. Existing `vqm_bench` VMAF repeated-pair workloads remain
separate microbenchmarks.

## Bootstrap and benchmark from a fresh clone

With the [benchmark prerequisites](../bench/README.md) installed, run:

```sh
uv run --no-project --python 3.12 bench/benchmark.py
```

This default video suite fetches the pinned Cloudinary, libjxl, libvmaf and
ColorVideoVDP originals, builds the native references and `vqm`, provisions
the locked CPU reference environment in the repository's `.venv/` using `uv`,
and downloads the hosted smoke inputs. `uv` can provision the bootstrap Python;
host Python does not need NumPy or PyTorch installed. The compatible
`python3 bench/benchmark.py` entry point accepts an existing Python 3.11+.
Existing environments are preserved. No reference checkout
is built or modified in place.
Checked-in independent original scores supply the agreement gate; regenerating
those scores remains the separate workflow below. `vqm_bench` is built for
device-information probing, while all clip scoring uses the normal CLI.

The default workload evaluates all five established metrics on 39 complete
clip pairs each, with 12 frames per clip, five rotating-order rounds, 20
timed iterations and three warmups. `--profile qualification` selects the
390-pair, 40-frame profile. `--setup-only` prepares dependencies without
timing; `--dry-run` shows the execution plan. `--metrics` selects a named
subset; `--rounds`, `--iterations`, `--warmup`, `--jobs` and `--out-dir`
control the run. The optional `--suite images` preserves the previous
synthetic-image peer comparison.

Reports default to a fresh UTC timestamp directory under
`artifacts/reports/video-benchmark/`. An explicit output directory must be
empty. Use these files to inspect a run.

| Path within the report | Contents |
| --- | --- |
| `summary.md` | Valid rounds and median core, decode, and CLI wall times by metric. |
| `summary.json` | Run parameters, provenance, each arm's outcome and report hashes, and timing distributions. |
| `round-NN/<metric>/` | Input manifest, pair list, complete CLI JSONL, agreement report, and candidate timing. |
| `round-NN/<metric>-workflow.log` | Output and diagnostics from the validation command. |

Core and decode measurements describe each clip's selected median-core
iteration. Full CLI process wall time includes repetitions, warmups, and setup.
Outer workflow
wall time additionally includes preparation, verification and quality gates.
A reference adapter's wall time is not presented as comparable CPU/GPU
compute speed. These timings remain specific to the declared inputs and
hardware; optimization claims require the repository's interleaved
baseline/candidate protocol.

The [fresh-repository benchmark record](../bench/notes/system/fresh-video-benchmark.md)
documents a completed single-command setup and all 25 default metric/round
records, with the exact source snapshot, raw evidence and timing boundaries.

## Fetch pinned hosted clips

```sh
python3 -m validation.temporal.fetch
python3 -m validation.temporal.fetch --check
```

[sources.json](../validation/temporal/sources.json) pins the first 40 complete
frames of 30 SD clips hosted in [Xiph's collection](https://media.xiph.org/video/derf/),
about 180 MB in total. Each entry records its HTTPS URL, requested byte range,
SHA-256, full upstream size, original header, geometry and rational timing.
The downloader requires the exact HTTP range response and complete frame
boundaries. It validates cached files offline and publishes fresh downloads
atomically. Corrupt existing files are preserved and reported for inspection.

`--only foreman_cif,bus_cif,mobile_cif` fetches the smoke sources. Original
prefixes live in ignored `validation/download/temporal/`. Per-source copyright
and README notices are linked by the catalogue; source media is downloaded
from its host, not redistributed in this repository. Some upstream timing was
inferred by the host; validation reproduces the declared timing rather than
claiming camera timestamp accuracy.

Preparation preserves the declared frame rate and declares square metric
geometry without rescaling. Metric families have separate canonical input
representations; their hashes and settings are part of each report.

- VMAF and CVVDP retain the original temporal representation. Source 4:2:0
  chroma is expanded once by exact nearest replication to 8-bit YUV444. VMAF
  receives those native coded samples. Original CVVDP receives an explicit
  FP32 limited-range BT.601 RGB conversion; the CLI uses its public GPU
  conversion. Both CVVDP arms use the frozen `standard_4k` angular density of
  75.402449345163049 PPD at every input size. Any conversion drift remains in
  the numerical report.
- SSIMULACRA 2 and Butteraugli use a separate spatial representation. The
  clipped source RGB is compressed into `[0.02, 0.98]`, encoded as limited-range
  BT.601 8-bit YUV444 and decoded again from those exact quantized bytes. The
  CPU decode is unclipped and remains safely within `[0, 1]`. The CLI receives
  this spatial Y4M with `--input-transfer srgb`; each pinned original receives
  the decoded FP32 sRGB-encoded values through a PFM image for every frame.
  Both arms pool final per-frame scores by arithmetic mean.
  The original spatial harness serializes frame scores to eight decimal places
  before pooling, contributing at most `5e-9` score units of decimal rounding
  per frame.

The source clips are shared across metrics, while spatial gamut conditioning
changes their metric inputs deliberately. The reports bind the spatial paths
and hashes separately from the temporal representation. Agreement within an
arm pair does not imply that every metric consumed identical pixels, or that
independent CPU and GPU RGB conversions were bit-identical.

## Run five-metric correlation and numerical diagnostics

```sh
.venv/bin/python -m validation.temporal.run --profile smoke
.venv/bin/python -m validation.temporal.run --profile qualification
```

The smoke corpus uses three originals, 13 distortion variants and 12 frames:
39 complete clip pairs per metric, or 195 metric/clip comparisons for the full
five-metric selection. Qualification uses all 30 originals and 40 frames:
390 pairs per metric, or 1,950 metric/clip comparisons. These describe the
input matrix; measured gate outcomes are recorded separately. Quantization,
weak/strong brightness flicker, frame freezing and isolated final-frame errors
cover several severities.
Weak flicker alone was uninformative under the original VMAF model; stronger
levels were added to obtain eligible comparisons without widening thresholds.
Stronger final-frame errors similarly supply original-score separation where
VMAF v1 clips its weaker impulse scores near the upper score bound.
Direct FIR and API tests separately cover impulses at startup and ring-wrap
positions, exact identity, frame association, buffer reuse and reset isolation.

Normal runs consume checked-in independent original scores and frozen policies
under `validation/temporal/oracles/`; they do not need the original Cloudinary,
libjxl or libvmaf executables, or PyTorch.
Hashes bind the entire prepared input manifest and resolved configurations.
Original-only margins are 1% of each configuration's original-score 5th-to-95th
percentile span with a positive numerical floor, frozen before candidate runs.
Incomplete subsets cannot pass the full qualification policy.

Raw reports go under `artifacts/reports/temporal-<profile>/`; generated media,
reference caches and RGB/YUV representations go under `.cache/validation/temporal/`.
Full preparation can use tens of GB. `--offline` forbids source downloads;
`--cache-dir`, `--download-dir` and `--out-dir` can choose other locations.
The aggregate `summary.json` keeps every selected metric visible alongside
its direct Spearman/Kendall result, ordering coverage and numerical outcome.
Per-metric canonical reports and agreement evidence retain the underlying data.

```sh
.venv/bin/python -m validation.temporal.run --profile qualification --offline \
  --iterations 5 --warmup 1 \
  --out-dir artifacts/reports/temporal-local
```

Ranking is the default for the registered smoke and qualification
original/policy pairs after their paired old/new audit. It selects direct
agreement and any required subjective gate while raw numerical errors remain
diagnostic. The [qualified-domain registry](../validation/qualified_domains.json)
binds each original score/input/configuration identity and frozen policy.
Unregistered or modified pairs default to `both`. Use that mode explicitly
to require the old/new conjunction on qualified inputs:

```sh
.venv/bin/python -m validation.temporal.run --profile qualification --gate-mode both
```

A numerical failure does not turn a direct ranking pass into an identical-score
claim. The conjunction fails whenever either requested gate fails. `numerical`
runs the old endpoint alone. These modes do not disable API, identity,
synchronization or numerical component tests.

The [2026-11-06 transition record](design/ranking-default-gates.md)
documents the qualified domains, default-resolution checks and retained
contracts. Historical result files keep the acceptance modes used when they
were measured. New configurations or input domains require their own
replacement agreement evidence before their numerical acceptance is retired.

## Regenerate independent originals

Fetch/build only the needed pinned references and follow
[CVVDP's CPU environment instructions](cvvdp-development.md) when regenerating
that original. The following workflow never edits `.reference/`:

```sh
python3 tools/fetch_references.py --only ssimulacra2,libjxl,vmaf,ColorVideoVDP
python3 tools/build_references.py --only cloudinary,libjxl,vmaf,cvvdp --jobs 8
.venv/bin/python -m validation.temporal.run \
  --profile qualification --reference generate --reference-only --offline \
  --out-dir artifacts/reports/temporal-original-regeneration
```

The optional original runtime evaluates Cloudinary SSIMULACRA 2 and libjxl
Butteraugli through their public image tools, libvmaf through its public video
CLI, and original CVVDP on CPU FP32 with one thread per process. Spatial image
scores are collected for every frame before arithmetic-mean pooling; VMAF
retains its ordered frame metrics and CVVDP its whole-clip latent pooling.
Up to four independent clips run concurrently. Input hashes and
model/executable/runtime fingerprints are retained. Reference caches verify
their own content digests; modified or incomplete results fail. Regeneration
writes reports for review and never overwrites checked-in oracles implicitly.

Challenge each frozen policy using conceptual output mutations:

```sh
.venv/bin/python -m validation.temporal.audit \
  --reference validation/temporal/oracles/qualification/vmaf-v0.json \
  --policy validation/temporal/oracles/qualification/vmaf-v0-policy.json \
  --output artifacts/reports/temporal-audit.json
```

The audit accepts identity, `+3`, positive scaling and increasing nonlinear
remapping, and rejects reversed order, collapsed scores and rotated sample
association. These recorded-score mutations supplement real GPU temporal
unit tests; they do not simulate every possible history defect.

Qualification covers this fixed SDR source/format/display/rate domain. It does
not establish a universal unseen-content error rate or subjective video
accuracy: these source clips have no MOS. HDR, other model/display settings,
high bit depth and extreme frame rates retain their existing independent
numerical tests until corresponding agreement corpora are qualified.
