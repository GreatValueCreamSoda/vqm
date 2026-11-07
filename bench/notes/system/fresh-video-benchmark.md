# Fresh-repository hosted-video benchmark

2026-11-06. This record checks whether one `uv` command can provision the
reference sources, native builds, Python runtime and hosted inputs, then run
the normal CLI benchmark for the five established metrics. It is a setup and
reproduction check with machine-specific timings, not a speedup claim.
The command completed with exit 0: all five setup commands and all 25
metric/round records passed. The final workspace's stricter aggregation
independently accepted those same complete raw records.

## Input state and command

The experiment copied 585 maintained working-tree files into
`/tmp/vqm-video-fresh-vovh5vt8`. The source was the current uncommitted working
tree, including the video workflow and concurrent PSNR-HVS work, rather than
the older Git HEAD alone. HEAD was
`a949b6177efb88890dd44860cc30dff45486a25d`. The snapshot initially contained
none of `build/`, `.venv/`, `.cache/`, `.reference/`, `validation/download/`
or `validation/data/`.

The exact command, from that snapshot's root, was:

```sh
uv run --no-project --python 3.12 bench/benchmark.py \
  --out-dir artifacts/reports/fresh-video-benchmark
```

Only the output location overrides the default workload. This is a fresh
repository setup on an existing development machine: system build packages,
the Vulkan driver, `uv` itself and its global interpreter/package caches may
already be present. It does not claim a clean operating-system installation
or that every Python wheel was downloaded again. The prerequisites are listed
in [the benchmark guide](../../README.md).

## Provisioning and scoring protocol

The command fetched the pinned Cloudinary SSIMULACRA 2, libjxl, Netflix VMAF
and ColorVideoVDP checkouts and initialized the required libjxl submodules.
It verified checkout cleanliness, built the native reference copies under
`build/`, provisioned `.venv/` with the locked CPU-only ColorVideoVDP runtime
and validated that original runtime. It configured the release build and
built `vqm` and `vqm_bench`; the latter supplies device information only.
The candidate scoring executable is always the normal `vqm` CLI.

The source catalogue provides three smoke prefixes, about 18 MB in total,
downloaded with exact ranges and checksums. Each prefix contains 40 complete
frames; the smoke workload prepares its first 12 frames and 13 deterministic
distortions. That gives 39 complete clip pairs per metric. Spatial metrics
use the separately pinned, gamut-conditioned encoded-sRGB representation;
VMAF and CVVDP retain their native temporal representation. The
[workflow guide](../../../docs/temporal-validation.md) defines these contracts.

The benchmark runs five rounds with 20 timed iterations and three warmups
per complete clip. Each round rotates metric order. Every one of
SSIMULACRA 2, Butteraugli, VMAF v0, VMAF v1 and CVVDP must produce five valid
records, for 25 metric/round records overall. One metric/round contains 468
frames in its selected score vector, 9,360 timed frame evaluations and 1,404
warmup frame evaluations. Across the matrix these are 234,000 timed and
35,100 warmup frame evaluations. They are repeated fixed inputs, not that
many independent clips.

Checked-in independent original scores qualify each arm using the registered
ranking default. The bootstrap builds the originals and verifies the runtime;
it does not regenerate those fixtures as part of timed scoring. Raw numerical
errors remain visible. PSNR-HVS is compiled in this source snapshot but is
outside the selected five-metric matrix and receives no qualification here.

## Timing interpretation

The CLI selects the iteration with median complete-clip core duration from
the 20 timed iterations and reports that iteration's core and decode times.
Core covers metric compute/sequence pushes and finalization. Pipeline
creation/reset, downloads, representation preparation and external reference
generation are outside core time. Decode is measured separately.

For a metric/round, the report retains the sum and median of those 39
selected clip core/decode times. Core seconds per frame is the core sum
divided by **468 selected frames**. It must not divide by 9,360 repeated timed
frames, nor multiply the reciprocal by 20. The published core ms/clip is the
median clip duration for that round; aggregate medians, minimum, maximum and
median absolute deviation are then computed across the five rounds.

Candidate process wall time includes all timed repetitions, all warmups,
decoding and CLI setup. Outer workflow wall time additionally includes input
preparation, hashing and agreement evaluation. Neither is interchangeable
with the sum of the selected median-iteration core times. Bootstrap clone,
build and environment steps have their own logs and wall durations.

Reference adapter/cache durations may include serialization, conversion and
process startup. They are not CPU/GPU compute comparisons and are not used as
reference speed denominators. No baseline arm is present, so this experiment
does not establish a performance improvement. Results apply to the recorded
machine, driver, source snapshot and SD smoke workload; they do not replace
the repository's interleaved optimization protocol or full qualification.

## Measured outcome

The device was NVIDIA GeForce RTX 5070, UUID
`7293d6cd964492010338cb259cfc9cf2`, NVIDIA driver 615.78.08 (raw Vulkan driver
version `2580775424`). The fresh candidate SHA-256 was
`f4a62c28165a18c7e8e662198e7fb092da842e95635d24be3b832599e4d21bd9`.
The build used GCC 16.2.1, CMake 4.4.5, Ninja 1.13.2 and `glslc` 2026.4 on
Linux x86-64. The `uv` runtime was 0.13.6; managed Python was CPython 3.12.14
with PyTorch 2.15.0+cpu and NumPy 2.5.4. Complete tool, package and model
fingerprints are retained in the [compact record](../measurements/fresh-video-benchmark.json).

Each timing below is the median across five rounds. The core range is the
minimum and maximum of the five per-round median clip durations, not an
uncertainty bound or the range across individual clips.

| Metric | Core median ms/clip | Core round range ms/clip | Decode median ms/clip | Candidate process wall median s | Valid rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| SSIMULACRA 2 | 1.840098 | 1.750426–1.890296 | 0.541284 | 2.347272 | 5 / 5 |
| Butteraugli | 3.713813 | 3.591135–3.871330 | 0.644862 | 4.326529 | 5 / 5 |
| VMAF v0 | 1.595202 | 1.473894–2.375141 | 0.653988 | 2.202362 | 5 / 5 |
| VMAF v1 | 3.774640 | 3.581217–6.012132 | 0.883160 | 4.451179 | 5 / 5 |
| CVVDP | 4.743369 | 4.665611–4.856965 | 0.856472 | 5.208520 | 5 / 5 |

All 25 arms resolved the omitted gate option to `ranking` and passed it.
Smoke SRCC and Kendall tau-b were 1.0 for every metric/round. SSIMULACRA 2's
numerical endpoint remained failed, with maximum absolute error about 1.78
and mean about 0.4023; the other four numerical endpoints passed. These
diagnostics were retained rather than widened or suppressed. This repeated
smoke result does not replace the separate
[390-pair qualification](all-metric-reference-correlation.md).

The fresh normal CLI additionally returned exactly 100 for a 12-frame
SSIMULACRA 2 identity clip and exactly 0 for Butteraugli. The original CPU
CVVDP runtime probe returned identity JOD 10. Those checks support preserved
identity contracts without changing any numerical component tests.

Separately, the final workspace passed all 26 release CTest suites, the 15
default-gate tests, 15 reference-bootstrap tests and 21 video-benchmark tests.
The recursive formatting check passed 257 maintained files. New Python
passes Ruff and YAPF; legacy lint debt was retained without new findings.

## Evidence and implementation boundary

The raw run lives at
`/tmp/vqm-video-fresh-vovh5vt8/artifacts/reports/fresh-video-benchmark/`.
Its setup logs, `source-files.json`, working-tree status, per-round raw CLI
outputs and canonical agreement reports remain unmodified. An unmodified
copy is retained under `artifacts/reports/ranking-default-gates/fresh-run/`
in the main workspace, including the original absolute command paths. The parent log
and initial-state metadata are under
`artifacts/reports/ranking-default-gates/fresh-video-benchmark.log` and
`artifacts/reports/ranking-default-gates/fresh-video-snapshot.json` in the
main working tree. The compact result retains their hashes, hardware and
runtime fingerprints, all 25 measurement records and timing distributions.

The exact 585 maintained source files are preserved separately in ignored
`artifacts/snapshots/fresh-video-benchmark-2026-11-06.tar.gz`, SHA-256
`8f435ff0ea1d64acfa86c5f8e4cc668e3455bde00ea906424061ccd812ccfa4e`.
That archive contains no builds, references, downloaded media or environment.
Its metadata is retained as `fresh-run/source-snapshot.json`. The versioned
compact record also embeds every source-file hash; source identity does not
depend on an old HEAD label or a temporary directory remaining available.

The measured snapshot's benchmark source predates subsequent hardening of
failed/incomplete aggregate reporting and additions of full numerical
diagnostics/model-asset hashes to aggregate provenance. Its source hash
manifest identifies that version exactly. The successful scoring and timing
path were unchanged; final validation independently verified all 25 raw
records rather than inferring completion from a permissive aggregate status.
It checked complete 39-clip/12-frame payloads, finite durations, repetition
counts, all report hashes and candidate fingerprints, and recomputed the
timing distributions.

The compact record includes supplemental static hashes of the pinned model
assets taken after measurement. Each compact round copy also includes the
full numerical-diagnostics object from its unchanged raw agreement report,
including max/mean error, offset and Pearson correlation. These explicit
post-run enrichments preserve the initial raw summary and rerun no timed work.

Rejected shortcuts were benchmarking only the first video frame, substituting
cached reference adapter wall time for CPU compute cost, multiplying the
selected-iteration frame rate by its repeat count, accepting missing rounds,
and testing the old clean HEAD while the requested workflow existed only in
the working tree. The observed result establishes that this source snapshot
can provision and run the documented matrix from empty repository-generated
directories under the stated prerequisites.
