# Frozen CID22 still ranking qualification

This fixture records 180 image pairs from 30 original CID22 files. On
2026-11-06, both ranking gates and the previous MCOS shortfall checks passed;
both strict numerical gates failed. The then-default `both` mode therefore failed.
No implementation arithmetic or thresholds were changed to obtain these results.

Following that paired audit, ranking became the default for these registered
original/policy identities on 2026-11-06. It includes the required subjective
noninferiority check and retains raw numerical errors as diagnostics.
`--gate-mode both` still requests the strict conjunction; unregistered or
modified policies keep that combined default. Identity and numerical component
tests remain required. [Qualified score gates](../../README.md#qualified-score-gates)

The selection was fixed before candidate scoring: take the lexicographically
first 30 original directories under `compressed/`, then select `cld_avif`
`q30.png`, `q60.png`, `q90.png` and `mozjpeg` `q30.jpg`, `q60.jpg`, `q90.jpg`
for each original. This uses filename quality settings without looking at
candidate scores or filtering on MOS. Each codec has three variants per source.
The complete source IDs, original source CSV SHA-256 and historical cache path
are retained in [selection.json](selection.json). The authoritative annotation
SHA-256 is:

```text
fbaf42f5031b0c520209e9739a352d3048d711c85c20552679304e9784af0cc6
```

These policies apply to this exact source-balanced subset. They are not default
policies for an arbitrary CID22 prefix or for the whole corpus. The metric
configuration IDs are `ssimulacra2:cid22` and `butteraugli:cid22`; both use stratum
`cid22` and codec-directory distortion families. Original-score ambiguity
margins are 0.1 SSIMULACRA 2 points and 0.01 Butteraugli distance, fixed before
candidate execution. Global limits are SRCC >= 0.999 and Kendall tau-b >= 0.99;
meaningful within-source macro error <= 0.005, codec-family error <= 0.01 and
worst-source error <= 0.05. Every policy requires all 30 source groups and the
`cid22` stratum requires 180 samples. The promoted policies additionally require
the separately planned human-label noninferiority gate: paired whole-source
bootstrap of the candidate-minus-original SRCC, margin 0.005, at least 30 source
groups, 10,000 resamples and seed 0. No frame-level resampling is involved.
Each policy also pins its canonical `input_manifest_hash` and explicitly requires
`min_total_sources = 30` and `min_total_samples = 180`. Changing pair membership,
raw bytes or selected MOS labels cannot silently reuse this policy. The
configuration hash is not pinned because the experiment configuration records
the actual candidate executable fingerprint, which can change after rebuilding.

## Reproduction

From the repository root, fetch and build the pinned originals if needed:

```sh
python3 tools/fetch_references.py --only cid22,ssimulacra2,libjxl
python3 tools/build_references.py --only cloudinary,libjxl --jobs 8
cmake --preset release
cmake --build --preset release --parallel 8
python3 validation/fixtures/cid22-balanced/prepare.py --check
python3 validation/fixtures/cid22-balanced/prepare.py
```

`prepare.py` checks the full source annotation hash, subset membership and both
raw input hashes against the retained original reports. It then recreates a
symlink-only corpus under `.cache/validation/cid22-balanced`; the original
corpus remains untouched. Missing or conflicting files fail without replacing
them. No binaries or source media are stored in this fixture.

The following commands explicitly run both the strict conjunction and ranking
acceptance. Omit `--gate-mode "$mode"` from either metric command to use its
qualified ranking default. In the retained measurement, each `both` command
exited 1 and each ranking result passed. All raw numerical and MCOS diagnostics
remain in the reports in either mode.

```sh
for mode in both ranking; do
  python3 validation/run_corpus.py \
    --data-root .cache/validation/cid22-balanced \
    --csv validation/fixtures/cid22-balanced/subset-mcos.csv \
    --metric ssimulacra2 --only ours,cloudinary \
    --agreement-policy validation/fixtures/cid22-balanced/ssimulacra2-policy.json \
    --gate-mode "$mode" --accuracy-tolerance 0.005 --strict-parity \
    --max-threshold 0.01 --mean-threshold 0.001 \
    --out-dir "artifacts/reports/ranking-stills-reproduction/$mode/ssimulacra2"
  python3 validation/run_corpus.py \
    --data-root .cache/validation/cid22-balanced \
    --csv validation/fixtures/cid22-balanced/subset-mcos.csv \
    --metric butteraugli --only ours,libjxl \
    --agreement-policy validation/fixtures/cid22-balanced/butteraugli-policy.json \
    --gate-mode "$mode" --accuracy-tolerance 0.005 --strict-parity \
    --max-threshold 0.01 --mean-threshold 0.001 \
    --out-dir "artifacts/reports/ranking-stills-reproduction/$mode/butteraugli"
done
```

Each run verifies the selected reference checkout against
`tools/references.lock`, fingerprints the actual files and executables before
scoring, and rechecks those fingerprints afterward. It writes complete
canonical original/candidate reports, their configuration and input manifests,
and a separate agreement result. The retained original reports here describe
the measured run; reproduction evaluates the pinned original again rather than
silently relabeling historical configuration hashes for a newly built candidate.

## Retained result

| Metric | SRCC | Kendall tau-b | Within-source errors / eligible pairs | Maximum absolute error | Mean absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| SSIMULACRA 2 | 0.9995514265 | 0.9905648665 | 0 / 450 | 2.02056257 | 0.3493186512 |
| Butteraugli | 0.9999814809 | 0.9990068281 | 0 / 450 | 0.03281712 | 0.0012097629 |

All within-source comparisons were eligible; none were excluded by ambiguity
margins and none collapsed to candidate ties. Each codec family had zero errors
in 90 eligible within-source comparisons. Global comparisons include different
original content: SSIMULACRA 2 reversed 76 of 16,110 global pairs, Butteraugli
reversed 8. Passing the chosen correlation limits does not mean an identical
global ordering.

Both legacy MCOS checks passed with a 0.005 shortfall tolerance. SSIMULACRA 2's
KRCC shortfall was 0.00496586, close to that frozen limit. Both numerical checks
failed the existing maximum 0.01 and mean 0.001 limits. These observed failures
remain recorded; they were not replaced with wider limits.

The planned subjective gate was evaluated separately from the retained paired
reports. Both gates passed with zero undefined bootstrap samples. The observed
SRCC deltas were -0.00236015515 for SSIMULACRA 2 and -0.000131691307 for
Butteraugli; their one-sided 95% lower bounds were -0.00437714034 and
-0.000751150057, both above -0.005. This uses 30 original-file groups and assumes
independence between those groups. It does not establish that related CID22
content is independent.

The candidate was `build/vqm_bench` SHA-256
`155eda72e223cdb87b08a300f99e9d6959c7228c2c28b75580854b6adff87ee0`.
Original pins and each measured executable hash are retained in the two
`*-configuration.json` files. The original measurement report is in Git at
`26bdffe:bench/notes/measurements/ranking-still-results.json`.

Raw logs and original historical commands remain under
`artifacts/reports/ranking-temporal/stills/`, using their original
`.cache/validation/ranking-temporal-stills-20261106` input path. This fixture is a
promotion of compact evidence; those historical logs were not rewritten.
The initial policies and CLI runs omitted bootstrap; the additional planned
subjective phase is retained in `stills/subjective/`. The promoted policies
require that passing gate. Both historical and promoted policy hashes appear
in the compact measurement report.
The final input-pinned policies were then exercised through real corpus runs
against the freshly prepared cache. Both ranking-mode processes exited 0,
including required subjective noninferiority; their input-manifest hashes
matched the historical reports exactly. This phase lives in `stills/input-pinned/`,
which also preserves the preceding subjective policies. No historical log was
rewritten.

This is a fixed-corpus still-image result covering two codecs and three quality
levels. Thirty original files do not establish independence for all related
CID22 content, an unseen-source error-rate bound or temporal video correctness.
