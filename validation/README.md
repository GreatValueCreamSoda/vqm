# Validation

## Reference correlation for the five established metrics

Reference faithfulness is measured directly against the pinned original scores.
The ranking gates require Spearman SRCC >= 0.999, Kendall tau-b >= 0.99 and at most
0.5% clearly ordered within-source failures with equal weight per source.
Required input strata, distortion families, minimum coverage and a 5% worst-source
cap prevent an easy majority from hiding a failed domain. Candidate ties on
clearly ordered original pairs count as failures. Ambiguity margins and input
hashes are frozen before candidate scoring.

The [hosted video workflow](temporal/README.md) downloads checksum-pinned
Xiph source prefixes, prepares the declared inputs and compares the normal
`vqm` CLI with independent originals for the five established metrics:
SSIMULACRA 2, Butteraugli, VMAF v0, VMAF v1 and CVVDP. After the
[local build and Python setup](../CONTRIBUTING.md#build-and-test), run the full
reference-correlation matrix:

```sh
.venv/bin/python -m validation.temporal.run --profile qualification
```

Qualification uses 390 complete clip pairs per metric from 30 source groups;
`--profile smoke` uses 39 pairs per metric from three groups. The runner's
`summary.json` reports each selected metric's direct correlation, ordering
coverage, numerical diagnostics and gate outcome. Its metric list and actual
outcomes describe the completed run. The earlier 390-pair VMAF/CVVDP reports
remain evidence for their original three-metric matrix.

These five metrics support progressive Y4M, JSON clip/frame results and batch
input. Spatial metrics pool independently scored frames by arithmetic mean;
VMAF and CVVDP retain their sequence contracts. The spatial and temporal
qualification paths share source clips but use separate, explicitly hashed
representations. The guide describes their color conversion and transfer
declarations. Direct API tests independently cover temporal filtering,
startup/tail rules, history ownership and reset behavior.

## Qualified score gates

Ranking is the default for the registered hosted SDR profiles
and frozen source-balanced CID22 policies. The
[qualified-domain registry](qualified_domains.json) binds the
original scores, input metadata, reference pin and frozen policy identity.
Temporal configuration hashes are also pinned; CID22's experiment configuration
can include a newly built candidate fingerprint while its original sample
scores and metadata remain fixed. Unregistered or modified original/policy
pairs keep the combined `both` default. Raw
numerical drift remains in every report. `--gate-mode both` explicitly requires
both acceptance endpoints; `numerical` selects the old score-error endpoint.
A ranking pass can coexist with a numerical failure and a failing conjunction.
The [frozen source-balanced CID22 fixture](fixtures/cid22-balanced/README.md)
provides separate still-image SSIMULACRA 2 and Butteraugli policies and
reproduction commands. Its required subjective gate checks the one-sided 95%
lower bound of candidate-minus-original SRCC against MOS is at least -0.005,
using 10,000 paired whole-source resamples.

`validation/agreement.py` evaluates canonical JSON/CSV reports independently
of scoring. It rejects mismatched sample sets, provenance, inputs, undeclared
domains and nonfinite values. A missing required reference or insufficient
coverage cannot pass. Constant groups have undefined correlation; they are
not assigned a perfect coefficient. Missing or undefined required subjective
evidence is inconclusive and fails qualification.

```sh
.venv/bin/python -m validation.agreement original.json candidate.json \
  --policy frozen-policy.json --require both --output agreement.json
```

The standalone evaluator resolves ranking by default for registered qualified
original/policy pairs and `both` for other pairs. Its `--require` option
explicitly selects either endpoint or their conjunction. This transition
preserves identity anchors and numerical component tests; it changes final
corpus-score acceptance only. The registry covers ten hosted metric/profile
combinations and two balanced still profiles. It matches canonical report
and policy identities independently of JSON key order or file location.

Spearman and Kendall preserve a common increasing score recalibration. Maximum
error, MAE, RMSE, Pearson correlation, offset and affine slope remain visible
diagnostics. Ranking agreement does not preserve the original score anchors,
absolute thresholds or distance interpretation; document a changed calibration
as part of its public contract. Retain numerical unit oracles for unchanged
feature/filter/model arithmetic and domains not qualified by a replacement
agreement corpus. No whole test should be skipped to remove one score pin.

The five-metric qualification passed ranking on all 390 pairs per metric.
SSIMULACRA 2 exceeded the numerical error limits, while the other four
metrics passed them. The run's full evidence is available in Git at
`26bdffe:bench/notes/system/all-metric-reference-correlation.md`.

## Corpus

The CID22 validation set: 49 originals, 4292 original/distorted pairs, all
512x512, distributed under CC BY-SA 4.0 by Cloudinary. Every distorted image
is stored as PNG for the AVIF, HEIC, JPEG 2000, WebP, JPEG XL and AV1 encodes,
or as JPEG for mozjpeg. The implementations read the same input files.
JPEG decoding and embedded ICC handling can still differ between tools.
8 codec families: `aom`, `vis_avif`, `cld_avif`, `cld_heic`, `cld_jp2`,
`cld_webp`, `libjxl`, `mozjpeg`.

The MCOS CSV has `reference_img,distorted_img,encoder,setting,bpp,MCOS,RMOS,
Elo,nb_pc_opinions`; `validation/common.py:enumerate_pairs` walks every codec
and setting, so a run covers all encodes, not a JPEG subset. Fetch the corpus
and pinned reference checkouts with

```sh
python3 tools/fetch_references.py
```

The default includes Cloudinary's official 1.4 GiB CID22 archive. Use
`--only cid22` to fetch only that corpus, or `--only ssimulacra2` for only the
Cloudinary implementation. `--only` accepts comma-separated names; adding
`--check` verifies the selected existing inputs without downloading. The
corpus can also be fetched directly with `python3 validation/fetch_cid22.py`.
The archive is pinned by SHA-256 in that script, using the digest of the archive
we obtained from Cloudinary; Cloudinary does not publish this digest. Setup
verifies the cached archive, extracted images, annotations and license. An
invalid existing archive or dataset is reported for inspection instead of
being overwritten. The archive and extracted files each require about 1.4 GiB.

## Tools

- `validation/references.json` -- per metric, the implementations to run, their
  command, and whether it is a batch harness. `direction` says whether a larger
  score is better, `reference` names the implementation the parity table and
  the accuracy gate compare against.
- `validation/run_corpus.py` -- runs the manifest over the corpus, writes
  `<metric>.csv` (raw scores next to MCOS) and `<metric>-summary.md` (parity
  and correlation tables), and prints the gate lines.
- `validation/common.py` -- pair enumeration, MCOS loading, score parsing and
  the KRCC/SRCC/PCC code the previous single-reference runner used.

**Batch mode.** Every entry drives an in-memory harness: `--pairs-file` names a
tab-separated list, `--score-only` prints exactly one line per pair in input
order (the score, or the literal `ERROR`), and `--iterations 1 --warmup 0
--threads N --decode-threads M` is appended for scoring runs. The thread counts
come from the manifest: the CPU references score with 16 in-process workers and
a 4-thread decode pool, the GPU arms with 3 pipelines and 16 decode threads (16
hardware threads here, 8 cores). A pair is decoded once, on a decode thread
ahead of the workers; nothing is re-decoded for `--iterations`, and `decode_ms`
is not reported in this mode.

Each implementation is **one process**: its harness spreads the pair list over
its own workers, so a run pays device setup once and no per-pair process
startup, and `--jobs` no longer splits a batch entry into slices (the manifest's
`jobs` is 1; `--jobs` still means per-pair invocations for an entry without
`batch`). Inside a process our pipeline and Vship's handler are created once per
geometry per worker -- Vship's API wants one handler per thread, so three
workers hold three handlers -- and the results are emitted in pair order after
the join, which is what keeps the one-line-per-pair contract without a reorder
thread. The manifest still supports per-pair commands (`batch` absent) and a
`score_line` field selecting the first (`butteraugli_main`) or last line.

The run reports a **scoring-time table** (pairs, seconds, pairs/s per
implementation) before the parity and correlation tables, timed around each
implementation's single invocation. Scores do not depend on the thread count:
the same list scored with one worker and with N is byte-identical for every
implementation here.

**Batch contract.** A process that exits non-zero fails every pair of that
implementation; a line count other than the pair count fails every pair with
the count; `ERROR` or an unparsable line fails that pair. Errors name the
implementation, the pair and the message.

## Legacy corpus gate semantics

- `--accuracy-tolerance` compares our KRCC/SRCC/PCC to the metric's reference
  implementation's; `ours` must not be more than the tolerance below it.
- `--strict-parity` additionally requires the cross-implementation max/mean
  bounds. The default run does not, because the sampled-vs-recursive Gaussian
  is a deliberate metric change: SSIMULACRA 2's parity line is expected to
  fail its default thresholds (max 2.9907, mean 0.3120). The pass criterion is
  MOS accuracy, not bit-parity. Strict parity also fails if the reference or a
  comparison implementation was not selected.
- The required lines are `accuracy vs MCOS: PASS` and a final `PASS`; a
  skipped gate prints `accuracy vs MCOS: SKIP (<reason>)` and does not fail.
- Exit status is 0 iff every selected implementation scored every selected
  pair, the accuracy gate passed (or was skipped), and any requested strict
  parity gate passed.
- Non-finite scores are rejected: they cannot produce meaningful correlation
  or parity statistics. Scoring-time pair counts include only successful
  scores and include process startup and decoding in the wall clock.

The runner writes local CSVs and Markdown summaries to
`artifacts/reports/validation/` by default. Use identical effective color
transforms when assessing numerical parity on JPEG pairs.
[Reference differences](../docs/divergence.md)

## Footnotes

- CID22's MCOS was collected for SSIMULACRA 2. Butteraugli is correlated on it
  with the score negated (lower is better), so a larger value always means
  better quality; this is a cross-metric sanity check, not Butteraugli's
  calibration corpus.
- fssimu2's scores differ by design: f16 planes, a 9-tap blur, and its PNG path
  truncates 16-bit input to 8 bits and drops alpha (matching its own CLI).
- Vship's Butteraugli differs from libjxl by about 0.3-1.2% (203-nit default,
  UHF weighting); it is a throughput reference, never the accuracy oracle.
- Vship-fast is measured through its Vulkan backend; its CUDA backend is
  upstream Vship code and is not used.
- Our alpha policy scores both dark and bright backgrounds whenever either
  image has alpha. The Cloudinary and libjxl in-memory comparison drivers use
  the same policy; this also handles transparency on the distorted image
  alone. Vship's C API has no alpha plane, so an image with genuinely
  transparent pixels is reported as a per-pair error rather than scored.
