# Ranking and temporal validation results

2026-11-06. The normal CLI now consumes complete progressive Y4M clips for
all five metrics. VMAF v0, VMAF v1 and CVVDP pass both the old numerical and new
ranking gates on the final hosted video corpus: **390 clip pairs per metric,
30 source groups, 13 variants and 40 frames per clip**. The source-balanced
still-image qualification passes ranking and paired subjective noninferiority
for SSIMULACRA 2 and Butteraugli while retaining their observed strict numerical
failures. No metric arithmetic or acceptance thresholds were widened.

## Question and method

The requested contract is preservation of the original metric's relative
quality ordering, allowing a common score offset or increasing recalibration.
Direct candidate/original Spearman and Kendall statistics therefore supplement
explicit within-source ordering decisions. Similar correlations with MOS are
not a substitute for direct agreement. Raw errors remain separately visible.
The [design record](../../../docs/design/ranking-and-temporal-validation-plan.md) describes the
proposed limits and their status as engineering choices.

The final policy requires SRCC >= 0.999, Kendall tau-b >= 0.99, at most 0.5%
source-balanced clearly ordered failures, at most 1% per distortion family and
at most 5% for any informative source. Required configurations and input strata
also have coverage gates. Exact returned-score ties on clearly ordered original
pairs count as failures. Original-only ambiguity margins use 1% of the original
5th-to-95th percentile span under a fixed rule; complete prepared input and
configuration manifests are pinned before candidate evaluation.

Videos are downloaded automatically from Xiph using exact, checksum-pinned
HTTP byte ranges. The first 40 complete frames of 30 source files total
179,444,008 bytes. Preparation declares square metric geometry without image
rescaling, preserves the declared native frame rate and exactly expands 4:2:0
chroma to YUV444. Both arms use these same canonical samples. Original CVVDP
gets an explicit CPU RGB conversion while the public CLI uses GPU conversion;
this small conversion difference is included in numerical drift.

Independent originals are pinned libvmaf
`4611e47a7330cd342498dcc130e1008efbeab6e4` and original base ColorVideoVDP
v0.5.7 `2a268bce8d56e2f3abde46df3927d8a633707a24`. VMAF uses its statically
linked public CLI, native 8-bit YUV444 and the float v0.6.1/SDR v1 3H models.
Its per-frame JSON has six-digit precision before clip-mean pooling. CVVDP
uses PyTorch 2.15.0+cpu, NumPy 2.5.4, one CPU thread per clip, FP32/highest
matrix precision, Standard4K photometry and replicated temporal padding.
Parallel workers evaluate independent whole clips, never separate frames of
one clip.

The candidate is the release `build/vqm` on NVIDIA GeForce RTX 5070, driver
615.78.08, device UUID `7293d6cd964492010338cb259cfc9cf2`. The working tree
includes pre-existing uncommitted work, so HEAD
`a949b6177efb88890dd44860cc30dff45486a25d` alone does not identify the measured
source. The [compact report](../measurements/ranking-temporal-results.json)
records the actual executable hash, source/policy/oracle hashes, original
runtime fingerprints and every gate outcome. No comparative speed claim is
made.

## Paired video gate results

| Metric | SRCC | Kendall tau-b | Clear within-source failures / eligible pairs | Maximum absolute error | Mean absolute error | Old numerical | New ranking |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| VMAF v0 | 0.999985029954 | 0.999551652641 | 0 / 2002 | 0.005348495826 | 0.000093969867 | PASS | PASS |
| VMAF v1 | 0.999999588278 | 0.999943238257 | 0 / 2030 | 0.006993603399 | 0.000815937619 | PASS | PASS |
| CVVDP | 1.0 | 1.0 | 0 / 2238 | 0.000014042005 | 0.000001645005 | PASS | PASS |

Old maximum-error limits are 0.01 for VMAF v0, 0.35 for VMAF v1 and 0.001 JOD
for CVVDP. Every required rate/configuration, input stratum and distortion
family passed. Each clip contributed one complete pooled score; per-frame
VMAF results remain in raw evidence. The three-metric conjunction exited 0.
All 1,170 paired scores are retained in the
[compact CSV](../measurements/ranking-temporal-scores.csv).

Both profiles passed seven conceptual output mutations: unchanged scores,
`+3`, positive scaling and increasing nonlinear mapping were accepted;
reversal, complete score collapse and rotated sample association were rejected.
The six profile/metric audit reports are retained under the raw evidence and
qualification audits are embedded in the compact report. These are gate
sensitivity tests on recorded scores, supported by independent real GPU
temporal tests rather than presented as production fault injection.

## Coverage changes and rejected shortcuts

The first three-source smoke matrix used two weak brightness-flicker levels.
Original VMAF scores barely separated them, making that family uninformative.
It failed the new coverage gate despite perfect direct rank agreement and
passing numerical checks. Two stronger flicker levels were added; limits were
not changed.

The intermediate 330-clip qualification then exposed insufficient VMAF v1
impulse separation: only 5 of 90 family comparisons exceeded the original-only
ambiguity margin, below the required 10% eligible fraction. Numerical parity
and correlation coefficients passed, but overall ranking qualification correctly
failed coverage. Two stronger final-frame errors increased the final matrix to
390 pairs. Its original-only self-audit established coverage before candidate
qualification. Final margins were frozen from those originals using the same
rule. Earlier input matrices, policies, failed reports and commands remain
under their original raw paths; they were not overwritten to imply a pass.

A long CLI run also produced every result and then intermittently crashed
during NVIDIA driver exit handling. Its core dump identified `vkDestroyDevice`
from the fallback context exit handler. The CLI now releases the retained
Vulkan context after all automatic pipelines have been destroyed, matching the
existing benchmark/test lifecycle. The full final candidate processes returned
zero. A concurrent rebuild was separately caught by the candidate-binary hash
guard and was rejected rather than accepted as qualification evidence.

Rejected approaches were inferring original agreement from equal MOS
correlations, ignoring ties/constant groups, allowing a large easy domain to
hide a failed family, widening thresholds to repair coverage and accepting
complete stdout from a process that exited unsuccessfully.

## Still image agreement and human ratings

The [frozen CID22 fixture](../../../validation/fixtures/cid22-balanced/README.md)
uses 180 pairs from 30 original files, two codecs and three quality levels per
codec. All 450 eligible within-source comparisons agree for both metrics.

| Metric | SRCC | Kendall tau-b | One-sided 95% lower bound on MOS delta SRCC | Strict numerical |
| --- | ---: | ---: | ---: | --- |
| SSIMULACRA 2 | 0.9995514265 | 0.9905648665 | -0.004377140338 | FAIL |
| Butteraugli | 0.9999814809 | 0.9990068281 | -0.000751150057 | FAIL |

Both subjective bounds pass the predeclared -0.005 margin using 10,000 paired
whole-source resamples, seed zero, with no undefined resamples. Both previous
MCOS checks also pass. Strict max/mean thresholds of 0.01/0.001 fail: SSIMULACRA
2 maximum/mean error is 2.02056257/0.3493186512; Butteraugli is
0.03281712/0.0012097629. These failures remain recorded. `both` correctly exits
1, and explicit ranking-only reproduction exits 0 with all numerical
diagnostics retained. Policies now pin the exact input manifest and source
coverage. The [still result summary](../measurements/ranking-still-results.json)
contains complete provenance and subjective evidence.

## Correctness and build checks

- Original release GPU harness: 227 tests, zero failures.
- Expanded release GPU harness: 240 tests, zero failures, GPU required.
- Release CTest: all 21 suites pass, including new gate, download, temporal
  workflow and public CLI tests.
- Hardware Vulkan synchronization validation: all 240 tests pass with no VUID
  or synchronization-hazard messages.
- Fresh Debug ASan/UBSan build: all 240 core tests and seven CLI video tests
  pass with no sanitizer diagnostics; leak detection disabled as prescribed.
- Recursive quality checks and clang-tidy pass for 48 compiled production C++
  files. New Python files pass Ruff and YAPF; affected legacy adapters retain
  133 pre-existing Ruff findings versus 160 before, with no new rule/file
  finding counts. Formatting checks pass for all affected Python.
- A fresh CLI-disabled, benchmark-disabled core configure/build passes without
  downloaded videos, original checkouts or historical artifacts as build inputs.

New GPU tests directly verify four-channel CVVDP impulse/step/ramp FIR output,
fractional coefficients, replicated startup, repeated ring wrap, exact changing
identity, reset and resident reuse. VMAF tests cover public short boundaries,
uneven reference transitions, per-frame spatial association, host/borrowed
buffer overwrite, reset with pending history and independent pipelines. CLI
tests exercise all five metrics and reject late malformed or unequal inputs
without partial clip output.

## Reproduction and retained numerical scope

Follow [the workflow guide](../../../docs/temporal-validation.md). The exact
final video command was:

```sh
python3 -m validation.temporal.run --profile qualification --offline \
  --out-dir artifacts/reports/ranking-temporal/qualification-final
```

Its default is `both`. Ordinary runs use checked-in originals and need neither
libvmaf nor PyTorch. Independent regeneration is optional and writes separate
reports for review. Source downloads, generated media and reference caches are
ignored; compact score/policy/provenance evidence is maintained source.

Numerical checks have not been globally retired. Explicit ranking-only corpus
acceptance is available after the paired audit, while numerical component
contracts and unsupported replacement domains remain hard. Existing reference
fixtures additionally cover HDR, high bit depth, alternate displays/models,
extreme frame rates and odd geometries. The hosted SDR result cannot justify
disabling those checks or whole mixed-purpose tests. A future calibration
change must identify its qualified domain and update its public score semantics.

This is a fixed-corpus agreement result with no subjective video ratings or
unseen-source error-rate claim. Thirty source files are not proof of 30
independent underlying capture conditions. The native-rate SD corpus also
does not replace high-resolution performance or unsupported-domain qualification.
