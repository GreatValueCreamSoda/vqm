# Reference correlation for the five established metrics

2026-11-06. This extends the hosted-video reference workflow to SSIMULACRA 2
and Butteraugli alongside VMAF v0, VMAF v1 and CVVDP. All five use the normal
`vqm` CLI for candidate evaluation. Independent original scores and policies
are maintained for both smoke and qualification profiles. PSNR-HVS was being
developed concurrently and is outside this qualification, as requested.
All five pass ranking on 390 clip pairs each. Numerical gates also pass for
four metrics; SSIMULACRA 2 retains its measured numerical failure.

## Question and reproducible method

Does each implementation preserve its original metric's ordering, including
within-source comparisons of clearly separated quality levels? A uniform
score offset such as `+3` is acceptable under the ranking contract. The
[policy design](../../../docs/design/ranking-and-temporal-validation-plan.md) sets engineering
limits: SRCC at least 0.999, Kendall tau-b at least 0.99, source-balanced clear
ordering error at most 0.5%, per-family error at most 1%, and worst informative
source error at most 5%. Coverage is required separately. These are explicit
acceptance choices for this fixed corpus, not a universal definition of
perceptual fidelity. Pearson correlation and raw numerical errors remain
diagnostics; Pearson alone cannot establish ordering preservation.

The source catalogue pins the first 40 complete frames of 30 Xiph-hosted SD
clips, 179,444,008 downloaded bytes. Every source has 13 deterministic
distortions, giving 390 complete clip pairs per metric and 1,950 comparisons
in the five-metric qualification. Smoke uses three sources and 12 frames,
39 pairs per metric. Preparation preserves declared rational frame rates.
Quantization, brightness flicker, freezes and isolated final-frame errors
provide several severities. This corpus has no human video quality ratings.

VMAF and CVVDP retain their existing native YUV444 representation, original
scores, policy hashes and input/configuration identities. Their historical
results remain in the [three-metric report](ranking-temporal-validation-results.md).
Spatial metrics use a separately pinned sRGB representation: clipped source
RGB is compressed into `[0.02, 0.98]`, quantized to limited-range BT.601
YUV444 8-bit, and decoded from those exact bytes. Gamut headroom makes
reference clipping a no-op. The CLI scores the spatial Y4M with explicit
`--input-transfer srgb`; originals receive the decoded encoded FP32 RGB in
bottom-up, interleaved, little-endian PFM files. Both pinned original readers
default these PFM samples to encoded sRGB. Spatial pixels deliberately differ
from the native temporal representation; each arm pair has matching hashes
and settings, rather than claiming identical pixels across metric families.

Independent originals are Cloudinary SSIMULACRA 2 at
`81feacf6a5b0bbd7ec11b4a881a4615033ab0b10`, libjxl Butteraugli at
`e47d2a2b5bb418f2947ce07f36cb048861db3820`, Netflix libvmaf at
`4611e47a7330cd342498dcc130e1008efbeab6e4` and base ColorVideoVDP v0.5.7 at
`2a268bce8d56e2f3abde46df3927d8a633707a24`. Spatial adapters call the existing
original in-process batch harnesses with one scoring/decode thread, one
iteration and no warmup. They pool final frame scores by arithmetic mean;
the harness serializes each original frame score to eight decimal places
before pooling. Its rounding contribution is at most approximately 5e-9
per frame. Butteraugli uses hf asymmetry 1, intensity target 80 and xmul 1.
VMAF pools final frame scores by mean; CVVDP retains whole-clip JOD.

Original-only ambiguity margins use the frozen rule of 1% of the original
5th-to-95th-percentile score span. Policies were saved before candidate
evaluation. No ranking or numerical limit was widened after seeing results.
The spatial raw-error limits are the strict 0.01 maximum / 0.001 mean
point-score checks also used by the still-image transition; the existing
temporal maximum limits remain VMAF v0 0.01, VMAF v1 0.35 and CVVDP 0.001.

## Reproduction

Build the regular release CLI, then run all five with automatic downloads:

```sh
python3 -m validation.temporal.run --profile qualification --gate-mode ranking
```

`--profile smoke` selects the smaller matrix. Omit `--gate-mode ranking` to
require both numerical and ranking gates. Every mode retains both diagnostic
results. Original regeneration, reference build requirements, colour
contracts and cache locations are described in the
[workflow guide](../../../docs/temporal-validation.md). Ordinary runs use
checked-in independent originals and require no original runtimes.

## Paired qualification results

All five pass the frozen ranking policy. Every required configuration, input
stratum and distortion family passes its coverage and agreement checks.

| Metric | SRCC | Kendall tau-b | Pearson | Clear within-source failures / eligible pairs | Old numerical | New ranking |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| SSIMULACRA 2 | 0.999971273771 | 0.997495221146 | 0.999973684280 | 0 / 2245 | FAIL | PASS |
| Butteraugli | 1.0 | 1.0 | 0.999999999988 | 0 / 2214 | PASS | PASS |
| VMAF v0 | 0.999985029954 | 0.999551652641 | 0.999999999897 | 0 / 2002 | PASS | PASS |
| VMAF v1 | 0.999999588278 | 0.999943238257 | 0.999999998292 | 0 / 2030 | PASS | PASS |
| CVVDP | 1.0 | 1.0 | 1.0 rounded | 0 / 2238 | PASS | PASS |

These within-source decisions exclude differences below the frozen
original-only ambiguity margin. They are distinct from the global rank
statistics: SSIMULACRA 2 has 95 reversed global pairs, including comparisons
across different source content. Its global coefficients still pass the
predeclared limits. No clearly separated within-source decision is reversed
or collapsed for any metric, and all source-balanced errors are zero.

| Metric | Maximum absolute error | Mean absolute error |
| --- | ---: | ---: |
| SSIMULACRA 2 | 5.262514484767 | 0.295218339595 |
| Butteraugli | 0.001781654205 | 0.000056372201 |
| VMAF v0 | 0.005348495826 | 0.000093969867 |
| VMAF v1 | 0.006993603399 | 0.000815937619 |
| CVVDP | 0.000014042005 | 0.000001645005 |

The combined `both` run exits 1 because SSIMULACRA 2 fails its maximum and
mean numerical limits. Standalone `validation.agreement --require ranking`
re-evaluation of each of those same saved original/candidate/policy triples
exits 0. The normal five-metric smoke command with `--gate-mode ranking`
also exits 0 while retaining the numerical failure in its output. Numerical
thresholds remain intact, including the default combined exit behavior.

The [compact report](../measurements/all-metric-reference-correlation.json)
records coefficients, numerical diagnostics, coverage statuses, reference
provenance, oracle/policy hashes, the combined summary and ranking recheck
commands. All 1,950 paired scores and their input hashes are retained in the
[score CSV](../measurements/all-metric-reference-correlation-scores.csv).
Raw candidate frame results, timings and logs remain under the stated raw
report directory. Generation provenance retains its original hashes; the
compact report separately records the spatial dispatcher's generation hash,
which the initial spatial provenance did not retain separately from its
adapter hash. Future generation records both hashes.

Both profiles pass all seven conceptual score mutations for all five metrics
(70 audit cases): identity, `+3`, positive scaling and an increasing nonlinear
mapping pass; reversal, complete collapse and rotated sample association
fail. These test gate sensitivity on saved original scores, rather than
claiming production GPU fault injection. Qualification audits are embedded
in the compact report.

The qualification run recorded here used both gates:

```sh
python3 -m validation.temporal.run --profile qualification --gate-mode both \
  --offline --out-dir artifacts/reports/all-metric-correlation/qualification-all-both
```

The candidate is the release CLI on NVIDIA GeForce RTX 5070, driver
615.78.08, UUID `7293d6cd964492010338cb259cfc9cf2`. Its executable SHA-256 is
`ec5c9eeefbc55f2b2c168d66afc19f887c0ed9de32e4d3e93b292d3bc8f8d541`.
The working tree contains substantial pre-existing and concurrent changes;
HEAD alone does not identify this binary. No comparative speed claim is made.

## Validation checks

All 23 release CTest suites pass. The release core harness with
`--require-gpu` passes all 240 tests with zero failed checks. The 19 new
all-metric workflow tests and 10 spatial-reference tests pass, including
real-process identity/nonidentity calls through both pinned spatial originals.
New and changed Python passes Ruff and YAPF; the recursive repository quality
check passes.

Regression tests reject a self-consistent oracle/policy pair with an incorrect
original revision, wrong or missing candidate metric labels, incomplete
spatial hashes, changed source bytes, truncated original frames and failed
original processes. Source/model/configuration identities remain bound to
their own metric and representation. Candidate batches retain separate
per-metric pair lists so spatial and native paths remain reviewable.

## Limits and retained contracts

This qualification covers the stated SDR inputs and default metric/model
configurations. It does not replace HDR, high-bit-depth, alternate-display or
alternate-model numerical oracles. Whole-clip file tests supplement the
independent core/API temporal tests. Numerical component contracts and exact
identity invariants remain enforced; numerical checks have not been globally
retired. A passing fixed-corpus correlation gate is not a guarantee about
unseen content or subjective video quality.
