# Ranking agreement and temporal validation plan

2026-11-06. Proposed testing policy for SSIMULACRA 2, Butteraugli, VMAF
v0.6.1, VMAF v1 and base ColorVideoVDP v0.5.7. The intended acceptance
criterion is preservation of the original metric's quality ordering. A
consistent score offset is acceptable. Numerical calibration is reported
separately. This plan adds temporal coverage and changes reference agreement
gates; it does not implement or relax the current gates.

Implementation status, 2026-11-06: the shared evaluator, hosted video workflow,
public CLI video path and temporal unit tests are implemented. The
[qualification report](../../bench/notes/system/ranking-temporal-validation-results.md) records paired
old/new results, mutation audits and the retained numerical scope. The original
proposal below remains the policy-design record.

## Meaning of agreement

Compare candidate scores directly with the pinned original on exactly the
same inputs and settings. A candidate equal to `original + 3` preserves rank
agreement. Any strictly increasing transformation also preserves ordering;
Pearson correlation additionally describes linear agreement, so it should be
diagnostic when ordering defines acceptance. These properties follow from
the definitions of [Spearman correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)
and [Pearson correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html).
Clipping after a shift can create ties and must be evaluated on the final
returned scores.

Use the description "ranking agreement with the pinned original on the
qualified input domain". A recalibrated implementation must describe its
score scale separately. Rank agreement alone does not retain the original
absolute thresholds, identity anchors or interpretation of score distances.
For example, a nonlinear remapping of CVVDP cannot inherit the original JOD
distance interpretation merely by preserving order.

Keep perceptual accuracy as a separate question: compare candidate and
original against the same MOS/MCOS observations. Equal correlations with MOS
do not establish that two implementations agree with each other. The existing
`validation/run_corpus.py` primarily gates differences in MCOS correlations;
its max/mean score-error gate is optional through `--strict-parity`.
`validation/cvvdp.py` instead currently gates maximum absolute JOD error.

## Proposed acceptance limits

These are strict initial engineering choices, not published definitions of
metric faithfulness. Audit current baselines and known faulty variants before
freezing a versioned policy. Freeze the limits and corpus before optimizing
candidates; do not select thresholds to admit a preferred result.

| Measurement | Proposed fixed corpus requirement |
| --- | --- |
| Direct candidate versus original Spearman SRCC | At least 0.999 |
| Direct candidate versus original Kendall tau-b | At least 0.99 |
| Clearly ordered comparisons within each source, averaged equally across sources | At most 0.5% order failures |
| Each adequately covered, predefined distortion family | At most 1% of clearly ordered comparisons fail, with equal source weights |
| Each adequately covered required input/configuration stratum | SRCC at least 0.999, tau-b at least 0.99 and source-averaged order failures at most 1% |
| Each source with eligible comparisons | At most 5% order failures; do not hide a catastrophic source in the average |
| Deliberately clear synthetic temporal ordering cases | Zero reversed or collapsed distinctions |
| Required inputs, original results and configuration provenance | Complete and matched; missing or uninformative required data cannot pass |

The first two limits apply separately to each metric/model and qualification
configuration. Report results globally, by source and by predefined input and
distortion strata. A large easy stratum must not compensate for a failed
required domain. Small groups with constant original scores have undefined
rank correlation; use their explicit identity or ordering assertions instead
of substituting a coefficient of one. Define minimum coverage for required
strata in the corpus manifest before qualification, including minimum numbers
of sources and eligible pairs. Missing required coverage blocks qualification.
Sources without eligible pairs do not enter the failure-rate average and do
not satisfy eligible-source coverage. Audit the excluded fraction against the
frozen manifest's coverage requirements.

Kendall has an interpretable connection to reversed comparisons. Without
ties, `tau = 1 - 2 * reversed_pairs / all_pairs`, so 0.99 allows 0.5% reversed
pairs. The simple conversion does not hold for tau-b when ties exist; report
concordant, reversed, original-tied and candidate-tied counts explicitly.
See [the documented tau-b definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html).
Spearman is a useful additional summary, not a substitute for counting actual
decisions that changed.

For each original image or clip, compare its distortion variants. A pair is
clearly ordered when its original score gap exceeds a fixed metric/configuration
margin. Set that margin using original repeatability and a practical reference
gap before examining candidate scores; it is not automatically a perceptual
just-noticeable difference. Pilot margins at 0.5%, 1% and 2% of a frozen
development set's original-score 5th-to-95th percentile span, then select and
freeze one using the input and failure audits. Use a positive numerical floor
from reference precision/repeatability. Do not estimate a new margin from
each candidate or holdout subset.

Count a reversal or an exact tie in final returned candidate scores on an
eligible original pair as an order failure. Report near-tie distances as
diagnostics instead of imposing an arbitrary candidate-scale tolerance that
would invalidate positive-rescaling invariance. Average each source's failure
fraction equally, then report the worst sources and family breakdown.
Publish eligible and excluded pair
counts and score-gap distributions so an excessively broad ambiguity margin
cannot hide failures. Keep raw rank statistics with their ordinary tie rules
alongside these margin-based counts.

Raw maximum error, MAE, RMSE, offset, fitted positive affine calibration,
Pearson correlation, score range and tie/clipping rates remain diagnostics.
They do not veto an otherwise accepted recalibration. Do not fit separate
content-specific mappings before measuring agreement. Do not gate on a
zero-correlation p-value: that tests association, not near-equivalence.

## Subjective accuracy and statistical uncertainty

Where human ratings exist, retain a separate noninferiority check. An initial
margin is a loss of at most 0.005 in SRCC versus the original, with the
one-sided 95% lower confidence bound on the paired difference at least
`-0.005`. This is also a proposed project margin. Report Kendall as supporting
evidence; report Pearson with a fixed, documented score-mapping procedure
when comparing subjective scales. Reference agreement and subjective
accuracy must have distinct result fields and pass/fail lines.
Apply each metric's declared better/worse direction before comparison with
MOS. Fit any subjective-scale mapping on development data and freeze it
before holdout evaluation.

Use a seeded paired bootstrap over original sources, carrying all derivatives
and complete sequences together, to report uncertainty in differences and
agreement statistics. About 10,000 resamples is a starting implementation
budget. Use the fifth percentile of the bootstrap distribution of paired
SRCC differences as the initial one-sided 95% lower bound. Compute pooled
SRCC with equal observation weights for compatibility with existing reports;
keep source-weighted ordering errors as a separate endpoint. Freeze minimum
source coverage in the manifest, with 30 distinct source groups as a pilot
floor rather than a guarantee of precision. Any undefined bootstrap statistic
or insufficient coverage makes subjective qualification inconclusive; do not
silently discard undefined resamples. Do not bootstrap individual video frames
or all pair comparisons as independent observations. The distinction between
total and within-cluster
rank correlation is supported by [Tu, Li and Shepherd](https://arxiv.org/abs/2402.11341);
source grouping and the specific bootstrap design here are engineering
choices for this corpus structure.

Fixed-corpus acceptance and claims about unseen content are different.
Bootstrap intervals can be degenerate when all observed errors are zero;
that does not establish zero risk on unseen content. A 30-to-50-source pilot
is useful for regression coverage, not proof of a universal rare-error bound.
Inconclusive subjective noninferiority requires more independent evidence,
not interpreting a nonsignificant difference as equivalence. Synthetic clips
without MOS can establish agreement with the original but cannot establish
subjective video accuracy.

## Temporal test corpus and unit coverage

One temporal qualification sample is one complete clip, scored independently
from reset. VMAF contributes its final pooled score, with every emitted frame
retained for diagnostics. CVVDP contributes whole-clip JOD; a prefix or average
of frame JOD values cannot substitute. Keep original content and every
derived clip in the same development or holdout partition. Count frame-rate,
duration and severity variants as related examples, not independent sources.

Start with a small deterministic local test collection and a separate qualification
pilot of 30 to 50 varied originals. Expand the latter according to coverage
and uncertainty, using licensed natural clips plus procedural patterns.
Include static and moving content, cuts, dark/bright scenes, grain, gradients
and opponent color. Derive several severities of compression, blur, noise,
flicker, freeze/duplication and isolated temporal errors. Match duration and
geometry for comparisons. Reference ordering supplies expectations; do not
assume every artifact has a universal severity ordering in every metric.

Extend the previous temporal test proposal with these priorities:

1. VMAF public sequences of lengths 0, 1, 2 and 3, then longer varied clips.
   Assert exact emission timing, feature/frame association, counts, indices,
   final-tail delivery and mean pooling. Test reference-only motion with a
   static reference/changing distortion and the converse. Use different
   spatial features on adjacent frames.
2. A direct CVVDP FIR probe using independent impulse, step and ramp histories
   for all four temporal channels. Check startup and positions around one and
   two history wraps. Activate the fractional coefficient arrays already in
   the fixtures. Existing full-score tests already cover short clips,
   rational rates and several ring boundaries; extend their diagnostic power.
3. Unequal temporal distortions that expose wrong pooling, omitted first/last
   frames, clipped distinctions and errors at the beginning, middle and end.
   Assert changing-but-identical clips preserve the declared identity through
   multiple wraps.
4. Multiframe reuse of host and borrowed Vulkan allocations immediately after
   Push. Compare against owning inputs. Run clip A, Reset, clip B against a
   fresh pipeline, including reset with a pending frame. Extend resident-slot
   release/reuse coverage and interleave independent pipelines to test isolation.
5. Unified Metrics with both VMAFs and CVVDP, checking alignment and one-time
   tail delivery. A non-identical CVVDP CLI clip with an error only near its
   end must expose first-frame-only ingestion. Current VMAF benchmark input
   loading repeats one decoded pair; ordered VMAF file ingestion would be a
   separate frontend change.

Structural and API tests remain hard gates: input validation, finite outputs,
history ownership, synchronization, frame accounting, reset isolation and
declared pooling behavior. Keep current public identity anchors by default;
if a future recalibration intentionally moves an anchor or bound, change its
public contract, documentation and tests together. Rank evidence alone must
not silently alter that contract.

Use numerical unit oracles to verify operations whose mathematical contracts
remain unchanged, and retain intermediate values to diagnose disagreements.
For an intentionally approximated component, specify and test its new local
contract, then qualify its final scores through the agreement gates. Remove
obsolete original final-score pins from mandatory acceptance rather than
silently loosening every tolerance or repinning each new candidate. Tight
equivalence between input paths running the same candidate remains useful
and does not prohibit a common recalibration.

## Implementation sequence

1. Add a shared agreement evaluator under `validation/`, reusing rank and
   correlation implementations in `common.py`. Join by stable sample IDs,
   require equal sample sets and input hashes, reject duplicates/nonfinite
   values, and preserve pinned-reference/configuration provenance. Store clip,
   source, distortion, severity, rate, duration and model/display identifiers.
2. Add CPU tests in `tests/test_validation.py`: offset and positive-scale
   invariance, increasing nonlinear mapping, reversed pairs, collapsed ties,
   misleading global agreement, constant data and missing/mismatched results.
   Challenge the policy with wrong frame order, stale history and dropped
   temporal information as well as harmless score remappings.
3. Add direct agreement reports to `run_corpus.py`, `compare_vmaf_scores.py`,
   `vmaf_cid22.py` and `cvvdp.py`. Required qualification must fail closed when
   the original or required coverage is absent; exploratory skip behavior can
   remain explicit. Keep numerical diagnostics available.
4. Extend offline original generators and a batch sequence driver, then commit
   compact scores/features and hashed manifests. Normal tests consume the
   fixtures without libvmaf, PyTorch or downloads. Preserve independent
   original oracles and read-only reference checkouts.
5. Run both old and proposed policies on frozen baselines and deliberate
   perturbations. Freeze per-domain margins and required strata. Expand the
   development corpus if the policy misses known failures; collect additional
   independent evidence if uncertainty prevents subjective qualification.
   Do not optimize against the holdout. If its results inform a policy or
   implementation adjustment, retire it into development and reserve a fresh
   holdout for the resulting qualification.
6. Make agreement the final-score acceptance gate and raw error diagnostic.
   Update AGENTS.md, docs/validation.md, metric development documents and
   superseded golden-score requirements together. Preserve historical reports
   as records of their original numerical qualification. Keep score-scale
   changes visible in release documentation.

Run Ruff and YAPF on changed Python, quality checks, release build,
`ctest --preset release` and `vqm_test --require-gpu` as applicable. Test-only
changes do not need a speed claim. Changes to production temporal arithmetic
or scheduling still need the existing interleaved correctness/performance
protocol and memory/resource checks.

## Existing data sanity check

The maintained `bench/notes/measurements/cvvdp-cid22-results.csv` contains
4,292 still-image pairs grouped under 49 originals. SHA-256:
`d85d70de7c9a3e0e06335e632748f33cb50fe03b20bc66bdfb8265622b30674f`.
Calling `validation/common.py:correlation_triple` on the aligned `vqm_jod`
and `reference_jod` columns gives SRCC **0.9999999992031778**, tau-b
**0.9999988597467612** and Pearson **0.9999999999979076**. Calling it on
`reference_jod + 3` versus `reference_jod` gives **1.0** for all three, while
MAE is **3.0**. This demonstrates the requested offset invariance using the
existing evaluator; it does not calibrate other metrics or temporal limits.

This calculation used Python/NumPy on retained score records, without a new
binary, build or GPU run. Original hardware/build/input provenance remains
in the [CVVDP qualification report](../../bench/notes/cvvdp/cvvdp-reference-qualification.md).
No current candidate is declared qualified by the proposed policy until its
complete required report exists.
