# Butteraugli score drift investigation

Date: 2026-11-06.

## Finding

Three GPU implementation defects explained almost all of the large score drift
from pinned libjxl: the MF/HF residual was formed after a nonlinear adjustment,
HF red-green suppression was missing, and four UHF Malta tap occurrences had
incorrect signs. Correcting them in an isolated build reduced full-CID22 mean
absolute score error by **99.624%**, from `0.0758241` to `0.0002853`.
The remaining large errors are associated with MozJPEG ICC profiles.

The initial investigation left production shaders and goldens unchanged. The
three fixes were subsequently adopted; the production qualification below
records the regression tests and gates. Historical corrections remain as
experimental patches, with a
standalone oracle runner and a deterministic synthetic reproducer in
[`bench/experiments/butteraugli_drift/`](../../experiments/butteraugli_drift/README.md).
The existing documentation's attribution to fp32 rounding has been corrected.

## Initial investigation provenance and method

- Baseline: `a949b6177efb88890dd44860cc30dff45486a25d` plus the preexisting
  dirty working tree, including source relocation and host refactors. `HEAD`
  alone is not the measured source. The relevant source/binary SHA-256 hashes
  are in the [compact results](../measurements/butteraugli-drift-results.json).
- Reference: read-only `.reference/libjxl`, pin
  `e47d2a2b5bb418f2947ce07f36cb048861db3820` in
  `tools/references.lock:10`. `tools/build_references.py --only libjxl --jobs 8`
  built its copy under `build/libjxl-src` / `build/libjxl-build`.
- Hardware: NVIDIA GeForce RTX 5070, driver `615.78.08`, x86_64 host,
  GCC `16.2.1 20260810`. CMake Release uses `-O3 -DNDEBUG`; shaders use
  `glslc -O --target-env=vulkan1.1`.
- Inputs: all 4,292 CID22 pairs, 49 originals, 512x512. Both arms use
  `hf_asymmetry=1`, `xmul=1`, `intensity_target=80`. The reference harness
  explicitly sets them in `bench/inmemory/libjxl_bench.cc:132-137` and reports
  the canonical maximum-diffmap score through `ComputeScore` at `:201-214`.
- Baseline core was configured/built with the release presets. Its full
  `./build/vqm_test --require-gpu` run passed all 206 tests with zero failed
  checks and no reported skips. The identity score was exactly `0.00000000`.
- Experimental builds use a copy of the measured working tree at
  `build/butteraugli-drift-src`, built separately under
  `build/butteraugli-drift-experiment`. Only the two shader files implicated
  below were changed in that copy. The suppression variant reads immutable
  sibling HF Y samples while loading the UHF input tile; the output HF/UHF
  regions are separate from that input, preserving dispatch/storage ordering.
- Fresh baseline and reference scores reproduced every historical score in
  `artifacts/reports/validation/butteraugli.csv` exactly at eight printed decimal
  places. Each cumulative arm scored every pair once; scores were checked for
  count and finiteness. These are correctness experiments, not timed speed
  comparisons.

Raw logs, pair lists, nine selected oracle comparisons, copied shader variants
and saved executables are under `artifacts/reports/butteraugli-drift/`.
Reference/build logs are also at
`artifacts/reports/butteraugli-drift-reference-build.log` and
`artifacts/reports/butteraugli-drift-build.log`.
The compact JSON preserves corpus statistics, hashes, selected results and
correlations; the [ICC CSV](../measurements/butteraugli-drift-icc.csv) preserves
the eight largest remaining input-profile cases.

## Source defects

### HF subtracts the range-adjusted MF value

`src/metrics/butteraugli/shaders/butter_blur.comp:73-80` computes
`aux = pre - value` after changing `value` with
`remove_range(blurred, 0.29)` or `amplify_range(blurred, 0.1)`.
Libjxl computes `HF = pre - blurred` before modifying MF:
`.reference/libjxl/lib/jxl/butteraugli/butteraugli.cc:449-466`.
The independent oracle preserves this order at
`tests/metrics/butteraugli/butter_ref.cc:324-335`.

The GPU therefore injects part of the MF nonlinear adjustment into HF. Both
the subsequent HF/UHF split and the error/masking calculations consume changed
bands. This alters the decomposition even with exact arithmetic.

### HF X omits intensity-dependent suppression

The reference calls `SuppressXByY(hf[1], &hf[0])` after MF/HF separation and
before HF/UHF separation (`butteraugli.cc:471-472`). It multiplies HF X by

```
0.653020556257 + (1 - 0.653020556257) * 46 / (Y_HF^2 + 46)
```

as defined at `:367-390`. The oracle applies it at `butter_ref.cc:340-349`.
Our dispatch sequence and shader omit this operation. Consequently, the GPU's
red-green high-frequency response differs according to local intensity detail.
The isolated correction applies the factor to every input sample used by the
UHF blur and to the center sample used for its residual.

### Four UHF Malta taps have the wrong x-offset sign

The first taps of shader terms 5, 6, 15 and 16, at
`src/metrics/butteraugli/shaders/butter_malta.comp:88`, `:94`, `:149`, `:155`,
use the wrong x sign. They belong to two directional patterns that appear
twice in the reference's intentional weighting. The two intended first taps
are `(x=+1,y=-4)` and `(x=-1,y=-4)`; see reference `butteraugli.cc:792`,
`:800`, `:925`, `:941`. The rest of each line follows the intended direction,
so the incorrect first sample changes its squared directional energy.

All reference/shader line offsets were compared. The LF unit swaps two whole
terms, which preserves the sum apart from association; it has no corresponding
coordinate defect. Intentional duplicated full-unit terms were preserved.
Review of defaults, two-level fold, opsin, weights, L2, mask, erosion, pooling
and scratch sequencing found no further substantive omission.

## Full-corpus cumulative ablations

All values compare raw canonical scores against the same freshly built pinned
libjxl arm. Each correction is added to the previous arm; these differences
are order-dependent attribution, not independent additive error budgets.

| GPU arm | Mean absolute error | Maximum | p99 | Pairs over 5% relative error |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.07582413 | 2.73060131 | 0.60755089 | 466 |
| Correct raw MF residual | 0.04988420 | 1.57180834 | 0.41691184 | 284 |
| Also restore `SuppressXByY` | 0.01895610 | 0.32447004 | 0.16324432 | 54 |
| Also correct Malta taps | 0.00028528 | 0.03281712 | 0.00589256 | 0 |

The baseline has 138 pairs over 10% relative error and a p99 absolute relative
error of 16.179%. The fully corrected build has a p99 absolute relative error
of 0.1934%. The number of pairs with absolute error above `0.01` falls from
3,481 to eight.

The worst baseline pair is `original/5458393.png` versus
`compressed/5458393/libjxl/e7_q50.png`:

| Arm | Score |
| --- | ---: |
| Pinned libjxl | 4.13950586 |
| Baseline GPU | 6.87010717 |
| Correct raw MF residual | 5.71131420 |
| Also suppress HF X | 4.13332081 |
| Also correct Malta taps | 4.13951015 |
| Independent double oracle | 4.13951473466 |
| Copied oracle using float arithmetic | 4.13950729370 |

The baseline error is +65.964%. The float experiment replaces `double` with
`float` in scratch copies of the oracle source/header and compiles with
`-fsingle-precision-constant`; it is a diagnostic, not an independent oracle.
Its close match decisively rejects a general fp32 explanation for this outlier.

Drift is content-dependent and has both signs. The top 25 baseline absolute
outliers use original `5458393`, but excluding it still leaves 405 pairs over
5% error. The largest negative baseline outlier is the Semarang City Logo
`cld_avif/q50.png` pair: `7.42851114` versus `7.95955420`. The combined
corrections bring MCOS KRCC/SRCC/PCC from `0.544504/0.733502/0.670087` to
`0.551090/0.741350/0.684256`, against reference
`0.551090/0.741350/0.684254`.

## Remaining ICC/input residual

After all three corrections:

| Distorted format | Pairs | Mean absolute error | Maximum |
| --- | ---: | ---: | ---: |
| PNG | 3,756 | 0.00001493 | 0.00043583 |
| JPEG | 536 | 0.00217974 | 0.03281712 |

All top 25 remaining errors, including all eight above `0.01`, are MozJPEG
JPEGs. Their `sRGB MozJPEG` ICC profiles contain a 26-entry sampled transfer
curve. Our loader ignores embedded ICC while libjxl applies it.

Removing only ICC APP2 segments from copies of those eight JPEGs, without
changing the compressed image bytes or decoded pixels, changes libjxl's
scores toward the corrected GPU scores. Seven of eight then agree within
`5.96e-6`; the largest residual shrinks from `0.03281712` to `0.00024080`.
For `2775196/mozjpeg/q95.jpg`, libjxl changes from `1.25949275` to
`1.28006268`, versus corrected GPU `1.28006065`. For the largest remaining
case, `2887497/mozjpeg/q60.jpg`, libjxl changes from `3.54057097` to
`3.50799465`, versus GPU `3.50775385`.

This isolates the large residual tail to profile interpretation rather than
JPEG decoding flags. It does not establish arbitrary ICC parity or prove that
every remaining numeric discrepancy has the same origin. The worst original
algorithm outlier's profiles describe sRGB, and CMS conversion preserved its
decoded samples; input profiles could not explain its 66% baseline error.

## Why the tests missed it

Before adoption, `TestButterAgainstReference` checked two smooth 32x32
gradients at a 5% tolerance. Odd-dimension oracle cases used constant colors,
and small-geometry cases checked only finite/nonzero scores and identity. Those inputs exercised
little of the color/texture/orientation response affected by the defects.
Golden tests pin the existing implementation's numbers and therefore cannot
establish reference equivalence.

The deterministic 64x64 checker generated by `make_checker.py` needs no corpus
and exposes the issue: baseline `2.42218304`, double oracle `2.57974731586`,
libjxl `2.57976222`, corrected GPU `2.57974863`. Baseline relative error is
6.11%. The generator and oracle runner preserve this as a compact candidate
for a regression test, now included in the production suite. The corrected
identity remains exactly zero.

`validation/run_corpus.py` gates raw-score parity only with `--strict-parity`.
Its usual gate compares correlations against MCOS. The historical memory run
used `--accuracy-tolerance 0.02`, which admitted the observed PCC drop of
0.0142. The fresh baseline fails even the default 0.01 accuracy tolerance.
The fully corrected build would still fail the default strict maximum `0.01`
raw-score threshold on eight ICC-bearing JPEGs, so input-profile and metric
parity need separately controlled comparisons.

## Reproduction and follow-up

Baseline command:

```sh
python3 validation/run_corpus.py --metric butteraugli --only ours,libjxl \
  --out-dir artifacts/reports/butteraugli-drift/baseline-corpus
```

Experimental arms use an unchanged full pair list and this score-only shape:

```sh
./build/butteraugli-drift-experiment/vqm_bench --metric butteraugli \
  --score-only --pairs-file artifacts/reports/butteraugli-drift/cid22.pairs \
  --iterations 1 --warmup 0 --threads 3 --decode-threads 16
```

The [experiment README](../../experiments/butteraugli_drift/README.md) gives
fresh source-copy, build, patch, checker, oracle and ICC commands, including
reconstruction of the pre-fix baseline from the corrected source. The
production qualification below adds checker/directional/color regression
inputs against the independent oracle, tightens assertions, updates moved
golden values and retains explicit corpus comparisons.
No performance claim follows from these score-only experiments. A production
hot-path change still requires the repository's interleaved resident and
streaming regression protocol, full correctness gates and quality checks.

During the initial investigation, research-tool C++ formatting, Python Ruff
lint and YAPF formatting were checked. The repository-wide quality command
was attempted and failed on unrelated
concurrent CVVDP declarations in `include/cvvdp.h` and `include/vqm_memory.h`;
those files were left untouched by this investigation. No production C++ lint,
sanitizer or performance gate is claimed for the experimental shader patches.

## Production adoption and qualification

The three corrections were applied to production later on 2026-11-06. HF now
subtracts the raw blurred MF, UHF input loading applies `SuppressXByY`, and the
four Malta tap signs match the pinned reference. The raw HF X/Y input planes
are immutable during UHF dispatch; the resulting HF/UHF planes occupy separate
storage. No additional dispatch, buffer, reference-cache policy or source
binding was introduced. The independent CPU oracle is unchanged.

`TestButterAgainstReference` now allows 0.1% relative error in score and
`distance_p3`. Three additional oracle regressions use a 64x64 checker,
65x47 simultaneous luminance/chroma detail, and a transposed 65x47 checker.
Each compares streaming, first resident evaluation and repeated cached
evaluation. Reintroducing each bug individually in an isolated copy proves
that these cases detect it:

| Single bug restored | Failing input | Score / oracle | Relative score error |
| --- | --- | ---: | ---: |
| Wrong MF residual | Checker | 2.40446949 / 2.57974732 | -6.79% |
| Missing suppression | Luminance/chroma | 2.41972351 / 2.44109058 | -0.875% |
| Wrong Malta signs | Checker | 2.59702873 / 2.57974732 | +0.670% |

The associated `distance_p3` checks also fail each mutant. The full corrected
three-case matrix passes. The gradient B score/p3 pins are now
`3.77568078` / `2.92297296`, with their existing `5e-3` absolute pin tolerance.
Gradient C measures `2.95148849` / `2.35224729`. SSIMULACRA 2 score pins remain
unchanged. Public/API documentation and gate guidance reflect this contract.

The [production results](../measurements/butteraugli-production-results.json)
preserve exact scores, source/binary/input hashes, mutation results, all
five timing samples per arm, medians and spreads. Raw logs are under
`artifacts/reports/butteraugli-production/`.

Correctness and quality checks actually run:

- Release configure/build with warnings as errors; `./build/vqm_test
  --require-gpu`: 227 tests, zero failed checks, no lack-of-device skips.
- ASan/UBSan preset configure/build and `ASAN_OPTIONS=detect_leaks=0
  UBSAN_OPTIONS=halt_on_error=1 ./build-asan/vqm_test --require-gpu`: the same
  227 tests passed without sanitizer findings.
- `ctest --preset release`: all 14 suites passed, including package consumers,
  CLI, sequence I/O and benchmark integration.
- `python3 tools/check_quality.py --lint`: 237 maintained files formatted and
  47 compiled production C++ files linted successfully.
- `python3 tools/fetch_references.py --check --only
  cid22,libjxl,ssimulacra2`: all selected pinned inputs verified offline.
- `python3 tests/make_goldens.py` regenerated CLI/benchmark fixtures. Both
  identities were checked: SSIMULACRA 2 `100.00000000`, Butteraugli score and
  p3 `0.00000000`. The active shader manifest retains 124 variants.
- Full Butteraugli CID22 against libjxl with `--accuracy-tolerance 0.005`:
  PASS. Every production score exactly matches the earlier combined-correction
  experiment at eight printed decimal places. No pair exceeds 5% relative
  error. The 3,756 PNG pairs pass the default raw-score thresholds; eight
  ICC-bearing JPEGs still exceed the default strict maximum, as explained
  above. No input color-management change is included in this fix.
- The 300-pair SSIMULACRA 2/Cloudinary accuracy gate at tolerance 0.005:
  PASS. Its sampled-vs-recursive blur residual is unchanged.

### Interleaved performance regression measurements

Baseline commit selection was explicit: `a949b6177efb88890dd44860cc30dff45486a25d`
plus the captured pre-fix dirty tree, rather than bare `HEAD`. A detached
`/tmp/vqm-butteraugli-base` worktree received that captured source and its own
Release build. Only Butteraugli shader arithmetic and explanatory production
comments differ between the measured arms. The same NVIDIA RTX 5070 / driver
615.78.08 measured both in one sitting.

Each geometry/mode alternated baseline A then fixed B for five rounds.
Resident runs used `--iterations 20 --warmup 3 --resident 2`; streaming used
`--iterations 8 --warmup 2` with no resident flag. Both used default f32 planes,
one worker and the existing `testsrc2`/bicubic down-up pairs under
`.cache/bench/pairs/synthetic-{1920x1080,3840x2160}/`. Every run's score,
checksum and stage table was checked. Per-arm scores were stable across all
rounds and modes; checksums matched warm-up plus timed call counts.

| Workload | Baseline pair ms, median [min..max] | Fixed pair ms, median [min..max] | Change |
| --- | ---: | ---: | ---: |
| 1080p resident | 2.082586 [2.074777..2.089237] | 2.079516 [2.068251..2.081396] | -0.15% |
| 1080p streaming | 3.474655 [3.464220..3.480820] | 3.465220 [3.449855..3.468720] | -0.27% |
| 4K resident | 9.285810 [9.266116..9.315001] | 9.360032 [9.338596..9.557487] | +0.80% |
| 4K streaming | 15.288186 [15.095611..15.309210] | 15.259556 [15.126376..15.457530] | -0.19% |

Pair-time medians stay within 1% of baseline. Restoring required arithmetic
has a measurable stage cost at 4K: separation changes from `2.532846` to
`2.571613` ms resident (+1.53%) and `5.063908` to `5.266312` ms streaming
(+4.00%). The unchanged mask stage also varies (+3.07% resident), so stage
changes should not all be attributed directly to instruction counts. The JSON
records every stage's samples and spread. This is a correctness correction;
no speedup claim or optimization speed gate is asserted.

After confirming the baseline worktree contained no unique authored work, its
source and binary were preserved at
`artifacts/snapshots/butteraugli-production-baseline-2026-11-06.tar.gz` and
`artifacts/reports/butteraugli-production/baseline-vqm_bench`. The worktree was
removed; the saved manifest and archive hashes identify the measured baseline.
