# Divergence from the references

Two tables: every per-stage difference between this implementation and the
SSIMULACRA 2 reference (Cloudinary's `ssimulacra2`) and between this
implementation and libjxl's Butteraugli. The class column says what the
difference is, not how much it matters:

- **exact math** -- a different but equivalent formula; contributes to the
  numeric residual only.
- **precision** -- same formula, different accumulation width.
- **fusion only** -- same arithmetic, fewer dispatches.
- **deliberate metric change** -- the score moves on purpose.
- **input surface** -- what the caller can feed in, not the metric.

## SSIMULACRA 2 vs Cloudinary `ssimulacra2`

| step | reference | here | measured effect | class |
| --- | --- | --- | --- | --- |
| input surface | libjxl 8-bit sRGB PNG/JPEG only; ICC through lcms2 | zimg's full surface: datatype, range, subsampling/siting, transfer, matrix, primaries, alpha; ICC/gAMA/cHRM/sBIT ignored, `--input-*` declares instead | scores only comparable on sRGB 8-bit | input surface |
| sRGB -> linear | Chebyshev rational polynomial (`.reference/ssimulacra2/src/lib/jxl/transfer_functions-inl.h:268-332`), about 2e-8 absolute on 8-bit | exact piecewise `pow` (`src/vulkan/shaders/color.glsl`) | part of the residual | exact math |
| XYB cube root | bit trick plus Newton refinement (`.reference/ssimulacra2/src/lib/jxl/fast_math-inl.h:173-205`, ~6 ulp) | `pow(v, 1/3)` | part of the residual | exact math |
| colour management | embedded ICC through lcms2 (`ssimulacra2.cc:437-439`) | declared transfer and primaries, no lcms | input surface | input surface |
| blur | recursive IIR `CreateRecursiveGaussian(1.5)` (`ssimulacra2.cc:82-106`; `lib/jxl/gauss_blur.cc:499`) | 13-tap sampled Gaussian, sigma 1.5, +/-3.5 sigma, normalized | **deliberate**: the dominant cause of the CID22 residual, ours higher on every pair | deliberate metric change |
| pruned channels | all 54 maps evaluated | seven `(channel, scale)` pairs whose six weights are all <= 0.01 skipped | mean drift 2.7e-4 (max 8.1e-4), resident pair 2.25 -> 1.21 ms | deliberate metric change |
| fusion | separate CPU passes | prepare fuses decode+XYB+pyramid; scale fuses blurs+maps+norms; one reduce dispatch | 108 norms within 1.6e-8 | fusion only |
| precision | double accumulation in places | fp32 partials, double host mean and fourth root | golden pins (93.17505696 / 92.12553218, +/-5e-3) | precision |
| half planes | none | f16 intermediate planes, opt-in | median abs diff 0.0538, max 0.5636, correlation change <= 5e-4 | deliberate metric change |
| alpha | lcms/alpha path, reference CLI checks the original for alpha | encoded RGB, worse of dark/bright backgrounds when either input has alpha | avoids missing transparency on the distorted input alone | input surface |

The previous release reproduced the recursive Gaussian and measured a mean
absolute deviation of 0.0269 over the same 4292 pairs. That is the attribution:
with the recursive blur the sampled-vs-recursive difference disappears and the
exact sRGB and cube root account for the rest. The sampled Gaussian is the
dominant remaining deviation and the one deliberate metric change in the table
that moves scores on every pair.

## SSIMULACRA 2 on CID22

The table records a historical run of
`python3 validation/run_corpus.py --metric all --jobs 8`. A rerun writes its
local summary to `artifacts/reports/validation/ssimulacra2-summary.md`.

| quantity | value |
| --- | --- |
| pairs | 4292 |
| ours vs Cloudinary, mean abs diff | 0.3120 |
| ours vs Cloudinary, max abs diff | 2.9907 |
| ours vs Cloudinary, pairs over 0.01 | most (the sampled blur) |
| ours vs Cloudinary, norms (fusion) | <= 1.6e-8 |
| ours vs Cloudinary, half planes (median / max) | 0.0538 / 0.5636 |

## Butteraugli vs libjxl `butteraugli.cc`

| step | reference | here | measured effect | class |
| --- | --- | --- | --- | --- |
| levels | `Make` builds a deeper `sub_` chain; `Diffmap` folds exactly one sub-level (`:1792-1822`, `:1829-1856`) | two levels, same fold `0.85*base + 0.5*sub` | equivalent for the comparator path | exact math |
| blur | `Separable5` or two transposed convolutions; `ComputeKernel` radius `max(1, int(2.25*sigma))` (`:78-88`, `:110-265`) | same radii and tap counts 5/7/13/15/33, truncate-and-renormalize borders | exact-ish, fp32 | precision |
| MF/HF separation | subtract raw blurred MF, then adjust MF range (`:449-466`) | same order | corrected on 2026-11-06; earlier shaders subtracted range-adjusted MF | exact math |
| HF color suppression | `SuppressXByY` before UHF separation (`:367-390`, `:472`) | same factor applied while loading UHF inputs | corrected on 2026-11-06; earlier shaders omitted it | fusion only |
| Malta | 16-line unit with intentionally duplicated terms (`:754-949`, `:882-946`) | same directional taps and duplicated terms | corrected on 2026-11-06; four earlier first-tap occurrences had wrong x-offset signs | exact math |
| two-level fold | `AddSupersampled2x(result, 0.5, base*0.85)` (`:1768-1784`) | same | exact | exact math |
| precision | double in several spots | fp32 device, double constants, no double-float escalation | corrected PNG subset MAE 0.00001493, max 0.00043583; includes remaining transfer/blur arithmetic differences | precision |
| ICC | applies embedded profiles | ignores PNG/JPEG ICC, assumes declared colors | corrected JPEG subset MAE 0.00217974, max 0.03281712; removing only ICC from the eight largest cases reduces their maximum error to 0.00024080 | input surface |
| intensity target | default 80 nits | 80, configurable | same | exact math |
| Vship comparison | Vship defaults to 203 nits and leaves UHF at "15 patterns" (~1.2%) | not an oracle | throughput only | input surface |

## Butteraugli on CID22

Butteraugli is lower-is-better, so its score is negated before correlating with
MCOS; the CSV carries raw scores. CID22 was collected for SSIMULACRA 2 and is
not Butteraugli's calibration corpus, so the correlation is a cross-metric
sanity check, not a libjxl claim.

The 2026-11-06 production correction restores the three reference operations
above. Earlier shaders had mean absolute score error `0.0758241`, maximum
`2.73060131` and 466 pairs above 5% relative error. Full-CID22 production
validation reduces these to `0.0002853`, `0.03281712` and zero pairs above 5%.
The [investigation](../bench/notes/system/butteraugli-drift.md) records the
cumulative ablations, independent oracle checks, ICC experiment and production
validation. Historical reports retain their original results.
