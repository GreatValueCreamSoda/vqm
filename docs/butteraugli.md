# Butteraugli

## Score scale

The canonical score is the diffmap's maximum
(`ButteraugliScoreFromDiffmap`, `.reference/libjxl/lib/jxl/butteraugli/butteraugli.cc:1954-1964`):
`0` means identical, larger is worse. `ButteraugliScore::distance_p3` is
libjxl's `ComputeDistanceP(p=3)` aggregate
(`.reference/libjxl/lib/extras/metrics.cc:42-145`), the number `cjxl` reports
next to the score. Identity is exactly `0.0`; the identity test enforces it.

## Two levels, not a pyramid

The comparator runs the base level plus one 2x2 subsample of linear RGB. When
that subsampled level is at least 8x8, `Diffmap` folds it with
`AddSupersampled2x(subresult, 0.5, result)`
(`butteraugli.cc:1829-1856`), which multiplies the base by
`1 - 0.3*0.5 = 0.85` and adds `0.5 * sub[y/2][x/2]`
(`AddSupersampled2x`, `:1768-1784`):

```
diffmap = 0.85 * base + 0.5 * sub[y/2][x/2]
```

The deeper `sub_` chain `Make` builds (`:1792-1822`) is never consulted, and
this is not a pyramid: it is exactly the reference comparator's one fold.
Below 8x8 the diffmap is zero.

## Opsin dynamics

`OpsinDynamicsImage` (`butteraugli.cc:1468-1540`): a sigma-1.2 blur, the
absorbance matrix and `Gamma` sensitivity, then
`x = m0 - m1`, `y = m0 + m1`, `b = m2`, with no 0.5 factor and libjxl's
clamps (`min` 1.7557 on x/y, 12.2265 on b). Level 0 also writes the 2x2
subsampled linear RGB that level 1 consumes (`SubSample2x`, `:1733-1766`).

## Frequency separation

`SeparateFrequencies` (`:545-558`) builds LF/MF/HF/UHF with

- `SeparateLFAndMF` (`:404-416`), sigma `7.15593339443`
- `SeparateMFAndHF` (`:418-474`), sigma `3.22489901262`
- `SeparateHFAndUHF` (`:476-536`), sigma `1.56416327805`
- a mask blur, sigma `2.7` (`kRadius`, `:1224`)

and the range operations `remove_range` / `amplify_range` / `maximum_clamp`
with `XybLowFreqToVals` (`:327-365`) and `SuppressXByY` (`:367-390`). The
opsin blur is sigma 1.2.

MF/HF separation forms HF by subtracting the raw blurred MF before changing
MF's range. The HF X channel then receives the reference's intensity-dependent
`SuppressXByY` factor before HF/UHF separation. The GPU applies that factor
while loading each UHF input sample, preserving the same signal decomposition.

## Malta

The UHF bands use the 16-line unit (`MaltaUnit`, `butteraugli.cc:754-949`),
including libjxl's four duplicated line terms (`:882-946`, kept because they
weight UHF energy). `kMaltaLen = 3.75`; the prefactor is
`mulli*sqrt(kWeight*w)/(2*len+1)` with `mulli = 0.39905817637` for the 16-line
unit and `0.611612573796` for the 5-line LF unit (`:1093`, `:1104`), weights
`0.5`/`0.33` (`:998-999`). Six bands are summed in libjxl's order -- UHF Y,
UHF X, HF Y, HF X, MF Y, MF X -- with the `hf_asymmetry` split
(`:1910-1928`); HF bands take `sqrt(hf_asymmetry)`, UHF bands take it directly.

## L2 terms

`L2Diff` (`:1320-1341`) and `L2DiffAsymmetric` (`:1364-1405`), weights
`wmul[9]` (`:72-79`): `400, 1.50815703118, 0, 2150, 10.6195433239,
16.2176043152, 29.2353797994, 0.844626970982, 0.703646627719`. The asymmetric
term adds the extra `0.8` multiplier inside the function and splits its weights
by `hf_asymmetry`; the too-small/too-big factors are `0.4` and `1.0`.

## Mask

`MaskPsychoImage` (`:1252-1266`) derives the mask from the HF/UHF channels of
both images (`CombineChannelsForMasking`, `:1110-1134`: `2.5` on X, `0.4` on
Y), runs `DiffPrecompute` (`:1136-1150`, `kMul = 6.19424080439`,
`kBias = 12.61050594197`), blurs it, and erodes with `FuzzyErosion`
(`:1174-1207`: three smallest of the pixel and its eight neighbours at step 3,
weighted `0.45`/`0.3`/`0.25`). Only image 0's eroded mask is used
(`Mask`, `:1214-1250`), and the Y channel of the error image receives
`ac[Y] += 10 * (blurred0 - blurred1)^2` (`kMaskToErrorMul`, `:1243-1246`).

## Combine and aggregates

`CombineChannelsToDiffmap` (`:1291-1318`) applies `MaskY` (`:1268`),
`MaskDcY` (`:1277`) and `MaskColor` (`:1286`), multiplies channel 0 by `xmul`,
takes the square root of the masked sum and does *no* final blur. The score is
the diffmap maximum (`:1954-1964`).

## GPU pipeline

22 dispatches for a pair whose reference chain is cached, 30 when it is
rebuilt: per level an opsin dispatch, three separation blurs, three Malta
dispatches (UHF/HF/LF), a fused L2 + mask-precompute pass, the mask blur, a
fused erosion + combine pass, plus the two-stage reduce. The reference chain is
cached for uploaded references by `(slot, generation, background)`, so a
released-and-reuploaded slot or a background change cannot serve a stale
reference. Caller-owned buffers are rebuilt because a retained mapping can
change their contents without changing the handle. The arena stores two
12-plane chains per level; intermediate storage is reused within the distorted
chain and between levels. With two levels this is roughly 34 full-resolution
float planes, plus the configured input regions and small tables. Query
memory requirements for the worker's device and configuration.
[GPU memory planning](memory.md)

## Precision and the oracle

Device arithmetic is fp32 with double constants; there is no double-float
escalation. `tests/metrics/butteraugli/butter_ref.cc` is an independent
double-precision port of libjxl. Tests compare scores and `distance_p3`
against that oracle without requiring a libjxl build. The gradient fixtures
and deterministic spatial-frequency, chroma-suppression and directional-Malta
cases permit 0.1% relative error in both aggregates. The spatial cases cover
streamed and cached resident inputs. Identity is exactly `0.0`. Pinned libjxl
supplies the external accuracy comparison.

The GPU path preserves three details from libjxl that affect the score.

- HF subtracts the raw blurred MF before MF's nonlinear range adjustment.
- HF X receives `SuppressXByY` before the UHF split.
- Four first-tap occurrences in two repeated UHF Malta directions have the
  reference's x-offset signs.

The largest residuals on CID22 come from MozJPEG ICC color management.
VQM uses the declared source colors, while libjxl applies embedded ICC
profiles. [Reference differences](divergence.md#butteraugli-on-cid22)

Vship remains a throughput reference; its default intensity target and UHF
weighting differ from pinned libjxl.

## Reference anchors

All in `.reference/libjxl/lib/jxl/butteraugli/butteraugli.cc` unless noted.

| our stage | reference |
| --- | --- |
| `ComputeKernel` radius rule | `:78-88` |
| `ConvolutionWithTranspose`, `Blur` | `:110-265` |
| `OpsinDynamicsImage` | `:1468-1540` |
| LF/MF/HF/UHF separation | `:404-558` |
| 16-line Malta unit, duplicates | `:754-949`, `:882-946` |
| 5-line LF Malta unit | `:579-750` |
| `MaltaDiffMapT`, wrappers | `:988-1108` |
| mask, erosion, combine | `:1110-1318` |
| L2 terms | `:1320-1405` |
| `Make`, `Diffmap`, fold | `:1792-1822`, `:1829-1856` |
| `DiffmapPsychoImage` | `:1894-1952` |
| `ButteraugliScoreFromDiffmap` | `:1954-1964` |
| `ComputeDistanceP` | `lib/extras/metrics.cc:42-145` |
