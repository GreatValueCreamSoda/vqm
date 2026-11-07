# SSIMULACRA 2.1

## Score scale

The score is in `-inf..100` and higher is better. `100` means the inputs are
mathematically identical: the error maps divide a difference of numerator and
denominator (`src/metrics/ssimulacra2/shaders/scale.comp`, `src/metrics/ssimulacra2/score.cc`), so for equal planes
every map value is exactly zero before any rounding, and anything that stops
the cancellation is the metric breaking, not drift to pin. Negative scores mean
very strong distortion.

## Pipeline

decoded image -> linear Rec.709 -> XYB -> 6-scale 2x2 box pyramid -> per-scale
blurs and error maps -> 108 norms -> weighted sum -> monotone score mapping.

The whole chain runs on the GPU. At six scales the CPU submits nine dispatches,
or eight when an unchanged resident reference pyramid is cached, and reads back
18 norm sums per scale; the score mapping is the only arithmetic in double, on
the host.

## Input and XYB

The shared GPU decoder uses the image's format descriptor to convert decoded
planes to linear Rec.709. The API reference lists the accepted formats,
color enums, and chroma filters. [Input formats](reference.md#input-formats)

XYB is the opsin absorbance matrix, the bias and the cube root. The scaled
form the metric uses is the reference's `MakePositiveXYB`
(`.reference/ssimulacra2/src/ssimulacra2.cc:209-220`):
`X = 14*X + 0.42`, `Y = Y + 0.01`, `B = B - Y + 0.55`
(`src/metrics/ssimulacra2/shaders/prepare.comp` carries the three offsets). Alpha is blended
against a background before the transfer decode, in encoded RGB, with the
background as the blend target; the CLI scores dark and bright backgrounds
(0.1 and 0.9) and reports the worse score, exactly as the reference CLI does
(`ssimulacra2.cc:421-440`).

## Pyramid

The base plane plus 2x2 box downsampling in linear RGB, at most 6 scales
(1:1 ... 1:32, `kSsimulacra2MaxScales`). Downsampling stops when a scale's width or height
falls below 8 (`kSsimulacra2MinDimension`); the reference checks the previous scale before
downsampling, so a 32x32 image runs scales 32, 16, 8 and 4. Fewer than six
scales change which weight rows the score uses; see *Score mapping*.

## Blur

Every map is built on a Gaussian blur of sigma 1.5 (`kSsimulacra2GaussianSigma`), sampled
and truncated at 3.5 sigma (`kSsimulacra2GaussianSupport`): radius 6, 13 taps, normalized
to sum one. The reference uses a recursive IIR (`CreateRecursiveGaussian(1.5)`,
`ssimulacra2.cc:82-106`) at the same sigma. The consequence is measured: on
CID22 our mean absolute deviation from the reference was 0.3120 and ours was
higher on every pair; the blur is the dominant cause of that residual. See
[divergence](divergence.md) for the comparison.

## Maps and norms

Per scale, per channel, the shader (`src/metrics/ssimulacra2/shaders/scale.comp`) evaluates:

- SSIM' map: `num_m = 1 - (mu1 - mu2)^2`,
  `num_s = 2*(s12 - mu1*mu2) + 0.0009`,
  `denom_s = (s11 - mu1^2) + (s22 - mu2^2) + 0.0009`,
  `d = max(1 - num_m*num_s/denom_s, 0)`.
  It publishes the mean and the fourth root of the mean fourth power.
- Ringing/blurring map: `d1 = (1 + |d - mu_d|)/(1 + |o - mu_o|) - 1`;
  `artifact = max(d1, 0)` and `detail_lost = max(-d1, 0)`, each with a mean
  and a fourth-root norm.

That is six norms per channel per scale: 18 per scale, 108 in total.

## Score mapping

`Ssimulacra2Result::Score()` (`src/metrics/ssimulacra2/pipeline.cc:369`) accumulates
`error = sum(weight * |norm|)` over 18 weight rows: channels X, Y, B and scales
1:1 ... 1:32, six weights per row (the 1-norm and 4-norm weights of SSIM',
ringing and blurring). Then:

```
scaled = 0.9562382616834844 * error
mapped = 2.326765642916932*scaled - 0.020884521182843837*scaled^2
       + 6.248496625763138e-05*scaled^3
score  = 100 if mapped <= 0, else 100 - 10*max(mapped, 0)^0.6276336467831387
```

The reference consumes the table as one flat 108-entry run
(`.reference/ssimulacra2/src/ssimulacra2.cc:261-417`), so a channel starts at
`c * scales.size()`, not at `c * 6`. A geometry with fewer than six scales
therefore reads a different row set, which is why pruning is
geometry-dependent: `Ssimulacra2ChannelPruned` and `Score()` index the same way.

## Pruning

A `(channel, scale)` pair is skipped when all six of its weights are `<= 0.01`
(Vship's threshold, `.reference/Vship/src/Vulkan/ssimu2/score.hpp:3-4`); seven
pairs qualify at six scales. The pruned norms are published as exactly zero,
which preserves the identity. Mean absolute score drift was 2.7e-4
in the measured CID22 comparison.
[Reference differences](divergence.md#ssimulacra-2-vs-cloudinary-ssimulacra2)

The pruning decision is made per geometry with the flat indexing above, so a
caller with a different scale count gets the reference's own behaviour.

## Half-precision planes

`Ssimulacra2Config::half_planes` stores the intermediate planes as f16 and builds
the `_half` shader variants. It is opt-in and SSIMULACRA 2 only. On the full
CID22 set the median absolute score difference was 0.0538 and the maximum was
0.5636; see [divergence](divergence.md).

## GPU pipeline

At six scales, up to nine dispatches per pair:

- `prepare` x2: decode (colour conversion, matrix, transfer, primaries), XYB,
  and the whole 2x2 box pyramid in one dispatch per image; each workgroup
  builds its part of the pyramid. The original's dispatch is omitted when an
  immutable resident reference and its background match the cached pyramid.
- `scale` x6, one per level: the three channels travel in `grid.z`; each
  workgroup filters the five blur quantities out of a shared tile, evaluates
  both error maps and reduces the six norm partials.
- `reduce` x1: the 108 norms.

The scale pass selects the first tile that fits the device's compute limits.
The 32×32 output tile uses 256 invocations and 21,120 bytes of shared memory.
The 32×16 fallback uses 128 invocations and 13,440 bytes. Prepare, scale, and
reduce use the same pruning predicate. The GPU sums the partials, and the host
computes the mean and fourth root in double precision.

## Reference anchors

| our stage | `.reference/ssimulacra2/src/ssimulacra2.cc` |
| --- | --- |
| `SSIMMap` | `:115-160` |
| `EdgeDiffMap` | `:162-207` |
| `Downsample` | `:44-66` |
| `Blur`, `CreateRecursiveGaussian(1.5)` | `:82-106`, `:85` |
| `MakePositiveXYB` | `:209-220` |
| `AlphaBlend` | `:222-240` |
| `Ssimulacra2Result::Score`, weight table | `:261-417`, `:264-405` |
| `ComputeSSIMULACRA2` | `:419-491` |

The XYB helpers come from libjxl: `OpsinAbsorbance`
(`.reference/libjxl/lib/jxl/enc_xyb.cc:50-80`), `LinearRGBToXYB` (`:83-105`) and
`ToXYB` (`:233-288`).
