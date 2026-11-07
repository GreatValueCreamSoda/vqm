# Original ColorVideoVDP reference and MATLAB review

Date: 2026-11-05. Scope: source and literature review for an implementation
plan. No metric implementation, reference execution, build, GPU validation or
performance measurement was performed. This review adds only this document;
third-party implementation source remains outside maintained repository files.

## Reference choice and the MATLAB premise

The original authors' implementation is
[gfxdisp/ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP), linked by the
[Cambridge project page](https://www.cl.cam.ac.uk/research/rainbow/projects/colorvideovdp/).
The paper is Mantiuk, Hanji, Ashraf, Asano and Chapiro, *ColorVideoVDP: A visual
difference predictor for image, video and display distortions*, SIGGRAPH 2024,
Article 129, [DOI 10.1145/3658144](https://doi.org/10.1145/3658144). The authors
provide a [paper PDF](https://www.cl.cam.ac.uk/~rkm38/pdfs/mantiuk2024_ColorVideoVDP.pdf)
and [supplement](https://www.cl.cam.ac.uk/~rkm38/pdfs/mantiuk2024_ColorVideoVDP_supplementary.pdf).
The paper and supplement explain the model; pinned executable source and data
must define numerical compatibility.

There is **no native MATLAB implementation of the complete metric** in the
official repository. The pinned [MATLAB README][matlab-readme] describes a
wrapper; [cvvdp.m][matlab] writes matrices and FPS into temporary MAT files,
launches the Python CLI in a conda environment, and parses its answer. Its
matrix arguments do not make it an independent mathematical oracle. The
`matlab/derive_LMS_colourspace/` directory is a color-transform derivation aid.
The separate [castleCSF project](https://www.cl.cam.ac.uk/research/rainbow/projects/castleCSF/)
does provide a MATLAB CSF model, but that is only one component of ColorVideoVDP.
FovVideoVDP is another distinct metric; its MATLAB implementation must not be
presented as an original ColorVideoVDP implementation.

Use release **v0.5.7**, commit
`2a268bce8d56e2f3abde46df3927d8a633707a24` (2026-08-02), as the modern
original-author reference candidate. [The release](https://github.com/gfxdisp/ColorVideoVDP/releases/tag/v0.5.7)
fixes short-video symmetric padding and returns the default padding to
`replicate`. The reviewed temporary checkout was
`/tmp/cvvdp-original-review-20261105`, detached at that commit. It is convenient
inspection material, not a reproducibility prerequisite. The observed current
main commit was `7ceb47b1d6755b8e11f8b5e237c40bb868b3ab76`; it was not used
for the stage review below.

The older v0.4.2 release is commit
`3a746fd45b798a712f72ac309efdb27a27a875c8`. Its default parameter JSON and
v0.5.7's differ only in the `version` string. Both identify the calibration as
`10/01/2024`, internal model `cvvdp_mult_mutual`, and have the same numerical
coefficients. That does **not** establish equivalent executable behavior:
source, batching, color input, interpolation and boundary handling changed.
The [changelog][changelog] records a batching-dependent score discrepancy fixed
in v0.5.3 and a YCbCr conversion change in v0.5.6. A peer matching these
coefficients cannot be assigned an exact upstream revision from coefficients
alone. Keep any Vship compatibility target separate from the original-author
target and report differences explicitly.

## Exact data and provenance

All following files are from the v0.5.7 commit above. SHA-256 values were
computed from the checked-out bytes.

| Input | SHA-256 |
| --- | --- |
| [cvvdp_parameters.json][parameters] | `ef4dedb44f11f33f509ea3d6d204bb23b03917259b8efbc077009ce53211762b` |
| [csf_lut_weber_fixed_size.json][lut] | `df39c869c0c53a9a2cb757811ecaf4ce0d44b4151f5ff9590f5bfe1ede940c18` |
| [display_models.json][displays] | `653979727e809fcd9e44424aff2615d54edde026d9feb17d8f07b7f0ff61b517` |
| [color_spaces.json][colors] | `30ce6752d35672d97a9d7fb7286170915a2ddfe4307782693f9ee176ef9491d1` |

For comparison, v0.4.2's parameter JSON hash is
`411218e52172b3c52bac2b78c09edd2161a5d6b07b51080a19a7d5e27fe87cb7`;
its default CSF LUT hash is identical to v0.5.7's. Parameter configuration
selection can be overridden by explicit paths or `CVVDP_PATH`; the validation
runner must record the actual loaded files and prevent accidental overrides.

The active base model selects `weber_g1` contrast, `weber_fixed_size` CSF,
`mult-mutual` masking, cross-channel masking and soft difference saturation.
Alternate configuration files, log-contrast models and `cvvdp-ml-*` are
different targets, not interchangeable modes of a single pinned score.

The bundled CSF LUT supplies four 32 by 32 tables of log sensitivity: sustained
achromatic, sustained red-green, sustained yellow-violet, and transient
achromatic. Its axes span luminance 0.005 to 10000 cd/m² and spatial frequency
0.1 to 64 cycles/degree; temporal samples are 0 and 5 Hz, with fixed
`ge_sigma=1.5`. [csf.py][csf] interpolates log sensitivity against log spatial
frequency, then against log background luminance. The spatial-frequency helper
extrapolates outside the axis; the uniformly sampled luminance helper clamps to
the endpoints. These policies need independent boundary fixtures. The `sigma`
argument to `sensitivity` is unused by this fixed LUT path.

Git history places the last LUT change at
[f8dc6bfcd5f641051cce558fccc65fe534bc9e7b](https://github.com/gfxdisp/ColorVideoVDP/commit/f8dc6bfcd5f641051cce558fccc65fe534bc9e7b),
dated 2024-01-04, with a message identifying a castleCSF2 fit of that date.
This review did not establish the exact separate castleCSF generator commit or
reproduction script. Preserve the bundled LUT bytes and this provenance;
regenerating with current castleCSF can change the reference.

The original repository's [LICENSE][license] is ordinary MIT, copyright 2024
Graphics and Displays group. No separate license notice was found for the
bundled LUT or base-model JSON in the pinned tree. The separate official
[castleCSF repository](https://github.com/gfxdisp/castleCSF) also declares MIT.
Any future vendoring should retain the original notices and data attribution;
this review vendors no code, tables or papers. Paper redistribution permissions
are distinct from the implementation's license. Upstream `third_party/`
includes additional provenance, so copying that entire directory would need
its own notice review.

## Mathematical stages that define the reference

### Display and opponent conversion

[display_model.py][display] first converts encoded RGB to absolute linear
emitted/reflected light, using the selected EOTF, luminance peak, contrast or
black level, ambient illuminance, reflectance and exposure. Input primaries
come from the selected color-space JSON. It then applies the RGB-to-XYZ,
XYZ-to-LMS2006 and D65 opponent transforms to produce achromatic, red-green
and yellow-violet channels. This is not a luma-only metric or an arbitrary
RGB-distance calculation. The CSF ultimately uses the reference's adaptation
luminance, so swapping reference and test need not preserve a score.

Relevant exact display distinctions include:

- sRGB linearizes values before mapping through peak and black luminance;
  ambient reflection adds `E_ambient * k_refl / pi`.
- PQ converts to absolute luminance, clips to the display's allowed range and
  adds black/reflected light. Linear HDR input must already contain absolute
  colorimetric values; RGB white 1 means approximately 1 nit, not display white.
- HLG includes an OOTF, luminance-dependent system gamma and an ambient-light
  correction. It cannot share the sRGB or PQ decoding path by relabeling input.
- Input display resolution and image dimensions are independent. Display
  geometry sets pixels/degree; a smaller image does not automatically expand
  to the entire screen. Resizing and chroma reconstruction are part of input
  provenance, not changes to conceal inside the metric.

For a fixed oracle case, record primaries, EOTF, range, chroma treatment, display
peak/black, ambient light, reflectance, exposure and pixels/degree. Named
display presets are useful, but their defining JSON must also be pinned.

### Temporal filtering and sequence boundaries

The [prediction driver][metric] treats **one total frame as an image**, even
if a positive FPS was supplied: it bypasses temporal filtering and uses only
three opponent channels. Two or more frames use four channels: three
sustained opponent channels and one transient achromatic channel. A repeated
still sequence therefore uses a different processing contract from a single
still.

For video, FIR length is `2 * ceil(0.250 * FPS / 2) + 1`, always odd. The three
sustained frequency responses use the calibrated exponential response; the
transient response is centered around 5 Hz. Each channel's taps are obtained
with an inverse real FFT and `fftshift`. The read loop applies the resulting
filter to the current frame and the preceding `N-1` frames. Thus the default
processing schedule is causal, despite the centered impulse-response taps;
re-centering it with future frames would change timing and boundaries.

Default `replicate` fills the missing prefix with the first frame. There is no
suffix extension or extra end-of-stream output in the reference loop. Optional
`symmetric` creates prefix samples from later frames and needs initial
read-ahead; v0.5.7 ping-pongs indices for videos shorter than the filter.
The helper's exact indexing should be fixture-pinned rather than replaced by a
library's generic mirror mode. History belongs to one sequence and must reset
at sequence start, not at arbitrary host batches. FPS must be positive for
multi-frame array input; the source accepts integer or floating frame rate.

Memory is dominated by two three-channel full-resolution temporal histories
plus the current temporal channels and pyramids. The reference processes
blocks sized by available memory and keeps per-frame scalar features for the
whole clip. Its default score can instead use a fixed history ring and a
running scalar accumulator, as derived below. That is a mathematical planning
observation, not a measured optimization or a proof of identical float rounding.
Symmetric startup, optional feature export, heatmaps, temporal resampling and
training add separate retention or latency requirements.

### Spatial pyramids and background contrast

[lpyr_dec.py][pyramid] builds a decimated Gaussian/Laplacian pyramid with
separable five-tap reduction, default kernel parameter 0.4, and ceil-halving
dimensions. Expansion inserts zeros, supplies custom edge values, filters and
uses the requested original level size. Intermediate band amplitudes are
scaled differently from the highest band and final baseband. The number of
bands depends on both minimum image dimension and pixels/degree; it is not
the fixed six-level SSIMULACRA 2 pyramid.

Default Weber contrast divides each image's bands by its own expanded
coarser sustained achromatic background, clamped below at 0.01. The
baseband instead uses an image-wide mean of its clamped background; dividing
it pointwise by itself would erase low-frequency differences. The contrast
code caps positive ratios at 1000; it does not symmetrically cap both signs.
The CSF background supplied later is the logarithm of the reference
background. Baseband spatial frequency is explicitly evaluated at 0.1 cpd.

The static source permits a baseband-only path at minimum dimension 2 or 3.
Dimension 1 leads to problematic level-count arithmetic. This is not a
runtime-verified supported-size statement: establish and document the accepted
minimum with the original oracle before publishing an API guarantee.

**Unresolved upstream edge issue:** [lpyr_dec.py line 206][pyramid-parity]
tests `x.shape[-2]` in the horizontal, columns-edge branch, although that index
is the input height. Odd rectangles with different width/height parity may
therefore differ from a conventional symmetric convolution. This observation
was not reproduced by execution. Add targeted fixtures, then choose an explicit
upstream-compatible or corrected-version policy; do not silently fix the
reference while describing the result as exact compatibility.

### Sensitivity, masking and difference response

For each spatial band/channel, the [metric][metric] evaluates the pinned CSF at
that band's frequency, at temporal frequency 0 or 5 Hz, using the reference
background and the configured sensitivity correction. Non-baseband contrasts
are multiplied by sensitivity and per-channel gains. Default mutual masking
uses the minimum test/reference absolute contrast, spatial Gaussian spreading
when the band is large enough, per-channel exponents and a learned 4 by 4
cross-channel matrix. The matrix's stored values are base-2 logarithms of
positive weights; transposing it changes the model.

The masked difference response raises the absolute contrast difference to
the calibrated power and divides by one plus masking. Soft saturation then
applies using `10^d_max` as the asymptote. The baseband follows a distinct
branch: absolute contrast difference times sensitivity, without this mutual
masking/saturation. Preserve that explicit branch and its substantial
channel-dependent pooling weights. Small bands also skip the Gaussian masking
spread when padding would be invalid; that is a numerical policy to test.

### Pooling, image correction and JOD

Do not substitute textbook p-norms indiscriminately. The pinned source loads
the exponents as tensors, so its `lp_norm` calls the differentiable
`safe_pow` helper for both the inner powers and outer root. Define
`g_p(x) = (x + 1e-5)^p - (1e-5)^p`. Its normalized reduction is
`g_(1/p)(sum(g_p(x_i)) / count)`; the unnormalized reduction omits division by
count. These tiny corrections affect tight comparisons, especially near zero.

The default score follows this order:

1. Reduce pixel differences within each band/channel/frame with normalized
   exponent 2.
2. Multiply channel weights and final-baseband weights, then reduce spatial
   bands with unnormalized exponent 4.
3. Reduce the three or four channels with unnormalized exponent 4.
4. For one total frame, multiply this result by `image_int =
   0.577918291091919`. For video, use a normalized exponent-2 reduction over
   all frames. The still correction comes after masking/saturation and
   channel pooling, before JOD conversion.
5. Convert the resulting nonnegative `Q` once to JOD.

Specifically, with `a=0.0439569391310215` and
`e=0.9302042722702026`, the final transform is linear for `Q <= 0.1`:
`JOD = 10 - a * 0.1^(e-1) * Q`. Above that threshold it is
`JOD = 10 - a * Q^e`. The branches meet at 0.1. There is no final [0,10]
clamp. Higher is better, identical valid inputs should produce 10 exactly,
and sufficiently severe differences can produce negative scores. These are
the original metric's contracts, distinct from this repository's existing
100-point or zero-distance identity conventions.

For score-only default video operation, let `u_f` be the result after step 3.
Maintain a frame count and `A = sum_f g_2(u_f)`; at the end compute
`Q = g_(1/2)(A/count)` and apply JOD. This bounds scalar feature memory
independently of clip length, alongside the FIR ring and frame workspaces.
Reproducing the scalar's reduction order/precision still needs a tolerance
study. Averaging per-frame JOD, averaging per-frame still-image predictions,
or moving the JOD power inside the temporal mean produces a different metric.

## Oracle practicality and validation requirements

The practical independent oracle is the **pinned official Python/PyTorch
implementation**, invoked directly on arrays with explicit configuration.
MATLAB is an optional front end to that same implementation. Its wrapper
downcasts doubles to singles and parses the CLI, whose ordinary output uses
four decimal places. That loses precision unnecessarily for golden-score and
stage comparisons. A future runner should preserve full-precision JOD,
`Q_per_ch`, actual loaded configuration hashes and environment provenance.

The Python package declares NumPy, SciPy, PyTorch, torchvision, imageio,
matplotlib, FFmpeg packages, PyEXR, einops and huggingface_hub. Normal video
CLI use additionally needs working FFmpeg/media dependencies. The base
hand-calibrated metric uses bundled tables; learned `cvvdp-ml-*` variants
introduce model download/checkpoint requirements and must be excluded from
this target. MATLAB wrapper use also needs a functional conda activation shell
and recent MATLAB syntax/support; Octave compatibility is not established.

Environment inspection found no `matlab`, `octave` or `conda` on PATH and no
importable `torch`, `torchvision` or `pycvvdp` in the active Python. NumPy and
SciPy were discoverable. No dependency installation was attempted. Accordingly
this review establishes source requirements, not a working oracle or measured
numeric tolerances.

The later implementation plan should include these independently checked cases:

- Exact identity for dark/bright SDR, HDR, stills and temporal sequences;
  finite output, expected direction, and a negative-JOD severe-difference case.
- Constant color shifts, luminance changes, saturation and uniform patches to
  exercise opponent conversion and the baseband separately.
- Impulses, flicker, moving edges, one-frame versus two-frame clips, lengths
  below/equal/above the temporal filter and multiple fractional frame rates.
- Prefix and final-frame impulses to detect timing or flush changes;
  replicate and symmetric startup, reset and chunk-size invariance.
- Odd/even and mixed-parity rectangles, tiny levels, gradients, boundary
  impulses, and spatial frequencies/luminance near or outside LUT limits.
- Full-precision per-stage/background/band/channel features against the
  original, with CPU/CUDA rounding differences measured rather than assumed.
- A pinned color/video corpus with original-author scores and subjective
  correlation checks under recorded display conditions. A peer-only comparison
  or existing SSIMULACRA 2 CID22 tolerance does not validate ColorVideoVDP.

Preserve an independent original-reference arm even when a peer is useful for
throughput. Defer saliency, learned regressors, heatmaps and arbitrary
frame-rate conversion unless explicitly included in the selected first target.
The principal remaining choices are the declared upstream revision, exact
numeric tolerances, mixed-parity edge policy, supported geometry/FPS limits,
and whether optional symmetric startup belongs in the initial API.

[matlab-readme]: https://github.com/gfxdisp/ColorVideoVDP/blob/2a268bce8d56e2f3abde46df3927d8a633707a24/matlab/README.md
[matlab]: https://github.com/gfxdisp/ColorVideoVDP/blob/2a268bce8d56e2f3abde46df3927d8a633707a24/matlab/cvvdp.m
[parameters]: https://github.com/gfxdisp/ColorVideoVDP/blob/2a268bce8d56e2f3abde46df3927d8a633707a24/pycvvdp/vvdp_data/cvvdp_parameters.json
[lut]: https://github.com/gfxdisp/ColorVideoVDP/blob/2a268bce8d56e2f3abde46df3927d8a633707a24/pycvvdp/vvdp_data/csf_lut_weber_fixed_size.json
[displays]: https://github.com/gfxdisp/ColorVideoVDP/blob/2a268bce8d56e2f3abde46df3927d8a633707a24/pycvvdp/vvdp_data/display_models.json
[colors]: https://github.com/gfxdisp/ColorVideoVDP/blob/2a268bce8d56e2f3abde46df3927d8a633707a24/pycvvdp/vvdp_data/color_spaces.json
[license]: https://github.com/gfxdisp/ColorVideoVDP/blob/2a268bce8d56e2f3abde46df3927d8a633707a24/LICENSE
[changelog]: https://github.com/gfxdisp/ColorVideoVDP/blob/2a268bce8d56e2f3abde46df3927d8a633707a24/ChangeLog.md
[csf]: https://github.com/gfxdisp/ColorVideoVDP/blob/2a268bce8d56e2f3abde46df3927d8a633707a24/pycvvdp/csf.py
[metric]: https://github.com/gfxdisp/ColorVideoVDP/blob/2a268bce8d56e2f3abde46df3927d8a633707a24/pycvvdp/cvvdp_metric.py
[pyramid]: https://github.com/gfxdisp/ColorVideoVDP/blob/2a268bce8d56e2f3abde46df3927d8a633707a24/pycvvdp/lpyr_dec.py
[pyramid-parity]: https://github.com/gfxdisp/ColorVideoVDP/blob/2a268bce8d56e2f3abde46df3927d8a633707a24/pycvvdp/lpyr_dec.py#L206
[display]: https://github.com/gfxdisp/ColorVideoVDP/blob/2a268bce8d56e2f3abde46df3927d8a633707a24/pycvvdp/display_model.py
