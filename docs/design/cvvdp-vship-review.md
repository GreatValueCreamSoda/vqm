# Vship CVVDP implementation review

Date: 2026-11-05. Scope: read-only source research for planning an independent
CVVDP implementation in VQM. No implementation was copied, no reference checkout
was changed, and no build, metric execution, GPU test or benchmark was run.

## Sources and limits

- VQM pins Vship in `tools/references.lock:12` to
  `91d03ec2f2e03e4817b64655a5ea58ab7ab3d3ba`, from
  <https://codeberg.org/Line-fr/Vship>. The local `.reference/Vship` HEAD matches
  this pin and `git status --short` is empty. Its commit date is
  `2026-10-19T12:49:19+02:00`; the subject names a CVVDP inverse-transform
  Nyquist-term fix for issue 33. The checkout is shallow, so this review cannot
  reconstruct release history or assign a source introduction date.
- `Makefile:22-24` identifies **Vship 5.1.2**, not a ColorVideoVDP model version.
  Both HIP/CUDA and Vulkan CVVDP implementations are present. Unless otherwise
  qualified, peer file references below are relative to `.reference/Vship/` at
  the pin above. [Pinned source tree](https://codeberg.org/Line-fr/Vship/src/commit/91d03ec2f2e03e4817b64655a5ea58ab7ab3d3ba).
- The direct comparison used the original ColorVideoVDP **v0.5.7** snapshot,
  `2a268bce8d56e2f3abde46df3927d8a633707a24`, commit date
  `2026-08-02T08:58:29+02:00`, obtained read-only by the parallel original-source
  review. Original paths below are relative to that snapshot, not Vship.
  [Pinned original source](https://github.com/gfxdisp/ColorVideoVDP/tree/2a268bce8d56e2f3abde46df3927d8a633707a24).
- Vship's `README.md:185-186` points to ColorVideoVDP's moving main branch;
  no upstream commit, model version, table generator, calibration date, parity
  corpus, or score tolerance was found in its CVVDP sources or documentation.
  Matching parameters alone does not establish implementation equivalence.
- `LICENSE:1-25` is headed MIT NON-AI and contains extra restrictions;
  `LICENSE:17-19` exempts the public API/color headers and compiled binaries
  from its training restriction. This review follows VQM's stricter repository
  rule: mechanisms are described with provenance, and Vship implementation code
  and tables are not copied into VQM. Use the original project as the normative
  algorithm/data source and Vship only as an independently executed comparison.

## Model and defaults

Vship implements a fixed base CVVDP model. It exposes display configuration but
does not expose alternate masking, contrast, pooling, CSF, temporal-filter or ML
model selection. Its numerical calibration agrees with the original base model
named `cvvdp_mult_mutual`, calibration date `10/01/2024`, `weber_g1` contrast,
`mult-mutual` masking, `weber_fixed_size` CSF and soft difference clamp
(`pycvvdp/vvdp_data/cvvdp_parameters.json:2-20,42,67-71`). The original review
also found these base coefficients unchanged between v0.4.2 and v0.5.7, so this
does not identify which upstream revision Vship originally targeted.

Read-only numerical comparison found all **22** scalar/vector parameters in
`src/HIP/cvvdp/parameters.hpp:5-64` equal to the corresponding values in original
v0.5.7's JSON after parsing their decimal values. Vulkan distributes those
parameters between `parameters.hpp`, `shaders/maskingModel.slang` and
`shaders/CSFModule.slang`. Notable contracts are spatial exponent 2, temporal
exponent 2, spatial-band and channel exponents 4, and image integration factor
`0.577918291091919`. The original JSON must remain the source for new generated
VQM model data; Vship is evidence of a peer's fixed configuration.

The VapourSynth and FFVship defaults are **standard_fhd**, no display resize,
no distortion-map output, with temporal behavior enabled:
`doc/CVVDP.md:11-20`, `src/Vulkan/cvvdp/vapoursynth.hpp:149-175`, and
`src/ffvship_utility/ffmpegmain.hpp:58-60`. The generic C init structure has no
implicit default values: callers supply its fields (`src/VshipAPI.h:153-162`).
In particular, a null display name becomes an empty key rather than
`standard_fhd` (`src/VshipLib.cpp:174-183`). Original v0.5.7's Python constructor
instead defaults to **standard_4k** (`pycvvdp/cvvdp_metric.py:109`). A parity
driver must specify the display explicitly in every arm.

The peer's standard_fhd display is 1920x1080, 24 inches, 0.6 m viewing distance,
200 cd/m² peak, contrast 1000 and ambient 250 lux. Default reflectance is 0.005
and exposure is 1 (`src/Vulkan/cvvdp/display_models.hpp:14-24,100-108`). It derives
pixels per degree from the physical geometry, reflected luminance from ambient
illuminance and reflectance, and black level from peak/contrast (`:145-160`).
Its standard_4k is 3840x2160, 30 inches, 0.7472 m with the same peak/contrast/
ambient (`:40-48`). HDR profiles set an HDR flag but do not themselves select
the input transfer function (`:49-99` and `colors.hpp:20-41`).

Custom display input may be a JSON string or file. The parser only accepts the
documented subset, rejects unknown selected-display fields, and collapses
BT.2020-PQ/HLG to HDR and BT.709-linear/sRGB/SDR to the SDR flag
(`display_models.hpp:166-182,214-267`). It is not a general upstream display
configuration implementation. The zoom profile's description says 4000 cd/m²
while the actual numeric peak is 10000 (`:90-98`); use values, not descriptions,
when fingerprinting a comparison arm.

## Actual frame and sequence semantics

This is a **true temporal metric**, not framewise CVVDP with scores averaged
afterward. Each handler owns two histories, three DKL-like color planes per
stored frame, and produces four temporally filtered channels: sustained
luminance, sustained red-green, sustained yellow-violet, transient luminance.
The history length is `2*ceil(0.25*fps/2)+1`, always odd for valid finite fps
(`src/Vulkan/cvvdp/temporalFilter.hpp:39-79,94-142`).

Temporal filters are built once on the CPU by sampling four frequency response
functions and applying a small handwritten inverse real DFT. There is **no FFT
library dependency** and no runtime per-frame FFT (`temporalFilter.hpp:7-28,
47-79`). Runtime filtering is a direct FIR convolution over the ring, one
256-thread dispatch per image. It uses current and past frames; missing early
history repeats the oldest stored frame (`shaders/temporalFilter.slang:1-3,
31-52`). There is no future-frame dependency, pending-frame queue, tail flush or
end-of-sequence operation in this peer API. There is temporal lag from applying
the centered filter to a past-only window, as in the original default replicate
path; zero future-frame dependency does not mean zero filter delay.

The first history entry has a special bypass: sustained channels equal the
current frame and transient luminance is exactly zero
(`shaders/temporalFilter.slang:13-20`). This path applies at every reset. This is
also what makes a newly initialized/reset handler a practical still-image arm.

For each evaluated frame, Vship spatially pools each band/channel using a
normalized L2 norm. It then sums fourth powers over bands/channels and takes
the fourth root on the CPU. It accumulates the square of that latent frame
quantity, then maps the RMS across evaluated frames to JOD. Exactly one
evaluated frame receives the image integration factor; two or more evaluated
frames do not (`src/Vulkan/cvvdp/main.hpp:94-128,430-439`). Thus:

- The returned value is the **ordered prefix score**, not independent frame JOD.
  The last returned value is the video score (`doc/CVVDP.md:109-112` and
  `src/FFVship.cpp:613-616`). Averaging those JOD values would be incorrect.
- Repeated Compute calls on a still pair grow a video sequence. A fair still
  benchmark must reset sequence state between calls or explicitly use a
  still-only interface in a future VQM implementation.
- Score-only reset preserves temporal history but resets count/sum; its first
  evaluated result nevertheless takes the image factor, even if history was
  preloaded. This follows `main.hpp:430-443` and is a peer semantic to test,
  not a contract to inherit silently.
- `Vship_Reset` is documented in `src/VshipAPI.h:205` as preserving score
  accumulation, but implementation `src/VshipLib.cpp:319-327` calls
  `flushTemporalRing`, which clears **both** history and accumulated score
  (`main.hpp:446-450`). `Vship_ResetScore` clears only score accumulation.
  A comparison harness must follow implementation behavior.

The public API supports one ordered stream per handler, explicit fps, two input
colorspaces, optional map output, map-resolution query, score reset, full reset,
and loading history without adding a score (`src/VshipAPI.h:153-212`;
`src/VshipLib.cpp:342-374`). It consumes CPU plane pointers/strides and blocks
until GPU completion (`main.hpp:395-439`); there is no public resident-input or
split Submit/Finish interface. FFVship forces one GPU worker, one CPU worker and
every frame for CVVDP, rejecting custom frame-index lists
(`src/FFVship.cpp:316-324`). It forces fps=1 for one-frame inputs and falls back
to 1 for invalid video fps (`:447-453`). The VapourSynth wrapper's invalid-fps
fallback is 60 (`vapoursynth.hpp:191-195`), and `disableTemporal` clears state
after each frame (`:76-78`). These wrappers are not interchangeable defaults.

## GPU dataflow and organization

The following are **Vship's mechanisms**, observed in the read-only peer;
they are not a proposed VQM implementation or measured optimal choices.

| Phase | Peer mechanism and provenance |
| --- | --- |
| Inputs and display | Two general converters upload and convert inputs to linear RGB. If shrinking to display, bicubic resize happens before history storage; if enlarging, it happens after temporal filtering, reducing history memory. Display photometry, primaries-to-XYZ and XYZ/LMS-to-DKL conversion precede storage (`main.hpp:278-363`; `colors.slang:1-55`). |
| Display fit | Optional aspect-preserving fit inside display resolution, with integer output dimensions (`main.hpp:184-197`). Resize is separable cubic Hermite interpolation through two transposed passes, 256 threads per workgroup (`resize.hpp:6-31`; `shaders/resize.slang:14-27`). |
| Temporal | Independent source/distorted 3-plane rings, four filters, direct FIR; float2 arithmetic pairs sustained/transient luminance and the two chromatic channels (`shaders/temporalFilter.slang:23-52`). |
| Spatial decomposition and CSF | For every non-base band, four sequential channel dispatches process both images using float2. Each 16x16 block fuses 5-tap pyramid reduction, expansion, contrast construction, and CSF weighting, keeping 23x23 inputs and 10x10 reduced samples in shared memory (`shaders/fromLpyrToCSF.slang:3-33,56-137`). Channel 0 writes background luminance used by the next channel dispatches (`lpyr.hpp:7-36`). |
| Masking | One 16x16 dispatch per non-base band handles all four channels, blurs the pointwise minimum absolute reference/test contrast, mixes cross-channel mask weights, computes masked differences and soft saturation, and writes weighted differences over the former reference planes (`shaders/maskingModel.slang:53-158`). The 32x32 shared tile is reused for four channel blurs. |
| Baseband | Reduce luminance to a mean separately for each image; divide each channel by its own scalar background; compute CSF-weighted absolute difference with baseband/channel weights (`main.hpp:75-84`; `shaders/lpyr.slang:94-109`; `shaders/maskingModel.slang:165-172`). |
| Spatial score pooling | Repeated 1024-thread shared-memory tree reductions give normalized L2 per band/channel; only at most 14x4 floats return to CPU (`pooling.hpp:5-41`; `shaders/pooling.slang:1-38`; `main.hpp:97-105,220`). |
| Final score | CPU uses a double sum for fourth-power band/channel aggregation, but returns the frame aggregate as float; temporal sum is double, and final JOD conversion accepts/returns float (`main.hpp:117-128,137,430-439`; `distmap_specifics.hpp:5-11`). |
| Optional map | Merge four channels with a fourth-power mean, reconstruct pyramid, map to `1-JOD/10`, and download float32 map when requested (`distmap_specifics.hpp:39-65`; `shaders/distmap_specifics.slang:22-35`). Map semantics should be checked independently of scalar score. |

Spatial band count depends on resolution and pixels per degree, with 0.2 cpd
minimum target and hard cap 14; each reduction rounds dimensions up
(`csf.hpp:20-32`; `lpyr.hpp:91-103`). Vship allocates 9 pyramid planes per image:
four current/final channels, background/scratch, and four CSF-weighted contrast
channels (`main.hpp:24-49`). The Vulkan backend allocates reusable scratch once,
records the spatial/pooling/map command buffer once, and re-records input and
optional readback command buffers per call. It executes chained command buffers
with semaphores and waits on a fence (`main.hpp:149-158,202-256,402-423`). The
recorded final stage always requests map reconstruction (`:253` passes true),
even when the caller does not request a download; this differs from the HIP
backend, which reconstructs only for non-null map output
(`src/HIP/cvvdp/main.hpp:149-155`).

HIP/CUDA uses two streams for the per-image work, joins them before the fused
spatial processing, and uses asynchronous temporary allocations/frees during
each computation (`src/HIP/cvvdp/main.hpp:31-47,78-98,127-158`). Vulkan shares the
scientific pipeline but has different lifetime/submission machinery. VQM should
not inherit either queue or allocation strategy without its own measurements.

## CSF data, precision and dependencies

The embedded CSF comprises two 32-value axes and four 32x32 log-sensitivity
planes: luminance backgrounds **0.005..10000 cd/m²**, spatial frequency
**0.1..64 cpd**, sustained Y/RG/YV at 0 Hz and transient Y at 5 Hz
(`csf_lut_weber.hpp:5-76`). Read-only parsing found all **4160** numerical values
identical to original v0.5.7's `csf_lut_weber_fixed_size.json`, flattening fields
`o0_c1`, `o0_c2`, `o0_c3`, `o5_c1` in that order. The original JSON also declares
`ge_sigma=1.5`. This identifies the data content, not Vship's table-generation
provenance.

The peer interpolates log sensitivity over log spatial frequency once on CPU
for every actual band/channel, uploading only `(1+4*bands)*32` float32 values.
It interpolates over log background luminance and exponentiates on GPU
(`csf.hpp:40-72`; `shaders/CSFModule.slang:5-16`). Frequency interpolation permits
extrapolation; luminance interpolation clamps to the endpoints. Background
luminance is reference-derived for sensitivity even though the two images have
their own Weber contrast denominator (`fromLpyrToCSF.slang:100-132`).

Source-table SHA-256 for both identical HIP and Vulkan headers:
`3c714ae572485ffd10c22a4835b5f8c3027b363e754f87cdcd333c380cce8589`.
Original v0.5.7 JSON SHA-256:
`df39c869c0c53a9a2cb757811ecaf4ce0d44b4151f5ff9590f5bfe1ede940c18`.
The file hashes differ because the wrappers/formats differ; the parsed numbers
were equal without tolerance. No new model file should be generated from the
peer header; generate from the independently pinned original data.

All CVVDP GPU planes, kernels and reduction values inspected are float32, with
explicit FMA in several stages. Vship does not need float64 shader support for
this metric; `src/VshipAPI.h:87-88` also marks float16 and float64 optional.
The general Vulkan peer runtime uses device addresses and several optional
storage/indexing features; this is not evidence that those features are
mathematical CVVDP requirements (`src/VshipAPI.h:78-96`). Its pooling kernel's
1024-thread workgroups require a suitable device workgroup limit and should not
be copied as VQM's portability floor.

Build dependencies are C++/make plus HIP or CUDA, or Vulkan headers/loader for
the Vulkan backend. Rebuilding the Vulkan shaders uses Slang SPIR-V 1.3;
precompiled shaders are checked in (`README.md:46-63`; `Makefile:249-256`).
FFVship additionally uses FFMS2/pkg-config. CVVDP has no runtime Python,
PyTorch, FFT, external CSF service, or neural-network weight dependency in this
peer implementation.

## Confirmed differences from original v0.5.7

These are source comparisons, not measured score/error results. They prevent
using Vship as a sole correctness oracle even though the model numbers match.

| Area | Difference and exact provenance |
| --- | --- |
| Temporal response sampling | Vship uses frequency `fps*i/(2*M)` for M real-spectrum entries (`temporalFilter.hpp:47-50`); original uses an inclusive 0..fps/2 linspace, spacing `fps/(2*(M-1))` (`pycvvdp/cvvdp_metric.py:1032-1034`). The frequency grids therefore differ. |
| First video frame | Vship bypasses filtering and sets transient to zero (`shaders/temporalFilter.slang:13-20`). Original knows whether the complete input is an image before evaluation; video mode filters all frames with replicated history (`cvvdp_metric.py:335-344,510-513,559-565`). |
| Numerical powers | Vship's masking and pooling use ordinary powers (`shaders/maskingModel.slang:119-143`; `shaders/pooling.slang:13-35`; `main.hpp:123-126,431-437`). Original applies epsilon-adjusted `safe_pow` in masking and when tensor exponents select the formula norm implementation (`cvvdp_metric.py:77-84,860,1019-1023`). It is a difference near zero, not merely a floating-point backend change. |
| Phase uncertainty blur | Vship uses radius 8 / sigma 3, renormalizing cropped valid samples at boundaries and applying it on every non-base band (`gaussianBlur.hpp:5-24`; `shaders/maskingModel.slang:56-103`). Original uses a 13x13 sigma-3 normalized Gaussian with reflect padding (`pycvvdp/utils.py:110-134`) and skips the blur if either dimension is <=6 (`cvvdp_metric.py:936-944`). |
| Defaults | Vship standard_fhd versus original standard_4k; custom display handling and SDR/HDR branches also differ in available configuration (`display_models.hpp:9-24,214-267`; original `cvvdp_metric.py:109`). Color/input metadata must be explicitly aligned before attributing any score gap to the metric core. |
| Optional models and padding | Vship fixes the base masking/contrast/filter/CSF and only implements replicated past history. Original supports multiple model/configuration choices and symmetric initial padding with read-ahead (`cvvdp_metric.py:177-212,515-537`). A conforming initial target should name the base configuration rather than promise every upstream option. |

Other sensitive parity surfaces remain to test: pyramid edge treatment on odd
dimensions, extra contrast scaling of nonzero bands, baseband averaging before
or after luminance clamp, CSF extrapolation, SDR/PQ/HLG photometry, bicubic
downscaling, map normalization/reconstruction and sequence reset boundaries.
They are not established equal by this review.

## Memory and validation implications for VQM planning

At an unchanged resolution, the two rings alone store **6*L*W*H float32
values**. The final scratch arena holds approximately **18*(4/3)*W*H** floats,
with exact pyramid dimensions rounded up, plus converter and optional resize
space (`main.hpp:202-213`; `temporalFilter.hpp:107-115`). Analytic 3840x2160
examples, excluding small tables/host readback, alignment and driver overhead:

| fps | L | Two histories | Histories plus approximate final scratch |
| --- | ---: | ---: | ---: |
| 24 | 7 | 1.393 GB | 2.190 GB |
| 60 | 17 | 3.384 GB | 4.180 GB |
| 120 | 31 | 6.171 GB | 6.967 GB |

These are allocation arithmetic, not memory measurements. The approximation in
`doc/CVVDP.md:116` is not the exact current Vulkan allocation formula. Explicit
memory queries, fps validation and pre-allocation failure checks are important
for VQM's plan. Still-only paths can avoid history entirely when mode is known
at initialization, as the original does.

Peer test evidence is narrow. `test/libvshipTestIdentity.cpp:364-389,436-468`
checks one identical pair per many input-format/resize/crop combinations and
expects CVVDP exactly 10, using a newly initialized 48 fps HDR handler. It does
not test a differing sequence against the original. `test/speedtest.cpp:46-68`
also only checks identity for CVVDP. README throughput tables
(`README.md:133-173`) identify one laptop and one clip but are peer claims under a different
protocol, not VQM performance evidence. No quantitative upstream parity corpus
or published tolerance was found in the pinned tree.

For a future implementation, retain three independent comparison roles:
original CPU/PyTorch (normative), original GPU where available (backend drift),
and Vship HIP/Vulkan (peer behavior/performance). Test raw per-band/channel
features and latent pooling quantities before JOD, not just the final rounded
score. Cover still mode, identical pairs, impulse/step/flicker sequences,
1/2/L-1/L/L+1 frame lengths, odd sizes and tiny bands, multiple fps values,
explicit display profiles, scene resets/preload, long accumulation, and optional
maps. Keep temporal boundary rules and still/video mode separate in the public
contract. No performance recommendation or pass/fail tolerance is established
by this source review.
