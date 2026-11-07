# VMAF

VQM exposes two Vulkan sequence pipelines: the floating-point v0.6.1 model in
`vmaf.h`, and the standard SDR v1.0.16 3H model in `vmaf_v1.h`. Both evaluate
ordered frame pairs and return per-frame scores and features. They accept
pre-uploaded images, strided host views and caller-owned Vulkan buffers as well
as owning `Image` inputs. Borrowed buffers must remain alive and unmodified
until `Push` returns; the caller owns and synchronizes borrowed Vulkan devices
and queues as described in [`vqm_vulkan.h`](../include/vqm_vulkan.h).

```cpp
#include <vmaf_v1.h>

// ref and dist are decoded SDR frames with the same geometry. The explicit
// mode accepts different supported source formats for each image or frame.
vqm::VmafV1Config config;
config.size = ref.size;
config.chroma_sampling = vqm::VmafChromaSampling::Yuv420;
vqm::VmafV1Pipeline pipeline(config);

if (auto ready = pipeline.Push(ref, dist)) {
    // ready->index and ready->score identify an emitted frame.
}
// Push the remaining pairs in sequence order, collecting every ready frame.
if (auto last = pipeline.Flush()) {
    // Collect the final frame, including the only frame of a still image.
}
auto mean = pipeline.PooledScore(); // arithmetic mean, or nullopt if empty
```

V0's first `Push` emits frame zero immediately; later frames need one frame of
look-ahead for motion2. With `temporal_weighting = false`, every V0 `Push`
emits its own frame and `Flush` emits none. V1's first `Push` emits nothing:
frame zero receives the first reference transition for a multi-frame sequence,
and is emitted after the second `Push`. A single-frame V1 sequence receives
zero motion on `Flush`.
For both pipelines, `Flush` completes the sequence and `Reset` starts another.
The decoded inputs must remain valid during `Push`. Every resident call prepares
both images; the retained previous reference is used for the motion feature.

## Inputs and precision

Both pipelines accept the SDR planar image formats supported by VQM, including
RGB, YUV and grayscale, integer and floating-point samples, and alpha. PQ and
HLG transfer functions are rejected because the implemented model is SDR.
The caller describes each source's color metadata and format; the same rules
and layout requirements used by SSIMULACRA 2 apply to the input.

`VmafChromaSampling::Yuv420` and `Yuv444` convert both images on the GPU to a
common BT.709 limited-range representation, even when the source is already
YUV. Decoding produces linear Rec.709, and encoding uses the display-referred
BT.709 transfer (gamma 2.4). Converted samples are rounded to 16-bit integer
codes and represented in the FP32 VMAF feature domain. The 4:2:0 chroma plane
uses a centered 2×2 box average. At odd right or bottom edges the last pixel is
replicated, yielding ceil(width/2) × ceil(height/2) chroma planes. These
explicit modes allow the two sources, and source formats across frames, to
differ as long as geometry stays fixed. Alpha is composited using the
configured `background`, default 0.5, following the same RGB-domain policy as
SSIMULACRA 2.

The default `Native` mode preserves compatible opaque integer coded YUV instead
of converting it. Native v0 also accepts coded grayscale and uses only luma.
Its first frame chooses native or converted coding for the sequence; a change
in native coding requires `Reset` or an explicit chroma mode so motion history
stays in one code domain. Native v1 requires each source format to match its
configured `input_format` throughout the sequence. When native v1 receives
another supported source format, it converts to 4:2:0. Existing coded YUV uses
8-bit bytes or 10-, 12-, or 16-bit words and is read as
`sample / 2^(bit_depth - 8) - 128`; this preserves the low-order input bits.

The two paths define different inputs to the model: explicit conversion can
change scores relative to `Native`, even for the same source. VMAF v0 uses
luma only, so choosing 4:2:0 or 4:4:4 in an explicit mode does not change its
chroma features; both modes request the common conversion. Native v0 streaming
uploads stage only luma when the pair uses the native code domain. Conversion
stages every plane, including mixed coded inputs and later coded inputs in a
sequence that started in the conversion domain. V1's
`encoded_bit_depth` defaults to the native input bit depth, or 16 for converted
inputs. Values 6–16 specify the original encode bit depth for CAMBI's
anti-dithering. Input and encoded geometry must match; encode-size overrides
are not yet implemented.

V1 needs each chroma plane to be at least 80×80. Its CAMBI path requires at
least one luma dimension of 216 pixels or more and all five scales to fit the
computed window. The effective CAMBI scale cannot exceed 1920×1080, and its
odd window cannot exceed 33 pixels. At or above 1920×1080 pixels by area, the
v1 model's high-resolution speedup halves both luma dimensions before the first
CAMBI scale. These restrictions are checked by the pipeline constructor.

The GPU feature stages use FP32. The conversion path quantizes its intermediate
to 16-bit codes; CAMBI deliberately uses a 10-bit integer code domain. The CPU
model predictors use double precision, as in libvmaf. V0 implements the float
v0.6.1 feature path for both shipped predictor tables. V1 targets
`vmaf_v1.0.16_3d0h.json`, whose reference uses integer ADM3 and motion3; VQM's
corresponding FP32 GPU features approximate those extractors. FP32 filters,
reductions and nonlinear operations can also differ in evaluation order and
rounding. Neither implementation promises bit-exact results. The v0.6.1
integer feature extractor and v1 phone, 4K-display, HFR and HDR model variants
are not implemented.

## v0.6.1 models

V0 exposes the two floating-point v0.6.1 predictor tables:

| Variant | Predictor | Canvas | Official float JSON |
| --- | --- | --- | --- |
| Standard 1080p, 3H | 211 support vectors | 1920×1080 | `vmaf_float_v0.6.1.json` |
| 4KTV, 1.5H | 262 support vectors | 3840×2160 | `vmaf_float_4k_v0.6.1.json` |

`VmafConfig::model` selects the table. The two generated constant sets live in
`src/metrics/vmaf/v0/vmaf_model_data.inc`. `tools/generate_vmaf_model.py --check`
verifies both against the pinned JSON and SHA-256 values. The upstream guide
describes the 1080p and 4KTV training and display assumptions.
[Pinned model guide](https://github.com/Netflix/vmaf/blob/4611e47a7330cd342498dcc130e1008efbeab6e4/resource/doc/models_v0.md)

The float and integer v0.6.1 JSONs contain identical slopes, intercepts, SVM
vectors, transform and clip. Their only meaningful difference is whether
`feature_names` selects the float or integer feature extractors. VQM runs the
float feature path, so loading an integer JSON would not produce the integer
extractor's scores. Exact integer-model parity requires implementing that
extractor; it is not a second predictor table.

NEG is a feature option, not another predictor. `VmafConfig::neg` sets
`adm_enhn_gain_limit = 1` and `vif_enhn_gain_limit = 1` for every VIF scale,
matching `vmaf_float_v0.6.1neg.json`. The 4K model has no official float NEG
JSON, but the integer 4K NEG JSON shares the same 262-vector predictor and only
adds those options; VQM exposes that combination as a derived 4K NEG mode.

The phone mode is the standard 1080 predictor's `score_transform`, enabled by
`VmafConfig::transform`. Netflix does not ship a 4K phone model and libvmaf's
`VMAF_MODEL_FLAG_ENABLE_TRANSFORM` is a no-op for the 4K JSON because it has no
`score_transform`. VQM applies the 1080 phone polynomial to 4K as a documented
derived extension; it is not an official Netflix 4K phone model. Because the
polynomial was calibrated on 1080 scores, 4K scores can saturate at 100 when it
is enabled.

`VmafConfig::temporal_weighting` defaults to `true` and selects the upstream
v0.6.1 feature order from the model JSON: ADM2, motion2, then VIF scales 0-3.
Setting it to `false` is a derived VQM mode, not an upstream model variant. It
forces motion2 to zero before prediction, skips the reference blur and motion
reduction passes, and emits each frame immediately because no look-ahead is
needed. This is useful when motion2 does not describe the content well, but it
is not expected to reproduce libvmaf scores: the predictor still expects the
six-feature layout and now receives a zero motion value. The benchmark flag
is `--vmaf-no-temporal`.

The `b`/`rb` v0.6.3 models are bootstrap collections for prediction confidence
intervals. Plain `b` resamples the training data and `rb` bootstraps prediction
residues. The upstream guide recommends `b`.
[Pinned confidence interval guide](https://github.com/Netflix/vmaf/blob/4611e47a7330cd342498dcc130e1008efbeab6e4/resource/doc/conf_interval.md)

`VmafConfig::fit` selects how the input geometry relates to the model canvas.
`VmafFitMode::Native` is the default and keeps the existing behavior: the
input is the feature geometry, so the caller controls any resampling.
`VmafFitMode::FitAndPad` scales each input uniformly to the largest size that
fits inside `VmafConfig::model_size`, centers it, and fills the remaining
canvas with the source's legal black code. A zero `model_size` uses the
selected model's canvas: 1920×1080 or 3840×2160. A 4:3 source therefore keeps
its aspect ratio and gains side bars; a 2.39:1 source gains top and bottom
bars. The input and model canvases must both be at least 17 pixels in each
dimension. If the source already equals the model canvas, the fit path is
skipped and the native path is used.

FitAndPad operates on coded luma, before VIF, ADM2 and motion. It is not a
color transform and does not change the input's range or matrix. The padded
area is part of the feature image: edges at the content boundary contribute to
the local filters, just as letterboxed pixels would in a pre-padded input.

## Shared implementation

The pipelines share the Vulkan runtime, resident-image handles, streaming uploads,
external-buffer binding, synchronization and readback. Both also use
`VmafLumaConverter` for native coded-plane conversion, `VmafSdrConverter` for
general SDR input, and `VmafMotion` for reference filtering and transition
differences. Changes in those components apply to both models; each pipeline
retains its own arena layout and feature orchestration.

ADM2 and ADM3 share the fused wavelet-transform and band-decoupling shader,
`src/metrics/vmaf/features/shaders/adm/vmaf_adm_dwt_decouple.comp`. The fusion first optimized
for v0 now also runs in v1, so both avoid separate distorted-band writes and
decoupling dispatches. The shader is compiled with enhancement-gain limits 100
and 1; v0 selects 100 normally and 1 for NEG, while ADM3 uses 1. Future
improvements to its common filtering and tiling therefore reach both models
while preserving that distinction. Their later ADM accumulation, masking and
score reductions remain model-specific.

V0 compiles an additional source-tiled variant of the decouple shader, which
stages a 34x34 luma neighborhood per image. ADM3 keeps the untiled variant.
Both fused paths consume the distorted detail bands in registers and reserve
no separate arena region for them.

V0's VIF and v1's chroma SpEED and CAMBI extractors have different algorithms and
remain separate. The recent cooperative covariance/Jacobi and CAMBI-window
optimizations benefit v1 only. Likewise, motion2 and motion3 share the transition
calculation but keep separate sequence timelines, boundary rules and predictors.
Feature or scheduling changes must retain each model's reference-validation
gates; sharing an implementation does not make its numerical rules interchangeable.

The [joint speed investigation](../bench/notes/vmaf/vmaf-speed-investigation.md)
records the later dependency scheduling, excluded ADM groups, motion tiling,
VIF shortcut and SpEED filtering/covariance changes, with measured alternatives
and both versions' correctness gates.

CAMBI's contrast shader stores its bounded 0–497 shared symbols in FP16 when
the logical device has enabled `shaderFloat16`, with the existing integer
shader as fallback. Those symbols are exact; histograms remain integer and
contrast/pooling arithmetic remains FP32. The global arena layout is unchanged.
The [FP16 study](../bench/notes/vmaf/vmaf-fp16.md) records why wider arithmetic
is still needed in VIF, ADM, motion and SpEED, including before accumulation.

## Reference and validation

Run these commands from the repository root after the local build and Python
setup. [Build and test](../CONTRIBUTING.md#build-and-test)

### Pinned libvmaf and sequence fixtures

`tools/references.lock` pins Netflix libvmaf 3.2.0 at commit
`4611e47a7330cd342498dcc130e1008efbeab6e4`. The reference build needs Meson,
Ninja, and C/C++ compilers. The build script copies the checkout to
`build/vmaf-src` and compiles it in `build/vmaf-build` with SIMD assembly and
embedded models disabled. The scalar build supports float and integer feature
extractors and loads model files by path. Use it as a correctness oracle.

```sh
python3 tools/fetch_references.py --only vmaf
python3 tools/build_references.py --only vmaf --jobs 8
python3 tests/make_vmaf_fixtures.py
.venv/bin/python -m unittest discover -s tests -p test_vmaf_oracle.py -v
```

The generator writes four synthetic 64×48 frames per input in planar YUV420 at
8, 10 and 12 bits. The 10- and 12-bit `low-bits` pairs differ in luma samples
while their upper eight luma bits match. The fixture manifest at
`build/fixtures/vmaf/manifest.json` records geometry, depth, frame count and
SHA-256 hashes.

The `vmaf_reference` driver calls libvmaf's public C API. Its `float` and
`integer` modes each report a score and six features per frame in CSV at
17-digit precision. Use the model file for the selected feature extractor.

```sh
build/vmaf-build/vmaf_reference float \
  build/vmaf-src/model/vmaf_float_v0.6.1.json \
  build/fixtures/vmaf/10bit-low-bits-ref.yuv \
  build/fixtures/vmaf/10bit-low-bits-dist.yuv 64 48 10

build/vmaf-build/vmaf_reference integer \
  build/vmaf-src/model/vmaf_v0.6.1.json \
  build/fixtures/vmaf/10bit-low-bits-ref.yuv \
  build/fixtures/vmaf/10bit-low-bits-dist.yuv 64 48 10
```

Both modes take raw little-endian YUV420 at 8, 10, 12 or 16 bits, with matching
even dimensions of at least 16×16 and equal, nonzero counts of complete frames.
The driver processes frames in order and flushes libvmaf before reading results
so the output includes the end-of-stream motion behavior. The native libvmaf
CLI at `build/vmaf-build/tools/vmaf` reports JSON values to six decimal places.
Use the CSV driver for tighter feature comparisons. Its separate `adm-debug`
mode reports the float ADM numerator and denominator at four scales and the
aggregate ADM2. Pass `-` in place of a model.

```sh
build/vmaf-build/vmaf_reference adm-debug - \
  build/fixtures/vmaf/8bit-moving-ref.yuv \
  build/fixtures/vmaf/8bit-moving-dist.yuv 64 48 8
```

VQM's C++ tests retain independent feature and frame-score oracles. Coverage
includes 4:4:4 and 4:2:0, low-order high-bit-depth differences, sequence
boundaries, CAMBI encoded-depth overrides, and borrowed buffers.

```sh
cmake --build --preset release --parallel 8
./build/vqm_test --require-gpu
ctest --preset release
```

### CID22 still-image comparison

`validation/vmaf_cid22.py` requires NumPy and Pillow from the local Python
setup. It converts the 4,292 annotated 512×512 pairs and scores each pair as an
independent one-frame sequence with zero motion. The reference uses the
v0.6.1 float model and the standard v1.0.16 SDR 3H model. Use `--limit 300`
for a shorter run, with the same limit on both commands.

```sh
python3 tools/fetch_references.py --only cid22
.venv/bin/python validation/vmaf_cid22.py --jobs 8
.venv/bin/python validation/vmaf_cid22.py --jobs 8 --vqm-v0 --vqm-v1 \
  --reference-scores artifacts/reports/vmaf-cid22/scores.csv \
  --out-dir artifacts/reports/vmaf-cid22-vqm
```

The candidate run requires `build/vqm_bench` and a working Vulkan device. It
runs one process per VQM model with `--score-only`. Use just
`--vqm-v0` or `--vqm-v1` to compare a single model. With `--reference-scores`,
the tool checks the previous run's input manifest, models, driver, annotations,
conversion and pair order before reusing its reference scores.

The conversion applies the BT.709 Y′CbCr matrix directly to sRGB-encoded 8-bit
RGB samples. It rounds to nearest, ties to even, in 10-bit limited range with
offset/scale 64/876 for Y and 512/896 for Cb and Cr. Chroma stays 4:4:4.
The tool ignores opaque alpha and rejects non-opaque alpha.

`inputs.csv` records geometry, paths and source/YUV SHA-256 hashes for the exact
bytes cached in `.cache/validation/vmaf-cid22-yuv/`. The full cache uses about
6.4 GiB. The VQM arms wrap those bytes in single-frame Y4M files in
`.cache/validation/vmaf-cid22-y4m/`, using about another 6.4 GiB. Use
`--prepare-only` to generate the input manifest without scoring.
`scores.csv` contains the per-pair scores. `summary.json` records correlations,
score differences and model, driver and input fingerprints.

To compare an implementation on the same inputs, pass `--vqm-scores PATH` with
one row per selected pair and `original`, `distorted`, and `v0` and/or `v1`
columns. To compare two optimization runs pair by pair, pass their score files
to the comparison tool. The default column is `vqm_v0`. Use
`--column vqm_v1` for v1 or specify separate reference and candidate columns.

```sh
.venv/bin/python validation/compare_vmaf_scores.py \
  BASELINE/scores.csv CANDIDATE/scores.csv
```

When both reports contain input hashes, the comparison requires them to match.
Optional `--max-absolute` and `--mean-absolute` thresholds fail the comparison
when drift exceeds the specified limits. The historical full-corpus comparison
records both models' correlations and numerical differences for this
conversion and workload. [CID22 results](../bench/notes/vmaf/vmaf-reference.md)

### Complete clips

The CLI accepts `--metric vmaf-v0` and `--metric vmaf-v1` for independent images
and complete progressive Y4M sequences:

```sh
./build/vqm --metric vmaf-v0 original.y4m distorted.y4m
./build/vqm --metric vmaf-v1 --json original.y4m distorted.y4m
./build/vqm --metric vmaf-v0 --json --pairs-file pairs.tsv \
  --iterations 5 --warmup 1
```

Each TSV row is `sample ID<TAB>original path<TAB>distorted path`. JSON contains the
arithmetic-mean clip score and every emitted frame in index order, with its score
and feature array. The v0 feature order is ADM2, motion2 and VIF scales
0–3; v1 uses CAMBI, SpEED chroma UV, ADM3 and motion3. The CLI flushes the final
pending frame before reporting a clip. Native coded input geometry and format
are preserved, with the shared `--input-*` color overrides available.

`--iterations` and `--warmup` repeat whole clips with reset history. Core timing
includes the final flush and pooling; decode is reported separately. Hosted
video validation uses this same executable:

```sh
.venv/bin/python -m validation.temporal.run --profile smoke --metrics vmaf-v0,vmaf-v1
```

The runner compares complete clip scores against pinned original-implementation
records. The registered profile requires ranking agreement and reports numerical
differences. Add `--gate-mode both` to require the numerical limits too.
[Frozen corpus and full qualification workflow](temporal-validation.md)

## Repeated-pair benchmark

`vqm_bench` retains the repeated-pair VMAF microbenchmark: it decodes one pair and
advances the sequence by reusing that pair, including the first frame of a Y4M
file. Its resident and streaming stage measurements describe that workload.
Use its `--vmaf-chroma native`, `--vmaf-chroma 420`, or `--vmaf-chroma 444` options
to select the input path described above.

```sh
./build/vqm_bench --metric vmaf-v1 --vmaf-chroma 420 \
  --pair original.png distorted.png --resident 2 --iterations 20 --warmup 3
```

The historical 1080p runs record the resident and streaming protocol, stage
timings and checksums. [1080p measurements](../bench/notes/vmaf/vmaf-notes.md)

## Reference licenses

Netflix's VMAF source and models use the BSD-2-Clause-Patent license. The
reference driver calls libvmaf's public C API. The reference build includes the
separately licensed libsvm dependency.
[Netflix VMAF notice](../licenses/NETFLIX-VMAF-LICENSE)
[libsvm notice](../licenses/LIBSVM-LICENSE)
