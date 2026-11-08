# Command line

Run these commands from the repository root after [building VQM](../README.md#build).

Pass the original first and the distorted rendition second. Use matching
dimensions and align video frames to the same content, frame rate and frame
count. The CLI reads PNG, JPEG and progressive Y4M, plus compressed clips with
[BestSource indexed hardware decoding](#bestsource-indexed-hardware-decoding).

```sh
./build/vqm original.png distorted.png
./build/vqm --metric ssimulacra2,butteraugli original.png distorted.png
./build/vqm --metric all --verbose original.y4m distorted.y4m
```

Use `--list-devices` to see compatible Vulkan compute devices with their
zero-based indexes, names, types, and UUIDs. Pass an index to `--device` to
select it for the whole invocation, including batch runs.

```sh
./build/vqm --list-devices
./build/vqm --device 0 --metric all original.y4m distorted.y4m
```

SSIMULACRA 2 is the default. Use a comma-separated list or repeat `--metric`
to run several metrics on the same pair. `all` runs the six metrics below
in table order. Each metric runs once in first-selected order. The pair must meet
every selected metric's input requirements.

The CLI processes frame pairs in order, running the selected metrics serially
for each frame before reading the next. Batch clip pairs are also processed one
at a time. GPU work within a metric can still run in parallel. This serial CLI
scheduler is distinct from the independent GPU workers used in throughput
benchmarks. SSIMULACRA 2, Butteraugli and PSNR-HVS have no temporal dependency;
the CLI does not currently distribute their frames across worker pipelines.

| CLI metric | Better score | How the CLI scores a clip |
| --- | --- | --- |
| [`ssimulacra2`](#ssimulacra-2) | Higher, up to 100 | Mean of frame scores |
| [`butteraugli`](#butteraugli) | Lower, down to 0 | Mean of frame diffmap maxima |
| [`vmaf-v0`](#vmaf-v0) | Higher, up to 100 | Mean of model-predicted frame scores |
| [`vmaf-v1`](#vmaf-v1) | Higher, up to 100 or 110 by preset | Mean of model-predicted frame scores |
| [`cvvdp`](#colorvideovdp) | Higher, up to 10 JOD | Pools whole-clip distortion, then converts to JOD |
| [`psnr-hvs`](#psnr-hvs) | Higher dB, up to `inf` | Averages weighted frame errors, then converts to dB |

Compare renditions of the same original using the same metric, color
interpretation, model and viewing settings. The metrics use different scales,
so compare their rankings separately.

PNG and JPEG use sRGB with BT.709 primaries. Y4M defaults to limited range,
the BT.470 BG matrix, and BT.709 transfer and primaries. Set `--input-range`,
`--input-matrix`, `--input-transfer` and `--input-primaries` when the samples
use different colorimetry. These overrides apply to both inputs. Run
`./build/vqm --help` for accepted values.

## SSIMULACRA 2

```sh
./build/vqm --metric ssimulacra2 original.png distorted.png
```

Use SSIMULACRA 2 for visible image artifacts such as blur and ringing. Identity
scores 100 on a discrete GPU. As rough landmarks from the original calibration,
70 is good visual quality, 90 is close to visually lossless, and negative
scores indicate severe distortion. For images with alpha, the CLI scores dark
and bright backgrounds and keeps the worse result.
[Original calibration](https://github.com/cloudinary/ssimulacra2#usage) ·
[VQM implementation](ssimulacra2.md)

## Butteraugli

```sh
./build/vqm --metric butteraugli original.png distorted.png
```

A frame's score is the maximum of its perceptual distance map. Identity scores
0, and a small area with a strong defect can drive the score up. For a clip,
**Frame max** is the worst frame score. Use `--raw` to see the per-frame
`distance_p3` as well, which pools distances across the image. The CLI keeps the
worse result across dark and bright backgrounds for images with alpha.
`--intensity-target` and `--hf-asymmetry` default to 80 nits and 1.0.
[Butteraugli score details](butteraugli.md#score-scale)

## VMAF v0

```sh
./build/vqm --metric vmaf-v0 --vmaf-model 4k --vmaf-scale --vmaf-neg \
  original.y4m distorted.y4m
```

The CLI uses floating-point v0.6.1 SDR predictors. Select `--vmaf-model 1080`
for 1080p at 3H (the default) or `--vmaf-model 4k` for 4K at 1.5H. Higher means
better predicted quality on the 0–100 scale. The clip score averages the frame
predictions, which include reference motion.
Video frames must be supplied in sequence order to preserve motion history;
`--vmaf-no-temporal` selects the frame-local v0 mode instead.

Use `--vmaf-neg` to disable enhancement gain when comparing encoders.
`--vmaf-transform` applies the phone score transform calibrated on the 1080p
model. VQM also offers both options with the 4K predictor as derived variants.
The phone transform can saturate 4K scores at 100.

`--vmaf-scale` and `--vmaf-pad` apply to both selected SDR VMAF versions.
Scaling fits each metric's input within its selected model canvas while
preserving aspect ratio. Padding adds black bars to reach 16:9 at the current
scale. Together the flags fit and pad to each model's full canvas. By default
both versions score at the input dimensions.
[VMAF v0 model details](vmaf.md#v061-models)

## VMAF v1

```sh
./build/vqm --metric vmaf-v1 --vmaf-v1-model 4k-1.5h --vmaf-scale \
  original.y4m distorted.y4m
```

VMAF v1 includes features for banding and chroma detail. Select a preset for
the viewing setup. The suffix `h` expresses viewing distance in screen heights.
Use the same preset when comparing renditions. Video frames must be supplied in
sequence order to preserve motion history.

| `--vmaf-v1-model` | Viewing setup | Score range |
| --- | --- | --- |
| `1080p-3h` (default) | SDR 1080p at 3H | 0–100 |
| `4k-1.5h` | SDR 4K at 1.5H | 0–100 |
| `4k-3h` | SDR 4K at 3H | 0–110 |
| `phone-5h` | SDR phone at 5H | 0–100 |
| `hdr-4k-1.5h` | HDR 4K at 1.5H | 0–100 |

Add `--vmaf-v1-hfr` to any preset for roughly 50/60 fps content. It selects
five-frame motion and smoothing. The HDR preset automatically scales the short
side to 2160 pixels, keeping the aspect ratio. The example below declares the
colorimetry of limited-range HDR10 in Y4M.

```sh
./build/vqm --metric vmaf-v1 --vmaf-v1-model hdr-4k-1.5h \
  --input-matrix bt2020ncl --input-transfer st2084 --input-primaries bt2020 \
  --input-range limited original-hdr.y4m distorted-hdr.y4m
```

[VMAF v1 presets and reference details](vmaf.md#v1016-sdr-models)

## ColorVideoVDP

```sh
./build/vqm --metric cvvdp --cvvdp-display compact --cvvdp-peak 200 \
  original.y4m distorted.y4m
```

ColorVideoVDP scores opaque SDR and HDR inputs in just-objectionable
differences (JOD) under a specified viewing setup. Identity scores 10 JOD and
lower scores indicate more objectionable distortion.
Negative scores are valid. Under the model's calibration, a 1 JOD advantage
corresponds to a predicted 75% preference in a paired comparison.
[ColorVideoVDP paper](https://www.cl.cam.ac.uk/~rkm38/pdfs/mantiuk2024_ColorVideoVDP.pdf)

The default `standard` is a 30-inch 16:9 display viewed from 0.7472 m. `compact` is
24 inches at 0.6 m. Override the aspect ratio, diagonal in inches or distance
in metres with `--cvvdp-aspect`, `--cvvdp-diagonal` and `--cvvdp-distance`.
`--cvvdp-peak` sets peak brightness in nits and defaults to 200. Keep these
conditions fixed when ranking encodes. Read **Aggregate** for the whole-clip
JOD and use the temporally filtered frame diagnostics to locate problem frames.
Video frames must be supplied in sequence order to preserve temporal history.
[CVVDP display and input options](cvvdp.md#display-and-input-interpretation)

## PSNR-HVS

```sh
./build/vqm --metric psnr-hvs --json original.y4m distorted.y4m
```

Use matching opaque 8-, 10- or 12-bit planar YUV 4:2:0, 4:2:2 or 4:4:4 inputs.
The metric measures their native sample codes. Higher dB means lower weighted
error. A gain of about 3 dB corresponds to halving that error. `inf` means
zero measured error, which includes identical frames and can also occur when
masking or block coverage excludes the differences. A clip reports `inf` only
when every frame has zero measured error. JSON writes infinity as the string
`"inf"`. Use the per-frame Y, Cb and Cr results to see which planes contribute
to the distortion.
PSNR-HVS scores each frame independently; it keeps no temporal history, although
the CLI pools frame errors into a clip score.
[PSNR-HVS inputs and error weighting](psnr-hvs.md)

## Reading the report

Use `--verbose` (or `-v`) for an aligned table. This run scored a three-frame,
320×240 clip with a translucent box over the middle frame. Scoring FPS
depends on the input, device and selected metrics.

```text
Frames: 3
Scoring FPS: 746.57
Metric             Aggregate       Frame min       Frame max   Finite stddev  Perfect
ssimulacra2        65.688104      -2.9356868             100       48.524348        2
butteraugli        29.944338               0       89.833015        42.34769        2
vmaf-v0            97.460145       95.452462       99.500102       1.6525997        0
vmaf-v1            98.612725       95.838175             100       1.9619032        2
cvvdp              9.6586087       9.5719113              10      0.19466893        1
psnr-hvs           37.861903       33.090691             inf               0        2
```

- **Aggregate** uses the clip calculation for each metric. PSNR-HVS stays
  finite here because the altered frame contributes error to the clip.
- **Frame min / max** show the extremes. Read the minimum for higher-is-better
  metrics and the maximum for Butteraugli to find the worst frame score.
- **Finite stddev** measures the spread of finite frame scores as a population
  standard deviation. One finite score gives 0. With no finite scores it
  displays `n/a`.
- **Perfect** counts scores that equal the metric's best value, including
  `inf` for PSNR-HVS. Motion can affect a frame's VMAF prediction, and clipping
  can put distorted frames at the ceiling.
- **Scoring FPS** divides the shared frame count by the sum of the selected
  metrics' core scoring times. Decode time is measured separately.

Plain text prints a bare score for one metric and labeled scores for multiple
metrics. `--raw` prints additional metric-specific values for a single metric.
`--json` emits one object per pair and metric, with the clip `score`, ordered
`frame_results` with zero-based indices, `core_seconds` and `decode_seconds`.
Add `--verbose` for `scoring_fps`, `frame_stats` including the finite-frame
mean, and CVVDP's diagnostic frame JODs. Without it, CVVDP returns its clip score with an empty
`frame_results` array. Unavailable statistics are `null`.

Use `--pairs-file` with `--json` to process a batch. Each TSV row holds a
unique sample ID, original path and distorted path. `--iterations` repeats
the complete clip with reset history and reports the iteration with median
combined core time. `--warmup` runs untimed clips first.

```sh
./build/vqm --metric vmaf-v0 --json --pairs-file pairs.tsv \
  --iterations 5 --warmup 1
```

[Complete video and batch examples](../validation/temporal/README.md#score-a-complete-video)

## BestSource indexed hardware decoding

On Linux, build with `-DVQM_ENABLE_BESTSOURCE=ON` to enable compressed video
input. The CLI detects compressed video pairs by their contents and decodes
them with BestSource. CMake fetches pinned BestSource R23 and libp2p sources,
verifies their archive hashes, and builds both into static libraries linked
into the CLI. [BestSource R23 release](https://github.com/vapoursynth/bestsource/releases/tag/R23)

The build needs a Python 3 interpreter to embed the GPU shaders. FFmpeg
and xxHash are external dependencies. Make their development packages
visible to `pkg-config` and their shared libraries visible to the runtime
linker. Use an FFmpeg build with Vulkan Video, libavutil 61.6 or newer, and
the Vulkan UUID device selector. For a non-system prefix, add its
`lib/pkgconfig` directory to `PKG_CONFIG_PATH` before configuring.

```sh
cmake --preset release -B build-bestsource -DVQM_ENABLE_BESTSOURCE=ON
cmake --build build-bestsource --parallel 8
./build-bestsource/vqm --start-frame 174 --frame-count 24 \
  --metric vmaf-v1 --json original.mp4 distorted.mp4
```

`--start-frame` is zero-based and applies to both inputs. It defaults to zero.
`--frame-count` sets the exact number of frames to score and must fit in both
clips. With no count, both inputs must end together. Metric history starts at
the first selected frame, and per-frame output indexes are relative to that
frame. BestSource builds and caches the frame index as needed. It can
temporarily download candidate frames while verifying a seek. Once selected,
frames are packed and scored in Vulkan device memory.

The backend requires the same physical GPU for decoding and scoring, Vulkan
opaque FD memory and semaphore sharing, and hardware support for the input
codec. `--device` selects that GPU for both operations. The device list
checks baseline compute support, so a listed device may lack Vulkan Video
support for a given codec or format. The backend accepts local progressive,
constant-rate 8-, 10- and 12-bit YUV 4:2:0 and 4:2:2 clips, subject to each
metric's format requirements. The CLI uses decoded color metadata and accepts
the same `--input-*` overrides as Y4M.
This backend is CLI-only and requires hardware decoding at runtime.
