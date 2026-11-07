# ColorVideoVDP

VQM implements the original **base ColorVideoVDP v0.5.7** calibration in FP32
Vulkan compute. It scores opaque images and fixed-frame-rate video in JOD:
higher is better, identical inputs score exactly **10**, and negative scores
are valid. A clip score pools distortion over the whole clip before converting
to JOD; it is not an average of per-frame JOD scores.

The pinned original is the numerical reference. Its MATLAB entry point wraps
Python. Vship informed the design research and is a separate comparison target;
its temporal, masking and boundary choices differ from the original. No Vship
implementation code is included. The original model data and attribution ship
with the library. Current checks and measured limits are in the
[implementation report](../bench/notes/cvvdp/cvvdp-implementation.md). See [reference reproduction](cvvdp-development.md) and the
[original design plan](design/cvvdp-implementation-plan.md).

## Images and clips

```cpp
#include <cvvdp.h>

vqm::CvvdpConfig config;
config.size = original.size;
config.display = vqm::CvvdpDisplay::Standard4K();
config.temporal = false;  // omit video history for independent images
vqm::CvvdpPipeline images(config);
double jod = images.ComputeImage(original, distorted).score;

config.temporal = true;
config.fps_numerator = 30000;
config.fps_denominator = 1001;
vqm::CvvdpPipeline video(config);
for (const auto& pair : frames) video.Push(pair.original, pair.distorted);
auto clip = video.Finalize();  // nullopt for an empty clip
video.Reset();
```

`Push` waits for input consumption and retains private converted history, so
host data and borrowed Vulkan buffers can be reused after it returns. The first
pair is retained until the next pair or `Finalize` establishes whether the input
is an image or video. A single-frame clip uses the image algorithm. Multiple
frames use four temporal channels, replicated startup and the original causal
FIR, including filtering of the first frame. No future-frame flush is needed.

Repeated `Finalize` returns the cached score. `Push` after finalization requires
`Reset`; `ComputeImage` requires a new/reset pipeline and leaves it ready for
another independent image. Reset preserves resident handles. Invalid host
descriptions are rejected before advancing the clip. Numerical processing
failures require Reset; a failed GPU submission requires reconstruction.

The individual API accepts `Image`, strided `ImageView`, `UploadedImage` and
`VulkanBufferView` pairs. Construct with `VulkanDevice` for caller-owned device
inputs. The binary semaphore overload follows the existing
[synchronization contract](vulkan-synchronization.md).

`MetricsConfig::cvvdp` enables the metric in the unified API. `ComputeImage`
populates `MetricsResult::cvvdp`. Ordered `Compute` results contain frame-aligned
scores for the other metrics; CVVDP is returned by `Metrics::Finalize` in its
typed `cvvdp` clip field. That result also includes the optional last pending
VMAF frame. A CVVDP-only `Compute` returns frame-index metadata with empty score
fields. `Flush` remains compatible and also finalizes CVVDP; a subsequent
`Finalize` retrieves the clip score without repeating the VMAF tail.

## Display and input interpretation

The CLI and benchmark use `--cvvdp-display` to select viewing geometry.
The presets use the dimensions and distance of the original display models.
VQM derives pixels per degree (PPD) for each input.
[Original display models](https://github.com/gfxdisp/ColorVideoVDP/blob/2a268bce8d56e2f3abde46df3927d8a633707a24/pycvvdp/vvdp_data/display_models.json)

| Viewing preset | Aspect | Diagonal | Distance |
| --- | --- | --- | --- |
| `standard` (default) | 16:9 | 30 in | 0.7472 m |
| `compact` | 16:9 | 24 in | 0.6 m |

Set display brightness with `--cvvdp-peak` in nits (default 200). Contrast
and ambient light default to 1,000:1 and 250 lux and can be set separately
with `--cvvdp-contrast` and `--cvvdp-ambient`.

Use `--cvvdp-aspect`, `--cvvdp-diagonal` (inches), and `--cvvdp-distance`
(metres) to override any of the preset's geometry values.

```sh
./build/vqm --metric cvvdp --cvvdp-aspect 21:9 --cvvdp-diagonal 34 \
  --cvvdp-distance 0.7 original.y4m distorted.y4m
```

This models centered viewing of square-pixel content fitted within the
display at its original aspect ratio. Matching aspects fill the screen.
Wider content fills the width and leaves space above and below, while
narrower content fills the height and leaves space at the sides. The
limiting dimension sets the effective pixel pitch in metres as
`min(display_width_m / frame_width, display_height_m / frame_height)`.
VQM scores the original frame samples and calculates central pixels per
degree from the angle subtended by one pixel.
[Original display geometry](https://github.com/gfxdisp/ColorVideoVDP/blob/2a268bce8d56e2f3abde46df3927d8a633707a24/pycvvdp/display_model.py#L453-L476)

Use `--cvvdp-ppd` for a fixed angular density. This option is mutually
exclusive with the geometry overrides.
CVVDP accepts 1 to 1,000,000 PPD.

The C++ library supplies fixed-PPD helpers for reproducing the original
display models. `Standard4K()` and `StandardHdr()` use 75.4024493452 PPD,
and `StandardFhd()` uses 37.8425038536 PPD. All library display parameters
can be overridden, including exposure and diffuse reflectivity.

Opaque RGB, grey and nonconstant-luminance YUV inputs use shared raw unpacking,
range handling and configurable chroma reconstruction, followed by metric-local
physical display conversion. CVVDP uses each input's primaries (BT.709, BT.2020
or Display P3) to map its display light to common DKL opponent-color channels.
To score the perceptual effect of gamut mapping, supply the mapped rendition
with its own primaries. Supported transfers are sRGB, BT.709/BT.601/BT.2020 SDR,
absolute linear, PQ and HLG. As in the original `BT.709` source preset, SDR
tags use the sRGB EOTF.
**Linear samples are absolute cd/m²**, unlike other VQM metrics' relative
linear convention. PQ and HLG use the configured physical display, not a fixed
100-nit normalization. HLG above 1000 nits requires positive ambient light for
the original surround correction. Alpha, CL/ICtCp encodings and unspecified
colorimetry are rejected. Non-finite samples are rejected after GPU completion.

This is the base model only: replicate padding, Weber contrast, bundled CSF LUT
and mutual masking. Heatmaps, symmetric temporal padding, variable frame rates,
learned CVVDP models and image resampling are not exposed. The original model's
mixed-parity pyramid edge behavior is retained for v0.5.7 compatibility. Some
uniform luminance changes are invariant under the original contrast model;
identity-score expectations should not be generalized to arbitrary content.

## Resources and tools

Temporal history is bounded by geometry and frame rate, independent of clip
length. The two FP32 three-channel rings alone use
`24 * width * height * (2*ceil(fps/8)+1)` bytes: about 807 MiB at 1080p60 and
3.15 GiB at 4K60. Scratch, source slots and staging are additional. Histories
are separate device-local allocations, exposed in memory queries and actual
usage as `auxiliary`; their heap costs are included in worker budgeting. Memory
policy governs the arena/upload path; histories always use device-local memory.

The supported timing range is 1–240 fps. Construction and memory queries reject
geometries/rates that exceed compute dispatch, 32-bit indexing or per-buffer
descriptor limits. The device needs 256-invocation workgroups; video also needs
six storage bindings. Query `GetMemoryRequirements(config)` before allocation.
A device with a 2 GiB descriptor limit cannot hold a single 4K120 history ring
with this implementation. Disable temporal storage for still-only work.

```sh
./build/vqm --metric cvvdp original.png distorted.png
./build/vqm --metric cvvdp original.y4m distorted.y4m
./build/vqm --metric cvvdp --verbose --json original.y4m distorted.y4m
./build/vqm --metric cvvdp --json original.y4m distorted.y4m \
  --iterations 5 --warmup 1
./build/vqm --metric cvvdp --json --pairs-file pairs.tsv
./build/vqm --metric cvvdp --cvvdp-peak 1500 --cvvdp-contrast 1000000 --cvvdp-ambient 10 \
  --input-transfer st2084 --input-primaries bt2020 --input-matrix bt2020ncl original.y4m distorted.y4m
./build/vqm_bench --metric cvvdp --pair original.png distorted.png \
  --iterations 20 --warmup 3 --resident 2
```

The CLI consumes an entire progressive Y4M stream with matched rational timing,
geometry and frame count. PNG/JPEG input is an independent image. Plain text and
`--raw` output retain their image and clip score formats; `--raw` also prints the
frame count. JSON contains one complete clip result, its frame count and separate
core/decode times. Its `frame_results` array is empty by default. Every
repetition resets the clip history, and the CLI reports the iteration with
median core time. `--pairs-file` reads TSV rows with sample ID, original path
and distorted path, allowing multiple independent clips in one process.

`--verbose` summarizes diagnostic frame JODs. With `--json` it also reports
each frame's index and JOD in `frame_results`. For video, each value maps that
frame's spatially pooled, temporally filtered distortion to JOD before
pooling across the clip. The clip score pools the frame distortions first,
then converts the pooled distortion to JOD. A one-frame input uses the image
JOD for its frame and clip scores.

Run the hosted smoke profile for CVVDP through the public CLI:

```sh
.venv/bin/python -m validation.temporal.run --profile smoke --metrics cvvdp
```

Its canonical full-resolution Y4M 4:4:4 inputs have the same pinned YUV bytes in
both arms. The original reference converts those samples independently on the
CPU. The CLI performs its normal GPU input conversion. The registered profile
requires ranking agreement and reports numerical clip-score differences.
Add `--gate-mode both` to require the numerical limits too. Source and display
settings, reference records and ranking policy are documented in
[temporal validation](temporal-validation.md).

`vqm_bench` uses independent images by default. Its `--cvvdp-video` stage-profiling
mode also accepts ordered Y4M clips and labels frame and clip work units
separately. It rejects interlaced input and invalid frame rates. The reader
uses coded pixel geometry and ignores the Y4M sample aspect ratio. Resample
non-square inputs before scoring if display geometry requires it. The core
library takes decoded frames and has no Python, PyTorch, MATLAB or FFmpeg
dependency.
