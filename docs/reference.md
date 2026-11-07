# VQM API and implementation guide

VQM is a C++20 Vulkan library implementing SSIMULACRA 2.1, Butteraugli,
floating-point VMAF v0.6.1, SDR and HDR VMAF v1, ColorVideoVDP and
FP32 PSNR-HVS. The core library takes decoded planar images and uses the
Vulkan loader and system threading support. GPU input
conversion handles the supported sample types, pixel ranges, alpha types,
transfer functions, matrices, and primaries. VMAF also has a native coded-input
path and rejects HDR transfers. The CLI decodes PNG and JPEG with system image
libraries. Its Y4M reader feeds complete clips through the metric APIs.
PSNR-HVS requires its native planar YUV domain; its reference ranking
qualification is pending.

Start with the [unified metrics guide](metrics.md) for still-image and video
workflows across the metric APIs. This reference covers build and installation,
the individual SSIMULACRA 2 and Butteraugli APIs, and their detailed GPU
implementation. The [VMAF guide](vmaf-development.md) covers both VMAF sequence
APIs, model assumptions, input restrictions, and reference validation. The
[memory guide](memory.md) describes allocation planning for every metric.

Derived from Cloudinary's `ssimulacra2` reference implementation
(Copyright (c) Jon Sneyers, Cloudinary), BSD-3-Clause; see `LICENSE`.
See [metric documentation](README.md) for the pipelines, reference
differences, validation, and performance methodology.

## Build and test

```sh
cmake --preset release
cmake --build --preset release -j
ctest --preset release --output-on-failure
```

`CMakePresets.json` also provides `debug` and `asan` (AddressSanitizer +
UndefinedBehaviorSanitizer) configure, build and test presets. Requires a C++20
compiler, CMake >= 3.25, the Vulkan loader and headers (`Vulkan::Vulkan`) and
`glslc` from the Vulkan SDK; the `release` preset builds with warnings as
errors. A Vulkan 1.1 compute device and driver are required at runtime. Apple
platforms are unsupported (there is no MoltenVK path or native CPU
implementation). libpng and libjpeg (`find_package(PNG)`, `find_package(JPEG)`)
are required by the CLI *and* by the tests, which decode
through the same readers. Configure with `-DVQM_BUILD_CLI=OFF
-DVQM_BUILD_BENCH=OFF -DVQM_BUILD_TESTS=OFF` to build only the core.
`VQM_WARNINGS_AS_ERRORS` controls strict compiler warnings. The standalone
core build needs no external reference checkout, downloaded corpus, PyTorch
or MATLAB installation. See [contributing](../CONTRIBUTING.md) for the
complete local Python integration-test setup.

To install the core library and its CMake package:

```sh
cmake --install build --prefix /path/to/prefix
```

An application can then use `find_package(vqm CONFIG REQUIRED)` and
link `vqm::core`. For an embedded build,
`add_subdirectory(path/to/vqm)` provides the same target; the CLI,
benchmark and tests default to off when this project is a subdirectory. The
core itself requires Vulkan but does not depend on PNG or JPEG.

## Usage

```sh
./build/vqm original.png distorted.png
./build/vqm --raw original.png distorted.png   # also print the 108 norms
./build/vqm --half-planes original.png distorted.png  # 16-bit planes
./build/vqm --chroma-filter lanczos original.y4m distorted.y4m
./build/vqm --input-matrix bt709 --input-range full a.y4m b.y4m
```

The score is in `-inf..100`, higher is better; 100 means mathematically
identical input. `--raw` prints the per-scale, per-channel, per-norm
intermediate values first (the reference tool's weight-tuning format), which is
useful when comparing stage by stage against the reference. `--half-planes`
opts into 16-bit plane storage (see *Half-precision planes* below). Its score
drift and timing depend on the workload; the historical measurements below
do not qualify every input domain for that mode.

### Input formats

`Image` carries a `ImageFormat` descriptor next to its packed sample buffer, and
every field of that descriptor is decoded on the GPU (see *Input conversion*
below):

| field | accepted |
| --- | --- |
| datatype | `Byte`, `Word`, `Half`, `Float`, with a separate bit depth (1..type depth) |
| colour family | `Grey`, `Rgb`, `Yuv` (planar, up to four planes: Y/R, U/G, V/B, alpha) |
| subsampling | 4:4:4, 4:2:2, 4:1:1, 4:2:0 in any combination, with all six chroma sitings |
| pixel range | full or limited (limited needs at least 8 bits) |
| alpha | none, straight or premultiplied, always full resolution |
| matrix coefficients | RGB, BT.709, FCC, BT.470BG/BT.601, SMPTE 170M/240M, YCgCo, BT.2020 non-constant and constant luminance, chromaticity-derived non-constant and constant luminance, BT.2100 ICtCp |
| transfer | BT.709/601/2020-10/2020-12 (BT.1886), BT.470M, BT.470BG, SMPTE 240M, linear, log 100, log 316, xvYCC, BT.1361, sRGB, ST 2084, ST 428, ARIB B67 |
| primaries | BT.709, BT.470M, BT.470BG, SMPTE 170M/240M (SMPTE C), film, BT.2020, ST 428 (XYZ), DCI-P3, P3-D65, EBU 3213-E |

That is zimg's input surface, minus field parity, active windows and packed
formats. The values are zimg's enum values, so a caller that already holds zimg
enums can pass them straight through. `ChromaFilter` picks the chroma
upsampling kernel: `Bilinear` (default), `Bicubic` (Catmull-Rom) or `Lanczos`
(3 taps), one choice per pipeline.

The CLI decodes PNG (8/16-bit), JPEG and progressive YUV4MPEG2. Its Y4M reader
supports `Cmono`, `C420`, `C420mpeg2`, `C420jpeg`, `C420paldv`, `C422`, `C444`,
`C411` and `C444alpha`, with `p9`, `p10`, `p12`, `p14` or `p16` suffixes for
high bit depth. For example, the reader accepts `Cmonop10`. FFmpeg's
`Cmono10` spelling is currently unsupported. Each metric still applies
its own input requirements.
PNG and JPEG are interpreted as sRGB with BT.709 primaries. Y4M defaults to the
BT.470BG matrix, BT.709 transfer and primaries, and limited range.
The reader ignores `XCOLORRANGE`. Pass `--input-range full` when both Y4M
inputs contain full-range samples.
An omitted `C` tag uses centered 4:2:0 chroma (`C420jpeg`), as specified by
[MJPEGtools' YUV4MPEG2 format declaration](https://svn.code.sf.net/p/mjpeg/Code/trunk/mjpeg_play/utils/yuv4mpeg.h).
The explicit `C420` spelling retains left siting for compatibility; use
`C420jpeg` or `C420mpeg2` to specify the siting unambiguously.
`--input-matrix`, `--input-transfer`, `--input-primaries` and `--input-range`
override whatever the decoder declared, on both images. Embedded ICC profiles,
`gAMA`, `cHRM` and `sBIT` are ignored, as before.

## Core API

```cpp
#include "vqm.h"

vqm::Image orig, dist;      // size, format, packed sample bytes
vqm::Ssimulacra2Config config{orig.size};
vqm::Ssimulacra2Pipeline pipeline(config);
vqm::Ssimulacra2Result msssim = pipeline.Compute(orig, dist);
double score = msssim.Score();      // msssim.scales holds the 18 norms per scale
```

For a sequence of pairs with one geometry, a bounded pool can keep several
pipelines in flight while returning results in input order:

```cpp
#include <array>
#include "vqm_pool.h"

vqm::Ssimulacra2Pool pool(config, 2);  // two complete pipeline workspaces
std::array<vqm::ImagePair, 2> pairs{{{orig0, dist0}, {orig1, dist1}}};
std::vector<double> scores = pool.ScoreMany(pairs);
```

`ComputeMany` returns each pair's complete `Ssimulacra2Result`; `ScoreMany` returns just
the quality scores. `ButteraugliPool::ComputeMany` returns `ButteraugliScore`
values. Both pools also accept `ImageViewPair` for borrowed, potentially padded
host planes. They accept `VulkanBufferPair` values when constructed with the
caller's `VulkanDevice`. Calls are synchronous, are made by one host
thread, and score one alpha background like `Ssimulacra2Pipeline::Compute`. All referenced
images and buffers must remain alive and unchanged until the call returns.
All pairs must have the configured geometry. The caller must synchronize other
users of a borrowed queue as described below. Each worker owns a complete
pipeline, so increasing the worker count also increases GPU memory use.

For separate host planes or padded rows, use a borrowed view instead of
constructing an owning image:

```cpp
#include "vqm_views.h"

vqm::ImageView original;  // populated from the reference planes
vqm::ImageView frame;
frame.size = {width, height};
frame.format = format;  // describe the input sample type and color space
frame.planes = {{{reinterpret_cast<const std::byte*>(r), r_stride},
                 {reinterpret_cast<const std::byte*>(g), g_stride},
                 {reinterpret_cast<const std::byte*>(b), b_stride}, {}}};
double score = vqm::Score(pipeline, original, frame);
```

The pointers and positive byte strides describe planar data. The views are
copied directly into the mapped upload or staging region, without temporary
packed images. Keep the planes readable until `Submit` returns (or until
`Score` returns). `Score` evaluates one alpha background, like `Compute`.

`ComputeWorstBackground` in `vqm_alpha.h` matches the CLI's alpha
policy: it evaluates dark and bright backgrounds when either image has alpha
and returns the worse metric result. Opaque inputs require one call. It supports
SSIMULACRA 2 and Butteraugli, with the image and view input types accepted by
their pipelines.

### Caller-owned Vulkan buffers

Applications that already own Vulkan buffers can borrow their device and
queue and pass the buffers directly to the metric:

```cpp
#include "vqm_vulkan.h"

vqm::VulkanDevice vulkan{physical_device, device, queue, queue_family};
vqm::Ssimulacra2Config config;
config.size = {width, height};
config.gpu_resident_images = 0;
config.streaming_inputs = false;
vqm::Ssimulacra2Pipeline pipeline(config, vulkan);

vqm::VulkanBufferView original{reference_buffer, reference_offset,
                                  reference_bytes, config.size, format};
vqm::VulkanBufferView distorted{distorted_buffer, distorted_offset,
                                   distorted_bytes, config.size, format};
double score = pipeline.Compute(original, distorted).Score();
```

`ButteraugliPipeline` accepts the same borrowed `VulkanDevice` and
`VulkanBufferView` arguments and returns its usual `ButteraugliScore`.
The instance must have been created for Vulkan 1.1 or later. Both buffers must
belong to the supplied logical device and have
`VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`. The shader reads them directly: there is
no CPU readback, PCIe upload, or intermediate pixel copy. The default layout is
the same tightly packed planar layout as `Image`; `bytes` is the available
range starting at `offset`. For padded planes within one buffer, set each
present plane's offset (relative to `offset`) and row stride in bytes:

```cpp
original.planes = {{{r_offset, r_stride}, {g_offset, g_stride},
                    {b_offset, b_stride}, {}}};
```

Plane starts must be word-aligned; row strides must be multiples of the sample
size. The `bytes` range must cover the full last 32-bit word read by each plane.
A `VkImage`, interleaved RGB and separate plane buffers are not supported by
this Vulkan interface.

The application owns the device, queue, buffers and their memory. Keep them
alive and the input contents unchanged until `Finish` returns, or until a
pipeline with an outstanding submission has finished destruction. Serialize
host access to the shared `VkQueue` across the application and all pipelines.
Submit producer work before the scoring call on that queue; the library adds a
memory dependency for its shader reads. For a different producer queue, arrange
the semaphore wait and any queue-family ownership transfer on the supplied
queue first, or pass binary semaphore dependencies to `Submit`:

```cpp
std::array waits{vqm::VulkanSemaphoreWait{
    producer_done, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT}};
std::array signals{consumer_ready};
vqm::VulkanSubmitSync sync{waits, signals};
std::uint64_t token = pipeline.Submit(original, distorted, sync);
// The application can submit a downstream GPU wait on consumer_ready here.
double score = pipeline.Finish(token).Score();
```

The overload accepts binary semaphores only; timeline semaphores are not
supported. The wait mask must include compute, top-of-pipe or all-commands,
and may contain only those stages or transfer. Semaphores do not transfer
queue-family ownership, and the caller must keep them alive until all queue
operations that use them complete. See [Vulkan synchronization](vulkan-synchronization.md)
for lifetime and queue-family details. Flush noncoherent host writes before
submission. Vulkan does not
expose a way to verify a buffer's actual allocation size, usage or owning
device from the handle, so those are caller responsibilities. Set
`storage_buffer_16bit_access` only when that feature was enabled on the borrowed
logical device and `half_planes` is needed.

`Image::data` holds the planes in zimg order and row-major with no padding,
each plane starting on a 4-byte boundary: `width * height` samples for the
luma/RGB planes and for alpha, `(width >> subsample_w) * (height >> subsample_h)`
for the two chroma planes, each sample one, two or four bytes according to
`ImageFormat::type`. `RequiredBytes(format, size)` returns exactly the length
`data` needs. `format.alpha` means the image is blended against `background`
(default `0.5`) after being decoded to RGB; a premultiplied image is divided by
its alpha first, so both alpha types decode to the same straight-alpha image.
A `Ssimulacra2Pipeline` fixes the image geometry at construction and allocates every
working buffer once, so scoring many pairs of the same size does not allocate
per call; it is not thread-safe, so use one instance per thread. The
constructor throws `std::invalid_argument` when a dimension is below 8 or the
geometry exceeds the device's storage-buffer range, and `std::runtime_error`
when no usable Vulkan device was found or pipeline creation failed. `Compute`
throws `std::invalid_argument` when an image size differs from the config, when
its format is not decodable (an unset transfer function or primaries set, a
constant-luminance matrix without the BT.709 curve, ICtCp without PQ/HLG and
BT.2020, an unsupported matrix coefficient, a bit depth above the datatype's),
or when `data` is not exactly `RequiredBytes(format, size)` long; every message
names the reason.

Scoring many pairs of the same source images can skip the upload entirely: park
the images on the device once, then score pairs of handles.

```cpp
vqm::Ssimulacra2Pipeline pipeline({orig.size});
vqm::UploadedImage a = pipeline.Upload(orig);
vqm::UploadedImage b = pipeline.Upload(dist);
vqm::Ssimulacra2Result msssim = pipeline.Compute(a, b);  // no upload per call
pipeline.Release(a);                                  // returns the slot
pipeline.Release(b);
```

`Upload` copies the image into device memory and returns an opaque handle;
`Compute(UploadedImage, UploadedImage)` scores two handles and never touches the
upload path. For many distortions of one original, upload the original once and
use `Compute(uploaded_original, distorted)`; only the distorted image is copied
each time. This mixed path needs `streaming_inputs = true` and is also available
on `ButteraugliPipeline`.
`Ssimulacra2Config::gpu_resident_images` sets how many images fit at once (default
2, i.e. one pair, 0 disables `Upload`); each slot costs
`4 * width * height` floats. `Ssimulacra2Config::half_planes` stores the
intermediate planes as 16-bit floats (see *Half-precision planes*). Both
`Upload` and the handle overload of `Compute` throw `std::invalid_argument`
when the pool is exhausted, when a handle has been released, or when the
geometry does not match.

## Pipeline

1. Input conversion, per sample: decode the raw sample with zimg's depth module
   (unpack, `range`/`depth` scale and offset), upsample subsampled chroma with
   the configured kernel, apply the matrix coefficients (or the
   constant-luminance or ICtCp op), apply straight or premultiplied alpha and
   the background blend, linearize with the declared transfer function, and
   convert the declared primaries to linear Rec.709. See *Input conversion*
   below.
2. Linear Rec.709 RGB -> XYB (opsin absorbance matrix, cube-root nonlinearity)
   with the reference's `MakePositiveXYB` rescaling.
3. Per scale (1:1 up to 1:32; downsampling is done in linear RGB and XYB is
   recomputed from the downsampled linear image):
   - blur the squares and cross-products and the images themselves with a
     separable Gaussian (sigma = 1.5, 13 taps, zero padded at the borders);
   - SSIM' error map and ringing/blurring edge-difference maps;
   - 1-norm and 4-norm per channel and map.
4. Weighted sum of the 108 norms and the reference's monotone mapping to
   scores at most 100; poor matches may score below zero. Nonfinite norms or
   overflow in the score mapping are rejected.

Filters and norm reductions use `float` on the device; the XYB planes can be
stored as 16-bit floats with `Ssimulacra2Config::half_planes`. The host
normalizes the norm sums and evaluates the score in `double`. Each pipeline
permits one submission in flight. There is exactly one implementation
of the metric: the stages above run as Vulkan compute dispatches (see below),
with no CPU tier and no backend selector. A `Ssimulacra2Pipeline` fixes the geometry —
and with it the scale pyramid and the device buffers — at construction and
reuses them for every `Compute` call.

### Input conversion

Arithmetic is `float32` throughout, for every input type. An integer sample is
only ever *unpacked* (bit extraction from a 32-bit arena word) and immediately
converted to `float` and normalised; the `Half` sample is widened with
`unpackHalf2x16`. Every later step is `float`: range expansion, chroma
filtering and its tap weights, the matrix/CL/ICtCp ops, the transfer decode,
the primaries matrix, the alpha unpremultiply and blend, the XYB conversion and
the downsample. No integer or fixed-point arithmetic is used anywhere, and the
decode is a pure rescale (`v * scale + offset`) with no dithering, because every
sample is consumed as a float. The host derives the constants (matrices,
scale/offset, luma weights) in `double` and narrows them once to `float`,
exactly as zimg does; the constants are ~40 words per image and no pixel data
is converted on the CPU.

The metric is defined on linear Rec.709, so HDR input is linearised with
zimg's display-referred scaling and no tone mapping or gamut compression: SDR
white is 1.0, PQ decodes to 0..100 (relative to a 100 cd/m^2 peak) and HLG to
0..10. `scene_referred`, `approximate_gamma`, `chromatic_adaptation` and
`peak_luminance` are fixed at zimg's defaults (off, off, off, 100) and are not
configurable.

Two deviations from zimg's own resampler/graph are deliberate, and follow from
the metric's alpha model: alpha is unpremultiplied in RGB space (zimg
premultiplies in the caller's plane space, i.e. on YUV values), and the
background blend is the metric's, applied after the matrix, rather than zimg's
premultiply/unpremultiply pair. The chroma upsampling is this implementation's
own bilinear/bicubic/lanczos evaluation of zimg's tap positions and edge
handling, not a bit-identical copy of zimg's resampler.

### Kernel specialization

`src/metrics/ssimulacra2/shaders/prepare.comp` is one source file compiled once per source
class, so each input path runs only its own code and registers: 4:4:4/GREY
carries no chroma filter, a linear-transfer kernel no EOTF, and a bilinear
kernel none of the six-tap sinc path. The axes are chroma (0 = 4:4:4/GREY,
1 = subsampled), transfer (0 = descriptor chain, 1 = sRGB, 2 = linear) and, for
subsampled input only, the chroma filter (0 = bilinear, 1 = bicubic,
2 = lanczos): 12 classes, each also built for the 16-bit plane mode. Datatype,
range, alpha type and the matrices are descriptor data rather than axes, since a
datatype branch costs one load plus two integer ops per sample.

## Deliberate differences from the reference

The reference is built on libjxl; it approximates several steps. This
implementation uses exact math instead of reproducing those approximations,
and a different, more accurate blur kernel; that is why the two scores are
close but not identical.

| Step | Reference | Here |
| --- | --- | --- |
| input conversion | libjxl's; sRGB-encoded 8-bit planes only | zimg's full input surface, decoded on the GPU (see *Input conversion*) |
| sRGB -> linear | Chebyshev rational polynomial (max error ~2e-8 absolute on 8-bit inputs) | exact piecewise `pow` |
| cube root in XYB | bit-trick + Newton iterations (6 ulp) | `pow(v, 1/3)` |
| colour management | embedded ICC profile run through lcms2 when the profile is not byte-identical to its internal sRGB | samples are decoded with their declared transfer function and primaries; `iCCP`, `gAMA`, `cHRM`, `sBIT`, JPEG ICC are ignored, and the CLI's `--input-*` flags supply the declaration where the container has none |
| blur | recursive Gaussian with Highway's unrolled/FMA kernels | sampled Gaussian of the same sigma, truncated at 4 sigma and normalized (13-tap separable FIR, evaluated in the fused scale dispatch) |
| pruned channels | every (channel, level) pair is evaluated | the seven whose six weights are all at most 0.01 are skipped; scores move by a mean of 2.7e-4 (see *Deliberately pruned channels*) |

Attribution: over the full set the mean deviation is 0.299 for PNG pairs and
0.406 for JPEG pairs. Nearly all of it is the blur kernel, not the exact-math
steps: the previous release reproduced the reference's recursive Gaussian and
measured 0.0269 over the same 4292 pairs (PNG 0.0255, JPEG 0.0371), a level
set by the reference's sRGB polynomial and 6-ulp cube root. The sampled
Gaussian is closer to the intended sigma = 1.5 filter than the IIR, but it is
not the filter the reference uses, so the two scores differ systematically:
ours is higher for every pair, by 0.31 on average and by up to 3.0 for the
highest-quality pairs, where the score mapping is steepest.

The blur is a 13-tap sampled Gaussian at the metric's sigma of 1.5, truncated
at 4 sigma and renormalized, applied separably in both directions. The
reference filters with a recursive (truncated-cosine IIR) Gaussian at the same
sigma; the sampled kernel has no design fit error and no serial recurrence, so
it is both closer to the intended Gaussian and cheaper to evaluate.

These differences leave the correlation with human opinion essentially
unchanged, at the cost of a small but real drop; see below.

## Vulkan backend

The metric runs the whole computation for one `Compute` call on a Vulkan
compute queue as up to nine dispatches at six scales: a *prepare* dispatch per
uncached image, which decodes
that image (chroma upsampling, matrix coefficients, alpha, transfer function,
primaries) and builds its **whole scale pyramid** -- the XYB plane of every
level, each level's 2x2 average taken from the one below it inside the same
workgroup, so the linear plane never reaches device memory; a *scale* dispatch
per level, its three channels in the grid's z (the five blurs, the SSIM' and
edge-diff maps and their norm partials); and one reduction that sums every
level's partials. Each
level's plane offsets, size and norm block live in a level table the prepare
and reduction dispatches read from the arena; the scale dispatch carries its
level's geometry as push constants instead. The entire pyramid is recorded
into one command buffer and submitted once, so the only host traffic after
submission is the 18 norm sums per scale. The host then converts those sums to
the 1-norm means and 4-norm fourth roots in `double`.

- **Build.** Required: configure fails without the Vulkan headers or `glslc`,
  and Apple platforms are rejected (no MoltenVK path). Shaders are compiled at
  build time and their SPIR-V is embedded in the binary, so no shader files are
  needed at run time. There is no surface, swapchain or macOS/MoltenVK path.
- **Device.** The first compute queue of a discrete GPU (falling back to any
  GPU) with at least 256 workgroup invocations and 16 KiB of shared memory.
  One `Ssimulacra2Pipeline` owns its buffers, command buffer and fence; the device and
  queue are process-wide. That singleton is destroyed from an exit handler
  registered when it is created, not from its static destructor: on the NVIDIA
  615.78 driver without a `DISPLAY` connection (any headless run) the
  static-destructor path calls `vkDestroyDevice` after the driver has torn
  itself down and dies with SIGSEGV *after* all output -- an exit code of 139
  for a scoring run that succeeded. The pre-fusion backend reproduced that
  static-destructor ordering failure. Later driver initialization can register
  additional cleanup, so the benchmark now releases the retained context
  explicitly after its pipelines are destroyed and before process-exit handlers.
  The library retains the exit-handler fallback.
- **Geometry follows the device.** The fused per-scale tile is built in two
  shapes: 32 x 32 output pixels (21 120 B of shared memory) over 256
  invocations and 16 x 32 (13 440 B) over 128; the pipeline reads
  `VkPhysicalDeviceLimits::maxComputeSharedMemorySize`,
  `maxComputeWorkGroupInvocations` and `maxComputeWorkGroupSize` and runs the
  first that fits. Both are the same source compiled per variant
  (`SS2_SCALE_ROWS`/`SS2_SCALE_OUT_W`/`SS2_SCALE_LANES`: the workgroup is 16 x
  lanes and each invocation owns four output rows, which the shader's `#if`
  enforces); forced onto the same GPU they agree to 1.2e-8 on every norm,
  because the tile only changes the summation *grouping* of the norm partials.
  A device with 16 KiB of shared memory gets the 16 x 32 tile. The tested
  RTX 5070 (48 KiB) and `lavapipe` (32 KiB) both selected 32 x 32.
  Historical 4K stage measurements informed the tile selection (resident, median
  of 20 iterations, three runs each, and re-checked with the level table in
  place): 32 x 32 over 256 invocations runs 1.8 ms, 16 x 32 over 128 runs
  2.0 ms, so the larger tile leads. Two more
  candidates were measured and rejected: a 32 x 64 tile (38 KB) at 1.65 ms, and
  512 invocations for the same 32 x 32 tile -- two output rows per invocation
  instead of four -- at 2.48 ms. That last one is the informative failure: it
  halves the rows per invocation, so shared loads per FMA go from 0.31 (16
  loads, 52 taps) to 0.54 (14 loads, 26 taps), and the stage feels that ratio
  rather than warp count. Losing a resident workgroup is worse still: the fused
  five-quantity variant needs 43.6 KB and measured 2.75 ms (see *Historical speed*). The
  rejected variant
  still agreed to 7.5e-9 on the norms and 7e-8 on the score against the
  capture, since depth regroups the partials too.
- **Memory.** Everything lives in one device-local buffer: the XYB pyramid
  (one plane set per level), the level table, the two raw source regions a
  streaming upload needs, the per-level norm partials, and one region per
  resident image. At 4K the arena measures 760 MB with the default two resident
  slots, and 507 MB with `Ssimulacra2Config::streaming_inputs = false`, which is
  what a caller that only uses `Upload` and `Compute` on resident handles
  wants: the pixel overloads then throw and the source regions are never
  reserved. The knobs
  that size it are `gpu_resident_images` (16 bytes per pixel per slot),
  `streaming_inputs` (32 bytes per pixel) and `input_bytes` (a pool that
  `AllocateImage` carves exactly `RequiredBytes(format)` from, for a caller
  that fills device memory itself); `half_planes` halves the pyramid only.
  Direct memory policy uses a host-visible device-local arena. The staged path holds two
  source slots when `streaming_inputs` is set, one slot for resident uploads
  when streaming is disabled, or none when neither path is configured, plus
  space for the descriptors and the norm readback. The source regions
  keep the file's own datatype, because nothing is converted on the host any
  more, so they are sized for the worst case of four `float` planes, 16 bytes
  per pixel; a byte image uploads 3 and a 16-bit one 6 or 8, and the shader
  decodes the samples in place. Without host-writable device memory there is
  also one host-visible staging buffer for those two regions (about 265 MB at
  4K) that the upload fills and `vkCmdCopyBuffer` drains. A resident slot
  costs `4 * width * height` floats, 16 bytes per pixel. The upload writes into that buffer directly when the device exposes
  `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` memory (resizable BAR: one
  host-side pass over PCIe, no staging buffer, no copy); otherwise a mapped
  staging buffer and `vkCmdCopyBuffer` take over and only the norms come back.
  A geometry whose arena exceeds `maxStorageBufferRange` is rejected with
  `std::invalid_argument`, and any Vulkan failure surfaces as
  `std::runtime_error` from the `Ssimulacra2Pipeline` constructor.

**Historical speed before resident-reference caching.** These measurements
describe the named builds and workloads. On a 3840x2160 pair, the metric alone
(`core_ms`, file decoding excluded) took **3.6 ms** streaming and **1.21 ms per pair** with the images
resident (823 pairs/s; 1.13 ms and 882 pairs/s with `half_planes`), against
**250.5 ms** on the AVX-512 tier retained in the `vulkan-only-checkpoint` tag
(RTX 5070, driver 615.78, resizable BAR). Streaming is 8 iterations after 2
warm-up runs, resident 20 after 3. The card's sustained clock moves by up to
10% between sittings -- one build measured 2.25 and 2.63 ms per resident pair
minutes apart -- so the numbers that separate two versions here were taken
back to back, interleaved. Measured that way against the binary before the
fused pyramid, the resident pair went from 2.45 to 2.25 ms (-8%, 408 to 445
pairs/s) and the prepare stage from 0.79 to 0.61 ms (-23%); the scale stage did
not move outside the noise. Against the binary before the pruned channels, the
same pair goes from 2.25 to 1.21 ms (-46%), prepare from 0.61 to 0.42 ms (-31%)
and scale from 1.72 to 0.74 ms (-57%), five runs per arm. One resident capture:

| stage | streaming ms | resident ms |
| --- | --- | --- |
| prepare (whole pyramid per image: raw sample -> linear -> XYB of the kept channels per level, 2x2 averages in registers; includes any upload copy) | 0.43 | 0.42 |
| scale (per level: five blurs, both maps, norms, for the channels the score can see) | 0.81 | 0.74 |
| untimed (upload memcpy, submit/wait, result) | 2.3 | 0.04 |
| **total** | **3.5** | **1.21** |

After resident-reference caching, five interleaved 4K runs on the same class
of device measured prepare at 0.42 -> 0.21 ms (both ranges rounded to those
values) for a repeated resident pair. The current benchmark forces both images
to be prepared, so its resident numbers are not directly comparable to that
cached-reference result. See `bench/notes/system/vship-notes.md` for the full stage tables
and the separate streaming and mixed-reference arms.

Streaming is upload bound: the untimed cell is the PCIe write of both images,
and an image that is already resident skips it entirely. `prepare` came down
from 0.99 ms over four changes -- an 8-bit image now uploads 3 bytes per pixel
instead of 16, the fused pyramid dropped the linear plane, the pruned channels
are no longer stored, and the cube root lost a sign branch its callers had
already made dead -- and with 4:2:0 input the chroma filters put it
at 1.09, 2.79 and 4.82 ms for bilinear, bicubic and lanczos in the
pre-pruning build.

The traffic is what both stage rewrites were about. Counted per pair, the
original split chain moved 4.63 GB through VRAM: 11.34 plane reads and 5 writes
for the five blurs, 9 more plane reads for the two map shaders, plus the whole
linear plane of every scale passed between three elementwise dispatches (19
float moves per pixel per image). Fusing the whole scale into one workgroup
left 0.50 GB. The prepare dispatch then built all six levels of each tile in
one workgroup. Within a 32 x 32 tile of base pixels, the 2x2 average
never leaves a register or the workgroup's shared ladder, and the stage moves
7 floats per base pixel per image -- 3 read from the source, 3 written at the
base level and 3/4 + 3/16 + ... for the rest -- against the 19 of the
elementwise chain. That put the pair at 0.80 GB of plane traffic -- 265 MB
written by prepare, 506 MB read by the scale dispatch's two source tiles -- on
the way to 1.8 GB for the split chain.

Pruning the channels the score cannot see (see *Deliberately pruned channels*
below) cuts both sides of that again: at 4K prepare writes 1.68 floats per base
pixel per image instead of 4.0 (112 MB for the pair) and the scale stage reads
its tiles for 13 785 workgroups instead of 32 691 (214 MB), so the pair moves
about 0.33 GB of plane traffic, 0.16 GB with 16-bit planes. The allocation does
not shrink -- the arena still holds a plane set per channel per level -- so the
*Memory* bullet above is unchanged.

The scale dispatch is the one that stopped being bandwidth bound: both passes
read four source values per output instead of the thirteen a plain 13-tap loop
reads, about 50 shared reads per output pixel per channel, and the stage ran at
1.7 ms before the pruning and 0.74 ms after it, against the 4.9 ms the same
arithmetic took before that rewrite. What limits it now is not the shared-load
count but how many workgroups the shared budget leaves resident. The occupancy
probe in `bench/notes/system/vship-notes.md` answers that by padding the block: at 21 120 B
the driver keeps at least three blocks per SM, 24 warps of the 64 the SM can
hold, the first step down is between 25 216 and 29 312 B, and every lost block
costs 13-19% of the stage. Two blocks, which this paragraph used to assume from
the 48 KB per-workgroup guarantee, is wrong -- the guarantee is per workgroup,
not per SM.

Three further attempts on that stage were measured and dropped. Giving an
invocation two output rows instead of four (512 invocations for the same 32 x
32 tile, same 21 120 B of shared memory, twice the warps) took the stage to
2.48 ms: shared loads per FMA go from 0.31 (16 loads, 52 taps) to 0.54 (14
loads, 26 taps). Replacing the norm ladder's eight halving stages with a
two-stage reduction -- eight contiguous blocks summed by eight threads, then
the eight block sums on thread 0, which saves six barriers -- took it to
1.68 ms; that variant also reads every block at a stride of 32 floats, so all
lanes of a warp hit one bank and the number does not separate the barrier
saving from the conflicts. The third was the shared-load ratio itself: the five
quantities' horizontal passes fused into one walk over the tile, loading the 16
source values once instead of once per quantity (32 shared loads instead of
80), keeping the squares and the cross product in registers, and writing five
planes so the vertical pass could read each one with no barrier between them --
ten barriers per channel down to three. It needs five mid planes, 43.6 KB for
the 32 x 32 tile and 27.8 KB for the 16 x 32 one, and measured 2.75 ms and
3.68 ms against the 1.76 ms of the unfused stage in the same sitting, at
identical scores. Both shapes sit past both occupancy steps the probe maps
(26-29 KB and 34-38 KB), so each loses two resident blocks: 26-38% of the stage
before their own extra traffic is counted, and the rest of the 56% is that
traffic -- five mid planes written per quantity instead of one. Losing the
blocks costs more than the 35% of shared loads the fusion saves, and the fusion's
extra traffic widens the gap rather than narrowing it. The shader is reverted;
the
numbers stay here because they are the measurement that says the shared-load
ratio is not the lever on this device.

In that capture, prepare wrote 112 MB and read 50 MB of source in 0.43 ms,
about 377 GB/s of the card's 672 GB/s. Scale read 214 MB in 0.74 ms, and
half-precision plane storage reduced the stage to 0.65 ms, about 12%. The
padding probe above measured at least three resident workgroups at 21,120 B.

See `bench/notes/system/vship-notes.md` for the Nsight Systems profile of this stage split
and the comparison with the GPU peers Vship and Vship-fast; `AGENTS.md` carries
the measurement protocol and the gates.

**Resident images.** `Ssimulacra2Pipeline::Upload` copies one image into a device slot and
`Ssimulacra2Pipeline::Compute(UploadedImage, UploadedImage)` scores two slots: the
per-pair pixel upload leaves the hot path, while color conversion and pyramid
construction remain in the scoring call. The library reuses the original's
prepared pyramid when its reference handle is unchanged. Slots are recycled
(`Release` puts one back) and each handle carries an upload generation, so a
handle that outlives its `Release` is rejected instead of silently scoring
whatever image took the slot next. The benchmark alternates reference handles
to measure full preparation of both inputs.

**Precision.** The GPU accumulates the norm sums in `float` and evaluates XYB's
cube root as `pow(v, 1/3)`, so scores are close to, not bit-identical with, the
AVX-512 tier this backend replaced: measured on the fixtures and on 33x17,
1919x1079 and 3840x2160 synthetic pairs, every norm the backend computes stays
within 3e-6 of that tier's result (it lives in the `vulkan-only-checkpoint`
tag) and the score within 1e-3. The seven pruned (channel, level) pairs are the
exception by construction: their norms are published as zero, as
*Deliberately pruned channels* below describes, which is the one place a norm
is not the tier's value. Both error maps divide the *difference* of
numerator and denominator (`(denom - num*num_s) / denom`, `(detail_dist -
detail_orig) / detail_orig`) instead of subtracting a quotient from 1, so equal
planes cancel exactly before any division rounds and identical inputs score
exactly `100` on the discrete-GPU drivers tested. Decoding the source datatype, the
transfer function, the matrix and the primaries conversion inside the prepare
pass did not move byte sRGB input: the five-pair capture this section was built
on (fixtures, alpha, 33x17, 1919x1079, 3840x2160) reproduces it to within
4e-4 on every pair, the alpha pair moving by one float rounding step more,
because the blend now sits inside the conversion pass rather than in a
channel loop of its own. The 4e-4 is the pruned channels of
*Deliberately pruned channels* below, which are the only remaining difference
on the four non-alpha pairs: against the unpruned build this capture moved by
1.7e-4 to 2.4e-4. The fused scale changes only the *grouping* of the norm partials (four
output pixels per invocation instead of one), not any blurred value: against
the pre-fusion capture the 108 norms move by at most 1.6e-8 and the score by
3e-7 -- and the tile variant moves them by the same order. The two attempts
that followed only regrouped the partials as well: the two-row depth variant
measured 7.5e-9 on the norms and 7e-8 on the score, the two-stage norm
reduction 3e-8 on the norms (1.7e-6 on the score for the alpha pair, where the
steep mapping amplifies a saturated Y 1:4 weight of 225). Neither touched a
blurred value, so both were performance questions only. Two
attempts to tighten the remaining 2.9e-6 were measured and rejected — one
Newton step on the cube root made it worse (3.5e-6), and Kahan compensation in
the partial reduction changed none of the 108 norms — so 2.9e-6 is where
`pow`-based XYB plus float accumulation lands. A software ICD can still differ
in the last ulp of a norm, which the steep score mapping turns into
`99.99...`; the error stays bounded by that rounding either way.

**Deliberately pruned channels.** Seven of the 18 (channel, level) pairs carry
weights of at most 0.01 in every one of their six score terms, so the maps they
produce cannot move the score by more than a hundredth of a norm:
`Ssimulacra2ChannelPruned` in `src/metrics/ssimulacra2/score.cc` finds them from the weight table itself
and the backend skips them -- prepare does not store those planes, the scale
dispatch returns before reading a tile, and the reduce dispatch writes those
norms as zero instead of summing partials nobody wrote. At 4K with six levels
the refused pairs are X at 1:1, B at 1:1, B at 1:2, B at 1:4 and all three
channels at 1:32, which is 58% of the scale stage's workgroups and 46% of the
resident pair's time (2.25 to 1.21 ms). The set is a property of the scale
count, not of the level index: `Score()` reads the weight table as one flat run
of `scales` rows per channel, so a four-level pair like the 32x32 fixture is
pruned differently -- X at 1:1, Y at 1:2 and B at 1:8, because those levels land
on rows that belong to another channel's label -- and the mask is rebuilt for
every geometry. In the historical 300-pair CID22 comparison, pruning moved the score by a mean
of 2.7e-4 and at most 8.1e-4, KRCC, SRCC and PCC are unchanged to four
decimals, and identical inputs still score exactly `100.00000000`, because
every pruned map value is exactly zero there and pruning removes only zeros.

**Half-precision planes.** `Ssimulacra2Config::half_planes` (CLI:
`--half-planes`) stores the plane region as `float16_t` and runs the second
SPIR-V build of the plane shaders, which halves the plane bytes of every
dispatch. Nothing else drops precision: the decoded sRGB pixels, alpha, the
resident slots and the norm partials stay `float32`, and the 16-bit type is
only ever loaded, stored and converted, never used for arithmetic. The cost is
the rounding of the plane values themselves, about 5e-4 relative.

In historical measurements on the fixtures, alpha/33x17 and the 1919x1079 and 3840x2160 pairs:
identical inputs still score exactly `100.00000000` (the two planes are rounded
the same way, so their difference still cancels exactly), and every norm stays
within 3e-4 of the exact run. The score moves downward by 0.01-0.06 on the
fixtures and the 33x17 pair, 0.24 on the 1919x1079 pair and 0.31 at 4K -- the
consequence of the ringing/blurring map being a difference of two nearly equal
terms, so what the rounding adds lands in the tail of its 4-norms.
`TestHalfPlanes` pins those numbers: a larger delta is a defect in the half
path, not a tolerance to relax. In exchange, a resident 4K pair runs at 1.13 ms
instead of 1.21 ms, the scale stage dropping from 0.74 to 0.65 ms while the
prepare stage holds at 0.42 ms: with the pruned channels gone, what prepare
writes is small enough that halving it no longer shows. At 1080p the flag buys
nothing measurable at all (0.33 ms per resident pair either way, the scale stage
0.15 ms in both): the pair carries a quarter of the plane bytes while the nine
dispatches, their launch latency and every workgroup's prologue cost the same,
so the share that half precision could halve fell below the noise. This run
measured about a 6% gain at 4K, compared with 1.2x in an earlier build before
pyramid fusion and channel pruning.
CID22 validation with the flag: see *Half-precision planes* under Validation.

*Historical FP16 experiments.* The tested card has `shaderFloat16`
(fPacked HFMA2 at twice the f32 rate), so the scale stage was measured with it
twice, and both variants stay out of the tree:

- **f16 arithmetic** (tile, horizontal scratch and blur accumulators as
  `float16_t`): the scale stage drops from 1.58 ms to 1.25 ms in the build
  that measured it, and a resident 4K pair from 2.63 ms to 2.01 ms (498
  pairs/s), but the 32x32 golden pair
  falls from 93.17 to **63.44**. That is not drift to pin, it is the metric
  breaking, and the damage is not where it is tempting to assume. Rounding
  *only* the five blur results to f16 precision, with all arithmetic still f32,
  splits it (same binary, same images):

  | rounded to f16 | 32x32 gradient_B | 32x32 gradient_C | 4K pair |
  | --- | --- | --- | --- |
  | nothing (f16 planes only) | 93.144 | 92.068 | 85.604 |
  | horizontal scratch only (internal, f32 arithmetic) | 83.481 | 83.701 | -3.765 |
  | `mu1`, `mu2` (blur of the images) | 93.098 | 92.058 | 79.441 |
  | `s11`, `s22`, `s12` (blur of squares and cross product) | 74.300 | 74.178 | -40.351 |
  | all five | 74.320 | 74.208 | -4937.995 |

  `mu` alone costs little at 32x32 and 6 points at 4K, and it is the *second*
  moments that destroy the score. SSIM' needs the local variance
  `s11 - mu1*mu1` in a denominator of `(s11 - mu1^2) + (s22 - mu2^2) + C2`,
  with `C2 = 9e-4`: a difference of two blurs of order 0.2-0.5 that has to
  resolve a quantity of order 1e-4-1e-3. The half-precision quantum at 0.2 is
  already ~6e-5, so rounding `s11` injects an error of the size of the variance
  itself; the denominator can shrink toward zero or flip sign, and the map
  divides by it -- which is why one 4K pair's score goes to -4938 instead of
  drifting by a few percent. In score-mapping terms the weighted error sum goes
  from `((100 - 93.17)/10)^(1/0.628) ~= 0.55` to `(36.56/10)^(1/0.628) ~= 7.9`,
  so the metric's own error estimate inflates about 14x.

  That also settles mixed precision: the horizontal scratch is the *cheapest*
  internal buffer to round and it still costs 10 points at 32x32 and 89 at 4K,
  so no buffer or operation inside the blur can be 16-bit -- not the
  accumulators, not the scratch, not the results. Only the plane storage can,
  because those values are already 16-bit. And since the 1.26x comes from
  keeping the whole accumulation chain packed in f16 (HFMA2), a design that
  narrows only some operations would pay an f16/f32 convert per operation and
  lose the packing: it would be no faster than the f32 path.
- **f16 tile storage with f32 arithmetic**: the tile values are exactly the
  16-bit plane values, so this is error-free by construction (scores reproduce
  the plane-rounding numbers bit for bit), and it makes the half build's shared
  footprint 19 520 B instead of 21 120 B. It measured no speed at all -- 1.51 ms
  against 1.53 ms for the scale stage, inside the run-to-run spread -- because
  the stage is not limited by shared-memory bytes, and the widening conversion
  on every load costs what the narrower loads save. Also rejected, as
  complexity without a return.

The scale stage is therefore limited by its instruction stream (five 13-tap
chains per output, about 50 shared reads per output pixel per channel, a
barrier at every reuse of the scratch) rather than by any memory throughput, so
a narrower type cannot help; the one decomposition that *would* cut those reads
was measured slower, because its five mid planes cost a resident workgroup (see
*Historical speed*).

**Historical software ICD results.** On the tested `lavapipe`, a 512x512 pair took 9.6 ms of core time (9.4
ms per pair with resident images), slower than the 7.4 ms AVX-512 tier recorded
in the `vulkan-only-checkpoint` tag, so a real GPU is what makes the backend
worth using. That lavapipe build reported the spec-minimum 128 MiB
`maxStorageBufferRange`, which caps the geometry at about one megapixel: with
the default two resident slots 1024x1024 fit and scored, while 1919x1079 was
rejected with the documented `std::invalid_argument`. It ran the 32 x 32
tile variant, because it reports 32 KiB of shared memory; a 16 KiB device gets
the 16 x 32 one, which is how the second variant stays covered.

## Butteraugli

`include/butteraugli.h` exposes `ButteraugliPipeline` beside `Ssimulacra2Pipeline`, with
the same geometry, `Upload`, `Submit`/`Finish` and caller-owned-buffer
contracts, running libjxl's Butteraugli as the metric (`--metric butteraugli`
in the CLI, `--metric butteraugli` in the benchmark). src/metrics/butteraugli/tables.cc and every
shader carry the libjxl line of each constant; `.reference/libjxl` is the
BSD-3-Clause source this is translated from, and nothing comes from the Vship
checkouts.

What it computes (butteraugli.cc line references in the sources): two levels --
the base and one 2x2 box subsample of the *linear* rgb, folded as
`0.85*base + 0.5*sub[y/2][x/2]`, which is exactly what Diffmap (:1829) folds and the deeper
`sub_` chain `Make` builds never contributes (:1854-1855).
Each level runs opsin dynamics (a sigma-1.2 mirror-bordered blur, the photopsin
absorbance matrix with its clamps, and the per-pixel log-gamma sensitivity),
the LF/MF/HF/UHF separation (sigmas 7.1559, 3.2249 and 1.5642 with
truncate-and-renormalize borders), six Malta band terms (the UHF unit keeps
libjxl's four duplicated line terms on purpose: dropping them under-weights
UHF), the asymmetric L2 terms, the psycho-visual mask with its 3-pixel
erosion, and one combine. There is no final blur.

The score is the diffmap's maximum, so identical images score exactly `0.0` and
larger is worse; `--raw` also prints `distance_p3`, libjxl's
ComputeDistanceP(3) aggregate. Precision: fp32 arithmetic on the device with
double-precision host constants (kernel taps, Malta prefactors). Tests pin the
agreement with a line-cited double-precision CPU port (`tests/metrics/butteraugli/butter_ref.cc`,
test-only). Gradient and spatial-frequency/chroma/directional cases permit
0.1% relative score and `distance_p3` error; the spatial cases also compare
streamed and cached resident inputs. A 1x1 anchor reproduces libjxl's
published band (`2.065716`;
the note in `tests/metrics/butteraugli/butter_test.cc` explains why the band is
loose). The standing oracle keeps the core tests independent of a libjxl
build; pinned libjxl is used for external checks.

The earlier 5% gate on two smooth gradients missed three implementation
defects. The 2026-11-06 correction forms HF from the raw MF blur before range
adjustment, restores `SuppressXByY` before the UHF split, and corrects four
UHF Malta tap signs. Full-CID22 production validation reduces mean absolute
score error from `0.0758241` to `0.0002853`. The remaining large tail comes
from MozJPEG ICC interpretation: our file loader assumes declared sRGB while
libjxl honors embedded profiles. See the [Butteraugli guide](butteraugli.md)
and [investigation](../bench/notes/system/butteraugli-drift.md) for residuals,
oracle evidence and validation results. Historical performance and corpus
reports below retain the values from their original builds.

`hf_asymmetry`, `xmul` and `intensity_target` are `ButteraugliConfig` fields,
with libjxl's defaults of 1.0/1.0/80 nits. Vship's 203-nit target is a different
default used only for throughput comparisons.

Historical cost before the fused erosion/combine and opsin tile changes: **11.1 ms/pair (1.34 ms/MPx)** at 4K resident; one 20-iteration run
splits it opsin 2.01, separate 2.53, malta 2.86, diff 1.50 (the L2 terms plus
the mask's precompute, one pass), mask 0.90, combine 0.70, reduce 0.23, untimed
0.35 ms. The corresponding 1080p measurement was 2.46 ms/pair. Three measured steps took the pair from 16.0 to
11.1 ms with the score unchanged: the blur taps moved from arena loads inside
every loop into registers (`separate` -42%), both blur passes took scale.comp's
shape so four outputs share one window of source values (`separate` -25%, the
pair -12%), and the L2 terms fused with the mask's precompute. Twenty-four
dispatches scored a cached pair, 32 when the reference was rebuilt. The current
pipeline uses 22 and 30 respectively. Five interleaved 4K runs after the opsin
and fused erosion changes measured the resident pair at 10.6 -> 9.2 ms
(ranges 10.6–10.6 and 9.2–9.2 ms at printed precision) and the streaming pair
at 17.3 -> 14.9 ms (ranges 17.2–17.3 and 14.9–14.9 ms). The reference
image's separated bands are cached, so a resident reference scored against many
distorted images pays for it once, while `Compute(Image, Image)` re-uploads and
rebuilds it every call. The historical Vship 5.1.2 CUDA comparison measured
2.7x resident and 1.66x streaming; it has not been repeated for this build.
`bench/notes/system/vship-notes.md` has the table and `bench/vship_bench.cc` is the driver
that produced it. What is left is Malta's window traffic: a 32x32 output
tile reads a 40x40 window (1.56x halo amplification), and the whole Malta unit
is only 8% of that stage, so a wider tile is the only lever there.

## Validation

`validation/fetch_cid22.py` downloads and unpacks the CID22 validation set
(49 originals, 4292 distorted images, ~1.4 GB, CC BY-SA 4.0).
`validation/run_corpus.py` scores the selected implementations, then reports
cross-implementation agreement and correlation with the subjective MCOS scores
shipped with the data set. The pinned references and their harnesses are built
under `build/`. The following command reproduces the legacy MCOS diagnostic.

```sh
python3 tools/fetch_references.py --only cid22,ssimulacra2
python3 tools/build_references.py --only cloudinary
.venv/bin/python validation/run_corpus.py --metric ssimulacra2 --only ours,cloudinary \
  --accuracy-tolerance 0.005
```

`--limit N` runs a smoke subset; `--strict-parity` additionally demands the
`--max-threshold`/`--mean-threshold` cross-implementation bounds. For current
acceptance, see the [reference correlation guide](validation.md) and the
[frozen balanced CID22 policies](../validation/fixtures/cid22-balanced/README.md).

### Historical results (full CID22 validation set, 4292 pairs, 0 failures)

| Metric | Value |
| --- | --- |
| max ǀour score − reference scoreǀ | 2.9907 |
| mean ǀdiffǀ | 0.3120 |
| median ǀdiffǀ | 0.2393 |
| p99 ǀdiffǀ | 1.4320 |
| RMSE | 0.4075 |
| pairs within 0.01 | 0 / 4292 |

The mean deviation is 2.3% of the standard deviation of the MCOS scores (13.3
points). It is a systematic offset rather than noise: every pair scores higher
here, because the sampled Gaussian and the reference's recursive filter have
slightly different frequency responses (about 1% apart in the mid band, where
the recursive design fits the intended Gaussian least well).

Correlation with the subjective MCOS scores:

| Implementation | KRCC | SRCC | PCC |
| --- | --- | --- | --- |
| reference binary | 0.7072 | 0.8901 | 0.8784 |
| this implementation | 0.7046 | 0.8885 | 0.8765 |

The three correlations sit 0.0016–0.0026 below the reference's. That is small
next to the sampling error of a single correlation at n = 4292 (standard error
≈ 0.010 for KRCC and ≈ 0.003–0.004 for SRCC/PCC at these levels). The blur
kernel reduced agreement with MCOS in that run. The legacy runner above still
reports MCOS and numerical parity diagnostics for newly built binaries.

### Half-precision planes (`--half-planes`)

These historical measurements compare the default and `--half-planes` paths
over the same 4292 pairs. To collect both score columns with the current runner:

```sh
.venv/bin/python validation/run_corpus.py --metric ssimulacra2 \
  --only ours,ours-half-planes,cloudinary
```

| Metric | Value |
| --- | --- |
| max ǀexact − halfǀ | 0.5636 |
| mean ǀdiffǀ | 0.0660 |
| median ǀdiffǀ | 0.0538 |
| p99 ǀdiffǀ | 0.2685 |
| RMSE | 0.0859 |
| pairs differing by > 0.01 | 3978 / 4292 |
| pairs differing by > 0.1 | 819 / 4292 |

Correlation with MCOS, which is what the flag protects:

| Implementation | KRCC | SRCC | PCC |
| --- | --- | --- | --- |
| exact planes | 0.7046 | 0.8885 | 0.8765 |
| 16-bit planes | 0.7051 | 0.8888 | 0.8768 |

The 0.0003-0.0005 differences are far inside the sampling error of a single
correlation at n = 4292 (≈0.010 for KRCC, ≈0.003-0.004 for SRCC/PCC), so the
flag does not measurably move the metric's agreement with human opinion; it
moves individual scores by 0.05 points (median absolute) and up to 0.56. The
shift is downward on 3991 of the 4292 pairs (mean signed −0.061) and upward on
the other 301: the rounding enters the ringing/blurring map, a difference of
two nearly equal terms, so it lands in the tail of that map's 4-norms and, for
near-lossless pairs whose 4-norms are pure noise, can go either way.

## Performance

The [benchmark guide](../bench/README.md) covers the hosted-video suite,
still-image comparisons, individual workloads and profiling commands.
The [performance guide](performance.md) explains the timing boundaries,
interleaved comparisons and how to interpret recorded measurements.

`vqm_bench` builds with `VQM_STAGE_TIMING` to report GPU stages. The CLI, tests
and installed core use the untimed library. The `vulkan-only-checkpoint` tag in
the original Git history preserves the scalar, AVX2, and AVX-512 comparisons.

## Style

Google C++ layout with four-space indentation, an 81-column limit, and braces
attached to the declaration line. A parameter or argument list that does not
fit on one line is broken after the open bracket, packed up to the column
limit, and closed by a bracket on its own line:

```cpp
void axis_taps(
    uint out_pos, float scale, float shift, uint dim,
    out float weights[VQM_TAPS], out int indices[VQM_TAPS], out float total
) {
```

The maintained check requires clang-format 23 for `.clang-format`'s bracket
options. The configuration fixes pointer alignment and preserves include order
for both C++ and GLSL. Format changed files with clang-format, then run the
recursive format and production lint checks:

```sh
cmake --preset release
python3 tools/check_quality.py --lint
```

Functions may nest at most three blocks (three nested loops, loop/loop/if, …).
clang-tidy counts the function body as one level, so the bound is
`NestingThreshold: 4` in `.clang-tidy`. The Release preset provides the compile
database for `check_quality.py`.

Python uses Ruff for linting and YAPF for formatting. See
[contributing](../CONTRIBUTING.md#formatting-and-provenance) for local commands.

## Repository layout

The [architecture guide](architecture.md) describes ownership and dependencies.
`tools/references.lock` pins original and peer source revisions. Fetch them
with `tools/fetch_references.py`; the build tool copies native references
under `build/` and never builds inside `.reference/`. See
[provenance](README.md#provenance) for upstream licenses and the distinction
between derived metric code and peer API comparisons.

`tests/make_goldens.py` writes manual CLI fixtures under `build/fixtures`.
Core tests construct their images in memory; input and CLI integration tests
generate temporary images and sequences and remove them afterward. Local
core builds and tests do not require the external corpora or reference
checkouts. The independent checked-in numerical oracles and qualified-domain
reference fixtures retain their provenance under `tests/metrics/` and
`validation/`; see [validation](validation.md) for acceptance rules.
