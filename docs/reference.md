# VQM API and Vulkan implementation

VQM operates on decoded planar samples or caller-owned Vulkan buffers.
This guide covers the shared input and resource contracts, plus the
SSIMULACRA 2 and Butteraugli compute paths. The individual metric guides
describe each score and its reference implementation.
[Metric guides](README.md) · [Unified metrics API](metrics.md)

## Input formats

`Image` carries a `ImageFormat` descriptor next to its packed sample buffer, and
the GPU decodes its samples using the descriptor fields below.

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
intermediate planes as 16-bit floats. Both
`Upload` and the handle overload of `Compute` throw `std::invalid_argument`
when the pool is exhausted, when a handle has been released, or when the
geometry does not match.

## Shared input conversion

SSIMULACRA 2 and Butteraugli convert input into linear Rec.709 on the GPU.
The shared input decoder uses FP32 arithmetic for every sample type. An integer sample is
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

The shared decoder linearizes HDR input with
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

## Vulkan execution

By default, VQM selects a Vulkan compute device and shares one queue
across pipelines. Applications can select the library-owned device
before first use or supply their own device and queue. Each pipeline
fixes its geometry and storage layout during construction and allows
one submission in flight. It records metric-specific passes and reads
back reduced results for host-side score conversion.
[Device selection](../include/vqm_device.h) ·
[Vulkan synchronization](vulkan-synchronization.md)

Streaming pipelines stage or directly upload the source's native
samples. Resident images retain device storage across calls. The
SSIMULACRA 2 pyramid and Butteraugli frequency bands can be cached for
an unchanged resident reference. Borrowed Vulkan buffers skip host
upload. Workgroup and buffer limits determine which layouts fit.
[GPU memory planning](memory.md) · [SSIMULACRA 2](ssimulacra2.md#gpu-pipeline) ·
[Butteraugli](butteraugli.md#gpu-pipeline)
