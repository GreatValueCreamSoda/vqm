# CVVDP implementation and qualification

2026-11-06. This implements the [integration plan](../../../docs/design/cvvdp-implementation-plan.md)
in the existing uncommitted audit/reorganization tree. The installed core now
supports the original base ColorVideoVDP v0.5.7 for images and ordered video;
it does not invoke a reference implementation at runtime.

## Scope and numerical contract

The original is pinned at `2a268bce8d56e2f3abde46df3927d8a633707a24`.
Calibration, CSF LUT, color matrices and display presets are generated from
that source with the original MIT notice. Vship remains read-only GPU design
research and an optional public-C-API comparison. No peer implementation code
is copied. Its MATLAB interface wraps Python, so the independent numerical
oracle is the original CPU FP32 PyTorch implementation.

The implementation contains metric-local physical display conversion, spatial
pyramids/CSF/mutual masking, epsilon-adjusted norms, temporal filter generation,
bounded owned history and whole-clip JOD pooling. It supports explicit viewing
conditions, SDR, PQ, HLG and absolute-linear samples. Identity is exactly 10;
negative scores are valid. A clip JOD is distinct from an average of frame JODs.
The original mixed-parity pyramid edge behavior is intentionally retained.

`CvvdpPipeline` accepts owning images, strided host views, resident images and
borrowed Vulkan buffers. The first pair is deferred until the stream's image
versus video mode is known. Inputs are consumed before Push returns, while the
history remains private. Finalize is repeatable and Reset retains uploaded
handles. Unified `Metrics` exposes a typed final clip result alongside any
pending VMAF tail, preserving frame alignment and semaphore ownership.

The public guide is [docs/cvvdp.md](../../../docs/cvvdp.md); reference and fixture
reproduction is in [CVVDP development](../../../docs/cvvdp-development.md).
Shaders and host code are colocated under `src/metrics/cvvdp/`, with coherent
pipeline, display, model and spatial responsibilities. The core has no Python,
PyTorch, MATLAB, FFmpeg or video-decoder dependency.

## Correctness and resource gates

- Release: **224 tests, zero failed checks**, with a GPU required.
- ASAN/UBSAN: **224 tests, zero failed checks**, GPU required,
  `ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1`.
- Khronos hardware synchronization validation: **224 tests**, zero errors.
- Software Vulkan full suite: **222 tests**, zero failures; the final CVVDP
  subset additionally runs **18 tests** against the final library, with
  synchronization validation and all 29 end-to-end oracle cases.
- CTest/package/embedded consumer/CLI/tool workflows: **14/14** passing. A
  separate fresh library-only configure/build passes without reference runtime
  dependencies. The installed consumer includes `cvvdp.h` and links its symbols.
- **29** original end-to-end cases cover images, short/long clips, one-frame
  deferral, ring wrapping, fractional/integral frame rates, 1/120/240 fps,
  HDR/P3/linear, tiny errors, exact identity and a negative video JOD. The score
  gate is absolute `1e-3` JOD; all original CPU repetitions have zero spread.
- **10** independent spatial feature cases pass on hardware and llvmpipe at
  `5e-6` absolute or `5e-5` relative tolerance. Coefficients are checked against
  original odd-length inverse FFT outputs at `2e-7` absolute tolerance.
- Direct/Staged paths, padded/borrowed buffers, immediate caller-buffer reuse,
  binary waits/signals, source switching, mixed-metric finalization, invalid
  configuration, non-finite input recovery and per-heap budgets are tested.
- SSIMULACRA 2 CID22/Cloudinary 300-pair accuracy regression gate passes with
  tolerance `0.005`; identity remains `100.00000000`, Butteraugli identity
  `0.00000000`, and CVVDP identity `10.00000000`.
- Formatting is clean; clang-tidy checks **47** compiled production C++ files.
  New Python/reference/quality tools pass Ruff and YAPF. Two legacy benchmark
  runners retain **214 pre-existing Ruff diagnostics**, the same count before
  and after this work; their YAPF reflow preserves the AST apart from the
  intended metric-selection changes. No broad legacy Python lint cleanup is
  claimed.

The full 4,292-pair comparison passes with maximum absolute error
**9.6611084e-7 JOD**, against the 1e-3 gate. It is recorded separately in
[reference qualification](cvvdp-reference-qualification.md), including the exact
final executable hash and decoded-input provenance.

Temporal histories use two device-local auxiliary allocations. Queries and
actual allocations agree, including heap/type/alignment charges, for 1080p and
4K at 24/30/60/120 fps on this RTX 5070. With streaming inputs and no resident
slots, 1080p60 uses 287,677,248 arena bytes plus 846,028,800 history bytes; 4K60
uses 1,150,682,368 plus 3,384,115,200. The 4K120 case also constructs successfully
on this device (6,171,033,600 total history bytes). Other devices may reject it
at their per-buffer descriptor limit. The reusable
[memory probe](../../experiments/cvvdp/README.md) checks each allocation against
its query; raw output is `artifacts/reports/cvvdp/memory-matrix.txt`.

## Initial performance baseline

RTX 5070, driver 615.78.08, GCC 16.2.1, glslc 2026.4, release/O3, one worker.
The exact measured candidate SHA-256 is
`51e144d5a07a082a9ef8dccf3eaf21ebf40ca5268bbeff2238374bc799ac5568`.
The display is standard_4k: 200 nits, contrast 1000, 250 lux, reflectivity .005,
75.4024493452 pixels/degree. Physical display geometry is independent of image
geometry. Raw stage tables, scores and checksums were checked for every run.

The following independent baselines use five runs each. Stills use the existing
synthetic RGB8 PNG pairs, resident 20/3 or streaming 8/2 iterations/warm-ups.
Video uses 32 genuinely changing 30-fps Y4M frames, eight complete timed clips
and two warm-up clips per run, with a fresh/reset temporal state each clip.
Decode and construction are excluded; video includes uploads and all clip
processing, with both ms/frame and ms/clip in its raw CSV. These separate
workloads are not a paired estimate of upload cost or image/video overhead.

| Geometry | Workload | Median ms/image or frame | Five-run range ms |
| --- | --- | ---: | ---: |
| 1920x1080 | resident | 2.859239 | 2.385111–4.355673 |
| 1920x1080 | streaming | 2.703861 | 2.609583–2.837505 |
| 1920x1080 | video | 4.126857 | 4.111876–4.183651 |
| 3840x2160 | resident | 8.232842 | 8.215782–8.508133 |
| 3840x2160 | streaming | 10.408051 | 10.395906–10.434380 |
| 3840x2160 | video | 17.571085 | 17.402405–18.077849 |

1080p video is about 242 frames/s and 4K video about 57 frames/s on these clips.
Resident 1080p runs have a broad spread, retained above without discarding
outliers. These are initial implementation measurements, not an optimization
claim or a promise for other displays, frame rates or content.

The separate [five-round Vship comparison](cvvdp-vship-still-comparison.md)
interleaves streaming arms: VQM measures 2.636 ms at 1080p and 10.888 ms at 4K,
versus 5.834/21.094 ms for the peer. Both use explicit standard_4k. Timer/reset
boundary differences and the peer's numerical differences are documented there;
the peer is not the correctness oracle.

## Existing-metric regression protocol

Baseline is the retained pre-CVVDP selected executable from the prior VMAF
investigation, SHA-256
`fbfdde4fb5c547b4def0edd9e33b527f9f20655624aece52bd0a8d3da6f266ba`,
with its preserved source snapshot. HEAD alone is not this working-tree
baseline. Five A/B-interleaved runs cover all four existing metrics at 1080p
and 4K, resident 20/3 and streaming 8/2: **160 runs**, with identical scores
and checksums per workload. No half-plane or multi-worker claim is made.

Fifteen initial core medians show no increase above 0.87%. The first VMAF v0
4K streaming sitting has a 4.52% increase, with isolated ADM timestamp spikes
and a candidate run at 7.59 ms. Its complete five-round follow-up measures
4.108861 ms baseline versus 4.138461 ms candidate (**+0.72%**). Both sittings and
all outliers remain in the compact evidence; this is no claim of a speedup.
A separate timing pass overlapped a test job and was discarded as a whole;
its raw files remain under `qualification-benchmark-interrupted`.

[Compact supporting data](../measurements/cvvdp-implementation-results.json)
retains every qualified run, score, checksum, stage table, range and input hash.
Raw logs, commands and the frozen binary are under
`artifacts/reports/cvvdp/`. Performance runs are unprofiled and performed while
reference/software-validation/test jobs are paused; those jobs resume afterward.

To reproduce a single still arm:

```sh
./build/vqm_bench --metric cvvdp \
  --pair .cache/bench/pairs/synthetic-1920x1080/orig.png \
         .cache/bench/pairs/synthetic-1920x1080/dist.png \
  --iterations 20 --warmup 3 --resident 2
```

The ordered test clips were generated with FFmpeg `testsrc2`, 32 frames at
30 fps, yuv420p, 1920x1080 or 3840x2160. Distortion uses
`lutyuv=y='floor(val/8)*8'`. Both use the tool's declared Y4M defaults
(BT.470BG matrix, BT.709 SDR tag interpreted with original sRGB EOTF, limited
range); no implicit spatial or temporal resampling is used.

```sh
ffmpeg -f lavfi -i testsrc2=size=1920x1080:rate=30 -frames:v 32 \
  -pix_fmt yuv420p .cache/bench/cvvdp/1080-original.y4m
ffmpeg -i .cache/bench/cvvdp/1080-original.y4m \
  -vf "lutyuv=y='floor(val/8)*8'" -frames:v 32 -pix_fmt yuv420p \
  .cache/bench/cvvdp/1080-distorted.y4m
./build/vqm_bench --metric cvvdp --cvvdp-video \
  --pair .cache/bench/cvvdp/1080-original.y4m \
         .cache/bench/cvvdp/1080-distorted.y4m --iterations 8 --warmup 2
```

## Limits and subsequent work

This is FP32 base-model support with replicated temporal startup. Alpha,
heatmaps, symmetric padding, variable-rate resampling, learned CVVDP models
and sharing one clip across workers are not supported. The original mixed-edge
quirk is retained for compatibility, not silently repaired. Full XR-DAVID and
subjective video-corpus correlation are extended validation, not claimed here.

Stage timing identifies spatial work and temporal history bandwidth as future
optimization targets. FP16 histories, fused FIR channels, reduced scratch or
cooperative-matrix changes need separate numerical qualification and the >=2%
target-stage speed gate. None is silently included in this initial baseline.

The complete measured source tree is retained locally at
`artifacts/snapshots/cvvdp-implementation-20261106-source.tar.gz`, SHA-256
`e5b590a8db307b34755c14acc7ec804cc3bc4dfa5e79f6917e32358425a96521`.
It includes the existing uncommitted audit work and the implementation; this
final snapshot-location paragraph was added after archive creation. Extract
into an empty directory when reconstructing it. No commit was created.
