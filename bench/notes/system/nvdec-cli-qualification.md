# NVDEC CLI qualification

2026-11-07. The CLI's optional NVDEC backend was evaluated for agreement
with the existing Y4M path and for decoded-frame transfers across the host
boundary. The release builds used an RTX 5070, NVIDIA driver 615.78.08,
Vulkan loader 1.4.363, FFmpeg 9.0.2 and CUDA Toolkit 13.4.92. The working
tree contained pre-existing changes at `f9d5024`. The local raw reports under
`artifacts/reports/temporal-nvdec/` record the executable fingerprints and
the canonical and encoded input hashes. The automatic-selection release
binary SHA-256 was
`fa38a3030e710bd3049c04a0f5b1903bd060b39672dade0450981d2398a680b0`.

## Method

The stock CLI ran the five-metric hosted video qualification with the
frozen original scores and default ranking policies.

```sh
.venv/bin/python -m validation.temporal.run --profile qualification --offline \
  --out-dir artifacts/reports/temporal-nvdec/y4m-baseline
```

The compressed arm used the same 30 source groups, 13 distortions per
source and 40 frames per clip. It used 420 unique lossless HEVC transports
for native YUV444 and 420 for the separate gamut-conditioned spatial
representation. For each canonical Y4M in `inputs.json`, encode it with
the original geometry and time base and verify its decoded pixels against
the corresponding canonical `yuv` SHA-256 in the manifest.

```sh
ffmpeg -i "$canonical_y4m" -an -c:v libx265 -preset ultrafast -pix_fmt yuv444p \
  -x265-params lossless=1:pools=1:frame-threads=1:log-level=error:colorprim=bt709:transfer=bt709:colormatrix=bt470bg:range=limited \
  -color_range tv -colorspace bt470bg -color_trc bt709 -color_primaries bt709 \
  -movflags +write_colr "$transport_mp4"
ffmpeg -v error -hwaccel none -i "$transport_mp4" -map 0:v:0 \
  -pix_fmt yuv444p -f rawvideo - | sha256sum
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,avg_frame_rate,nb_frames "$transport_mp4"
```

Use the verified MP4 paths in place of the corresponding Y4M paths in the
stock `<metric>-pairs.tsv` files. Run the same CLI command recorded in each
`<metric>-candidate.log` with the substituted pair list and the NVDEC-enabled
executable. The CLI selects NVDEC for the compressed inputs.
Keep `--input-transfer srgb` for SSIMULACRA 2 and Butteraugli. Compare
`sample_id`, `frames`, `score` and `frame_results` in the two JSONL outputs.
The compressed-arm aggregate used the stock temporal agreement functions
and bound each sample to the canonical hashes after the transport checks.

## Qualification results

Both input paths scored 390 clip pairs per metric. Every clip score and
emitted frame record matched exactly between the compressed and Y4M paths.

| Metric | Spearman | Kendall tau-b | Ranking | Legacy numerical |
| --- | ---: | ---: | --- | --- |
| SSIMULACRA 2 | 0.999971274 | 0.997495221 | Pass | Fail |
| Butteraugli | 1.000000000 | 1.000000000 | Pass | Pass |
| VMAF v0 | 0.999985030 | 0.999551653 | Pass | Pass |
| VMAF v1 | 0.999999588 | 0.999943238 | Pass | Pass |
| CVVDP | 1.000000000 | 1.000000000 | Pass | Pass |

The unchanged SSIMULACRA 2 numerical diagnostic had a maximum absolute
error of 5.262514 and mean absolute error of 0.295218 against the pinned
original in both arms. PSNR-HVS used independent CLI fixture parity checks.

## Transfer trace

Two H.264 NVENC MP4 files contained 12 frames at 256×160 and 30000/1001 fps.
The original used FFmpeg's `testsrc2` source and the distorted clip applied
`gblur=sigma=1.2` before encoding. Both were 8-bit YUV420 with limited BT.709
color and B frames enabled. Nsight Systems 2026.4.1 traced the scoring
process with `--trace=cuda,nvvideo,vulkan --sample=none --cpuctxsw=none`.
The CUDA memory summary showed 48 device-to-device copies and no host-to-device
or device-to-host copies. There were 24 `PackNvdecFrame` kernel launches, one
per decoded source frame. The Vulkan scorer reads back its normal score
data. CUDA Compute Sanitizer memcheck reported zero errors on this fixture.

The design uses GPU packing because decoded semiplanar UV and
MSB-aligned words need layout changes before the existing planar metric
shaders can consume them. Exported buffers and binary semaphores keep
the handoff on one physical GPU. A CPU decode fallback would break the
requested frame residency and is rejected. [CUDA external resource interop](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/04-special-topics/graphics-interop.html) · [Vulkan external sharing](https://docs.vulkan.org/spec/latest/chapters/resources.html#resources-external-sharing)
