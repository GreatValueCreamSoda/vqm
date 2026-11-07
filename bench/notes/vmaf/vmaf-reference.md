# VMAF reference results on CID22

This historical run compared VQM with the scalar Netflix libvmaf 3.2.0 build
at commit `4611e47a7330cd342498dcc130e1008efbeab6e4`. It used the v0.6.1 float
model and the v1.0.16 standard SDR 3H model. The developer guide describes the
reference build, sequence fixtures and still-image comparison commands.
[VMAF reference and validation](../../../docs/vmaf-development.md#reference-and-validation)

Each of the 4,292 annotated 512×512 CID22 pairs was an independent one-frame
sequence with zero motion. Both implementations used the same planar
YUV444P10LE samples. Conversion applied the BT.709 matrix to sRGB-encoded
8-bit RGB, then rounded to nearest, ties to even, in 10-bit limited range.
Y used offset 64 and scale 876, and Cb and Cr used offset 512 and scale 896.
The conversion identifier was
`rgb8-srgb-encoded-bt709-limited-yuv444p10le-rne-v1`.

## Correlation against CID22 subjective scores

| implementation | KRCC | SRCC | PCC |
| --- | ---: | ---: | ---: |
| libvmaf v0.6.1 float | 0.59537 | 0.79482 | 0.75519 |
| VQM v0.6.1 float | 0.59538 | 0.79482 | 0.75520 |
| libvmaf v1.0.16 SDR 3H | 0.56397 | 0.76186 | 0.71780 |
| VQM v1.0.16 SDR 3H | 0.56395 | 0.76184 | 0.71780 |

## Score differences against libvmaf

| VQM model | Maximum absolute | Mean absolute | 99th-percentile absolute | RMSE |
| --- | ---: | ---: | ---: | ---: |
| v0.6.1 float | 0.017094 | 0.000196 | 0.001473 | 0.000508 |
| v1.0.16 SDR 3H | 0.004679 | 0.001700 | 0.003448 | 0.001798 |

During the first run the process exited with SIGSEGV in an NVIDIA driver worker
while destroying the Vulkan device. The cause remains unknown. One full
ASan/UBSan run and three further full Release runs passed and produced identical
score files. The run writes `vqm-v0.log` and `vqm-v0.stdout` for diagnostics.

These correlations and numerical differences describe the annotated still-image
pairs at their coded geometry with this conversion and pooling. CID22 gathered
subjective ratings for an image-quality study.
