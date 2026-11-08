# VQM documentation

Start with [supported metrics, performance and building](../README.md).

## Usage and workflows

- [Command line and score interpretation](cli.md)
- [C++ library quick start](library.md)
- [Build, test and contributing](../CONTRIBUTING.md)
- [Benchmarking and profiling](../bench/README.md)
- [Reference validation](../validation/README.md)

## Metrics

| Metric | Guide |
| --- | --- |
| SSIMULACRA 2.1 | [Pipeline, norms and score mapping](ssimulacra2.md) |
| Butteraugli | [Diffmap, two spatial levels and final score](butteraugli.md) |
| VMAF v0.6.1 and v1 SDR/HDR | [Features, model presets and sequence processing](vmaf.md) |
| ColorVideoVDP v0.5.7 | [Perceptual pipeline, viewing conditions and JOD](cvvdp.md) |
| FP32 PSNR-HVS | [Native YUV inputs, contrast masking and dB pooling](psnr-hvs.md) |

## Implementation

- [Reference differences](divergence.md)
- [Image formats and Vulkan implementation](reference.md)
- [Unified metrics and frame alignment](metrics.md)
- [Component ownership](architecture.md)
- [GPU memory planning](memory.md)
- [Shared metric scratch](shared-workspace.md)
- [Vulkan synchronization](vulkan-synchronization.md)

## Provenance

Metric implementations and comparison tools have distinct upstream provenance:

- **Cloudinary SSIMULACRA 2** (BSD-3-Clause, Copyright (c) Jon Sneyers,
  Cloudinary): this project is derived from it. Translations and derived
  weights cite `.reference/ssimulacra2/src/*.cc` by function and line.
- **libjxl** (BSD-3-Clause, Copyright (c) the JPEG XL Project Authors): the
  Butteraugli reference and the later SSIMULACRA 2 home. Formulas are
  translated and cite `.reference/libjxl/lib/jxl/butteraugli/butteraugli.cc`
  and related files by function and line. The source and installed package
  include the [JPEG XL license](../licenses/JPEG-XL-LICENSE) and its
  [upstream patent grant](../licenses/JPEG-XL-PATENTS).
- **Netflix VMAF** (BSD-2-Clause-Patent, Copyright Netflix, Inc.): the source
  for VMAF features and models. See the
  [Netflix notice](../licenses/NETFLIX-VMAF-LICENSE) and the separate
  [libsvm notice](../licenses/LIBSVM-LICENSE).
- **ColorVideoVDP** (MIT, Copyright the Graphics and Displays group): the base
  v0.5.7 display, spatial and temporal models, with generated calibration/CSF
  data and independent reference oracles. See the
  [ColorVideoVDP notice](../licenses/COLORVIDEOVDP-LICENSE).
- **Xiph PSNR-HVS** (permissive two-clause notice, Copyright Xiph.Org and
  contributors): weighting and masking from the pinned libvmaf third-party
  source; the FP32 adaptation retains `licenses/XIPH-PSNR-HVS-LICENSE`.
- **Vship and Vship-fast** (MIT, with a no-AI clause): mechanisms only. Their
  public C API is called for the throughput comparison; their file and line is
  named where a mechanism was read, and none of their code is copied.
