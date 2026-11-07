# VQM documentation

The main README covers the build, command line and library quick start.
[Project README](../README.md)

## Metrics

| Metric | Guide |
| --- | --- |
| SSIMULACRA 2.1 | [Pipeline, norms and score mapping](ssimulacra2.md) |
| Butteraugli | [Diffmap, two spatial levels and final score](butteraugli.md) |
| VMAF v0.6.1 and v1.0.16 3H | [Sequence APIs, model limits and reference checks](vmaf-development.md) |
| ColorVideoVDP v0.5.7 | [Image and video API, viewing conditions and JOD](cvvdp.md) · [Reference implementation and oracles](cvvdp-development.md) |
| FP32 PSNR-HVS | [Native YUV inputs, dB scores and qualification status](psnr-hvs.md) |

## Library and development

| Topic | Guide |
| --- | --- |
| C++ API and input layouts | [API and implementation reference](reference.md) |
| Unified still-image and sequence API | [Metrics](metrics.md) |
| GPU allocations and worker sizing | [Memory planning](memory.md) |
| Shared spatial scratch storage | [Shared workspace](shared-workspace.md) |
| Borrowed buffers, devices and queues | [Vulkan synchronization](vulkan-synchronization.md) |
| Repository layout and component ownership | [Architecture](architecture.md) |
| Local tests, sanitizers and formatting | [Contributing](../CONTRIBUTING.md) |

## Design records

These dated records capture the original scope, design choices and acceptance
criteria. The guides above describe the current APIs and workflows.

| Date | Record |
| --- | --- |
| 2026-11-05 | [ColorVideoVDP implementation](design/cvvdp-implementation-plan.md) · [Integration review](design/cvvdp-integration-review.md) · [Original and MATLAB review](design/cvvdp-matlab-review.md) · [Vship source review](design/cvvdp-vship-review.md) |
| 2026-11-06 | [PSNR-HVS implementation](design/psnr-hvs-implementation-plan.md) |
| 2026-11-06 | [Ranking agreement and temporal validation](design/ranking-and-temporal-validation-plan.md) · [Adopted ranking defaults](design/ranking-default-gates.md) |

## Repository audits

- [Correctness and organization audit](audits/audit-correctness-readability.md)
- [Detailed code, structure and correctness review](audits/multi-angle-review.md)
- [GitHub readiness audit](audits/github-readiness-audit-20261106.md)

## Reference correlation

The hosted qualification matrix uses pinned independent originals for
SSIMULACRA 2, Butteraugli, VMAF v0, VMAF v1 and CVVDP. Run it after building
the CLI and preparing the Python environment described in the main README.

```sh
.venv/bin/python -m validation.temporal.run --profile qualification
```

The qualification profile covers 390 complete clip pairs per metric.
`--profile smoke` selects 39 pairs per metric. The aggregate `summary.json`
records correlation, coverage, numerical diagnostics and the gate outcome for
each selected metric. Use the workflow guide for input formats, frozen scores
and ranking policies. [Hosted video validation](temporal-validation.md)

The CID22 workflows provide separate still-image and human-rating evidence.
Metric-specific differences from the originals are documented separately.
[Validation guide](validation.md) · [Reference differences](divergence.md)

## Measurements

Start with the benchmark guide for local stage timing, complete clips and
peer comparisons. The methodology guide describes timing boundaries and
workload selection. [Benchmark guide](../bench/README.md) ·
[Performance methodology](performance.md)

Raw reports go under `artifacts/reports/`. Dated results, research conclusions
and reproduction instructions live in the versioned archive. Reusable
probes have their own index. [Research and measurement archive](../bench/notes/README.md) ·
[Experiments](../bench/experiments/README.md)

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
