# VQM architecture

Code is grouped by the component that owns its behavior. Each metric owns its
input policy, feature computation, memory layout, command recording, shaders
and score conversion. Shared code describes image storage or operates Vulkan
resources without interpreting a metric's features.

```text
include/
  vqm_image.h                  common image descriptions and handles
  ssimulacra2.h                SSIMULACRA 2 API
  butteraugli.h                Butteraugli API
  psnr_hvs.h                   frame-local native YUV PSNR-HVS API
  vmaf.h, vmaf_v1.h            versioned VMAF APIs
  cvvdp.h                      display-aware image and clip JOD API
  vqm_metrics.h                combined evaluation and frame alignment
  vqm.h                        convenience umbrella
src/
  input.*, image_view.*        image validation, geometry and host views
  handles.*                    resident-handle identity and ownership
  metrics/
    metrics.cc                 combined metric evaluation
    shared_workspace.*         serial spatial-metric composition
    ssimulacra2/                facade, score, layout, Vulkan passes, shaders/
    butteraugli/                facade, tables, layout, Vulkan passes, shaders/
    psnr_hvs/                   facade, layout, FP32 DCT/masking, shaders/
    vmaf/
      features/                ADM/ADM3, VIF, preprocessing and their shaders/
      cambi/, speed/           complete v1 feature passes and their shaders/
      v0/, v1/                 model tables/evaluation and temporal pipelines
    cvvdp/                     display conversion, spatial model, clip state,
                               generated calibration/CSF data and shaders/
  vulkan/                      device, storage, descriptors and synchronization
    shaders/                   common GLSL bindings and source decoding
  timing.*                     benchmark-only timing implementation
tests/
  core/                        common resource and API contracts
  input/                       formats, conversion and decoding
  metrics/                     metric tests and oracles mirroring src/metrics/
bench/driver/                  routing, metric measurements and shared modes
bench/video.py                 hosted-video benchmark setup, timing and reports
tools/image_io/                PNG, JPEG and Y4M decoding outside the core library
validation/temporal/            hosted clips, reproducible representations and score oracles
```

Public leaf headers include shared image types or the specific metric
declarations they need. They do not depend on the umbrella. Existing callers
can continue including `vqm.h`; Vulkan interoperation is opt-in through
`vqm_vulkan.h`. Code using SSIMULACRA 2 declarations through an incidental
include of another leaf header should include `ssimulacra2.h` or `vqm.h`
explicitly.

Cross-component implementation includes are rooted at `src`, for example
`metrics/vmaf/features/vmaf_vif.h` or `vulkan/vulkan_runtime.h`. A header in the
same directory can use a local include. CMake exposes one implementation include
root, so moving a component cannot silently change which header an include finds.

A pipeline validates and describes its inputs, records metric-owned passes
through `VulkanRuntime`, then converts readback into metric results. The layout
planner used by construction is also used by memory queries. Every planner
returns the common `BufferMemoryPlan`; metric-specific planning declarations
remain beside their layouts.

`Metrics` aligns per-frame results from selected metrics and returns CVVDP's
whole-clip score separately from `Finalize`. `SharedMetricWorkspace` serializes
SSIMULACRA 2 and Butteraugli in one arena. VMAF and CVVDP sequences keep private
temporal state and allocations. These are distinct ownership contracts;
adding a metric requires an explicit decision about its state lifetime.

CVVDP's pipeline file owns input validation, memory planning, ordered history
and finalization. The display module converts shared raw samples into physical
display light and DKL; it does not use the other metrics' relative-linear
100-nit decoder. The spatial module records the Gaussian/Laplacian pyramid,
reference-derived CSF, mutual masking and pixel pooling. The model module
provides temporal coefficients, band/channel norms and the final JOD transform;
the pipeline retains the bounded temporal pooling sum and frame count.
Generated model tables come from the pinned original, with independent oracle
fixtures under `tests/metrics/cvvdp/`.

One unknown-length CVVDP clip retains its first converted pair until the second
input or finalization selects the original video or image algorithm. Separate
device-local allocations hold the two temporal rings; scratch and current
planes share the arena. The optional runtime storage bindings and every
auxiliary allocation are included in memory queries and per-heap budgets.
Image-only pipelines omit those rings and retain the original four bindings.
No borrowed source buffer becomes temporal history. See the supported input,
state and resource contracts in [the CVVDP guide](cvvdp.md).

VMAF host passes are consolidated by responsibility: preprocessing contains
coded-plane conversion, SDR conversion, fit/pad and reference motion; ADM2 and
ADM3 share one implementation file; CAMBI and SpEED each keep their full stage
sequence in one file. Each version's pipeline file includes its temporal state
transitions. Model evaluation stays separate beside the generated tables, with
the temporal declarations in one internal header per version. Individual shader
files remain separate for compilation and GPU debugging.

Implementation interfaces identify their scope: `VulkanSsimulacra2Pipeline`
names one metric, while `SourceDecodeVariant` describes shared input decoding.
Benchmark selection uses one metric enum. Generic pipeline traits use the name
`Pipeline` rather than naming a particular metric's implementation.

The shared decoder and color-option tables use `vqm::image_io` and depend on
the common image header. `vqm::cli` owns command-line parsing and presentation.
Benchmarks, reference drivers and tests use the same image I/O module directly.

Shared rules have one owner: resident upload/release transactions live in
`ResidentTable`; context selection and pipeline-array creation/cleanup live in
the Vulkan runtime; named color flags and their application live in
`InputColorOptions`; serial benchmark decoding lives in `PairDecoder`. Shared
workspace activation has one state transition. Metric-specific numerical and
temporal rules stay beside their implementations, and CPU reference oracles
remain independent for verification.

Shaders live with the metric or feature that records them. Only GLSL bindings
and source decoding are shared under `src/vulkan/shaders`. Embedded shader
identifiers remain stable when files move; the active identifiers are listed
in `build/generated/vulkan_shader_manifest.txt`. Numerical test oracles can add
test-only variants without changing production shaders.

The canonical executables are `vqm`, `vqm_bench` and `vqm_test`.
Only `vqm_bench` links the core built with stage timing.

Design decisions and their source reviews live in `docs/design/`. Repository-wide
audits live in `docs/audits/`. Benchmark and qualification evidence lives in
`bench/notes/`, with reusable probes under `bench/experiments/`. Scripts create
ignored `artifacts/reports/` for raw local results and `.cache/` for generated
benchmark pairs and corpus conversions as needed. The `build/` and
`build-*/` directories contain disposable build outputs, including copied
reference build trees and generated fixtures. No unique research document or
required maintained source belongs there. See the complete
[workspace retention policy](../AGENTS.md#workspace-files-and-retention) before
cleaning a workspace.
