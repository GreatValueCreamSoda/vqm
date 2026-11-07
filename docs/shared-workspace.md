# Shared metric scratch

The library-owned `MetricPipelines` collection and unified `Metrics` pipeline
share one overwriteable GPU arena across all selected metrics: SSIMULACRA 2,
Butteraugli, VMAF v0, VMAF v1, CVVDP, and PSNR-HVS. Initialization queries their
requirements, allocates the maximum
scratch size rather than the sum, and binds it internally to each pipeline.
The CLI constructs `MetricPipelines` using public configuration and borrows its
selected pipelines; it never plans, allocates, or accesses internal scratch.
The frame loop does not allocate or pass a buffer manually. A fallback upload
and readback staging buffer, when required, is also sized to the maximum.

Metric layouts declare which arena ranges must survive between calls. Each
metric retains these in its own device-local buffer: immutable lookup tables
are captured at initialization and restored before dispatch; mutable VMAF
motion planes are bound directly as private shader storage, with no per-frame
save or restore copies. Both VMAF v1 HFR history planes and the rotating work
plane remain private. CVVDP's FIR
history buffers are already private and are never shared. For a one-frame clip,
finalization reloads the prepared frame from that private history rather than
relying on scratch surviving the other metrics. CPU-side sequence state and
Vulkan command objects also remain metric-owned. Metric arithmetic is
unchanged. Motion-history addressing uses the private plane binding, while
other feature buffers retain their existing arena offsets.

All selected metrics execute serially on each frame. Switching metrics may
freely overwrite scratch, but not another metric's retained state. Outstanding
submissions prevent another borrower from starting or constructing a pipeline
until the first submission is finished. A worker is not thread-safe; use a
separate composed pipeline and scratch allocation for each concurrent worker.
Resident image handles and mapped input pools are unavailable on shared scratch.
Standalone metric pipelines retain their existing allocation and handle APIs.
With only one metric selected, `MetricPipelines` uses its standalone allocation
path and does not duplicate retained state for sharing.

The CLI reuses initialized pipelines across frames, repetitions, and compatible
pairs. Geometry, input format, or rate changes create a new set. Metric options,
alpha-background policy, frame ordering, and temporal reset/finalization rules
are unchanged. GPU-resident BestSource inputs remain borrowed on the same GPU
without CPU input staging.

Use `Metrics` for frame-aligned, combined scores and unified sequence handling.
Use `MetricPipelines` when individual APIs are needed, such as the CLI's raw
features, per-metric timing, requested metric order, and alpha-background policy.
Its typed getters return null for unselected metrics. Returned pipeline pointers
are non-owning and remain valid across collection moves while the owner lives;
do not move individual pipelines out of the collection. The library retains
scratch and private state for the collection's lifetime. Callers submit work
serially and follow each metric's reset and finalization contracts.

`Metrics::GetMemoryUsage()` reports the arena and optional staging buffer once
in `scratch`. Metric entries contain only private `auxiliary` allocations;
`spatial` combines the two spatial metrics' private storage. Queries made with
the same complete configuration and device predict these allocations and their
per-heap charges. They exclude borrowed inputs and driver overhead. See
[memory planning](memory.md).

Multi-metric `MetricPipelines::GetMemoryUsage()` uses the same accounting and
requirements plan. With a single metric, the collection reports the standalone
arena and staging allocations in that metric's entry instead, with no `scratch`
entry. Query its normalized individual configuration for that standalone plan;
`GetMemoryRequirements(MetricsConfig)` describes the always-shared `Metrics`
path, including when `Metrics` selects just one metric.

## Two-metric spatial workspace

The existing `SharedMetricWorkspace` API remains available for serial
SSIMULACRA 2 and Butteraugli evaluation only. It allocates scratch and private
tables at construction, not per frame:

```cpp
#include <vqm_shared.h>

vqm::SharedMetricWorkspace metrics({.size = {1920, 1080}});
auto memory = metrics.GetMemoryUsage();
auto ssimu2 = metrics.ComputeSsimulacra2(original, distorted).Score();
auto butter = metrics.ComputeButteraugli(original, distorted).score;
```

The `Image`, `ImageView`, and borrowed `VulkanBufferView` overloads are
synchronous. Construct with a caller-owned `VulkanDevice` to use external
buffers; the external buffer lifetime, queue synchronization, semaphore, and
queue-family ownership rules are the same as for the individual pipelines.
Use one workspace per concurrently active worker. Inputs must be re-supplied
for each metric. `GetMemoryUsage()` reports the one shared arena, any staging
buffer, and private metadata allocations in `auxiliary`.
Before construction, pass the same `SharedMetricConfig` to
`GetMemoryRequirements()` in `vqm_memory.h` to query those allocations.
