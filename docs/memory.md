# GPU memory planning

Each standalone pipeline allocates its working buffers when it is constructed
and reuses them for every image or video frame of its configured geometry.
`vqm_memory.h` can query those buffer requirements before constructing a
pipeline. Use the same full metric configuration and Vulkan device for the
query and the pipeline: geometry, input format, resident slots, streaming,
half-planes and the device's limits can all change the result.

```cpp
#include <array>
#include <vector>
#include <vqm_memory.h>

vqm::Ssimulacra2Config config;
config.size = {1920, 1080};
config.gpu_resident_images = 0;

// Uses the same default device selection as Ssimulacra2Pipeline.
auto requirements = vqm::GetMemoryRequirements(config);

// A device may provide a snapshot of its current remaining heap budgets.
// If the extension is unavailable, the application must supply budgets itself.
auto available = vqm::AvailableHeapBudgets(requirements);
std::vector<std::uint64_t> budgets = available.value_or(
    std::vector<std::uint64_t>(requirements.heaps.size(), 0)
);
// Supply budgets if unavailable, and reserve application-specific headroom.

const std::array per_worker{requirements};
std::size_t workers = vqm::MaxWorkersForBudgetWorstCase(per_worker, budgets);
```

`GetMemoryRequirements` is overloaded for `Ssimulacra2Config`,
`ButteraugliConfig`, `VmafConfig`, `VmafV1Config`, `CvvdpConfig`,
`PsnrHvsConfig` and `SharedMetricConfig`. The [unified metrics API](metrics.md)
adds an overload for `MetricsConfig`.
Pass a `VulkanDevice` as the second argument when the pipeline will borrow an
application-owned device.
The returned `direct` and `staged` scenarios give the buffer bytes, actual
Vulkan allocation requirements, memory type and heap index. The direct arena is
host-visible and device-local; when it is unavailable or allocation fails, the
runtime falls back to a device-local arena with a host-visible staging buffer.
On unified-memory devices the staging buffer may use a device-local heap too.

Every configuration has a `memory_policy`. `MemoryPolicy::Automatic` prefers
the direct path and falls back to staging if necessary. Set
`MemoryPolicy::Direct` or `MemoryPolicy::Staged` to require one path; the
constructor then fails if it cannot use that path. `Staged` deliberately does
not map the arena, so `AllocateImage` and `MapImage` are unavailable even if
the selected device-local memory type happens to be host-visible. Set the
policy before querying, and use the same configuration for construction.
Every pipeline also exposes `GetMemoryUsage()` to report the allocations
actually retained after construction.

For a known allocation scenario, `MaxWorkersForBudget` accepts one scenario or
several scenarios per worker and adds their allocation sizes by heap.
`MaxWorkersForBudgetWorstCase` instead charges each heap for the largest
supported direct/staged alternative of each selected metric. It is conservative
and can be used when the eventual fallback path is not known yet. Use a separate
allocation per concurrently active standalone pipeline; merely submitting more
work to the same pipeline does not make it concurrent.

These numbers cover library-owned Vulkan buffers and the allocation sizes
reported for them, not driver-owned shader/pipeline/descriptor memory, other
applications, or changing available memory. Heap capacity is not free memory.
If `VK_EXT_memory_budget` is supported, `AvailableHeapBudgets` returns the
driver's current advisory budget minus usage; otherwise it returns `nullopt`.
The caller's budgets must leave appropriate headroom; passing the arithmetic
test cannot guarantee that a later allocation or dispatch will succeed. To
assess a particular resolution or aspect ratio, query that geometry and test
the desired worker count against the available budgets. Different aspect
ratios and metric formats have different limits, so there is no single largest
image size independent of the workload.

## Device limits

The shared Vulkan context requires 256 invocations per compute workgroup and
16 KiB of workgroup shared memory. Individual metric planners check additional
requirements before querying or allocating buffers. SSIMULACRA 2 and Butteraugli
both require at least 256 invocations along the X axis for fixed shader passes.
Butteraugli also needs 29,376 bytes of workgroup shared memory for the opsin pass.
SSIMULACRA 2 selects its scale tile from the device's limits.

## CVVDP temporal storage

CVVDP's queried scenarios and `PipelineMemoryUsage` include an `auxiliary`
vector. Each entry is a separate device-local history buffer with its own
allocation size, memory type and heap. Arena memory policy does not turn these
histories into upload staging. Budget helpers count all auxiliary allocations
by heap, including in the conservative direct/staged calculation. Do not add
only `arena` and `staging` when estimating CVVDP memory. Still-only configurations
with `temporal = false` allocate no histories. Construction and queries share
one planner and reject per-buffer descriptor/index limits before allocation.
See [CVVDP resources](cvvdp.md#resources) for formulas and examples.
