# Sharing non-temporal metric scratch

`SharedMetricWorkspace` computes SSIMULACRA 2 and Butteraugli sequentially with
one arena sized to the larger metric and, if needed, one shared staging buffer.
The arena and staging buffers are allocated at construction, not per frame.
The two backends retain separate Vulkan pipeline and command objects; those and
driver-managed allocations are not included in the memory report.

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
Use one workspace per concurrently active worker; a workspace is not
thread-safe. Its inputs must be re-supplied for each metric, and it does not
expose resident handles or caller-mapped `ImageBuffer` allocations, because
the other metric would overwrite them. Each switch republishes the small
metric-specific GPU tables.

`GetMemoryUsage()` reports the one actual shared arena and any staging buffer,
including Vulkan allocation sizes and their memory types and heaps. It does
not report externally owned buffers or Vulkan implementation/driver overhead.
Before constructing the workspace, pass the same `SharedMetricConfig` to
`GetMemoryRequirements()` in `vqm_memory.h`; see [memory planning](memory.md).
VMAF v0 and v1 keep temporal state between frames and are not included in this
workspace; their individual pipelines remain separate.

The [shared-memory report](../bench/notes/system/shared-memory.md) records
measured memory requirements, serial throughput and correctness comparisons.
