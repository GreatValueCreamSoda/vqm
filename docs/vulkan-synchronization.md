# Vulkan semaphore submission

A pipeline constructed with a borrowed `VulkanDevice` can accept binary
semaphore dependencies when submitting two external `VulkanBufferView`s:

```cpp
#include <array>
#include "vqm_vulkan.h"

std::array waits{vqm::VulkanSemaphoreWait{
    producer_done, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT}};
std::array signals{consumer_ready};
vqm::VulkanSubmitSync sync{waits, signals};
std::uint64_t token = pipeline.Submit(original, distorted, sync);
// Submit downstream work waiting on consumer_ready if needed.
auto result = pipeline.Finish(token);
```

The spans are read only during `Submit`; the semaphore objects themselves
must remain alive until all queue operations that reference them complete.
Inputs must remain alive and unmodified until `Finish`, or until destruction of
an in-flight pipeline completes. `Finish` still waits on the pipeline's fence
and returns its CPU-visible score. The signal semaphores can be consumed by a
subsequently submitted GPU operation without waiting for `Finish` first.

Only binary semaphores are supported. Vulkan does not expose a way to verify a
semaphore's type or originating device from its handle; the caller supplies
valid same-device binaries. Each wait needs a previously submitted signal;
each signal must be unsignaled when its signal operation executes. The lists
may be empty. Null handles, repeated handles, a semaphore appearing in both
lists, and unsupported stage masks are rejected before recording. Allowed wait
stage bits are top-of-pipe, compute shader, transfer and all-commands; each mask
must include top-of-pipe, compute shader or all-commands so input reads cannot
run ahead of the wait.

The caller must externally synchronize host access to the borrowed `VkQueue`
across the application and all pipelines. A semaphore does not transfer queue
family ownership. For exclusive buffers produced in another family, the
application must record a matching release and acquire and synchronize those
operations. For example, submit the acquire on the scoring queue with a wait
on the producer's signal, signal a second binary semaphore, and pass that second
semaphore as the pipeline wait. Buffers shared across queues in the same family
need no ownership transfer, but still need the producer-to-consumer semaphore
when there is no other valid dependency. Noncoherent host writes must be
flushed before submission.
