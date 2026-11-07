// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "butteraugli.h"
#include "cvvdp.h"
#include "psnr_hvs.h"
#include "vmaf.h"
#include "vmaf_v1.h"
#include "ssimulacra2.h"
#include "vqm_memory_types.h"
#include "vqm_shared.h"

namespace vqm {

// A possible allocation path. Automatic prefers `direct`, but may use
// `staged` if a host-visible device-local allocation is unavailable or fails.
// Staging can itself use a device-local heap on a unified-memory device.
// `supported` indicates a compatible memory type, not available capacity.
struct MemoryScenario {
    bool supported = false;
    MemoryAllocation arena;
    MemoryAllocation staging;
    std::vector<MemoryAllocation> auxiliary = {};
};

struct MemoryHeapInfo {
    std::uint64_t size_bytes = 0;
    bool device_local = false;
    // VK_EXT_memory_budget snapshot, when the physical device supports it.
    // Both values can change as other applications allocate memory.
    bool budget_available = false;
    std::uint64_t budget_bytes = 0;
    std::uint64_t usage_bytes = 0;
};

// Per-pipeline buffer requirements for the selected device. Allocation sizes
// are queried from Vulkan without allocating device memory. They exclude
// driver-owned memory for pipelines, descriptors, command buffers and other
// applications. Heap sizes are capacities, not free or reserved memory.
struct MetricMemoryRequirements {
    MemoryScenario direct;
    MemoryScenario staged;
    std::vector<MemoryHeapInfo> heaps;
    MetricScratchRequirements scratch{};
};

// Estimated remaining bytes per heap from the device's budget snapshot, or
// nullopt if VK_EXT_memory_budget is unavailable. This is advisory and can
// change immediately; applications should reserve additional headroom.
[[nodiscard]] std::optional<std::vector<std::uint64_t>> AvailableHeapBudgets(
    const MetricMemoryRequirements& requirements
);

// Plan one pipeline before construction using its complete configuration and
// the default device, or the supplied borrowed device. Planning validates the
// metric's geometry and buffer limits and throws if they are unsupported; it
// cannot guarantee a later allocation or dispatch succeeds. Automatic reports
// both alternatives; a strict memory policy marks the other unsupported.
[[nodiscard]] MetricMemoryRequirements GetMemoryRequirements(
    const Ssimulacra2Config& config
);
[[nodiscard]] MetricMemoryRequirements GetMemoryRequirements(
    const Ssimulacra2Config& config, const VulkanDevice& device
);
[[nodiscard]] MetricMemoryRequirements GetMemoryRequirements(
    const ButteraugliConfig& config
);
[[nodiscard]] MetricMemoryRequirements GetMemoryRequirements(
    const ButteraugliConfig& config, const VulkanDevice& device
);
[[nodiscard]] MetricMemoryRequirements GetMemoryRequirements(
    const PsnrHvsConfig& config
);
[[nodiscard]] MetricMemoryRequirements GetMemoryRequirements(
    const PsnrHvsConfig& config, const VulkanDevice& device
);
[[nodiscard]] MetricMemoryRequirements GetMemoryRequirements(
    const VmafConfig& config
);
[[nodiscard]] MetricMemoryRequirements GetMemoryRequirements(
    const VmafConfig& config, const VulkanDevice& device
);
[[nodiscard]] MetricMemoryRequirements GetMemoryRequirements(
    const VmafV1Config& config
);
[[nodiscard]] MetricMemoryRequirements GetMemoryRequirements(
    const VmafV1Config& config, const VulkanDevice& device
);
[[nodiscard]] MetricMemoryRequirements GetMemoryRequirements(
    const SharedMetricConfig& config
);
[[nodiscard]] MetricMemoryRequirements GetMemoryRequirements(
    const CvvdpConfig& config
);
[[nodiscard]] MetricMemoryRequirements GetMemoryRequirements(
    const CvvdpConfig& config, const VulkanDevice& device
);
[[nodiscard]] MetricMemoryRequirements GetMemoryRequirements(
    const SharedMetricConfig& config, const VulkanDevice& device
);

// Maximum number of independent pipelines fitting the caller-supplied byte
// budget for each Vulkan memory heap, counting the scenario's arena, staging
// and auxiliary allocations. Throws for an unsupported scenario, too few heap
// budgets or arithmetic overflow. This is arithmetic, not an allocation
// guarantee; leave room for driver resources and other users of the device.
[[nodiscard]] std::size_t MaxWorkersForBudget(
    const MemoryScenario& scenario, std::span<const std::uint64_t> heap_budgets
);
// As above, when each worker holds several independent metric pipelines.
// Pass the selected direct or staged scenario for each pipeline; allocations
// that use the same heap are added before computing the worker count.
[[nodiscard]] std::size_t MaxWorkersForBudget(
    std::span<const MemoryScenario> per_worker,
    std::span<const std::uint64_t> heap_budgets
);

// Conservative count when the runtime may choose direct or staged memory for
// each pipeline. Charges each heap for the largest supported alternative of
// each metric, then adds the metrics. This may be lower than an actual mixed
// allocation, but does not assume that the preferred direct allocation works.
[[nodiscard]] std::size_t MaxWorkersForBudgetWorstCase(
    std::span<const MetricMemoryRequirements> per_worker,
    std::span<const std::uint64_t> heap_budgets
);

}  // namespace vqm
