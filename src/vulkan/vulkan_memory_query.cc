// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include "vqm_memory.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "metrics/vmaf/vmaf_memory.h"
#include "metrics/cvvdp/cvvdp_internal.h"
#include "metrics/psnr_hvs/layout.h"
#include "metrics/ssimulacra2/layout.h"
#include "metrics/butteraugli/layout.h"
#include "metrics/shared_workspace_internal.h"
#include "vulkan_runtime.h"
#include "vulkan_support.h"

namespace vqm {
namespace {

struct BufferRequirements {
    std::uint64_t bytes = 0;
    VkMemoryRequirements requirements{};
};

BufferRequirements QueryBuffer(
    VkDevice device, std::uint64_t bytes, VkBufferUsageFlags usage
) {
    if (bytes == 0) return {};
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = bytes;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    internal::Check(
        vkCreateBuffer(device, &info, nullptr, &buffer), "vkCreateBuffer"
    );
    BufferRequirements result;
    result.bytes = bytes;
    vkGetBufferMemoryRequirements(device, buffer, &result.requirements);
    vkDestroyBuffer(device, buffer, nullptr);
    return result;
}

std::optional<std::uint32_t> FindType(
    const VkPhysicalDeviceMemoryProperties& properties, std::uint32_t mask,
    VkMemoryPropertyFlags flags
) {
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((mask & (1u << i)) != 0 &&
            (properties.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return std::nullopt;
}

MemoryAllocation MakeAllocation(
    const VkPhysicalDeviceMemoryProperties& properties,
    const BufferRequirements& buffer, std::uint32_t type
) {
    return MemoryAllocation{buffer.bytes, buffer.requirements.size, type,
                            properties.memoryTypes[type].heapIndex};
}

void QueryHeapBudgets(
    VkPhysicalDevice physical_device, std::vector<MemoryHeapInfo>& heaps
) {
#ifdef VK_EXT_memory_budget
    std::uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(
            physical_device, nullptr, &count, nullptr
        ) != VK_SUCCESS) {
        return;
    }
    if (count == 0) return;
    std::vector<VkExtensionProperties> extensions(count);
    if (vkEnumerateDeviceExtensionProperties(
            physical_device, nullptr, &count, extensions.data()
        ) != VK_SUCCESS) {
        return;
    }
    const bool available = std::any_of(
        extensions.begin(), extensions.begin() + count,
        [](const auto& extension) {
            return std::string_view(extension.extensionName) ==
                   VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
        }
    );
    if (!available) return;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    properties.pNext = &budget;
    vkGetPhysicalDeviceMemoryProperties2(physical_device, &properties);
    for (std::size_t i = 0; i < heaps.size(); ++i) {
        heaps[i].budget_available = true;
        heaps[i].budget_bytes = budget.heapBudget[i];
        heaps[i].usage_bytes = budget.heapUsage[i];
    }
#else
    (void)physical_device;
    (void)heaps;
#endif
}

MetricMemoryRequirements QueryAllocations(
    const internal::VulkanContext& context, std::uint64_t arena_bytes,
    std::uint64_t staging_bytes, std::span<const std::uint64_t> auxiliary_bytes
) {
    constexpr VkBufferUsageFlags kArenaUsage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    constexpr VkBufferUsageFlags kStagingUsage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    constexpr VkMemoryPropertyFlags kDirect =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    constexpr VkMemoryPropertyFlags kStaging =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    const BufferRequirements arena =
        QueryBuffer(context.device, arena_bytes, kArenaUsage);
    const BufferRequirements staging =
        QueryBuffer(context.device, staging_bytes, kStagingUsage);
    const auto& properties = context.memory_properties;
    MetricMemoryRequirements result;
    result.heaps.reserve(properties.memoryHeapCount);
    for (std::uint32_t i = 0; i < properties.memoryHeapCount; ++i) {
        const auto& heap = properties.memoryHeaps[i];
        result.heaps.push_back(
            MemoryHeapInfo{heap.size,
                           (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0}
        );
    }
    QueryHeapBudgets(context.physical_device, result.heaps);

    const auto direct =
        FindType(properties, arena.requirements.memoryTypeBits, kDirect);
    if (direct) {
        result.direct.supported = true;
        result.direct.arena = MakeAllocation(properties, arena, *direct);
    }
    const auto device_local = FindType(
        properties, arena.requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    const auto host_staging =
        staging_bytes == 0
            ? std::optional<std::uint32_t>{}
            : FindType(
                  properties, staging.requirements.memoryTypeBits, kStaging
              );
    if (device_local && (staging_bytes == 0 || host_staging)) {
        result.staged.supported = true;
        result.staged.arena = MakeAllocation(properties, arena, *device_local);
        if (host_staging) {
            result.staged.staging =
                MakeAllocation(properties, staging, *host_staging);
        }
    }
    for (const auto bytes : auxiliary_bytes) {
        const auto buffer = QueryBuffer(context.device, bytes, kArenaUsage);
        const auto type = FindType(
            properties, buffer.requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        if (!type) {
            result.direct = {};
            result.staged = {};
            break;
        }
        const auto allocation = MakeAllocation(properties, buffer, *type);
        if (result.direct.supported)
            result.direct.auxiliary.push_back(allocation);
        if (result.staged.supported)
            result.staged.auxiliary.push_back(allocation);
    }
    return result;
}

template <typename Config, typename Planner>
MetricMemoryRequirements QueryMetric(
    const Config& config, const VulkanDevice* device, Planner planner
) {
    internal::ValidateMemoryPolicy(config.memory_policy);
    auto context = internal::RequireVulkanContext(device);
    const auto plan = planner(config, *context);
    auto result = QueryAllocations(
        *context, plan.arena_bytes, plan.fallback_staging_bytes,
        plan.auxiliary_bytes
    );
    if (config.memory_policy == MemoryPolicy::Direct) result.staged = {};
    if (config.memory_policy == MemoryPolicy::Staged) result.direct = {};
    return result;
}

void AddBytes(std::uint64_t& total, std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - total) {
        throw std::overflow_error("memory: heap allocation size overflow");
    }
    total += value;
}

void AddScenario(
    std::vector<std::uint64_t>& charge, const MemoryScenario& scenario
) {
    if (!scenario.supported || scenario.arena.allocation_bytes == 0) {
        throw std::invalid_argument("memory: unsupported or empty scenario");
    }
    std::vector<const MemoryAllocation*> allocations{&scenario.arena,
                                                     &scenario.staging};
    for (const auto& allocation : scenario.auxiliary)
        allocations.push_back(&allocation);
    for (const auto* allocation : allocations) {
        if (allocation->allocation_bytes == 0) continue;
        if (allocation->heap >= charge.size()) {
            throw std::invalid_argument("memory: missing heap budget");
        }
        AddBytes(charge[allocation->heap], allocation->allocation_bytes);
    }
}

std::size_t CountWorkers(
    std::span<const std::uint64_t> charge, std::span<const std::uint64_t> budgets
) {
    std::uint64_t workers = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t i = 0; i < charge.size(); ++i) {
        if (charge[i] != 0) workers = std::min(workers, budgets[i] / charge[i]);
    }
    return static_cast<std::size_t>(
        std::min<std::uint64_t>(workers, std::numeric_limits<std::size_t>::max())
    );
}

}  // namespace

std::optional<std::vector<std::uint64_t>> AvailableHeapBudgets(
    const MetricMemoryRequirements& requirements
) {
    std::vector<std::uint64_t> budgets;
    budgets.reserve(requirements.heaps.size());
    for (const auto& heap : requirements.heaps) {
        if (!heap.budget_available) return std::nullopt;
        budgets.push_back(
            heap.budget_bytes > heap.usage_bytes
                ? heap.budget_bytes - heap.usage_bytes
                : 0
        );
    }
    return budgets;
}

MetricMemoryRequirements GetMemoryRequirements(const Ssimulacra2Config& config) {
    return QueryMetric(config, nullptr, internal::PlanSsimulacra2Memory);
}
MetricMemoryRequirements GetMemoryRequirements(
    const Ssimulacra2Config& config, const VulkanDevice& device
) {
    return QueryMetric(config, &device, internal::PlanSsimulacra2Memory);
}
MetricMemoryRequirements GetMemoryRequirements(const ButteraugliConfig& config) {
    return QueryMetric(config, nullptr, internal::PlanButteraugliMemory);
}
MetricMemoryRequirements GetMemoryRequirements(
    const ButteraugliConfig& config, const VulkanDevice& device
) {
    return QueryMetric(config, &device, internal::PlanButteraugliMemory);
}
MetricMemoryRequirements GetMemoryRequirements(const VmafConfig& config) {
    return QueryMetric(config, nullptr, internal::PlanVmafMemory);
}
MetricMemoryRequirements GetMemoryRequirements(
    const VmafConfig& config, const VulkanDevice& device
) {
    return QueryMetric(config, &device, internal::PlanVmafMemory);
}
MetricMemoryRequirements GetMemoryRequirements(const VmafV1Config& config) {
    return QueryMetric(config, nullptr, internal::PlanVmafV1Memory);
}
MetricMemoryRequirements GetMemoryRequirements(
    const VmafV1Config& config, const VulkanDevice& device
) {
    return QueryMetric(config, &device, internal::PlanVmafV1Memory);
}
MetricMemoryRequirements GetMemoryRequirements(
    const SharedMetricConfig& config
) {
    return QueryMetric(config, nullptr, internal::PlanSharedMetricMemory);
}
MetricMemoryRequirements GetMemoryRequirements(
    const SharedMetricConfig& config, const VulkanDevice& device
) {
    return QueryMetric(config, &device, internal::PlanSharedMetricMemory);
}

MetricMemoryRequirements GetMemoryRequirements(const PsnrHvsConfig& config) {
    return QueryMetric(config, nullptr, internal::PlanPsnrHvsMemory);
}
MetricMemoryRequirements GetMemoryRequirements(
    const PsnrHvsConfig& config, const VulkanDevice& device
) {
    return QueryMetric(config, &device, internal::PlanPsnrHvsMemory);
}

MetricMemoryRequirements GetMemoryRequirements(const CvvdpConfig& config) {
    return QueryMetric(config, nullptr, internal::PlanCvvdpMemory);
}
MetricMemoryRequirements GetMemoryRequirements(
    const CvvdpConfig& config, const VulkanDevice& device
) {
    return QueryMetric(config, &device, internal::PlanCvvdpMemory);
}

std::size_t MaxWorkersForBudget(
    std::span<const MemoryScenario> per_worker,
    std::span<const std::uint64_t> heap_budgets
) {
    if (per_worker.empty()) {
        throw std::invalid_argument("memory: no scenarios");
    }
    std::vector<std::uint64_t> per_heap(heap_budgets.size(), 0);
    for (const auto& scenario : per_worker) {
        AddScenario(per_heap, scenario);
    }
    return CountWorkers(per_heap, heap_budgets);
}

std::size_t MaxWorkersForBudget(
    const MemoryScenario& scenario, std::span<const std::uint64_t> heap_budgets
) {
    return MaxWorkersForBudget(std::span(&scenario, 1), heap_budgets);
}

std::size_t MaxWorkersForBudgetWorstCase(
    std::span<const MetricMemoryRequirements> per_worker,
    std::span<const std::uint64_t> heap_budgets
) {
    if (per_worker.empty()) {
        throw std::invalid_argument("memory: no requirements");
    }
    std::vector<std::uint64_t> total(heap_budgets.size(), 0);
    for (const auto& requirements : per_worker) {
        std::vector<std::uint64_t> largest(heap_budgets.size(), 0);
        bool supported = false;
        for (const auto* scenario :
             {&requirements.direct, &requirements.staged}) {
            if (!scenario->supported) continue;
            supported = true;
            std::vector<std::uint64_t> charge(heap_budgets.size(), 0);
            AddScenario(charge, *scenario);
            for (std::size_t i = 0; i < charge.size(); ++i) {
                largest[i] = std::max(largest[i], charge[i]);
            }
        }
        if (!supported) {
            throw std::invalid_argument("memory: unsupported requirements");
        }
        for (std::size_t i = 0; i < total.size(); ++i) {
            AddBytes(total[i], largest[i]);
        }
    }
    return CountWorkers(total, heap_budgets);
}

}  // namespace vqm
