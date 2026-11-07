#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

#include "vqm_memory.h"
#include "vulkan/vulkan_shared_storage.h"

namespace vqm::internal {

struct MetricScratchAccess {
    template <class Pipeline, class Config>
    static std::unique_ptr<Pipeline> Create(
        Config config, const VulkanDevice* device,
        std::shared_ptr<SharedMetricStorage> scratch
    ) {
        if (scratch && config.gpu_resident_images != 0)
            throw std::invalid_argument(
                "shared scratch cannot retain resident images"
            );
        if constexpr (requires { config.input_bytes; }) {
            if (scratch && config.input_bytes != 0)
                throw std::invalid_argument(
                    "shared scratch cannot retain input handles"
                );
        }
        return std::unique_ptr<Pipeline>(
            new Pipeline(std::move(config), device, std::move(scratch))
        );
    }
};

class MetricScratchBuilder {
public:
    void Add(const MetricMemoryRequirements& requirements) {
        const auto& scratch = requirements.scratch;
        if (!scratch.reusable)
            throw std::invalid_argument(
                "metric cannot share scratch with retained input handles"
            );
        arena_bytes_ = std::max(arena_bytes_, scratch.arena_bytes);
        staging_bytes_ =
            std::max(staging_bytes_, scratch.fallback_staging_bytes);
    }

    std::shared_ptr<SharedMetricStorage> Build(
        const VulkanDevice* device, MemoryPolicy policy = MemoryPolicy::Automatic
    ) const {
        return std::make_shared<SharedMetricStorage>(
            RequireVulkanContext(device), arena_bytes_, staging_bytes_, policy
        );
    }

    BufferMemoryPlan MemoryPlan() const {
        return {arena_bytes_, staging_bytes_};
    }

private:
    std::uint64_t arena_bytes_ = 0;
    std::uint64_t staging_bytes_ = 0;
};

MetricMemoryRequirements QueryMemoryPlan(
    const BufferMemoryPlan& plan, MemoryPolicy policy, const VulkanDevice* device
);

MetricMemoryRequirements PrivateMemoryRequirements(
    const MetricMemoryRequirements& requirements, MemoryPolicy policy,
    const VulkanDevice* device
);

inline void AppendPrivateMemory(
    PipelineMemoryUsage& destination, const PipelineMemoryUsage& source
) {
    destination.auxiliary.insert(
        destination.auxiliary.end(), source.auxiliary.begin(),
        source.auxiliary.end()
    );
}

}  // namespace vqm::internal
