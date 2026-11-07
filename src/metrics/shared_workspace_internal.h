// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include "vqm_shared.h"
#include "vulkan/memory_plan.h"

namespace vqm::internal {

struct VulkanContext;

// Peak sequential arena/staging requirements of the shared spatial workspace.
[[nodiscard]] BufferMemoryPlan PlanSharedMetricMemory(
    const SharedMetricConfig& config, const VulkanContext& context
);

}  // namespace vqm::internal
