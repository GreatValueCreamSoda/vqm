// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// VMAF: Copyright (c) Netflix, Inc. BSD-2-Clause-Patent;
// see licenses/NETFLIX-VMAF-LICENSE.

#pragma once

#include "vulkan/memory_plan.h"

namespace vqm {
struct VmafConfig;
struct VmafV1Config;
namespace internal {
struct VulkanContext;

// These are the same layouts the pipeline constructors use. Neither allocates
// GPU resources. A returned plan does not imply that available memory suffices.
[[nodiscard]] BufferMemoryPlan PlanVmafMemory(
    const VmafConfig& config, const VulkanContext& context
);
[[nodiscard]] BufferMemoryPlan PlanVmafV1Memory(
    const VmafV1Config& config, const VulkanContext& context
);

}  // namespace internal
}  // namespace vqm
