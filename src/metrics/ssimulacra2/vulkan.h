// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <cstddef>
#include <memory>
#include "metrics/ssimulacra2/ssimulacra2_internal.h"

namespace vqm::internal {

struct VulkanContext;
class SharedMetricStorage;

// Throws std::runtime_error on Vulkan failure, and with `half_planes` on a
// device without 16-bit storage buffers. Throws std::invalid_argument when the
// geometry exceeds the device's storage-buffer range. `resident_images` is how
// many images the arena reserves room for in Ssimulacra2Pipeline::Upload.
std::unique_ptr<Ssimulacra2GpuPipeline> CreateVulkanSsimulacra2Pipeline(
    const Ssimulacra2ScaleLayout& scales, const GaussianKernel& kernel,
    std::size_t resident_images, bool half_planes, ChromaFilter chroma_filter,
    bool streaming_inputs, std::size_t input_bytes, MemoryPolicy memory_policy,
    const VulkanDevice* borrowed_device = nullptr
);

std::unique_ptr<Ssimulacra2GpuPipeline> CreateSharedVulkanSsimulacra2Pipeline(
    const Ssimulacra2ScaleLayout& scales, const GaussianKernel& kernel,
    const Ssimulacra2Config& config, std::shared_ptr<VulkanContext> context,
    std::shared_ptr<SharedMetricStorage> storage
);

}  // namespace vqm::internal
