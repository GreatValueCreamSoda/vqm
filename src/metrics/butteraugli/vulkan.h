// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// The Butteraugli backend's factory, kept in its own header so
// src/metrics/butteraugli/tables.cc can name it without pulling in the Vulkan
// headers.

#pragma once

#include <cstddef>
#include <memory>

#include "metrics/butteraugli/butteraugli_internal.h"
#include "butteraugli.h"

namespace vqm {
struct ButteraugliConfig;
}

namespace vqm::internal {

struct VulkanContext;
class SharedMetricStorage;

std::unique_ptr<ButteraugliGpuPipeline> CreateVulkanButteraugliPipeline(
    const ButteraugliGeometry& geometry, const ButteraugliParams& params,
    std::size_t resident_images, ChromaFilter chroma_filter,
    bool streaming_inputs, std::size_t input_bytes, MemoryPolicy memory_policy,
    const VulkanDevice* borrowed_device = nullptr
);

std::unique_ptr<ButteraugliGpuPipeline> CreateSharedVulkanButteraugliPipeline(
    const ButteraugliGeometry& geometry, const ButteraugliParams& params,
    const ButteraugliConfig& config, std::shared_ptr<VulkanContext> context,
    std::shared_ptr<SharedMetricStorage> storage
);

}  // namespace vqm::internal
