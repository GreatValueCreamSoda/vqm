// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <cstddef>
#include <memory>

#include "vulkan_runtime.h"

namespace vqm::internal {

// Backing storage for a serial metric workspace. Pipeline command objects and
// descriptors remain private. A pipeline may access the storage only while its
// owning workspace is executing it; retained state belongs to each runtime.
class SharedMetricStorage {
public:
    SharedMetricStorage(
        std::shared_ptr<VulkanContext> context, std::size_t arena_bytes,
        std::size_t staging_bytes, MemoryPolicy memory_policy
    );
    VulkanRuntime& backing() noexcept { return backing_; }
    const VulkanRuntime& backing() const noexcept { return backing_; }
    void RequireIdle(const VulkanRuntime* caller) const;
    void Submitted(VulkanRuntime* caller) noexcept;
    void Release(const VulkanRuntime* caller) noexcept;

private:
    VulkanRuntime backing_;
    VulkanRuntime* borrower_ = nullptr;
};

}  // namespace vqm::internal
