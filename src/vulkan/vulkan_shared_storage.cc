// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include "vulkan_shared_storage.h"

#include <stdexcept>
#include <utility>

namespace vqm::internal {
namespace {
VulkanRuntime::Config BackingConfig(MemoryPolicy memory_policy) {
    VulkanRuntime::Config result;
    result.memory_policy = memory_policy;
    return result;
}
}  // namespace

SharedMetricStorage::SharedMetricStorage(
    std::shared_ptr<VulkanContext> context, std::size_t arena_bytes,
    std::size_t staging_bytes, MemoryPolicy memory_policy
)
    : backing_(std::move(context), BackingConfig(memory_policy)) {
    if (arena_bytes == 0) throw std::invalid_argument("shared arena is empty");
    backing_.CreateArena(arena_bytes);
    if (!backing_.arena_visible()) {
        if (staging_bytes == 0)
            throw std::invalid_argument("shared staging arena is empty");
        backing_.CreateStaging(staging_bytes);
    }
}

void SharedMetricStorage::RequireIdle(const VulkanRuntime* caller) const {
    if (borrower_ && borrower_ != caller && borrower_->pending())
        throw std::logic_error("shared scratch: another metric is in flight");
}

void SharedMetricStorage::Submitted(VulkanRuntime* caller) noexcept {
    borrower_ = caller;
}

void SharedMetricStorage::Release(const VulkanRuntime* caller) noexcept {
    if (borrower_ == caller) borrower_ = nullptr;
}

}  // namespace vqm::internal
