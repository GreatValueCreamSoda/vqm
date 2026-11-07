// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <cstdint>
#include <vector>

namespace vqm::internal {

// Logical buffer sizes shared by metric layout planners. Vulkan allocation
// requirements may reserve more bytes; staging is omitted for a host-visible
// arena. A plan describes requirements, not available allocation capacity.
struct BufferMemoryPlan {
    std::uint64_t arena_bytes = 0;
    std::uint64_t fallback_staging_bytes = 0;
    std::vector<std::uint64_t> auxiliary_bytes = {};
};

}  // namespace vqm::internal
