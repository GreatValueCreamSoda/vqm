// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <cstdint>
#include <vector>

namespace vqm {

// Select the arena allocation path. Automatic prefers Direct and falls back to
// Staged on allocation failure. Direct and Staged require that exact path;
// construction fails if it is unavailable, instead of silently switching.
enum class MemoryPolicy : std::uint8_t { Automatic, Direct, Staged };

// One library-owned Vulkan buffer and its bound device-memory allocation.
// An absent buffer has zero sizes; heap and memory_type are then unused.
struct MemoryAllocation {
    std::uint64_t buffer_bytes = 0;
    std::uint64_t allocation_bytes = 0;
    std::uint32_t memory_type = 0;
    std::uint32_t heap = 0;
};

// The allocations retained by one pipeline. Other Vulkan objects and driver
// resources are not included.
struct PipelineMemoryUsage {
    bool uses_staging = false;
    MemoryAllocation arena;
    MemoryAllocation staging;
    // Metric-private device-local storage (for example temporal histories).
    // Every allocation is reported separately with its actual heap/type.
    std::vector<MemoryAllocation> auxiliary = {};
};

}  // namespace vqm
