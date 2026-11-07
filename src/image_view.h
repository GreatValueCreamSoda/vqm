// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <array>
#include <cstddef>

#include "input.h"
#include "vqm_views.h"

namespace vqm::internal {

struct ImageViewLayout {
    std::array<PlaneGeometry, 4> planes;
    std::size_t bytes = 0;
};

// Validate the format, pointers, strides, and pointer arithmetic once before
// beginning an upload. The caller still owns the source allocation bounds.
[[nodiscard]] ImageViewLayout DescribeImageView(const ImageView& view);

// Pack a validated view into exactly layout.bytes bytes at destination.
// Destination may be ordinary host memory or a mapped Vulkan upload region.
void PackImageView(
    std::byte* destination, const ImageView& view, const ImageViewLayout& layout,
    std::uint8_t plane_mask = 0x0f
);

}  // namespace vqm::internal
