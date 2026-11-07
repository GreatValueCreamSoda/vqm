// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <array>
#include <cstddef>

#include "vqm_image.h"

namespace vqm {

class Ssimulacra2Pipeline;

// A host plane with a positive row stride, in bytes. The caller owns the data
// and must ensure the complete plane remains readable for the call. The API
// cannot determine the bounds of an allocation from its pointer.
struct PlaneView {
    const std::byte* data = nullptr;
    std::size_t stride_bytes = 0;
};

// Non-owning, planar host image: Y/R, U/G, V/B, alpha, in the same sample and
// chroma order as Image. Each present plane may reside in a separate allocation
// and may have padding at the end of its rows. Absent planes are ignored.
struct ImageView {
    Size size;
    ImageFormat format;
    std::array<PlaneView, 4> planes{};
};

// Copies a host view into the packed owning Image layout. Each present plane
// needs a non-null pointer and a stride at least as long as its row. Throws
// std::invalid_argument for an invalid image or plane and std::overflow_error
// when its geometry or stride cannot be addressed. This is a CPU copy.
[[nodiscard]] Image CopyImage(const ImageView& view);

// Copies both views directly to the pipeline's upload regions and scores them.
// The background has the same single-background semantics as
// Ssimulacra2Pipeline::Compute; the CLI's two-background alpha policy is not
// applied.
[[nodiscard]] double Score(
    Ssimulacra2Pipeline& pipeline, const ImageView& original,
    const ImageView& distorted, float background = 0.5f
);

}  // namespace vqm
