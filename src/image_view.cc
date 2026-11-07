// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include "vqm_views.h"
#include "ssimulacra2.h"

#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "image_view.h"

namespace vqm::internal {

ImageViewLayout DescribeImageView(const ImageView& view) {
    if (view.size.width == 0 || view.size.height == 0) {
        throw std::invalid_argument("image view: size must be non-zero");
    }
    ValidateImage(view.format, view.size, "image view");
    ImageViewLayout layout{PlaneLayout(view.format, view.size),
                           RequiredBytes(view.format, view.size)};
    for (std::size_t plane = 0; plane < layout.planes.size(); ++plane) {
        const auto& geometry = layout.planes[plane];
        if (!geometry.present()) continue;
        const std::size_t row_bytes = geometry.width * geometry.sample_bytes;
        const PlaneView& source = view.planes[plane];
        if (source.data == nullptr || source.stride_bytes < row_bytes) {
            throw std::invalid_argument(
                "image view: missing plane or short stride"
            );
        }
        // Bound the last source row offset, including its readable bytes, to a
        // representable pointer difference before doing any pointer arithmetic.
        const auto max_offset =
            static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
        if (row_bytes > max_offset ||
            (geometry.height > 1 &&
             source.stride_bytes >
                 (max_offset - row_bytes) / (geometry.height - 1))) {
            throw std::overflow_error("image view: plane stride is too large");
        }
    }

    return layout;
}

void PackImageView(
    std::byte* destination, const ImageView& view, const ImageViewLayout& layout,
    std::uint8_t plane_mask
) {
    for (std::size_t plane = 0; plane < layout.planes.size(); ++plane) {
        const auto& geometry = layout.planes[plane];
        if (!geometry.present() || (plane_mask & (1u << plane)) == 0) continue;
        const std::size_t row_bytes = geometry.width * geometry.sample_bytes;
        const PlaneView& source = view.planes[plane];
        std::byte* target = destination + geometry.offset;
        if (source.stride_bytes == row_bytes) {
            std::memcpy(target, source.data, row_bytes * geometry.height);
        } else {
            for (std::size_t row = 0; row < geometry.height; ++row) {
                std::memcpy(
                    target + row * row_bytes,
                    source.data + row * source.stride_bytes, row_bytes
                );
            }
        }
        // Plane starts and the final length are word-aligned. Initialize the
        // at-most-three unused trailing bytes without clearing all pixel data.
        const std::size_t end = geometry.offset + row_bytes * geometry.height;
        const std::size_t aligned_end = (end + 3) & ~std::size_t{3};
        if (end < aligned_end) {
            std::memset(destination + end, 0, aligned_end - end);
        }
    }
}

}  // namespace vqm::internal

namespace vqm {

Image CopyImage(const ImageView& view) {
    const internal::ImageViewLayout layout = internal::DescribeImageView(view);
    Image image{view.size, view.format, {}};
    image.data.resize(layout.bytes);
    internal::PackImageView(
        reinterpret_cast<std::byte*>(image.data.data()), view, layout
    );
    return image;
}

double Score(
    Ssimulacra2Pipeline& pipeline, const ImageView& original,
    const ImageView& distorted, float background
) {
    if (original.size != distorted.size) {
        throw std::invalid_argument("image views must have the same size");
    }
    return pipeline.Compute(original, distorted, background).Score();
}

}  // namespace vqm
