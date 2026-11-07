// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include "handles.h"

#include <atomic>
#include <stdexcept>
#include <string>

namespace vqm::internal {

std::uint64_t NextHandleIncarnation() {
    static std::atomic<std::uint64_t> next{1};
    std::uint64_t candidate = next.load(std::memory_order_relaxed);
    for (;;) {
        if (candidate > kHandleIncarnationMax) {
            throw std::overflow_error(
                "pipeline handle identifier space exhausted"
            );
        }
        if (next.compare_exchange_weak(
                candidate, candidate + 1, std::memory_order_relaxed,
                std::memory_order_relaxed
            )) {
            return candidate;
        }
    }
}

void ResidentTable::Resize(std::size_t capacity) {
    slots_.assign(capacity, Slot{});
}

std::size_t ResidentTable::RequireFree() const {
    for (std::size_t index = 0; index < slots_.size(); ++index) {
        if (!slots_[index].live) return index;
    }
    throw std::invalid_argument(kResidentCapacityError);
}

UploadedImage ResidentTable::Acquire(
    std::size_t index, std::uint32_t device, Size size, std::uint64_t generation
) {
    Slot& slot = slots_[index];
    slot.device = device;
    slot.live = true;
    slot.generation = generation;
    const std::uint64_t handle = (generation << kResidentIndexBits) |
                                 (static_cast<std::uint64_t>(index) + 1u);
    return UploadedImage{handle, size};
}

std::optional<std::size_t> ResidentTable::Index(
    const UploadedImage& image
) const {
    const std::uint64_t index = image.slot & kResidentIndexMask;
    if (index == 0 || index > slots_.size()) return std::nullopt;
    const std::size_t candidate = static_cast<std::size_t>(index - 1);
    const Slot& slot = slots_[candidate];
    const std::uint64_t generation = image.slot >> kResidentIndexBits;
    if (!slot.live || slot.generation != generation) {
        return std::nullopt;
    }
    return candidate;
}

const ResidentTable::Slot& ResidentTable::Require(
    const UploadedImage& image, Size size, const char* name
) const {
    const std::optional<std::size_t> index = Index(image);
    if (!index) {
        throw std::invalid_argument(
            std::string(name) + ": not a live resident image"
        );
    }
    if (image.size != size) {
        throw std::invalid_argument(
            std::string(name) +
            ": image size does not match the pipeline geometry"
        );
    }
    return at(*index);
}

}  // namespace vqm::internal
