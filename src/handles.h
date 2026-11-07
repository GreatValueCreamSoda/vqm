// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// Generation-protected resident handles shared by metric pipelines. This
// module owns handle identity and live-slot bookkeeping; image layout and
// content validation live in input.h.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vqm_image.h"

namespace vqm::internal {

// UploadedImage::slot packs a globally unique incarnation with the pool index:
// [incarnation:52][index + 1:12]. Incarnations never wrap or get reused, even
// across pipelines and handle types, so stale and foreign handles cannot alias.
inline constexpr std::uint32_t kResidentIndexBits = 12;
inline constexpr std::uint32_t kResidentIndexMask =
    (1u << kResidentIndexBits) - 1u;
inline constexpr std::uint64_t kHandleIncarnationMax =
    UINT64_MAX >> kResidentIndexBits;

// Process-wide unique identifiers shared by image handles and submissions.
// Throws std::overflow_error instead of reusing an identifier at exhaustion.
[[nodiscard]] std::uint64_t NextHandleIncarnation();

inline constexpr const char* kResidentCapacityError =
    "resident image capacity exhausted; raise gpu_resident_images";

// One facade's resident pool: one entry per configured slot, each holding the
// generation its handle carries and the backend's device slot index.
class ResidentTable {
public:
    struct Slot {
        bool live = false;
        std::uint64_t generation = 0;
        std::uint32_t device = 0;
    };

    void Resize(std::size_t capacity);
    [[nodiscard]] std::size_t capacity() const noexcept { return slots_.size(); }

    // Reserve identity before the backend upload and publish it only after
    // that upload succeeds. The caller performs its idle and image checks;
    // upload(capacity) returns the backend's optional device slot.
    template <class UploadFn>
    [[nodiscard]] UploadedImage Upload(Size size, UploadFn&& upload) {
        const std::size_t index = RequireFree();
        const std::uint64_t generation = NextHandleIncarnation();
        const std::optional<std::uint32_t> device =
            std::forward<UploadFn>(upload)(capacity());
        if (!device) throw std::invalid_argument(kResidentCapacityError);
        return Acquire(index, *device, size, generation);
    }

    // Stale releases are harmless. Keep the handle live if the backend release
    // throws; once released, no stale device index remains in the dead slot.
    // The caller retains any idle check required before liveness validation.
    template <class ReleaseFn>
    void Release(const UploadedImage& image, ReleaseFn&& release) {
        const std::optional<std::size_t> index = Index(image);
        if (!index) return;
        Slot& slot = at(*index);
        std::forward<ReleaseFn>(release)(slot.device);
        slot.device = 0;
        slot.live = false;
    }

    // First free slot; throws the shared capacity error if the pool is full.
    [[nodiscard]] std::size_t RequireFree() const;

    [[nodiscard]] Slot& at(std::size_t index) { return slots_[index]; }
    [[nodiscard]] const Slot& at(std::size_t index) const {
        return slots_[index];
    }

    // Marks `index` live with a previously reserved unique incarnation and
    // returns the packed handle and image geometry.
    [[nodiscard]] UploadedImage Acquire(
        std::size_t index, std::uint32_t device, Size size,
        std::uint64_t generation
    );

    // Pool index of a live, generation-matching handle; std::nullopt when the
    // handle is stale, out of range or belongs to an earlier upload.
    [[nodiscard]] std::optional<std::size_t> Index(
        const UploadedImage& image
    ) const;

    // Validate liveness before geometry, preserving the facade error contract.
    [[nodiscard]] const Slot& Require(
        const UploadedImage& image, Size size, const char* name
    ) const;

private:
    std::vector<Slot> slots_;
};

}  // namespace vqm::internal
