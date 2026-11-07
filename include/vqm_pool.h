// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "butteraugli.h"
#include "ssimulacra2.h"

namespace vqm {

// Non-owning input pairs. Both objects and the memory they reference must stay
// alive and unchanged until ComputeMany/ScoreMany returns, including if the
// call throws. Vulkan buffers follow the synchronization contract documented
// in vqm_vulkan.h.
struct ImagePair {
    const Image& original;
    const Image& distorted;
};

struct ImageViewPair {
    const ImageView& original;
    const ImageView& distorted;
};

struct VulkanBufferPair {
    const VulkanBufferView& original;
    const VulkanBufferView& distorted;
};

// A bounded group of independent pipelines for one geometry. ComputeMany
// submits one pair per available pipeline from the calling thread and returns
// results in input order. Each pipeline has at most one submission in flight;
// inputs can overlap on the device. The pool is not thread-safe. A borrowed
// VkQueue must be externally synchronized against other users of that queue.
//
// Construction allocates `workers` full pipelines and their working memory;
// it throws std::invalid_argument when workers is zero. Set the underlying
// config's resident/streaming options to the input paths the application needs.
// Vulkan pairs require the borrowed-device constructor. All images must match
// the configured size. The background policy is the same single background as
// Ssimulacra2Pipeline::Compute, including for alpha images.
//
// If a pair fails, already-submitted work is drained before the exception is
// rethrown. If draining itself fails, later calls throw std::logic_error; the
// pool must then be destroyed. Destruction waits for any pending GPU work.
class Ssimulacra2Pool {
public:
    Ssimulacra2Pool(Ssimulacra2Config config, std::size_t workers);
    Ssimulacra2Pool(
        Ssimulacra2Config config, std::size_t workers, const VulkanDevice& device
    );
    ~Ssimulacra2Pool();
    Ssimulacra2Pool(Ssimulacra2Pool&&) noexcept;
    Ssimulacra2Pool& operator=(Ssimulacra2Pool&&) noexcept;
    Ssimulacra2Pool(const Ssimulacra2Pool&) = delete;
    Ssimulacra2Pool& operator=(const Ssimulacra2Pool&) = delete;

    [[nodiscard]] std::size_t Workers() const noexcept;
    [[nodiscard]] std::vector<Ssimulacra2Result> ComputeMany(
        std::span<const ImagePair> pairs, float background = 0.5f
    );
    [[nodiscard]] std::vector<Ssimulacra2Result> ComputeMany(
        std::span<const ImageViewPair> pairs, float background = 0.5f
    );
    [[nodiscard]] std::vector<Ssimulacra2Result> ComputeMany(
        std::span<const VulkanBufferPair> pairs, float background = 0.5f
    );
    [[nodiscard]] std::vector<double> ScoreMany(
        std::span<const ImagePair> pairs, float background = 0.5f
    );
    [[nodiscard]] std::vector<double> ScoreMany(
        std::span<const ImageViewPair> pairs, float background = 0.5f
    );
    [[nodiscard]] std::vector<double> ScoreMany(
        std::span<const VulkanBufferPair> pairs, float background = 0.5f
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The same bounded ordering, lifetime and synchronization contract as
// Ssimulacra2Pool, with Butteraugli's complete score and distance_p3 per pair.
class ButteraugliPool {
public:
    ButteraugliPool(ButteraugliConfig config, std::size_t workers);
    ButteraugliPool(
        ButteraugliConfig config, std::size_t workers, const VulkanDevice& device
    );
    ~ButteraugliPool();
    ButteraugliPool(ButteraugliPool&&) noexcept;
    ButteraugliPool& operator=(ButteraugliPool&&) noexcept;
    ButteraugliPool(const ButteraugliPool&) = delete;
    ButteraugliPool& operator=(const ButteraugliPool&) = delete;

    [[nodiscard]] std::size_t Workers() const noexcept;
    [[nodiscard]] std::vector<ButteraugliScore> ComputeMany(
        std::span<const ImagePair> pairs, float background = 0.5f
    );
    [[nodiscard]] std::vector<ButteraugliScore> ComputeMany(
        std::span<const ImageViewPair> pairs, float background = 0.5f
    );
    [[nodiscard]] std::vector<ButteraugliScore> ComputeMany(
        std::span<const VulkanBufferPair> pairs, float background = 0.5f
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vqm
