// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include "vqm_pool.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vqm_views.h"
#include "vqm_vulkan.h"

namespace vqm {
namespace {

// Pipelines own their workspaces, command buffers and tokens. Only this host
// thread calls Submit/Finish, so each slot stays idle until its prior token has
// been consumed. The caller still synchronizes other users of a borrowed queue.
template <typename PipelineType, typename Config>
class PoolState {
public:
    PoolState(
        const Config& config, std::size_t workers, const VulkanDevice* device
    ) {
        if (workers == 0)
            throw std::invalid_argument("pool needs at least one worker");
        slots_.reserve(workers);
        for (std::size_t i = 0; i < workers; ++i) {
            if (device) {
                slots_.emplace_back(PipelineType(config, *device));
            } else {
                slots_.emplace_back(PipelineType(config));
            }
        }
    }

    [[nodiscard]] std::size_t Workers() const noexcept { return slots_.size(); }

    template <typename Output, typename Pair, typename Convert>
    std::vector<Output> Run(
        std::span<const Pair> pairs, float background, Convert convert
    ) {
        if (poisoned_) {
            throw std::logic_error("pool cannot be reused after a failed drain");
        }
        std::vector<Output> results(pairs.size());
        try {
            for (std::size_t index = 0; index < pairs.size(); ++index) {
                Slot& slot = slots_[index % slots_.size()];
                FinishSlot(slot, results, convert);
                const Pair& pair = pairs[index];
                slot.token = slot.pipeline.Submit(
                    pair.original, pair.distorted, background
                );
                slot.result_index = index;
                slot.pending = true;
            }
            for (Slot& slot : slots_) FinishSlot(slot, results, convert);
        } catch (...) {
            const std::exception_ptr primary = std::current_exception();
            // Preserve the error that interrupted the batch while ensuring
            // every earlier submission completes before its input is reused.
            DrainPendingAfterFailure();
            std::rethrow_exception(primary);
        }
        return results;
    }

private:
    struct Slot {
        explicit Slot(PipelineType pipeline_in)
            : pipeline(std::move(pipeline_in)) {}
        PipelineType pipeline;
        std::uint64_t token = 0;
        std::size_t result_index = 0;
        bool pending = false;
    };

    template <typename Output, typename Convert>
    static void FinishSlot(
        Slot& slot, std::vector<Output>& results, Convert& convert
    ) {
        if (!slot.pending) return;
        auto value = slot.pipeline.Finish(slot.token);
        slot.pending = false;
        results[slot.result_index] = convert(std::move(value));
    }

    void DrainPendingAfterFailure() noexcept {
        bool drain_failed = false;
        for (Slot& slot : slots_) {
            if (!slot.pending) continue;
            try {
                (void)slot.pipeline.Finish(slot.token);
                slot.pending = false;
            } catch (...) {
                drain_failed = true;
            }
        }
        if (drain_failed) {
            // Pipeline destruction makes its final best-effort fence wait
            // before releasing Vulkan resources, so no live slot escapes to
            // the caller after the error.
            slots_.clear();
            poisoned_ = true;
        }
    }

    std::vector<Slot> slots_;
    bool poisoned_ = false;
};

template <typename T>
T& RequirePool(std::unique_ptr<T>& impl) {
    if (!impl) throw std::logic_error("pool has been moved from");
    return *impl;
}

}  // namespace

struct Ssimulacra2Pool::Impl
    : PoolState<Ssimulacra2Pipeline, Ssimulacra2Config> {
    using PoolState::PoolState;
};

Ssimulacra2Pool::Ssimulacra2Pool(Ssimulacra2Config config, std::size_t workers)
    : impl_(std::make_unique<Impl>(config, workers, nullptr)) {}

Ssimulacra2Pool::Ssimulacra2Pool(
    Ssimulacra2Config config, std::size_t workers, const VulkanDevice& device
)
    : impl_(std::make_unique<Impl>(config, workers, &device)) {}

Ssimulacra2Pool::~Ssimulacra2Pool() = default;
Ssimulacra2Pool::Ssimulacra2Pool(Ssimulacra2Pool&&) noexcept = default;
Ssimulacra2Pool& Ssimulacra2Pool::operator=(Ssimulacra2Pool&&) noexcept =
    default;

std::size_t Ssimulacra2Pool::Workers() const noexcept {
    return impl_ ? impl_->Workers() : 0;
}

std::vector<Ssimulacra2Result> Ssimulacra2Pool::ComputeMany(
    std::span<const ImagePair> pairs, float background
) {
    return RequirePool(impl_).Run<Ssimulacra2Result>(
        pairs, background, [](Ssimulacra2Result value) { return value; }
    );
}

std::vector<Ssimulacra2Result> Ssimulacra2Pool::ComputeMany(
    std::span<const ImageViewPair> pairs, float background
) {
    return RequirePool(impl_).Run<Ssimulacra2Result>(
        pairs, background, [](Ssimulacra2Result value) { return value; }
    );
}

std::vector<Ssimulacra2Result> Ssimulacra2Pool::ComputeMany(
    std::span<const VulkanBufferPair> pairs, float background
) {
    return RequirePool(impl_).Run<Ssimulacra2Result>(
        pairs, background, [](Ssimulacra2Result value) { return value; }
    );
}

std::vector<double> Ssimulacra2Pool::ScoreMany(
    std::span<const ImagePair> pairs, float background
) {
    return RequirePool(impl_).Run<double>(
        pairs, background, [](Ssimulacra2Result value) { return value.Score(); }
    );
}

std::vector<double> Ssimulacra2Pool::ScoreMany(
    std::span<const ImageViewPair> pairs, float background
) {
    return RequirePool(impl_).Run<double>(
        pairs, background, [](Ssimulacra2Result value) { return value.Score(); }
    );
}

std::vector<double> Ssimulacra2Pool::ScoreMany(
    std::span<const VulkanBufferPair> pairs, float background
) {
    return RequirePool(impl_).Run<double>(
        pairs, background, [](Ssimulacra2Result value) { return value.Score(); }
    );
}

struct ButteraugliPool::Impl
    : PoolState<ButteraugliPipeline, ButteraugliConfig> {
    using PoolState::PoolState;
};

ButteraugliPool::ButteraugliPool(ButteraugliConfig config, std::size_t workers)
    : impl_(std::make_unique<Impl>(config, workers, nullptr)) {}

ButteraugliPool::ButteraugliPool(
    ButteraugliConfig config, std::size_t workers, const VulkanDevice& device
)
    : impl_(std::make_unique<Impl>(config, workers, &device)) {}

ButteraugliPool::~ButteraugliPool() = default;
ButteraugliPool::ButteraugliPool(ButteraugliPool&&) noexcept = default;
ButteraugliPool& ButteraugliPool::operator=(ButteraugliPool&&) noexcept =
    default;

std::size_t ButteraugliPool::Workers() const noexcept {
    return impl_ ? impl_->Workers() : 0;
}

std::vector<ButteraugliScore> ButteraugliPool::ComputeMany(
    std::span<const ImagePair> pairs, float background
) {
    return RequirePool(impl_).Run<ButteraugliScore>(
        pairs, background, [](ButteraugliScore value) { return value; }
    );
}

std::vector<ButteraugliScore> ButteraugliPool::ComputeMany(
    std::span<const ImageViewPair> pairs, float background
) {
    return RequirePool(impl_).Run<ButteraugliScore>(
        pairs, background, [](ButteraugliScore value) { return value; }
    );
}

std::vector<ButteraugliScore> ButteraugliPool::ComputeMany(
    std::span<const VulkanBufferPair> pairs, float background
) {
    return RequirePool(impl_).Run<ButteraugliScore>(
        pairs, background, [](ButteraugliScore value) { return value; }
    );
}

}  // namespace vqm
