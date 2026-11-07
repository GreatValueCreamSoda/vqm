// Copyright 2001-2012 Xiph.Org and contributors.
// BSD-2-Clause; see licenses/XIPH-PSNR-HVS-LICENSE.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "psnr_hvs.h"

namespace vqm::internal {

class PsnrHvsGpuPipeline {
public:
    explicit PsnrHvsGpuPipeline(
        const PsnrHvsConfig& config, const VulkanDevice* device = nullptr,
        std::shared_ptr<SharedMetricStorage> scratch = {}
    );
    ~PsnrHvsGpuPipeline();
    PsnrHvsGpuPipeline(const PsnrHvsGpuPipeline&) = delete;
    PsnrHvsGpuPipeline& operator=(const PsnrHvsGpuPipeline&) = delete;

    void RequireIdle() const;
    [[nodiscard]] PipelineMemoryUsage GetMemoryUsage() const;
    [[nodiscard]] std::optional<std::uint32_t> Upload(
        const Image& image, std::size_t capacity
    );
    void Release(std::uint32_t slot);
    [[nodiscard]] ImageBuffer AllocateImage(const ImageFormat& format);
    [[nodiscard]] std::byte* MapImage(ImageBuffer image);
    void Release(ImageBuffer image);

    [[nodiscard]] std::uint64_t Submit(
        const Image& original, const Image& distorted
    );
    [[nodiscard]] std::uint64_t Submit(
        const ImageView& original, const ImageView& distorted
    );
    [[nodiscard]] std::uint64_t Submit(
        std::uint32_t original, std::uint32_t distorted
    );
    [[nodiscard]] std::uint64_t Submit(
        std::uint32_t original, const Image& distorted
    );
    [[nodiscard]] std::uint64_t Submit(
        const ImageBuffer& original, const ImageBuffer& distorted
    );
    [[nodiscard]] std::uint64_t Submit(
        const VulkanBufferView& original, const VulkanBufferView& distorted
    );
    [[nodiscard]] std::uint64_t Submit(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        const VulkanSubmitSync& sync
    );
    [[nodiscard]] PsnrHvsScore Finish(std::uint64_t token);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vqm::internal
