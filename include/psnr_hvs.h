// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
// PSNR-HVS weighting and masking: Copyright 2001-2012 Xiph.Org and contributors.
// See licenses/XIPH-PSNR-HVS-LICENSE.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "vqm_image.h"
#include "vqm_memory_types.h"

namespace vqm {

struct PsnrHvsConfig {
    Size size;
    // Raw upload slots, each reserving 16 bytes per configured image pixel.
    // Zero disables Upload; both images are prepared on every comparison.
    std::size_t gpu_resident_images = 2;
    // Bytes reserved for AllocateImage, carved into 64-byte aligned regions.
    std::size_t input_bytes = 0;
    // Reserve raw source storage for host image/view submissions.
    bool streaming_inputs = true;
    MemoryPolicy memory_policy = MemoryPolicy::Automatic;
};

struct PsnrHvsScore {
    // Combined and individual plane scores in dB, promoted from FP32 results.
    // Higher is better; zero
    // measured error produces positive infinity, including masked changes.
    double score = 0.0;
    double y = 0.0;
    double cb = 0.0;
    double cr = 0.0;
    // Mean perceptually weighted squared errors / (2^depth - 1)^2, Y/Cb/Cr.
    // The combined score is -10*log10(0.8*Y + 0.1*Cb + 0.1*Cr).
    std::array<double, 3> normalized_error{};
};

// Frame-local libvmaf/Xiph PSNR-HVS on a Vulkan compute device. Supports
// opaque planar YUV420/422/444: Byte at depth 8 or Word at depth 10/12.
// Inputs have matching geometry and format; all planes are at least 8x8.
// Samples are valid code values at the declared depth. Range and transfer
// metadata do not cause conversion; there is no RGB or alpha adapter.
// Complete 8x8 windows use stride 7 and ignore uncovered trailing pixels.
//
// One pipeline is one worker, is not thread-safe and allows one submission
// in flight. Compute is Submit followed by Finish. Invalid inputs/handles
// throw std::invalid_argument; Submit/Upload/AllocateImage/MapImage/Release
// throw std::logic_error while work is in flight. Finish rejects unknown or
// consumed tokens. See vqm_views.h and vqm_vulkan.h for view and borrowed
// buffer lifetime/synchronization requirements.
//
// Sample processing, DCT, masking, reduction, normalization and dB conversion
// use FP32 regardless of input storage. Integer operations only unpack storage
// and address data. Final scores need not be numerically identical to libvmaf's
// integer DCT and serial FP32 sum. Ranking qualification is separate.
// Query vqm_memory.h for allocation requirements before construction.
class PsnrHvsPipeline {
public:
    explicit PsnrHvsPipeline(PsnrHvsConfig config);
    PsnrHvsPipeline(PsnrHvsConfig config, const VulkanDevice& device);
    ~PsnrHvsPipeline();
    PsnrHvsPipeline(PsnrHvsPipeline&&) noexcept;
    PsnrHvsPipeline& operator=(PsnrHvsPipeline&&) noexcept;
    PsnrHvsPipeline(const PsnrHvsPipeline&) = delete;
    PsnrHvsPipeline& operator=(const PsnrHvsPipeline&) = delete;

    [[nodiscard]] PsnrHvsScore Compute(
        const Image& original, const Image& distorted
    );
    // Host data is copied before Submit returns and may then be reused.
    [[nodiscard]] PsnrHvsScore Compute(
        const ImageView& original, const ImageView& distorted
    );
    [[nodiscard]] UploadedImage Upload(const Image& image);
    // Unknown or already released handles are ignored while idle.
    void Release(UploadedImage image);
    [[nodiscard]] PsnrHvsScore Compute(
        const UploadedImage& original, const UploadedImage& distorted
    );
    [[nodiscard]] PsnrHvsScore Compute(
        const UploadedImage& original, const Image& distorted
    );

    [[nodiscard]] std::uint64_t Submit(
        const Image& original, const Image& distorted
    );
    [[nodiscard]] std::uint64_t Submit(
        const ImageView& original, const ImageView& distorted
    );
    [[nodiscard]] std::uint64_t Submit(
        const UploadedImage& original, const UploadedImage& distorted
    );
    [[nodiscard]] std::uint64_t Submit(
        const UploadedImage& original, const Image& distorted
    );

    // Allocations have exactly RequiredBytes(format, config.size) bytes.
    // Mapping requires a host-visible arena. Finish before modifying pixels
    // used by a pending submission, including through a retained mapping.
    [[nodiscard]] ImageBuffer AllocateImage(const ImageFormat& format);
    [[nodiscard]] std::byte* MapImage(ImageBuffer buffer);
    void Release(ImageBuffer buffer);
    [[nodiscard]] PsnrHvsScore Compute(
        const ImageBuffer& original, const ImageBuffer& distorted
    );
    [[nodiscard]] std::uint64_t Submit(
        const ImageBuffer& original, const ImageBuffer& distorted
    );

    [[nodiscard]] PsnrHvsScore Compute(
        const VulkanBufferView& original, const VulkanBufferView& distorted
    );
    [[nodiscard]] std::uint64_t Submit(
        const VulkanBufferView& original, const VulkanBufferView& distorted
    );
    [[nodiscard]] std::uint64_t Submit(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        const VulkanSubmitSync& sync
    );
    [[nodiscard]] PsnrHvsScore Finish(std::uint64_t token);
    [[nodiscard]] PipelineMemoryUsage GetMemoryUsage() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vqm
