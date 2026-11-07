// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// Butteraugli portions: Copyright (c) the JPEG XL Project Authors.
// BSD-3-Clause; see licenses/JPEG-XL-LICENSE.

// Butteraugli on the same Vulkan device as Ssimulacra2Pipeline. The metric is
// libjxl's ButteraugliComparator (BSD-3-Clause);
// src/metrics/butteraugli/tables.cc and the shaders carry the source line of
// every constant.
//
// The score is the diffmap's maximum (ButteraugliScoreFromDiffmap), so 0.0
// means the two images are identical on this metric and larger is worse; the
// conventional reading is that values below ~1.0 are visually lossless.
// distance_p3 is libjxl's aggregate (ComputeDistanceP with p = 3), the number
// cjxl reports beside the score.
//
// Precision: fp32 arithmetic on the device with double-precision constants
// (kernel taps, Malta weights) computed on the host. Gradient and spatial
// tests compare score and distance_p3 against our independent double-precision
// CPU oracle at 0.1%. File-loader ICC differences from libjxl are separate;
// see docs/butteraugli.md. Identity scores exactly 0.0. There
// is no half-plane mode: Butteraugli's planes feed differences, and rounding
// them would move the metric rather than trade precision for speed.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "vqm_image.h"
#include "vqm_memory_types.h"

namespace vqm {

struct ButteraugliConfig {
    Size size;
    // How many images may be resident in device memory at once (see
    // Ssimulacra2Config::gpu_resident_images). 0 disables Upload().
    std::size_t gpu_resident_images = 2;
    // Kernel for upsampling subsampled chroma planes during the decode.
    ChromaFilter chroma_filter = ChromaFilter::Bilinear;
    // Bytes reserved for caller-owned image buffers (AllocateImage).
    std::size_t input_bytes = 0;
    // Whether Compute(Image, Image) will be used; see
    // Ssimulacra2Config::streaming_inputs.
    bool streaming_inputs = true;
    // libjxl's ButteraugliParams (butteraugli.h:34-44). The defaults are
    // libjxl's: hf_asymmetry and xmul neutral, intensity_target 80 nits for a
    // 1.0 input value. A different target rescales the whole sensitivity
    // curve; Vship uses 203 for HDR-tuned comparisons. hf_asymmetry and
    // intensity_target must be finite and positive; xmul must be finite and
    // nonnegative. Invalid values throw std::invalid_argument.
    float hf_asymmetry = 1.0f;
    float xmul = 1.0f;
    float intensity_target = 80.0f;
    MemoryPolicy memory_policy = MemoryPolicy::Automatic;
};

struct ButteraugliScore {
    // The diffmap maximum. 0.0 for identical images.
    double score = 0.0;
    // ComputeDistanceP(p = 3) over the same diffmap.
    double distance_p3 = 0.0;
};

// Reusable Butteraugli instance for one geometry. Same threading and handle
// contracts as Ssimulacra2Pipeline: not thread-safe, one submission in flight,
// Finish throws for an unknown token, and a second Submit/Upload/AllocateImage/
// MapImage/Release while one is in flight throws std::logic_error.
//
// For two levels, working planes occupy approximately 136 B per pixel (about
// 1.13 GB at 4K), plus tables and configured input storage: 32 B per pixel for
// streaming_inputs, 16 B per pixel per resident slot, and input_bytes. Staging
// may require additional memory. Query GetMemoryRequirements in vqm_memory.h
// for device-specific allocation sizes and GetMemoryUsage for actual usage.
// The reference image's separated bands are cached for an uploaded reference
// at the same background, so scoring several distorted images against it does
// not repeat that work. Caller-owned buffers may be modified through retained
// mappings, and Compute(Image, Image) re-uploads pixels; both rebuild it.
class ButteraugliPipeline {
public:
    explicit ButteraugliPipeline(ButteraugliConfig config);
    // Borrows the caller's Vulkan device and queue; see vqm_vulkan.h
    // for lifetime, synchronization and image buffer requirements.
    ButteraugliPipeline(ButteraugliConfig config, const VulkanDevice& device);
    ~ButteraugliPipeline();
    ButteraugliPipeline(ButteraugliPipeline&&) noexcept;
    ButteraugliPipeline& operator=(ButteraugliPipeline&&) noexcept;
    ButteraugliPipeline(const ButteraugliPipeline&) = delete;
    ButteraugliPipeline& operator=(const ButteraugliPipeline&) = delete;

    [[nodiscard]] ButteraugliScore Compute(
        const Image& original, const Image& distorted, float background = 0.5f
    );
    // Host plane views are copied into the upload region during Submit.
    [[nodiscard]] ButteraugliScore Compute(
        const ImageView& original, const ImageView& distorted,
        float background = 0.5f
    );

    [[nodiscard]] UploadedImage Upload(const Image& image);
    void Release(UploadedImage image);
    [[nodiscard]] ButteraugliScore Compute(
        const UploadedImage& original, const UploadedImage& distorted,
        float background = 0.5f
    );
    // Reuse an uploaded original while streaming each distorted image. The
    // pipeline must have streaming_inputs enabled.
    [[nodiscard]] ButteraugliScore Compute(
        const UploadedImage& original, const Image& distorted,
        float background = 0.5f
    );

    [[nodiscard]] std::uint64_t Submit(
        const Image& original, const Image& distorted, float background = 0.5f
    );
    [[nodiscard]] std::uint64_t Submit(
        const ImageView& original, const ImageView& distorted,
        float background = 0.5f
    );
    [[nodiscard]] std::uint64_t Submit(
        const UploadedImage& original, const UploadedImage& distorted,
        float background = 0.5f
    );
    [[nodiscard]] std::uint64_t Submit(
        const UploadedImage& original, const Image& distorted,
        float background = 0.5f
    );

    [[nodiscard]] ImageBuffer AllocateImage(const ImageFormat& format);
    [[nodiscard]] std::byte* MapImage(ImageBuffer buffer);
    void Release(ImageBuffer buffer);
    [[nodiscard]] ButteraugliScore Compute(
        const ImageBuffer& original, const ImageBuffer& distorted,
        float background = 0.5f
    );
    [[nodiscard]] std::uint64_t Submit(
        const ImageBuffer& original, const ImageBuffer& distorted,
        float background = 0.5f
    );

    // Direct GPU buffer input on a borrowed device. Externally writable
    // references are rebuilt on every call rather than cached.
    [[nodiscard]] ButteraugliScore Compute(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        float background = 0.5f
    );
    [[nodiscard]] std::uint64_t Submit(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        float background = 0.5f
    );
    [[nodiscard]] std::uint64_t Submit(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        const VulkanSubmitSync& sync, float background = 0.5f
    );

    [[nodiscard]] ButteraugliScore Finish(std::uint64_t token);
    [[nodiscard]] PipelineMemoryUsage GetMemoryUsage() const;

private:
    friend struct internal::MetricScratchAccess;
    ButteraugliPipeline(
        ButteraugliConfig config, const VulkanDevice* device,
        std::shared_ptr<internal::SharedMetricStorage> scratch
    );
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vqm
