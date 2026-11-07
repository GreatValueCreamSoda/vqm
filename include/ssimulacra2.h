// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "vqm_image.h"
#include "vqm_memory_types.h"

namespace vqm {

// The 18 error-map norms of one scale (three channels, two norms, three maps).
struct Ssimulacra2Scale {
    static constexpr std::size_t kSsimNormCount = 6;
    static constexpr std::size_t kEdgeNormCount = 12;
    // [channel * 2 + norm], channel X, Y, B; norm 0 = 1-norm, 1 = 4-norm.
    std::array<double, kSsimNormCount> avg_ssim{};
    // [channel * 4 + n]: ringing n = 0, 1; blurring n = 2, 3.
    std::array<double, kEdgeNormCount> avg_edgediff{};
};

struct Ssimulacra2Result {
    std::vector<Ssimulacra2Scale> scales;
    // Weighted sum of up to 108 norms, mapped to a quality score at most 100.
    // Poor matches may score below zero. Throws std::invalid_argument for more
    // than six scales, nonfinite norms, or overflow in the score mapping.
    [[nodiscard]] double Score() const;
};

struct Ssimulacra2Config {
    Size size;
    // How many images may be resident in device memory at once. A slot costs
    // 16 bytes per pixel: up to four input planes of up to four bytes each.
    // 0 disables Upload(). Default 2 holds one pair.
    std::size_t gpu_resident_images = 2;
    // Stores the intermediate XYB planes as 16-bit floats, which halves the
    // plane traffic of every dispatch at the cost of rounding those planes to
    // ~1e-3 relative. The default keeps planes at full precision; the device
    // must expose storageBuffer16BitAccess or the constructor throws
    // std::runtime_error. Input decoding is unaffected.
    bool half_planes = false;
    // Kernel for upsampling subsampled chroma planes, one choice per pipeline.
    ChromaFilter chroma_filter = ChromaFilter::Bilinear;
    // Bytes of the arena reserved for caller-owned image buffers
    // (AllocateImage). Each allocation takes exactly RequiredBytes(its format)
    // out of this pool, 64-byte aligned, so a caller that fills the memory
    // itself never pays for a worst-case slot; 0 reserves none and
    // AllocateImage throws. This needs device-local memory that is host
    // visible (resizable BAR), which is what Upload already prefers.
    std::size_t input_bytes = 0;
    // Whether Compute(Image, Image) will be used. That path copies decoded
    // pixels into the device and needs a source region of 32 bytes per pixel,
    // plus the same amount again in a staging buffer on a device whose arena is
    // not host-visible. A caller that scores only uploaded images can set this
    // false; the constructor then omits both source regions in the arena.
    // The pixel overloads then throw std::invalid_argument.
    bool streaming_inputs = true;
    MemoryPolicy memory_policy = MemoryPolicy::Automatic;
};

// Reusable metric instance for one image geometry. All working buffers are
// allocated by the constructor and reused by every stage and every Compute
// call. The metric runs on a Vulkan compute device: there is no CPU
// implementation. A Ssimulacra2Pipeline is not thread-safe: use one instance per
// thread.
//
// What one Ssimulacra2Pipeline allocates, per pixel of the configured geometry:
//   32 B  source regions for Compute(Image, Image)   -- only with
//         Ssimulacra2Config::streaming_inputs
//   16 B  per resident slot (Ssimulacra2Config::gpu_resident_images, default 2)
//  ~32 B  the XYB plane pyramid, ~16 B with half_planes
//   plus the norm partials and the level table, which are kilobytes.
// Nothing else is hidden: on a device whose device-local memory is host
// visible (resizable BAR) there is no staging buffer at all. Otherwise it
// holds two source slots (32 B per pixel) for streaming, one slot (16 B per
// pixel) for resident uploads when streaming is disabled, or no source slots
// when neither is configured, plus descriptors and norm readback space.
class Ssimulacra2Pipeline {
public:
    // Throws std::invalid_argument if a dimension is below 8 or if the
    // geometry exceeds the device's storage-buffer range. Throws
    // std::runtime_error when no usable Vulkan device was found or pipeline
    // creation failed.
    explicit Ssimulacra2Pipeline(Ssimulacra2Config config);
    // Uses a caller-owned Vulkan device and queue; see vqm_vulkan.h.
    Ssimulacra2Pipeline(Ssimulacra2Config config, const VulkanDevice& device);
    ~Ssimulacra2Pipeline();
    Ssimulacra2Pipeline(Ssimulacra2Pipeline&&) noexcept;
    Ssimulacra2Pipeline& operator=(Ssimulacra2Pipeline&&) noexcept;
    Ssimulacra2Pipeline(const Ssimulacra2Pipeline&) = delete;
    Ssimulacra2Pipeline& operator=(const Ssimulacra2Pipeline&) = delete;

    // Throws std::invalid_argument if an image dimension differs from the
    // config, if the image format is not supported (see ImageFormat), or if
    // `data` is not exactly RequiredBytes(format, size) long. `background` is
    // the alpha blend target, applied to the encoded RGB values.
    [[nodiscard]] Ssimulacra2Result Compute(
        const Image& original, const Image& distorted, float background = 0.5f
    );

    // Accepts separately allocated, optionally padded host planes. Samples are
    // copied into the existing upload region before Submit returns, so their
    // storage only needs to remain readable until that point. See
    // vqm_views.h for plane and stride requirements.
    [[nodiscard]] Ssimulacra2Result Compute(
        const ImageView& original, const ImageView& distorted,
        float background = 0.5f
    );

    // Copies `image` into device memory (Vulkan). Throws std::invalid_argument
    // when `image` does not match the pipeline geometry or when every resident
    // slot is in use.
    [[nodiscard]] UploadedImage Upload(const Image& image);
    // Returns the slot to the pool. Unknown or already released handles are
    // ignored.
    void Release(UploadedImage image);
    // Scores two resident images. Throws std::invalid_argument for a stale
    // handle or a geometry mismatch.
    [[nodiscard]] Ssimulacra2Result Compute(
        const UploadedImage& original, const UploadedImage& distorted,
        float background = 0.5f
    );
    // Reuse an uploaded original while streaming each distorted image. The
    // pipeline must have streaming_inputs enabled.
    [[nodiscard]] Ssimulacra2Result Compute(
        const UploadedImage& original, const Image& distorted,
        float background = 0.5f
    );

    // Submits a pair and returns at once; Finish(token) waits for it and gives
    // the score, so Compute is Submit followed by Finish. One submission may be
    // in flight per Ssimulacra2Pipeline because its command buffer and image
    // pyramid are shared. Use a separate Ssimulacra2Pipeline per worker to
    // overlap submissions.
    //
    // Finish throws std::invalid_argument for an unknown or already finished
    // token, and a second Submit while one is in flight throws std::logic_error
    // -- call Finish first. Upload throws std::logic_error in the same state,
    // because it rewrites a resident slot the in-flight pair may be reading.
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

    // Caller-owned input memory, carved out of Ssimulacra2Config::input_bytes.
    // AllocateImage throws std::invalid_argument when the pool is zero or full
    // or the format is not decodable, and std::runtime_error when the arena is
    // not host visible (use Upload on such a device).
    //
    // The mapping is persistent and coherent: there is no unmap, no flush, and
    // no per-write bookkeeping. What there is, is ordering. Submit does not
    // block, so the pixels of a buffer must be written before the Submit that
    // scores it, and that buffer must not be touched again until its Finish
    // returns. AllocateImage, MapImage and Release therefore throw
    // std::logic_error while a submission is in flight, the same way Upload
    // does; they exist to catch a caller that would otherwise race the GPU or
    // hand a span to the next allocation while it is still being read.
    [[nodiscard]] ImageBuffer AllocateImage(const ImageFormat& format);
    [[nodiscard]] std::byte* MapImage(ImageBuffer buffer);
    void Release(ImageBuffer buffer);
    [[nodiscard]] Ssimulacra2Result Compute(
        const ImageBuffer& original, const ImageBuffer& distorted,
        float background = 0.5f
    );
    [[nodiscard]] std::uint64_t Submit(
        const ImageBuffer& original, const ImageBuffer& distorted,
        float background = 0.5f
    );

    // Reads caller-owned VkBuffers directly on the borrowed device. See
    // vqm_vulkan.h for layout, lifetime and synchronization rules.
    [[nodiscard]] Ssimulacra2Result Compute(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        float background = 0.5f
    );
    [[nodiscard]] std::uint64_t Submit(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        float background = 0.5f
    );
    // Adds binary semaphore dependencies to an external-buffer submission.
    // See VulkanSubmitSync for stage, lifetime and queue-ownership rules.
    [[nodiscard]] std::uint64_t Submit(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        const VulkanSubmitSync& sync, float background = 0.5f
    );

    [[nodiscard]] Ssimulacra2Result Finish(std::uint64_t token);
    // Actual library-owned buffer allocations retained by this pipeline.
    [[nodiscard]] PipelineMemoryUsage GetMemoryUsage() const;

private:
    friend struct internal::MetricScratchAccess;
    Ssimulacra2Pipeline(
        Ssimulacra2Config config, const VulkanDevice* device,
        std::shared_ptr<internal::SharedMetricStorage> scratch
    );
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vqm
