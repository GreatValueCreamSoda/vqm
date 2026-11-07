// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vulkan/vulkan.h>

#include "vqm_image.h"

namespace vqm {
struct VulkanDevice;
}

namespace vqm::internal {

// Vulkan device and single compute queue. The default context owns its
// instance and device; an interop context borrows the caller's handles.
struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    VkPhysicalDeviceLimits limits{};
    VkPhysicalDeviceMemoryProperties memory_properties{};
    bool timestamps = false;     // timestampValidBits > 0 on that family
    float timestamp_period = 0;  // nanoseconds per tick
    std::uint32_t timestamp_valid_bits = 0;
    // storageBuffer16BitAccess, enabled at device creation; what
    // Ssimulacra2Config::half_planes needs.
    bool sixteen_bit_storage = false;
    // shaderFloat16 is separate from 16-bit storage and must be enabled for
    // half-float values in shader function/workgroup memory.
    bool shader_float16 = false;
    bool owns_device = true;
    // Every pipeline shares `queue` (one device, one queue, one submission in
    // flight per pipeline), and VkQueue requires external synchronization, so
    // two pipelines computing on two threads must serialize the submit itself.
    // The lock covers vkQueueSubmit only: recording, the fence reset and the
    // fence wait are per runtime, and the second pipeline's work still overlaps
    // the first one's on the device. Measured: three streaming pipelines on
    // three threads without it lost the device (vkWaitForFences -4).
    std::mutex queue_mutex;

    ~VulkanContext();
};

// Shared singleton; nullptr when Vulkan is not usable. Allocation failures may
// throw; callers must not classify every construction error as missing hardware.
std::shared_ptr<VulkanContext> AcquireVulkanContext();

// Timestamp counters wrap at the queue family's valid bit width, which may
// be smaller than the 64-bit host query result.
[[nodiscard]] constexpr std::uint64_t ElapsedTimestampTicks(
    std::uint64_t begin, std::uint64_t end, std::uint32_t valid_bits
) noexcept {
    const std::uint64_t mask =
        valid_bits >= 64 ? UINT64_MAX : (std::uint64_t{1} << valid_bits) - 1;
    return (end - begin) & mask;
}

// Terminal executable cleanup, before driver/layer process-exit handlers.
// All pipelines and worker threads must be gone; acquiring a default context
// afterward is unsupported. Throws if another shared context owner remains.
void ShutdownDefaultVulkanContext();

// Wraps a caller-owned device and compute queue. The caller keeps all handles
// alive and synchronizes its own queue operations with the pipeline's. Vulkan
// cannot verify that the non-null handles belong together or which features
// were enabled on a logical device; the caller must provide accurate values.
// Throws std::invalid_argument for null handles or an unsupported queue/device.
std::shared_ptr<VulkanContext> BorrowVulkanContext(const VulkanDevice& device);

// Select the shared default context or wrap the supplied borrowed device.
// The nullable form preserves probing/factory behavior when Vulkan is absent;
// the required form reports the common construction error instead.
std::shared_ptr<VulkanContext> AcquireVulkanContext(
    const VulkanDevice* borrowed_device
);
std::shared_ptr<VulkanContext> RequireVulkanContext(
    const VulkanDevice* borrowed_device
);

}  // namespace vqm::internal

// The per-image constant block the prepare shader reads through binding 2. It
// holds everything that depends only on the image's format: where its planes
// are, how to decode one sample, and the matrices and scales of its colour
// conversion. It is derived on the host once per image and applied per sample
// on the GPU, so no pixel data is converted on the CPU.

namespace vqm::internal {

// std430 layout, 44 words. Padded to whole words by construction.
struct SourceDesc {
    std::uint32_t plane_word[4];   //  0  arena word offset of each plane
    std::uint32_t enums0;          //  4  see BuildSourceDesc
    std::uint32_t enums1;          //  5  bit 0 identity1, bit 1 identity2
    float scale[3];                //  6  luma, chroma, alpha
    float offset[3];               //  9  luma, chroma, alpha
    float matrix1[9];              // 12  encoded->RGB, or ICtCp->LMS
    float matrix2[9];              // 21  source primaries -> linear Rec.709
    float kr, kg, kb;              // 30  CL op and HLG luminance OOTF weights
    float to_linear_scale;         // 33  PQ/HLG peak-luminance factor
    float shift_x, shift_y;        // 34  chroma siting, chroma-pixel units
    float cl_nb, cl_pb;            // 36  constant-luminance corner values
    float cl_nr, cl_pr;            // 38
    std::uint32_t row_samples[4];  // 40  per-plane row stride in samples
};
inline constexpr std::size_t kSourceDescWords = 44;
static_assert(sizeof(SourceDesc) == kSourceDescWords * sizeof(std::uint32_t));

// `plane_words` are the arena word offsets the planes start at; the builder
// derives everything else from `format` and `chroma_filter`.
[[nodiscard]] SourceDesc BuildSourceDesc(
    const ImageFormat& format, const std::array<std::uint32_t, 4>& plane_words,
    ChromaFilter chroma_filter
);

}  // namespace vqm::internal

// The compile-time classes of src/metrics/ssimulacra2/shaders/prepare.comp. The
// shader is one source file built once per class, so every input path runs only
// its own code: a 4:4:4 kernel carries no chroma filter, a linear-transfer
// kernel no EOTF, and a bilinear kernel none of the six-tap sinc code.
//
//   chroma    0 = 4:4:4 or GREY, 1 = subsampled (the chroma filter)
//   transfer  0 = descriptor chain, 1 = sRGB, 2 = linear
//   filter    0 = bilinear, 1 = bicubic, 2 = lanczos (chroma 1 only)

namespace vqm::internal {

struct SourceDecodeVariant {
    std::uint8_t chroma = 0;
    std::uint8_t transfer = 0;
    std::uint8_t filter = 0;
    [[nodiscard]] constexpr bool operator==(
        const SourceDecodeVariant&
    ) const noexcept = default;
};

// Every variant the build produces: 3 for the 4:4:4 class and 9 for the
// subsampled one. Order and membership are shared with CMakeLists.txt.
inline constexpr std::array<SourceDecodeVariant, 12> kSourceDecodeVariants{{
    {0, 0, 0},
    {0, 1, 0},
    {0, 2, 0},
    {1, 0, 0},
    {1, 0, 1},
    {1, 0, 2},
    {1, 1, 0},
    {1, 1, 1},
    {1, 1, 2},
    {1, 2, 0},
    {1, 2, 1},
    {1, 2, 2},
}};

// The class `format` needs, given the pipeline's chroma filter:
//   - chroma follows the subsampling;
//   - transfer is the identity (2) for linear, the sRGB curve (1) for sRGB,
//     and the descriptor chain (0) otherwise. The constant-luminance and
//     ICtCp ops produce or consume linear values in ways the pinned classes do
//     not model, so those always take the generic kernel;
//   - filter only exists for the subsampled class.
[[nodiscard]] SourceDecodeVariant SelectSourceDecodeVariant(
    const ImageFormat& format, ChromaFilter chroma_filter
);

// Index of `variant` in kSourceDecodeVariants, which is its pipeline slot.
[[nodiscard]] std::size_t SourceDecodeVariantIndex(SourceDecodeVariant variant);

}  // namespace vqm::internal
