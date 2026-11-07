// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// VMAF: Copyright (c) Netflix, Inc. BSD-2-Clause-Patent;
// see licenses/NETFLIX-VMAF-LICENSE.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include "vmaf.h"
#include "vqm_image.h"
#include "vqm_memory_types.h"
namespace vqm {
// Pinned standard SDR vmaf_v1.0.16_3d0h. Native mode requires the configured
// input format throughout the sequence and reads opaque integer YUV directly;
// other native inputs convert to 16-bit limited-range BT.709. Explicit 4:2:0
// or 4:4:4 modes convert both images, even matching YUV, so input formats may
// differ and change across the sequence. PQ and HLG inputs are not supported.
struct VmafV1Config {
    Size size;  // source geometry
    // Required source format for Native mode; ignored in explicit modes.
    ImageFormat input_format;
    // CAMBI's encoded sample bit depth; 0 uses the native input bit depth or
    // 16 when converting. Values 6..16 are supported. CAMBI uses the feature
    // geometry selected by fit.
    std::uint32_t encoded_bit_depth = 0;
    std::size_t gpu_resident_images = 2;
    bool streaming_inputs = true;
    bool clip = true;
    MemoryPolicy memory_policy = MemoryPolicy::Automatic;
    VmafChromaSampling chroma_sampling = VmafChromaSampling::Native;
    float background = 0.5f;  // encoded RGB background for alpha compositing
    // Scaling uses the standard 1920x1080 model canvas. Padding uses its 16:9
    // aspect ratio.
    VmafFitMode fit = VmafFitMode::Native;
};
struct VmafV1Frame {
    std::uint64_t index = 0;
    double cambi = 0;
    double speed_chroma_uv = 0;
    double adm3 = 0;
    double motion3 = 0;
    double score = 0;
};
// Push consumes one ordered frame pair and waits for its GPU work. Frame zero
// is emitted on the second Push; every subsequent frame needs one-frame
// lookahead. Flush emits the final frame, including a single-frame sequence.
// PooledScore is the mean of scores after Flush. Reset starts a new sequence.
// Borrowed Vulkan buffers must remain alive and unmodified through Push; use
// VulkanSubmitSync to wait on a producer when necessary. The caller owns and
// synchronizes any borrowed Vulkan device and queue, as in vqm_vulkan.h.
class VmafV1Pipeline {
public:
    explicit VmafV1Pipeline(VmafV1Config config);
    VmafV1Pipeline(VmafV1Config config, const VulkanDevice& device);
    ~VmafV1Pipeline();
    VmafV1Pipeline(VmafV1Pipeline&&) noexcept;
    VmafV1Pipeline& operator=(VmafV1Pipeline&&) noexcept;
    VmafV1Pipeline(const VmafV1Pipeline&) = delete;
    VmafV1Pipeline& operator=(const VmafV1Pipeline&) = delete;
    [[nodiscard]] UploadedImage Upload(const Image& image);
    void Release(UploadedImage image);
    [[nodiscard]] std::optional<VmafV1Frame> Push(
        const Image& original, const Image& distorted
    );
    [[nodiscard]] std::optional<VmafV1Frame> Push(
        const ImageView& original, const ImageView& distorted
    );
    [[nodiscard]] std::optional<VmafV1Frame> Push(
        const UploadedImage& original, const UploadedImage& distorted
    );
    [[nodiscard]] std::optional<VmafV1Frame> Push(
        const VulkanBufferView& original, const VulkanBufferView& distorted
    );
    [[nodiscard]] std::optional<VmafV1Frame> Push(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        const VulkanSubmitSync& sync
    );
    [[nodiscard]] std::optional<VmafV1Frame> Flush();
    [[nodiscard]] std::optional<double> PooledScore() const;
    [[nodiscard]] PipelineMemoryUsage GetMemoryUsage() const;
    void Reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace vqm
