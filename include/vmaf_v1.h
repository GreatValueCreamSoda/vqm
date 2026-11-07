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
// Pinned SDR v1.0.16 viewing presets. Standard4K is 2160p at 1.5H;
// Tv4K is 2160p at 3H and has a 110-point score ceiling.
// Hdr4K is the pinned HDR v1.1.8 2160p model at 1.5H.
enum class VmafV1Model : std::uint8_t {
    Standard1080,
    Standard4K,
    Tv4K,
    Phone,
    Hdr4K
};

// SDR Native mode requires the configured input format throughout the
// sequence and reads opaque integer YUV directly.
// Other SDR Native inputs convert to 16-bit limited-range BT.709. Explicit 4:2:0
// or 4:4:4 modes convert both images, even matching YUV, so input formats may
// differ and change across the sequence. Hdr4K instead accepts PQ BT.2020 YUV
// (BT.2020 non-constant luminance or ICtCp) on either side and prepares
// limited-range 16-bit ICtCp 4:2:0 at a 2160-pixel short side. It requires
// Native fit and chroma settings; the source formats may differ.
struct VmafV1Config {
    Size size;  // source geometry
    // Required source format for SDR Native mode.
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
    // Scaling uses the selected model canvas. Padding uses its 16:9 aspect
    // ratio.
    VmafFitMode fit = VmafFitMode::Native;
    VmafV1Model model = VmafV1Model::Standard1080;
    bool hfr = false;  // Pinned five-frame motion model for about 50/60 fps.
};
struct VmafV1Frame {
    std::uint64_t index = 0;
    double cambi = 0;
    double speed_chroma_uv = 0;
    double adm3 = 0;
    double motion3 = 0;
    double score = 0;
};
// Push consumes one ordered frame pair and waits for its GPU work. Results lag
// by one frame, or two with hfr. Flush closes the input and emits one remaining
// result per call. PooledScore is the mean after draining. Reset starts a new
// sequence.
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
    friend struct internal::MetricScratchAccess;
    VmafV1Pipeline(
        VmafV1Config config, const VulkanDevice* device,
        std::shared_ptr<internal::SharedMetricStorage> scratch
    );
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace vqm
