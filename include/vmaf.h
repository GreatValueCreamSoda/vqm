// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// VMAF: Copyright (c) Netflix, Inc. BSD-2-Clause-Patent;
// see licenses/NETFLIX-VMAF-LICENSE.

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

#include "vqm_image.h"
#include "vqm_memory_types.h"

namespace vqm {

// Native preserves supported coded YUV input subsampling; other input formats
// use 4:2:0. Explicit options convert both inputs to BT.709 limited-range YUV
// at the selected subsampling. VMAF v0 uses luma only.
enum class VmafChromaSampling : std::uint8_t { Native, Yuv420, Yuv444 };

// The two shipped v0.6.1 predictor tables. Standard1080 is the 211-support-
// vector 1080p/3H model; Standard4K is the 262-support-vector 4KTV/1.5H model.
// NEG and the phone transform are feature/score options, not separate tables.
enum class VmafModel : std::uint8_t { Standard1080, Standard4K };

// Native keeps the input geometry, as VMAF has always done here. FitAndPad
// scales each input uniformly to fit inside model_size, centers it, and fills
// the remaining canvas with the source's black code. VMAF v0 reads luma only,
// so the fit and padding are applied to luma.
enum class VmafFitMode : std::uint8_t { Native, FitAndPad };

// The float VMAF v0.6.1 model. Supports SDR planar inputs accepted by VQM.
// Native coded integer YUV/grey is used directly; other inputs and explicit
// chroma modes are converted to 16-bit BT.709 limited-range coded values.
// Native sequences retain the coding chosen on their first frame; use an
// explicit chroma mode when input formats may change during a sequence.
struct VmafConfig {
    Size size;  // source geometry; the feature canvas in Native mode
    VmafModel model = VmafModel::Standard1080;
    VmafFitMode fit = VmafFitMode::Native;
    // FitAndPad canvas override. Zero uses the selected model's native canvas.
    Size model_size = {};
    std::size_t gpu_resident_images = 2;
    bool streaming_inputs = true;
    // NEG disables enhancement gain in ADM2 and every VIF scale. The 4K NEG
    // mode is derived because Netflix ships no float 4K NEG JSON.
    bool neg = false;
    // The 1080 phone score transform. Applied to 4K as a derived extension;
    // Netflix's 4K model does not declare a transform. It can saturate 4K at
    // 100.
    bool transform = false;
    // Include motion2 in the v0.6.1 predictor and compute the reference motion
    // pyramid. Disabling this is a derived VQM extension: motion2 is forced to
    // zero and frames are emitted immediately without temporal look-ahead.
    bool temporal_weighting = true;
    bool clip = true;
    MemoryPolicy memory_policy = MemoryPolicy::Automatic;
    VmafChromaSampling chroma_sampling = VmafChromaSampling::Native;
    float background = 0.5f;  // encoded RGB background for alpha compositing
};

struct VmafFrame {
    std::uint64_t index = 0;
    double adm2 = 0.0;
    double motion2 = 0.0;
    std::array<double, 4> vif{};
    double score = 0.0;
};

// Ordered, single-threaded video sequence. With temporal weighting, Push scores
// frame 0 immediately and later frames once the next reference frame is
// available; Flush emits the final pending frame. With temporal weighting
// disabled, every Push emits its own frame immediately and Flush emits none.
// Call Reset before starting another sequence. Each Push waits for GPU
// completion, including borrowed buffers.
// PooledScore returns the arithmetic mean after Flush (nullopt for no frames).
class VmafPipeline {
public:
    explicit VmafPipeline(VmafConfig config);
    VmafPipeline(VmafConfig config, const VulkanDevice& device);
    ~VmafPipeline();
    VmafPipeline(VmafPipeline&&) noexcept;
    VmafPipeline& operator=(VmafPipeline&&) noexcept;
    VmafPipeline(const VmafPipeline&) = delete;
    VmafPipeline& operator=(const VmafPipeline&) = delete;

    [[nodiscard]] UploadedImage Upload(const Image& image);
    void Release(UploadedImage image);
    [[nodiscard]] std::optional<VmafFrame> Push(
        const Image& original, const Image& distorted
    );
    [[nodiscard]] std::optional<VmafFrame> Push(
        const ImageView& original, const ImageView& distorted
    );
    [[nodiscard]] std::optional<VmafFrame> Push(
        const UploadedImage& original, const UploadedImage& distorted
    );
    [[nodiscard]] std::optional<VmafFrame> Push(
        const VulkanBufferView& original, const VulkanBufferView& distorted
    );
    [[nodiscard]] std::optional<VmafFrame> Push(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        const VulkanSubmitSync& sync
    );
    [[nodiscard]] std::optional<VmafFrame> Flush();
    [[nodiscard]] std::optional<double> PooledScore() const;
    [[nodiscard]] PipelineMemoryUsage GetMemoryUsage() const;
    void Reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vqm
