// ColorVideoVDP model: Copyright (c) 2024, Graphics and Displays Group.
// MIT; see licenses/COLORVIDEOVDP-LICENSE. VQM implementation: BSD-3-Clause.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "vqm_image.h"
#include "vqm_memory_types.h"

namespace vqm {

namespace internal {
struct CvvdpFrameAccess;
}  // namespace internal

// Physical display parameters, independent of the input image dimensions.
// Luminance is cd/m^2, illuminance is lux, reflectivity is diffuse. Angular
// sampling is the display's central pixels per visual degree; no resizing is
// performed. Default values resolve the original standard_4k preset.
struct CvvdpDisplay {
    double peak_luminance = 200.0;
    double contrast = 1000.0;
    double ambient_illuminance = 250.0;
    double reflectivity = 0.005;
    double pixels_per_degree = 75.402449345163049;
    double exposure = 1.0;
    [[nodiscard]] static CvvdpDisplay Standard4K();
    [[nodiscard]] static CvvdpDisplay StandardFhd();
    [[nodiscard]] static CvvdpDisplay StandardHdr();
};

// Original base ColorVideoVDP v0.5.7 calibration; replicated causal temporal
// padding. Opaque RGB, grey and nonconstant-luminance YUV inputs are supported
// with sRGB, BT.709/BT.2020 SDR, absolute linear, PQ or HLG transfer. Linear
// samples are absolute luminance, unlike the relative-linear convention of
// other VQM metrics. Alpha and special CL/ICtCp encodings are rejected.
struct CvvdpConfig {
    Size size;
    CvvdpDisplay display = {};
    std::uint32_t fps_numerator = 30;
    std::uint32_t fps_denominator = 1;
    // Disable temporal storage for independent images; Push then throws.
    bool temporal = true;
    std::size_t gpu_resident_images = 2;
    bool streaming_inputs = true;
    MemoryPolicy memory_policy = MemoryPolicy::Automatic;
    ChromaFilter chroma_filter = ChromaFilter::Bilinear;
};

// Whole-image or whole-clip JOD. Higher is better; identity is exactly 10.
// Negative JOD is valid. This is never an average of frame JOD values.
struct CvvdpScore {
    double score = 10.0;
    std::uint64_t frames = 0;
};

// One geometry/display/frame rate and one ordered clip per pipeline; not
// thread-safe. Push consumes its inputs before returning and retains only
// private converted history. The first pair is deferred until a second Push
// or Finalize chooses video versus the original one-image algorithm.
// Finalize returns nullopt for an empty clip and caches its result on repeated
// calls. Push after Finalize requires Reset. Invalid host input can be retried;
// a processing failure requires Reset. A failed GPU submission requires
// reconstructing the pipeline. Reset preserves uploaded handles.
// ComputeImage is independent of clip state and needs a new/reset pipeline.
class CvvdpPipeline {
public:
    explicit CvvdpPipeline(CvvdpConfig config);
    CvvdpPipeline(CvvdpConfig config, const VulkanDevice& device);
    ~CvvdpPipeline();
    CvvdpPipeline(CvvdpPipeline&&) noexcept;
    CvvdpPipeline& operator=(CvvdpPipeline&&) noexcept;
    CvvdpPipeline(const CvvdpPipeline&) = delete;
    CvvdpPipeline& operator=(const CvvdpPipeline&) = delete;

    [[nodiscard]] UploadedImage Upload(const Image& image);
    void Release(UploadedImage image);
    [[nodiscard]] CvvdpScore ComputeImage(const Image&, const Image&);
    [[nodiscard]] CvvdpScore ComputeImage(const ImageView&, const ImageView&);
    [[nodiscard]] CvvdpScore ComputeImage(
        const UploadedImage&, const UploadedImage&
    );
    [[nodiscard]] CvvdpScore ComputeImage(
        const VulkanBufferView&, const VulkanBufferView&
    );
    [[nodiscard]] CvvdpScore ComputeImage(
        const VulkanBufferView&, const VulkanBufferView&, const VulkanSubmitSync&
    );
    void Push(const Image&, const Image&);
    void Push(const ImageView&, const ImageView&);
    void Push(const UploadedImage&, const UploadedImage&);
    void Push(const VulkanBufferView&, const VulkanBufferView&);
    void Push(
        const VulkanBufferView&, const VulkanBufferView&, const VulkanSubmitSync&
    );
    [[nodiscard]] std::optional<CvvdpScore> Finalize();
    void Reset();
    [[nodiscard]] PipelineMemoryUsage GetMemoryUsage() const;

private:
    friend struct internal::MetricScratchAccess;
    CvvdpPipeline(
        CvvdpConfig config, const VulkanDevice* device,
        std::shared_ptr<internal::SharedMetricStorage> scratch
    );
    friend struct internal::CvvdpFrameAccess;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vqm
