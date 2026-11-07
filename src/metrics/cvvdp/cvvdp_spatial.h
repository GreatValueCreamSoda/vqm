// ColorVideoVDP: Copyright (c) 2024, Graphics and Displays group.
// MIT; see licenses/COLORVIDEOVDP-LICENSE.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <vulkan/vulkan.h>

#include "vqm_image.h"

namespace vqm::internal {

class VulkanRuntime;

// The pinned base model's spatial pyramid, CSF and mutual masking. Inputs are
// channel-major FP32 DKL planes: Y, rg, yv, and (for video) transient Y. Three
// channels select the image path; four select already temporally filtered video.
// Record does not upload, submit, read back or pool across bands/channels/time.
class CvvdpSpatial {
public:
    explicit CvvdpSpatial(VulkanRuntime& runtime);
    ~CvvdpSpatial();
    CvvdpSpatial(const CvvdpSpatial&) = delete;
    CvvdpSpatial& operator=(const CvvdpSpatial&) = delete;

    [[nodiscard]] static std::size_t Levels(Size size, double pixels_per_degree);
    [[nodiscard]] static std::size_t ScratchWords(
        Size size, double pixels_per_degree
    );
    [[nodiscard]] static std::size_t TableWords(
        Size size, double pixels_per_degree
    );
    [[nodiscard]] static std::size_t PartialCount(Size size);
    [[nodiscard]] static constexpr std::size_t PushBytes() noexcept {
        return 20 * sizeof(std::uint32_t);
    }

    // Populate tables for this geometry and viewing distance. The first 40
    // words contain model constants; the rest hold one 32-luminance CSF row
    // per band/channel, with frequency already interpolated on the host.
    static void FillTables(
        Size size, double pixels_per_degree, std::span<float> destination
    );

    // Offsets are absolute arena words. Reserve ScratchWords() scratch,
    // TableWords() initialized tables, 4*PartialCount() partials and 4*Levels()
    // result words, all disjoint from the input planes and each other. Output
    // is pixel-pooled distortion, [channel][band], before band/channel weights.
    void Record(
        Size size, double pixels_per_degree, std::uint32_t channels,
        std::uint32_t reference, std::uint32_t distorted, std::uint32_t scratch,
        std::uint32_t tables, std::uint32_t partials, std::uint32_t results
    );

private:
    VulkanRuntime* runtime_;
    std::array<VkPipeline, 4> pipelines_{};
};

}  // namespace vqm::internal
