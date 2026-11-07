// ColorVideoVDP: Graphics and Displays Group (2024), MIT.
// VQM implementation: BSD-3-Clause; see LICENSE and licenses/.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "cvvdp.h"
#include "vulkan/memory_plan.h"
#include "vulkan/vulkan_runtime.h"

namespace vqm::internal {

[[nodiscard]] BufferMemoryPlan PlanCvvdpMemory(
    const CvvdpConfig&, const VulkanContext&
);
void ValidateCvvdpDisplay(const CvvdpDisplay& display);
void ValidateCvvdpFormat(const ImageFormat& format);
void ValidateCvvdpInput(const ImageFormat& format, const CvvdpDisplay& display);
[[nodiscard]] std::size_t CvvdpTemporalTaps(const CvvdpConfig& config);
[[nodiscard]] std::vector<float> CvvdpTemporalFilters(const CvvdpConfig& config);
// Original tensor-exponent norm semantics, including the epsilon correction.
[[nodiscard]] double CvvdpSafePow(double value, double exponent);
[[nodiscard]] double CvvdpFrameDistortion(
    std::span<const float> bands, std::size_t channels
);
[[nodiscard]] double CvvdpJod(double distortion);
[[nodiscard]] double CvvdpImageJod(std::span<const float> bands);

class CvvdpPrepare {
public:
    CvvdpPrepare(VulkanRuntime&, const CvvdpDisplay&, ChromaFilter);
    ~CvvdpPrepare();
    void Record(
        std::size_t image, std::uint32_t descriptor, std::uint32_t output,
        Size size, const ImageFormat& format, std::uint32_t status
    );

private:
    VulkanRuntime& runtime_;
    CvvdpDisplay display_;
    ChromaFilter filter_;
    std::array<VkPipeline, 3> pipelines_{};
};

}  // namespace vqm::internal
