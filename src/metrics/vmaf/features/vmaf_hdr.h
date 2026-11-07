// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vulkan/vulkan.h>
#include "vqm_image.h"

namespace vqm::internal {

class VulkanRuntime;

void ValidateVmafHdrFormat(const ImageFormat& format);
[[nodiscard]] Size VmafHdrGeometry(Size source, std::size_t short_side);

struct VmafHdrBuffers {
    std::array<std::uint32_t, 3> working{};
    std::uint32_t source = 0;
    std::uint32_t horizontal = 0;
};

// Prepare limited-range 16-bit PQ ICtCp in the centered FP32 code domain.
// The source, horizontal and three working regions are reused across images.
class VmafHdrConverter {
public:
    explicit VmafHdrConverter(VulkanRuntime& runtime);
    ~VmafHdrConverter();
    VmafHdrConverter(const VmafHdrConverter&) = delete;
    VmafHdrConverter& operator=(const VmafHdrConverter&) = delete;

    [[nodiscard]] static constexpr std::size_t PushBytes() {
        return 9 * sizeof(std::uint32_t);
    }

    void Record(
        std::size_t image, std::uint32_t desc, const ImageFormat& format,
        Size source, Size target, const VmafHdrBuffers& buffers,
        const std::array<std::uint32_t, 3>& output
    );

private:
    void Resize(
        std::uint32_t source, std::uint32_t destination, std::uint32_t scratch,
        Size from, Size to, std::array<float, 2> shift, unsigned plane,
        bool quantize
    );

    VulkanRuntime& runtime_;
    std::array<VkPipeline, 2> pipelines_{};
};

}  // namespace vqm::internal
