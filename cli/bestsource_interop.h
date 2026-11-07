// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include "vqm_vulkan.h"

struct AVBufferRef;
struct AVFrame;
class BSGpuHasher;

namespace vqm::cli {

class BestSourceDevice {
public:
    BestSourceDevice();
    ~BestSourceDevice();

    [[nodiscard]] const VulkanDevice& vulkan() const noexcept;
    [[nodiscard]] const std::array<std::uint8_t, VK_UUID_SIZE>&
    uuid() const noexcept;

private:
    friend class BestSourceInterop;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The reader owns one reusable planar buffer. BestSource's Vulkan device
// and the scoring device each bind the same external allocation.
class BestSourceInterop {
public:
    BestSourceInterop(
        BestSourceDevice& device, AVBufferRef& decoder_device, Size size,
        const ImageFormat& format
    );
    ~BestSourceInterop();

    [[nodiscard]] VulkanBufferView Pack(
        const AVFrame& frame, BSGpuHasher& exporter
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vqm::cli
