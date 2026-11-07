// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#pragma once

#include <memory>
#include <string>

#include "io.h"
#include "vqm_vulkan.h"

namespace vqm::cli {

// Probe a local file's container without opening a decoder or a GPU device.
[[nodiscard]] bool IsCompressedVideo(const std::string& path);

class NvdecDevice {
public:
    NvdecDevice();
    ~NvdecDevice();

    [[nodiscard]] const VulkanDevice& vulkan() const noexcept;

private:
    friend class NvdecReader;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The returned view remains valid until the next call to Next or destruction.
// Scoring consumes it synchronously before the reader reuses its GPU buffer.
class NvdecReader {
public:
    NvdecReader(const std::string& path, NvdecDevice& device);
    ~NvdecReader();

    [[nodiscard]] Size size() const noexcept;
    [[nodiscard]] const ImageFormat& format() const noexcept;
    [[nodiscard]] image_io::FrameRate frame_rate() const noexcept;
    [[nodiscard]] bool is_video() const noexcept { return true; }
    [[nodiscard]] image_io::SequenceReadResult Next(
        VulkanBufferView& view, std::string& error
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vqm::cli
