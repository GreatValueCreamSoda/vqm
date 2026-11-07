// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include "bestsource_interop.h"
#include "io.h"

namespace vqm::cli {

// Probe by content and select the first video track that is not cover art.
[[nodiscard]] std::optional<int> FindBestSourceTrack(const std::string& path);

class BestSourceReader {
public:
    BestSourceReader(
        const std::string& path, int track, BestSourceDevice& device,
        std::int64_t start, std::optional<std::int64_t> count
    );
    ~BestSourceReader();

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
