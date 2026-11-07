// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace vqm {

using DeviceUuid = std::array<std::uint8_t, 16>;

enum class DeviceType { Other, IntegratedGpu, DiscreteGpu, VirtualGpu, Cpu };

struct DeviceInfo {
    DeviceUuid uuid;
    std::string name;
    DeviceType type;
};

// List physical devices meeting VQM's baseline Vulkan compute requirements,
// including software devices. Individual metrics can require additional
// features or limits. Returns an empty list if Vulkan cannot enumerate any
// compatible devices. Enumeration does not initialize the default device.
[[nodiscard]] std::vector<DeviceInfo> ListDevices();

// Select the process-wide device for pipelines, pools, and queries using
// VQM-owned Vulkan resources. Call before the first operation that uses
// the default device. Throws std::logic_error after default initialization.
// The first use reports an error if the UUID is unavailable or unusable.
// Without a selection, VQM prefers a discrete GPU. Borrowed VulkanDevice
// overloads remain available for per-pipeline device selection.
void SetDefaultDevice(const DeviceUuid& uuid);

}  // namespace vqm
