// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <array>
#include <cstdint>
#include <span>

#include <vulkan/vulkan.h>

#include "vqm_image.h"

namespace vqm {

// Borrowed Vulkan handles. The caller owns them and must keep the physical
// device, its instance, the device and queue alive until every
// pipeline, pool or workspace using them is destroyed. The
// instance must have been created for Vulkan 1.1 or later, and the physical
// device must support it. The queue must come from a compute-capable family of
// `device`. Set
// storage_buffer_16bit_access only if that feature was enabled when creating
// the logical device; it is required for Ssimulacra2Config::half_planes.
// Set shader_float16 only when shaderFloat16 was enabled (through Vulkan 1.2
// or VK_KHR_shader_float16_int8). VMAF uses an FP32 fallback when it is false.
//
// VkQueue access requires external synchronization. The caller must serialize
// queue submissions by the application and all Pipelines sharing this queue;
// an internal mutex cannot synchronize calls made outside this library.
struct VulkanDevice {
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    bool storage_buffer_16bit_access = false;
    bool shader_float16 = false;
};

// A binary semaphore to wait on before reading external inputs. The mask must
// include COMPUTE_SHADER, TOP_OF_PIPE or ALL_COMMANDS; allowed bits are those
// three and TRANSFER. This restriction is valid for every compute queue and
// ensures the input shader reads cannot run ahead of the wait.
struct VulkanSemaphoreWait {
    VkSemaphore semaphore = VK_NULL_HANDLE;
    VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
};

// Optional dependencies for Submit(VulkanBufferView, VulkanBufferView, sync).
// Empty lists are valid. Semaphores must be binary and belong to the borrowed
// device: Vulkan cannot query their type or device from a handle. Each wait
// must have a previously submitted signal, each signal must be unsignaled when
// the signal operation executes, and handles may not repeat in either list or
// appear in both lists. Keep semaphores alive until every queue operation that
// uses them has completed, including any downstream wait after Finish.
//
// Semaphores do not transfer queue-family ownership; the caller must arrange
// any required release and acquire operations. Queue host synchronization and
// external buffer lifetime rules still apply. Array storage is needed only
// during Submit. Timeline semaphores are not supported by this overload.
struct VulkanSubmitSync {
    std::span<const VulkanSemaphoreWait> waits{};
    std::span<const VkSemaphore> signals{};
};

// A plane's offset is relative to VulkanBufferView::offset. Rows are separated
// by stride_bytes; each plane starts on a four-byte boundary and each row
// stride is a multiple of its sample size. Multiple planes may overlap because
// the metric only reads them.
struct VulkanPlaneLayout {
    VkDeviceSize offset_bytes = 0;
    VkDeviceSize stride_bytes = 0;
};

// Borrowed GPU buffer containing one planar image. By default (`planes` all
// zero), it uses the tightly packed layout documented by Image. To use custom
// offsets or padded rows, set both fields for every present plane; absent
// planes must be zero. `offset` and each plane start must be aligned to four
// bytes; `bytes` is the available length beginning at offset and must cover
// each plane, including padding to the end of its last 32-bit word. The buffer
// must be created with
// VK_BUFFER_USAGE_STORAGE_BUFFER_BIT on the borrowed device. The caller owns
// the buffer and must keep it and its memory alive and unmodified until Finish
// returns or an outstanding submission completes during pipeline destruction.
// The actual buffer bounds and usage cannot be queried from VkBuffer;
// callers must provide accurate values.
//
// Before Submit, GPU writes must be ordered on the supplied queue. When the
// producer uses another queue, pass a VulkanSubmitSync wait or arrange the
// semaphore wait beforehand. The caller must arrange any required queue-family
// ownership transfer. Flush writes to noncoherent host-visible memory before
// submission. The pipeline
// inserts a device memory barrier before its shader reads. The buffer remains
// on the GPU and is read directly; no pixel upload or intermediate copy occurs.
struct VulkanBufferView {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize bytes = 0;
    Size size;
    ImageFormat format;
    std::array<VulkanPlaneLayout, 4> planes{};
};

}  // namespace vqm
