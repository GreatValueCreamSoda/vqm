// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vulkan/vulkan.h>

#include "vqm_vulkan.h"
#include "vulkan/vulkan_support.h"

namespace inmem {

// Benchmark-owned device-local inputs. Pixels are staged and copied exactly
// once before the timing loop; the metric then reads these external buffers
// directly on the same device and queue as its caller.
class VulkanInputs {
public:
    VulkanInputs(const vqm::Image& original, const vqm::Image& distorted)
        : context_(vqm::internal::AcquireVulkanContext()) {
        if (!context_)
            throw std::runtime_error("vulkan: no usable compute device");
        try {
            Create(original, distorted);
        } catch (...) {
            Destroy();
            throw;
        }
    }
    ~VulkanInputs() { Destroy(); }
    VulkanInputs(const VulkanInputs&) = delete;
    VulkanInputs& operator=(const VulkanInputs&) = delete;

    [[nodiscard]] vqm::VulkanDevice device() const {
        return {context_->physical_device, context_->device, context_->queue,
                context_->queue_family, context_->sixteen_bit_storage};
    }
    [[nodiscard]] const vqm::VulkanBufferView& original() const {
        return views_[0];
    }
    [[nodiscard]] const vqm::VulkanBufferView& distorted() const {
        return views_[1];
    }

private:
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
    };

    static void Check(VkResult result, const char* operation) {
        if (result != VK_SUCCESS)
            throw std::runtime_error(
                std::string(operation) +
                " failed: " + std::to_string(static_cast<int>(result))
            );
    }

    [[nodiscard]] std::uint32_t FindMemoryType(
        std::uint32_t bits, VkMemoryPropertyFlags required
    ) const {
        for (std::uint32_t i = 0;
             i < context_->memory_properties.memoryTypeCount; ++i) {
            if ((bits & (1u << i)) != 0 &&
                (context_->memory_properties.memoryTypes[i].propertyFlags &
                 required) == required)
                return i;
        }
        throw std::runtime_error("vulkan: no compatible benchmark memory type");
    }

    void Allocate(
        Buffer& result, VkDeviceSize bytes, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties
    ) {
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = bytes;
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(
            vkCreateBuffer(
                context_->device, &buffer_info, nullptr, &result.buffer
            ),
            "vkCreateBuffer"
        );
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(
            context_->device, result.buffer, &requirements
        );
        VkMemoryAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex =
            FindMemoryType(requirements.memoryTypeBits, properties);
        Check(
            vkAllocateMemory(
                context_->device, &allocate, nullptr, &result.memory
            ),
            "vkAllocateMemory"
        );
        Check(
            vkBindBufferMemory(
                context_->device, result.buffer, result.memory, 0
            ),
            "vkBindBufferMemory"
        );
    }

    void Create(const vqm::Image& original, const vqm::Image& distorted) {
        if (original.size != distorted.size || original.data.empty() ||
            distorted.data.empty())
            throw std::invalid_argument(
                "external images must share a nonempty geometry"
            );
        const std::array<const vqm::Image*, 2> images{&original, &distorted};
        const VkDeviceSize first = original.data.size();
        const VkDeviceSize second = distorted.data.size();
        if (second > std::numeric_limits<VkDeviceSize>::max() - first)
            throw std::overflow_error("external image staging size overflows");
        for (std::size_t i = 0; i < images.size(); ++i) {
            const auto& image = *images[i];
            Allocate(
                inputs_[i], image.data.size(),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            );
            views_[i].buffer = inputs_[i].buffer;
            views_[i].bytes = image.data.size();
            views_[i].size = image.size;
            views_[i].format = image.format;
        }
        Allocate(
            staging_, first + second, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        );
        void* mapped = nullptr;
        Check(
            vkMapMemory(
                context_->device, staging_.memory, 0, VK_WHOLE_SIZE, 0, &mapped
            ),
            "vkMapMemory"
        );
        std::memcpy(mapped, original.data.data(), original.data.size());
        std::memcpy(
            static_cast<std::byte*>(mapped) + first, distorted.data.data(),
            distorted.data.size()
        );
        VkMappedMemoryRange flush{};
        flush.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        flush.memory = staging_.memory;
        flush.size = VK_WHOLE_SIZE;
        const VkResult flush_result =
            vkFlushMappedMemoryRanges(context_->device, 1, &flush);
        vkUnmapMemory(context_->device, staging_.memory);
        Check(flush_result, "vkFlushMappedMemoryRanges");

        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = context_->queue_family;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        Check(
            vkCreateCommandPool(context_->device, &pool_info, nullptr, &pool_),
            "vkCreateCommandPool"
        );
        VkCommandBufferAllocateInfo command_info{};
        command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_info.commandPool = pool_;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        Check(
            vkAllocateCommandBuffers(context_->device, &command_info, &command),
            "vkAllocateCommandBuffers"
        );
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        Check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
        const VkBufferCopy a{0, 0, first};
        const VkBufferCopy b{first, 0, second};
        vkCmdCopyBuffer(command, staging_.buffer, inputs_[0].buffer, 1, &a);
        vkCmdCopyBuffer(command, staging_.buffer, inputs_[1].buffer, 1, &b);
        Check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        Check(
            vkCreateFence(context_->device, &fence_info, nullptr, &fence_),
            "vkCreateFence"
        );
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        {
            std::lock_guard<std::mutex> lock(context_->queue_mutex);
            Check(
                vkQueueSubmit(context_->queue, 1, &submit, fence_),
                "vkQueueSubmit"
            );
        }
        pending_ = true;
        Check(
            vkWaitForFences(context_->device, 1, &fence_, VK_TRUE, UINT64_MAX),
            "vkWaitForFences"
        );
        pending_ = false;
        ReleaseUploadResources();
    }

    void ReleaseUploadResources() noexcept {
        const VkDevice device = context_->device;
        if (fence_ != VK_NULL_HANDLE) vkDestroyFence(device, fence_, nullptr);
        if (pool_ != VK_NULL_HANDLE)
            vkDestroyCommandPool(device, pool_, nullptr);
        if (staging_.buffer != VK_NULL_HANDLE)
            vkDestroyBuffer(device, staging_.buffer, nullptr);
        if (staging_.memory != VK_NULL_HANDLE)
            vkFreeMemory(device, staging_.memory, nullptr);
        fence_ = VK_NULL_HANDLE;
        pool_ = VK_NULL_HANDLE;
        staging_ = {};
    }

    void Destroy() noexcept {
        if (!context_) return;
        const VkDevice device = context_->device;
        if (pending_) (void)vkDeviceWaitIdle(device);
        ReleaseUploadResources();
        for (const Buffer& buffer : inputs_) {
            if (buffer.buffer != VK_NULL_HANDLE)
                vkDestroyBuffer(device, buffer.buffer, nullptr);
            if (buffer.memory != VK_NULL_HANDLE)
                vkFreeMemory(device, buffer.memory, nullptr);
        }
    }

    std::shared_ptr<vqm::internal::VulkanContext> context_;
    std::array<Buffer, 2> inputs_{};
    Buffer staging_{};
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    bool pending_ = false;
    std::array<vqm::VulkanBufferView, 2> views_{};
};

}  // namespace inmem
