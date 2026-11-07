#include "vulkan_runtime.h"

#include <limits>
#include <stdexcept>

#include "vulkan_shared_storage.h"

namespace vqm::internal {

void VulkanRuntime::InitializeSharedState() {
    if (!shared_) return;
    RequireIdle();
    if (!config_.resident_slots.empty() || config_.input_pool_size != 0)
        throw std::invalid_argument(
            "shared scratch cannot retain input handles"
        );
    for (const auto& range : config_.persistent_ranges) {
        if (range.bytes == 0 || range.offset_bytes % 4 || range.bytes % 4 ||
            range.offset_bytes > arena_bytes_ ||
            range.bytes > arena_bytes_ - range.offset_bytes ||
            range.bytes >
                std::numeric_limits<VkDeviceSize>::max() - persistent_bytes_)
            throw std::invalid_argument("invalid persistent scratch range");
        persistent_bytes_ += range.bytes;
    }
    if (persistent_bytes_ == 0) return;
    persistent_ = CreateAllocation(
        persistent_bytes_,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    ResetCommandPool();
    BeginCommandBuffer();
    SaveSharedState(true);
    EndCommandBuffer();
    SubmitAndWait();
    persistent_ready_ = true;
}

void VulkanRuntime::RestoreSharedState() {
    if (!persistent_ready_ || persistent_planes_) return;
    Barrier(
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT
    );
    VkDeviceSize offset = 0;
    for (const auto& range : config_.persistent_ranges) {
        const VkBufferCopy copy{offset, range.offset_bytes, range.bytes};
        vkCmdCopyBuffer(
            command_buffer_, persistent_.buffer, arena_.buffer, 1, &copy
        );
        offset += range.bytes;
    }
    Barrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT
    );
}

void VulkanRuntime::SaveSharedState(bool initialize) {
    if (persistent_planes_ || (!initialize && !persistent_ready_)) return;
    Barrier(
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
    );
    VkDeviceSize offset = 0;
    for (const auto& range : config_.persistent_ranges) {
        if (initialize || !range.immutable) {
            const VkBufferCopy copy{range.offset_bytes, offset, range.bytes};
            vkCmdCopyBuffer(
                command_buffer_, arena_.buffer, persistent_.buffer, 1, &copy
            );
        }
        offset += range.bytes;
    }
}

std::uint32_t VulkanRuntime::BindPersistentPlanes() {
    if (!shared_) return 0;
    RequireIdle();
    if (!persistent_ready_ || config_.persistent_ranges.size() != 1 ||
        config_.persistent_ranges.front().immutable)
        throw std::logic_error(
            "vulkan: private planes require one mutable persistent range"
        );
    const VkDescriptorBufferInfo buffer{persistent_.buffer, 0,
                                        persistent_bytes_};
    std::array<VkWriteDescriptorSet, 2> writes{};
    for (std::size_t index = 0; index < descriptor_sets_.size(); ++index) {
        auto& write = writes[index];
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptor_sets_[index];
        write.dstBinding = 1;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer;
    }
    vkUpdateDescriptorSets(
        ctx_->device, static_cast<std::uint32_t>(writes.size()), writes.data(),
        0, nullptr
    );
    persistent_planes_ = true;
    return static_cast<std::uint32_t>(
        config_.persistent_ranges.front().offset_bytes / sizeof(float)
    );
}

}  // namespace vqm::internal
