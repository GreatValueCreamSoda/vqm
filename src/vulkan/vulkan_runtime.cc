// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include "vulkan_runtime.h"
#include "vulkan_shared_storage.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "handles.h"
#include "vulkan_shaders.h"

namespace vqm::internal {

void ValidateMemoryPolicy(MemoryPolicy policy) {
    switch (policy) {
        case MemoryPolicy::Automatic:
        case MemoryPolicy::Direct:
        case MemoryPolicy::Staged:
            return;
    }
    throw std::invalid_argument("vulkan: invalid memory policy");
}

void Check(VkResult result, const char* what) {
    if (result == VK_SUCCESS) return;
    throw std::runtime_error(
        std::string("vulkan: ") + what + " failed with VkResult " +
        std::to_string(static_cast<int>(result))
    );
}

VkDeviceSize Bytes(std::size_t floats) {
    return static_cast<VkDeviceSize>(floats) * sizeof(float);
}

VkDeviceSize PlaneBytes(std::uint32_t elements, bool half) {
    return static_cast<VkDeviceSize>(elements) * (half ? 2u : 4u);
}

std::uint32_t Groups(std::size_t extent, std::uint32_t per_group) {
    const std::size_t groups = (extent + per_group - 1) / per_group;
    return static_cast<std::uint32_t>(std::max<std::size_t>(groups, 1));
}

namespace {

const shaders::Blob& FindBlob(std::string_view name) {
    for (const shaders::Blob& blob : shaders::kBlobs) {
        if (name == blob.name) return blob;
    }
    throw std::runtime_error(
        "vulkan: missing embedded shader " + std::string(name)
    );
}

#ifdef VQM_STAGE_TIMING
constexpr std::uint32_t kMaxTimestamps = 256;
#endif

}  // namespace

VulkanRuntime::VulkanRuntime(
    std::shared_ptr<VulkanContext> context, Config config,
    std::shared_ptr<SharedMetricStorage> shared
)
    : ctx_(std::move(context)),
      shared_(std::move(shared)),
      config_(std::move(config)) {
    if (!ctx_) throw std::invalid_argument("vulkan: no device context");
    ValidateMemoryPolicy(config_.memory_policy);
    resident_format_.assign(config_.resident_slots.size(), ImageFormat{});
    resident_free_.assign(config_.resident_slots.size(), true);
    input_pool_ = config_.input_pool;
    input_pool_size_ = config_.input_pool_size;
}

VulkanRuntime::~VulkanRuntime() { Destroy(); }

PipelineMemoryUsage VulkanRuntime::GetMemoryUsage() const {
    PipelineMemoryUsage usage;
    usage.uses_staging = staging_.buffer != VK_NULL_HANDLE;
    usage.arena = MemoryAllocation{arena_bytes_, arena_.allocation_bytes,
                                   arena_.memory_type, arena_.heap_index};
    if (usage.uses_staging) {
        usage.staging =
            MemoryAllocation{staging_bytes_, staging_.allocation_bytes,
                             staging_.memory_type, staging_.heap_index};
    }
    for (std::size_t i = 0; i < auxiliary_.size(); ++i) {
        const auto& allocation = auxiliary_[i];
        usage.auxiliary.push_back(
            {config_.auxiliary_bytes[i], allocation.allocation_bytes,
             allocation.memory_type, allocation.heap_index}
        );
    }
    return usage;
}

void VulkanRuntime::Destroy() noexcept {
    const VkDevice device = ctx_ ? ctx_->device : VK_NULL_HANDLE;
    arena_bytes_ = 0;
    staging_bytes_ = 0;
    arena_visible_ = false;
    pending_ = false;
    if (device == VK_NULL_HANDLE) return;

#ifdef VQM_STAGE_TIMING
    if (queries_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device, queries_, nullptr);
        queries_ = VK_NULL_HANDLE;
    }
#endif
    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
        descriptor_sets_.fill(VK_NULL_HANDLE);
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (descriptor_set_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout_, nullptr);
        descriptor_set_layout_ = VK_NULL_HANDLE;
    }
    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, command_pool_, nullptr);
        command_pool_ = VK_NULL_HANDLE;
        command_buffer_ = VK_NULL_HANDLE;
    }
    if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device, fence_, nullptr);
        fence_ = VK_NULL_HANDLE;
    }
    if (!shared_ && staging_map_ != nullptr) {
        vkUnmapMemory(device, staging_.memory);
        staging_map_ = nullptr;
    }
    if (!shared_ && arena_map_ != nullptr) {
        vkUnmapMemory(device, arena_.memory);
        arena_map_ = nullptr;
    }
    for (auto& allocation : auxiliary_) {
        if (allocation.buffer)
            vkDestroyBuffer(device, allocation.buffer, nullptr);
        if (allocation.memory) vkFreeMemory(device, allocation.memory, nullptr);
    }
    auxiliary_.clear();
    if (!shared_) {
        for (Allocation* allocation : {&staging_, &arena_}) {
            if (allocation->buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, allocation->buffer, nullptr);
                allocation->buffer = VK_NULL_HANDLE;
            }
            if (allocation->memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, allocation->memory, nullptr);
                allocation->memory = VK_NULL_HANDLE;
            }
        }
    }
}

std::uint32_t VulkanRuntime::FindMemoryType(
    std::uint32_t type_bits, VkMemoryPropertyFlags required
) const {
    const std::optional<std::uint32_t> type =
        FindMemoryTypeIfSupported(type_bits, required);
    if (!type) {
        throw std::runtime_error("vulkan: no memory type matches the request");
    }
    return *type;
}

std::optional<std::uint32_t> VulkanRuntime::FindMemoryTypeIfSupported(
    std::uint32_t type_bits, VkMemoryPropertyFlags required
) const {
    const VkPhysicalDeviceMemoryProperties& memory = ctx_->memory_properties;
    for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) == 0) continue;
        if ((memory.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    return std::nullopt;
}

VulkanRuntime::Allocation VulkanRuntime::CreateAllocation(
    VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties
) {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    Allocation allocation;
    Check(
        vkCreateBuffer(ctx_->device, &info, nullptr, &allocation.buffer),
        "vkCreateBuffer"
    );

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(
        ctx_->device, allocation.buffer, &requirements
    );

    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    try {
        allocate.memoryTypeIndex =
            FindMemoryType(requirements.memoryTypeBits, properties);
    } catch (...) {
        vkDestroyBuffer(ctx_->device, allocation.buffer, nullptr);
        throw;
    }

    const VkResult allocated =
        vkAllocateMemory(ctx_->device, &allocate, nullptr, &allocation.memory);
    if (allocated != VK_SUCCESS) {
        vkDestroyBuffer(ctx_->device, allocation.buffer, nullptr);
        Check(allocated, "vkAllocateMemory");
    }
    const VkResult bound = vkBindBufferMemory(
        ctx_->device, allocation.buffer, allocation.memory, 0
    );
    if (bound != VK_SUCCESS) {
        vkFreeMemory(ctx_->device, allocation.memory, nullptr);
        vkDestroyBuffer(ctx_->device, allocation.buffer, nullptr);
        Check(bound, "vkBindBufferMemory");
    }
    allocation.allocation_bytes = requirements.size;
    allocation.memory_type = allocate.memoryTypeIndex;
    allocation.heap_index =
        ctx_->memory_properties.memoryTypes[allocate.memoryTypeIndex].heapIndex;
    return allocation;
}

void VulkanRuntime::CreateArena(VkDeviceSize bytes) {
    if (shared_) {
        VulkanRuntime& owner = shared_->backing();
        if (ctx_.get() != owner.ctx_.get() || bytes > owner.arena_bytes_)
            throw std::invalid_argument(
                "shared arena is too small or on a different device"
            );
        if ((config_.memory_policy == MemoryPolicy::Direct &&
             !owner.arena_visible_) ||
            (config_.memory_policy == MemoryPolicy::Staged &&
             owner.arena_visible_)) {
            throw std::invalid_argument("shared arena memory policy mismatch");
        }
        arena_ = owner.arena_;
        arena_bytes_ = bytes;
        arena_visible_ = owner.arena_visible_;
        arena_map_ = owner.arena_map_;
        return;
    }
    // Prefer device-local memory the host may write directly (resizable BAR):
    // the upload then lands in the arena itself, so neither the staging buffer
    // nor the DMA copy into it is needed. Without that memory type, or when the
    // allocation for it fails, fall back to plain device-local memory and the
    // staging buffer.
    constexpr VkMemoryPropertyFlags kVisible =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = bytes;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    Check(
        vkCreateBuffer(ctx_->device, &info, nullptr, &arena_.buffer),
        "vkCreateBuffer"
    );

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(ctx_->device, arena_.buffer, &requirements);
    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;

    const std::optional<std::uint32_t> visible =
        FindMemoryTypeIfSupported(requirements.memoryTypeBits, kVisible);
    if (config_.memory_policy == MemoryPolicy::Direct && !visible) {
        throw std::runtime_error(
            "vulkan: direct memory policy requires host-visible device-local "
            "memory"
        );
    }
    if (config_.memory_policy != MemoryPolicy::Staged && visible) {
        allocate.memoryTypeIndex = *visible;
        const VkResult result =
            vkAllocateMemory(ctx_->device, &allocate, nullptr, &arena_.memory);
        if (config_.memory_policy == MemoryPolicy::Direct &&
            result != VK_SUCCESS) {
            arena_.memory = VK_NULL_HANDLE;
            Check(result, "vkAllocateMemory (direct arena)");
        }
        arena_visible_ = result == VK_SUCCESS;
    }
    if (!arena_visible_) {
        arena_.memory = VK_NULL_HANDLE;
        allocate.memoryTypeIndex = FindMemoryType(
            requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        Check(
            vkAllocateMemory(ctx_->device, &allocate, nullptr, &arena_.memory),
            "vkAllocateMemory"
        );
    }

    const VkResult bound =
        vkBindBufferMemory(ctx_->device, arena_.buffer, arena_.memory, 0);
    if (bound != VK_SUCCESS) {
        vkFreeMemory(ctx_->device, arena_.memory, nullptr);
        arena_.memory = VK_NULL_HANDLE;
        vkDestroyBuffer(ctx_->device, arena_.buffer, nullptr);
        arena_.buffer = VK_NULL_HANDLE;
        Check(bound, "vkBindBufferMemory");
    }
    arena_.allocation_bytes = requirements.size;
    arena_.memory_type = allocate.memoryTypeIndex;
    arena_.heap_index =
        ctx_->memory_properties.memoryTypes[allocate.memoryTypeIndex].heapIndex;
    arena_bytes_ = bytes;

    if (arena_visible_) {
        void* mapped = nullptr;
        Check(
            vkMapMemory(
                ctx_->device, arena_.memory, 0, VK_WHOLE_SIZE, 0, &mapped
            ),
            "vkMapMemory"
        );
        arena_map_ = static_cast<float*>(mapped);
    }
}

void VulkanRuntime::CreateAuxiliaryStorage() {
    if (!auxiliary_.empty())
        throw std::logic_error("vulkan: auxiliary storage already created");
    auxiliary_.reserve(config_.auxiliary_bytes.size());
    for (const auto bytes : config_.auxiliary_bytes) {
        auxiliary_.push_back(CreateAllocation(
            bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        ));
    }
}

void VulkanRuntime::CreateStaging(std::size_t bytes) {
    if (shared_) {
        VulkanRuntime& owner = shared_->backing();
        if (bytes > owner.staging_bytes_)
            throw std::invalid_argument("shared staging arena is too small");
        staging_ = owner.staging_;
        staging_bytes_ = bytes;
        staging_map_ = owner.staging_map_;
        return;
    }
    staging_ = CreateAllocation(
        bytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    staging_bytes_ = static_cast<VkDeviceSize>(bytes);
    void* mapped = nullptr;
    Check(
        vkMapMemory(ctx_->device, staging_.memory, 0, VK_WHOLE_SIZE, 0, &mapped),
        "vkMapMemory"
    );
    staging_map_ = static_cast<float*>(mapped);
}

void VulkanRuntime::CreateCommandObjects() {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                      VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = ctx_->queue_family;
    Check(
        vkCreateCommandPool(ctx_->device, &pool_info, nullptr, &command_pool_),
        "vkCreateCommandPool"
    );

    VkCommandBufferAllocateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    buffer_info.commandPool = command_pool_;
    buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    buffer_info.commandBufferCount = 1;
    Check(
        vkAllocateCommandBuffers(ctx_->device, &buffer_info, &command_buffer_),
        "vkAllocateCommandBuffers"
    );

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    Check(
        vkCreateFence(ctx_->device, &fence_info, nullptr, &fence_),
        "vkCreateFence"
    );

#ifdef VQM_STAGE_TIMING
    if (ctx_->timestamps) {
        VkQueryPoolCreateInfo query_info{};
        query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_info.queryCount = kMaxTimestamps;
        Check(
            vkCreateQueryPool(ctx_->device, &query_info, nullptr, &queries_),
            "vkCreateQueryPool"
        );
    }
#endif
}

void VulkanRuntime::CreateDescriptors(
    VkDeviceSize plane_base_bytes, VkDeviceSize plane_bytes
) {
    // Binding 0 is the whole arena (f32), binding 1 its plane region, which is
    // the same buffer at an offset so the shaders can address 16-bit planes
    // without a second allocation, and binding 2 is the whole arena again as
    // words, where each image's constant block lives. Binding 3 is the raw
    // source view: the arena for internal images, or a caller's VkBuffer.
    const auto binding_count = static_cast<std::uint32_t>(4 + auxiliary_.size());
    if (binding_count > ctx_->limits.maxPerStageDescriptorStorageBuffers ||
        binding_count > ctx_->limits.maxDescriptorSetStorageBuffers)
        throw std::invalid_argument(
            "vulkan: storage bindings exceed device limits"
        );
    std::vector<VkDescriptorSetLayoutBinding> bindings(binding_count);
    for (std::size_t i = 0; i < bindings.size(); ++i) {
        bindings[i].binding = static_cast<std::uint32_t>(i);
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo set_info{};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    set_info.pBindings = bindings.data();
    Check(
        vkCreateDescriptorSetLayout(
            ctx_->device, &set_info, nullptr, &descriptor_set_layout_
        ),
        "vkCreateDescriptorSetLayout"
    );

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.offset = 0;
    range.size = static_cast<std::uint32_t>(config_.push_constant_size);

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &descriptor_set_layout_;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;
    Check(
        vkCreatePipelineLayout(
            ctx_->device, &layout_info, nullptr, &pipeline_layout_
        ),
        "vkCreatePipelineLayout"
    );

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 2 * binding_count;
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 2;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    Check(
        vkCreateDescriptorPool(
            ctx_->device, &pool_info, nullptr, &descriptor_pool_
        ),
        "vkCreateDescriptorPool"
    );

    VkDescriptorSetAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorPool = descriptor_pool_;
    const std::array<VkDescriptorSetLayout, 2> layouts{descriptor_set_layout_,
                                                       descriptor_set_layout_};
    allocate.descriptorSetCount = static_cast<std::uint32_t>(layouts.size());
    allocate.pSetLayouts = layouts.data();
    Check(
        vkAllocateDescriptorSets(
            ctx_->device, &allocate, descriptor_sets_.data()
        ),
        "vkAllocateDescriptorSets"
    );

    std::vector<VkDescriptorBufferInfo> buffer_info(binding_count);
    buffer_info[0].buffer = arena_.buffer;
    buffer_info[0].offset = 0;
    buffer_info[0].range = arena_bytes_;
    buffer_info[1].buffer = arena_.buffer;
    buffer_info[1].offset = plane_base_bytes;
    buffer_info[1].range = plane_bytes;
    buffer_info[2].buffer = arena_.buffer;
    buffer_info[2].offset = 0;
    buffer_info[2].range = arena_bytes_;
    buffer_info[3] = buffer_info[2];

    for (std::size_t i = 0; i < auxiliary_.size(); ++i) {
        buffer_info[i + 4] = {auxiliary_[i].buffer, 0,
                              config_.auxiliary_bytes[i]};
    }
    std::vector<VkWriteDescriptorSet> writes(2 * binding_count);
    for (std::size_t image = 0; image < descriptor_sets_.size(); ++image) {
        for (std::size_t binding = 0; binding < binding_count; ++binding) {
            auto& write = writes[image * binding_count + binding];
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptor_sets_[image];
            write.dstBinding = static_cast<std::uint32_t>(binding);
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &buffer_info[binding];
        }
        source_bindings_[image] = buffer_info[3];
    }
    vkUpdateDescriptorSets(
        ctx_->device, static_cast<std::uint32_t>(writes.size()), writes.data(),
        0, nullptr
    );
}

VkPipeline VulkanRuntime::CreatePipeline(std::string_view name) {
    const shaders::Blob& blob = FindBlob(name);
    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = blob.word_count * sizeof(std::uint32_t);
    module_info.pCode = blob.words;
    VkShaderModule module = VK_NULL_HANDLE;
    Check(
        vkCreateShaderModule(ctx_->device, &module_info, nullptr, &module),
        "vkCreateShaderModule"
    );

    VkComputePipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = pipeline_layout_;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult result = vkCreateComputePipelines(
        ctx_->device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline
    );
    vkDestroyShaderModule(ctx_->device, module, nullptr);
    Check(result, "vkCreateComputePipelines");
    return pipeline;
}

void VulkanRuntime::DestroyPipelines(
    std::span<const VkPipeline> pipelines
) noexcept {
    for (VkPipeline pipeline : pipelines)
        if (pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(ctx_->device, pipeline, nullptr);
}

void VulkanRuntime::BeginCommandBuffer() {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    Check(
        vkBeginCommandBuffer(command_buffer_, &begin_info),
        "vkBeginCommandBuffer"
    );
    if (shared_) {
        Barrier(
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT
        );
    }
#ifdef VQM_STAGE_TIMING
    if (queries_ != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(command_buffer_, queries_, 0, kMaxTimestamps);
    }
#endif
}

void VulkanRuntime::EndCommandBuffer() {
    Check(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");
}

void VulkanRuntime::ResetCommandPool() {
    Check(
        vkResetCommandPool(ctx_->device, command_pool_, 0), "vkResetCommandPool"
    );
}

void VulkanRuntime::BindDescriptorSet() { BindSource(0); }

void VulkanRuntime::BindSource(std::size_t image) {
    vkCmdBindDescriptorSets(
        command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
        &descriptor_sets_[image], 0, nullptr
    );
}

void VulkanRuntime::Dispatch(
    VkPipeline pipeline, const void* params, std::size_t params_bytes,
    std::uint32_t groups_x, std::uint32_t groups_y, std::uint32_t groups_z
) {
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(
        command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        static_cast<std::uint32_t>(params_bytes), params
    );
    vkCmdDispatch(
        command_buffer_, std::max(groups_x, 1u), std::max(groups_y, 1u),
        std::max(groups_z, 1u)
    );
}

void VulkanRuntime::Barrier(
    VkPipelineStageFlags src_stage, VkAccessFlags src_access,
    VkPipelineStageFlags dst_stage, VkAccessFlags dst_access
) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(
        command_buffer_, src_stage, dst_stage, 0, 1, &barrier, 0, nullptr, 0,
        nullptr
    );
}

void VulkanRuntime::ComputeBarrier() {
    Barrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
    );
}

void VulkanRuntime::HostToComputeBarrier() {
    Barrier(
        VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
    );
}

void VulkanRuntime::ComputeToHostBarrier() {
    Barrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT
    );
}

void VulkanRuntime::CopyUpload(
    std::size_t staging_floats, std::uint32_t arena_floats, std::size_t floats
) {
    VkBufferCopy copy{};
    copy.srcOffset = Bytes(staging_floats);
    copy.dstOffset = Bytes(arena_floats);
    copy.size = Bytes(floats);
    vkCmdCopyBuffer(command_buffer_, staging_.buffer, arena_.buffer, 1, &copy);
}

void VulkanRuntime::FillZero(std::uint32_t arena_floats, std::size_t floats) {
    vkCmdFillBuffer(
        command_buffer_, arena_.buffer, Bytes(arena_floats), Bytes(floats), 0u
    );
}

void VulkanRuntime::RecordStagedCopies(std::span<const CopyRange> copies) {
    // The host filled the staging buffer before recording; make the writes
    // visible to the copies that read them on the compute queue.
    Barrier(
        VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
    );
    for (const CopyRange& copy : copies) {
        CopyUpload(copy.staging_floats, copy.arena_floats, copy.floats);
    }
    Barrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
    );
}

void VulkanRuntime::RecordReadback(
    std::uint32_t arena_floats, std::size_t staging_floats, std::size_t floats
) {
    Barrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
    );
    VkBufferCopy readback{};
    readback.srcOffset = Bytes(arena_floats);
    readback.dstOffset = Bytes(staging_floats);
    readback.size = Bytes(floats);
    vkCmdCopyBuffer(
        command_buffer_, arena_.buffer, staging_.buffer, 1, &readback
    );
    Barrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT
    );
}

void VulkanRuntime::PublishStaged(
    std::uint32_t arena_floats, std::size_t staging_floats, std::size_t floats
) {
    const std::array<CopyRange, 1> copies{
        CopyRange{staging_floats, arena_floats, floats}};
    RequireIdle();
    ResetCommandPool();
    BeginCommandBuffer();
    RecordStagedCopies(copies);
    EndCommandBuffer();
    SubmitAndWait();
}

void VulkanRuntime::RequireIdle() const {
    if (pending_) {
        throw std::logic_error(
            "vulkan: a submission is already in flight; Finish it first"
        );
    }
}

std::uint64_t VulkanRuntime::BeginSubmission() {
    RequireIdle();
    ResetCommandPool();
    ResetTiming();
    token_ = NextHandleIncarnation();
    return token_;
}

void VulkanRuntime::QueueSubmit() {
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer_;
    QueueSubmitInfo(submit);
}

void VulkanRuntime::ValidateSubmitSync(const VulkanSubmitSync& sync) {
    if (sync.waits.size() > UINT32_MAX || sync.signals.size() > UINT32_MAX ||
        (!sync.waits.empty() && sync.waits.data() == nullptr) ||
        (!sync.signals.empty() && sync.signals.data() == nullptr)) {
        throw std::invalid_argument("vulkan: invalid semaphore list");
    }
    constexpr VkPipelineStageFlags supported =
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    constexpr VkPipelineStageFlags covers_compute =
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    std::unordered_set<VkSemaphore> handles;
    for (const VulkanSemaphoreWait& wait : sync.waits) {
        if (wait.semaphore == VK_NULL_HANDLE ||
            (wait.stages & ~supported) != 0 ||
            (wait.stages & covers_compute) == 0 ||
            !handles.insert(wait.semaphore).second) {
            throw std::invalid_argument(
                "vulkan: invalid or duplicate wait semaphore"
            );
        }
    }
    for (const VkSemaphore signal : sync.signals) {
        if (signal == VK_NULL_HANDLE || !handles.insert(signal).second) {
            throw std::invalid_argument(
                "vulkan: invalid or duplicate signal semaphore"
            );
        }
    }
}

void VulkanRuntime::QueueSubmit(const VulkanSubmitSync& sync) {
    std::vector<VkSemaphore> waits;
    std::vector<VkPipelineStageFlags> stages;
    waits.reserve(sync.waits.size());
    stages.reserve(sync.waits.size());
    for (const VulkanSemaphoreWait& wait : sync.waits) {
        waits.push_back(wait.semaphore);
        stages.push_back(wait.stages);
    }
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = static_cast<std::uint32_t>(waits.size());
    submit.pWaitSemaphores = waits.data();
    submit.pWaitDstStageMask = stages.data();
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer_;
    submit.signalSemaphoreCount =
        static_cast<std::uint32_t>(sync.signals.size());
    submit.pSignalSemaphores = sync.signals.data();
    QueueSubmitInfo(submit);
}

void VulkanRuntime::QueueSubmitInfo(const VkSubmitInfo& submit) {
    Check(vkResetFences(ctx_->device, 1, &fence_), "vkResetFences");
    // The queue is process-wide and VkQueue is externally synchronized, so a
    // second pipeline submitting from another thread waits here. Only the call
    // itself is serialized: the fence is this runtime's, so the wait below and
    // the other pipeline's execution stay independent.
    {
        std::lock_guard<std::mutex> lock(ctx_->queue_mutex);
        Check(vkQueueSubmit(ctx_->queue, 1, &submit, fence_), "vkQueueSubmit");
    }
    pending_ = true;
}

void VulkanRuntime::WaitFence() {
    Check(
        vkWaitForFences(ctx_->device, 1, &fence_, VK_TRUE, UINT64_MAX),
        "vkWaitForFences"
    );
}

void VulkanRuntime::SubmitAndWait() {
    QueueSubmit();
    WaitFence();
    // A one-shot copy is not a token submission: QueueSubmit armed the
    // pair-in-flight flag, but nothing will call Finish for it, so leaving it
    // set would make every later Submit throw std::logic_error. Nothing on a
    // BAR-visible device reaches this path (the staged copies go straight into
    // the arena), which is why the omission only bites on a device without
    // host-visible device-local memory.
    pending_ = false;
}

void VulkanRuntime::FinishSubmission(std::uint64_t token) {
    if (!pending_ || token != token_) {
        throw std::invalid_argument("vulkan: no submission with that token");
    }
    WaitFence();
    pending_ = false;
    CollectTimings();
}

#ifdef VQM_STAGE_TIMING
void VulkanRuntime::ResetTiming() {
    timed_ranges_.clear();
    next_timestamp_ = 0;
    open_timestamp_ = -1;
}

void VulkanRuntime::OpenStage(Stage stage) {
    if (queries_ == VK_NULL_HANDLE || next_timestamp_ + 2 > kMaxTimestamps) {
        open_timestamp_ = -1;
        return;
    }
    vkCmdWriteTimestamp(
        command_buffer_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queries_,
        next_timestamp_
    );
    open_timestamp_ = static_cast<std::int32_t>(next_timestamp_++);
    open_stage_ = stage;
}

void VulkanRuntime::CloseStage() {
    if (open_timestamp_ < 0) return;
    vkCmdWriteTimestamp(
        command_buffer_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queries_,
        next_timestamp_
    );
    timed_ranges_.push_back(
        TimedRange{open_stage_, static_cast<std::uint32_t>(open_timestamp_),
                   next_timestamp_++}
    );
    open_timestamp_ = -1;
}

void VulkanRuntime::CollectTimings() {
    if (queries_ == VK_NULL_HANDLE || timed_ranges_.empty()) return;
    std::array<std::uint64_t, kMaxTimestamps> stamps{};
    Check(
        vkGetQueryPoolResults(
            ctx_->device, queries_, 0, next_timestamp_,
            sizeof(stamps[0]) * next_timestamp_, stamps.data(),
            sizeof(stamps[0]), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
        ),
        "vkGetQueryPoolResults"
    );
    // Local first, then one add: another pipeline may be finishing on another
    // thread, and the accumulator is shared.
    StageTiming delta;
    for (const TimedRange& range : timed_ranges_) {
        const double ticks = static_cast<double>(ElapsedTimestampTicks(
            stamps[range.begin], stamps[range.end], ctx_->timestamp_valid_bits
        ));
        delta.seconds[static_cast<std::size_t>(range.stage)] +=
            ticks * static_cast<double>(ctx_->timestamp_period) * 1e-9;
    }
    AddStageTimings(delta);
}
#else
void VulkanRuntime::ResetTiming() {}
void VulkanRuntime::OpenStage(Stage) {}
void VulkanRuntime::CloseStage() {}
void VulkanRuntime::CollectTimings() {}
#endif  // VQM_STAGE_TIMING

}  // namespace vqm::internal
