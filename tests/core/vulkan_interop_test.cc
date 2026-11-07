// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// Exercise the public Vulkan integration using a device and GPU buffers owned
// by the caller. Pixel writes originate in a Vulkan transfer, not the metric's
// Upload path or a host pointer into its arena.

#include <algorithm>
#include <cmath>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <vulkan/vulkan.h>

#include "butteraugli.h"
#include "input.h"
#include "vqm_metrics.h"
#include "vqm_pool.h"
#include "vqm_shared.h"
#include "vqm_vulkan.h"
#include "support.h"

namespace {

void RequireVk(VkResult result) {
    if (result != VK_SUCCESS)
        throw std::runtime_error("Vulkan test setup failed");
}

VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

class CallerVulkan {
public:
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
    };

    CallerVulkan() {
        try {
            Initialize();
        } catch (...) {
            Destroy();
            throw;
        }
    }
    ~CallerVulkan() { Destroy(); }
    CallerVulkan(const CallerVulkan&) = delete;
    CallerVulkan& operator=(const CallerVulkan&) = delete;

    [[nodiscard]] vqm::VulkanDevice Device() const {
        return {physical_, device_, queue_, queue_family_, false};
    }

    [[nodiscard]] bool HasSecondQueue() const noexcept {
        return queue_count_ > 1;
    }

    Buffer MakeBuffer(
        VkDeviceSize bytes, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties
    ) {
        Buffer out;
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = bytes;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        RequireVk(vkCreateBuffer(device_, &info, nullptr, &out.buffer));
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, out.buffer, &requirements);
        VkPhysicalDeviceMemoryProperties memory_properties{};
        vkGetPhysicalDeviceMemoryProperties(physical_, &memory_properties);
        std::uint32_t index = memory_properties.memoryTypeCount;
        for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            const bool compatible =
                (requirements.memoryTypeBits & (1u << i)) != 0;
            const bool required =
                (memory_properties.memoryTypes[i].propertyFlags & properties) ==
                properties;
            if (compatible && required) {
                index = i;
                break;
            }
        }
        if (index == memory_properties.memoryTypeCount) {
            vkDestroyBuffer(device_, out.buffer, nullptr);
            throw std::runtime_error(
                "required Vulkan test memory type unavailable"
            );
        }
        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = index;
        const VkResult allocated =
            vkAllocateMemory(device_, &allocation, nullptr, &out.memory);
        if (allocated != VK_SUCCESS) {
            vkDestroyBuffer(device_, out.buffer, nullptr);
            RequireVk(allocated);
        }
        const VkResult bound =
            vkBindBufferMemory(device_, out.buffer, out.memory, 0);
        if (bound != VK_SUCCESS) {
            vkDestroyBuffer(device_, out.buffer, nullptr);
            vkFreeMemory(device_, out.memory, nullptr);
            RequireVk(bound);
        }
        buffers_.push_back(out);
        return out;
    }

    VkSemaphore MakeSemaphore() {
        VkSemaphoreCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkSemaphore semaphore = VK_NULL_HANDLE;
        RequireVk(vkCreateSemaphore(device_, &info, nullptr, &semaphore));
        semaphores_.push_back(semaphore);
        return semaphore;
    }

    // Consume the library's signal in a subsequent queue submission. A
    // finite fence wait makes a missing signal fail instead of hanging tests.
    void WaitForSignals(
        std::span<const VkSemaphore> semaphores, bool other_queue = false
    ) {
        std::vector<VkPipelineStageFlags> stages(
            semaphores.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
        );
        VkFenceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        RequireVk(vkCreateFence(device_, &info, nullptr, &fence));
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount =
            static_cast<std::uint32_t>(semaphores.size());
        submit.pWaitSemaphores = semaphores.data();
        submit.pWaitDstStageMask = stages.data();
        const VkQueue queue = other_queue ? auxiliary_queue_ : queue_;
        RequireVk(vkQueueSubmit(queue, 1, &submit, fence));
        const VkResult waited =
            vkWaitForFences(device_, 1, &fence, VK_TRUE, 5000000000ull);
        if (waited != VK_SUCCESS) {
            std::fprintf(
                stderr, "timed out waiting for library semaphore signal\n"
            );
            std::_Exit(EXIT_FAILURE);
        }
        vkDestroyFence(device_, fence, nullptr);
    }

    // Write both images to staging, then issue actual GPU copies into the
    // caller's device buffers. The ordered variant supplies its own barrier;
    // the queued variant relies on the library's same-queue dependency.
    void Produce(
        const Buffer& a, VkDeviceSize a_offset, const vqm::Image& original,
        const Buffer& b, VkDeviceSize b_offset, const vqm::Image& distorted,
        bool wait_for_completion = true,
        std::span<const VkSemaphore> signals = {}, bool other_queue = false
    ) {
        const VkDeviceSize a_bytes = original.data.size();
        const VkDeviceSize b_bytes = distorted.data.size();
        const VkDeviceSize b_source = AlignUp(a_bytes, 4);
        const Buffer staging = MakeBuffer(
            b_source + b_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        );
        void* mapped = nullptr;
        RequireVk(
            vkMapMemory(device_, staging.memory, 0, VK_WHOLE_SIZE, 0, &mapped)
        );
        auto* bytes = static_cast<std::byte*>(mapped);
        std::memcpy(bytes, original.data.data(), original.data.size());
        std::memcpy(
            bytes + b_source, distorted.data.data(), distorted.data.size()
        );
        VkMappedMemoryRange flush{};
        flush.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        flush.memory = staging.memory;
        flush.size = VK_WHOLE_SIZE;
        RequireVk(vkFlushMappedMemoryRanges(device_, 1, &flush));
        vkUnmapMemory(device_, staging.memory);

        RequireVk(vkResetCommandPool(device_, command_pool_, 0));
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        RequireVk(vkBeginCommandBuffer(command_buffer_, &begin));
        const VkBufferCopy copy_a{0, a_offset, a_bytes};
        const VkBufferCopy copy_b{b_source, b_offset, b_bytes};
        vkCmdCopyBuffer(command_buffer_, staging.buffer, a.buffer, 1, &copy_a);
        vkCmdCopyBuffer(command_buffer_, staging.buffer, b.buffer, 1, &copy_b);
        if (wait_for_completion) {
            VkMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(
                command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr,
                0, nullptr
            );
        }
        RequireVk(vkEndCommandBuffer(command_buffer_));
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_buffer_;
        submit.signalSemaphoreCount = static_cast<std::uint32_t>(signals.size());
        submit.pSignalSemaphores = signals.data();
        const VkQueue queue = other_queue ? auxiliary_queue_ : queue_;
        RequireVk(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
        if (wait_for_completion) RequireVk(vkQueueWaitIdle(queue));
    }

    void CheckAlive() const { RequireVk(vkDeviceWaitIdle(device_)); }

private:
    void Initialize() {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.apiVersion = VK_API_VERSION_1_1;
        app.pApplicationName = "ssimulacra2 interop test";
        VkInstanceCreateInfo instance_info{};
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_info.pApplicationInfo = &app;
        RequireVk(vkCreateInstance(&instance_info, nullptr, &instance_));
        std::uint32_t count = 0;
        RequireVk(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
        if (count == 0) throw std::runtime_error("no Vulkan physical device");
        std::vector<VkPhysicalDevice> devices(count);
        RequireVk(vkEnumeratePhysicalDevices(instance_, &count, devices.data()));
        for (VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(candidate, &properties);
            if (properties.apiVersion < VK_API_VERSION_1_1 ||
                properties.limits.maxComputeWorkGroupInvocations < 256 ||
                properties.limits.maxComputeSharedMemorySize < 16384)
                continue;
            std::uint32_t families_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(
                candidate, &families_count, nullptr
            );
            std::vector<VkQueueFamilyProperties> families(families_count);
            vkGetPhysicalDeviceQueueFamilyProperties(
                candidate, &families_count, families.data()
            );
            for (std::uint32_t family = 0; family < families_count; ++family) {
                if ((families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0)
                    continue;
                if (physical_ == VK_NULL_HANDLE ||
                    properties.deviceType ==
                        VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                    physical_ = candidate;
                    queue_family_ = family;
                    queue_count_ = std::min(families[family].queueCount, 2u);
                }
                break;
            }
            if (physical_ == candidate &&
                properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                break;
        }
        if (physical_ == VK_NULL_HANDLE)
            throw std::runtime_error("no compatible compute device");
        const std::array<float, 2> priorities{1.0f, 1.0f};
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = queue_family_;
        queue_info.queueCount = queue_count_;
        queue_info.pQueuePriorities = priorities.data();
        VkDeviceCreateInfo device_info{};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        RequireVk(vkCreateDevice(physical_, &device_info, nullptr, &device_));
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
        vkGetDeviceQueue(
            device_, queue_family_, queue_count_ > 1 ? 1 : 0, &auxiliary_queue_
        );
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = queue_family_;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        RequireVk(
            vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_)
        );
        VkCommandBufferAllocateInfo cmd_info{};
        cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_info.commandPool = command_pool_;
        cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_info.commandBufferCount = 1;
        RequireVk(
            vkAllocateCommandBuffers(device_, &cmd_info, &command_buffer_)
        );
    }

    void Destroy() noexcept {
        if (device_ != VK_NULL_HANDLE) {
            (void)vkDeviceWaitIdle(device_);
            if (command_pool_ != VK_NULL_HANDLE)
                vkDestroyCommandPool(device_, command_pool_, nullptr);
            for (const Buffer& buffer : buffers_) {
                vkDestroyBuffer(device_, buffer.buffer, nullptr);
                vkFreeMemory(device_, buffer.memory, nullptr);
            }
            for (VkSemaphore semaphore : semaphores_)
                vkDestroySemaphore(device_, semaphore, nullptr);
            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
        }
    }

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkQueue auxiliary_queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;
    std::uint32_t queue_count_ = 0;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    std::vector<Buffer> buffers_;
    std::vector<VkSemaphore> semaphores_;
};

vqm::Image MakeSubsampledYuv() {
    using namespace vqm;
    constexpr Size size{34, 18};
    ImageFormat format;
    format.type = PixelType::Word;
    format.depth = 10;
    format.family = ColorFamily::Yuv;
    format.matrix = MatrixCoefficients::Bt709;
    format.transfer = TransferCharacteristics::Srgb;
    format.primaries = ColorPrimaries::Bt709;
    format.subsample_w = 1;
    format.subsample_h = 1;
    format.alpha = AlphaType::Straight;
    std::array<std::vector<float>, 4> values;
    for (std::size_t y = 0; y < size.height; ++y) {
        for (std::size_t x = 0; x < size.width; ++x) {
            values[0].push_back(
                static_cast<float>((3 * x + 7 * y) % 80 + 10) / 100.0f
            );
            values[3].push_back(static_cast<float>((x + 2 * y) % 9 + 1) / 10.0f);
        }
    }
    for (std::size_t y = 0; y < size.height / 2; ++y) {
        for (std::size_t x = 0; x < size.width / 2; ++x) {
            values[1].push_back(
                static_cast<float>(static_cast<int>((x + y) % 11) - 5) / 50.0f
            );
            values[2].push_back(
                static_cast<float>(static_cast<int>((2 * x + y) % 9) - 4) / 40.0f
            );
        }
    }
    return vqm::test::PackPlanes(format, size, values);
}

struct PaddedImage {
    vqm::Image payload;
    std::array<vqm::VulkanPlaneLayout, 4> planes{};
};

PaddedImage AddPlanePadding(const vqm::Image& input) {
    const auto packed = vqm::internal::PlaneLayout(input.format, input.size);
    PaddedImage result;
    result.payload = input;
    std::size_t end = 4;
    for (std::size_t p = 0; p < packed.size(); ++p) {
        const auto& geometry = packed[p];
        if (!geometry.present()) continue;
        const std::size_t row = geometry.width * geometry.sample_bytes;
        const std::size_t stride = row + geometry.sample_bytes * (p + 1);
        end = (end + 3) / 4 * 4;
        result.planes[p] = {end, stride};
        end += (geometry.height - 1) * stride + row;
        end = (end + 3) / 4 * 4;
    }
    result.payload.data.assign(end, 0xcd);
    for (std::size_t p = 0; p < packed.size(); ++p) {
        const auto& geometry = packed[p];
        if (!geometry.present()) continue;
        const std::size_t row = geometry.width * geometry.sample_bytes;
        for (std::size_t y = 0; y < geometry.height; ++y) {
            std::memcpy(
                result.payload.data.data() + result.planes[p].offset_bytes +
                    y * result.planes[p].stride_bytes,
                input.data.data() + geometry.offset + y * row, row
            );
        }
    }
    return result;
}

}  // namespace

VQM_TEST(TestSharedMetricsBorrowedVulkanBuffers) {
    if (!vqm::test::GpuAvailable()) {
        vqm::test::MarkSkip("no usable Vulkan device", true);
        return;
    }
    CallerVulkan caller;
    const auto original = vqm::test::MakeGradient(0);
    const auto distorted = vqm::test::MakeGradient(1);
    const VkDeviceSize bytes = original.data.size();
    constexpr VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const auto a =
        caller.MakeBuffer(bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const auto b =
        caller.MakeBuffer(bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    caller.Produce(a, 0, original, b, 0, distorted);
    const vqm::VulkanBufferView va{a.buffer, 0, bytes, original.size,
                                   original.format};
    const vqm::VulkanBufferView vb{b.buffer, 0, bytes, distorted.size,
                                   distorted.format};

    vqm::Ssimulacra2Pipeline standalone(
        {.size = original.size}, caller.Device()
    );
    vqm::ButteraugliPipeline butter({.size = original.size}, caller.Device());
    const double expected_s = standalone.Compute(va, vb).Score();
    const auto expected_b = butter.Compute(va, vb);
    vqm::SharedMetricWorkspace workspace(
        {.size = original.size, .streaming_inputs = false}, caller.Device()
    );
    CHECK_NEAR(workspace.ComputeSsimulacra2(va, vb).Score(), expected_s, 1e-9);
    CHECK_NEAR(
        workspace.ComputeButteraugli(va, vb).score, expected_b.score, 1e-9
    );
    CHECK_NEAR(workspace.ComputeSsimulacra2(va, vb).Score(), expected_s, 1e-9);
    CHECK_THROWS_AS(
        workspace.ComputeButteraugli(original, distorted), std::invalid_argument
    );
}

// The unified call submits multiple metric pipelines on one queue. A binary
// producer semaphore must be consumed once, and the consumer semaphore must
// be signaled only after all selected metrics have completed.
VQM_TEST(TestMetricsBorrowedBuffersWithSemaphores) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm;
    CallerVulkan caller;
    const auto original = test::MakeGradient(0);
    const auto distorted = test::MakeGradient(1);
    const VkDeviceSize bytes = original.data.size();
    constexpr auto usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const auto a =
        caller.MakeBuffer(bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const auto b =
        caller.MakeBuffer(bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const VulkanBufferView va{a.buffer, 0, bytes, original.size,
                              original.format};
    const VulkanBufferView vb{b.buffer, 0, bytes, distorted.size,
                              distorted.format};

    MetricsConfig config;
    config.size = original.size;
    config.original_format = original.format;
    config.distorted_format = distorted.format;
    config.streaming_inputs = false;
    config.ssimulacra2 = Ssimulacra2Config{.size = config.size};
    config.butteraugli = ButteraugliConfig{.size = config.size};
    Metrics metrics(config, caller.Device());
    Ssimulacra2Pipeline ssimu({.size = config.size}, caller.Device());
    ButteraugliPipeline butter({.size = config.size}, caller.Device());
    const auto expected_ssimu = ssimu.Compute(original, distorted).Score();
    const auto expected_butter = butter.Compute(original, distorted).score;

    const auto ready = caller.MakeSemaphore();
    const auto done = caller.MakeSemaphore();
    const std::array<VkSemaphore, 1> producer_signals{ready};
    const std::array<VulkanSemaphoreWait, 1> waits{
        {{ready, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT}}};
    const std::array<VkSemaphore, 1> signals{done};
    const std::array<VkSemaphore, 1> overlap{ready};
    caller.Produce(
        a, 0, original, b, 0, distorted, false, producer_signals, true
    );

    // Validate the entire original sync object before splitting it across the
    // submissions, including a semaphore present in both lists.
    CHECK_THROWS_AS(
        metrics.Compute(va, vb, VulkanSubmitSync{waits, overlap}),
        std::invalid_argument
    );
    const auto result =
        metrics.Compute(va, vb, VulkanSubmitSync{waits, signals});
    CHECK(result.has_value());
    if (result) {
        CHECK(result->frame_index == 0);
        CHECK(result->ssimulacra2.has_value());
        CHECK(result->butteraugli.has_value());
        if (result->ssimulacra2)
            CHECK_NEAR(*result->ssimulacra2, expected_ssimu, 1e-9);
        if (result->butteraugli)
            CHECK_NEAR(*result->butteraugli, expected_butter, 1e-9);
    }
    caller.WaitForSignals(signals, true);
    CHECK(!metrics.Flush());
}

VQM_TEST(TestBorrowedVulkanBuffers) {
    if (!vqm::test::GpuAvailable()) return;
    CallerVulkan caller;
    const auto orig = vqm::test::MakeGradient(0);
    const auto dist = vqm::test::MakeGradient(1);
    const auto replacement = vqm::test::MakeGradient(2);
    const VkDeviceSize bytes = orig.data.size();
    // VulkanBufferView promises word-aligned byte offsets, not necessarily the
    // device's stronger storage-buffer descriptor offset alignment.
    constexpr VkDeviceSize a_offset = 4;
    constexpr VkDeviceSize b_offset = 12;
    constexpr VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const auto a = caller.MakeBuffer(
        a_offset + bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    const auto b = caller.MakeBuffer(
        b_offset + bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    caller.Produce(a, a_offset, orig, b, b_offset, dist);
    const vqm::VulkanBufferView va{a.buffer, a_offset, bytes, orig.size,
                                   orig.format};
    const vqm::VulkanBufferView vb{b.buffer, b_offset, bytes, dist.size,
                                   dist.format};
    // A Vulkan buffer is meaningful only on the VkDevice that created it.
    // Self-created pipeline devices must reject these handles before use.
    {
        vqm::Ssimulacra2Config config;
        config.size = orig.size;
        config.input_bytes = 2 * orig.data.size();
        vqm::Ssimulacra2Pipeline pipeline(config);
        CHECK_THROWS_AS(pipeline.Compute(va, vb), std::invalid_argument);
        vqm::ButteraugliConfig butter_config;
        butter_config.size = orig.size;
        vqm::ButteraugliPipeline butter(butter_config);
        CHECK_THROWS_AS(butter.Compute(va, vb), std::invalid_argument);
    }
    {
        vqm::Ssimulacra2Config config;
        config.size = orig.size;
        auto bad_device = caller.Device();
        bad_device.physical_device = VK_NULL_HANDLE;
        CHECK_THROWS_AS(
            vqm::Ssimulacra2Pipeline(config, bad_device), std::invalid_argument
        );
        bad_device = caller.Device();
        bad_device.device = VK_NULL_HANDLE;
        CHECK_THROWS_AS(
            vqm::Ssimulacra2Pipeline(config, bad_device), std::invalid_argument
        );
        bad_device = caller.Device();
        bad_device.queue = VK_NULL_HANDLE;
        CHECK_THROWS_AS(
            vqm::Ssimulacra2Pipeline(config, bad_device), std::invalid_argument
        );
        bad_device = caller.Device();
        bad_device.queue_family = UINT32_MAX;
        CHECK_THROWS_AS(
            vqm::Ssimulacra2Pipeline(config, bad_device), std::invalid_argument
        );
        vqm::Ssimulacra2Pipeline pipeline(config, caller.Device());
        const double expected = pipeline.Compute(orig, dist).Score();
        const double identity = pipeline.Compute(va, va).Score();
        CHECK(identity >= 99.5 && identity <= 100.0);
        CHECK_NEAR(pipeline.Compute(va, vb).Score(), expected, 1e-9);
        CHECK(expected < 99.5);
        auto external_only = config;
        external_only.gpu_resident_images = 0;
        external_only.streaming_inputs = false;
        vqm::Ssimulacra2Pipeline direct(external_only, caller.Device());
        CHECK_NEAR(direct.Compute(va, vb).Score(), expected, 1e-9);
        const auto resident_a = pipeline.Upload(orig);
        const auto resident_b = pipeline.Upload(dist);
        CHECK_NEAR(
            pipeline.Compute(resident_a, resident_b).Score(), expected, 1e-9
        );
        pipeline.Release(resident_a);
        pipeline.Release(resident_b);
        CHECK_NEAR(pipeline.Compute(va, vb).Score(), expected, 1e-9);
        const auto token = pipeline.Submit(va, vb);
        CHECK_THROWS_AS(pipeline.Submit(va, vb), std::logic_error);
        CHECK_NEAR(pipeline.Finish(token).Score(), expected, 1e-9);
        CHECK_THROWS_AS(pipeline.Finish(token), std::invalid_argument);
        auto invalid = va;
        invalid.buffer = VK_NULL_HANDLE;
        CHECK_THROWS_AS(pipeline.Compute(invalid, vb), std::invalid_argument);
        invalid = va;
        invalid.bytes = bytes - 4;
        CHECK_THROWS_AS(pipeline.Compute(invalid, vb), std::invalid_argument);
        invalid = va;
        invalid.offset += 1;
        CHECK_THROWS_AS(pipeline.Compute(invalid, vb), std::invalid_argument);
        invalid = va;
        ++invalid.size.width;
        CHECK_THROWS_AS(pipeline.Compute(invalid, vb), std::invalid_argument);

        // Reusing a VkBuffer must read its new GPU contents, not a cached
        // reference pyramid for the prior contents at the same handle.
        caller.Produce(a, a_offset, replacement, b, b_offset, dist);
        const double changed = pipeline.Compute(replacement, dist).Score();
        CHECK(std::abs(changed - expected) > 1e-4);
        CHECK_NEAR(pipeline.Compute(va, vb).Score(), changed, 1e-9);
    }
    {
        vqm::Ssimulacra2Config config;
        config.size = orig.size;
        vqm::Ssimulacra2Pipeline pipeline(config, caller.Device());
        (void)pipeline.Submit(va, vb);
        // Destruction must wait before destroying resources referenced by the
        // pending command buffer. The caller may use its device immediately.
    }
    caller.CheckAlive();
    // Butteraugli uses the same runtime but separate opsin dispatches and
    // reference cache, so validate both reported distances and a mutation.
    caller.Produce(a, a_offset, orig, b, b_offset, dist);
    {
        vqm::ButteraugliConfig config;
        config.size = orig.size;
        vqm::ButteraugliPipeline pipeline(config, caller.Device());
        const auto expected = pipeline.Compute(orig, dist);
        const auto identity = pipeline.Compute(va, va);
        CHECK(identity.score == 0.0);
        CHECK(identity.distance_p3 == 0.0);
        CHECK(expected.score > 0.0);
        CHECK(expected.distance_p3 > 0.0);
        const auto actual = pipeline.Compute(va, vb);
        CHECK_NEAR(actual.score, expected.score, 1e-9);
        CHECK_NEAR(actual.distance_p3, expected.distance_p3, 1e-9);
        auto external_only = config;
        external_only.gpu_resident_images = 0;
        external_only.streaming_inputs = false;
        vqm::ButteraugliPipeline direct(external_only, caller.Device());
        const auto direct_result = direct.Compute(va, vb);
        CHECK_NEAR(direct_result.score, expected.score, 1e-9);
        CHECK_NEAR(direct_result.distance_p3, expected.distance_p3, 1e-9);
        const auto resident_a = pipeline.Upload(orig);
        const auto resident_b = pipeline.Upload(dist);
        const auto resident_result = pipeline.Compute(resident_a, resident_b);
        CHECK_NEAR(resident_result.score, expected.score, 1e-9);
        CHECK_NEAR(resident_result.distance_p3, expected.distance_p3, 1e-9);
        pipeline.Release(resident_a);
        pipeline.Release(resident_b);
        CHECK_NEAR(pipeline.Compute(va, vb).score, expected.score, 1e-9);
        const auto token = pipeline.Submit(va, vb);
        CHECK_THROWS_AS(pipeline.Submit(va, vb), std::logic_error);
        const auto result = pipeline.Finish(token);
        CHECK_NEAR(result.score, expected.score, 1e-9);
        CHECK_NEAR(result.distance_p3, expected.distance_p3, 1e-9);
        CHECK_THROWS_AS(pipeline.Finish(token), std::invalid_argument);
        auto invalid = va;
        invalid.buffer = VK_NULL_HANDLE;
        CHECK_THROWS_AS(pipeline.Compute(invalid, vb), std::invalid_argument);
        invalid = va;
        invalid.bytes = bytes - 4;
        CHECK_THROWS_AS(pipeline.Compute(invalid, vb), std::invalid_argument);
        invalid = va;
        invalid.offset += 1;
        CHECK_THROWS_AS(pipeline.Compute(invalid, vb), std::invalid_argument);
        invalid = va;
        ++invalid.size.width;
        CHECK_THROWS_AS(pipeline.Compute(invalid, vb), std::invalid_argument);

        caller.Produce(a, a_offset, replacement, b, b_offset, dist);
        const auto changed = pipeline.Compute(replacement, dist);
        CHECK(std::abs(changed.score - expected.score) > 1e-4);
        const auto updated = pipeline.Compute(va, vb);
        CHECK_NEAR(updated.score, changed.score, 1e-9);
        CHECK_NEAR(updated.distance_p3, changed.distance_p3, 1e-9);
    }
    {
        vqm::ButteraugliConfig config;
        config.size = orig.size;
        vqm::ButteraugliPipeline pipeline(config, caller.Device());
        (void)pipeline.Submit(va, vb);
    }
    // The pipeline borrowed the logical device and must not destroy it.
    caller.CheckAlive();
}

VQM_TEST(TestBorrowedVulkanBufferFormats) {
    if (!vqm::test::GpuAvailable()) return;
    CallerVulkan caller;
    const std::vector<vqm::Image> originals{
        vqm::test::MakeRampImage(33, 17, true, vqm::PixelType::Word),
        vqm::test::MakeRampImage(35, 19, false, vqm::PixelType::Float),
        MakeSubsampledYuv(),
    };
    for (const auto& original : originals) {
        auto distorted = original;
        std::fill_n(distorted.data.begin(), 256, std::uint8_t{0});
        const VkDeviceSize bytes = original.data.size();
        constexpr VkDeviceSize a_offset = 4;
        constexpr VkDeviceSize b_offset = 12;
        constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        const auto a = caller.MakeBuffer(
            a_offset + bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        const auto b = caller.MakeBuffer(
            b_offset + bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        caller.Produce(a, a_offset, original, b, b_offset, distorted);
        const vqm::VulkanBufferView va{a.buffer, a_offset, bytes, original.size,
                                       original.format};
        const vqm::VulkanBufferView vb{b.buffer, b_offset, bytes, distorted.size,
                                       distorted.format};
        constexpr float background = 0.1f;
        {
            vqm::Ssimulacra2Config config;
            config.size = original.size;
            vqm::Ssimulacra2Pipeline pipeline(config, caller.Device());
            const double expected =
                pipeline.Compute(original, distorted, background).Score();
            const double identity = pipeline.Compute(va, va, background).Score();
            CHECK(identity >= 99.5 && identity <= 100.0);
            CHECK(expected < 99.5);
            CHECK_NEAR(
                pipeline.Compute(va, vb, background).Score(), expected, 1e-9
            );
        }
        {
            vqm::ButteraugliConfig config;
            config.size = original.size;
            vqm::ButteraugliPipeline pipeline(config, caller.Device());
            const auto expected =
                pipeline.Compute(original, distorted, background);
            const auto identity = pipeline.Compute(va, va, background);
            CHECK(identity.score == 0.0);
            CHECK(identity.distance_p3 == 0.0);
            CHECK(expected.score > 0.0);
            const auto actual = pipeline.Compute(va, vb, background);
            CHECK_NEAR(actual.score, expected.score, 1e-9);
            CHECK_NEAR(actual.distance_p3, expected.distance_p3, 1e-9);
        }
    }
}

VQM_TEST(TestBorrowedVulkanQueuedProducer) {
    if (!vqm::test::GpuAvailable()) return;
    CallerVulkan caller;
    const auto original = vqm::test::MakeGradient(0);
    const auto replacement = vqm::test::MakeGradient(2);
    const auto distorted = vqm::test::MakeGradient(1);
    const VkDeviceSize bytes = original.data.size();
    constexpr VkDeviceSize a_offset = 4;
    constexpr VkDeviceSize b_offset = 12;
    constexpr VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const auto a = caller.MakeBuffer(
        a_offset + bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    const auto b = caller.MakeBuffer(
        b_offset + bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    const vqm::VulkanBufferView va{a.buffer, a_offset, bytes, original.size,
                                   original.format};
    const vqm::VulkanBufferView vb{b.buffer, b_offset, bytes, distorted.size,
                                   distorted.format};

    vqm::Ssimulacra2Config config;
    config.size = original.size;
    vqm::Ssimulacra2Pipeline pipeline(config, caller.Device());
    const double expected = pipeline.Compute(original, distorted).Score();
    vqm::ButteraugliConfig butter_config;
    butter_config.size = original.size;
    vqm::ButteraugliPipeline butter(butter_config, caller.Device());
    const auto butter_expected = butter.Compute(replacement, distorted);

    // The caller queues copies then immediately queues scoring on the same
    // VkQueue. It waits on neither the producer nor an intermediate fence, and
    // the producer records no transfer-to-compute barrier. Its command pool and
    // staging allocations remain alive until scoring has finished.
    caller.Produce(a, a_offset, original, b, b_offset, distorted, false);
    const auto token = pipeline.Submit(va, vb);
    CHECK_NEAR(pipeline.Finish(token).Score(), expected, 1e-9);

    // The previous Finish also guarantees that the first producer command
    // completed, so its pool can be reset before queueing the next producer.
    caller.Produce(a, a_offset, replacement, b, b_offset, distorted, false);
    const auto butter_token = butter.Submit(va, vb);
    const auto actual = butter.Finish(butter_token);
    CHECK_NEAR(actual.score, butter_expected.score, 1e-9);
    CHECK_NEAR(actual.distance_p3, butter_expected.distance_p3, 1e-9);
}

VQM_TEST(TestBorrowedVulkanPaddedPlanes) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm;
    CallerVulkan caller;
    const std::vector<Image> originals{
        test::MakeRampImage(33, 17, false, PixelType::Byte),
        test::MakeRampImage(33, 17, true, PixelType::Word),
        test::MakeRampImage(35, 19, false, PixelType::Float),
        MakeSubsampledYuv(),
    };
    for (const Image& original : originals) {
        Image distorted = original;
        std::fill_n(distorted.data.begin(), 256, std::uint8_t{0});
        const PaddedImage a_padded = AddPlanePadding(original);
        const PaddedImage b_padded = AddPlanePadding(distorted);
        const VkDeviceSize a_bytes = a_padded.payload.data.size();
        const VkDeviceSize b_bytes = b_padded.payload.data.size();
        constexpr VkDeviceSize a_offset = 4;
        constexpr VkDeviceSize b_offset = 12;
        constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        const auto a = caller.MakeBuffer(
            a_offset + a_bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        const auto b = caller.MakeBuffer(
            b_offset + b_bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        caller.Produce(
            a, a_offset, a_padded.payload, b, b_offset, b_padded.payload
        );
        const VulkanBufferView va{a.buffer,        a_offset,
                                  a_bytes,         original.size,
                                  original.format, a_padded.planes};
        const VulkanBufferView vb{b.buffer,         b_offset,
                                  b_bytes,          distorted.size,
                                  distorted.format, b_padded.planes};
        constexpr float background = 0.1f;
        Ssimulacra2Config config;
        config.size = original.size;
        Ssimulacra2Pipeline pipeline(config, caller.Device());
        const double expected =
            pipeline.Compute(original, distorted, background).Score();
        CHECK_NEAR(pipeline.Compute(va, vb, background).Score(), expected, 1e-9);
        CHECK(pipeline.Compute(va, va, background).Score() >= 99.5);
        ButteraugliConfig butter_config;
        butter_config.size = original.size;
        ButteraugliPipeline butter(butter_config, caller.Device());
        const auto butter_expected =
            butter.Compute(original, distorted, background);
        const auto actual = butter.Compute(va, vb, background);
        CHECK_NEAR(actual.score, butter_expected.score, 1e-9);
        CHECK_NEAR(actual.distance_p3, butter_expected.distance_p3, 1e-9);
        CHECK(butter.Compute(va, va, background).score == 0.0);

        // Different row offsets and alignments are still validated prior to
        // recording, leaving the pipeline usable after each rejection.
        auto bad = va;
        bad.planes[0].offset_bytes += 1;
        CHECK_THROWS_AS(pipeline.Compute(bad, vb), std::invalid_argument);
        bad = va;
        bad.planes[0].stride_bytes = 1;
        CHECK_THROWS_AS(pipeline.Compute(bad, vb), std::invalid_argument);
        bad = va;
        bad.planes[0].stride_bytes = std::numeric_limits<VkDeviceSize>::max();
        CHECK_THROWS_AS(pipeline.Compute(bad, vb), std::invalid_argument);
        bad = va;
        bad.bytes = 4;
        CHECK_THROWS_AS(pipeline.Compute(bad, vb), std::invalid_argument);
        bad = va;
        bad.planes[1].stride_bytes = 0;
        CHECK_THROWS_AS(pipeline.Compute(bad, vb), std::invalid_argument);
        bad = va;
        bad.planes[0].offset_bytes =
            std::numeric_limits<VkDeviceSize>::max() - 3;
        CHECK_THROWS_AS(pipeline.Compute(bad, vb), std::invalid_argument);
        if (original.format.alpha == AlphaType::None) {
            bad = va;
            bad.planes[3] = {4, 4};
            CHECK_THROWS_AS(pipeline.Compute(bad, vb), std::invalid_argument);
        }
        CHECK_NEAR(pipeline.Compute(va, vb, background).Score(), expected, 1e-9);

        // Replacing pixels in the same external buffer must rebuild its
        // reference pyramid, including with non-packed plane geometry.
        caller.Produce(
            a, a_offset, b_padded.payload, b, b_offset, b_padded.payload
        );
        CHECK(pipeline.Compute(va, vb, background).Score() >= 99.5);
        CHECK(butter.Compute(va, vb, background).score == 0.0);
    }
}

VQM_TEST(TestBorrowedVulkanPools) {
    using namespace vqm;
    if (!test::GpuAvailable()) return;
    CallerVulkan caller;
    const auto original = test::MakeGradient(0);
    const auto distorted = test::MakeGradient(1);
    const VkDeviceSize bytes = original.data.size();
    constexpr VkDeviceSize a_offset = 4;
    constexpr VkDeviceSize b_offset = 12;
    constexpr VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const auto a = caller.MakeBuffer(
        a_offset + bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    const auto b = caller.MakeBuffer(
        b_offset + bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    caller.Produce(a, a_offset, original, b, b_offset, distorted);
    const VulkanBufferView va{a.buffer, a_offset, bytes, original.size,
                              original.format};
    const VulkanBufferView vb{b.buffer, b_offset, bytes, distorted.size,
                              distorted.format};
    const std::array<VulkanBufferPair, 4> pairs{
        {{va, va}, {va, vb}, {vb, va}, {vb, vb}}};

    Ssimulacra2Config config;
    config.size = original.size;
    config.gpu_resident_images = 0;
    config.streaming_inputs = false;
    Ssimulacra2Pipeline reference(config, caller.Device());
    std::array<double, 4> expected{};
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        expected[i] =
            reference.Compute(pairs[i].original, pairs[i].distorted).Score();
    }
    Ssimulacra2Pool pool(config, 2, caller.Device());
    const auto actual = pool.ScoreMany(pairs);
    CHECK(actual.size() == pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        CHECK_NEAR(actual[i], expected[i], 1e-9);
    }
    auto invalid = vb;
    invalid.bytes = 0;
    const std::array<VulkanBufferPair, 3> bad{
        {{va, vb}, {vb, va}, {va, invalid}}};
    CHECK_THROWS_AS(pool.ComputeMany(bad), std::invalid_argument);
    CHECK_NEAR(pool.ScoreMany(std::span(pairs).first(1))[0], expected[0], 1e-9);

    ButteraugliConfig butter;
    butter.size = original.size;
    butter.gpu_resident_images = 0;
    butter.streaming_inputs = false;
    ButteraugliPipeline butter_ref(butter, caller.Device());
    std::array<ButteraugliScore, 4> butter_expected{};
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        butter_expected[i] =
            butter_ref.Compute(pairs[i].original, pairs[i].distorted);
    }
    ButteraugliPool butter_pool(butter, 2, caller.Device());
    const auto butter_actual = butter_pool.ComputeMany(pairs);
    CHECK(butter_actual.size() == pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        CHECK_NEAR(butter_actual[i].score, butter_expected[i].score, 1e-9);
        CHECK_NEAR(
            butter_actual[i].distance_p3, butter_expected[i].distance_p3, 1e-9
        );
    }
    CHECK_THROWS_AS(butter_pool.ComputeMany(bad), std::invalid_argument);
    CHECK_NEAR(
        butter_pool.ComputeMany(std::span(pairs).first(1))[0].score,
        butter_expected[0].score, 1e-9
    );
}

VQM_TEST(TestBorrowedVulkanSemaphoreSubmission) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm;
    CallerVulkan caller;
    std::printf(
        "  semaphore integration: %s\n", caller.HasSecondQueue()
                                             ? "two queues in one family"
                                             : "single queue fallback"
    );
    const Image original = test::MakeGradient(0);
    const Image distorted = test::MakeGradient(1);
    const VkDeviceSize bytes = original.data.size();
    constexpr VkDeviceSize a_offset = 4;
    constexpr VkDeviceSize b_offset = 12;
    constexpr VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const auto a = caller.MakeBuffer(
        a_offset + bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    const auto b = caller.MakeBuffer(
        b_offset + bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    const VulkanBufferView va{a.buffer, a_offset, bytes, original.size,
                              original.format};
    const VulkanBufferView vb{b.buffer, b_offset, bytes, distorted.size,
                              distorted.format};
    const auto exercise = [&](auto& pipeline, auto score) {
        const double expected = score(pipeline.Compute(original, distorted));
        const std::array<VkSemaphore, 3> ready{caller.MakeSemaphore(),
                                               caller.MakeSemaphore(),
                                               caller.MakeSemaphore()};
        const std::array<VkSemaphore, 2> done{caller.MakeSemaphore(),
                                              caller.MakeSemaphore()};
        const std::array<VulkanSemaphoreWait, 3> waits{{
            {ready[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_TRANSFER_BIT},
            {ready[1], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT},
            {ready[2], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT},
        }};
        const VulkanSubmitSync sync{waits, done};

        // All invalid combinations are rejected before recording or queueing
        // work, so the pipeline remains usable immediately afterward.
        const std::array<VulkanSemaphoreWait, 1> null_wait{
            {{VK_NULL_HANDLE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT}}};
        CHECK_THROWS_AS(
            pipeline.Submit(va, vb, VulkanSubmitSync{null_wait, {}}),
            std::invalid_argument
        );
        const std::array<VulkanSemaphoreWait, 1> zero_wait{{{ready[0], 0}}};
        CHECK_THROWS_AS(
            pipeline.Submit(va, vb, VulkanSubmitSync{zero_wait, {}}),
            std::invalid_argument
        );
        const std::array<VulkanSemaphoreWait, 1> transfer_only{
            {{ready[0], VK_PIPELINE_STAGE_TRANSFER_BIT}}};
        CHECK_THROWS_AS(
            pipeline.Submit(va, vb, VulkanSubmitSync{transfer_only, {}}),
            std::invalid_argument
        );
        const std::array<VulkanSemaphoreWait, 1> unsupported_stage{
            {{ready[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_HOST_BIT}}};
        CHECK_THROWS_AS(
            pipeline.Submit(va, vb, VulkanSubmitSync{unsupported_stage, {}}),
            std::invalid_argument
        );
        const std::array<VulkanSemaphoreWait, 2> duplicate_wait{
            {waits[0], waits[0]}};
        CHECK_THROWS_AS(
            pipeline.Submit(va, vb, VulkanSubmitSync{duplicate_wait, {}}),
            std::invalid_argument
        );
        const std::array<VkSemaphore, 1> null_signal{VK_NULL_HANDLE};
        CHECK_THROWS_AS(
            pipeline.Submit(va, vb, VulkanSubmitSync{{}, null_signal}),
            std::invalid_argument
        );
        const std::array<VkSemaphore, 2> duplicate_signal{done[0], done[0]};
        CHECK_THROWS_AS(
            pipeline.Submit(va, vb, VulkanSubmitSync{{}, duplicate_signal}),
            std::invalid_argument
        );
        const std::array<VkSemaphore, 1> overlapping_signal{ready[0]};
        CHECK_THROWS_AS(
            pipeline.Submit(va, vb, VulkanSubmitSync{waits, overlapping_signal}),
            std::invalid_argument
        );

        // Queue writes and signal both producer semaphores. Do not wait on the
        // CPU or add a producer-side transfer-to-compute barrier.
        // When available the producer and consumer use a second queue in the
        // same family, making the semaphore dependency genuinely cross-queue.
        // On a one-queue device the test still exercises binary waits/signals.
        caller.Produce(
            a, a_offset, original, b, b_offset, distorted, false, ready, true
        );
        const auto token = pipeline.Submit(va, vb, sync);
        CHECK_THROWS_AS(pipeline.Submit(va, vb, sync), std::logic_error);
        // Submit a consumer wait before the host waits for the score.
        caller.WaitForSignals(done, true);
        CHECK_NEAR(score(pipeline.Finish(token)), expected, 1e-9);

        // Empty lists are a valid synchronized overload with no dependencies.
        const auto empty = pipeline.Submit(va, vb, VulkanSubmitSync{});
        CHECK_NEAR(score(pipeline.Finish(empty)), expected, 1e-9);
    };
    Ssimulacra2Config config;
    config.size = original.size;
    Ssimulacra2Pipeline ssimu(config, caller.Device());
    exercise(ssimu, [](const Ssimulacra2Result& result) {
        return result.Score();
    });
    ButteraugliConfig butter_config;
    butter_config.size = original.size;
    ButteraugliPipeline butter(butter_config, caller.Device());
    exercise(butter, [](const ButteraugliScore& result) {
        return result.score;
    });
}
