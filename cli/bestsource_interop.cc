// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
#include "bestsource_interop.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/uuid.h>
}

#include "gpuhash.h"
#include "input.h"
#include "vulkan/vulkan_support.h"

namespace vqm::cli {
namespace {

constexpr auto kMemoryHandle = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
constexpr auto kSemaphoreHandle =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
constexpr auto kBufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

void Check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(
            std::string("BestSource Vulkan: ") + operation + " failed (" +
            std::to_string(result) + ")"
        );
}

template <class Function>
Function DeviceProc(VkDevice device, const char* name) {
    auto fn = reinterpret_cast<Function>(vkGetDeviceProcAddr(device, name));
    if (!fn)
        throw std::runtime_error(
            std::string("BestSource Vulkan requires ") + name
        );
    return fn;
}

struct Fd {
    int value = -1;
    ~Fd() {
        if (value >= 0) (void)close(value);
    }
};

using DeviceRef = std::unique_ptr<AVBufferRef, decltype([](AVBufferRef* value) {
                                      av_buffer_unref(&value);
                                  })>;

DeviceRef CreateDevice(const std::optional<DeviceUuid>& requested) {
    // 36 UUID characters plus the null terminator required by av_uuid_unparse.
    std::array<char, 37> uuid{};
    if (requested) av_uuid_unparse(requested->data(), uuid.data());
    AVBufferRef* ref = nullptr;
    const int status = av_hwdevice_ctx_create(
        &ref, AV_HWDEVICE_TYPE_VULKAN, requested ? uuid.data() : nullptr,
        nullptr, 0
    );
    DeviceRef owner(ref);
    if (status < 0)
        throw std::runtime_error(
            std::string("BestSource: cannot create FFmpeg Vulkan device") +
            (requested ? std::string(" for UUID ") + uuid.data() : "")
        );
    return owner;
}

struct DeviceSide {
    AVHWDeviceContext& av;
    const AVVulkanDeviceContext& context;
    VulkanDevice vulkan{};

    explicit DeviceSide(AVBufferRef& ref)
        : av(*reinterpret_cast<AVHWDeviceContext*>(ref.data)),
          context(*static_cast<const AVVulkanDeviceContext*>(av.hwctx)) {
        for (auto* feature = static_cast<const VkBaseInStructure*>(
                 context.device_features.pNext
             );
             feature; feature = feature->pNext) {
            if (feature->sType ==
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES)
                vulkan.storage_buffer_16bit_access =
                    reinterpret_cast<const VkPhysicalDeviceVulkan11Features*>(
                        feature
                    )
                        ->storageBuffer16BitAccess != 0;
            if (feature->sType ==
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
                vulkan.shader_float16 =
                    reinterpret_cast<const VkPhysicalDeviceVulkan12Features*>(
                        feature
                    )
                        ->shaderFloat16 != 0;
        }
        // BestSource's exporter selects the first compute family and queue 0.
        for (int i = 0; i < context.nb_qf; ++i) {
            if (!(context.qf[i].flags & VK_QUEUE_COMPUTE_BIT)) continue;
            vulkan.physical_device = context.phys_dev;
            vulkan.device = context.act_dev;
            vulkan.queue_family = static_cast<std::uint32_t>(context.qf[i].idx);
            VkDeviceQueueInfo2 info{};
            info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2;
            info.flags = context.queue_flags;
            info.queueFamilyIndex = vulkan.queue_family;
            vkGetDeviceQueue2(vulkan.device, &info, &vulkan.queue);
            if (vulkan.queue) return;
            break;
        }
        throw std::runtime_error(
            "BestSource: Vulkan device has no compute queue"
        );
    }

    [[nodiscard]] DeviceUuid Uuid() const {
        return internal::VulkanDeviceUuid(vulkan.physical_device);
    }
};

void RequireExternalSharing(VkPhysicalDevice device) {
    VkPhysicalDeviceExternalBufferInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    buffer_info.usage = kBufferUsage;
    buffer_info.handleType = kMemoryHandle;
    VkExternalBufferProperties buffer{};
    buffer.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    vkGetPhysicalDeviceExternalBufferProperties(device, &buffer_info, &buffer);

    VkPhysicalDeviceExternalSemaphoreInfo semaphore_info{};
    semaphore_info.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
    semaphore_info.handleType = kSemaphoreHandle;
    VkExternalSemaphoreProperties semaphore{};
    semaphore.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;
    vkGetPhysicalDeviceExternalSemaphoreProperties(
        device, &semaphore_info, &semaphore
    );

    constexpr auto memory_features = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT |
                                     VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
    constexpr auto semaphore_features =
        VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT |
        VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT;
    if ((buffer.externalMemoryProperties.externalMemoryFeatures &
         memory_features) != memory_features ||
        (semaphore.externalSemaphoreFeatures & semaphore_features) !=
            semaphore_features)
        throw std::runtime_error(
            "BestSource requires Vulkan opaque FD memory and semaphore sharing"
        );
}

// Match FFmpeg's queue locking on devices using its legacy synchronization.
#if FF_API_VULKAN_SYNC_QUEUES
class QueueLock {
public:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    explicit QueueLock(const DeviceSide& side) : side_(side) {
        if (side_.context.lock_queue)
            side_.context.lock_queue(&side_.av, side_.vulkan.queue_family, 0);
    }
    ~QueueLock() {
        if (side_.context.unlock_queue)
            side_.context.unlock_queue(&side_.av, side_.vulkan.queue_family, 0);
    }
#pragma GCC diagnostic pop
    QueueLock(const QueueLock&) = delete;
    QueueLock& operator=(const QueueLock&) = delete;

private:
    const DeviceSide& side_;
};
#else
struct QueueLock {
    explicit QueueLock(const DeviceSide&) {}
};
#endif

class Allocation {
public:
    explicit Allocation(const DeviceSide& side) : device_(side.vulkan.device) {}
    ~Allocation() {
        vkDestroyBuffer(device_, buffer, nullptr);
        vkFreeMemory(device_, memory, nullptr);
    }
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;

    [[nodiscard]] VkMemoryRequirements CreateBuffer(VkDeviceSize bytes) {
        VkExternalMemoryBufferCreateInfo external{};
        external.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        external.handleTypes = kMemoryHandle;
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.pNext = &external;
        info.size = bytes;
        info.usage = kBufferUsage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(
            vkCreateBuffer(device_, &info, nullptr, &buffer),
            "create shared buffer"
        );
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, buffer, &requirements);
        return requirements;
    }

    void AllocateMemory(
        VkDeviceSize bytes, std::uint32_t type, const void* external
    ) {
        VkMemoryDedicatedAllocateInfo dedicated{};
        dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        dedicated.pNext = external;
        dedicated.buffer = buffer;
        VkMemoryAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        info.pNext = &dedicated;
        info.allocationSize = bytes;
        info.memoryTypeIndex = type;
        Check(
            vkAllocateMemory(device_, &info, nullptr, &memory),
            "allocate shared memory"
        );
    }

    void BindMemory() {
        Check(
            vkBindBufferMemory(device_, buffer, memory, 0), "bind shared memory"
        );
    }

    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;

private:
    VkDevice device_;
};

std::uint32_t DeviceLocalType(VkPhysicalDevice physical, std::uint32_t allowed) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i)
        if ((allowed & (1u << i)) && (properties.memoryTypes[i].propertyFlags &
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            return i;
    throw std::runtime_error(
        "BestSource: no common device-local external memory type"
    );
}

void ShareBuffer(
    const DeviceSide& scorer, Allocation& exported, Allocation& imported,
    VkDeviceSize bytes
) {
    const auto a = exported.CreateBuffer(bytes);
    const auto b = imported.CreateBuffer(bytes);
    if (a.size != b.size)
        throw std::runtime_error(
            "BestSource: incompatible external buffer allocations"
        );
    const auto type = DeviceLocalType(
        scorer.vulkan.physical_device, a.memoryTypeBits & b.memoryTypeBits
    );

    VkExportMemoryAllocateInfo export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    export_info.handleTypes = kMemoryHandle;
    exported.AllocateMemory(a.size, type, &export_info);
    exported.BindMemory();
    auto export_fd = DeviceProc<PFN_vkGetMemoryFdKHR>(
        scorer.vulkan.device, "vkGetMemoryFdKHR"
    );
    VkMemoryGetFdInfoKHR request{};
    request.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    request.memory = exported.memory;
    request.handleType = kMemoryHandle;
    Fd fd;
    Check(
        export_fd(scorer.vulkan.device, &request, &fd.value),
        "export frame memory"
    );

    VkImportMemoryFdInfoKHR import_info{};
    import_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
    import_info.handleType = kMemoryHandle;
    import_info.fd = fd.value;
    imported.AllocateMemory(a.size, type, &import_info);
    fd.value = -1;
    imported.BindMemory();
}

class SemaphorePair {
public:
    SemaphorePair(VkDevice a, VkDevice b) : devices_{a, b} {}
    ~SemaphorePair() {
        for (std::size_t i = 0; i < devices_.size(); ++i)
            vkDestroySemaphore(devices_[i], handles[i], nullptr);
    }
    SemaphorePair(const SemaphorePair&) = delete;
    SemaphorePair& operator=(const SemaphorePair&) = delete;

    void Create() {
        VkExportSemaphoreCreateInfo external{};
        external.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
        external.handleTypes = kSemaphoreHandle;
        VkSemaphoreCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        info.pNext = &external;
        Check(
            vkCreateSemaphore(devices_[0], &info, nullptr, &handles[0]),
            "create external semaphore"
        );
        info.pNext = nullptr;
        Check(
            vkCreateSemaphore(devices_[1], &info, nullptr, &handles[1]),
            "create imported semaphore"
        );
        auto export_fd = DeviceProc<PFN_vkGetSemaphoreFdKHR>(
            devices_[0], "vkGetSemaphoreFdKHR"
        );
        auto import_fd = DeviceProc<PFN_vkImportSemaphoreFdKHR>(
            devices_[1], "vkImportSemaphoreFdKHR"
        );
        VkSemaphoreGetFdInfoKHR request{};
        request.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
        request.semaphore = handles[0];
        request.handleType = kSemaphoreHandle;
        Fd fd;
        Check(export_fd(devices_[0], &request, &fd.value), "export semaphore");
        VkImportSemaphoreFdInfoKHR import{};
        import.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
        import.semaphore = handles[1];
        import.handleType = kSemaphoreHandle;
        import.fd = fd.value;
        Check(import_fd(devices_[1], &import), "import semaphore");
        fd.value = -1;
    }

    std::array<VkSemaphore, 2> handles{};

private:
    std::array<VkDevice, 2> devices_;
};

class Commands {
public:
    explicit Commands(const DeviceSide& side) : side_(side) {}
    ~Commands() { vkDestroyCommandPool(side_.vulkan.device, pool_, nullptr); }
    Commands(const Commands&) = delete;
    Commands& operator=(const Commands&) = delete;

    void Create() {
        VkCommandPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        info.queueFamilyIndex = side_.vulkan.queue_family;
        Check(
            vkCreateCommandPool(side_.vulkan.device, &info, nullptr, &pool_),
            "create transfer command pool"
        );
        VkCommandBufferAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocation.commandPool = pool_;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = static_cast<std::uint32_t>(cmds_.size());
        Check(
            vkAllocateCommandBuffers(
                side_.vulkan.device, &allocation, cmds_.data()
            ),
            "allocate transfer command"
        );
    }

    void Reset() {
        Check(
            vkResetCommandPool(side_.vulkan.device, pool_, 0),
            "reset transfer pool"
        );
    }

    void Transfer(VkBuffer buffer, bool release, VkSemaphore semaphore) {
        const VkCommandBuffer cmd = cmds_[release ? 0 : 1];
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        Check(
            vkBeginCommandBuffer(cmd, &begin), "begin frame ownership transfer"
        );
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask =
            release ? VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT : 0;
        barrier.dstAccessMask =
            release ? 0 : VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.srcQueueFamilyIndex =
            release ? side_.vulkan.queue_family : VK_QUEUE_FAMILY_EXTERNAL;
        barrier.dstQueueFamilyIndex =
            release ? VK_QUEUE_FAMILY_EXTERNAL : side_.vulkan.queue_family;
        barrier.buffer = buffer;
        barrier.size = VK_WHOLE_SIZE;
        constexpr VkPipelineStageFlags stages =
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        vkCmdPipelineBarrier(
            cmd, stages, stages, 0, 0, nullptr, 1, &barrier, 0, nullptr
        );
        Check(vkEndCommandBuffer(cmd), "end frame ownership transfer");

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        if (release) {
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores = &semaphore;
        } else {
            submit.waitSemaphoreCount = 1;
            submit.pWaitSemaphores = &semaphore;
            submit.pWaitDstStageMask = &stages;
        }
        const QueueLock lock(side_);
        Check(
            vkQueueSubmit(side_.vulkan.queue, 1, &submit, VK_NULL_HANDLE),
            "submit frame ownership transfer"
        );
    }

    [[nodiscard]] VkResult Wait() const {
        const QueueLock lock(side_);
        return vkQueueWaitIdle(side_.vulkan.queue);
    }

private:
    const DeviceSide& side_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    std::array<VkCommandBuffer, 2> cmds_{};
};

VkDeviceSize Align(VkDeviceSize value, VkDeviceSize alignment) {
    if (value > std::numeric_limits<VkDeviceSize>::max() - (alignment - 1))
        throw std::overflow_error("BestSource: planar buffer size overflow");
    return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace

struct BestSourceDevice::Impl {
    DeviceRef device;
    DeviceSide side{*device};
    DeviceUuid uuid = side.Uuid();

    explicit Impl(const std::optional<DeviceUuid>& requested)
        : device(CreateDevice(requested)) {
        if (requested && uuid != *requested)
            throw std::runtime_error(
                "BestSource: FFmpeg did not select the requested Vulkan device"
            );
        RequireExternalSharing(side.vulkan.physical_device);
    }
};

BestSourceDevice::BestSourceDevice(const std::optional<DeviceUuid>& requested)
    : impl_(std::make_unique<Impl>(requested)) {}
BestSourceDevice::~BestSourceDevice() = default;

const VulkanDevice& BestSourceDevice::vulkan() const noexcept {
    return impl_->side.vulkan;
}

const DeviceUuid& BestSourceDevice::uuid() const noexcept { return impl_->uuid; }

struct BestSourceInterop::Impl {
    Impl(
        const DeviceSide& scorer, AVBufferRef& decoder_device, Size size,
        const ImageFormat& format
    )
        : decoder(decoder_device),
          exported(scorer),
          imported(decoder),
          scoring_commands(scorer),
          decode_commands(decoder),
          released(scorer.vulkan.device, decoder.vulkan.device),
          packed(decoder.vulkan.device, scorer.vulkan.device) {
        if (decoder.Uuid() != scorer.Uuid())
            throw std::runtime_error(
                "BestSource decoder and scorer selected different GPUs"
            );
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(
            scorer.vulkan.physical_device, &properties
        );
        const auto alignment = std::max<VkDeviceSize>(
            4, properties.limits.minStorageBufferOffsetAlignment
        );
        view.size = size;
        view.format = format;
        const auto planes = internal::PlaneLayout(format, size);
        for (std::size_t i = 0; i < targets.size(); ++i) {
            const auto& plane = planes[i];
            const auto stride = Align(plane.width * plane.sample_bytes, 4);
            const auto offset = Align(view.bytes, alignment);
            const auto bytes = stride * plane.height;
            view.planes[i] = {offset, stride};
            targets[i] = {VK_NULL_HANDLE, offset, bytes,
                          static_cast<std::ptrdiff_t>(stride)};
            view.bytes = offset + bytes;
        }
        if (view.bytes > properties.limits.maxStorageBufferRange)
            throw std::invalid_argument(
                "BestSource: packed frame exceeds storage buffer limit"
            );
        ShareBuffer(scorer, exported, imported, view.bytes);
        for (auto& target : targets) target.Buffer = imported.buffer;
        view.buffer = exported.buffer;
        scoring_commands.Create();
        decode_commands.Create();
        released.Create();
        packed.Create();
    }

    VulkanBufferView Pack(const AVFrame& frame, BSGpuHasher& exporter) {
        scoring_commands.Reset();
        decode_commands.Reset();
        try {
            scoring_commands.Transfer(
                exported.buffer, true, released.handles[0]
            );
            decode_commands.Transfer(
                imported.buffer, false, released.handles[1]
            );
            exporter.ExportAsPlanarGPU(
                &frame, static_cast<int>(view.size.width),
                static_cast<int>(view.size.height), targets.data(),
                VK_NULL_HANDLE, 0
            );
            decode_commands.Transfer(imported.buffer, true, packed.handles[0]);
            scoring_commands.Transfer(exported.buffer, false, packed.handles[1]);
            // Retire both command pools and the export before reusing the
            // commands or letting the source release the decoded frame.
            Check(scoring_commands.Wait(), "finish frame ownership transfer");
            Check(decode_commands.Wait(), "finish frame export commands");
            if (!exporter.FinishExports())
                throw std::runtime_error(
                    "BestSource: Vulkan frame export failed"
                );
        } catch (...) {
            // A wait is only queued after the matching signal was submitted.
            (void)decode_commands.Wait();
            (void)scoring_commands.Wait();
            (void)exporter.FinishExports();
            throw;
        }
        return view;
    }

    DeviceSide decoder;
    Allocation exported, imported;
    Commands scoring_commands, decode_commands;
    SemaphorePair released, packed;
    std::array<BSGpuPlaneTarget, 3> targets{};
    VulkanBufferView view{};
};

BestSourceInterop::BestSourceInterop(
    BestSourceDevice& device, AVBufferRef& decoder_device, Size size,
    const ImageFormat& format
)
    : impl_(
          std::make_unique<Impl>(
              device.impl_->side, decoder_device, size, format
          )
      ) {}

BestSourceInterop::~BestSourceInterop() = default;

VulkanBufferView BestSourceInterop::Pack(
    const AVFrame& frame, BSGpuHasher& exporter
) {
    return impl_->Pack(frame, exporter);
}

}  // namespace vqm::cli
