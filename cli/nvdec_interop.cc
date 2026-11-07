// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#include "nvdec_interop.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "input.h"
#include "nvdec_pack_ptx.h"

namespace vqm::cli {
namespace {

constexpr auto kMemoryHandle = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
constexpr auto kSemaphoreHandle =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
constexpr VkBufferUsageFlags kBufferUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

void CheckVk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("NVDEC Vulkan: ") + operation + " failed (" +
            std::to_string(result) + ")"
        );
    }
}

void CheckCuda(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS) {
        const char* error = nullptr;
        (void)cuGetErrorName(result, &error);
        throw std::runtime_error(
            std::string("NVDEC CUDA: ") + operation + " failed (" +
            (error != nullptr ? error : std::to_string(result)) + ")"
        );
    }
}

void CheckImport(CUresult result, int fd, const char* operation) {
    // CUDA takes ownership only when an opaque FD import succeeds.
    if (result != CUDA_SUCCESS) (void)close(fd);
    CheckCuda(result, operation);
}

struct DeviceChoice {
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    CUdevice cuda = 0;
    int cuda_ordinal = 0;
    std::uint32_t queue_family = 0;
    bool float16_extension = false;
};

std::optional<std::uint32_t> ComputeQueueFamily(VkPhysicalDevice device) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, queues.data());
    for (std::uint32_t i = 0; i < count; ++i) {
        if (queues[i].queueCount != 0 &&
            (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0)
            return i;
    }
    return std::nullopt;
}

std::vector<VkExtensionProperties> DeviceExtensions(VkPhysicalDevice device) {
    std::uint32_t count = 0;
    CheckVk(
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr),
        "count device extensions"
    );
    std::vector<VkExtensionProperties> extensions(count);
    CheckVk(
        vkEnumerateDeviceExtensionProperties(
            device, nullptr, &count, extensions.data()
        ),
        "enumerate device extensions"
    );
    extensions.resize(count);
    return extensions;
}

bool SupportsInteropHandles(VkPhysicalDevice device) {
    VkPhysicalDeviceExternalBufferInfo buffer{};
    buffer.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    buffer.usage = kBufferUsage;
    buffer.handleType = kMemoryHandle;
    VkExternalBufferProperties buffer_properties{};
    buffer_properties.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    vkGetPhysicalDeviceExternalBufferProperties(
        device, &buffer, &buffer_properties
    );

    VkPhysicalDeviceExternalSemaphoreInfo semaphore{};
    semaphore.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
    semaphore.handleType = kSemaphoreHandle;
    VkExternalSemaphoreProperties semaphore_properties{};
    semaphore_properties.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;
    vkGetPhysicalDeviceExternalSemaphoreProperties(
        device, &semaphore, &semaphore_properties
    );
    return (buffer_properties.externalMemoryProperties.externalMemoryFeatures &
            VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0 &&
           (semaphore_properties.externalSemaphoreFeatures &
            VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) != 0;
}

std::optional<DeviceChoice> InspectDevice(
    VkPhysicalDevice device, CUdevice cuda, int cuda_ordinal, const CUuuid& uuid
) {
    VkPhysicalDeviceIDProperties identity{};
    identity.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &identity;
    vkGetPhysicalDeviceProperties2(device, &properties);
    static_assert(sizeof(uuid.bytes) == VK_UUID_SIZE);
    if (std::memcmp(identity.deviceUUID, uuid.bytes, VK_UUID_SIZE) != 0 ||
        properties.properties.apiVersion < VK_API_VERSION_1_1)
        return std::nullopt;

    const auto family = ComputeQueueFamily(device);
    const auto extensions = DeviceExtensions(device);
    const auto has_extension = [&](const char* name) {
        return std::ranges::any_of(extensions, [&](const auto& extension) {
            return std::strcmp(extension.extensionName, name) == 0;
        });
    };
    if (!family || !has_extension(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) ||
        !has_extension(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME) ||
        !SupportsInteropHandles(device))
        return std::nullopt;
    return DeviceChoice{
        device, cuda, cuda_ordinal, *family,
        has_extension(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)};
}

DeviceChoice ChooseDevice(VkInstance instance) {
    std::uint32_t count = 0;
    CheckVk(
        vkEnumeratePhysicalDevices(instance, &count, nullptr),
        "count physical devices"
    );
    std::vector<VkPhysicalDevice> devices(count);
    CheckVk(
        vkEnumeratePhysicalDevices(instance, &count, devices.data()),
        "enumerate physical devices"
    );
    devices.resize(count);
    int cuda_count = 0;
    CheckCuda(cuDeviceGetCount(&cuda_count), "count devices");
    for (int index = 0; index < cuda_count; ++index) {
        CUdevice cuda = 0;
        CUuuid uuid{};
        CheckCuda(cuDeviceGet(&cuda, index), "get device");
        CheckCuda(cuDeviceGetUuid(&uuid, cuda), "get device UUID");
        for (VkPhysicalDevice device : devices) {
            if (auto choice = InspectDevice(device, cuda, index, uuid))
                return *choice;
        }
    }
    throw std::runtime_error(
        "NVDEC requires a matching NVIDIA CUDA and Vulkan device with opaque "
        "FD memory and semaphore export support"
    );
}

void CreateDevice(const DeviceChoice& choice, VulkanDevice& vulkan) {
    VkPhysicalDeviceShaderFloat16Int8Features float16{};
    float16.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    VkPhysicalDevice16BitStorageFeatures storage16{};
    storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    if (choice.float16_extension) storage16.pNext = &float16;
    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &storage16;
    vkGetPhysicalDeviceFeatures2(choice.physical, &features);
    storage16.uniformAndStorageBuffer16BitAccess = VK_FALSE;
    storage16.storagePushConstant16 = VK_FALSE;
    storage16.storageInputOutput16 = VK_FALSE;
    float16.shaderInt8 = VK_FALSE;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue{};
    queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue.queueFamilyIndex = choice.queue_family;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    const std::array extensions{VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                                VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
                                VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME};
    VkDeviceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.pNext = &storage16;
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue;
    info.enabledExtensionCount = choice.float16_extension ? 3u : 2u;
    info.ppEnabledExtensionNames = extensions.data();
    CheckVk(
        vkCreateDevice(choice.physical, &info, nullptr, &vulkan.device),
        "create device"
    );
    vulkan.physical_device = choice.physical;
    vulkan.queue_family = choice.queue_family;
    vulkan.storage_buffer_16bit_access = storage16.storageBuffer16BitAccess;
    vulkan.shader_float16 = float16.shaderFloat16;
    vkGetDeviceQueue(vulkan.device, vulkan.queue_family, 0, &vulkan.queue);
}

struct ExternalSemaphore {
    VkSemaphore vulkan = VK_NULL_HANDLE;
    CUexternalSemaphore cuda = nullptr;
};

void CreateSemaphore(
    VkDevice device, PFN_vkGetSemaphoreFdKHR export_fd,
    ExternalSemaphore& semaphore
) {
    VkExportSemaphoreCreateInfo export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    export_info.handleTypes = kSemaphoreHandle;
    VkSemaphoreCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    info.pNext = &export_info;
    CheckVk(
        vkCreateSemaphore(device, &info, nullptr, &semaphore.vulkan),
        "create external semaphore"
    );
    VkSemaphoreGetFdInfoKHR fd_info{};
    fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    fd_info.semaphore = semaphore.vulkan;
    fd_info.handleType = kSemaphoreHandle;
    int fd = -1;
    CheckVk(export_fd(device, &fd_info, &fd), "export semaphore FD");
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC handle{};
    handle.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD;
    handle.handle.fd = fd;
    CheckImport(
        cuImportExternalSemaphore(&semaphore.cuda, &handle), fd,
        "import semaphore FD"
    );
}

std::uint32_t DeviceLocalType(VkPhysicalDevice physical, std::uint32_t allowed) {
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        if ((allowed & (1u << i)) != 0 &&
            (memory.memoryTypes[i].propertyFlags &
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0)
            return i;
    }
    throw std::runtime_error("NVDEC Vulkan: no device-local frame memory type");
}

void ValidateSource(
    Size size, const ImageFormat& format, const NvdecSource& source
) {
    // Color metadata is interpreted by the metric after the CLI overrides.
    if (format.family != ColorFamily::Yuv || format.alpha != AlphaType::None ||
        (format.type != PixelType::Byte && format.type != PixelType::Word) ||
        size.width == 0 || size.height == 0 ||
        size.width > std::numeric_limits<unsigned>::max() ||
        size.height > std::numeric_limits<unsigned>::max()) {
        throw std::invalid_argument("invalid NVDEC YUV frame format or size");
    }
    const auto layout = internal::PlaneLayout(format, size);
    const auto storage_bits = layout[0].sample_bytes * 8;
    const auto depth = internal::EffectiveDepth(format);
    if (depth > storage_bits || source.shift > storage_bits - depth) {
        throw std::invalid_argument("invalid NVDEC sample bit alignment");
    }
    const std::size_t present_planes = source.semiplanar ? 2u : 3u;
    for (std::size_t i = 0; i < present_planes; ++i) {
        const auto& plane = layout[i];
        const std::size_t row_bytes = plane.width * plane.sample_bytes *
                                      ((i == 1 && source.semiplanar) ? 2u : 1u);
        if (source.data[i] == 0 || source.stride[i] < row_bytes ||
            source.stride[i] % plane.sample_bytes != 0) {
            throw std::invalid_argument("invalid NVDEC frame pointer or stride");
        }
    }
}

void LaunchPack(
    CUfunction kernel, CUdeviceptr dst, Size size, const ImageFormat& format,
    NvdecSource source
) {
    auto width = static_cast<unsigned>(size.width);
    auto height = static_cast<unsigned>(size.height);
    unsigned subsample_w = format.subsample_w;
    unsigned subsample_h = format.subsample_h;
    unsigned sample_bytes = format.type == PixelType::Byte ? 1u : 2u;
    unsigned semiplanar = source.semiplanar ? 1u : 0u;
    void* args[]{&dst,
                 &source.data[0],
                 &source.data[1],
                 &source.data[2],
                 &source.stride[0],
                 &source.stride[1],
                 &source.stride[2],
                 &width,
                 &height,
                 &subsample_w,
                 &subsample_h,
                 &sample_bytes,
                 &semiplanar,
                 &source.shift};
    constexpr unsigned threads = 128;
    const std::size_t words = (size.width * size.height * sample_bytes + 3) / 4;
    const auto groups = static_cast<unsigned>((words + threads - 1) / threads);
    CheckCuda(
        cuLaunchKernel(
            kernel, groups, 3, 1, threads, 1, 1, 0, source.stream, args, nullptr
        ),
        "launch frame packing kernel"
    );
}

}  // namespace

CurrentCudaContext::CurrentCudaContext(CUcontext context) {
    CheckCuda(cuCtxPushCurrent(context), "push primary context");
}

CurrentCudaContext::~CurrentCudaContext() {
    CUcontext context = nullptr;
    (void)cuCtxPopCurrent(&context);
}

struct NvdecBuffer::Impl {
    VkDevice device = VK_NULL_HANDLE;
    CUcontext context = nullptr;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    CUexternalMemory external = nullptr;
    CUdeviceptr pointer = 0;
    VkDeviceSize bytes = 0;
    bool first_use = true;

    ~Impl() {
        if (external != nullptr && cuCtxPushCurrent(context) == CUDA_SUCCESS) {
            if (pointer != 0) (void)cuMemFree(pointer);
            (void)cuDestroyExternalMemory(external);
            CUcontext popped = nullptr;
            (void)cuCtxPopCurrent(&popped);
        }
        if (buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, buffer, nullptr);
        if (memory != VK_NULL_HANDLE) vkFreeMemory(device, memory, nullptr);
    }

    void Allocate(
        const VulkanDevice& vulkan, CUcontext cuda_context,
        PFN_vkGetMemoryFdKHR export_fd, VkDeviceSize size
    ) {
        device = vulkan.device;
        context = cuda_context;
        bytes = size;
        VkExternalMemoryBufferCreateInfo external_info{};
        external_info.sType =
            VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        external_info.handleTypes = kMemoryHandle;
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.pNext = &external_info;
        buffer_info.size = bytes;
        buffer_info.usage = kBufferUsage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        CheckVk(
            vkCreateBuffer(device, &buffer_info, nullptr, &buffer),
            "create external frame buffer"
        );

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, buffer, &requirements);
        VkMemoryDedicatedAllocateInfo dedicated{};
        dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        dedicated.buffer = buffer;
        VkExportMemoryAllocateInfo export_info{};
        export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        export_info.pNext = &dedicated;
        export_info.handleTypes = kMemoryHandle;
        VkMemoryAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        info.pNext = &export_info;
        info.allocationSize = requirements.size;
        info.memoryTypeIndex =
            DeviceLocalType(vulkan.physical_device, requirements.memoryTypeBits);
        CheckVk(
            vkAllocateMemory(device, &info, nullptr, &memory),
            "allocate external frame memory"
        );
        CheckVk(
            vkBindBufferMemory(device, buffer, memory, 0), "bind frame buffer"
        );

        VkMemoryGetFdInfoKHR fd_info{};
        fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        fd_info.memory = memory;
        fd_info.handleType = kMemoryHandle;
        int fd = -1;
        CheckVk(export_fd(device, &fd_info, &fd), "export frame memory FD");
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC handle{};
        handle.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        handle.handle.fd = fd;
        handle.size = requirements.size;
        handle.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
        CheckImport(
            cuImportExternalMemory(&external, &handle), fd,
            "import frame memory FD"
        );
        CUDA_EXTERNAL_MEMORY_BUFFER_DESC mapping{};
        mapping.size = bytes;
        CheckCuda(
            cuExternalMemoryGetMappedBuffer(&pointer, external, &mapping),
            "map frame buffer"
        );
    }
};

struct NvdecInterop::Impl {
    VkInstance instance = VK_NULL_HANDLE;
    VulkanDevice vulkan{};
    CUdevice cuda_device = 0;
    int cuda_ordinal = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    ExternalSemaphore released;
    ExternalSemaphore packed;
    PFN_vkGetMemoryFdKHR export_memory_fd = nullptr;
    VkCommandPool commands = VK_NULL_HANDLE;
    std::array<VkCommandBuffer, 2> transfers{};
    VkFence acquired = VK_NULL_HANDLE;
    VkDeviceSize max_buffer_range = 0;
    bool failed = false;

    ~Impl() {
        const VkDevice device = vulkan.device;
        if (device != VK_NULL_HANDLE) (void)vkDeviceWaitIdle(device);
        if (context != nullptr) {
            if (cuCtxPushCurrent(context) == CUDA_SUCCESS) {
                if (module != nullptr) (void)cuModuleUnload(module);
                if (packed.cuda != nullptr)
                    (void)cuDestroyExternalSemaphore(packed.cuda);
                if (released.cuda != nullptr)
                    (void)cuDestroyExternalSemaphore(released.cuda);
                CUcontext popped = nullptr;
                (void)cuCtxPopCurrent(&popped);
            }
            (void)cuDevicePrimaryCtxRelease(cuda_device);
        }
        if (acquired != VK_NULL_HANDLE)
            vkDestroyFence(device, acquired, nullptr);
        if (commands != VK_NULL_HANDLE)
            vkDestroyCommandPool(device, commands, nullptr);
        if (packed.vulkan != VK_NULL_HANDLE)
            vkDestroySemaphore(device, packed.vulkan, nullptr);
        if (released.vulkan != VK_NULL_HANDLE)
            vkDestroySemaphore(device, released.vulkan, nullptr);
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
    }

    void Init() {
        CheckCuda(cuInit(0), "initialize driver");
        VkApplicationInfo application{};
        application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        application.pApplicationName = "vqm NVDEC";
        application.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.pApplicationInfo = &application;
        CheckVk(vkCreateInstance(&info, nullptr, &instance), "create instance");
        const auto choice = ChooseDevice(instance);
        CreateDevice(choice, vulkan);
        cuda_device = choice.cuda;
        cuda_ordinal = choice.cuda_ordinal;
        CheckCuda(
            cuDevicePrimaryCtxRetain(&context, cuda_device),
            "retain primary context"
        );
        const CurrentCudaContext current(context);
        CheckCuda(
            cuModuleLoadData(&module, kNvdecPackPtx), "load frame packing PTX"
        );
        CheckCuda(
            cuModuleGetFunction(&kernel, module, "PackNvdecFrame"),
            "get frame packing kernel"
        );
        export_memory_fd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
            vkGetDeviceProcAddr(vulkan.device, "vkGetMemoryFdKHR")
        );
        const auto export_semaphore_fd =
            reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
                vkGetDeviceProcAddr(vulkan.device, "vkGetSemaphoreFdKHR")
            );
        if (export_memory_fd == nullptr || export_semaphore_fd == nullptr)
            throw std::runtime_error("NVDEC Vulkan: missing opaque FD export");
        CreateSemaphore(vulkan.device, export_semaphore_fd, released);
        CreateSemaphore(vulkan.device, export_semaphore_fd, packed);
        CreateCommands();
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(vulkan.physical_device, &properties);
        max_buffer_range = properties.limits.maxStorageBufferRange;
    }

    void CreateCommands() {
        VkCommandPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = vulkan.queue_family;
        CheckVk(
            vkCreateCommandPool(vulkan.device, &pool, nullptr, &commands),
            "create ownership transfer command pool"
        );
        VkCommandBufferAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.commandPool = commands;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = static_cast<std::uint32_t>(transfers.size());
        CheckVk(
            vkAllocateCommandBuffers(vulkan.device, &info, transfers.data()),
            "allocate ownership transfer commands"
        );
        VkFenceCreateInfo fence{};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        CheckVk(
            vkCreateFence(vulkan.device, &fence, nullptr, &acquired),
            "create ownership transfer fence"
        );
    }

    void RecordOwnership(
        VkCommandBuffer command, VkBuffer buffer, bool acquire, bool first_use
    ) const {
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CheckVk(
            vkBeginCommandBuffer(command, &begin), "begin ownership transfer"
        );
        if (first_use) {
            // Establish initial Vulkan ownership before releasing the buffer.
            // This also initializes the alignment padding between planes.
            vkCmdFillBuffer(command, buffer, 0, VK_WHOLE_SIZE, 0);
        }
        const VkPipelineStageFlags producer_stage =
            first_use ? VK_PIPELINE_STAGE_TRANSFER_BIT
                      : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        const VkAccessFlags producer_access =
            first_use ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = acquire ? 0u : producer_access;
        barrier.dstAccessMask =
            acquire ? VkAccessFlags{VK_ACCESS_SHADER_READ_BIT} : 0u;
        barrier.srcQueueFamilyIndex =
            acquire ? VK_QUEUE_FAMILY_EXTERNAL : vulkan.queue_family;
        barrier.dstQueueFamilyIndex =
            acquire ? vulkan.queue_family : VK_QUEUE_FAMILY_EXTERNAL;
        barrier.buffer = buffer;
        barrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(
            command,
            acquire ? VkPipelineStageFlags{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT}
                    : producer_stage,
            acquire ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                    : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 1, &barrier, 0, nullptr
        );
        CheckVk(vkEndCommandBuffer(command), "end ownership transfer");
    }

    void ReleaseToCuda(VkBuffer buffer, bool first_use) const {
        RecordOwnership(transfers[0], buffer, false, first_use);
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &transfers[0];
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &released.vulkan;
        CheckVk(
            vkQueueSubmit(vulkan.queue, 1, &submit, VK_NULL_HANDLE),
            "release frame buffer to CUDA"
        );
    }

    void AcquireForVulkan(VkBuffer buffer) const {
        RecordOwnership(transfers[1], buffer, true, false);
        constexpr VkPipelineStageFlags wait = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &packed.vulkan;
        submit.pWaitDstStageMask = &wait;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &transfers[1];
        CheckVk(
            vkQueueSubmit(vulkan.queue, 1, &submit, acquired),
            "acquire packed frame buffer"
        );
        CheckVk(
            vkWaitForFences(vulkan.device, 1, &acquired, VK_TRUE, UINT64_MAX),
            "wait for frame ownership"
        );
        CheckVk(
            vkResetFences(vulkan.device, 1, &acquired), "reset transfer fence"
        );
    }

    void Pack(
        NvdecBuffer::Impl& frame, Size size, const ImageFormat& format,
        const NvdecSource& source
    ) {
        try {
            ReleaseToCuda(frame.buffer, frame.first_use);
            CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS wait{};
            CheckCuda(
                cuWaitExternalSemaphoresAsync(
                    &released.cuda, &wait, 1, source.stream
                ),
                "wait for Vulkan frame buffer"
            );
            LaunchPack(kernel, frame.pointer, size, format, source);
            CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal{};
            CheckCuda(
                cuSignalExternalSemaphoresAsync(
                    &packed.cuda, &signal, 1, source.stream
                ),
                "signal packed frame buffer"
            );
            // Check for asynchronous CUDA failures before submitting a Vulkan
            // wait that would otherwise be stranded on an unsignaled semaphore.
            CheckCuda(
                cuStreamSynchronize(source.stream), "complete frame packing"
            );
            AcquireForVulkan(frame.buffer);
            frame.first_use = false;
        } catch (...) {
            failed = true;
            (void)cuStreamSynchronize(source.stream);
            (void)vkQueueWaitIdle(vulkan.queue);
            throw;
        }
    }
};

NvdecBuffer::NvdecBuffer() = default;
NvdecBuffer::~NvdecBuffer() = default;

NvdecInterop::NvdecInterop() : impl_(std::make_unique<Impl>()) { impl_->Init(); }
NvdecInterop::~NvdecInterop() = default;

const VulkanDevice& NvdecInterop::vulkan() const noexcept {
    return impl_->vulkan;
}
CUcontext NvdecInterop::context() const noexcept { return impl_->context; }
int NvdecInterop::cuda_ordinal() const noexcept { return impl_->cuda_ordinal; }

VulkanBufferView NvdecInterop::Pack(
    NvdecBuffer& dst, Size size, const ImageFormat& format,
    const NvdecSource& src
) {
    if (impl_->failed)
        throw std::runtime_error("NVDEC frame interop failed earlier");
    ValidateSource(size, format, src);
    const VkDeviceSize bytes = RequiredBytes(format, size);
    if (bytes > impl_->max_buffer_range)
        throw std::invalid_argument(
            "NVDEC frame exceeds Vulkan storage buffer range"
        );
    if (dst.impl_ && dst.impl_->device != impl_->vulkan.device)
        throw std::invalid_argument(
            "NVDEC frame buffer belongs to another device"
        );
    const CurrentCudaContext current(impl_->context);
    if (!dst.impl_ || dst.impl_->bytes != bytes) {
        auto frame = std::make_unique<NvdecBuffer::Impl>();
        frame->Allocate(
            impl_->vulkan, impl_->context, impl_->export_memory_fd, bytes
        );
        dst.impl_ = std::move(frame);
    }
    impl_->Pack(*dst.impl_, size, format, src);
    return {dst.impl_->buffer, 0, bytes, size, format};
}

}  // namespace vqm::cli
