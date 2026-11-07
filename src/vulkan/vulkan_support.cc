// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

#include "vqm_vulkan.h"
#include "vulkan_support.h"

// The process-wide Vulkan device. One instance, one discrete (preferred)
// physical device with a compute queue, one queue. Creation is lazy and
// failure is reported as a null context, so the library reports "no usable
// compute device" instead of aborting.

namespace vqm::internal {
namespace {

std::once_flag g_init;
std::shared_ptr<VulkanContext> g_context;

struct DeviceChoice {
    VkPhysicalDevice device = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    std::uint32_t timestamp_valid_bits = 0;
    bool sixteen_bit_storage = false;
    bool shader_float16 = false;
    VkPhysicalDeviceProperties properties{};
};

bool HasDeviceExtension(VkPhysicalDevice device, const char* name) {
    std::uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr) !=
        VK_SUCCESS)
        return false;
    std::vector<VkExtensionProperties> extensions(count);
    if (vkEnumerateDeviceExtensionProperties(
            device, nullptr, &count, extensions.data()
        ) != VK_SUCCESS)
        return false;
    for (std::uint32_t i = 0; i < count; ++i) {
        if (std::strcmp(extensions[i].extensionName, name) == 0) return true;
    }
    return false;
}

// The shared baseline requires 256 invocations per workgroup and 16 KiB of
// shared memory. Metric layouts check the limits of their own shaders, including
// per-axis workgroup sizes and any optional variants.
std::optional<DeviceChoice> Inspect(VkPhysicalDevice device) {
    DeviceChoice choice;
    choice.device = device;
    vkGetPhysicalDeviceProperties(device, &choice.properties);
    if (choice.properties.apiVersion < VK_API_VERSION_1_1) return std::nullopt;
    if (choice.properties.limits.maxComputeWorkGroupInvocations < 256u) {
        return std::nullopt;
    }
    if (choice.properties.limits.maxComputeSharedMemorySize < 16384u) {
        return std::nullopt;
    }

    VkPhysicalDevice16BitStorageFeatures storage16{};
    storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    // The owned instance requests Vulkan 1.1, so explicitly enable the KHR
    // extension before using Float16, even on a newer physical device.
    const bool float16_extension =
        HasDeviceExtension(device, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
    VkPhysicalDeviceShaderFloat16Int8Features float16{};
    float16.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    if (float16_extension) storage16.pNext = &float16;
    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &storage16;
    vkGetPhysicalDeviceFeatures2(device, &features);
    choice.sixteen_bit_storage = storage16.storageBuffer16BitAccess == VK_TRUE;
    choice.shader_float16 = float16.shaderFloat16 == VK_TRUE;

    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        device, &family_count, families.data()
    );
    for (std::uint32_t i = 0; i < family_count; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
        choice.queue_family = i;
        choice.timestamp_valid_bits = families[i].timestampValidBits;
        return choice;
    }
    return std::nullopt;
}

std::optional<DeviceChoice> PickDevice(
    const std::vector<VkPhysicalDevice>& devices, bool discrete_only
) {
    for (VkPhysicalDevice device : devices) {
        const std::optional<DeviceChoice> choice = Inspect(device);
        if (!choice) continue;
        const bool discrete = choice->properties.deviceType ==
                              VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        if (discrete_only && !discrete) continue;
        return choice;
    }
    return std::nullopt;
}

std::shared_ptr<VulkanContext> CreateContext() {
    auto context = std::make_shared<VulkanContext>();

    VkApplicationInfo application{};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "vqm";
    application.pEngineName = "vqm";
    application.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application;
    if (vkCreateInstance(&instance_info, nullptr, &context->instance) !=
        VK_SUCCESS) {
        return nullptr;
    }

    std::uint32_t device_count = 0;
    if (vkEnumeratePhysicalDevices(context->instance, &device_count, nullptr) !=
            VK_SUCCESS ||
        device_count == 0) {
        return nullptr;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    if (vkEnumeratePhysicalDevices(
            context->instance, &device_count, devices.data()
        ) != VK_SUCCESS) {
        return nullptr;
    }

    std::optional<DeviceChoice> choice = PickDevice(devices, true);
    if (!choice) choice = PickDevice(devices, false);
    if (!choice) return nullptr;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = choice->queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    // Requested unconditionally when the device has it, so one process-wide
    // device serves both plane modes; the flag is left off otherwise and the
    // half-plane pipeline is then rejected where it is created.
    VkPhysicalDevice16BitStorageFeatures storage16{};
    storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    storage16.storageBuffer16BitAccess =
        choice->sixteen_bit_storage ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceShaderFloat16Int8Features float16{};
    float16.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    float16.shaderFloat16 = choice->shader_float16 ? VK_TRUE : VK_FALSE;
    const char* float16_extension = VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME;
    if (choice->shader_float16) {
        storage16.pNext = &float16;
        device_info.enabledExtensionCount = 1;
        device_info.ppEnabledExtensionNames = &float16_extension;
    }
    device_info.pNext = &storage16;
    if (vkCreateDevice(
            choice->device, &device_info, nullptr, &context->device
        ) != VK_SUCCESS) {
        return nullptr;
    }

    context->physical_device = choice->device;
    context->queue_family = choice->queue_family;
    context->limits = choice->properties.limits;
    vkGetDeviceQueue(context->device, choice->queue_family, 0, &context->queue);
    vkGetPhysicalDeviceMemoryProperties(
        choice->device, &context->memory_properties
    );
    context->timestamp_period = context->limits.timestampPeriod;
    context->timestamp_valid_bits = choice->timestamp_valid_bits;
    context->timestamps =
        choice->timestamp_valid_bits > 0 && context->timestamp_period > 0.0f;
    context->sixteen_bit_storage = choice->sixteen_bit_storage;
    context->shader_float16 = choice->shader_float16;
    return context;
}

// Destroys the process-wide device and instance from an exit handler rather
// than from the singleton's static destructor. The order matters: a static
// destructor runs at the very end of exit, by which point the driver's own
// teardown has already run, and on the NVIDIA 615.78 driver without a DISPLAY
// connection (every headless run: CI, containers, `env -i`) vkDestroyDevice
// then dispatches into unmapped driver code -- a SIGSEGV *after* all output,
// exit code 139. Reproduced with the pre-fusion backend too, so it predates the
// fused prepare pass. This fallback runs before the singleton's static
// destructor. Later driver initialization can register additional cleanup;
// the benchmark and test harness therefore release their contexts explicitly
// before exit. A later instance can also leave validation-layer teardown ahead
// of this fallback, even when every pipeline has already been destroyed.
void DestroyContext() { g_context.reset(); }

}  // namespace

VulkanContext::~VulkanContext() {
    if (owns_device) {
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
    }
}

std::shared_ptr<VulkanContext> AcquireVulkanContext() {
    std::call_once(g_init, [] {
        g_context = CreateContext();
        if (g_context) std::atexit(DestroyContext);
    });
    return g_context;
}

void ShutdownDefaultVulkanContext() {
    if (g_context && g_context.use_count() != 1)
        throw std::logic_error(
            "vulkan: default context shutdown requires all pipelines to be gone"
        );
    // A driver may initialize more components after device creation. Releasing
    // the final owner here avoids relying on their atexit registration order.
    // The existing exit handler remains an idempotent fallback.
    g_context.reset();
}

std::shared_ptr<VulkanContext> BorrowVulkanContext(const VulkanDevice& device) {
    if (device.physical_device == VK_NULL_HANDLE ||
        device.device == VK_NULL_HANDLE || device.queue == VK_NULL_HANDLE) {
        throw std::invalid_argument(
            "vulkan: borrowed device handles must not be null"
        );
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device.physical_device, &properties);
    if (properties.apiVersion < VK_API_VERSION_1_1) {
        throw std::invalid_argument(
            "vulkan: borrowed device requires Vulkan 1.1"
        );
    }
    if (properties.limits.maxComputeWorkGroupInvocations < 256u ||
        properties.limits.maxComputeSharedMemorySize < 16384u) {
        throw std::invalid_argument(
            "vulkan: borrowed device lacks required compute limits"
        );
    }

    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
        device.physical_device, &family_count, nullptr
    );
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        device.physical_device, &family_count, families.data()
    );
    if (device.queue_family >= family_count ||
        !(families[device.queue_family].queueFlags & VK_QUEUE_COMPUTE_BIT) ||
        families[device.queue_family].queueCount == 0) {
        throw std::invalid_argument(
            "vulkan: borrowed queue family must support compute"
        );
    }

    if (device.storage_buffer_16bit_access) {
        VkPhysicalDevice16BitStorageFeatures storage16{};
        storage16.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        VkPhysicalDeviceFeatures2 features{};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.pNext = &storage16;
        vkGetPhysicalDeviceFeatures2(device.physical_device, &features);
        if (storage16.storageBuffer16BitAccess != VK_TRUE) {
            throw std::invalid_argument(
                "vulkan: borrowed device does not support 16-bit storage"
            );
        }
    }

    if (device.shader_float16) {
        if (properties.apiVersion < VK_API_VERSION_1_2 &&
            !HasDeviceExtension(
                device.physical_device, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME
            )) {
            throw std::invalid_argument(
                "vulkan: borrowed device does not support shaderFloat16"
            );
        }
        VkPhysicalDeviceShaderFloat16Int8Features float16{};
        float16.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
        VkPhysicalDeviceFeatures2 features{};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.pNext = &float16;
        vkGetPhysicalDeviceFeatures2(device.physical_device, &features);
        if (float16.shaderFloat16 != VK_TRUE) {
            throw std::invalid_argument(
                "vulkan: borrowed device does not support shaderFloat16"
            );
        }
    }

    auto context = std::make_shared<VulkanContext>();
    context->owns_device = false;
    context->physical_device = device.physical_device;
    context->device = device.device;
    context->queue = device.queue;
    context->queue_family = device.queue_family;
    context->limits = properties.limits;
    vkGetPhysicalDeviceMemoryProperties(
        device.physical_device, &context->memory_properties
    );
    context->timestamp_period = properties.limits.timestampPeriod;
    context->timestamp_valid_bits =
        families[device.queue_family].timestampValidBits;
    context->timestamps = families[device.queue_family].timestampValidBits > 0 &&
                          context->timestamp_period > 0.0f;
    // Vulkan exposes physical support but not the features enabled on an
    // existing VkDevice. Never infer enabled features from physical support.
    context->sixteen_bit_storage = device.storage_buffer_16bit_access;
    context->shader_float16 = device.shader_float16;
    return context;
}

std::shared_ptr<VulkanContext> AcquireVulkanContext(
    const VulkanDevice* borrowed_device
) {
    return borrowed_device ? BorrowVulkanContext(*borrowed_device)
                           : AcquireVulkanContext();
}

std::shared_ptr<VulkanContext> RequireVulkanContext(
    const VulkanDevice* borrowed_device
) {
    auto context = AcquireVulkanContext(borrowed_device);
    if (!context) throw std::runtime_error("vulkan: no usable compute device");
    return context;
}

}  // namespace vqm::internal
