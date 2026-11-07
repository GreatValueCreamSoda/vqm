// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// Standalone correctness probe. It contains no performance timers.
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(
            std::string(operation) + ": VkResult " + std::to_string(result)
        );
}

bool HasExtension(VkPhysicalDevice device, const char* name) {
    std::uint32_t count = 0;
    Check(
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr),
        "enumerate extension count"
    );
    std::vector<VkExtensionProperties> extensions(count);
    Check(
        vkEnumerateDeviceExtensionProperties(
            device, nullptr, &count, extensions.data()
        ),
        "enumerate extensions"
    );
    return std::any_of(extensions.begin(), extensions.end(), [&](const auto& e) {
        return std::strcmp(e.extensionName, name) == 0;
    });
}

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    float* mapped = nullptr;
};

class ProbeDevice {
public:
    ProbeDevice() { Initialize(); }
    ~ProbeDevice() {
        if (device) {
            vkDeviceWaitIdle(device);
            if (commands) vkDestroyCommandPool(device, commands, nullptr);
            if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
            if (descriptors)
                vkDestroyDescriptorSetLayout(device, descriptors, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
    ProbeDevice(const ProbeDevice&) = delete;
    ProbeDevice& operator=(const ProbeDevice&) = delete;

    Buffer Allocate(std::size_t bytes) const {
        Buffer buffer;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(
            vkCreateBuffer(device, &info, nullptr, &buffer.handle),
            "create buffer"
        );
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, buffer.handle, &requirements);
        VkPhysicalDeviceMemoryProperties memory;
        vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        std::uint32_t type = memory.memoryTypeCount;
        for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            const auto required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (memory.memoryTypes[i].propertyFlags & required) == required) {
                type = i;
                break;
            }
        }
        if (type == memory.memoryTypeCount)
            throw std::runtime_error("no host-visible coherent buffer memory");
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        Check(
            vkAllocateMemory(device, &allocation, nullptr, &buffer.memory),
            "allocate memory"
        );
        Check(
            vkBindBufferMemory(device, buffer.handle, buffer.memory, 0),
            "bind buffer"
        );
        void* mapped = nullptr;
        Check(
            vkMapMemory(device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &mapped),
            "map buffer"
        );
        buffer.mapped = static_cast<float*>(mapped);
        return buffer;
    }

    void Release(Buffer& buffer) const {
        if (buffer.mapped) vkUnmapMemory(device, buffer.memory);
        if (buffer.handle) vkDestroyBuffer(device, buffer.handle, nullptr);
        if (buffer.memory) vkFreeMemory(device, buffer.memory, nullptr);
        buffer = {};
    }

    std::vector<float> Run(
        const std::string& path, const std::vector<float>& inputs,
        std::size_t output_count
    ) const {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("missing shader " + path);
        const std::size_t size = static_cast<std::size_t>(file.tellg());
        if (size % 4) throw std::runtime_error("invalid SPIR-V byte count");
        std::vector<std::uint32_t> code(size / 4);
        file.seekg(0);
        file.read(reinterpret_cast<char*>(code.data()), size);
        VkShaderModuleCreateInfo shader_info{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader_info.codeSize = size;
        shader_info.pCode = code.data();
        VkShaderModule shader;
        Check(
            vkCreateShaderModule(device, &shader_info, nullptr, &shader),
            "create shader"
        );
        VkComputePipelineCreateInfo pipeline_info{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.flags =
            VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        pipeline_info.stage.module = shader;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = layout;
        VkPipeline pipeline;
        Check(
            vkCreateComputePipelines(
                device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline
            ),
            "create pipeline"
        );
        vkDestroyShaderModule(device, shader, nullptr);
        Buffer input = Allocate(inputs.size() * sizeof(float));
        Buffer output = Allocate(output_count * sizeof(float));
        std::copy(inputs.begin(), inputs.end(), input.mapped);
        std::fill(
            output.mapped, output.mapped + output_count,
            std::numeric_limits<float>::quiet_NaN()
        );
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
        VkDescriptorPoolCreateInfo pool_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        VkDescriptorPool pool;
        Check(
            vkCreateDescriptorPool(device, &pool_info, nullptr, &pool),
            "descriptor pool"
        );
        VkDescriptorSetAllocateInfo allocation{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = pool;
        allocation.descriptorSetCount = 1;
        allocation.pSetLayouts = &descriptors;
        VkDescriptorSet set;
        Check(
            vkAllocateDescriptorSets(device, &allocation, &set), "descriptor set"
        );
        const VkDescriptorBufferInfo buffers[] = {
            {input.handle, 0, inputs.size() * sizeof(float)},
            {output.handle, 0, output_count * sizeof(float)}};
        std::array<VkWriteDescriptorSet, 2> writes{};
        for (std::uint32_t i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffers[i];
        }
        vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
        VkCommandBufferAllocateInfo command_info{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_info.commandPool = commands;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        VkCommandBuffer command;
        Check(
            vkAllocateCommandBuffers(device, &command_info, &command),
            "command buffer"
        );
        VkCommandBufferBeginInfo begin{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        Check(vkBeginCommandBuffer(command, &begin), "begin command");
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(
            command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0,
            nullptr
        );
        vkCmdDispatch(command, 1, 1, 1);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(
            command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr
        );
        Check(vkEndCommandBuffer(command), "end command");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        Check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "queue submit");
        Check(vkQueueWaitIdle(queue), "queue wait");
        std::vector<float> result(output.mapped, output.mapped + output_count);
        vkFreeCommandBuffers(device, commands, 1, &command);
        vkDestroyDescriptorPool(device, pool, nullptr);
        Release(input);
        Release(output);
        vkDestroyPipeline(device, pipeline, nullptr);
        return result;
    }

    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceSubgroupProperties subgroup{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceCooperativeMatrixPropertiesKHR stages{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR};
    std::vector<VkCooperativeMatrixPropertiesKHR> shapes;

private:
    void Initialize() {
        VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application.pApplicationName = "VQM cooperative matrix precision probe";
        application.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        info.pApplicationInfo = &application;
        Check(vkCreateInstance(&info, nullptr, &instance), "create instance");
        std::uint32_t count = 0;
        Check(
            vkEnumeratePhysicalDevices(instance, &count, nullptr), "device count"
        );
        std::vector<VkPhysicalDevice> devices(count);
        Check(
            vkEnumeratePhysicalDevices(instance, &count, devices.data()),
            "devices"
        );
        for (auto candidate : devices) {
            VkPhysicalDeviceProperties candidate_properties;
            vkGetPhysicalDeviceProperties(candidate, &candidate_properties);
            if (candidate_properties.apiVersion >= VK_API_VERSION_1_1 &&
                HasExtension(
                    candidate, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
                )) {
                physical = candidate;
                properties = candidate_properties;
                if (properties.deviceType ==
                    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                    break;
            }
        }
        if (!physical) throw std::runtime_error("no cooperative-matrix device");
        for (auto extension : {VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
                               VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME,
                               VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME})
            if (!HasExtension(physical, extension))
                throw std::runtime_error(
                    std::string("missing extension ") + extension
                );
        subgroup.pNext = &stages;
        VkPhysicalDeviceProperties2 extended{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        extended.pNext = &subgroup;
        vkGetPhysicalDeviceProperties2(physical, &extended);
        if (subgroup.subgroupSize != 32 ||
            !(stages.cooperativeMatrixSupportedStages &
              VK_SHADER_STAGE_COMPUTE_BIT))
            throw std::runtime_error(
                "probe requires compute matrices and subgroup size 32"
            );
        const auto get_shapes = reinterpret_cast<
            PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            vkGetInstanceProcAddr(
                instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"
            )
        );
        if (!get_shapes)
            throw std::runtime_error("missing cooperative-matrix query");
        Check(get_shapes(physical, &count, nullptr), "matrix shape count");
        shapes.resize(count);
        for (auto& shape : shapes)
            shape.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        Check(get_shapes(physical, &count, shapes.data()), "matrix shapes");
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        VkPhysicalDeviceShaderFloat16Int8Features half{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        VkPhysicalDeviceVulkanMemoryModelFeatures memory{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES};
        VkPhysicalDeviceSubgroupSizeControlFeatures full_subgroups{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
        cooperative.pNext = &half;
        half.pNext = &memory;
        memory.pNext = &full_subgroups;
        VkPhysicalDeviceFeatures2 features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &cooperative;
        vkGetPhysicalDeviceFeatures2(physical, &features);
        if (!cooperative.cooperativeMatrix || !half.shaderFloat16 ||
            !memory.vulkanMemoryModel || !full_subgroups.computeFullSubgroups)
            throw std::runtime_error(
                "missing cooperative/FP16/memory-model feature"
            );
        cooperative.cooperativeMatrixRobustBufferAccess = VK_FALSE;
        half.shaderInt8 = VK_FALSE;
        memory.vulkanMemoryModelDeviceScope = VK_FALSE;
        memory.vulkanMemoryModelAvailabilityVisibilityChains = VK_FALSE;
        full_subgroups.subgroupSizeControl = VK_FALSE;
        std::uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(
            physical, &family_count, nullptr
        );
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(
            physical, &family_count, families.data()
        );
        std::uint32_t family = family_count;
        for (std::uint32_t i = 0; i < family_count; ++i)
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                family = i;
                break;
            }
        if (family == family_count) throw std::runtime_error("no compute queue");
        const float priority = 1;
        VkDeviceQueueCreateInfo queue_info{
            VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        const char* extensions[] = {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
                                    VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
                                    VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME,
                                    VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME};
        VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device_info.pNext = &cooperative;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.enabledExtensionCount = std::size(extensions);
        device_info.ppEnabledExtensionNames = extensions;
        Check(
            vkCreateDevice(physical, &device_info, nullptr, &device),
            "create device"
        );
        vkGetDeviceQueue(device, family, 0, &queue);
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        for (std::uint32_t i = 0; i < bindings.size(); ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo descriptor_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        descriptor_info.bindingCount = bindings.size();
        descriptor_info.pBindings = bindings.data();
        Check(
            vkCreateDescriptorSetLayout(
                device, &descriptor_info, nullptr, &descriptors
            ),
            "descriptor layout"
        );
        VkPipelineLayoutCreateInfo layout_info{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &descriptors;
        Check(
            vkCreatePipelineLayout(device, &layout_info, nullptr, &layout),
            "pipeline layout"
        );
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.queueFamilyIndex = family;
        Check(
            vkCreateCommandPool(device, &pool, nullptr, &commands),
            "command pool"
        );
    }

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
};

float Half(float value) {
    if (std::abs(value) >= 65520.0f)
        return std::copysign(std::numeric_limits<float>::infinity(), value);
    const int exponent =
        value == 0 ? -14 : std::max(-14, std::ilogb(std::abs(value)));
    const float step = std::ldexp(1.0f, exponent - 10);
    return std::nearbyint(value / step) * step;
}

std::vector<float> Inputs(
    const std::string& scene, unsigned m, unsigned n, unsigned k
) {
    std::vector<float> data(m * k + k * n);
    for (unsigned row = 0; row < m; ++row) {
        for (unsigned depth = 0; depth < k; ++depth) {
            float value = 100.5f;
            if (scene == "above-half-product-range") value = 512.0f;
            if (scene == "small-cancellation")
                value = depth % 4 < 2 ? 1.0009765625f : 1.0f;
            if (scene == "lost-operand-bits")
                value = 127.0f + (depth % 2) / 256.0f;
            if (scene == "mixed-sign")
                value = Half(
                    static_cast<int>((row * 37 + depth * 71 + 19) % 199) /
                        17.0f -
                    5.0f
                );
            data[row * k + depth] = value;
        }
    }
    for (unsigned depth = 0; depth < k; ++depth) {
        for (unsigned col = 0; col < n; ++col) {
            float value = 100.5f;
            if (scene == "cancellation") value = depth % 2 ? -100.25f : 100.5f;
            if (scene == "above-half-product-range") value = 512.0f;
            if (scene == "small-cancellation") {
                const float terms[] = {1.0009765625f, -1.0f, -1.0009765625f,
                                       1.0f};
                value = terms[depth % 4];
            }
            if (scene == "lost-operand-bits") value = depth % 2 ? 1.0f : -1.0f;
            if (scene == "mixed-sign")
                value = Half(
                    static_cast<int>((col * 23 + depth * 59 + 43) % 179) /
                        19.0f -
                    4.0f
                );
            data[m * k + depth * n + col] = value;
        }
    }
    return data;
}

void Number(double value) {
    if (std::isfinite(value))
        std::cout << value;
    else
        std::cout << "null";
}

bool Analyze(
    const std::string& scene, unsigned m, unsigned n, unsigned k,
    const std::vector<float>& inputs, const std::vector<float>& result
) {
    double maximum_error = 0, first_exact = 0, first_original = 0,
           first_half_product = 0;
    bool passed = true;
    for (unsigned row = 0; row < m; ++row) {
        for (unsigned col = 0; col < n; ++col) {
            double exact = 0, original = 0, magnitude = 0, half_product = 0;
            for (unsigned depth = 0; depth < k; ++depth) {
                const float a = inputs[row * k + depth];
                const float b = inputs[m * k + depth * n + col];
                const double product = static_cast<double>(Half(a)) * Half(b);
                original += static_cast<double>(a) * b;
                exact += product;
                magnitude += std::abs(product);
                half_product += Half(static_cast<float>(product));
            }
            const double value = result[row * n + col];
            const double error = std::abs(value - exact);
            maximum_error = std::max(maximum_error, error);
            // This is a probe acceptance bound, not a Vulkan precision
            // guarantee.
            const double tolerance =
                2 * k * std::numeric_limits<float>::epsilon() * magnitude + 1e-7;
            passed &= std::isfinite(value) && error <= tolerance;
            if (scene != "mixed-sign")
                passed &= value == static_cast<float>(exact);
            if (!row && !col) {
                first_exact = exact;
                first_original = original;
                first_half_product = half_product;
            }
        }
    }
    std::cout << "{\"scene\":\"" << scene << "\",\"shape\":[" << m << ',' << n
              << ',' << k << "],\"gpu_first\":" << result[0]
              << ",\"half_operand_exact_first\":" << first_exact
              << ",\"original_exact_first\":" << first_original
              << ",\"half_product_sum_first\":";
    Number(first_half_product);
    std::cout << ",\"max_absolute_error_to_half_operand_exact\":"
              << maximum_error
              << ",\"passed_probe_bound\":" << (passed ? "true" : "false")
              << '}';
    return passed;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: platform_probe SPIRV_DIRECTORY\n";
        return 2;
    }
    try {
        if (std::fegetround() != FE_TONEAREST)
            throw std::runtime_error(
                "probe CPU oracle requires round-to-nearest-even"
            );
        ProbeDevice probe;
        std::cout
            << std::setprecision(17) << "{\"device\":\""
            << probe.properties.deviceName
            << "\",\"vendor_id\":" << probe.properties.vendorID
            << ",\"device_id\":" << probe.properties.deviceID
            << ",\"api_version\":" << probe.properties.apiVersion
            << ",\"driver_version_raw\":" << probe.properties.driverVersion
            << ",\"subgroup_size\":" << probe.subgroup.subgroupSize
            << ",\"cooperative_supported_stages\":"
            << probe.stages.cooperativeMatrixSupportedStages
            << ",\"enabled\":[\"VK_KHR_cooperative_matrix\",\"VK_KHR_shader_"
               "float16_int8\",\"VK_KHR_vulkan_memory_model\",\"VK_EXT_subgroup_"
               "size_control\"],\"cases\":[";
        bool first = true, passed = true;
        unsigned tested = 0;
        for (const auto& shape : probe.shapes) {
            if (shape.AType != VK_COMPONENT_TYPE_FLOAT16_KHR ||
                shape.BType != VK_COMPONENT_TYPE_FLOAT16_KHR ||
                shape.CType != VK_COMPONENT_TYPE_FLOAT32_KHR ||
                shape.ResultType != VK_COMPONENT_TYPE_FLOAT32_KHR ||
                shape.scope != VK_SCOPE_SUBGROUP_KHR ||
                shape.saturatingAccumulation)
                continue;
            const auto m = shape.MSize, n = shape.NSize, k = shape.KSize;
            const std::string shader =
                std::string(argv[1]) + '/' + std::to_string(m) + '-' +
                std::to_string(n) + '-' + std::to_string(k) + ".spv";
            for (const auto* scene :
                 {"product-precision", "cancellation",
                  "above-half-product-range", "small-cancellation",
                  "lost-operand-bits", "mixed-sign"}) {
                auto inputs = Inputs(scene, m, n, k);
                auto result = probe.Run(shader, inputs, m * n);
                if (!first) std::cout << ',';
                first = false;
                passed &= Analyze(scene, m, n, k, inputs, result);
                ++tested;
            }
        }
        std::cout << "],\"tested_cases\":" << tested
                  << ",\"passed\":" << (passed && tested ? "true" : "false")
                  << "}\n";
        return passed && tested ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
