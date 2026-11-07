// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// Research-only subgroup observation. Uses the ordinary runtime context and
// pipeline creation flags; no required subgroup size, full-subgroup flag or
// performance timestamps are added. See the v1_headroom README for commands.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vulkan/vulkan_runtime.h"
#include "vulkan/vulkan_support.h"

namespace {
constexpr std::size_t kInvocations = 256;
constexpr std::size_t kWords = 4 + kInvocations;
constexpr std::size_t kBytes = kWords * sizeof(std::uint32_t);

void Check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(
            std::string(operation) + ": VkResult " + std::to_string(result)
        );
}

std::vector<std::uint32_t> ReadShader(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open supplied SPIR-V shader");
    const auto length = file.tellg();
    if (length < 20 || length % 4 != 0)
        throw std::runtime_error("invalid SPIR-V file size");
    std::vector<std::uint32_t> words(static_cast<std::size_t>(length) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), length);
    if (!file || words[0] != 0x07230203u)
        throw std::runtime_error("invalid SPIR-V shader contents");
    return words;
}

class ProbePipeline {
public:
    ProbePipeline(vqm::internal::VulkanRuntime& runtime, const char* path)
        : device_(runtime.context().device) {
        const auto code = ReadShader(path);
        VkShaderModuleCreateInfo module_info{};
        module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        module_info.codeSize = code.size() * sizeof(std::uint32_t);
        module_info.pCode = code.data();
        VkShaderModule module = VK_NULL_HANDLE;
        Check(
            vkCreateShaderModule(device_, &module_info, nullptr, &module),
            "vkCreateShaderModule"
        );
        VkComputePipelineCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = module;
        info.stage.pName = "main";
        info.layout = runtime.pipeline_layout();
        const auto result = vkCreateComputePipelines(
            device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline_
        );
        vkDestroyShaderModule(device_, module, nullptr);
        Check(result, "vkCreateComputePipelines");
    }
    ~ProbePipeline() { vkDestroyPipeline(device_, pipeline_, nullptr); }
    ProbePipeline(const ProbePipeline&) = delete;
    ProbePipeline& operator=(const ProbePipeline&) = delete;
    [[nodiscard]] VkPipeline handle() const noexcept { return pipeline_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

int Probe(const char* shader) {
    using namespace vqm::internal;
    const auto context = AcquireVulkanContext();
    if (!context) throw std::runtime_error("no Vulkan compute device");
    VkPhysicalDeviceSubgroupProperties subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(context->physical_device, &properties);
    const bool basic =
        (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
        (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0;
    std::cout << "device: " << properties.properties.deviceName << '\n'
              << "default_subgroup_size: " << subgroup.subgroupSize << '\n'
              << "compute_basic_supported: " << basic << '\n';
    if (!basic) throw std::runtime_error("compute subgroup BASIC unsupported");

    VulkanRuntime::Config config;
    config.base_size = {16, 16};
    config.base_pixels = kInvocations;
    config.streaming = false;
    config.push_constant_size = sizeof(std::uint32_t);
    VulkanRuntime runtime(context, config);
    runtime.CreateArena(kBytes);
    runtime.CreateStaging(kBytes);
    runtime.CreateCommandObjects();
    runtime.CreateDescriptors(0, kBytes);
    const ProbePipeline pipeline(runtime, shader);
    runtime.ResetCommandPool();
    runtime.BeginCommandBuffer();
    runtime.BindDescriptorSet();
    // The runtime always publishes a push range; this shader ignores it.
    const std::uint32_t unused = 0;
    runtime.Dispatch(pipeline.handle(), &unused, sizeof(unused), 1, 1, 1);
    runtime.RecordReadback(0, 0, kWords);
    runtime.EndCommandBuffer();
    runtime.SubmitAndWait();
    std::array<std::uint32_t, kWords> result{};
    std::memcpy(result.data(), runtime.staging_map(), kBytes);
    std::array<unsigned, kInvocations> coverage{};
    bool in_range = true;
    for (std::size_t i = 4; i < result.size(); ++i) {
        if (result[i] >= coverage.size())
            in_range = false;
        else
            ++coverage[result[i]];
    }
    const bool exact =
        in_range &&
        std::all_of(coverage.begin(), coverage.end(), [](unsigned count) {
            return count == 1;
        });
    const bool selected = result[0] == 32 && result[1] == 8;
    std::cout << "actual_subgroup_size: " << result[0] << '\n'
              << "actual_num_subgroups: " << result[1] << '\n'
              << "workgroup_invocations: " << result[2] << '\n'
              << "logical_slots_cover_256_once: " << exact << '\n'
              << "subgroup_row_condition: " << result[3] << '\n';
    if (result[2] != kInvocations || result[3] != unsigned(selected) ||
        (selected && !exact))
        throw std::runtime_error("inconsistent subgroup probe output");
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: subgroup_rows_probe SHADER.spv\n";
        return 2;
    }
    int result = 1;
    try {
        result = Probe(argv[1]);
        vqm::internal::ShutdownDefaultVulkanContext();
    } catch (const std::exception& error) {
        result = 1;
        std::cerr << "subgroup probe failed: " << error.what() << '\n';
        try {
            vqm::internal::ShutdownDefaultVulkanContext();
        } catch (const std::exception&) {
        }
    }
    return result;
}
