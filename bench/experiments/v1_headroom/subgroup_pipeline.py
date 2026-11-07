#!/usr/bin/env python3
"""Install a capability-selected CAMBI subgroup pipeline in an isolated copy.

Original portable uint/half modules keep their names and compile without any
subgroup capability. Two additional modules opt in via VQM_CAMBI_SUBGROUP_ROWS.
Unsupported subgroup properties select the portable modules. This installer
also adds forced-portable/automatic and conservative borrowed-device checks;
the maintained checkout is deliberately refused.
"""

import argparse
from pathlib import Path

from subgroup_rows import CAMBI, ROOT, contrast_function, replace_once


def make_shader(shader: str) -> str:
    """Build the experiment shader source from the baseline.

    :param shader: Original GLSL source text.
    :returns: Shader with capability-selected subgroup row mapping.
    :raises ValueError: If the expected barrier or source anchor is absent.
    """
    original = contrast_function(shader)
    mapping = """#ifdef VQM_CAMBI_SUBGROUP_ROWS
    // All 256 invocations reach this workgroup-uniform test before any
    // pixel-bound return. Eight 32-slot subgroups therefore form full rows.
    bool subgroup_rows = gl_SubgroupSize == 32u && gl_NumSubgroups == 8u;
    uint lane = subgroup_rows ? gl_SubgroupInvocationID : gl_LocalInvocationIndex % 32u;
    uint row = subgroup_rows ? gl_SubgroupID : gl_LocalInvocationIndex / 32u;
    uint logical_index = row * 32u + lane;
#else
    uint lane = gl_LocalInvocationIndex % 32u;
    uint row = gl_LocalInvocationIndex / 32u;
#endif"""
    changed = replace_once(
        original, """    uint lane = gl_LocalInvocationIndex % 32u;
    uint row = gl_LocalInvocationIndex / 32u;""", "__CAMBI_ROW_MAPPING__"
    )
    changed = changed.replace("gl_LocalInvocationIndex", "CAMBI_LOGICAL_INDEX")
    changed = replace_once(changed, "__CAMBI_ROW_MAPPING__", mapping)
    loop = "    for (uint x = 0u; x < 16u; ++x) {"
    before_loop, slide = changed.split(loop)
    if slide.count("barrier();") != 2:
        raise ValueError("expected exactly two final-loop barriers")
    slide = slide.replace(
        "        barrier();", """#ifdef VQM_CAMBI_SUBGROUP_ROWS
        if (subgroup_rows)
            subgroupBarrier();
        else
#endif
            barrier();"""
    )
    changed = before_loop + loop + slide
    shader = replace_once(shader, original, changed)
    shader = replace_once(
        shader, "#version 450", """#version 450
#ifdef VQM_CAMBI_SUBGROUP_ROWS
#extension GL_KHR_shader_subgroup_basic : require
#define CAMBI_LOGICAL_INDEX logical_index
#else
#define CAMBI_LOGICAL_INDEX gl_LocalInvocationIndex
#endif"""
    )
    assert shader.count("void contrast_map() {") == 1
    assert shader.count("subgroupBarrier();") == 2
    assert before_loop.count("barrier();") == original.split(loop
                                                             )[0].count("barrier();")
    return shader


def install(source: Path) -> None:
    """Install the research variant in an isolated source copy.

    :param source: Directory containing the isolated source copy.
    :raises ValueError: If the target or expected source layout is unsuitable for the
        variant.
    """
    source = source.resolve()
    if source == ROOT or source.is_relative_to(ROOT) or ROOT.is_relative_to(source):
        raise ValueError(
            "use a fresh isolated source copy outside the maintained checkout"
        )
    paths = (
        CAMBI / "shaders/vmaf_cambi_contrast.comp", CAMBI / "vmaf_cambi.h",
        CAMBI / "vmaf_cambi.cc", Path("CMakeLists.txt"),
        Path("tests/metrics/vmaf/cambi/vmaf_cambi_contrast_test.cc")
    )
    shader, header, host, cmake, tests = ((source / path).read_text() for path in paths)
    if "VQM_CAMBI_SUBGROUP_ROWS" in shader:
        raise ValueError("subgroup probe already installed; use a fresh source copy")
    shader = make_shader(shader)
    cmake = replace_once(
        cmake,
        "vqm_add_shader(vmaf_cambi_contrast_half metrics/vmaf/cambi/shaders/vmaf_cambi_contrast.comp -DVQM_CAMBI_HALF_SOURCE=1)",
        """vqm_add_shader(vmaf_cambi_contrast_half metrics/vmaf/cambi/shaders/vmaf_cambi_contrast.comp -DVQM_CAMBI_HALF_SOURCE=1)
vqm_add_shader(vmaf_cambi_contrast_subgroup metrics/vmaf/cambi/shaders/vmaf_cambi_contrast.comp -DVQM_CAMBI_SUBGROUP_ROWS=1)
vqm_add_shader(vmaf_cambi_contrast_subgroup_half metrics/vmaf/cambi/shaders/vmaf_cambi_contrast.comp -DVQM_CAMBI_SUBGROUP_ROWS=1 -DVQM_CAMBI_HALF_SOURCE=1)"""
    )
    header = replace_once(
        header,
        "    explicit VmafCambiContrast(VulkanRuntime& runtime, bool allow_half = true);",
        """    explicit VmafCambiContrast(VulkanRuntime& runtime, bool allow_half = true,
                               bool allow_subgroups = true);"""
    )
    header = replace_once(
        header,
        "    [[nodiscard]] bool UsesHalfSource() const noexcept { return half_source_; }",
        """    [[nodiscard]] bool UsesHalfSource() const noexcept { return half_source_; }
    [[nodiscard]] bool UsesSubgroupPipeline() const noexcept { return subgroup_pipeline_; }
    [[nodiscard]] static constexpr bool SupportsSubgroupRows(
        const VkPhysicalDeviceSubgroupProperties& properties) noexcept {
        return properties.subgroupSize == 32 &&
               (properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
               (properties.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0;
    }"""
    )
    header = replace_once(
        header, "    bool half_source_ = false;", """    bool half_source_ = false;
    bool subgroup_pipeline_ = false;"""
    )
    host = replace_once(
        host, """namespace {
struct CambiContrastPush {""", """namespace {
bool SupportsCambiSubgroupRows(VulkanRuntime& runtime) {
    VkPhysicalDeviceSubgroupProperties subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(runtime.context().physical_device, &properties);
    // BASIC subgroup operations are core Vulkan 1.1, without an enabled-device
    // feature. Float16 storage remains independently negotiated below.
    return VmafCambiContrast::SupportsSubgroupRows(subgroup);
}
struct CambiContrastPush {"""
    )
    host = replace_once(
        host,
        "VmafCambiContrast::VmafCambiContrast(VulkanRuntime& runtime, bool allow_half)",
        """VmafCambiContrast::VmafCambiContrast(VulkanRuntime& runtime, bool allow_half,
                                   bool allow_subgroups)"""
    )
    host = replace_once(
        host, "      half_source_(allow_half && runtime.context().shader_float16),",
        """      half_source_(allow_half && runtime.context().shader_float16),
      subgroup_pipeline_(allow_subgroups && SupportsCambiSubgroupRows(runtime)),"""
    )
    host = replace_once(
        host,
        '          half_source_ ? "vmaf_cambi_contrast_half" : "vmaf_cambi_contrast"',
        """          subgroup_pipeline_
              ? (half_source_ ? "vmaf_cambi_contrast_subgroup_half"
                              : "vmaf_cambi_contrast_subgroup")
              : (half_source_ ? "vmaf_cambi_contrast_half" : "vmaf_cambi_contrast")"""
    )
    tests = replace_once(
        tests,
        "        vqm::Size size, bool allow_half = true, bool enabled_float16 = true\n",
        """        vqm::Size size, bool allow_half = true, bool enabled_float16 = true,
        bool allow_subgroups = true
"""
    )
    tests = replace_once(
        tests, "        contrast_.emplace(runtime_, allow_half);",
        "        contrast_.emplace(runtime_, allow_half, allow_subgroups);"
    )
    tests = replace_once(
        tests,
        "    bool UsesHalfSource() const { return contrast_->UsesHalfSource(); }",
        """    bool UsesHalfSource() const { return contrast_->UsesHalfSource(); }
    bool UsesSubgroupPipeline() const { return contrast_->UsesSubgroupPipeline(); }"""
    )
    tests += """

VQM_TEST(TestVmafCambiSubgroupEligibility) {
    using vqm::internal::VmafCambiContrast;
    VkPhysicalDeviceSubgroupProperties properties{};
    CHECK(!VmafCambiContrast::SupportsSubgroupRows(properties));
    properties.subgroupSize = 32;
    properties.supportedStages = VK_SHADER_STAGE_COMPUTE_BIT;
    properties.supportedOperations = VK_SUBGROUP_FEATURE_BASIC_BIT;
    CHECK(VmafCambiContrast::SupportsSubgroupRows(properties));
    for (const std::uint32_t size : {1u, 16u, 64u, 128u}) {
        properties.subgroupSize = size;
        CHECK(!VmafCambiContrast::SupportsSubgroupRows(properties));
    }
    properties.subgroupSize = 32;
    properties.supportedStages = VK_SHADER_STAGE_VERTEX_BIT;
    CHECK(!VmafCambiContrast::SupportsSubgroupRows(properties));
    properties.supportedStages = VK_SHADER_STAGE_COMPUTE_BIT;
    properties.supportedOperations = VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    CHECK(!VmafCambiContrast::SupportsSubgroupRows(properties));
    properties.supportedOperations |= VK_SUBGROUP_FEATURE_BASIC_BIT;
    CHECK(VmafCambiContrast::SupportsSubgroupRows(properties));
}

VQM_TEST(TestVmafCambiSubgroupAndPortableModulesMatchExactly) {
    if (!vqm::test::GpuAvailable()) return;
    constexpr vqm::Size size{65, 37};
    ContrastDevice portable_uint(size, false, true, false);
    ContrastDevice portable_half(size, true, true, false);
    ContrastDevice subgroup_uint(size, false, true, true);
    ContrastDevice subgroup_half(size, true, true, true);
    ContrastDevice borrowed_uint(size, true, false, true);
    ContrastDevice borrowed_portable(size, true, false, false);
    const auto context = vqm::internal::AcquireVulkanContext();
    VkPhysicalDeviceSubgroupProperties properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 physical{};
    physical.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    physical.pNext = &properties;
    vkGetPhysicalDeviceProperties2(context->physical_device, &physical);
    const bool eligible = vqm::internal::VmafCambiContrast::SupportsSubgroupRows(properties);
    CHECK(!portable_uint.UsesSubgroupPipeline());
    CHECK(!portable_half.UsesSubgroupPipeline());
    CHECK(subgroup_uint.UsesSubgroupPipeline() == eligible);
    CHECK(subgroup_half.UsesSubgroupPipeline() == eligible);
    CHECK(borrowed_uint.UsesSubgroupPipeline() == eligible);
    CHECK(!borrowed_portable.UsesSubgroupPipeline());
    CHECK(!portable_uint.UsesHalfSource());
    CHECK(portable_half.UsesHalfSource() == context->shader_float16);
    CHECK(subgroup_half.UsesHalfSource() == context->shader_float16);
    CHECK(!borrowed_uint.UsesHalfSource());
    CHECK(!borrowed_portable.UsesHalfSource());
    for (unsigned test_case = 0; test_case < 8; ++test_case) {
        const unsigned window = test_case == 0 ? 1u :
            (test_case == 1 || test_case == 6) ? 17u : 33u;
        const auto expected = portable_uint.Compute(test_case, window);
        for (auto* device : {&portable_half, &subgroup_uint, &subgroup_half,
                             &borrowed_uint, &borrowed_portable}) {
            const auto actual = device->Compute(test_case, window);
            CHECK(actual.size() == expected.size());
            for (std::size_t i = 0; i < actual.size(); ++i)
                CHECK(std::bit_cast<std::uint32_t>(actual[i]) ==
                      std::bit_cast<std::uint32_t>(expected[i]));
        }
    }
}
"""
    # Validate every fragile source anchor before writing any source.
    for path, text in zip(paths, (shader, header, host, cmake, tests), strict=True):
        (source / path).write_text(text)


def main() -> None:
    """Install the capability-selected CAMBI subgroup pipelines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    args = parser.parse_args()
    install(args.source_dir)
    print(
        "Installed capability-selected CAMBI subgroup modules with portable fallback and tests."
    )


if __name__ == "__main__":
    main()
