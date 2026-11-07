#!/usr/bin/env python3
"""Install research-only 32-lane row-local CAMBI contrast barriers.

Cross-row construction retains workgroup barriers. Only the final sliding
loop's two row-private barriers become subgroupBarrier. A physical-device
constructor guard requires compute BASIC operations and default subgroup32;
a uniform shader guard preserves the portable implementation if the actual
workgroup does not contain exactly eight 32-lane subgroups. No new device
features are enabled. Both existing half/uint shader variants use the probe.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAMBI = Path("src/metrics/vmaf/cambi")


def replace_once(text: str, before: str, after: str) -> str:
    """Replace one exact source anchor.

    :param text: Source text to inspect or modify.
    :param before: Source anchor to find.
    :param after: Replacement text.
    :returns: Source text with the replacement applied.
    :raises ValueError: If the source anchor does not occur exactly once.
    """
    if text.count(before) != 1:
        raise ValueError(f"expected one unchanged source anchor: {before[:100]}")
    return text.replace(before, after)


def contrast_function(text: str) -> str:
    """Extract the CAMBI contrast function body.

    :param text: Source text to inspect or modify.
    :returns: Full contrast function source including its braces.
    :raises ValueError: If the contrast function is missing or unbalanced.
    """
    start = text.index("void contrast_map() {")
    opening = text.index("{", start)
    depth = 1
    for end in range(opening + 1, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start:end + 1]
    raise ValueError("unbalanced contrast function")


def make_shader(shader: str, skip_inactive_rows: bool = False) -> str:
    """Build the experiment shader source from the baseline.

    :param shader: Original GLSL source text.
    :param skip_inactive_rows: Whether inactive subgroups skip row processing.
    :returns: Shader with experimental subgroup row mapping.
    :raises ValueError: If the baseline shader layout does not match the probe.
    """
    original = contrast_function(shader)
    mapping = """    // Compute subgroup size is command-uniform; NumSubgroups is uniform
    // within the workgroup. Eight 32-slot groups contain all 256 live lanes.
    // Other packing keeps the original local-index mapping and barriers.
    bool subgroup_rows = gl_SubgroupSize == 32u && gl_NumSubgroups == 8u;
    uint lane = subgroup_rows ? gl_SubgroupInvocationID : gl_LocalInvocationIndex % 32u;
    uint row = subgroup_rows ? gl_SubgroupID : gl_LocalInvocationIndex / 32u;
    uint logical_index = row * 32u + lane;"""
    subgroup = replace_once(
        original, """    uint lane = gl_LocalInvocationIndex % 32u;
    uint row = gl_LocalInvocationIndex / 32u;""", "__CAMBI_ROW_MAPPING__"
    )
    # Every cooperative index, including dense clone construction, follows
    # this virtual ordering. Physical local invocation order is irrelevant.
    if subgroup.count("gl_LocalInvocationIndex"
                      ) != original.count("gl_LocalInvocationIndex") - 2:
        raise ValueError("unexpected cooperative-index mapping")
    subgroup = subgroup.replace("gl_LocalInvocationIndex", "logical_index")
    subgroup = replace_once(subgroup, "__CAMBI_ROW_MAPPING__", mapping)
    loop = "    for (uint x = 0u; x < 16u; ++x) {"
    before_loop, slide = subgroup.split(loop)
    if slide.count("barrier();") != 2:
        raise ValueError("expected exactly two final-loop barriers")
    slide = slide.replace(
        "barrier();", """if (subgroup_rows)
            subgroupBarrier();
        else
            barrier();"""
    )
    if skip_inactive_rows:
        before_loop += """    // Cross-row publication is complete. An inactive logical row has
    // sixteen zero centers and therefore sixteen exactly zero contrasts.
    // This predicate is uniform within one complete subgroup; no remaining
    // workgroup barrier can wait for the returning lanes.
    if (subgroup_rows && !row_active) {
        if (lane < 16u) {
            uint out_x = gl_WorkGroupID.x * 16u + lane;
            uint out_y = gl_WorkGroupID.y * 8u + row;
            if (out_x < pc.width && out_y < pc.height)
                arena[pc.contrast + out_y * pc.width + out_x] = 0.0;
        }
        return;
    }
"""
    subgroup = before_loop + loop + slide
    shader = replace_once(shader, original, subgroup)
    shader = replace_once(
        shader, "#version 450", """#version 450
#extension GL_KHR_shader_subgroup_basic : require"""
    )
    assert subgroup.count("gl_LocalInvocationIndex") == 2
    assert shader.count("void contrast_map() {") == 1
    assert subgroup.count("subgroupBarrier();") == 2
    assert before_loop.count("barrier();") == original.split(loop
                                                             )[0].count("barrier();")
    return shader


def install(source: Path, skip_inactive_rows: bool = False) -> None:
    """Install the research variant in an isolated source copy.

    :param source: Directory containing the isolated source copy.
    :param skip_inactive_rows: Whether inactive subgroups skip row processing.
    :raises ValueError: If the target or expected source layout is unsuitable for the
        variant.
    """
    source = source.resolve()
    if source == ROOT or source.is_relative_to(ROOT) or ROOT.is_relative_to(source):
        raise ValueError(
            "use a fresh isolated source copy outside the maintained checkout"
        )
    shader_path = source / CAMBI / "shaders/vmaf_cambi_contrast.comp"
    host_path = source / CAMBI / "vmaf_cambi.cc"
    shader, host = shader_path.read_text(), host_path.read_text()
    if "bool subgroup_rows" in shader:
        raise ValueError("subgroup row probe already exists; use a fresh source copy")
    shader = make_shader(shader, skip_inactive_rows)
    host = replace_once(
        host, """namespace {
struct CambiContrastPush {""", """namespace {
// Research-only local-device guard, before any contrast pipeline is created.
// No default-subgroup or new-feature promise is added to the installed API.
void RequireCambiSubgroupRows(VulkanRuntime& runtime) {
    VkPhysicalDeviceSubgroupProperties subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(runtime.context().physical_device, &properties);
    if (subgroup.subgroupSize != 32 ||
        (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0 ||
        (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) == 0)
        throw std::runtime_error("CAMBI row probe requires default compute subgroup32 BASIC");
}
struct CambiContrastPush {"""
    )
    host = replace_once(
        host,
        """VmafCambiContrast::VmafCambiContrast(VulkanRuntime& runtime, bool allow_half)
    : runtime_(&runtime),""",
        """VmafCambiContrast::VmafCambiContrast(VulkanRuntime& runtime, bool allow_half)
    : runtime_((RequireCambiSubgroupRows(runtime), &runtime)),"""
    )
    # Preflight every source anchor before either file is written.
    shader_path.write_text(shader)
    host_path.write_text(host)


def main() -> None:
    """Install the requested CAMBI subgroup-row shader experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument(
        "--skip-inactive-rows", action="store_true",
        help="after cross-row publication, return complete inactive row subgroups"
    )
    args = parser.parse_args()
    install(args.source_dir, args.skip_inactive_rows)
    print(
        "Installed research-only subgroup32 CAMBI row barriers" +
        (" and inactive-row skip." if args.skip_inactive_rows else ".")
    )


if __name__ == "__main__":
    main()
