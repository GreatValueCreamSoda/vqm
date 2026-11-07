#!/usr/bin/env python3
"""Install subgroup-per-rank bisection in the original one-dispatch solver.

The maintained checkout is refused. Householder, Cholesky, the QR guard and
fallback stay unchanged. The normal/wide blob names use a generated shader,
while the forced-legacy oracle continues compiling the original shader file.
"""

import argparse
import re
from pathlib import Path

from solver_phases import HERE, ROOT, block_body, replace_once


def make_subgroup_shader(original: str) -> str:
    """Generate the subgroup solver variant.

    :param original: Original source text.
    :returns: Solver GLSL with subgroup bisection installed.
    :raises ValueError: If the expected baseline source anchors are absent.
    """
    tree = (HERE / "solver_phase_bisect.comp").read_text()
    start = tree.index("    float lo = arena[aux + 49u], hi = arena[aux + 50u];")
    stop = tree.index("    if (lane == 0u) {", start)
    search = tree[start:stop]
    search = replace_once(
        search, "float lo = arena[aux + 49u], hi = arena[aux + 50u];",
        "float lo = lower_bound, hi = upper_bound;"
    )
    search = re.sub(r"\blane\b", "subgroup_lane", search)
    helper = """
// One rank per complete subgroup, using the same five-decision trial tree as
// the split-solver experiment. There are no workgroup barriers in this helper.
float subgroup_eigenvalue(uint rank, out bool converged_out) {
    uint subgroup_lane = gl_SubgroupInvocationID;
""" + search + """    converged_out = converged && !invalid;
    return lo * 0.5 + hi * 0.5;
}

"""
    original = replace_once(
        original, "#version 450", """#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require"""
    )
    original = replace_once(original, "void main() {", helper + "void main() {")
    anchor = "    if (lane < 25u && bounds_valid) {"
    body = block_body(original, anchor)
    tail = body[body.index("        arena[eigen(ch, lane)] = value;"):]
    tail = re.sub(r"\blane\b", "rank", tail)
    replacement = """    if (bounds_valid) {
        uint rank_batches = (25u + gl_NumSubgroups - 1u) / gl_NumSubgroups;
        for (uint batch = 0u; batch < rank_batches; ++batch) {
            uint rank = gl_SubgroupID + batch * gl_NumSubgroups;
            if (rank < 25u) {
                bool converged;
                float value = subgroup_eigenvalue(rank, converged);
                if (gl_SubgroupInvocationID == 0u) {
""" + tail + """                }
            }
        }
    }"""
    return replace_once(original, anchor + body + "}", replacement)


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
    cmake_path = source / "CMakeLists.txt"
    cmake = cmake_path.read_text()
    if "vmaf_speed_stats_solver_prepare" in cmake or "vmaf_speed_pool_subgroup" in cmake:
        raise ValueError(
            "use a fresh source copy without another solver/pooling installer"
        )
    shaders = source / "src/metrics/vmaf/speed/shaders"
    original_path = shaders / "vmaf_speed_stats_eigen.comp"
    shader = make_subgroup_shader(original_path.read_text())
    for suffix, flags in [("", ""), ("_wide", " -DVQM_SPEED_EIGEN_LANES=512")]:
        before = f"vqm_add_shader(vmaf_speed_stats_eigen{suffix} metrics/vmaf/speed/shaders/vmaf_speed_stats_eigen.comp{flags})"
        after = before.replace("/vmaf_speed_stats_eigen.comp", "/solver_subgroup.comp")
        cmake = replace_once(cmake, before, after)
    support_path = source / "src/vulkan/vulkan_support.cc"
    support = replace_once(
        support_path.read_text(), "    device_info.pNext = &storage16;",
        """    device_info.pNext = &storage16;
    // Research-only local-device promises; this is not portable negotiation.
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    subgroup.subgroupSizeControl = VK_TRUE;
    subgroup.computeFullSubgroups = VK_TRUE;
    storage16.pNext = &float16;
    float16.pNext = &subgroup;
    const char* solver_extensions[] = {VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
                                      VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME};
    device_info.enabledExtensionCount = 2;
    device_info.ppEnabledExtensionNames = solver_extensions;"""
    )
    runtime_path = source / "src/vulkan/vulkan_runtime.cc"
    runtime = replace_once(
        runtime_path.read_text(), "    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;",
        """    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size{};
    subgroup_size.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroup_size.requiredSubgroupSize = 32;
    if (name == "vmaf_speed_stats_eigen" || name == "vmaf_speed_stats_eigen_wide") {
        info.stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        info.stage.pNext = &subgroup_size;
    }"""
    )
    # Check every fragile anchor before any source mutation.
    cmake_path.write_text(cmake)
    support_path.write_text(support)
    runtime_path.write_text(runtime)
    (shaders / "solver_subgroup.comp").write_text(shader)


def main() -> None:
    """Install the subgroup-per-rank solver experiment."""
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    args = parser.parse_args()
    install(args.source_dir)
    print(
        "Installed one-subgroup-per-rank bisection without extra dispatches or scratch"
    )


if __name__ == "__main__":
    main()
