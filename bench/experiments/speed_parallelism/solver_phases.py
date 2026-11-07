#!/usr/bin/env python3
"""Install the split Householder / parallel-tree Sturm / QR solver experiment.

Only a fresh isolated source copy is accepted. The original solver remains an
independent test oracle. This prototype assumes native 32-lane compute subgroups
with shuffle support; the production library's device negotiation is untouched.
"""

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def replace_once(text: str, before: str, after: str) -> str:
    """Replace one exact source anchor.

    :param text: Source text to inspect or modify.
    :param before: Source anchor to find.
    :param after: Replacement text.
    :returns: Source text with the replacement applied.
    :raises ValueError: If the source anchor does not occur exactly once.
    """
    if text.count(before) != 1:
        raise ValueError(f"expected exactly one baseline anchor: {before[:90]}")
    return text.replace(before, after)


def block_body(text: str, anchor: str) -> str:
    """Extract the source block after an exact anchor.

    :param text: Source text to inspect or modify.
    :param anchor: Source anchor immediately before the block.
    :returns: Source text between the anchored braces.
    :raises ValueError: If the anchor is absent or the block is unterminated.
    """
    start = text.index(anchor)
    first = text.index("{", start)
    depth = 1
    for index in range(first + 1, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[first + 1:index]
    raise ValueError(f"unterminated source block: {anchor}")


def make_phase_shaders(original: str) -> dict[str, str]:
    """Generate shader sources for the experimental solver phases.

    :param original: Original source text.
    :returns: Shader source indexed by output filename.
    """
    preamble, body = original.split("void main() {", 1)
    helper = "\nuint solver_aux(uint ch) { return pc.scratch + 2604u * pc.op + 56u * ch; }\n"
    preamble += helper
    prepare = "void main() {" + body.split("    bool bounds_valid = failed == 0u;",
                                           1)[0]
    prepare = replace_once(
        prepare, "        minimum_eigen = 0.0;", """        minimum_eigen = 0.0;
        arena[solver_aux(ch) + 51u] = 2.0;
        arena[solver_aux(ch) + 52u] = 0.0;"""
    )
    prepare = replace_once(
        prepare, "        if (lane == 0u) arena[singular(ch)] = 2.0;",
        """        if (lane == 0u) {
            arena[singular(ch)] = 2.0;
            arena[solver_aux(ch) + 51u] = 2.0;
            arena[solver_aux(ch) + 52u] = 1.0;
        }"""
    )
    prepare += """    uint aux = solver_aux(ch);
    if (lane < 25u) arena[aux + lane] = diagonal[lane];
    if (lane < 24u) arena[aux + 25u + lane] = subdiagonal[lane];
    if (lane == 0u) {
        arena[aux + 49u] = lower_bound;
        arena[aux + 50u] = upper_bound;
        arena[aux + 51u] = float(failed);
    }
}
"""
    guard = block_body(
        original, "    if (lane == 0u) {\n        // Heuristic, not a formal bound:"
    )
    fallback = block_body(original, "    if (fallback != 0u)")
    qr = block_body(fallback, "        if (lane == 0u)")
    qr = replace_once(
        qr, "diagonal[row] = matrix[row * 25u + row];",
        "diagonal[row] = arena[aux + row];"
    )
    qr = replace_once(
        qr, "subdiagonal[row] = matrix[(row + 1u) * 25u + row];",
        "subdiagonal[row] = arena[aux + 25u + row];"
    )
    finish = """void main() {
    uint ch = gl_WorkGroupID.x, lane = gl_LocalInvocationIndex;
    if (lane != 0u) return;
    uint aux = solver_aux(ch);
    if (arena[aux + 52u] != 0.0) {
        arena[singular(ch)] = 2.0;
        return;
    }
    failed = uint(arena[aux + 51u]);
    for (uint row = 0u; row < 25u; ++row)
        diagonal[row] = arena[aux + row];
    for (uint row = 0u; row < 24u; ++row)
        subdiagonal[row] = arena[aux + 25u + row];
    lower_bound = arena[aux + 49u];
    upper_bound = arena[aux + 50u];
    minimum_eigen = arena[eigen(ch, 0u)];
    if (failed == 0u) {
        for (uint rank = 0u; rank < 25u; ++rank) {
            float value = arena[eigen(ch, rank)];
            if (isnan(value) || isinf(value)) failed = 2u;
        }
        uint count = sturm_count(1e-6);
        if (count > 25u) failed = 2u;
        else if (count > 0u) failed = max(failed, 1u);
    }
""" + guard + "\n    if (fallback != 0u) {\n" + qr + "\n    }\n    arena[singular(ch)] = float(failed);\n}\n"
    return {
        "solver_phase_prepare.comp": preamble + prepare,
        "solver_phase_finish.comp": preamble + finish,
        "solver_phase_bisect.comp": (HERE / "solver_phase_bisect.comp").read_text()
    }


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
    shader_dir = source / "src/metrics/vmaf/speed/shaders"
    original = (shader_dir / "vmaf_speed_stats_eigen.comp").read_text()
    shaders = make_phase_shaders(original)
    header_path = source / "src/metrics/vmaf/speed/vmaf_speed.h"
    header = replace_once(
        header_path.read_text(), "        return 2604;", "        return 2604 + 2 * 56;"
    )
    header = replace_once(
        header, "    std::array<VkPipeline, 4> pipelines_{};",
        "    std::array<VkPipeline, 6> pipelines_{};"
    )
    host_path = source / "src/metrics/vmaf/speed/vmaf_speed.cc"
    host = host_path.read_text()
    host = replace_once(
        host, '? "vmaf_speed_stats_eigen_wide"',
        '? "vmaf_speed_stats_solver_prepare_wide"'
    )
    host = replace_once(
        host, ': "vmaf_speed_stats_eigen";', ': "vmaf_speed_stats_solver_prepare";'
    )
    host = replace_once(
        host,
        '                           "vmaf_speed_stats_pool", "vmaf_speed_stats_reduce"};',
        '                           "vmaf_speed_stats_pool", "vmaf_speed_stats_reduce",\n'
        '                           "vmaf_speed_stats_solver_bisect", "vmaf_speed_stats_solver_finish"};'
    )
    host = replace_once(
        host,
        "    runtime_->Dispatch(pipelines_[1], &pc, sizeof(pc), 2 * channels, 2, 1);",
        """    pc.op = channels;
    runtime_->Dispatch(pipelines_[1], &pc, sizeof(pc), 2 * channels, 2, 1);
    runtime_->ComputeBarrier();
    runtime_->Dispatch(pipelines_[4], &pc, sizeof(pc), 25, 2 * channels, 1);
    runtime_->ComputeBarrier();
    runtime_->Dispatch(pipelines_[5], &pc, sizeof(pc), 2 * channels, 1, 1);
    pc.op = 1;"""
    )
    cmake_path = source / "CMakeLists.txt"
    cmake = cmake_path.read_text()
    anchor = "vqm_add_shader(vmaf_speed_stats_eigen_wide metrics/vmaf/speed/shaders/vmaf_speed_stats_eigen.comp -DVQM_SPEED_EIGEN_LANES=512)"
    cmake = replace_once(
        cmake, anchor, anchor + """
vqm_add_shader(vmaf_speed_stats_solver_prepare metrics/vmaf/speed/shaders/solver_phase_prepare.comp)
vqm_add_shader(vmaf_speed_stats_solver_prepare_wide metrics/vmaf/speed/shaders/solver_phase_prepare.comp -DVQM_SPEED_EIGEN_LANES=512)
vqm_add_shader(vmaf_speed_stats_solver_bisect metrics/vmaf/speed/shaders/solver_phase_bisect.comp)
vqm_add_shader(vmaf_speed_stats_solver_finish metrics/vmaf/speed/shaders/solver_phase_finish.comp -DVQM_SPEED_EIGEN_LANES=32)"""
    )
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
    if (name == "vmaf_speed_stats_solver_bisect") {
        info.stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        info.stage.pNext = &subgroup_size;
    }"""
    )
    # All fragile anchors are checked before writing any source file.
    header_path.write_text(header)
    host_path.write_text(host)
    cmake_path.write_text(cmake)
    support_path.write_text(support)
    runtime_path.write_text(runtime)
    for name, text in shaders.items():
        (shader_dir / name).write_text(text)


def main() -> None:
    """Install the experimental solver phase shaders and dispatches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    args = parser.parse_args()
    install(args.source_dir)
    print("Installed three-phase current SpEED solver research variant")


if __name__ == "__main__":
    main()
