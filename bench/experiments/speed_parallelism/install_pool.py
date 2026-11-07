#!/usr/bin/env python3
"""Install a full-32-lane-subgroup pooling experiment in a fresh source copy.

Scores temporarily alias dead means/covariance storage only for <=1300 blocks.
The original two-level sum order and old partial/memory layouts are preserved;
larger geometries retain the original shaders. This is a local-device trial,
not production feature negotiation.
"""

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def replace(path: Path, old: str, new: str) -> None:
    """Replace one exact anchor in a source file.

    :param path: File to read, hash or update.
    :param old: Source anchor to find.
    :param new: Replacement text.
    :raises ValueError: If the source anchor does not occur exactly once.
    """
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f"expected one unchanged anchor in {path}: {old[:65]}")
    path.write_text(text.replace(old, new))


def main() -> None:
    """Install the pooling experiment in the selected source copy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--subgroups", type=int, choices=(1, 2, 4), default=2)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    if source == ROOT or source.is_relative_to(ROOT) or ROOT.is_relative_to(source):
        parser.error("use an isolated source copy outside the maintained checkout")
    support = source / "src/vulkan/vulkan_support.cc"
    replace(
        support, "    device_info.pNext = &storage16;",
        """    device_info.pNext = &storage16;
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    subgroup.subgroupSizeControl = VK_TRUE;
    subgroup.computeFullSubgroups = VK_TRUE;
    storage16.pNext = &float16;
    float16.pNext = &subgroup;
    const char* pool_extensions[] = {VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
                                    VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME};
    device_info.enabledExtensionCount = 2;
    device_info.ppEnabledExtensionNames = pool_extensions;"""
    )
    runtime = source / "src/vulkan/vulkan_runtime.cc"
    replace(
        runtime, "    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;",
        """    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size{};
    subgroup_size.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroup_size.requiredSubgroupSize = 32;
    if (name == "vmaf_speed_pool_subgroup") {
        info.stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        info.stage.pNext = &subgroup_size;
    }"""
    )
    header = source / "src/metrics/vmaf/speed/vmaf_speed.h"
    replace(
        header, "    std::array<VkPipeline, 4> pipelines_{};",
        "    std::array<VkPipeline, 6> pipelines_{};"
    )
    host = source / "src/metrics/vmaf/speed/vmaf_speed.cc"
    replace(
        host,
        '                           "vmaf_speed_stats_pool", "vmaf_speed_stats_reduce"};',
        '                           "vmaf_speed_stats_pool", "vmaf_speed_stats_reduce",\n'
        '                           "vmaf_speed_pool_subgroup", "vmaf_speed_pool_reduce_alias"};'
    )
    old = """    runtime_->Dispatch(
        pipelines_[2], &pc, sizeof(pc), static_cast<std::uint32_t>(groups),
        channels, 1
    );
    runtime_->ComputeBarrier();
    runtime_->Dispatch(pipelines_[3], &pc, sizeof(pc), channels, 1, 1);"""
    new = f"""    const auto blocks = (size.width / 5) * (size.height / 5);
    // After factorization, [kMeans,kCholesky) is dead until the next frame.
    // Source, factors, eigenvalues and status remain live and disjoint.
    const bool alias_pool = blocks <= 1300;
    runtime_->Dispatch(
        pipelines_[alias_pool ? 4 : 2], &pc, sizeof(pc),
        alias_pool ? Groups(blocks, {args.subgroups}) : static_cast<std::uint32_t>(groups),
        channels, 1
    );
    runtime_->ComputeBarrier();
    runtime_->Dispatch(pipelines_[alias_pool ? 5 : 3], &pc, sizeof(pc), channels, 1, 1);"""
    replace(host, old, new)
    shaders = source / "src/metrics/vmaf/speed/shaders"
    for name in ("pool_subgroup.comp", "pool_reduce_alias.comp"):
        (shaders / name).write_text((HERE / name).read_text())
    cmake = source / "CMakeLists.txt"
    anchor = "vqm_add_shader(vmaf_speed_stats_reduce metrics/vmaf/speed/shaders/vmaf_speed_stats_reduce.comp)"
    replace(
        cmake, anchor, anchor + f"""
vqm_add_shader(vmaf_speed_pool_subgroup metrics/vmaf/speed/shaders/pool_subgroup.comp
  -DSPEED_POOL_SUBGROUPS={args.subgroups} -DSPEED_POOL_WRITE_SCORES=2)
vqm_add_shader(vmaf_speed_pool_reduce_alias metrics/vmaf/speed/shaders/pool_reduce_alias.comp)"""
    )
    print(
        f"Installed {args.subgroups} subgroups per pooling workgroup, exact reduction alias arm."
    )


if __name__ == "__main__":
    main()
