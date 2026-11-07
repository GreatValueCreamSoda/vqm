#!/usr/bin/env python3
"""Install the SpEED cooperative-matrix experiment into an isolated source copy.

The production checkout is deliberately refused. This local-device research
variant requires all named extensions, a 32-lane subgroup, and FP16/FP32
16x16x16 cooperative matrices; it does not implement portable negotiation.
"""

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def replace_once(path: Path, before: str, after: str) -> None:
    """Replace one exact anchor in a source file.

    :param path: File to read, hash or update.
    :param before: Source anchor to find.
    :param after: Replacement text.
    :raises ValueError: If the source anchor does not occur exactly once.
    """
    text = path.read_text()
    if text.count(before) != 1:
        raise ValueError(
            f"expected exactly one baseline anchor in {path}: {before[:70]}"
        )
    path.write_text(text.replace(before, after))


def install(
    source: Path, products: int, split_k: int, full_tiles: bool = False
) -> None:
    """Install the research variant in an isolated source copy.

    :param source: Directory containing the isolated source copy.
    :param products: Number of matrix products used in the variant.
    :param split_k: Number of disjoint matrix inner-dimension partitions.
    :param full_tiles: Whether to include the lower matrix tile.
    :raises ValueError: If the target or expected source layout is unsuitable for the
        variant.
    """
    if source == ROOT or ROOT.is_relative_to(source) or source.is_relative_to(ROOT):
        raise ValueError("use an isolated source copy outside the production checkout")
    support = source / "src/vulkan/vulkan_support.cc"
    replace_once(
        support, "    device_info.pNext = &storage16;",
        """    device_info.pNext = &storage16;
    // Research-only: the platform probe verifies these local-device promises.
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative{};
    cooperative.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    cooperative.cooperativeMatrix = VK_TRUE;
    VkPhysicalDeviceVulkanMemoryModelFeatures memory{};
    memory.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
    memory.vulkanMemoryModel = VK_TRUE;
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    subgroup.subgroupSizeControl = VK_TRUE;
    subgroup.computeFullSubgroups = VK_TRUE;
    storage16.pNext = &float16;
    float16.pNext = &cooperative;
    cooperative.pNext = &memory;
    memory.pNext = &subgroup;
    const char* research_extensions[] = {
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
        VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME,
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME};
    device_info.enabledExtensionCount = 4;
    device_info.ppEnabledExtensionNames = research_extensions;"""
    )
    runtime = source / "src/vulkan/vulkan_runtime.cc"
    replace_once(
        runtime, "    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;",
        """    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size{};
    subgroup_size.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroup_size.requiredSubgroupSize = 32;
    if (name == "vmaf_speed_stats_cov_coop") {
        info.stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        info.stage.pNext = &subgroup_size;
    }"""
    )
    header = source / "src/metrics/vmaf/speed/vmaf_speed.h"
    streams = 2 if products == 2 else 1
    replace_once(
        header, "        return 2604;",
        f"        return 2604 + 2 * {split_k} * 4 * 256 * {streams};"
    )
    replace_once(
        header, "    std::array<VkPipeline, 4> pipelines_{};",
        "    std::array<VkPipeline, 6> pipelines_{};"
    )
    host = source / "src/metrics/vmaf/speed/vmaf_speed.cc"
    replace_once(
        host,
        '                           "vmaf_speed_stats_pool", "vmaf_speed_stats_reduce"};',
        '                           "vmaf_speed_stats_pool", "vmaf_speed_stats_reduce",\n'
        '                           "vmaf_speed_stats_cov_coop", "vmaf_speed_stats_cov_merge"};'
    )
    triangular = products != 2 and not full_tiles
    tiles = 3 if triangular else 4
    replace_once(
        host,
        "    runtime_->Dispatch(pipelines_[0], &pc, sizeof(pc), 650 * channels, 1, 1);",
        f"""    pc.op = channels;
    runtime_->Dispatch(pipelines_[4], &pc, sizeof(pc), {tiles}, 2 * channels, {split_k});
    runtime_->ComputeBarrier();
    runtime_->Dispatch(pipelines_[5], &pc, sizeof(pc), Groups(650 * channels, 64), 1, 1);
    pc.op = 1;"""
    )
    shader_dir = source / "src/metrics/vmaf/speed/shaders"
    for name in ("speed_covariance.comp", "speed_covariance_merge.comp"):
        (shader_dir / name).write_text((HERE / name).read_text())
    cmake = source / "CMakeLists.txt"
    anchor = "vqm_add_shader(vmaf_speed_stats_cov metrics/vmaf/speed/shaders/vmaf_speed_stats_cov.comp)"
    extra = f"""
vqm_add_shader(vmaf_speed_stats_cov_coop metrics/vmaf/speed/shaders/speed_covariance.comp
  -DCOOP_SPLIT_K={split_k} -DCOOP_PRODUCTS={products} -DCOOP_TRIANGULAR={int(triangular)} --target-spv=spv1.3)
vqm_add_shader(vmaf_speed_stats_cov_merge metrics/vmaf/speed/shaders/speed_covariance_merge.comp
  -DCOOP_SPLIT_K={split_k} -DCOOP_PRODUCTS={products})"""
    replace_once(cmake, anchor, anchor + extra)


def main() -> None:
    """Install the selected cooperative-matrix covariance variant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument(
        "--products", type=int, choices=(1, 2, 3, 4), default=1,
        help="HH, HH+HL with transposed cross term, three or four products"
    )
    parser.add_argument(
        "--split-k", type=int, choices=(1, 2, 4, 8, 16, 32, 64, 128), default=8
    )
    parser.add_argument(
        "--full-tiles", action="store_true",
        help="also compute the unused lower tile in one/three/four-product arms"
    )
    args = parser.parse_args()
    install(args.source_dir.resolve(), args.products, args.split_k, args.full_tiles)
    print(
        f"Installed research variant: products={args.products}, split_k={args.split_k}"
    )


if __name__ == "__main__":
    main()
