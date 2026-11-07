#!/usr/bin/env python3
"""Route the isolated direct spectral tests through the three-phase solver.

Requires solver_phases.py to have been installed in that source copy first.
Forced-legacy devices retain the unchanged one-dispatch QR shader as the
independent oracle. This changes test source only, never benchmark/library code.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def replace_once(text: str, before: str, after: str) -> str:
    """Replace one exact source anchor.

    :param text: Source text to inspect or modify.
    :param before: Source anchor to find.
    :param after: Replacement text.
    :returns: Source text with the replacement applied.
    :raises ValueError: If the source anchor does not occur exactly once.
    """
    if text.count(before) != 1:
        raise ValueError(f"expected one baseline test anchor: {before[:85]}")
    return text.replace(before, after)


def install(source: Path) -> None:
    """Install the research variant in an isolated source copy.

    :param source: Directory containing the isolated source copy.
    :raises ValueError: If the target or expected source layout is unsuitable for the
        variant.
    """
    source = source.resolve()
    if source == ROOT or source.is_relative_to(ROOT) or ROOT.is_relative_to(source):
        raise ValueError(
            "use an isolated phase-solver source copy outside the maintained checkout"
        )
    cmake = (source / "CMakeLists.txt").read_text()
    if "vqm_add_shader(vmaf_speed_stats_solver_bisect " not in cmake:
        raise ValueError("install solver_phases.py in this copy first")
    path = source / "tests/metrics/vmaf/speed/vmaf_speed_stats_test.cc"
    text = path.read_text()
    start = text.index("class EigenDevice {")
    end = text.index("\nbool IsCpuVulkanDevice()", start)
    original = text[start:end]
    changed = replace_once(
        original, "static constexpr std::uint32_t kWords = 2604;",
        "static constexpr std::uint32_t kWords = 2604 + 2 * 56;"
    )
    changed = replace_once(
        changed, """        pipeline_ = runtime_.CreatePipeline(
            force_legacy_qr ? "vmaf_speed_stats_eigen_legacy"
            : wide          ? "vmaf_speed_stats_eigen_wide"
                            : "vmaf_speed_stats_eigen"
        );""", """        phases_ = !force_legacy_qr;
        if (phases_) {
            const std::array names{
                wide ? "vmaf_speed_stats_solver_prepare_wide"
                     : "vmaf_speed_stats_solver_prepare",
                "vmaf_speed_stats_solver_bisect", "vmaf_speed_stats_solver_finish"};
            pipelines_ = runtime_.CreatePipelines(names);
        } else {
            pipelines_[0] = runtime_.CreatePipeline("vmaf_speed_stats_eigen_legacy");
        }"""
    )
    changed = replace_once(
        changed,
        "        vkDestroyPipeline(runtime_.context().device, pipeline_, nullptr);",
        "        runtime_.DestroyPipelines(pipelines_);"
    )
    changed = replace_once(
        changed, """        const std::array<std::uint32_t, 9> pc{};
        runtime_.Dispatch(pipeline_, pc.data(), sizeof(pc), 2, 2, 1);
        runtime_.ComputeBarrier();""", """        std::array<std::uint32_t, 9> pc{};
        // One U/V pair is two matrices; phases use pc.op as this pair count.
        if (phases_) pc[8] = 1;
        runtime_.Dispatch(pipelines_[0], pc.data(), sizeof(pc), 2, 2, 1);
        runtime_.ComputeBarrier();
        if (phases_) {
            runtime_.Dispatch(pipelines_[1], pc.data(), sizeof(pc), 25, 2, 1);
            runtime_.ComputeBarrier();
            runtime_.Dispatch(pipelines_[2], pc.data(), sizeof(pc), 2, 1, 1);
            runtime_.ComputeBarrier();
        }"""
    )
    changed = replace_once(
        changed, "    VkPipeline pipeline_ = VK_NULL_HANDLE;",
        "    std::array<VkPipeline, 3> pipelines_{};\n    bool phases_ = false;"
    )
    # Other GPU fixtures and the direct matrix assertions remain unchanged.
    path.write_text(text[:start] + changed + text[end:])


def main() -> None:
    """Install the phased solver comparison tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    args = parser.parse_args()
    install(args.source_dir)
    print(
        "Direct spectral/stress tests now exercise phases versus the unchanged legacy oracle"
    )


if __name__ == "__main__":
    main()
