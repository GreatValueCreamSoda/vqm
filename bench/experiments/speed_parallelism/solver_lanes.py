#!/usr/bin/env python3
"""Change only the current SpEED wide-solver lane count in an isolated copy.

This preserves the shader's arithmetic expressions, scratch layout, dispatch
count and workgroup grid. It tests whether the current solver benefits from the
512-lane choice inherited from an older numerical implementation.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ANCHOR = (
    "vqm_add_shader(vmaf_speed_stats_eigen_wide "
    "metrics/vmaf/speed/shaders/vmaf_speed_stats_eigen.comp "
    "-DVQM_SPEED_EIGEN_LANES=512)"
)


def install(source: Path, lanes: int) -> None:
    """Install the research variant in an isolated source copy.

    :param source: Directory containing the isolated source copy.
    :param lanes: Workgroup lane count.
    :raises ValueError: If the target or expected source layout is unsuitable for the
        variant.
    """
    source = source.resolve()
    if source == ROOT or source.is_relative_to(ROOT) or ROOT.is_relative_to(source):
        raise ValueError("use a fresh source copy outside the maintained checkout")
    cmake = source / "CMakeLists.txt"
    text = cmake.read_text()
    if text.count(ANCHOR) != 1:
        raise ValueError("expected one unchanged wide-solver declaration")
    cmake.write_text(text.replace(ANCHOR, ANCHOR.replace("=512)", f"={lanes})")))


def main() -> None:
    """Set the experimental wide solver lane count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument(
        "--lanes", type=int, choices=(32, 64, 128, 256, 512), required=True
    )
    args = parser.parse_args()
    install(args.source_dir, args.lanes)
    print(f"Current SpEED wide-solver variant uses {args.lanes} invocations")


if __name__ == "__main__":
    main()
