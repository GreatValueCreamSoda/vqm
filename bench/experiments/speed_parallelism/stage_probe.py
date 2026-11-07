#!/usr/bin/env python3
"""Add diagnostic SpEED GPU timestamps to an isolated source copy.

Extra query boundaries perturb execution. Use this binary for attribution;
compare optimizations using ordinary vqm_bench binaries without these probes.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAGES = ("Filter", "Means", "Covariance", "Solve", "Pool", "Reduce", "Combine")


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
            f"baseline anchor missing or repeated in {path}: {before[:70]}"
        )
    path.write_text(text.replace(before, after))


def bracket(code: str, stage: str) -> str:
    """Wrap one source snippet with stage timing calls.

    :param code: Source snippet to bracket.
    :param stage: Timing stage identifier.
    :returns: Source snippet with surrounding stage timing calls.
    """
    return (
        "#ifdef VQM_STAGE_TIMING\n"
        f"    runtime_->OpenStage(Stage::VmafV1Speed{stage});\n"
        "#endif\n" + code + "\n#ifdef VQM_STAGE_TIMING\n"
        "    runtime_->CloseStage();\n#endif"
    )


def main() -> None:
    """Install additional solver and covariance stage timers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    if source == ROOT or source.is_relative_to(ROOT) or ROOT.is_relative_to(source):
        parser.error("use an isolated copy outside the maintained checkout")
    replace_once(
        source / "src/timing.h", "    VmafV1Speed,",
        "    VmafV1Speed,\n" + "".join(f"    VmafV1Speed{x},\n"
                                       for x in STAGES).rstrip()
    )
    replace_once(
        source / "src/timing.cc", '"vmaf_v1_speed",',
        '"vmaf_v1_speed", ' + ", ".join(f'"speed_{x.lower()}"' for x in STAGES) + ","
    )
    replace_once(
        source / "src/metrics/vmaf/v1/vmaf_v1_pipeline.cc",
        "        runtime.OpenStage(Stage::VmafV1Speed);\n", ""
    )
    host = source / "src/metrics/vmaf/speed/vmaf_speed.cc"
    phases = {
        "Filter":
        "    filter_.RecordBatch(size, planes);",
        "Means":
        "    runtime_->Dispatch(pipelines_[0], &pc, sizeof(pc), 50 * channels, 1, 1);\n    runtime_->ComputeBarrier();",
        "Covariance":
        "    runtime_->Dispatch(pipelines_[0], &pc, sizeof(pc), 650 * channels, 1, 1);\n    runtime_->ComputeBarrier();",
        "Solve":
        "    runtime_->Dispatch(pipelines_[1], &pc, sizeof(pc), 2 * channels, 2, 1);\n    runtime_->ComputeBarrier();",
        "Pool":
        "    runtime_->Dispatch(\n        pipelines_[2], &pc, sizeof(pc), static_cast<std::uint32_t>(groups),\n        channels, 1\n    );\n    runtime_->ComputeBarrier();",
        "Reduce":
        "    runtime_->Dispatch(pipelines_[3], &pc, sizeof(pc), channels, 1, 1);\n    runtime_->ComputeBarrier();",
        "Combine":
        "    runtime_->Dispatch(combine_, &push, sizeof(push), 1, 1, 1);\n    runtime_->ComputeBarrier();",
    }
    for stage, code in phases.items():
        replace_once(host, code, bracket(code, stage))
    print("Installed seven diagnostic GPU timestamp brackets.")


if __name__ == "__main__":
    main()
