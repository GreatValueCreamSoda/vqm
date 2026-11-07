#!/usr/bin/env python3
"""Install six internal CAMBI timestamp categories in an isolated source copy.

The ordinary CAMBI bracket is replaced by non-nested Prepare, Decimate, Mode,
Contrast, Pool and Combine ranges. Five scales accumulate in the same six
categories. At 1080p/4K one V1 submission uses 44 timestamp queries and fits
even a 64-query pool. This is diagnostic instrumentation, not a speed change.
"""

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAMBI = Path("src/metrics/vmaf/cambi/vmaf_cambi.cc")
PIPELINE = Path("src/metrics/vmaf/v1/vmaf_v1_pipeline.cc")
RUNTIME = Path("src/vulkan/vulkan_runtime.cc")


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


def open_stage(name: str, indent: str) -> str:
    """Generate the stage opening instrumentation.

    :param name: Name of the coefficient array, stage or case.
    :param indent: Indentation of the inserted source lines.
    :returns: Source snippet opening the selected timing category.
    """
    return f"#ifdef VQM_STAGE_TIMING\n{indent}runtime_->OpenStage(Stage::VmafV1Cambi{name});\n#endif\n"


def close_stage(indent: str) -> str:
    """Generate the stage closing instrumentation.

    :param indent: Indentation of the inserted source lines.
    :returns: Source snippet closing the current timing category.
    """
    return f"#ifdef VQM_STAGE_TIMING\n{indent}runtime_->CloseStage();\n#endif\n"


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
    paths = (Path("src/timing.h"), Path("src/timing.cc"), CAMBI, PIPELINE, RUNTIME)
    header, timing, cambi, pipeline, runtime = ((source / p).read_text() for p in paths)

    names = ("Prepare", "Decimate", "Mode", "Contrast", "Pool", "Combine")
    header = replace_once(
        header, "    VmafV1Cambi,\n",
        "".join(f"    VmafV1Cambi{name},\n" for name in names)
    )
    timing = replace_once(
        timing, '"vmaf_v1_cambi"',
        ", ".join(f'"vmaf_v1_cambi_{name.lower()}"' for name in names)
    )
    pipeline = replace_once(
        pipeline, """        runtime.CloseStage();
        runtime.OpenStage(Stage::VmafV1Cambi);""", "        runtime.CloseStage();"
    )
    pipeline = replace_once(
        pipeline, """        runtime.CloseStage();
        runtime.OpenStage(Stage::VmafV1Motion);""",
        "        runtime.OpenStage(Stage::VmafV1Motion);"
    )

    # Contrast owns both mode dispatches and the contrast map; it is the only
    # component that splits its range internally. The composite never opens
    # an enclosing contrast range, so OpenStage cannot overwrite a live range.
    cambi = replace_once(
        cambi, """    runtime_->ComputeBarrier();
    for (std::uint32_t phase = 0; phase < 2; ++phase)""",
        open_stage("Mode", "    ") + """    runtime_->ComputeBarrier();
    for (std::uint32_t phase = 0; phase < 2; ++phase)"""
    )
    cambi = replace_once(
        cambi,
        "    // Sliding histograms bound each workgroup to 16x8 outputs and reuse\n",
        close_stage("    ") + open_stage("Contrast", "    ") +
        "    // Sliding histograms bound each workgroup to 16x8 outputs and reuse\n"
    )
    cambi = replace_once(
        cambi, """        Groups(size.height, 8), 1
    );
    runtime_->ComputeBarrier();
}""", """        Groups(size.height, 8), 1
    );
    runtime_->ComputeBarrier();
""" + close_stage("    ") + "}"
    )
    cambi = replace_once(
        cambi, "    prepare_.Record(\n",
        open_stage("Prepare", "    ") + "    prepare_.Record(\n"
    )
    cambi = replace_once(
        cambi, """        {b.input, b.prepared, b.derivative, b.mask}
    );
""", """        {b.input, b.prepared, b.derivative, b.mask}
    );
""" + close_stage("    ")
    )
    cambi = replace_once(
        cambi, "            decimate_.Record(\n",
        open_stage("Decimate", "            ") + "            decimate_.Record(\n"
    )
    cambi = replace_once(
        cambi, "            current = VmafCambiDecimate::OutputSize(current);\n",
        close_stage("            ") +
        "            current = VmafCambiDecimate::OutputSize(current);\n"
    )
    cambi = replace_once(
        cambi, "    pool_.RecordBatch(\n",
        open_stage("Pool", "    ") + "    pool_.RecordBatch(\n"
    )
    cambi = replace_once(
        cambi,
        "    const CambiCombinePush push{b.scale_scores, b.result, plan.window};\n",
        close_stage("    ") + open_stage("Combine", "    ") +
        "    const CambiCombinePush push{b.scale_scores, b.result, plan.window};\n"
    )
    cambi = replace_once(
        cambi, """    runtime_->Dispatch(combine_, &push, sizeof(push), 1, 1, 1);
    runtime_->ComputeBarrier();
}""", """    runtime_->Dispatch(combine_, &push, sizeof(push), 1, 1, 1);
    runtime_->ComputeBarrier();
""" + close_stage("    ") + "}"
    )

    capacities = re.findall(r"constexpr std::uint32_t kMaxTimestamps = (\d+);", runtime)
    if len(capacities) != 1 or int(capacities[0]) < 44:
        raise ValueError("expected one timestamp query capacity of at least 44")
    # Check all anchors, capacities and stage names before writing any file.
    assert "Stage::VmafV1Cambi)" not in pipeline
    assert cambi.count("runtime_->OpenStage(Stage::VmafV1Cambi") == 6
    assert cambi.count("runtime_->CloseStage();") == 6
    for name in names:
        assert header.count(f"VmafV1Cambi{name},") == 1
        assert timing.count(f'"vmaf_v1_cambi_{name.lower()}"') == 1
    # Runtime capacity is inspected only; the probe leaves it unchanged.
    for path, text in zip(paths[:-1], (header, timing, cambi, pipeline), strict=True):
        (source / path).write_text(text)


def main() -> None:
    """Install separate CAMBI timing categories in a source copy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    args = parser.parse_args()
    install(args.source_dir)
    print("Installed six non-nested CAMBI categories; 44 timestamps at 1080p/4K.")


if __name__ == "__main__":
    main()
