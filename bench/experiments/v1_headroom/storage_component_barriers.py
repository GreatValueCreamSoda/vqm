#!/usr/bin/env python3
"""Remove repeated CAMBI component-entry dependencies in an isolated copy.

Each producer's existing trailing compute barrier remains. The composite can
therefore skip its immediately following consumer's entry barrier. Standalone
callers retain both boundaries by default. This arm changes no shader/layout.
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
    header_path, host_path = (
        source / CAMBI / name for name in ("vmaf_cambi.h", "vmaf_cambi.cc")
    )
    header, host = header_path.read_text(), host_path.read_text()
    header = replace_once(
        header, "    void Record(Size size, const VmafCambiDecimateBuffers& buffers);",
        """    void Record(Size size, const VmafCambiDecimateBuffers& buffers,
                bool synchronize_input = true);"""
    )
    header = replace_once(
        header,
        "    void Record(Size size, unsigned window, const VmafCambiContrastBuffers& b);",
        """    void Record(Size size, unsigned window, const VmafCambiContrastBuffers& b,
                bool synchronize_input = true);"""
    )
    header = replace_once(
        header, "        bool allow_packed = true\n",
        "        bool allow_packed = true, bool synchronize_input = true\n"
    )
    host = replace_once(
        host,
        "void VmafCambiDecimate::Record(Size size, const VmafCambiDecimateBuffers& b) {",
        """void VmafCambiDecimate::Record(Size size, const VmafCambiDecimateBuffers& b,
                               bool synchronize_input) {"""
    )
    host = replace_once(
        host, """                                 b.mask_out};
    runtime_->ComputeBarrier();""", """                                 b.mask_out};
    if (synchronize_input) runtime_->ComputeBarrier();"""
    )
    host = replace_once(
        host, """void VmafCambiContrast::Record(
    Size size, unsigned window, const VmafCambiContrastBuffers& b
)""", """void VmafCambiContrast::Record(
    Size size, unsigned window, const VmafCambiContrastBuffers& b,
    bool synchronize_input
)"""
    )
    host = replace_once(
        host, """    runtime_->ComputeBarrier();
    for (std::uint32_t phase = 0; phase < 2; ++phase)""",
        """    if (synchronize_input) runtime_->ComputeBarrier();
    for (std::uint32_t phase = 0; phase < 2; ++phase)"""
    )
    host = replace_once(
        host,
        "    const VmafCambiPoolBuffers& buffers, double ratio, bool allow_packed\n",
        """    const VmafCambiPoolBuffers& buffers, double ratio, bool allow_packed,
    bool synchronize_input
"""
    )
    host = replace_once(
        host, """    const std::uint32_t packed_bit = packed ? 256 : 0;
    runtime_->ComputeBarrier();""",
        """    const std::uint32_t packed_bit = packed ? 256 : 0;
    if (synchronize_input) runtime_->ComputeBarrier();"""
    )
    host = replace_once(
        host,
        "                {pixels_src, mask_src, b.scale_input, b.scale_mask[mask_bank]}",
        "                {pixels_src, mask_src, b.scale_input, b.scale_mask[mask_bank]}, false"
    )
    host = replace_once(
        host, """            {pixels_src, mask_src, b.contrast_horizontal, b.filtered,
             contrast_offset}
""", """            {pixels_src, mask_src, b.contrast_horizontal, b.filtered,
             contrast_offset}, false
"""
    )
    host = replace_once(
        host, """        plan.scale_pixels, {b.contrast, b.pool_histogram, b.pool_state,
                            b.pool_partials, b.scale_scores}
""", """        plan.scale_pixels, {b.contrast, b.pool_histogram, b.pool_state,
                            b.pool_partials, b.scale_scores}, 0.6, true, false
"""
    )
    host = replace_once(
        host, "    auto current = size;\n",
        """    // Every component's trailing barrier publishes its output. The next
    // component skips only its redundant entry barrier; standalone calls
    // retain their existing complete synchronization contracts by default.
    auto current = size;
"""
    )
    # Check every anchor before writing either file.
    header_path.write_text(header)
    host_path.write_text(host)


def main() -> None:
    """Install the CAMBI component barrier experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    args = parser.parse_args()
    install(args.source_dir)
    print("Installed composite-only CAMBI component-entry barrier skip.")


if __name__ == "__main__":
    main()
