#!/usr/bin/env python3
"""Install an exact CAMBI final-scan shortcut in a fresh isolated source copy.

The selector can prove the sum strictly above the selected threshold is zero.
After its histogram is dead, bin zero carries that proof to the partial pass.
Canonical threshold/rank state, final arithmetic, layouts and dispatches stay
unchanged. The maintained checkout is refused deliberately.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
POOL = Path("src/metrics/vmaf/cambi/shaders/pool")


def replace_once(text: str, before: str, after: str) -> str:
    """Replace one exact source anchor.

    :param text: Source text to inspect or modify.
    :param before: Source anchor to find.
    :param after: Replacement text.
    :returns: Source text with the replacement applied.
    :raises ValueError: If the source anchor does not occur exactly once.
    """
    if text.count(before) != 1:
        raise ValueError(f"expected one unchanged shader anchor: {before[:80]}")
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
    selector_path = source / POOL / "vmaf_cambi_pool_select.comp"
    partial_path = source / POOL / "vmaf_cambi_pool_partial.comp"
    selector = selector_path.read_text()
    partial = partial_path.read_text()
    selector = replace_once(
        selector, "shared uint rank_snapshot;", "shared uint rank_snapshot, zero_sum;"
    )
    selector = replace_once(
        selector, "    if (lane == 0u) rank_snapshot = data[a.state + 1u];",
        """    if (lane == 0u) {
        rank_snapshot = data[a.state + 1u];
        zero_sum = 0u;
    }"""
    )
    selector = replace_once(
        selector, """    if (rank_snapshot == 0u) {
        data[a.histogram + lane] = 0u;
        return;
    }""", """    if (rank_snapshot == 0u) {
        // No histogram work remains. Bin zero carries the optional exact
        // zero-sum proof from an earlier completed selection pass.
        if (lane != 0u) data[a.histogram + lane] = 0u;
        return;
    }"""
    )
    selector = replace_once(
        selector, "            zero_count = a.count - other;",
        """            zero_count = a.count - other;
            // The first scan counted every nonzero high byte and ORed all
            // tiny positive keys into prefix. Together these prove all-zero.
            zero_sum = uint(other == 0u && prefix == 0u);"""
    )
    selector = replace_once(
        selector, "                prefix |= bin << (a.pass * 8u);",
        """                prefix |= bin << (a.pass * 8u);
                // At the final byte the key is exact. Rank is the original
                // selected count minus all strictly greater keys. Equality
                // therefore proves every selected value is a threshold tie.
                if (a.pass == 0u && rank == a.selected) zero_sum = 1u;"""
    )
    selector = replace_once(
        selector, "    data[a.histogram + lane] = 0u;",
        """    // Active intermediate passes leave a clean histogram. Completed
    // zero maps and the final pass may repurpose dead bin zero as a proof;
    // the initial clear resets it before the next pooling operation.
    data[a.histogram + lane] = lane == 0u ? zero_sum : 0u;"""
    )
    partial = replace_once(
        partial, """    uint lane = gl_LocalInvocationIndex;
    float value = 0.0;""", """    uint lane = gl_LocalInvocationIndex;
    if (data[a.histogram] != 0u) {
        // Selector proved there are no values strictly above threshold.
        // The original scan and add tree would produce positive zero.
        if (lane == 0u) data[a.partials + a.group] = floatBitsToUint(0.0);
        return;
    }
    float value = 0.0;"""
    )
    # Validate every fragile anchor before writing either source file.
    selector_path.write_text(selector)
    partial_path.write_text(partial)


def main() -> None:
    """Install the CAMBI zero-sum scan shortcut."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    args = parser.parse_args()
    install(args.source_dir)
    print("Installed exact CAMBI zero-above-threshold scan shortcut.")


if __name__ == "__main__":
    main()
