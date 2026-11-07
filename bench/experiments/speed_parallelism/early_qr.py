#!/usr/bin/env python3
"""Install an early SpEED QR selection experiment in an isolated source copy.

The prefix mode uses a cheap Rayleigh gate, then certifies fallback using a
short prefix of the existing rank-zero bisection. Rayleigh mode skips that
certificate and is a heuristic control. Neither mode changes Householder or
legacy QR arithmetic. Production-source paths are deliberately refused.
"""

import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SHADER = Path("src/metrics/vmaf/speed/shaders/vmaf_speed_stats_eigen.comp")


def replace_once(text: str, old: str, new: str) -> str:
    """Replace one exact source anchor.

    :param text: Source text to inspect or modify.
    :param old: Source anchor to find.
    :param new: Replacement text.
    :returns: Source text with the replacement applied.
    :raises ValueError: If the source anchor does not occur exactly once.
    """
    if text.count(old) != 1:
        raise ValueError(f"expected one unchanged shader anchor: {old[:70]}")
    return text.replace(old, new)


def install(
    source: Path, mode: str, margin_scale: float, probe_iterations: int,
    gate: str = "diagonal"
) -> None:
    """Install the research variant in an isolated source copy.

    :param source: Directory containing the isolated source copy.
    :param mode: Numerical or installation mode.
    :param margin_scale: Scale of the safety margin for certification.
    :param probe_iterations: Maximum number of prefix iterations.
    :param gate: Rule used to decide whether to probe the QR branch.
    :raises ValueError: If the target or expected source layout is unsuitable for the
        variant.
    """
    if source == ROOT or source.is_relative_to(ROOT) or ROOT.is_relative_to(source):
        raise ValueError("use an isolated source copy outside the maintained checkout")
    if mode == "rayleigh" and gate != "diagonal":
        raise ValueError("rayleigh mode requires the diagonal gate")
    path = source / SHADER
    text = path.read_text()
    text = replace_once(
        text, "shared uint fallback;", "shared uint fallback, early_qr;"
    )
    selection = """    if (lane == 0u) {
        early_qr = 0u;
        if (bounds_valid) {
            float smallest_diagonal = diagonal[0];
            for (uint row = 1u; row < 25u; ++row)
                smallest_diagonal = min(smallest_diagonal, diagonal[row]);
            float scale = max(abs(lower_bound), abs(upper_bound));
            float guard = 1e-6 + 1e-3 * scale;
            float margin = MARGIN * scale;
            // Rayleigh's principle bounds the exact stored tridiagonal's
            // smallest eigenvalue. FP32 Sturm behavior needs a separate check.
            if (smallest_diagonal < guard - margin) {
SELECTION
            }
        }
    }
""".replace("MARGIN", f"{margin_scale:.9e}")
    if mode == "rayleigh":
        body = """                // Heuristic control: unlike prefix mode, this does not
                // certify the current FP32 bisection's eventual fallback.
                early_qr = 1u;"""
    else:
        body = """                float lo = lower_bound, hi = upper_bound;
                for (uint probe = 0u; probe < ITERATIONSu; ++probe) {
                    // Once rank zero's upper endpoint enters the existing
                    // guard, continuing that search cannot leave the guard.
                    if (hi <= guard) {
                        early_qr = 1u;
                        break;
                    }
                    float mid = lo * 0.5 + hi * 0.5;
                    if (mid == lo || mid == hi) {
                        if (mid <= guard) early_qr = 1u;
                        break;
                    }
                    uint count = sturm_count(mid);
                    if (count > 25u) {
                        // The unchanged rank-zero search would report a
                        // failed bisection and select the same QR fallback.
                        early_qr = 1u;
                        break;
                    }
                    if (count == 0u)
                        lo = mid;
                    else
                        hi = mid;
                }
                if (hi <= guard) early_qr = 1u;""".replace(
            "ITERATIONS", str(probe_iterations)
        )
    selection = selection.replace("SELECTION", body)
    if gate == "sturm":
        selection = selection.replace(
            "            if (smallest_diagonal < guard - margin) {",
            "            uint guard_count = sturm_count(guard);\n"
            "            if (guard_count > 0u && guard_count <= 25u) {"
        )
    elif gate == "always":
        selection = selection.replace(
            "            if (smallest_diagonal < guard - margin) {",
            "            if (true) {"
        )
    elif gate != "diagonal":
        raise ValueError("gate must be diagonal, sturm or always")
    anchor = """    bool bounds_valid = failed == 0u;
    barrier();
    if (lane < 25u && bounds_valid) {"""
    replacement = """    bool bounds_valid = failed == 0u;
""" + selection + """    barrier();
    if (lane < 25u && bounds_valid && early_qr == 0u) {"""
    text = replace_once(text, anchor, replacement)
    # Every modification is validated before this single source-file write.
    path.write_text(text)


def main() -> None:
    """Install the selected early QR branch experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--mode", choices=("prefix", "rayleigh"), default="prefix")
    parser.add_argument("--margin-scale", type=float, default=1e-6)
    parser.add_argument("--probe-iterations", type=int, default=16)
    parser.add_argument(
        "--gate", choices=("diagonal", "sturm", "always"), default="diagonal",
        help="controls whether to try the prefix certificate"
    )
    args = parser.parse_args()
    if not math.isfinite(args.margin_scale) or args.margin_scale < 0:
        parser.error("--margin-scale must be finite and nonnegative")
    if args.probe_iterations < 1 or args.probe_iterations > 320:
        parser.error("--probe-iterations must be in 1..320")
    if args.mode == "rayleigh" and args.gate != "diagonal":
        parser.error("--mode rayleigh requires --gate diagonal")
    install(
        args.source_dir.resolve(), args.mode, args.margin_scale, args.probe_iterations,
        args.gate
    )
    print(
        f"Installed early-QR {args.mode} arm, margin={args.margin_scale:g}, "
        f"probe_iterations={args.probe_iterations}, gate={args.gate}."
    )


if __name__ == "__main__":
    main()
