#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Write the benchmark corpus: CID22 pairs plus synthetic sizes.

The CID22 part is the first --count pairs of the validation set (all 512x512).
The synthetic part is one pair per size, `testsrc2` against a bicubic
down/up-scaled copy of itself, so the benchmark covers geometries the corpus
does not. Pair lists are tab-separated absolute paths under --out-dir, and
all.txt concatenates them.

Missing CID22 or FFmpeg is reported and skipped. --synthetic-only skips the
corpus entirely. Exit 1 if no pair list could be written. Invalid synthetic
geometries are rejected before any output is written.
"""

import argparse
import os
import shutil
import subprocess
import sys
from typing import TextIO


def parse_args() -> argparse.Namespace:
    """Parse corpus selection and synthetic generation options.

    :returns: Requested pair list settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default="validation/data/CID22_validation_set"
    )
    parser.add_argument("--out-dir", default=".cache/bench/pairs")
    parser.add_argument(
        "--count", type=int, default=100, help="CID22 pairs to list (0 = all)"
    )
    parser.add_argument(
        "--synthetic-only", action="store_true",
        help="do not read or include the CID22 corpus"
    )
    parser.add_argument(
        "--sizes", default="512x512,1920x1080,3840x2160", help=(
            "comma-separated WxH synthetic pairs with positive, even dimensions"
        )
    )
    parser.add_argument(
        "--force", action="store_true", help="regenerate the synthetic images"
    )
    return parser.parse_args()


def write_pair_file(path: str, pairs: list[tuple[str, str]]) -> int:
    """Write a tab-separated benchmark pair list.

    :param path: Destination list path.
    :param pairs: Original and distorted paths in benchmark order.
    :returns: Number of pairs written.
    :raises OSError: If the list cannot be written.
    """
    with open(path, "w") as f:
        for original, distorted in pairs:
            f.write(f"{original}\t{distorted}\n")
    print(path)
    return len(pairs)


def _parse_size(size: str) -> tuple[int, int]:
    """Parse a geometry that FFmpeg can generate without rounding it down.

    :param size: Geometry from the synthetic size list.
    :returns: Positive even width and height.
    :raises SystemExit: If the geometry is malformed or has invalid dimensions.
    """
    try:
        width, height = (int(part) for part in size.lower().split("x"))
    except ValueError:
        raise SystemExit(f"bad --sizes entry {size!r}, expected WxH") from None
    if min(width, height) <= 0 or width % 2 or height % 2:
        raise SystemExit(
            f"bad --sizes entry {size!r}, dimensions must be positive and even"
        )
    return width, height


def synthetic_pairs(out_dir: str, sizes: list[tuple[int, int]],
                    force: bool) -> list[str]:
    """Generate and list one synthetic pair per geometry.

    :param out_dir: Directory for images and pair lists.
    :param sizes: Validated width and height pairs.
    :param force: Whether to regenerate existing images.
    :returns: Written pair list paths, empty when FFmpeg is unavailable.
    :raises subprocess.CalledProcessError: If image generation fails.
    """
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found: skipping synthetic pairs")
        return []
    written = []
    for width, height in sizes:
        directory = os.path.join(out_dir, f"synthetic-{width}x{height}")
        os.makedirs(directory, exist_ok=True)
        original = os.path.join(directory, "orig.png")
        distorted = os.path.join(directory, "dist.png")
        if force or not os.path.isfile(original):
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc2=size={width}x{height}:rate=1", "-frames:v", "1",
                    "-pix_fmt", "rgb24", original
                ], check=True
            )
        if force or not os.path.isfile(distorted):
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc2=size={width}x{height}:rate=1", "-vf",
                    "scale=iw/2:ih/2:flags=bicubic,"
                    "scale=iw*2:ih*2:flags=bicubic", "-frames:v", "1",
                    "-pix_fmt", "rgb24", distorted
                ], check=True
            )
        pair_file = os.path.join(out_dir, f"synthetic-{width}x{height}.txt")
        write_pair_file(pair_file, [(original, distorted)])
        written.append(pair_file)
    return written


def _cid22_pairs(args: argparse.Namespace) -> list[str]:
    """Write the selected CID22 pair list if the corpus is available."""
    if args.synthetic_only:
        return []
    if not os.path.isdir(args.data_root):
        print("CID22 missing: run python3 validation/fetch_cid22.py")
        return []
    # Synthetic-only runs need no NumPy or validation dependencies.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    from validation.common import enumerate_pairs

    pairs = enumerate_pairs(args.data_root)
    if args.count:
        pairs = pairs[:args.count]
    absolute = [
        (
            os.path.abspath(os.path.join(args.data_root, original)),
            os.path.abspath(os.path.join(args.data_root, distorted))
        ) for original, distorted in pairs
    ]
    pair_file = os.path.join(args.out_dir, f"cid22-{len(absolute)}.txt")
    write_pair_file(pair_file, absolute)
    return [pair_file]


def _append_pairs(out: TextIO, path: str) -> int:
    """Append a pair list to the combined output and count its rows."""
    total = 0
    with open(path) as source:
        for line in source:
            out.write(line)
            total += 1
    return total


def main() -> int:
    """Prepare the benchmark's combined and per-geometry pair lists.

    :returns: Zero when at least one list was written.
    :raises SystemExit: If a requested synthetic size is invalid.
    """
    args = parse_args()
    sizes = [
        _parse_size(size) for part in args.sizes.split(",")
        if (size := part.strip())
    ]
    os.makedirs(args.out_dir, exist_ok=True)
    pair_files = _cid22_pairs(args)
    pair_files.extend(synthetic_pairs(args.out_dir, sizes, args.force))

    if not pair_files:
        print("no pair list could be written", file=sys.stderr)
        return 1

    combined = os.path.join(args.out_dir, "all.txt")
    with open(combined, "w") as out:
        total = sum(_append_pairs(out, path) for path in pair_files)
    print(combined)
    print(f"{total} pair(s) in {len(pair_files)} list(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
