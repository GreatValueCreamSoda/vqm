#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Generate the deterministic 64x64 Butteraugli drift diagnostic pair."""

import argparse
from pathlib import Path

from PIL import Image

DEFAULT_OUTPUT = Path("artifacts/reports/butteraugli-drift/synthetic")


def _clamp(value: int) -> int:
    """Clamp an integer to the byte sample range.

    :param value: Sample before clamping.
    :returns: A value between zero and 255.
    """
    return max(0, min(255, value))


def _pixels(x: int, y: int) -> tuple[bytes, bytes]:
    """Construct the original and distorted RGB samples at one coordinate.

    :param x: Zero-based column.
    :param y: Zero-based row.
    :returns: Two three-byte RGB samples.
    """
    value = 64 + 128 * ((x // 3 + y // 3) % 2)
    original = (value, _clamp(value + x % 7 - 3), _clamp(value + y % 5 - 2))
    noise = (13 * x + 7 * y + 3 * x * y) % 11 - 5
    change = noise + (5 if (x // 8 + y // 8) % 2 else -5)
    distorted = (_clamp(channel + change) for channel in original)
    return bytes(original), bytes(distorted)


def _write_images(output: Path) -> tuple[Path, Path]:
    """Write both opaque RGB PNG images using only integer pixel formulas.

    :param output: Existing directory that receives the images.
    :returns: Original and distorted image paths.
    :raises OSError: If an output image cannot be written.
    """
    original = bytearray()
    distorted = bytearray()
    for y in range(64):
        for x in range(64):
            first, second = _pixels(x, y)
            original.extend(first)
            distorted.extend(second)
    original_path = output / "64-checker-a.png"
    distorted_path = output / "64-checker-b.png"
    Image.frombytes("RGB", (64, 64), bytes(original)).save(original_path)
    Image.frombytes("RGB", (64, 64), bytes(distorted)).save(distorted_path)
    return original_path, distorted_path


def main() -> None:
    """Write a diagnostic image pair and a tab-separated pair list.

    :raises OSError: If the output directory or files cannot be written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    options = parser.parse_args()
    options.out_dir.mkdir(parents=True, exist_ok=True)
    original, distorted = _write_images(options.out_dir)
    pair_list = options.out_dir / "checker.pairs"
    pair_list.write_text(f"{original}\t{distorted}\n", encoding="utf-8")
    print(pair_list)


if __name__ == "__main__":
    main()
