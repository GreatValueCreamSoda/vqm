#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Dev-only generator for the CLI fixtures and the golden scores.

Writes synthetic PNG pairs to build/fixtures and, when --ref-binary is given,
prints the reference scores to copy into
tests/metrics/ssimulacra2/pipeline_test.cc. Not run by
ctest: the C++ test builds the same pixel data in memory.
"""

import argparse
import os
import struct
import subprocess
import zlib
from collections.abc import Iterable
from pathlib import Path

WIDTH = 32
HEIGHT = 32


def write_png(
    path: str | Path, width: int, height: int, channels: int,
    rows: Iterable[Iterable[int]]
) -> None:
    """Write an eight-bit RGB or RGBA PNG from packed rows.

    :param path: Destination or source filesystem path.
    :param width: Image width in pixels.
    :param height: Image height in pixels.
    :param channels: Number of interleaved RGB or RGBA channels.
    :param rows: Rows of packed eight-bit channel values.
    :raises KeyError: If the channel count is not RGB or RGBA.
    :raises OSError: If the file cannot be written.
    """
    color_type = {3: 2, 4: 6}[channels]

    def chunk(tag: bytes, data: bytes) -> bytes:
        out = struct.pack(">I", len(data)) + tag + data
        return out + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    )
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def base_pixels(x: int, y: int) -> tuple[int, int, int]:
    """Return the RGB gradient samples at one pixel.

    :param x: Horizontal coordinate in the plane.
    :param y: Vertical coordinate in the plane.
    :returns: Red, green and blue sample codes.
    """
    r8 = (x * 255) // (WIDTH - 1)
    g8 = (y * 255) // (HEIGHT - 1)
    b8 = ((x * 7 + y * 11) * 13) & 255
    return r8, g8, b8


def gradient() -> list[list[int]]:
    """Generate rows for the undistorted RGB fixture.

    :returns: Rows of interleaved channel codes.
    """
    rows = []
    for y in range(HEIGHT):
        row = []
        for x in range(WIDTH):
            row.extend(base_pixels(x, y))
        rows.append(row)
    return rows


def gradient_b() -> list[list[int]]:
    """Generate a checkerboard increase in the red channel.

    :returns: Rows of interleaved channel codes.
    """
    rows = []
    for y in range(HEIGHT):
        row = []
        for x in range(WIDTH):
            r8, g8, b8 = base_pixels(x, y)
            r8 = min(r8 + 16, 255) if ((x + y) & 1) == 0 else r8
            row.extend((r8, g8, b8))
        rows.append(row)
    return rows


def gradient_c() -> list[list[int]]:
    """Generate a gradient with the lowest three bits removed.

    :returns: Rows of interleaved channel codes.
    """
    rows = []
    for y in range(HEIGHT):
        row = []
        for x in range(WIDTH):
            r8, g8, b8 = base_pixels(x, y)
            row.extend((r8 & 0xF8, g8 & 0xF8, b8 & 0xF8))
        rows.append(row)
    return rows


def alpha_pixels(x: int, y: int) -> tuple[int, int, int]:
    """Return the RGB samples for an alpha fixture pixel.

    :param x: Horizontal coordinate in the plane.
    :param y: Vertical coordinate in the plane.
    :returns: Red, green and blue sample codes.
    """
    r8 = (x * 255) // (WIDTH - 1)
    g8 = (y * 255) // (HEIGHT - 1)
    b8 = (x * y) & 255
    return r8, g8, b8


def alpha_a() -> list[list[int]]:
    """Generate the reference RGBA gradient and opacity ramp.

    :returns: Rows of interleaved channel codes.
    """
    rows = []
    for y in range(HEIGHT):
        row = []
        for x in range(WIDTH):
            r8, g8, b8 = alpha_pixels(x, y)
            a8 = (x * 255) // (WIDTH - 1)
            row.extend((r8, g8, b8, a8))
        rows.append(row)
    return rows


def alpha_b() -> list[list[int]]:
    """Generate the distorted RGBA gradient with the same opacity ramp.

    :returns: Rows of interleaved channel codes.
    """
    rows = []
    for y in range(HEIGHT):
        row = []
        for x in range(WIDTH):
            r8, g8, b8 = alpha_pixels(x, y)
            a8 = (x * 255) // (WIDTH - 1)
            row.extend((255 - r8, g8, (b8 + 64) & 255, a8))
        rows.append(row)
    return rows


def main() -> None:
    """Write local fixtures and optionally print pinned reference scores."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="build/fixtures")
    parser.add_argument(
        "--ref-binary", default=None,
        help="reference ssimulacra2 binary to score the pairs"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fixtures = {
        "gradient.png": (3, gradient()),
        "gradient_B.png": (3, gradient_b()),
        "gradient_C.png": (3, gradient_c()),
        "alpha_A.png": (4, alpha_a()),
        "alpha_B.png": (4, alpha_b()),
    }
    for name, (channels, rows) in fixtures.items():
        path = os.path.join(args.out_dir, name)
        write_png(path, WIDTH, HEIGHT, channels, rows)
        print("wrote", path)

    if args.ref_binary:
        pairs = [
            ("gradient.png", "gradient_B.png"),
            ("gradient.png", "gradient_C.png"), ("alpha_A.png", "alpha_B.png")
        ]
        for orig, dist in pairs:
            result = subprocess.run(
                [
                    args.ref_binary,
                    os.path.join(args.out_dir, orig),
                    os.path.join(args.out_dir, dist)
                ], capture_output=True, text=True
            )
            score = result.stdout.strip().splitlines()[-1]
            print(f"reference {orig} vs {dist}: {score}")


if __name__ == "__main__":
    main()
