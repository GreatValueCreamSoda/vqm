#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Generate deterministic planar YUV420 VMAF sequence fixtures.

The low-bit fixtures have matching upper eight bits in every luma sample and
are deliberately not downconverted. Files and their hashes are local build
artifacts, not checked-in upstream data.
"""
import argparse
import hashlib
import json
import struct
from collections.abc import Sequence
from pathlib import Path


def encode_plane(samples: Sequence[int], bitdepth: int) -> bytes:
    """Serialize a plane as eight-bit codes or little-endian words.

    :param samples: Unsigned plane sample codes.
    :param bitdepth: Declared number of bits per sample.
    :returns: Serialized plane samples.
    """
    if bitdepth == 8:
        return bytes(samples)
    return struct.pack(f"<{len(samples)}H", *samples)


def _luma_pair(width: int, height: int, shift: int, frame: int,
               low_bits: bool) -> tuple[list[int], list[int]]:
    """Build the original and distorted luma codes for one frame.

    :param width: Image width in pixels.
    :param height: Image height in pixels.
    :param shift: Number of bits below the upper eight.
    :param frame: Zero-based frame index.
    :param low_bits: Whether distortion affects only the low bits.
    :returns: Original and distorted sample lists in raster order.
    """
    original, distorted = [], []
    for y in range(height):
        for x in range(width):
            base = 16 + (
                7 * x + 13 * y + 17 * frame + 29 * ((x // 7 + frame) % 2)
            ) % 220
            code = base << shift
            changed = (
                code + 1 + (x + y + frame) % ((1 << shift) - 1) if low_bits else
                min(235 << shift, code + ((x + y + frame) % 9 << shift))
            )
            original.append(code)
            distorted.append(changed)
    return original, distorted


def make_pair(width: int, height: int, bitdepth: int, frames: int,
              low_bits: bool) -> tuple[bytes, bytes]:
    """Build two deterministic YUV420 clips for one bit-depth case.

    :param width: Image width in pixels.
    :param height: Image height in pixels.
    :param bitdepth: Declared number of bits per sample.
    :param frames: Number of complete video frames.
    :param low_bits: Keep distortions in bits below the upper eight.
    :returns: Original and distorted clips.
    """
    shift = bitdepth - 8
    neutral = 128 << shift
    chroma = encode_plane([neutral] * (width * height // 4), bitdepth) * 2
    reference, distorted = bytearray(), bytearray()
    for frame in range(frames):
        ref_y, dist_y = _luma_pair(width, height, shift, frame, low_bits)
        reference.extend(encode_plane(ref_y, bitdepth))
        distorted.extend(encode_plane(dist_y, bitdepth))
        reference.extend(chroma)
        distorted.extend(chroma)
    return bytes(reference), bytes(distorted)


def main() -> None:
    """Write the YUV fixtures and a manifest of their hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/fixtures/vmaf")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--frames", type=int, default=4)
    args = parser.parse_args()
    if args.width < 16 or args.height < 16 or args.width % 2 or args.height % 2:
        parser.error("width and height must be even and at least 16")
    if args.frames < 1:
        parser.error("frames must be positive")
    directory = Path(args.out_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cases = []
    for depth in (8, 10, 12):
        for label, low_bits in [("moving", False),
                                *([("low-bits", True)] if depth > 8 else [])]:
            name = f"{depth}bit-{label}"
            ref, dist = make_pair(args.width, args.height, depth, args.frames, low_bits)
            ref_path, dist_path = directory / f"{name}-ref.yuv", directory / f"{name}-dist.yuv"
            ref_path.write_bytes(ref)
            dist_path.write_bytes(dist)
            cases.append(
                {
                    "name": name,
                    "width": args.width,
                    "height": args.height,
                    "bitdepth": depth,
                    "pixel_format": "420",
                    "frames": args.frames,
                    "reference": ref_path.name,
                    "distorted": dist_path.name,
                    "reference_sha256": hashlib.sha256(ref).hexdigest(),
                    "distorted_sha256": hashlib.sha256(dist).hexdigest()
                }
            )
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    print(manifest)


if __name__ == "__main__":
    main()
