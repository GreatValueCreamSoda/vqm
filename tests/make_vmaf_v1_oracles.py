#!/usr/bin/env python3
# Copyright 2016-2020 Netflix, Inc.
# BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
"""Generate public V1 API integration fixtures with pinned libvmaf's CLI.

    python3 tools/fetch_references.py --only vmaf
    python3 tools/build_references.py --only vmaf --jobs 8
    python3 tests/make_vmaf_v1_oracles.py > tests/metrics/vmaf/pipeline/vmaf_v1_reference.inc

The C++ test generates the same 10-bit coded samples, including nontrivial
chroma, gradients, banding, motion, and four independent frame pairs.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
W, H, FRAMES = 512, 384, 4
VIEWING_SIZE = (640, 480)
CLI = ROOT / "build/vmaf-build/tools/vmaf"
MODEL = ROOT / "build/vmaf-src/model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json"
HDR_CLI = ROOT / "build/vmaf-hdr-build/tools/vmaf"
HDR_MODEL = MODEL.parent.with_name("hdr_vmaf_v1.1.8") / "hdr_vmaf_v1.1.8_1d5h_2160.json"
FEATURE_PREFIXES = ("cambi_", "speed_chroma_uv_", "integer_adm3_", "integer_motion3_")


def sample(plane: int, x: int, y: int, frame: int, distorted: bool) -> int:
    """Compute the synthetic ten-bit code for one plane sample.

    :param plane: Planar channel index in Y, Cb, Cr order.
    :param x: Horizontal coordinate in the plane.
    :param y: Vertical coordinate in the plane.
    :param frame: Zero-based position in the clip.
    :param distorted: Generate the distorted variant.
    :returns: Clamped ten-bit sample code.
    """
    if plane == 0:
        value = 80 + ((x // 8 * 5 + y // 9 * 3 + frame * 11) % 151)
        if distorted:
            offset = 2 if (x // 23 + y // 19 + frame) % 5 == 0 else 0
            value = ((value + 3) // 8) * 8 + offset
    elif plane == 1:
        value = 512 + (x * 3 + y * 5 + frame * 29) % 241 - 120
        if distorted:
            value += (x * 5 + y * 7 + frame * 13) % 65 - 32
    else:
        value = 512 + (x * 7 + y * 2 + frame * 17) % 201 - 100
        if distorted:
            value += (x * 11 + y * 3 + frame * 9) % 49 - 24
    return max(0, min(1023, value))


def sequence(
    chroma: str, distorted: bool, size: tuple[int, int] = (W, H), count: int = FRAMES
) -> bytearray:
    """Build the packed little-endian YUV clip for an oracle case.

    :param chroma: YUV chroma subsampling code.
    :param distorted: Generate the distorted variant.
    :param size: Full luma width and height.
    :param count: Number of frames to generate.
    :returns: Packed planar samples for the complete clip.
    """
    out = bytearray()
    chroma_size = (size[0] // 2, size[1] // 2) if chroma == "420" else size
    for frame in range(count):
        for plane in range(3):
            width, height = chroma_size if plane else size
            data = [
                sample(plane, x, y, frame, distorted) for y in range(height)
                for x in range(width)
            ]
            out.extend(struct.pack(f"<{len(data)}H", *data))
    return out


def oracle(
    chroma: str, count: int, base: Path, frames: str, *, encoded8: bool = False,
    model_file: Path = MODEL, size: tuple[int, int] = (W, H),
    hdr: tuple[str, int] | None = None,
) -> None:
    """Print one C++ oracle table from the pinned VMAF CLI.

    :param chroma: YUV chroma subsampling code.
    :param count: Number of frames to score.
    :param base: Directory containing the reference and distorted input.
    :param frames: Suffix distinguishing the generated C++ table.
    :param encoded8: Use the CAMBI eight-bit encoder override.
    :param model_file: Pinned model to evaluate.
    :param size: Full luma width and height.
    :param hdr: Optional HDR input matrix and code depth.
    :raises ValueError: If the reference emits the wrong number of frames.
    :raises subprocess.CalledProcessError: If the reference command fails.
    """
    result = base / "reference.json"
    model = f"path={model_file}:name=v1" + (":cambi.enc_bitdepth=8" if encoded8 else "")
    colors = _hdr_color_options(hdr[0]) if hdr else []
    depth = hdr[1] if hdr else 10
    subprocess.run(
        [
            str(HDR_CLI if hdr else CLI), "--reference",
            str(base / "ref.yuv"), "--distorted",
            str(base / "dist.yuv"), "--width",
            str(size[0]), "--height",
            str(size[1]), "--pixel_format", chroma, "--bitdepth",
            str(depth), *colors, "--frame_cnt",
            str(count), "--model", model, "--json", "--output",
            str(result), "--threads", "1", "--quiet"
        ], check=True, capture_output=True, text=True
    )
    data = json.loads(result.read_text())
    if len(data["frames"]) != count:
        raise ValueError("reference did not score the requested frame count")
    print(f"constexpr std::array<V1Oracle, {count}> kOracle{chroma}{frames}{{{{")
    features = data["frames"][0]["metrics"]
    names = [_feature_name(features, prefix) for prefix in FEATURE_PREFIXES]
    for frame in data["frames"]:
        values = [frame["metrics"][name] for name in (*names, "v1")]
        print("    {" + ", ".join(f"{value:.6f}" for value in values) + "},")
    print("}};")


def _feature_name(metrics: dict[str, float], prefix: str) -> str:
    matches = [name for name in metrics if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one feature with prefix {prefix}")
    return matches[0]


def main() -> None:
    """Generate the maintained VMAF v1 score and feature tables.

    :raises SystemExit: If the pinned reference build is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdr", action="store_true")
    hdr = parser.parse_args().hdr
    cli, model = (HDR_CLI, HDR_MODEL) if hdr else (CLI, MODEL)
    if not cli.is_file() or not model.is_file():
        raise SystemExit("build the pinned VMAF reference first")
    print(
        "// Generated by make_vmaf_v1_oracles.py; values are libvmaf CLI's six-digit JSON."
    )
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        if hdr:
            _emit_hdr_cases(base)
            return
        for chroma in ("444", "420"):
            _emit_chroma_case(chroma, base)
        _emit_viewing_cases(base)
        _emit_hfr_cases(base)


def _emit_chroma_case(chroma: str, base: Path) -> None:
    """Generate fixtures and print the applicable tables for one chroma format.

    :param chroma: Chroma subsampling code for the generated clips.
    :param base: Directory for temporary reference and distorted clips.
    :raises OSError: If a fixture file cannot be written.
    :raises subprocess.CalledProcessError: If the reference command fails.
    :raises ValueError: If the reference emits an incorrect frame count.
    """
    (base / "ref.yuv").write_bytes(sequence(chroma, False))
    (base / "dist.yuv").write_bytes(sequence(chroma, True))
    oracle(chroma, 1, base, "Single")
    oracle(chroma, FRAMES, base, "Video")
    if chroma == "444":
        oracle(chroma, FRAMES, base, "VideoEnc8", encoded8=True)


def _emit_viewing_cases(base: Path) -> None:
    chroma = "444"
    (base / "ref.yuv").write_bytes(sequence(chroma, False, VIEWING_SIZE))
    (base / "dist.yuv").write_bytes(sequence(chroma, True, VIEWING_SIZE))
    for name, suffix in (("Close4K", "1d5h_2160"), ("Tv4K", "3d0h_2160"), ("Phone",
                                                                           "5d0h")):
        oracle(
            chroma, FRAMES, base, name, size=VIEWING_SIZE,
            model_file=MODEL.with_name(f"vmaf_v1.0.16_{suffix}.json")
        )


def _emit_hfr_cases(base: Path) -> None:
    chroma = "420"
    count = 6
    (base / "ref.yuv").write_bytes(sequence(chroma, False, count=count))
    (base / "dist.yuv").write_bytes(sequence(chroma, True, count=count))
    model = MODEL.parent.with_name("vmaf_v1.0.16_hfr") / "vmaf_v1.0.16_hfr_3d0h.json"
    for length in (2, 3, count):
        oracle(chroma, length, base, f"Hfr{length}", model_file=model)


def _hdr_color_options(matrix: str) -> list[str]:
    return [
        item for side in ("ref", "dist") for key, value in (
            ("range", "limited"), ("primaries", "bt2020"), ("trc", "pq"),
            ("matrix", matrix)
        ) for item in (f"--color_{key}_{side}", value)
    ]


def _hdr_plane(
    x: NDArray[np.int32], y: NDArray[np.int32], width: int, plane: int, frame: int,
    distorted: bool
) -> NDArray[np.int32]:
    if plane == 0:
        value = 125 + (x + y // 3 + 63 * frame) % (width // 2) * 480 // (width // 2)
        if distorted:
            value = (value + 2) // 5 * 5 + ((x // 47 + y // 53 + frame) % 5 == 0)
        return value
    value = 442 + (x // 5 * (5 + plane) + y // 7 * (7 - plane) + frame * 3) % 141
    if distorted:
        value += (x // 9 * 3 + y // 13 * 7 + frame * 7) % 21 - 10
    return value


def _hdr_frame(
    size: tuple[int, int], frame: int, depth: int, distorted: bool
) -> bytearray:
    import numpy as np  # noqa: PLC0415 - only required for HDR fixtures.

    packed = bytearray()
    for plane in range(3):
        width, height = (size[0] // 2, size[1] // 2) if plane else size
        x = np.arange(width, dtype=np.int32)[None, :]
        y = np.arange(height, dtype=np.int32)[:, None]
        value = _hdr_plane(x, y, width, plane, frame, distorted)
        if depth == 16:
            value = value * 64 + (x + 2 * y + frame) % 64
        packed.extend(value.astype("<u2").tobytes())
    return packed


def _write_hdr(
    path: Path, size: tuple[int, int], count: int, depth: int, distorted: bool
) -> None:
    with path.open("wb") as out:
        for frame in range(count):
            out.write(_hdr_frame(size, frame, depth, distorted))


def _emit_hdr_cases(base: Path) -> None:
    hfr = HDR_MODEL.parent.with_name("hdr_vmaf_v1.1.8_hfr"
                                     ) / ("hdr_vmaf_v1.1.8_hfr_1d5h_2160.json")
    for name, size, count, model, color in (("HdrNclScaled", (1280, 720), 1, HDR_MODEL,
                                             ("bt2020nc",
                                              10)), ("HdrNclHfr", (3840, 2160), 3, hfr,
                                                     ("bt2020nc", 10)),
                                            ("HdrIctcp", (3840, 2160), 3, HDR_MODEL,
                                             ("ictcp", 16)),
                                            ):
        _write_hdr(base / "ref.yuv", size, count, color[1], False)
        _write_hdr(base / "dist.yuv", size, count, color[1], True)
        oracle("420", count, base, name, model_file=model, size=size, hdr=color)


if __name__ == "__main__":
    main()
