# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Score complete clips using the pinned original spatial metric executables.

Both pinned readers interpret PFM pixels as encoded sRGB through their default
ColorHints. This adapter writes the independently decoded RGB float32 samples
without integer quantization, then evaluates every frame through the existing
Cloudinary or libjxl in-process batch harness. The clip result is the arithmetic
mean of those original frame scores, matching the public CLI's pooling policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

from .corpus import ROOT
from .corpus import digest

PINS = {
    "ssimulacra2": "81feacf6a5b0bbd7ec11b4a881a4615033ab0b10",
    "butteraugli": "e47d2a2b5bb418f2947ce07f36cb048861db3820",
}
STEPS = {"ssimulacra2": "cloudinary", "butteraugli": "libjxl"}
BINARIES = {
    "ssimulacra2": "build/reference-build/ss2_cloudinary_bench",
    "butteraugli": "build/libjxl-build/tools/ss2_libjxl_bench",
}
HARNESSES = {
    "ssimulacra2": "bench/inmemory/cloudinary_bench.cc",
    "butteraugli": "bench/inmemory/libjxl_bench.cc",
}
FloatImage = NDArray[np.float32]

# Decoder semantics belong to the pinned originals: Cloudinary's
# src/lib/extras/dec/pnm.cc:323 and color_hints.cc:58, and libjxl's
# lib/extras/dec/pnm.cc:486 and color_hints.cc:75 preserve float samples and
# select encoded sRGB. PFM serialization, frame ordering and clip pooling here
# are this repository's adapter; no original implementation is copied.


def _check_metric(metric: str) -> None:
    if metric not in PINS:
        raise ValueError(f"unsupported original spatial metric: {metric}")


def _reference_sources(metric: str) -> list[str]:
    prefix = (
        ".reference/ssimulacra2/src"
        if metric == "ssimulacra2" else ".reference/libjxl"
    )
    kernels = (
        [prefix + "/ssimulacra2.cc"] if metric == "ssimulacra2" else [
            prefix + "/lib/jxl/enc_butteraugli_comparator.cc", prefix +
            "/lib/jxl/butteraugli/butteraugli.cc"
        ]
    )
    return [
        *kernels, prefix + "/lib/extras/dec/pnm.cc",
        prefix + "/lib/extras/dec/color_hints.cc"
    ]


def spatial_provenance(metric: str) -> dict[str, object]:
    """Fingerprint the original, its adapter and fixed scoring settings.

    :param metric: SSIMULACRA 2 or Butteraugli metric name.
    :returns: Source revision, executable/build/source hashes and input policy.
    :raises RuntimeError: If a required pinned original build is missing or
        stale.
    :raises ValueError: If the selected metric is unsupported.
    """
    _check_metric(metric)
    stamp = ROOT / "build/reference-fingerprints" / f"{STEPS[metric]}.json"
    binary = ROOT / BINARIES[metric]
    if not stamp.is_file() or not binary.is_file():
        raise RuntimeError(
            f"build pinned spatial reference with "
            f"tools/build_references.py --only {STEPS[metric]}"
        )
    fingerprint = json.loads(stamp.read_text())
    if fingerprint.get("revision") != PINS[metric]:
        raise RuntimeError(
            "spatial reference build fingerprint has the wrong pinned revision"
        )
    files = [
        HARNESSES[metric], "bench/inmemory/driver.h", "bench/inmemory/pairs.h",
        "bench/inmemory/prefetch.h", *_reference_sources(metric)
    ]
    info: dict[str, object] = {
        "revision": PINS[metric],
        "executable_sha256": digest(binary),
        "build_fingerprint_sha256": digest(stamp),
        "build_fingerprint": fingerprint,
        "source_sha256": {
            name: digest(ROOT / name)
            for name in files
        },
        "runner_sha256": digest(Path(__file__)),
        "numpy": np.__version__,
        "threads": 1,
        "decode_threads": 1,
        "frame_iterations": 1,
        "frame_warmup": 0,
        "pooling": "arithmetic-mean-of-original-frame-scores",
        "input": "bottom-up-interleaved-little-endian-FP32-PFM-encoded-sRGB",
        "transfer": "sRGB via pinned PNM/PFM reader default ColorHints",
    }
    if metric == "butteraugli":
        info["parameters"] = {
            "hf_asymmetry": 1.0,
            "intensity_target": 80.0,
            "xmul": 1.0
        }
    return info


def write_pfm(path: Path, encoded_rgb: FloatImage) -> str:
    """Serialize planar encoded RGB float32 without clipping or quantization.

    The pinned original PFM readers both flip file rows vertically and apply
    their default sRGB color encoding. PFM is used as an exact floating-point
    transport for those readers; its transfer interpretation is declared in
    reference provenance instead of relying on other PFM readers' defaults.

    :param path: Temporary PFM output path.
    :param encoded_rgb: Finite [0, 1] float32 samples in [3, H, W] order.
    :returns: SHA-256 of the exact serialized original input file.
    :raises ValueError: If shape, precision, range or values are invalid.
    """
    values = np.asarray(encoded_rgb)
    if (values.ndim != 3 or values.shape[0] != 3 or min(values.shape) < 1
            or values.dtype.kind != "f" or values.dtype.itemsize != 4):
        raise ValueError("PFM requires three nonempty planar float32 channels")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1
                                                                     ):
        raise ValueError("PFM requires finite encoded RGB samples in [0, 1]")
    height, width = values.shape[1:]
    header = f"PF\n{width} {height}\n-1.0\n".encode("ascii")
    pixels = values.transpose(1, 2, 0)[::-1].astype("<f4", copy=False).tobytes()
    path.write_bytes(header + pixels)
    return digest(path)


def _dimensions(row: dict[str, object]) -> tuple[int, int, int]:
    values = tuple(row[name] for name in ("frames", "height", "width"))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in values):
        raise ValueError(
            "spatial reference requires positive integer clip geometry/count"
        )
    return values


def _rgb_input(row: dict[str, object], side: str) -> tuple[Path, str]:
    spatial = "spatial_original" in row or "spatial_distorted" in row
    name = "spatial_" + side if spatial else side
    hashes = row.get(name + "_hashes")
    if name not in row or not isinstance(hashes, dict):
        raise ValueError(
            "spatial reference input paths and hash metadata are incomplete"
        )
    expected = hashes.get("rgb")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}",
                                                     expected) is None:
        raise ValueError(
            "spatial reference requires a pinned RGB sidecar SHA-256"
        )
    return Path(str(row[name]) + ".rgb"), expected


def _read_rgb(stream: BinaryIO, height: int, width: int) -> FloatImage:
    count = 3 * height * width * 4
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("truncated spatial RGB sidecar")
    return np.frombuffer(data, dtype="<f4").reshape(3, height, width)


def _prepare_frames(row: dict[str, object],
                    directory: Path) -> tuple[Path, list[dict[str, object]]]:
    count, height, width = _dimensions(row)
    original, original_hash = _rgb_input(row, "original")
    distorted, distorted_hash = _rgb_input(row, "distorted")
    hashes = [hashlib.sha256(), hashlib.sha256()]
    frames = []
    pair_lines = []
    with original.open("rb") as reference, distorted.open("rb") as test:
        for index in range(count):
            ref = _read_rgb(reference, height, width)
            dist = _read_rgb(test, height, width)
            hashes[0].update(ref.tobytes())
            hashes[1].update(dist.tobytes())
            first, second = directory / (
                f"{index}-reference.pfm"
            ), directory / (f"{index}-distorted.pfm")
            frames.append(
                {
                    "index": index,
                    "original_pfm_sha256": write_pfm(first, ref),
                    "distorted_pfm_sha256": write_pfm(second, dist)
                }
            )
            pair_lines.append(f"{first}\t{second}\n")
        extra = reference.read(1) or test.read(1)
    if extra:
        raise ValueError(
            "spatial RGB sidecar contains extra frames or trailing bytes"
        )
    if [checksum.hexdigest() for checksum in hashes] != [original_hash,
                                                         distorted_hash]:
        raise ValueError(
            "spatial RGB sidecar does not match its pinned input hash"
        )
    pairs = directory / "pairs.tsv"
    pairs.write_text("".join(pair_lines))
    return pairs, frames


def _scores(stdout: str, count: int) -> list[float]:
    lines = stdout.splitlines()
    if len(lines) != count:
        raise ValueError(
            f"original spatial reference returned {len(lines)} scores for "
            f"{count} frames"
        )
    try:
        values = [float(line) for line in lines]
    except ValueError as error:
        raise ValueError(
            "original spatial reference returned an invalid frame score"
        ) from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            "original spatial reference returned a nonfinite frame score"
        )
    return values


def spatial_score(row: dict[str, object], metric: str,
                  directory: Path) -> dict[str, object]:
    """Score all clip frames with a pinned original spatial batch harness.

    Spatial-specific input paths and hashes take precedence over generic rows.
    Temporary PFM files are removed after the original returns; their ordered
    hashes, command and stdout/stderr are retained as numerical provenance.

    :param row: Prepared clip geometry and independently decoded RGB sidecars.
    :param metric: SSIMULACRA 2 or Butteraugli metric name.
    :param directory: Raw reference evidence/cache destination.
    :returns: Arithmetic-mean clip score, frame scores and exact input
        provenance.
    :raises ValueError: If clip inputs or returned frame scores are invalid.
    :raises RuntimeError: If the original executable fails.
    """
    _check_metric(metric)
    directory.mkdir(parents=True, exist_ok=True)
    count, _, _ = _dimensions(row)
    with tempfile.TemporaryDirectory(prefix="spatial-frames-",
                                     dir=directory) as temporary:
        pairs, frame_inputs = _prepare_frames(row, Path(temporary).resolve())
        command = [
            str(ROOT / BINARIES[metric]), "--metric",
            "ssimu2" if metric == "ssimulacra2" else "butteraugli",
            "--pairs-file",
            str(pairs), "--score-only", "--threads", "1", "--decode-threads",
            "1", "--iterations", "1", "--warmup", "0"
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300, check=False
        )
        (directory / "spatial.stdout").write_text(result.stdout)
        (directory / "spatial.stderr").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(
                result.stderr or result.stdout
                or "original spatial reference failed"
            )
        values = _scores(result.stdout, count)
    return {
        "score":
        math.fsum(values) / count,
        "frames":
        count,
        "frame_results": [
            {
                "index": index,
                "score": value
            } for index, value in enumerate(values)
        ],
        "command":
        command,
        "frame_inputs":
        frame_inputs,
        "frame_input_protocol":
        "encoded-sRGB-FP32-PFM-pinned-reader-default-v1"
    }
