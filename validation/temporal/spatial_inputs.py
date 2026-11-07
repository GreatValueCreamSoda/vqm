"""Prepare common in-gamut sRGB video inputs for both spatial metric arms."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from . import corpus
from .fetch import Source
from .fetch import load_manifest

PROTOCOL = "clipped-rgb-gamut0.02-0.98-bt601-limited444-rne-fp32-v1"
SPATIAL_METRICS = ("ssimulacra2", "butteraugli")


def encode_frame(rgb: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Map encoded RGB to a safe gamut and encode limited-range bytes.

    :param rgb: Three full-resolution encoded RGB planes in [0, 1].
    :returns: Full-resolution YUV444 byte planes.
    :raises ValueError: If the source is not finite, normalized three-plane RGB.
    """
    if rgb.ndim != 3 or rgb.shape[0] != 3 or not np.isfinite(rgb).all():
        raise ValueError("expected finite planar RGB")
    if np.min(rgb) < 0 or np.max(rgb) > 1:
        raise ValueError("source RGB must lie in [0, 1]")
    red, green, blue = rgb.astype(np.float64) * .96 + .02
    luma = .299 * red + .587 * green + .114 * blue
    cb = (blue - luma) / 1.772
    cr = (red - luma) / 1.402
    return np.rint(np.stack((16 + 219 * luma, 128 + 224 * cb, 128 + 224 * cr))
                   ).astype(np.uint8)


def write_spatial_clip(source_rgb: Path, target: Path,
                       source: Source) -> dict[str, str]:
    """Serialize video with matching CLI and reference RGB conversions.

    :param source_rgb: Existing clipped FP32 RGB representation.
    :param target: New canonical Y4M path.
    :param source: Pinned source geometry and rational timing.
    :returns: Hashes of the actual Y4M, raw YUV and encoded FP32 RGB files.
    :raises ValueError: If a frame would be clipped by the reference conversion.
    """
    rgb = np.fromfile(source_rgb,
                      dtype="<f4").reshape(-1, 3, source.height, source.width)
    values = np.stack([encode_frame(frame).reshape(-1) for frame in rgb])
    for frame in values:
        decoded = corpus.rgb_frame(frame, source.width, source.height)
        if not (np.min(decoded) > 0 and np.max(decoded) < 1):
            raise ValueError(
                "spatial preparation failed to preserve RGB gamut headroom"
            )
    return corpus.write_clip(target, values, source)


def _prepared_side(row: dict[str, object], side: str,
                   source: Source) -> tuple[str, dict[str, str]]:
    original = Path(str(row[side]))
    rgb = Path(str(original) + ".rgb")
    expected = row[side + "_hashes"]["rgb"]
    if corpus.digest(rgb) != expected:
        raise ValueError(f"{rgb}: original RGB hash mismatch")
    target = original.with_suffix(".spatial.y4m")
    metadata = target.with_suffix(".json")
    identity = {
        "protocol": PROTOCOL,
        "input_rgb_sha256": expected,
        "width": source.width,
        "height": source.height,
        "fps": source.fps
    }
    if metadata.exists():
        recorded = json.loads(metadata.read_text())
        if recorded.get("identity") == identity and _valid_files(
                target, recorded["hashes"]):
            return str(target), recorded["hashes"]
    hashes = write_spatial_clip(rgb, target, source)
    metadata.write_text(
        json.dumps({
            "identity": identity,
            "hashes": hashes
        }, indent=2) + "\n"
    )
    return str(target), hashes


def _valid_files(target: Path, hashes: dict[str, str]) -> bool:
    paths = {
        "y4m": target,
        "yuv": target.with_suffix(".yuv"),
        "rgb": Path(str(target) + ".rgb")
    }
    return all(
        path.is_file() and corpus.digest(path) == hashes[name]
        for name, path in paths.items()
    )


def ensure_spatial(
    rows: list[dict[str, object]], sources: Sequence[Source] | None = None
) -> list[dict[str, object]]:
    """Add fingerprinted spatial inputs while preserving temporal inputs.

    :param rows: Prepared hosted video pairs.
    :param sources: Optional verified source records, primarily for offline
        tests.
    :returns: Copies with complete spatial paths, hashes and protocol
        identifiers.
    :raises ValueError: If source identity or original RGB data is invalid.
    """
    by_id = {
        source.id: source
        for source in
        (load_manifest(corpus.MANIFEST) if sources is None else sources)
    }
    cached: dict[str, tuple[str, dict[str, str]]] = {}
    result = []
    for row in rows:
        identifier = str(row["sample_id"]).split("/", 1)[0]
        if identifier not in by_id:
            raise ValueError(f"unknown hosted source: {identifier}")
        result.append(_spatial_row(row, by_id[identifier], cached))
    return result


def _spatial_row(
    row: dict[str, object], source: Source,
    cached: dict[str, tuple[str, dict[str, str]]]
) -> dict[str, object]:
    value = dict(row, spatial_protocol=PROTOCOL)
    for side in ("original", "distorted"):
        path = str(row[side])
        if path not in cached:
            cached[path] = _prepared_side(row, side, source)
        value["spatial_" + side], value["spatial_" + side +
                                        "_hashes"] = cached[path]
    return value
