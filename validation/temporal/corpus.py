"""Prepare byte-identical temporal inputs and deterministic distortion variants."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

from .fetch import Source, load_manifest, select_sources

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("sources.json")
ByteArray = NDArray[np.uint8]


@dataclass(frozen=True)
class Variant:
    """One fixed distortion recipe, independent of candidate results."""

    name: str
    family: str
    strength: int


VARIANTS = (
    Variant("quantize8", "quantization", 8), Variant("quantize32", "quantization", 32),
    Variant("flicker8", "flicker", 8), Variant("flicker24", "flicker",
                                               24), Variant("flicker64", "flicker", 64),
    Variant("flicker96", "flicker", 96), Variant("freeze3", "freeze",
                                                 3), Variant("freeze8", "freeze", 8),
    Variant("pulse12", "impulse", 12), Variant("pulse32", "impulse",
                                               32), Variant("pulse64", "impulse", 64),
    Variant("pulse128", "impulse", 128), Variant("pulse224", "impulse", 224),
)
SMOKE_SOURCES = "foreman_cif,bus_cif,mobile_cif"


def digest(path: Path) -> str:
    """Hash one input without retaining its contents.

    :param path: Existing regular file.
    :returns: SHA-256 hexadecimal digest.
    """
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def json_digest(value: object) -> str:
    """Hash canonical JSON metadata.

    :param value: JSON-compatible configuration or manifest.
    :returns: SHA-256 hexadecimal digest.
    """
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_source(source: Source, directory: Path, frames: int) -> ByteArray:
    """Read a verified prefix's tightly packed 8-bit 4:2:0 frames.

    :param source: Pinned source description.
    :param directory: Downloaded prefix directory.
    :param frames: Number of initial frames to retain.
    :returns: Frame-major packed YUV bytes.
    :raises ValueError: If source bytes or framing differ from the pin.
    """
    path = directory / source.filename
    if not source.chroma.startswith("420") or source.width % 2 or source.height % 2:
        raise ValueError("temporal preparation requires even 8-bit YUV420 sources")
    if not 1 <= frames <= source.frames:
        raise ValueError("requested frame count exceeds the pinned prefix")
    if digest(path) != source.sha256:
        raise ValueError(f"{path}: source checksum mismatch; run temporal.fetch")
    size = source.width * source.height * 3 // 2
    values = []
    with path.open("rb") as stream:
        stream.readline()
        for _ in range(frames):
            values.append(_read_frame(stream, size, path))
    if any(value.size != size for value in values):
        raise ValueError(f"{path}: truncated source")
    return np.stack(values)


def _read_frame(stream: BinaryIO, size: int, path: Path) -> ByteArray:
    if stream.readline() != b"FRAME\n":
        raise ValueError(f"{path}: missing frame marker")
    return np.frombuffer(stream.read(size), dtype=np.uint8)


def distort(values: ByteArray, variant: Variant, pixels: int) -> ByteArray:
    """Apply a fully integer, deterministic distortion recipe.

    :param values: Frame-major packed YUV420 samples.
    :param variant: Fixed recipe and strength.
    :param pixels: Number of luma samples in one frame.
    :returns: Independently owned distorted frames.
    :raises ValueError: If the recipe is unknown.
    """
    if variant.family == "identity":
        return values.copy()
    if variant.family == "freeze":
        indices = np.arange(len(values)) // variant.strength * variant.strength
        return values[indices].copy()
    if variant.family == "quantization":
        rounded = (values.astype(np.uint16) + variant.strength // 2)
        rounded = rounded // variant.strength * variant.strength
        return np.clip(rounded, 0, 255).astype(np.uint8)
    result = values.copy()
    if variant.family == "flicker":
        signs = np.where(np.arange(len(values)) % 2, 1, -1)[:, None]
        luma = result[:, :pixels].astype(np.int16) + signs * variant.strength
        result[:, :pixels] = np.clip(luma, 0, 255).astype(np.uint8)
        return result
    if variant.family == "impulse":
        index = len(values) - 1
        luma = result[index, :pixels].astype(np.int16) + variant.strength
        result[index, :pixels] = np.clip(luma, 0, 255).astype(np.uint8)
        return result
    raise ValueError(f"unknown distortion family: {variant.family}")


def rgb_frame(frame: ByteArray, width: int, height: int) -> NDArray[np.float32]:
    """Decode one frame once to shared planar RGB float32 samples.

    The original consumes this serialized result; the CLI receives the same
    canonical YUV444 samples and performs its public input conversion.

    :param frame: Packed 8-bit YUV444 samples.
    :param width: Even source width.
    :param height: Even source height.
    :returns: Planar encoded RGB, clipped to [0, 1], as little-endian FP32.
    """
    pixels = width * height
    y = (frame[:pixels].reshape(height, width).astype(np.float32) - 16) / 219
    u = frame[pixels:2 * pixels].reshape(height, width)
    v = frame[2 * pixels:].reshape(height, width)
    cb = (u.astype(np.float32) - 128) / 224
    cr = (v.astype(np.float32) - 128) / 224
    planes = (
        y + np.float32(1.402) * cr,
        y - np.float32(0.344136) * cb - np.float32(0.714136) * cr,
        y + np.float32(1.772) * cb
    )
    return np.clip(np.stack(planes), 0, 1).astype("<f4")


def write_clip(path: Path, values: ByteArray, source: Source) -> dict[str, str]:
    """Write canonical Y4M, raw YUV and a shared RGB sidecar.

    :param path: Canonical Y4M destination.
    :param values: Frame-major packed YUV444 samples.
    :param source: Source geometry and native rational timing.
    :returns: SHA-256 hashes of all three representations.
    """
    header = (
        f"YUV4MPEG2 W{source.width} H{source.height} F{source.fps} "
        "Ip A1:1 C444\n"
    ).encode()
    with path.open("wb") as y4m, path.with_suffix(".yuv").open("wb") as raw:
        y4m.write(header)
        for frame in values:
            data = frame.tobytes()
            y4m.write(b"FRAME\n" + data)
            raw.write(data)
    rgb_path = Path(str(path) + ".rgb")
    with rgb_path.open("wb") as rgb:
        for frame in values:
            rgb.write(rgb_frame(frame, source.width, source.height).tobytes())
    return {
        "y4m": digest(path),
        "yuv": digest(path.with_suffix(".yuv")),
        "rgb": digest(rgb_path)
    }


def prepare_source(
    source: Source, download: Path, cache: Path, frames: int, variants: tuple[Variant,
                                                                              ...]
) -> list[dict[str, object]]:
    """Prepare all requested variants of one verified source.

    :param source: Pinned source description.
    :param download: Original prefix directory.
    :param cache: Derived-input destination.
    :param frames: Number of complete source frames.
    :param variants: Distortion recipes to generate.
    :returns: Ordered sample records with complete input hashes.
    """
    values = expand_chroma(
        read_source(source, download, frames), source.width, source.height
    )
    directory = cache / source.id
    directory.mkdir(parents=True, exist_ok=True)
    original = directory / "original.y4m"
    original_hashes = write_clip(original, values, source)
    rows = []
    for variant in variants:
        target = directory / (variant.name + ".y4m")
        hashes = write_clip(
            target, distort(values, variant, source.width * source.height), source
        )
        rows.append(
            {
                "sample_id": f"{source.id}/{variant.name}",
                "source_id": source.source_id,
                "distortion_family": variant.family,
                "severity": variant.strength,
                "stratum": "sdr-native-444",
                "original": str(original.resolve()),
                "distorted": str(target.resolve()),
                "width": source.width,
                "height": source.height,
                "fps": source.fps,
                "frames": frames,
                "original_hashes": original_hashes,
                "distorted_hashes": hashes,
                "source_sha256": source.sha256,
            }
        )
    return rows


def expand_chroma(values: ByteArray, width: int, height: int) -> ByteArray:
    """Expand source 4:2:0 chroma once with an exact nearest-neighbor rule.

    :param values: Frame-major packed 8-bit YUV420 samples.
    :param width: Even source width.
    :param height: Even source height.
    :returns: Frame-major packed YUV444 bytes shared by both implementations.
    """
    pixels = width * height
    frames = len(values)
    chroma = values[:, pixels:].reshape(frames, 2, height // 2, width // 2)
    expanded = np.repeat(np.repeat(chroma, 2, axis=2), 2, axis=3)
    return np.concatenate(
        (values[:, :pixels], expanded.reshape(frames, 2 * pixels)), axis=1
    )


def prepare(
    download: Path, cache: Path, profile: str, only: str | None = None,
    manifest: Path = MANIFEST
) -> list[dict[str, object]]:
    """Prepare a reproducible smoke or full hosted-video qualification corpus.

    :param download: Verified original prefixes.
    :param cache: Disposable derived corpus destination.
    :param profile: Either smoke or qualification.
    :param only: Optional source IDs; qualification still requires full coverage.
    :param manifest: Pinned source manifest.
    :returns: Prepared clip-pair records.
    :raises ValueError: If the profile or requested sources are invalid.
    """
    if profile not in ("smoke", "qualification"):
        raise ValueError("profile must be smoke or qualification")
    sources = select_sources(
        load_manifest(manifest), only
        or (SMOKE_SOURCES if profile == "smoke" else None)
    )
    variants = VARIANTS
    frames = 12 if profile == "smoke" else 40
    rows = []
    for source in sources:
        rows.extend(prepare_source(source, download, cache, frames, variants))
    return rows


def verify_inputs(rows: list[dict[str, object]]) -> None:
    """Reject prepared media changed between reference and candidate execution.

    :param rows: Prepared clip records with expected representation hashes.
    :raises ValueError: If any prepared file no longer matches its fingerprint.
    """
    files = {}
    for row in rows:
        files.update(_clip_hashes(row))
    for path, expected in files.items():
        if digest(path) != expected:
            raise ValueError(f"{path}: prepared input changed during validation")


def _clip_hashes(row: dict[str, object]) -> dict[Path, str]:
    files = {}
    for side in ("original", "distorted"):
        path = Path(str(row[side]))
        hashes = row[side + "_hashes"]
        files.update(
            {
                path: hashes["y4m"],
                path.with_suffix(".yuv"): hashes["yuv"],
                Path(str(path) + ".rgb"): hashes["rgb"]
            }
        )
    return files
