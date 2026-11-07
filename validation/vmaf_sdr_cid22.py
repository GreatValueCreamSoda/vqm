#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Compare VQM's RGB-to-SDR VMAF path against CID22 and pinned libvmaf.

The full corpus is evaluated as independent stills. VQM consumes the original
RGB images with explicit 4:4:4 (both models) and 4:2:0 (v1) conversion. An
independent CPU conversion to 16-bit limited BT.709 YUV444 is also scored with
libvmaf for the first --reference-count pairs. Results from the older 10-bit
encoded-sRGB conversion are a separate MCOS comparison, not parity references.
Pillow and NumPy are required; decoding 16-bit PNG inputs also requires ffmpeg.

Existing legacy vqm-v0.csv and vqm-v1.csv may be provided via --baseline-dir;
otherwise their native-YUV scores are regenerated using vmaf_cid22.py helpers.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vmaf_cid22
from common import Pair, correlation_triple, enumerate_pairs, load_mcos
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from PIL import __version__ as pillow_version
from vmaf_cid22 import ConvertedFrame, CorpusRow, VmafPaths

ROOT = vmaf_cid22.ROOT
LEGACY_CONVERSION = vmaf_cid22.CONVERSION
LEGACY_YUV_CACHE = vmaf_cid22.YUV_CACHE

CONVERSION = "srgb-rgb8-rgb16-display-to-bt709-limited-yuv444p16le-rne-v2"
REVISION = "4611e47a7330cd342498dcc130e1008efbeab6e4"


def convert_rgb(rgb: NDArray[np.uint8] | NDArray[np.uint16]) -> bytes:
    """Convert SDR RGB through the sRGB EOTF and display BT.709.

    :param rgb: RGB8 or RGB16 samples in height, width and channel order.
    :returns: Planar little-endian YUV444P16 bytes.
    :raises ValueError: If the input is not an RGB8 or RGB16 image.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype not in (np.uint8, np.uint16):
        raise ValueError("expected RGB8 or RGB16 image")
    maximum = 255.0 if rgb.dtype == np.uint8 else 65535.0
    srgb = rgb.astype(np.float64) / maximum
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055)**2.4)
    encoded = np.maximum(linear, 0.0)**(1.0 / 2.4)
    red, green, blue = (encoded[:, :, i] for i in range(3))
    y = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    components = (
        16.0 + 219.0 * y, 128.0 + (112.0 / 0.9278) * (blue - y),
        128.0 + (112.0 / 0.7874) * (red - y)
    )
    return b"".join(
        np.rint(np.clip(plane * 256.0, 0, 65535)).astype("<u2").tobytes()
        for plane in components
    )


def _decode_png16(path: Path, width: int, height: int,
                  has_alpha: bool) -> NDArray[np.uint16]:
    # rgb48/rgba64 preserve the decoded PNG's original 16-bit values.
    # Pillow's RGB conversion truncates 16-bit-per-channel PNGs to 8.
    pixel_format = "rgba64le" if has_alpha else "rgb48le"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i",
        str(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", pixel_format,
        "pipe:1"
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=120)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"{path}: ffmpeg is required to preserve 16-bit PNG samples"
        ) from error
    if result.returncode:
        raise RuntimeError(f"{path}: ffmpeg: {result.stderr.decode(errors='replace')}")
    channels = 4 if has_alpha else 3
    if len(result.stdout) != width * height * channels * 2:
        raise ValueError(f"{path}: incorrect RGB16 decoded frame length")
    samples = np.frombuffer(result.stdout, dtype="<u2").reshape(height, width, channels)
    if has_alpha and np.any(samples[:, :, 3] != 65535):
        raise ValueError(f"{path}: non-opaque alpha is unsupported")
    return samples[:, :, :3]


def read_source(path: Path) -> NDArray[np.uint8] | NDArray[np.uint16]:
    """Decode a still while preserving the original sample precision.

    :param path: Source image.
    :returns: RGB samples with opaque alpha removed.
    :raises ValueError: If the image mode, alpha or decoded length is invalid.
    :raises RuntimeError: If FFmpeg is required and unavailable or fails.
    """
    with path.open("rb") as source:
        header = source.read(29)
    png16 = (
        len(header) == 29 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[24] == 16
    )
    with Image.open(path) as image:
        width, height = image.size
        if png16:
            return _decode_png16(path, width, height, header[25] in (4, 6))
        return vmaf_cid22.decode_rgb8(image, path)


def convert_file_16(path: Path, cache: Path) -> ConvertedFrame:
    """Create and verify one content-addressed planar YUV444P16LE frame.

    :param path: Source image.
    :param cache: Existing directory for converted samples.
    :returns: Converted frame path, dimensions and input fingerprints.
    :raises ValueError: If the source image is invalid.
    :raises RuntimeError: If FFmpeg fails to decode a 16-bit PNG.
    """
    source_hash = vmaf_cid22.digest(path)
    key = hashlib.sha256((CONVERSION + ":" + source_hash).encode()).hexdigest()
    target = cache / (key + ".yuv")
    rgb = read_source(path)
    height, width = rgb.shape[:2]
    raw = convert_rgb(rgb)
    raw_hash = hashlib.sha256(raw).hexdigest()
    if not target.is_file() or vmaf_cid22.digest(target) != raw_hash:
        vmaf_cid22.write_cache_bytes(target, ".vmaf-sdr-", raw)
    return {
        "file": str(target),
        "width": width,
        "height": height,
        "source_sha256": source_hash,
        "yuv_sha256": raw_hash
    }


def _score_rows(reader: csv.DictReader, path: Path, model: str) -> dict[Pair, float]:
    values = {}
    if not {"original", "distorted", model}.issubset(reader.fieldnames or []):
        raise ValueError(f"{path}: missing original, distorted or {model}")
    for row in reader:
        key = row["original"], row["distorted"]
        if key in values:
            raise ValueError(f"{path}: duplicate pair {key}")
        value = float(row[model])
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite score for {key}")
        values[key] = value
    return values


def read_score_csv(path: Path, model: str, pairs: list[Pair]) -> list[float]:
    """Load selected finite scores by their exact image pair keys.

    :param path: Existing score CSV.
    :param model: Column containing the requested model's scores.
    :param pairs: Selected pairs in output order.
    :returns: One score per selected pair in that order.
    :raises ValueError: If a pair is missing, duplicated or has an invalid score.
    """
    with path.open(newline="") as source:
        values = _score_rows(csv.DictReader(source), path, model)
    missing = set(pairs) - values.keys()
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} selected pairs")
    return [values[pair] for pair in pairs]


def score_rgb(pairs_path: Path, count: int, metric: str, chroma: str,
              out_dir: Path) -> list[float]:
    """Score the RGB pair list with one VQM model and chroma mode.

    :param pairs_path: Tab-separated reference and distortion paths.
    :param count: Number of expected scores.
    :param metric: VMAF model selector, "v0" or "v1".
    :param chroma: Chroma conversion mode, "420" or "444".
    :param out_dir: Destination for stdout and diagnostic logs.
    :returns: Finite VQM scores in pair-list order.
    :raises ValueError: If the command fails or produces invalid scores.
    """
    executable = ROOT / "build/vqm_bench"
    command = [
        str(executable), "--metric", "vmaf-" + metric, "--vmaf-chroma", chroma,
        "--score-only", "--pairs-file",
        str(pairs_path), "--iterations", "1", "--warmup", "0"
    ]
    print(f"VQM {metric} {chroma}: scoring {count} independent still pairs", flush=True)
    result = subprocess.run(
        command, text=True, capture_output=True, timeout=120 + 2 * count
    )
    log = out_dir / f"vqm-{metric}-{chroma}.log"
    log.write_text(" ".join(command) + "\n" + result.stderr)
    (out_dir / f"vqm-{metric}-{chroma}.stdout").write_text(result.stdout)
    lines = result.stdout.splitlines()
    if result.returncode or len(lines) != count:
        raise ValueError(
            f"{log}: exit {result.returncode}, expected {count} "
            f"scores but got {len(lines)}"
        )
    values = [float(line) for line in lines]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{log}: non-finite score")
    return values


def reference_score_16(row: CorpusRow, executable: Path, model0: Path,
                       model1: Path) -> list[float]:
    """Score one 16-bit pair with both pinned libvmaf models.

    :param row: Prepared pair paths and frame dimensions.
    :param executable: Pinned libvmaf driver.
    :param model0: VMAF v0 model path.
    :param model1: VMAF v1 model path.
    :returns: The v0 and v1 frame scores in that order.
    :raises RuntimeError: If the driver fails.
    :raises ValueError: If its output is not one finite score per model.
    """
    command = [
        str(executable), "models",
        str(model0), row["original_yuv"], row["distorted_yuv"],
        str(row["width"]),
        str(row["height"]), "16",
        str(model1)
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"{row['distorted']}: {result.stderr.strip()}")
    values = list(csv.DictReader(result.stdout.splitlines()))
    if len(values) != 1 or values[0].get("frame") != "0":
        raise ValueError(f"{row['distorted']}: expected one reference frame")
    scores = [float(values[0][name]) for name in ("v0", "v1")]
    if not all(math.isfinite(value) for value in scores):
        raise ValueError(f"{row['distorted']}: non-finite reference score")
    return scores


def correlations(values: ArrayLike, mcos: ArrayLike) -> dict[str, float]:
    """Compute rank and linear correlations against subjective MCOS.

    :param values: Metric scores for the selected pairs.
    :param mcos: Subjective scores in matching order.
    :returns: Finite Kendall tau-b, Spearman and Pearson correlations by name.
    :raises ValueError: If a correlation is undefined.
    """
    result = correlation_triple(values, mcos)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("correlation requires at least two varying values")
    return dict(zip(("KRCC", "SRCC", "PCC"), result, strict=True))


def difference(actual: ArrayLike, reference: ArrayLike) -> dict[str, float]:
    """Summarize the score shifts of the first arm against the second.

    :param actual: First score vector.
    :param reference: Corresponding scores to subtract.
    :returns: Signed, absolute and squared difference statistics.
    """
    delta = np.asarray(actual) - np.asarray(reference)
    absolute = np.abs(delta)
    return {
        "mean_signed": float(np.mean(delta)),
        "mean_absolute": float(np.mean(absolute)),
        "max_absolute": float(np.max(absolute)),
        "p99_absolute": float(np.percentile(absolute, 99)),
        "rmse": float(np.sqrt(np.mean(delta**2)))
    }


def _verify_baseline_summary(
    status: dict[str, bool | int], summary_path: Path, manifest_path: Path,
    mcos_file: Path
) -> None:
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text())
    if summary.get("conversion") != LEGACY_CONVERSION:
        raise ValueError(f"{summary_path}: unexpected legacy conversion")
    if summary.get("mcos_sha256") != vmaf_cid22.digest(mcos_file):
        raise ValueError(f"{summary_path}: MCOS file hash differs")
    status["mcos_verified"] = True
    if manifest_path.is_file():
        if summary.get("input_manifest_sha256") != vmaf_cid22.digest(manifest_path):
            raise ValueError(f"{summary_path}: input manifest hash differs")
        status["manifest_verified"] = True


def _index_manifest(reader: csv.DictReader,
                    manifest_path: Path) -> dict[Pair, dict[str, str]]:
    manifest = {}
    required = {
        "original", "distorted", "original_source_sha256", "distorted_source_sha256"
    }
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"{manifest_path}: missing source hash columns")
    for row in reader:
        key = row["original"], row["distorted"]
        if key in manifest:
            raise ValueError(f"{manifest_path}: duplicate pair {key}")
        manifest[key] = row
    return manifest


def _record_source_hash(
    expected: dict[str, str], path: str, value: str, manifest_path: Path
) -> None:
    if not is_sha256(value):
        raise ValueError(f"{manifest_path}: invalid source hash for {path}")
    if path in expected and expected[path] != value:
        raise ValueError(f"{manifest_path}: conflicting source hashes for {path}")
    expected[path] = value


def _expected_source_hashes(pairs: list[Pair], manifest_path: Path) -> dict[str, str]:
    with manifest_path.open(newline="") as source:
        manifest = _index_manifest(csv.DictReader(source), manifest_path)
    expected = {}
    for pair in pairs:
        if pair not in manifest:
            raise ValueError(f"{manifest_path}: missing pair {pair}")
        row = manifest[pair]
        for path, field in ((pair[0], "original_source_sha256"),
                            (pair[1], "distorted_source_sha256")):
            _record_source_hash(expected, path, row[field], manifest_path)
    return expected


def validate_baseline(
    baseline_dir: Path, pairs: list[Pair], data_root: Path, mcos_file: Path, jobs: int
) -> dict[str, bool | int]:
    """Verify available prior manifest and annotation provenance.

    :param baseline_dir: Directory with the previous report and input manifest.
    :param pairs: Selected pairs that must appear in the previous manifest.
    :param data_root: Current CID22 source directory.
    :param mcos_file: Current annotation CSV.
    :param jobs: Concurrent source hashing workers.
    :returns: Whether annotation and manifest hashes matched and how many sources were checked.
    :raises ValueError: If a required source or recorded digest differs.
    """
    status = {
        "mcos_verified": False,
        "manifest_verified": False,
        "source_files_verified": 0
    }
    summary_path = baseline_dir / "summary.json"
    manifest_path = baseline_dir / "inputs.csv"
    _verify_baseline_summary(status, summary_path, manifest_path, mcos_file)
    if not manifest_path.is_file():
        print(
            f"warning: {manifest_path} missing; baseline source hashes not verified",
            file=sys.stderr, flush=True
        )
        return status
    expected = _expected_source_hashes(pairs, manifest_path)
    paths = sorted(expected)
    actual = vmaf_cid22.parallel_map(
        "baseline source hashes", lambda path: vmaf_cid22.digest(data_root / path),
        paths, jobs
    )
    for path, value in zip(paths, actual, strict=True):
        if value != expected[path]:
            raise ValueError(f"baseline source hash differs: {path}")
    status["source_files_verified"] = len(paths)
    return status


def is_sha256(value: str) -> bool:
    """Check that a string is a canonical lowercase SHA-256 digest.

    :param value: Digest text from a manifest.
    :returns: Whether it is exactly 64 lowercase hexadecimal digits.
    """
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def ffmpeg_version() -> str | None:
    """Read the FFmpeg version for the conversion provenance report.

    :returns: Version line, or None if FFmpeg is unavailable.
    :raises RuntimeError: If an installed FFmpeg returns no version.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], text=True, capture_output=True, timeout=10
        )
    except FileNotFoundError:
        return None
    if result.returncode or not result.stdout.splitlines():
        raise RuntimeError("ffmpeg -version failed")
    return result.stdout.splitlines()[0]


def legacy_scores(
    pairs: list[Pair], data_root: Path, out_dir: Path, jobs: int,
    baseline_dir: Path | None
) -> dict[str, list[float]]:
    """Reuse or regenerate the older encoded RGB conversion's scores.

    :param pairs: Selected still-image pairs.
    :param data_root: Extracted CID22 directory.
    :param out_dir: Destination for regenerated legacy scores.
    :param jobs: Concurrent conversion and Y4M preparation workers.
    :param baseline_dir: Prior verified score directory to reuse, if supplied.
    :returns: Score vectors for v0 and v1 in pair order.
    :raises ValueError: If a score or pair geometry is invalid.
    """
    if baseline_dir:
        return {
            model: read_score_csv(baseline_dir / f"vqm-{model}.csv", model, pairs)
            for model in ("v0", "v1")
        }
    cache = ROOT / LEGACY_YUV_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    paths = sorted({path for pair in pairs for path in pair})
    converted = vmaf_cid22.parallel_map(
        "legacy convert", lambda path: vmaf_cid22.convert_file(data_root / path, cache),
        paths, jobs
    )
    lookup = dict(zip(paths, converted, strict=True))
    rows = []
    for original, distorted in pairs:
        ref, dist = lookup[original], lookup[distorted]
        if (ref["width"], ref["height"]) != (dist["width"], dist["height"]):
            raise ValueError(f"geometry mismatch: {original}, {distorted}")
        rows.append(
            {
                "original": original,
                "distorted": distorted,
                "original_yuv": ref["file"],
                "distorted_yuv": dist["file"],
                "width": ref["width"],
                "height": ref["height"]
            }
        )
    legacy_dir = out_dir / "legacy"
    legacy_dir.mkdir(exist_ok=True)
    return {
        model: vmaf_cid22.vqm_scores(rows, legacy_dir, jobs, model)
        for model in ("v0", "v1")
    }


def parse_args() -> argparse.Namespace:
    """Parse the SDR comparison's corpus, baseline and output options.

    :returns: Validated command-line options.
    :raises SystemExit: If an option is invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "validation/data/CID22_validation_set"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "artifacts/reports/vmaf-sdr-cid22"
    )
    parser.add_argument(
        "--cache", type=Path, default=ROOT / ".cache/validation/vmaf-sdr-cid22-yuv16"
    )
    parser.add_argument(
        "--baseline-dir", type=Path,
        help="prior report containing vqm-v0.csv and vqm-v1.csv"
    )
    parser.add_argument(
        "--reuse-rgb-dir", type=Path,
        help="reuse matching VQM RGB scores from an earlier report"
    )
    parser.add_argument(
        "--verify-baseline-only", action="store_true",
        help="verify prior source hashes and exit without scoring"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="first N pairs, zero selects the full corpus"
    )
    parser.add_argument(
        "--reference-count", type=int, default=300,
        help="number of first pairs independently scored by libvmaf"
    )
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.limit < 0 or args.reference_count < 2 or args.jobs < 1:
        parser.error(
            "--limit must be non-negative, --reference-count >= 2, and --jobs positive"
        )
    if args.verify_baseline_only and not args.baseline_dir:
        parser.error("--verify-baseline-only requires --baseline-dir")
    return args


@dataclass(frozen=True)
class _CorpusInputs:
    data_root: Path
    out_dir: Path
    cache: Path
    pairs: list[Pair]
    mcos_file: Path
    mcos: list[float]


_ScoreArms = dict[str, list[float]]


def _prepare_corpus(args: argparse.Namespace) -> _CorpusInputs:
    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    cache = args.cache.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    pairs = enumerate_pairs(str(data_root))
    if args.limit:
        pairs = pairs[:args.limit]
    if len(pairs) < 2:
        raise ValueError("at least two CID22 pairs are required")
    mcos_file = data_root / "CID22_validation_set.csv"
    annotations = load_mcos(str(mcos_file))
    if missing := set(pairs) - annotations.keys():
        raise ValueError(f"{len(missing)} selected pairs lack MCOS")
    mcos = [annotations[pair] for pair in pairs]
    return _CorpusInputs(data_root, out_dir, cache, pairs, mcos_file, mcos)


def _reference_assets() -> tuple[VmafPaths, Path]:
    executable = ROOT / "build/vmaf-build/vmaf_reference"
    model0 = ROOT / "build/vmaf-src/model/vmaf_float_v0.6.1.json"
    model1 = ROOT / "build/vmaf-src/model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json"
    bench = ROOT / "build/vqm_bench"
    for path in (executable, model0, model1, bench):
        if not path.is_file():
            raise ValueError(f"missing {path}; build VQM and pinned VMAF references")

    return (executable, model0, model1), bench


def _reuse_rgb_scores(
    prior: Path, pairs: list[Pair], mcos_file: Path, pairs_path: Path, bench: Path
) -> _ScoreArms:
    previous_summary = json.loads((prior / "summary.json").read_text())
    previous_scores = prior / "scores.csv"
    for key, expected in (("vqm_benchmark_sha256", vmaf_cid22.digest(bench)),
                          ("mcos_sha256", vmaf_cid22.digest(mcos_file)),
                          ("rgb_pairs_sha256", vmaf_cid22.digest(pairs_path)),
                          ("scores_sha256", vmaf_cid22.digest(previous_scores))):
        if previous_summary.get(key) != expected:
            raise ValueError(f"previous RGB scores do not match: {key}")
    return {
        name: read_score_csv(previous_scores, "vqm_" + name, pairs)
        for name in ("v0_444", "v1_444", "v1_420")
    }


def _rgb_scores(
    inputs: _CorpusInputs, reuse_dir: Path | None, bench: Path
) -> _ScoreArms:
    pairs, data_root, out_dir = inputs.pairs, inputs.data_root, inputs.out_dir
    pairs_path = out_dir / "rgb-pairs.txt"
    with pairs_path.open("w") as output:
        for original, distorted in pairs:
            output.write(f"{data_root / original}\t{data_root / distorted}\n")
    if reuse_dir:
        return _reuse_rgb_scores(reuse_dir, pairs, inputs.mcos_file, pairs_path, bench)
    return {
        "v0_444": score_rgb(pairs_path, len(pairs), "v0", "444", out_dir),
        "v1_444": score_rgb(pairs_path, len(pairs), "v1", "444", out_dir),
        "v1_420": score_rgb(pairs_path, len(pairs), "v1", "420", out_dir)
    }


def _prepare_reference_rows(
    reference_pairs: list[Pair], data_root: Path, cache: Path, jobs: int
) -> list[CorpusRow]:
    paths = sorted({path for pair in reference_pairs for path in pair})
    converted = vmaf_cid22.parallel_map(
        "CPU SDR conversion", lambda path: convert_file_16(data_root / path, cache),
        paths, jobs
    )
    lookup = dict(zip(paths, converted, strict=True))
    reference_rows = []
    for original, distorted in reference_pairs:
        ref, dist = lookup[original], lookup[distorted]
        if (ref["width"], ref["height"]) != (dist["width"], dist["height"]):
            raise ValueError(f"geometry mismatch: {original}, {distorted}")
        reference_rows.append(
            {
                "original": original,
                "distorted": distorted,
                "width": ref["width"],
                "height": ref["height"],
                "original_yuv": ref["file"],
                "distorted_yuv": dist["file"],
                "original_source_sha256": ref["source_sha256"],
                "distorted_source_sha256": dist["source_sha256"],
                "original_yuv_sha256": ref["yuv_sha256"],
                "distorted_yuv_sha256": dist["yuv_sha256"]
            }
        )
    return reference_rows


def _reference_scores(
    inputs: _CorpusInputs, reference_count: int, jobs: int, assets: VmafPaths
) -> _ScoreArms:
    reference_pairs = inputs.pairs[:min(reference_count, len(inputs.pairs))]
    reference_rows = _prepare_reference_rows(
        reference_pairs, inputs.data_root, inputs.cache, jobs
    )
    values = vmaf_cid22.parallel_map(
        "libvmaf", lambda row: reference_score_16(row, *assets), reference_rows, jobs
    )
    reference = {"v0": [row[0] for row in values], "v1": [row[1] for row in values]}
    for row, (v0, v1) in zip(reference_rows, values, strict=True):
        row.update(libvmaf_v0=v0, libvmaf_v1=v1)
    reference_csv = inputs.out_dir / "reference-444.csv"
    with reference_csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=reference_rows[0].keys())
        writer.writeheader()
        writer.writerows(reference_rows)

    return reference


def _write_scores(
    inputs: _CorpusInputs, baseline: _ScoreArms, observed: _ScoreArms
) -> None:
    out_dir, pairs, mcos = inputs.out_dir, inputs.pairs, inputs.mcos
    score_csv = out_dir / "scores.csv"
    with score_csv.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "original", "distorted", "MCOS", "legacy_v0", "legacy_v1", "vqm_v0_444",
                "vqm_v1_444", "vqm_v1_420"
            )
        )
        for i, (original, distorted) in enumerate(pairs):
            writer.writerow(
                (
                    original, distorted, mcos[i], baseline["v0"][i], baseline["v1"][i],
                    observed["v0_444"][i], observed["v1_444"][i], observed["v1_420"][i]
                )
            )


def _score_statistics(
    baseline: _ScoreArms, observed: _ScoreArms, reference: _ScoreArms, mcos: list[float]
) -> dict[str, object]:
    count = len(reference["v0"])
    full_correlations = {
        "legacy_v0": correlations(baseline["v0"], mcos),
        "legacy_v1": correlations(baseline["v1"], mcos)
    }
    full_correlations.update(
        {
            "vqm_" + name: correlations(values, mcos)
            for name, values in observed.items()
        }
    )
    subset_correlations = {}
    subset_parity = {}
    for model in ("v0", "v1"):
        subset_correlations["libvmaf_" +
                            model] = correlations(reference[model], mcos[:count])
        subset_correlations["vqm_" + model] = correlations(
            observed[model + "_444"][:count], mcos[:count]
        )
        subset_parity[model] = difference(
            observed[model + "_444"][:count], reference[model]
        )
    return {
        "full_corpus_correlations": full_correlations,
        "legacy_score_shifts": {
            model: difference(observed[model + "_444"], baseline[model])
            for model in ("v0", "v1")
        },
        "v1_420_vs_444": difference(observed["v1_420"], observed["v1_444"]),
        "reference_subset_correlations": subset_correlations,
        "reference_subset_score_difference": subset_parity,
    }


def _summary(
    inputs: _CorpusInputs, baseline_dir: Path | None,
    baseline_provenance: dict[str, bool | int] | None, assets: VmafPaths, bench: Path,
    count: int, statistics: dict[str, object]
) -> dict[str, object]:
    executable, model0, model1 = assets
    pairs, mcos_file, out_dir = inputs.pairs, inputs.mcos_file, inputs.out_dir
    pairs_path, reference_csv, score_csv = (
        out_dir / name for name in ("rgb-pairs.txt", "reference-444.csv", "scores.csv")
    )
    info = {
        "pairs": len(pairs),
        "reference_pairs": count,
        "conversion": CONVERSION,
        "legacy_conversion": LEGACY_CONVERSION,
        "legacy_source": str(baseline_dir.resolve()) if baseline_dir else "regenerated",
        "legacy_provenance": baseline_provenance,
        "legacy_note":
        "Legacy encoded-sRGB 10-bit conversion is a different input convention and truncates RGB16 PNGs to RGB8; score differences are not same-input implementation parity.",
        "reference_note":
        "CPU float64 and GPU FP32 conversion can differ at quantization boundaries; PNG/JPEG decoders may also differ.",
        "temporal": "independent one-frame sequences; motion=0",
        "libvmaf_revision": REVISION,
        "driver_sha256": vmaf_cid22.digest(executable),
        "v0_model_sha256": vmaf_cid22.digest(model0),
        "v1_model_sha256": vmaf_cid22.digest(model1),
        "vqm_benchmark_sha256": vmaf_cid22.digest(bench),
        "mcos_sha256": vmaf_cid22.digest(mcos_file),
        "rgb_pairs_sha256": vmaf_cid22.digest(pairs_path),
        "reference_manifest_sha256": vmaf_cid22.digest(reference_csv),
        "scores_sha256": vmaf_cid22.digest(score_csv),
        "numpy": np.__version__,
        "pillow": pillow_version,
        "ffmpeg": ffmpeg_version(),
        **statistics
    }
    if baseline_dir:
        info["baseline_sha256"] = {
            model: vmaf_cid22.digest(baseline_dir / f"vqm-{model}.csv")
            for model in ("v0", "v1")
        }
    return info


def main() -> None:
    """Compare VQM's SDR conversion with CID22 ratings and pinned libvmaf."""
    args = parse_args()
    inputs = _prepare_corpus(args)
    baseline_provenance = None
    if args.baseline_dir:
        baseline_provenance = validate_baseline(
            args.baseline_dir, inputs.pairs, inputs.data_root, inputs.mcos_file,
            args.jobs
        )
    if args.verify_baseline_only:
        print(json.dumps(baseline_provenance, indent=2))
        return
    assets, bench = _reference_assets()
    baseline = legacy_scores(
        inputs.pairs, inputs.data_root, inputs.out_dir, args.jobs, args.baseline_dir
    )
    observed = _rgb_scores(inputs, args.reuse_rgb_dir, bench)
    reference = _reference_scores(inputs, args.reference_count, args.jobs, assets)
    _write_scores(inputs, baseline, observed)
    statistics = _score_statistics(baseline, observed, reference, inputs.mcos)
    info = _summary(
        inputs, args.baseline_dir, baseline_provenance, assets, bench,
        len(reference["v0"]), statistics
    )
    (inputs.out_dir / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"vmaf-sdr-cid22: {error}", file=sys.stderr)
        sys.exit(1)
