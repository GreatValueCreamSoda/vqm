#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Score CID22 still pairs with pinned libvmaf v0.6.1 float and v1 SDR.

Inputs are cached as planar YUV444P10LE so another implementation can consume
exactly the same bytes. Each pair is an independent one-frame sequence. This
is a still-image experiment: temporal features, including motion, are zero.
See docs/vmaf-development.md for setup and bench/notes/vmaf/vmaf-reference.md
for the recorded CID22 comparison.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypedDict, TypeVar, cast

import numpy as np
from common import Pair, correlation_triple, enumerate_pairs, load_mcos
from compare_vmaf_scores import direct_correlations, pair_input_hash, pair_sample_id
from numpy.typing import NDArray
from PIL import Image
from PIL import __version__ as pillow_version

ROOT = Path(__file__).resolve().parent.parent
YUV_CACHE = Path(".cache/validation/vmaf-cid22-yuv")
CONVERSION = "rgb8-srgb-encoded-bt709-limited-yuv444p10le-rne-v1"
REFERENCE_REVISION = "4611e47a7330cd342498dcc130e1008efbeab6e4"
AGREEMENT_CONFIGURATION = "cid22-yuv444p10le"


class ConvertedFrame(TypedDict):
    """Describe an independently converted frame and its source fingerprint."""

    file: str
    width: int
    height: int
    source_sha256: str
    yuv_sha256: str


CorpusRow = dict[str, object]
VmafPaths = tuple[Path, Path, Path]
_Input = TypeVar("_Input")
_Result = TypeVar("_Result")


def digest(path: str | Path) -> str:
    """Hash a file without loading it into memory in full.

    :param path: File to read.
    :returns: Lowercase SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_rgb(rgb: NDArray[np.uint8]) -> bytes:
    """Convert encoded RGB through BT.709 to 10-bit TV range with even ties.

    :param rgb: RGB8 samples in height, width and channel order.
    :returns: Planar little-endian YUV444P10 bytes.
    :raises ValueError: If the input is not an RGB8 image.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected RGB8 image")
    r, g, b = (rgb[:, :, i].astype(np.float64) for i in range(3))
    y = (2126 * r + 7152 * g + 722 * b) / 10000.0
    planes = [
        np.rint(64 + 876 * y / 255),
        np.rint(512 + 896 * (b - y) / (255 * 2 * 0.9278)),
        np.rint(512 + 896 * (r - y) / (255 * 2 * 0.7874))
    ]
    return b"".join(np.clip(p, 0, 1023).astype("<u2").tobytes() for p in planes)


def decode_rgb8(image: Image.Image, path: Path) -> NDArray[np.uint8]:
    """Decode an RGB or grayscale image, checking that any alpha is opaque.

    :param image: Open Pillow image.
    :param path: Image path for error messages.
    :returns: Decoded RGB8 samples.
    :raises ValueError: If the mode is unsupported or alpha is nonopaque.
    """
    if image.mode in ("RGBA", "LA"):
        if image.getchannel("A").getextrema() != (255, 255):
            raise ValueError(f"{path}: non-opaque alpha is unsupported")
    elif image.mode not in ("RGB", "L"):
        raise ValueError(f"{path}: unsupported image mode {image.mode}")
    return np.asarray(image.convert("RGB"))


def write_cache_bytes(target: Path, prefix: str, *chunks: bytes) -> None:
    """Publish converted bytes atomically from a temporary file.

    :param target: Path to publish in the cache.
    :param prefix: Prefix for the private temporary file name.
    :param chunks: Payload parts to write in order.
    """
    fd, temporary = tempfile.mkstemp(prefix=prefix, dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.writelines(chunks)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def convert_file(path: Path, cache: Path) -> ConvertedFrame:
    """Convert one still and verify or repair its content-addressed cache.

    :param path: Source image.
    :param cache: Existing directory for planar YUV bytes.
    :returns: Converted frame path, geometry and input fingerprints.
    :raises ValueError: If the image format or geometry is unsupported.
    """
    source_hash = digest(path)
    key = hashlib.sha256((CONVERSION + ":" + source_hash).encode()).hexdigest()
    target = cache / f"{key}.yuv"
    # Decode even cached inputs: validating dimensions, mode and RGB sample
    # type makes accidental file replacement or a wrong-format corpus fail.
    with Image.open(path) as image:
        rgb = decode_rgb8(image, path)
    height, width = rgb.shape[:2]
    if min(width, height) < 17:
        raise ValueError(f"{path}: geometry too small for VMAF")
    raw = convert_rgb(rgb)
    # The output hash validates a pre-existing cache rather than trusting its
    # filename. Recompute if missing or damaged.
    expected = hashlib.sha256(raw).hexdigest()
    if not target.exists() or digest(target) != expected:
        write_cache_bytes(target, ".vmaf-", raw)
    return {
        "file": str(target.resolve()),
        "width": width,
        "height": height,
        "source_sha256": source_hash,
        "yuv_sha256": expected
    }


def reference_score(row: CorpusRow, executable: Path, model0: Path,
                    model1: Path) -> list[float]:
    """Score one raw 10-bit pair with both pinned libvmaf models.

    :param row: Prepared pair paths and frame dimensions.
    :param executable: Pinned libvmaf reference driver.
    :param model0: VMAF v0 model path.
    :param model1: VMAF v1 model path.
    :returns: The v0 and v1 frame scores in that order.
    :raises RuntimeError: If the driver fails.
    :raises ValueError: If its output is not one finite score per model.
    """
    argv = [
        str(executable), "models",
        str(model0), row["original_yuv"], row["distorted_yuv"],
        str(row["width"]),
        str(row["height"]), "10",
        str(model1)
    ]
    result = subprocess.run(argv, text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"{row['distorted']}: {result.stderr.strip()}")
    values = list(csv.DictReader(result.stdout.splitlines()))
    if len(values) != 1 or values[0].get("frame") != "0":
        raise ValueError(f"{row['distorted']}: expected exactly frame zero")
    scores = [float(values[0][name]) for name in ("v0", "v1")]
    if not all(math.isfinite(value) for value in scores):
        raise ValueError(f"{row['distorted']}: non-finite score")
    return scores


def write_y4m(raw_path: str | Path, output_dir: Path, width: int, height: int) -> str:
    """Wrap the exact raw 10-bit LE samples in a one-frame Y4M file.

    :param raw_path: Planar YUV444P10LE data.
    :param output_dir: Existing directory for wrappers.
    :param width: Coded frame width.
    :param height: Coded frame height.
    :returns: Path to the verified Y4M wrapper.
    :raises ValueError: If the byte length disagrees with the dimensions.
    """
    source = Path(raw_path)
    raw = source.read_bytes()
    if len(raw) != width * height * 3 * 2:
        raise ValueError(f"{source}: wrong YUV444P10LE frame length")
    header = f"YUV4MPEG2 W{width} H{height} F1:1 Ip A0:0 C444p10\nFRAME\n".encode()
    target = output_dir / (source.stem + ".y4m")
    expected = hashlib.sha256(header + raw).digest()
    if not target.exists() or bytes.fromhex(digest(target)) != expected:
        write_cache_bytes(target, ".vmaf-y4m-", header, raw)
    return str(target)


def _wrapper_sizes(rows: list[CorpusRow]) -> dict[str, tuple[int, int]]:
    source_sizes = {}
    files = (
        (row[key], (row["width"], row["height"])) for row in rows
        for key in ("original_yuv", "distorted_yuv")
    )
    for path, size in files:
        if source_sizes.setdefault(path, size) != size:
            raise ValueError("same YUV file has mismatched dimensions")
    return source_sizes


def _write_vqm_pairs(rows: list[CorpusRow], output_dir: Path, jobs: int) -> Path:
    cache = ROOT / ".cache/validation/vmaf-cid22-y4m"
    cache.mkdir(parents=True, exist_ok=True)
    source_sizes = _wrapper_sizes(rows)
    sources = sorted(source_sizes)
    wrappers = parallel_map(
        "Y4M", lambda path: write_y4m(path, cache, *source_sizes[path]), sources, jobs
    )
    lookup = dict(zip(sources, wrappers, strict=True))
    pairs_file = output_dir / "vqm-pairs.txt"
    with pairs_file.open("w") as output:
        for row in rows:
            output.write(
                f"{lookup[row['original_yuv']]}\t{lookup[row['distorted_yuv']]}\n"
            )
    return pairs_file


def vqm_scores(rows: list[CorpusRow], output_dir: Path, jobs: int,
               model: str) -> list[float]:
    """Score exact inputs with one VQM model and save per-pair results.

    :param rows: Prepared reference/distorted still pairs.
    :param output_dir: Destination for pair list, logs and scores.
    :param jobs: Concurrent workers wrapping raw files in Y4M.
    :param model: VMAF version selector, "v0" or "v1".
    :returns: Finite scores in the same order as the supplied rows.
    :raises ValueError: If the model, executable or score output is invalid.
    """
    if model not in ("v0", "v1"):
        raise ValueError(f"unsupported VQM model: {model}")
    executable = ROOT / "build/vqm_bench"
    if not executable.is_file():
        raise ValueError(f"missing {executable}; run cmake --build build -j8")
    pairs_file = _write_vqm_pairs(rows, output_dir, jobs)
    command = [
        str(executable), "--metric", "vmaf-" + model, "--score-only", "--input-matrix",
        "bt709", "--pairs-file",
        str(pairs_file)
    ]
    print(f"VQM {model}: scoring {len(rows)} independent still pairs", flush=True)
    result = subprocess.run(
        command, text=True, capture_output=True, timeout=120 + 2 * len(rows)
    )
    log_file = output_dir / f"vqm-{model}.log"
    log_file.write_text(" ".join(command) + "\n" + result.stderr)
    (output_dir / f"vqm-{model}.stdout").write_text(result.stdout)
    lines = result.stdout.splitlines()
    if result.returncode or len(lines) != len(rows):
        raise ValueError(
            f"VQM exited {result.returncode}; expected {len(rows)} scores, "
            f"got {len(lines)}; see {log_file}"
        )
    values = [float(line) for line in lines]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("VQM produced non-finite scores")
    score_file = output_dir / f"vqm-{model}.csv"
    with score_file.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["original", "distorted", model])
        writer.writerows(
            (row["original"], row["distorted"], value)
            for row, value in zip(rows, values, strict=True)
        )
    return values


def parallel_map(
    label: str, fn: Callable[[_Input], _Result], values: Sequence[_Input], jobs: int
) -> list[_Result]:
    """Process values concurrently while preserving their input order.

    :param label: Operation name for progress messages.
    :param fn: Operation to run for each input.
    :param values: Ordered inputs.
    :param jobs: Maximum number of concurrent workers.
    :returns: Results in input order.
    """
    results: list[_Result | None] = [None] * len(values)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(fn, value): i for i, value in enumerate(values)}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results[futures[future]] = future.result()
            _print_progress(label, completed, len(values))
    return cast(list[_Result], results)


def _print_progress(label: str, completed: int, total: int) -> None:
    if completed % 250 == 0 or completed == total:
        print(f"{label}: {completed}/{total}", flush=True)


def _supplied_rows(path: Path,
                   require_hashes: bool) -> tuple[list[str], dict[Pair, CorpusRow]]:
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        models = [name for name in ("v0", "v1") if name in fields]
        if not models or not {"original", "distorted"}.issubset(fields):
            raise ValueError(
                "VQM scores need original,distorted and at least one of v0,v1"
            )
        if require_hashes and not {"original_yuv_sha256", "distorted_yuv_sha256"
                                   }.issubset(fields):
            raise ValueError(
                "ranking acceptance requires supplied VQM decoded-input hashes"
            )
        rows = list(reader)
    provided = {}
    for value in rows:
        key = value["original"], value["distorted"]
        if key in provided:
            raise ValueError(f"duplicate VQM pair {key}")
        scores = {name: float(value[name]) for name in models}
        if not all(math.isfinite(score) for score in scores.values()):
            raise ValueError("VQM contains non-finite scores")
        provided[key] = {**value, **scores}
    return models, provided


def _apply_supplied(args: argparse.Namespace, rows: list[CorpusRow],
                    pairs: list[Pair]) -> list[str]:
    if not args.vqm_scores:
        return []
    models, supplied = _supplied_rows(args.vqm_scores, bool(args.agreement_policy))
    if set(supplied) != set(pairs):
        raise ValueError("VQM score pairs do not exactly match selected CID22 pairs")
    for row in rows:
        value = supplied[(row["original"], row["distorted"])]
        hashes = ("original_yuv_sha256", "distorted_yuv_sha256")
        if args.agreement_policy and any(value[name] != row[name] for name in hashes):
            raise ValueError(
                "supplied VQM decoded-input hashes differ from prepared inputs"
            )
        row.update({"vqm_" + model: value[model] for model in models})
    return models


def _agreement_sample(row: CorpusRow, column: str) -> dict[str, object]:
    pair = str(row["original"]), str(row["distorted"])
    return {
        "sample_id":
        pair_sample_id(pair),
        "source_id":
        pair[0],
        "configuration_id":
        AGREEMENT_CONFIGURATION,
        "stratum":
        "still-sdr",
        "distortion_family":
        Path(pair[1]).parent.name,
        "input_hash":
        pair_input_hash(
            (str(row["original_yuv_sha256"]), str(row["distorted_yuv_sha256"]))
        ),
        "score":
        row[column],
        "mos":
        row["mcos"]
    }


def build_agreement_reports(
    rows: list[CorpusRow], model: str, model_hash: str, manifest_hash: str,
    candidate_revision: str
) -> tuple[dict, dict]:
    """Build canonical single-frame reports from the measured CID22 input manifest.

    :param rows: Exact paired scores and decoded input hashes.
    :param model: VMAF v0 or v1 model identity.
    :param model_hash: Hash of the pinned model JSON.
    :param manifest_hash: Hash of the prepared input manifest.
    :param candidate_revision: Supplied revision or fingerprint of the invoked binary.
    :returns: Original and candidate canonical reports with matched provenance.
    :raises ValueError: If canonical scores or provenance are invalid.
    """
    from agreement import report_from_rows

    configuration = {
        "conversion": CONVERSION,
        "model_sha256": model_hash,
        "chroma": "444",
        "bitdepth": 10,
        "geometry": "native",
        "temporal": "independent-single-frame-motion-zero"
    }
    configuration_hash = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    shared = {
        "metric": "vmaf-" + model,
        "configuration_hash": configuration_hash,
        "input_manifest_hash": manifest_hash
    }
    reference = report_from_rows(
        [_agreement_sample(row, "libvmaf_" + model) for row in rows],
        implementation="libvmaf", revision=REFERENCE_REVISION, **shared
    )
    candidate = report_from_rows(
        [_agreement_sample(row, "vqm_" + model) for row in rows], implementation="vqm",
        revision=candidate_revision, **shared
    )
    return reference, candidate


def _record_agreement(
    args: argparse.Namespace, rows: list[CorpusRow], models: list[str],
    info: dict[str, object]
) -> bool:
    from agreement import evaluate_agreement

    policy = None if not args.agreement_policy else json.loads(
        args.agreement_policy.read_text()
    )
    policy_model = None if policy is None else str(policy.get("metric", "")
                                                   ).removeprefix("vmaf-")
    if policy is not None and policy_model not in models:
        raise ValueError("agreement policy requires a matching selected VQM model")
    binary_models = {
        model
        for model, requested in (("v0", args.vqm_v0), ("v1", args.vqm_v1)) if requested
    }
    reports = {}
    for model in models:
        revision = (
            "binary-sha256:" +
            str(info["vqm_benchmark_sha256"]) if model in binary_models else
            args.candidate_revision or "unreported-external-scores"
        )
        reference, candidate = build_agreement_reports(
            rows, model, str(info[model + "_model_sha256"]),
            str(info["input_manifest_sha256"]), revision
        )
        for label, report in (("reference", reference), ("candidate", candidate)):
            target = args.out_dir / f"{model}-{label}-agreement.json"
            target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if model == policy_model:
            reports[model] = evaluate_agreement(reference, candidate, policy)
    info["agreement"] = reports if policy else {"status": "not_requested"}
    return policy is None or reports[policy_model]["proposed_status"] == "pass"


def parse_args() -> argparse.Namespace:
    """Read corpus, scoring and optional frozen acceptance settings.

    :returns: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "validation/data/CID22_validation_set"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "artifacts/reports/vmaf-cid22"
    )
    parser.add_argument("--cache", type=Path, default=ROOT / YUV_CACHE)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--limit", type=int, default=0,
        help="first N pairs, zero selects the full corpus"
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--vqm-scores", type=Path,
        help="optional CSV: original,distorted and v0 and/or v1"
    )
    parser.add_argument(
        "--vqm-v0", action="store_true",
        help="also score exact inputs with the VQM v0 benchmark"
    )
    parser.add_argument(
        "--vqm-v1", action="store_true",
        help="also score exact inputs with the VQM v1 benchmark"
    )
    parser.add_argument(
        "--reference-scores", type=Path,
        help="reuse a prior scores.csv with matching summary.json"
    )
    parser.add_argument(
        "--agreement-policy", type=Path,
        help="frozen direct-ranking policy for one selected VMAF model"
    )
    parser.add_argument(
        "--candidate-revision",
        help="revision of supplied --vqm-scores (required with a policy)"
    )
    args = parser.parse_args()
    if args.jobs < 1 or args.limit < 0:
        parser.error("--jobs must be positive and --limit non-negative")
    if args.agreement_policy and args.prepare_only:
        parser.error("--agreement-policy cannot be combined with --prepare-only")
    if args.agreement_policy and args.vqm_scores and not args.candidate_revision:
        parser.error("ranking acceptance of --vqm-scores requires --candidate-revision")
    return args


def _prepare_rows(args: argparse.Namespace) -> tuple[list[Pair], list[CorpusRow], Path]:
    data_root = args.data_root.resolve()
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = enumerate_pairs(str(data_root))
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        raise ValueError("no CID22 pairs selected")
    mcos_file = data_root / "CID22_validation_set.csv"
    mcos = load_mcos(str(mcos_file))
    paths = sorted({name for pair in pairs for name in pair})
    converted = parallel_map(
        "convert", lambda path: convert_file(data_root / path, cache), paths, args.jobs
    )
    lookup = dict(zip(paths, converted, strict=True))
    rows = []
    for original, distorted in pairs:
        ref, dist = lookup[original], lookup[distorted]
        if (ref["width"], ref["height"]) != (dist["width"], dist["height"]):
            raise ValueError(f"geometry mismatch: {original}, {distorted}")
        if (original, distorted) not in mcos:
            raise ValueError(f"missing MCOS: {original}, {distorted}")
        rows.append(
            {
                "original": original,
                "distorted": distorted,
                "mcos": mcos[(original, distorted)],
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
    return pairs, rows, mcos_file


def _write_rows(path: Path, rows: list[CorpusRow]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _reference_paths() -> VmafPaths:
    executable = ROOT / "build/vmaf-build/vmaf_reference"
    model0 = ROOT / "build/vmaf-src/model/vmaf_float_v0.6.1.json"
    model1 = ROOT / "build/vmaf-src/model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json"
    for path in (executable, model0, model1):
        if not path.is_file():
            raise ValueError(
                f"missing {path}; run python3 tools/build_references.py --only vmaf"
            )
    return executable, model0, model1


def _reuse_reference_scores(
    path: Path, rows: list[CorpusRow], manifest: Path, mcos_file: Path,
    assets: VmafPaths
) -> list[tuple[float, float]]:
    executable, model0, model1 = assets
    prior_summary = json.loads(path.with_name("summary.json").read_text())
    for key, value in (("input_manifest_sha256", digest(manifest)), ("v0_model_sha256",
                                                                     digest(model0)),
                       ("v1_model_sha256", digest(model1)), ("driver_sha256",
                                                             digest(executable)),
                       ("mcos_sha256", digest(mcos_file)), ("conversion", CONVERSION)):
        if prior_summary.get(key) != value:
            raise ValueError(f"prior reference does not match: {key}")
    with path.open(newline="") as source:
        previous = list(csv.DictReader(source))
    if len(previous) != len(rows):
        raise ValueError("prior reference pair count differs")
    results = []
    for row, old in zip(rows, previous, strict=True):
        if (row["original"], row["distorted"]) != (old["original"], old["distorted"]):
            raise ValueError("prior reference pair order differs")
        values = float(old["libvmaf_v0"]), float(old["libvmaf_v1"])
        if not all(math.isfinite(value) for value in values):
            raise ValueError("prior reference contains non-finite scores")
        results.append(values)
    print(f"libvmaf: reused {len(results)} matching reference scores", flush=True)
    return results


def _add_reference_scores(
    args: argparse.Namespace, rows: list[CorpusRow], manifest: Path, mcos_file: Path,
    assets: VmafPaths
) -> None:
    if args.reference_scores:
        results = _reuse_reference_scores(
            args.reference_scores, rows, manifest, mcos_file, assets
        )
    else:
        results = parallel_map(
            "libvmaf", lambda row: reference_score(row, *assets), rows, args.jobs
        )
    for row, (v0, v1) in zip(rows, results, strict=True):
        row["libvmaf_v0"], row["libvmaf_v1"] = v0, v1


def _add_vqm_scores(args: argparse.Namespace, rows: list[CorpusRow],
                    pairs: list[Pair]) -> list[str]:
    vqm_models = _apply_supplied(args, rows, pairs)
    for model, requested in (("v0", args.vqm_v0), ("v1", args.vqm_v1)):
        if not requested:
            continue
        if model in vqm_models:
            raise ValueError(f"VQM {model} supplied twice")
        vqm_models.append(model)
        values = vqm_scores(rows, args.out_dir, args.jobs, model)
        for row, value in zip(rows, values, strict=True):
            row["vqm_" + model] = value
    return vqm_models


def _correlations(rows: list[CorpusRow],
                  vqm_models: list[str]) -> dict[str, dict[str, float]]:
    names = ["libvmaf_v0", "libvmaf_v1"]
    if vqm_models:
        names += ["vqm_" + name for name in vqm_models]
    correlations = {
        name:
        dict(
            zip(
                ("KRCC", "SRCC", "PCC"),
                correlation_triple(
                    [row[name] for row in rows], [row["mcos"] for row in rows]
                ), strict=True
            )
        )
        for name in names
    }
    if not all(math.isfinite(value) for values in correlations.values()
               for value in values.values()):
        raise ValueError("not enough varying data for correlations")
    return correlations


def _parity(rows: list[CorpusRow],
            vqm_models: list[str]) -> dict[str, dict[str, float]]:
    parity = {}
    for model in ("v0", "v1"):
        if model in vqm_models:
            difference = np.asarray(
                [row["vqm_" + model] - row["libvmaf_" + model] for row in rows]
            )
            absolute = np.abs(difference)
            parity[model] = {
                "max_absolute": float(absolute.max()),
                "mean_absolute": float(absolute.mean()),
                "p99_absolute": float(np.percentile(absolute, 99)),
                "rmse": float(np.sqrt(np.mean(difference**2)))
            }
    return parity


def _build_summary(
    args: argparse.Namespace, rows: list[CorpusRow], vqm_models: list[str],
    manifest: Path, mcos_file: Path, assets: VmafPaths
) -> dict[str, object]:
    executable, model0, model1 = assets
    correlations = _correlations(rows, vqm_models)
    parity = _parity(rows, vqm_models)
    info = {
        "pairs": len(rows),
        "conversion": CONVERSION,
        "libvmaf_revision": REFERENCE_REVISION,
        "v0_model_sha256": digest(model0),
        "v1_model_sha256": digest(model1),
        "driver_sha256": digest(executable),
        "mcos_sha256": digest(mcos_file),
        "input_manifest_sha256": digest(manifest),
        "numpy": np.__version__,
        "pillow": pillow_version,
        "temporal": "independent single-frame sequences; motion=0",
        "correlations": correlations,
        "parity": parity,
        "direct_agreement": {
            model:
            direct_correlations(
                [row["libvmaf_" + model]
                 for row in rows], [row["vqm_" + model] for row in rows]
            )
            for model in vqm_models
        }
    }
    if args.vqm_v0 or args.vqm_v1:
        info["vqm_benchmark_sha256"] = digest(ROOT / "build/vqm_bench")
    if args.vqm_scores:
        info["vqm_scores_sha256"] = digest(args.vqm_scores)
    return info


def main() -> int:
    """Run the still-image experiment and optional reference-ranking acceptance.

    :returns: Zero unless an explicitly requested agreement policy fails.
    """
    args = parse_args()
    pairs, rows, mcos_file = _prepare_rows(args)
    manifest = args.out_dir / "inputs.csv"
    _write_rows(manifest, rows)
    print(f"input manifest: {manifest}", flush=True)
    if args.prepare_only:
        return 0
    assets = _reference_paths()
    _add_reference_scores(args, rows, manifest, mcos_file, assets)
    vqm_models = _add_vqm_scores(args, rows, pairs)
    _write_rows(args.out_dir / "scores.csv", rows)
    info = _build_summary(args, rows, vqm_models, manifest, mcos_file, assets)
    agreement_passed = _record_agreement(args, rows, vqm_models, info)
    (args.out_dir / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0 if agreement_passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"vmaf-cid22: {error}", file=sys.stderr)
        sys.exit(1)
