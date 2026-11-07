#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Compare VQM CVVDP with pinned original CPU FP32 on identical decoded RGB8.

CID22 is decoded once with Pillow into lossless RGB8 PNGs. Both arms consume
those exact samples, so JPEG decoder differences cannot masquerade as metric
error. Scores are raw tensor predictions from the original authors' v0.5.7.
Reference dependencies are optional; see docs/cvvdp-development.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TextIO

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
PIN = "2a268bce8d56e2f3abde46df3927d8a633707a24"
# Fixed viewing geometries from the pinned original display models.
DISPLAY_PPD = {
    "standard_4k": "75.402449345163049",
    "standard_fhd": "37.842503853602885",
}
sys.path.insert(0, str(ROOT / ".reference/ColorVideoVDP"))

import numpy as np  # noqa: E402
import pycvvdp  # noqa: E402
import torch  # noqa: E402
from agreement import default_gate_mode, evaluate_agreement, report_from_rows  # noqa: E402
from agreement_policy import AgreementPolicy  # noqa: E402
from common import enumerate_pairs, load_mcos  # noqa: E402
from PIL import Image  # noqa: E402
from PIL import __version__ as pillow_version  # noqa: E402
from pycvvdp.display_model import vvdp_display_photo_eotf  # noqa: E402


@dataclasses.dataclass(frozen=True)
class Source:
    """One source and its canonical decoded RGB8 PNG representation."""

    source: str
    source_sha256: str
    decoded: str
    rgb_sha256: str
    width: int
    height: int


def digest(path: Path) -> str:
    """Hash one file without retaining its full contents.

    :param path: Existing input file.
    :returns: Hexadecimal SHA-256.
    """
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def canonical_source(path: Path, cache: Path) -> Source:
    """Decode once to lossless RGB8 samples shared by both implementations.

    :param path: PNG/JPEG input path.
    :param cache: Generated RGB8 PNG directory.
    :returns: Source and decoded-sample provenance.
    :raises ValueError: For nonopaque alpha or an unsupported sample mode.
    """
    with Image.open(path) as image:
        if image.mode in ("RGBA", "LA") and image.getchannel("A").getextrema() != (255,
                                                                                   255):
            raise ValueError(f"{path}: nonopaque alpha is unsupported")
        if image.mode not in ("RGB", "L", "RGBA", "LA"):
            raise ValueError(f"{path}: expected RGB8/grey8, found {image.mode}")
        rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.tobytes()
    rgb_hash = hashlib.sha256(f"RGB8:{width}x{height}:".encode() + pixels).hexdigest()
    target = cache / f"{rgb_hash}.png"
    if target.exists():
        with Image.open(target) as existing:
            valid = existing.mode == "RGB" and existing.size == rgb.size and existing.tobytes(
            ) == pixels
        if not valid:
            raise ValueError(f"{target}: decoded cache was modified")
    else:
        rgb.save(target, format="PNG")
    return Source(
        str(path.resolve()), digest(path), str(target.resolve()), rgb_hash, width,
        height
    )


def selected_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    """Read a custom pair list or select a deterministic CID22 subset.

    :param args: Parsed paths and selection options.
    :returns: Ordered source path pairs.
    :raises ValueError: For malformed pair lists or empty selections.
    """
    if args.pairs_file:
        rows = read_pair_list(args.pairs_file)
    else:
        rows = [
            (args.corpus / a, args.corpus / b)
            for a, b in enumerate_pairs(str(args.corpus))
        ]
    if args.limit and len(rows) > args.limit:
        indices = np.linspace(0, len(rows) - 1, args.limit, dtype=np.int64)
        rows = [rows[int(index)] for index in indices]
    if not rows:
        raise ValueError("no input pairs selected")
    return rows


def read_pair_list(path: Path) -> list[tuple[Path, Path]]:
    """Read the repository's original-tab-distorted pair format.

    :param path: UTF-8 pair list.
    :returns: Ordered path pairs.
    :raises ValueError: For malformed lines.
    """
    pairs = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"{path}:{number}: expected original<TAB>distorted")
        pairs.append((Path(fields[0]), Path(fields[1])))
    return pairs


def prepare_pairs(pairs: list[tuple[Path, Path]],
                  cache: Path) -> list[tuple[Source, Source]]:
    """Canonicalize each distinct input once and validate paired geometries.

    :param pairs: Ordered original/distorted paths.
    :param cache: Generated lossless RGB cache.
    :returns: Ordered decoded source records.
    :raises ValueError: If a pair's dimensions differ.
    """
    cache.mkdir(parents=True, exist_ok=True)
    paths = dict.fromkeys(path for pair in pairs for path in pair)
    sources = {path: canonical_source(path, cache) for path in paths}
    rows = []
    for original, distorted in pairs:
        a, b = sources[original], sources[distorted]
        if (a.width, a.height) != (b.width, b.height):
            raise ValueError(f"{original}, {distorted}: unequal dimensions")
        rows.append((a, b))
    return rows


def metric_for(display: str) -> pycvvdp.cvvdp:
    """Construct one explicitly sRGB, CPU FP32 original metric.

    :param display: Original physical preset name.
    :returns: Original metric reused across pairs and geometries.
    """
    resolved = pycvvdp.vvdp_display_photometry.load(display, [])
    photometry = vvdp_display_photo_eotf(
        resolved.Y_peak, contrast=resolved.contrast, source_colorspace="sRGB",
        E_ambient=resolved.E_ambient, k_refl=resolved.k_refl, exposure=resolved.exposure
    )
    return pycvvdp.cvvdp(
        display_name=display, display_photometry=photometry, device=torch.device("cpu"),
        quiet=True
    )


def raw_tensor(path: str) -> torch.Tensor:
    """Read canonical PNG samples directly into a normalized FP32 tensor.

    :param path: Canonical RGB8 PNG path.
    :returns: [1,3,1,H,W] FP32 tensor with exact code/255 conversion.
    """
    with Image.open(path) as image:
        rgb = np.asarray(image).copy()
    return torch.from_numpy(rgb).permute(2, 0,
                                         1).unsqueeze(0).unsqueeze(2).float() / 255.0


def cache_key(a: Source, b: Source, display: str) -> str:
    """Identify all numerical inputs to one deterministic reference score.

    :param a: Reference decoded pixels.
    :param b: Distorted decoded pixels.
    :param display: Physical preset name.
    :returns: Cache key including implementation, precision and runtime version.
    """
    settings = [
        PIN, torch.__version__, "cpu-fp32-highest-threads1", display, a.rgb_sha256,
        b.rgb_sha256
    ]
    return hashlib.sha256(json.dumps(settings).encode()).hexdigest()


def load_score_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Load append-only original score evidence without trusting stale settings.

    :param path: JSONL cache path.
    :returns: Records indexed by their complete numerical-input key.
    """
    if not path.exists():
        return {}
    return {
        row["key"]: row
        for row in (json.loads(line) for line in path.read_text().splitlines())
    }


def original_scores(rows: list[tuple[Source, Source]],
                    args: argparse.Namespace) -> list[dict[str, Any]]:
    """Evaluate original scores, reusing only fully identified cached inputs.

    :param rows: Canonical paired sources.
    :param args: Display and output/cache options.
    :returns: Ordered scores with CPU evaluation time and cache provenance.
    """
    metric = metric_for(args.display)
    cached = load_score_cache(args.reference_cache)
    args.reference_cache.parent.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.monotonic()
    with args.reference_cache.open("a") as stream:
        for index, (a, b) in enumerate(rows):
            key = cache_key(a, b, args.display)
            results.append(cached_original(metric, a, b, key, cached, stream))
            print_progress(index + 1, len(rows), started)
    return results


def cached_original(
    metric: pycvvdp.cvvdp, a: Source, b: Source, key: str,
    cached: dict[str, dict[str, Any]], stream: TextIO
) -> dict[str, Any]:
    """Reuse or append a complete-input-keyed independent original result.

    :param metric: Pinned original model.
    :param a: Original source.
    :param b: Distorted source.
    :param key: Complete numerical input identity.
    :param cached: Previously verified input-keyed records.
    :param stream: Append-only reference evidence file.
    :returns: Original score and timing record.
    """
    if key in cached:
        return cached[key]
    result = evaluate_original(metric, a, b, key)
    stream.write(json.dumps(result) + "\n")
    stream.flush()
    cached[key] = result
    return result


def print_progress(count: int, total: int, started: float) -> None:
    """Report bounded, concise progress during long reference runs.

    :param count: Completed pair count.
    :param total: Selected pair count.
    :param started: Monotonic start time.
    """
    if count % 20 == 0 or count == total:
        print(
            f"original {count}/{total}, {time.monotonic() - started:.1f}s elapsed",
            flush=True
        )


def evaluate_original(metric: pycvvdp.cvvdp, a: Source, b: Source,
                      key: str) -> dict[str, Any]:
    """Score one decoded pair through the original raw-tensor API.

    :param metric: Original model with resolved physical display.
    :param a: Original source.
    :param b: Distorted source.
    :param key: Complete numerical-input cache key.
    :returns: Raw JOD and elapsed original evaluation time.
    :raises ValueError: For a non-finite reference result.
    """
    original, distorted = raw_tensor(a.decoded), raw_tensor(b.decoded)
    start = time.monotonic()
    with torch.inference_mode():
        score, _ = metric.predict(distorted, original, frames_per_second=0)
    value = float(score)
    if not math.isfinite(value):
        raise ValueError(f"{b.source}: original returned a non-finite score")
    return {"key": key, "score": value, "seconds": time.monotonic() - start}


def candidate_scores(rows: list[tuple[Source, Source]],
                     args: argparse.Namespace) -> list[float]:
    """Run one VQM process over the same decoded samples.

    :param rows: Ordered canonical pair records.
    :param args: Candidate binary, display and output options.
    :returns: Finite JOD values in pair order.
    :raises ValueError: For process failures, missing or malformed scores.
    """
    pairs_file = args.out_dir / "candidate-pairs.txt"
    pairs_file.write_text("".join(f"{a.decoded}\t{b.decoded}\n" for a, b in rows))
    snapshot = args.out_dir / "candidate-binary"
    shutil.copyfile(args.binary, snapshot)
    snapshot.chmod(args.binary.stat().st_mode)
    device = subprocess.run(
        [str(snapshot.resolve()), "--device-info"], text=True, capture_output=True,
        check=True, timeout=60
    )
    json.loads(device.stdout)  # Reject malformed provenance before scoring.
    (args.out_dir / "device.json").write_text(device.stdout)
    command = [
        str(snapshot.resolve()), "--metric", "cvvdp", "--score-only", "--cvvdp-ppd",
        DISPLAY_PPD[args.display], "--pairs-file",
        str(pairs_file.resolve())
    ]
    result = subprocess.run(
        command, text=True, capture_output=True, timeout=120 + 5 * len(rows)
    )
    (args.out_dir / "candidate.stdout").write_text(result.stdout)
    (args.out_dir /
     "candidate.log").write_text(json.dumps(command) + "\n" + result.stderr)
    values = result.stdout.splitlines()
    if result.returncode or len(values) != len(rows):
        raise ValueError(
            f"candidate exit {result.returncode}, {len(values)}/{len(rows)} scores; see candidate.log"
        )
    scores = [float(line) for line in values]
    if not all(map(math.isfinite, scores)):
        raise ValueError("candidate returned a non-finite score")
    return scores


def write_report(
    rows: list[tuple[Source, Source]], original: list[dict[str, Any]],
    candidate: list[float] | None, args: argparse.Namespace
) -> bool:
    """Record complete pair provenance and the fixed absolute JOD gate.

    :param rows: Canonical source records.
    :param original: Ordered original scores and timings.
    :param candidate: Ordered VQM scores, absent for reference-only execution.
    :param args: Physical preset, candidate and output settings.
    :returns: Whether every candidate score satisfies the requested gate.
    """
    values = comparison_rows(rows, original, candidate)
    errors = [row["absolute_error"] for row in values if "absolute_error" in row]
    passed = bool(errors) and max(errors) <= args.tolerance
    metadata = candidate_provenance(args.out_dir) if candidate is not None else {}
    report = {
        "upstream_commit": PIN,
        "torch": torch.__version__,
        "pillow": pillow_version,
        "precision": "cpu-fp32-highest-threads1",
        "display": args.display,
        "binary": str(args.binary.resolve()),
        "tolerance": args.tolerance,
        "reference_only": candidate is None,
        "pairs": len(values),
        "max_absolute_error": max(errors, default=None),
        "mean_absolute_error": sum(errors) / len(errors) if errors else None,
        "passed": passed if candidate is not None else None,
        "results": values,
        **metadata
    }
    print(f'{len(values)} pairs, max absolute JOD error {report["max_absolute_error"]}')
    selected_passed = passed
    if getattr(args, "agreement_policy", None) is not None:
        agreement = write_agreement(rows, original, candidate, args, passed)
        selected_passed = agreement["selected_gate_passed"]
        report.update(
            gate_mode=agreement["gate_mode"], selected_gate_passed=selected_passed
        )
    else:
        print("REFERENCE ONLY" if candidate is None else "PASS" if passed else "FAIL")
    (args.out_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return selected_passed


def _json_digest(value: object) -> str:
    """Hash a deterministic, finite JSON representation of actual run inputs."""
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _agreement_configuration(args: argparse.Namespace) -> dict[str, object]:
    """Identify the physical display, source interpretation and pinned model data."""
    data = ROOT / ".reference/ColorVideoVDP/pycvvdp/vvdp_data"
    return {
        "metric": "cvvdp",
        "model": "base-v0.5.7",
        "reference_revision": PIN,
        "display": args.display,
        "input": "opaque-RGB8-sRGB",
        "input_geometry": "native",
        "frames_per_second": 0,
        "precision": "cpu-fp32-highest-threads1",
        "decoder": f"Pillow-{pillow_version}",
        "display_models_sha256": digest(data / "display_models.json"),
        "color_spaces_sha256": digest(data / "color_spaces.json")
    }


def _agreement_sample(index: int, a: Source, b: Source, score: float,
                      display: str) -> dict[str, object]:
    """Describe one pair using the decoded samples consumed by both arms."""
    parts = Path(b.source).parts
    family = parts[parts.index("compressed") +
                   2] if "compressed" in parts[:-2] else "custom"
    return {
        "sample_id": f"pair-{index:06d}",
        "source_id": a.rgb_sha256,
        "configuration_id": display,
        "stratum": "rgb8",
        "distortion_family": family,
        "input_hash": _json_digest([a.rgb_sha256, b.rgb_sha256]),
        "score": score
    }


def _agreement_reports(
    rows: list[tuple[Source, Source]], original: list[dict[str, Any]],
    candidate: list[float], args: argparse.Namespace
) -> tuple[dict[str, object], dict[str, object]]:
    """Build canonical arms with a common real configuration and decoded-input manifest."""
    reference_rows = [
        _agreement_sample(index, a, b, float(score["score"]), args.display)
        for index, ((a, b), score) in enumerate(zip(rows, original, strict=True))
    ]
    _attach_mos(reference_rows, rows, args)
    candidate_rows = [
        {
            **row, "score": score
        } for row, score in zip(reference_rows, candidate, strict=True)
    ]
    configuration = _agreement_configuration(args)
    inputs = [
        {
            key: value
            for key, value in row.items() if key != "score"
        } for row in reference_rows
    ]
    for name, value in (("configuration", configuration), ("inputs", inputs)):
        (args.out_dir / f"agreement-{name}.json").write_text(
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    metadata = {
        "metric": "cvvdp",
        "configuration_hash": _json_digest(configuration),
        "input_manifest_hash": _json_digest(inputs)
    }
    reference = report_from_rows(
        reference_rows, implementation="ColorVideoVDP", revision=PIN, **metadata
    )
    observed = report_from_rows(
        candidate_rows, implementation="vqm",
        revision=f"sha256:{digest(args.out_dir / 'candidate-binary')}", **metadata
    )
    return reference, observed


def _relative_pair(a: Source, b: Source, root: Path) -> tuple[str, str] | None:
    """Identify CID22 annotation keys only for sources inside the selected corpus."""
    original, distorted = Path(a.source), Path(b.source)
    if not original.is_relative_to(root) or not distorted.is_relative_to(root):
        return None
    return str(original.relative_to(root)), str(distorted.relative_to(root))


def _attach_mos(
    samples: list[dict[str, object]], rows: list[tuple[Source, Source]],
    args: argparse.Namespace
) -> None:
    """Retain available authoritative CID22 labels for explicitly requested bootstrap gates."""
    corpus = getattr(args, "corpus", None)
    if corpus is None:
        return
    annotations = Path(corpus) / "CID22_validation_set.csv"
    if not annotations.is_file():
        return
    labels = load_mcos(str(annotations))
    root = Path(corpus).resolve()
    for sample, (a, b) in zip(samples, rows, strict=True):
        key = _relative_pair(a, b, root)
        if key in labels:
            sample["mos"] = labels[key]


def write_agreement(
    rows: list[tuple[Source, Source]], original: list[dict[str, Any]],
    candidate: list[float] | None, args: argparse.Namespace, numerical_passed: bool
) -> dict[str, object]:
    """Write independent ranking evidence and resolve the requested gate selection.

    :param rows: Canonical paired decoded inputs.
    :param original: Complete pinned-original score records.
    :param candidate: Complete candidate scores; required for qualification.
    :param args: Frozen policy path, output location and requested gate mode.
    :param numerical_passed: Outcome of the existing absolute-JOD gate.
    :returns: Complete agreement diagnostics and resolved gate mode/outcome.
    :raises ValueError: If policy, pin, provenance or score evidence is invalid.
    :raises OSError: If policy or provenance files cannot be read or written.
    """
    if candidate is None:
        raise ValueError(
            "agreement qualification requires candidate and original scores"
        )
    policy = json.loads(args.agreement_policy.read_text(encoding="utf-8"))
    AgreementPolicy.parse(policy)
    reference, observed = _agreement_reports(rows, original, candidate, args)
    result = evaluate_agreement(reference, observed, policy)
    ranking_passed = result["proposed_status"] == "pass"
    numerical_gate = numerical_passed and result["numerical"]["status"] != "fail"
    mode = getattr(args, "gate_mode", None) or default_gate_mode(reference, policy)
    passed = (
        ranking_passed if mode == "ranking" else
        numerical_gate if mode == "numerical" else ranking_passed and numerical_gate
    )
    result.update(
        gate_mode=mode, legacy_numerical_passed=numerical_passed,
        numerical_gate_passed=numerical_gate, selected_gate_passed=passed
    )
    for name, value in (("original", reference), ("candidate", observed), ("agreement",
                                                                           result)):
        (args.out_dir / f"{name}.json"
         ).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(f"ranking agreement: {result['proposed_status'].upper()}")
    print(f"numerical parity: {'PASS' if numerical_gate else 'FAIL'}")
    print(f"selected gates ({mode}): {'PASS' if passed else 'FAIL'}")
    return result


def candidate_provenance(out_dir: Path) -> dict[str, Any]:
    """Identify the retained candidate executable and its selected device.

    :param out_dir: Qualification output directory.
    :returns: Binary path/hash and Vulkan device identity.
    """
    snapshot = out_dir / "candidate-binary"
    return {
        "measured_binary": str(snapshot.resolve()),
        "binary_sha256": digest(snapshot),
        "device": json.loads((out_dir / "device.json").read_text())
    }


def comparison_rows(
    rows: list[tuple[Source, Source]], original: list[dict[str, Any]],
    candidate: list[float] | None
) -> list[dict[str, Any]]:
    """Join score arms and decoded provenance without changing pair order.

    :param rows: Canonical input records.
    :param original: Original score records.
    :param candidate: Optional candidate scores.
    :returns: Serializable pair comparisons.
    """
    values = []
    for index, ((a, b), reference) in enumerate(zip(rows, original, strict=True)):
        row = {
            "original": dataclasses.asdict(a),
            "distorted": dataclasses.asdict(b),
            "reference": reference
        }
        if candidate is not None:
            row.update(
                candidate=candidate[index],
                absolute_error=abs(candidate[index] - reference["score"])
            )
        values.append(row)
    return values


def _agreement_options(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Reject inadmissible qualification options before loading or scoring inputs."""
    if args.gate_mode and args.agreement_policy is None:
        parser.error("--gate-mode requires --agreement-policy")
    if args.agreement_policy is None:
        return
    if args.reference_only:
        parser.error("--agreement-policy requires both original and candidate arms")
    try:
        settings = AgreementPolicy.parse(json.loads(args.agreement_policy.read_text()))
    except (OSError, ValueError) as error:
        parser.error(f"invalid agreement policy: {error}")
    if settings.metric != "cvvdp" or settings.reference_revision != PIN:
        parser.error("agreement policy metric or original pin does not match CVVDP")


def arguments() -> argparse.Namespace:
    """Parse validation inputs and fixed correctness limits.

    :returns: Validated command options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "validation/data/CID22_validation_set"
    )
    parser.add_argument("--pairs-file", type=Path)
    parser.add_argument(
        "--limit", type=int, default=300,
        help="uniform subset size; 0 selects all pairs"
    )
    parser.add_argument("--display", choices=tuple(DISPLAY_PPD), default="standard_4k")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/vqm_bench")
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "artifacts/reports/cvvdp/cid22"
    )
    parser.add_argument(
        "--rgb-cache", type=Path, default=ROOT / ".cache/validation/cvvdp-rgb8"
    )
    parser.add_argument(
        "--reference-cache", type=Path,
        default=ROOT / ".cache/validation/cvvdp-reference.jsonl"
    )
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument(
        "--agreement-policy", type=Path,
        help="explicit frozen policy for canonical ranking qualification"
    )
    parser.add_argument(
        "--gate-mode", choices=("both", "ranking", "numerical"), help=
        "requested agreement gates (default: ranking for qualified domains; both otherwise)"
    )
    args = parser.parse_args()
    if args.limit < 0 or not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("limit must be nonnegative and tolerance finite and positive")
    _agreement_options(args, parser)
    return args


def main() -> int:
    """Run one reproducible original/candidate validation workload.

    :returns: Zero for a passing comparison or completed reference-only run.
    """
    args = arguments()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/fetch_references.py"), "--check", "--only",
            "ColorVideoVDP"
        ], check=True
    )
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = prepare_pairs(selected_pairs(args), args.rgb_cache)
    original = original_scores(rows, args)
    candidate = None if args.reference_only else candidate_scores(rows, args)
    passed = write_report(rows, original, candidate, args)
    return 0 if passed or args.reference_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
