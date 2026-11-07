"""Run old numerical and new ranking gates on reproducible hosted video clips."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..agreement import evaluate_agreement, load_report, report_from_rows
from ..gate_defaults import default_gate_mode
from . import corpus
from .cli_results import validate_cli_result
from .corpus import MANIFEST, ROOT, SMOKE_SOURCES
from .fetch import DEFAULT_OUTPUT, fetch_sources, load_manifest, select_sources
from .reference import PINS, provenance, score_reference, verify_references
from .spatial_inputs import PROTOCOL as SPATIAL_PROTOCOL
from .spatial_inputs import SPATIAL_METRICS, ensure_spatial

METRICS = (*SPATIAL_METRICS, "vmaf-v0", "vmaf-v1", "cvvdp")
OLD_MAX = {
    "ssimulacra2": .01,
    "butteraugli": .01,
    "vmaf-v0": .01,
    "vmaf-v1": .35,
    "cvvdp": .001
}
ORACLES = Path(__file__).with_name("oracles")


def arguments() -> argparse.Namespace:
    """Parse reproducible input, reference and rollout options.

    :returns: Parsed command-line options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("smoke", "qualification"), default="smoke"
    )
    parser.add_argument("--metrics", default=",".join(METRICS))
    parser.add_argument(
        "--only", help="source IDs; partial qualification cannot pass full policy"
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="verify existing downloads without network"
    )
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / ".cache/validation/temporal"
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/vqm")
    parser.add_argument(
        "--reference", choices=("checked", "generate"), default="checked"
    )
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--oracle-dir", type=Path, default=ORACLES)
    parser.add_argument(
        "--gate-mode", choices=("both", "ranking", "numerical"), help=
        "gates to require (default: ranking for registered qualified domains, both otherwise)"
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    selected = args.metrics.split(",")
    if not set(selected) <= set(METRICS) or len(selected) != len(set(selected)):
        parser.error("--metrics must select unique names from " + ",".join(METRICS))
    if args.iterations < 1 or args.warmup < 0 or args.jobs < 1:
        parser.error("iterations/jobs must be positive and warmup nonnegative")
    if args.reference_only and args.reference != "generate":
        parser.error("--reference-only requires --reference generate")
    args.out_dir = args.out_dir or ROOT / "artifacts/reports" / f"temporal-{args.profile}"
    return args


def configuration_id(row: dict[str, object], metric: str) -> str:
    """Name a separately qualified model and timing configuration.

    :param row: Prepared clip metadata.
    :param metric: Temporal metric/model name.
    :returns: Stable configuration identifier.
    """
    if metric in SPATIAL_METRICS:
        return f"{metric}-srgb444-8bit-frame-mean"
    return f"cvvdp-standard4k-{row['fps']}" if metric == "cvvdp" else f"{metric}-native444-8bit"


def sample_metadata(row: dict[str, object], metric: str) -> dict[str, object]:
    """Identify the exact prepared representation consumed by both arms.

    :param row: Prepared clip and representation hashes.
    :param metric: Temporal metric/model name.
    :returns: Canonical sample fields excluding its score.
    """
    representation = SPATIAL_PROTOCOL if metric in SPATIAL_METRICS else "YUV444-8bit-BT601-limited-and-reference-RGB-FP32"
    prefix = "spatial_" if metric in SPATIAL_METRICS else ""
    try:
        original = row[prefix + "original_hashes"]
        distorted = row[prefix + "distorted_hashes"]
    except KeyError as error:
        raise ValueError("missing required spatial input hashes") from error
    identity = {
        "original": original,
        "distorted": distorted,
        "width": row["width"],
        "height": row["height"],
        "fps": row["fps"],
        "frames": row["frames"],
        "representation": representation
    }
    return {
        "sample_id": row["sample_id"],
        "source_id": row["source_id"],
        "stratum":
        "sdr-spatial-srgb444" if metric in SPATIAL_METRICS else row["stratum"],
        "distortion_family": row["distortion_family"],
        "configuration_id": configuration_id(row, metric),
        "input_hash": corpus.json_digest(identity)
    }


def report(
    rows: list[dict[str, object]], values: list[dict[str, object]], metric: str,
    implementation: str, revision: str
) -> dict[str, object]:
    """Build a portable report with content hashes independent of cache paths.

    :param rows: Prepared clip records.
    :param values: Scores in matching input order.
    :param metric: Temporal metric/model name.
    :param implementation: Original or candidate arm name.
    :param revision: Pinned original revision or measured binary fingerprint.
    :returns: Validated canonical agreement report.
    """
    metadata = [sample_metadata(row, metric) for row in rows]
    samples = [
        dict(item, score=value["score"])
        for item, value in zip(metadata, values, strict=True)
    ]
    settings = {
        "metric": metric,
        "protocol": 1,
        "display": "standard_4k",
        "vmaf_input": "native444-8bit",
        "rgb": "BT601-limited-nearest-FP32-sRGB",
        "configurations": sorted({str(item["configuration_id"])
                                  for item in metadata})
    }
    if metric in SPATIAL_METRICS:
        settings = _spatial_settings(metric, metadata)
    return report_from_rows(
        samples, metric=metric, implementation=implementation, revision=revision,
        configuration_hash=corpus.json_digest(settings),
        input_manifest_hash=corpus.json_digest(metadata)
    )


def _spatial_settings(metric: str, metadata: list[dict[str,
                                                       object]]) -> dict[str, object]:
    return {
        "metric": metric,
        "protocol": SPATIAL_PROTOCOL,
        "input_transfer": "srgb",
        "input_matrix": "bt470bg",
        "input_range": "limited",
        "pooling": "arithmetic-mean-final-frame-score",
        "configurations": sorted({str(item["configuration_id"])
                                  for item in metadata}),
        "half_planes": False,
        "hf_asymmetry": 1.0,
        "xmul": 1.0,
        "intensity_target": 80.0
    }


def frozen_policy(reference: dict[str, object], profile: str) -> dict[str, object]:
    """Freeze original-only ambiguity margins before observing candidate results.

    :param reference: Original report generated from the hosted corpus.
    :param profile: Smoke or full qualification coverage.
    :returns: Explicit policy with immutable input/configuration pins.
    """
    samples = reference["samples"]
    configurations = sorted({str(row["configuration_id"]) for row in samples})
    margins = {}
    for config in configurations:
        values = [
            float(row["score"]) for row in samples if row["configuration_id"] == config
        ]
        span = float(np.percentile(values, 95) - np.percentile(values, 5))
        margins[config] = max(1e-6, .01 * span)
    coverage = {
        "min_sources": 1,
        "min_samples": 2,
        "min_eligible_pairs": 1,
        "min_eligible_fraction": .1
    }
    result = {
        "schema_version": 1,
        "metric": reference["metric"],
        "reference_revision": reference["revision"],
        "margins": margins,
        "configuration_hash": reference["configuration_hash"],
        "input_manifest_hash": reference["input_manifest_hash"],
        "min_total_sources": 30 if profile == "qualification" else 3,
        "min_total_samples": 390 if profile == "qualification" else 39,
        "coverage": coverage,
        "required_strata": {
            str(row["stratum"]): coverage
            for row in samples
        },
        "numerical_max": OLD_MAX[str(reference["metric"])],
        "subjective": {
            "required": False
        }
    }
    if reference["metric"] in SPATIAL_METRICS:
        result["direction"
               ] = "lower" if reference["metric"] == "butteraugli" else "higher"
        result["numerical_mean"] = .001
    return result


def generated_original(
    rows: list[dict[str, object]], metric: str, args: argparse.Namespace
) -> tuple[dict[str, object], dict[str, object]]:
    """Generate independent original evidence and freeze its acceptance policy.

    :param rows: Prepared clip records.
    :param metric: Temporal metric/model name.
    :param args: Output and cache locations.
    :returns: Original canonical report and original-only frozen policy.
    """
    identity = provenance(metric)
    (args.out_dir / f"{metric}-original-provenance.json"
     ).write_text(json.dumps(identity, indent=2) + "\n")
    values = _original_values(rows, metric, args, identity)
    (args.out_dir /
     f"{metric}-original-details.json").write_text(json.dumps(values) + "\n")
    original = report(rows, values, metric, "pinned-original", PINS[metric])
    return original, frozen_policy(original, args.profile)


def _original_task(
    task: tuple[dict[str, object], str, Path, dict[str, object]]
) -> dict[str, object]:
    return score_reference(*task)


def _original_values(
    rows: list[dict[str, object]], metric: str, args: argparse.Namespace,
    identity: dict[str, object]
) -> list[dict[str, object]]:
    tasks = [(row, metric, args.cache_dir / "reference", identity) for row in rows]
    workers = min(4, getattr(args, "jobs", 1))
    if workers == 1:
        return [_original_task(task) for task in tasks]
    executor = (
        ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) if metric == "cvvdp" else ThreadPoolExecutor(max_workers=workers)
    )
    values = []
    with executor:
        for index, value in enumerate(executor.map(_original_task, tasks)):
            values.append(value)
            print(
                f"{metric} original {index + 1}/{len(rows)} {rows[index]['sample_id']}",
                flush=True
            )
    return values


def original_and_policy(
    rows: list[dict[str, object]], metric: str, args: argparse.Namespace
) -> tuple[dict[str, object], dict[str, object]]:
    """Load checked evidence or generate originals before running the candidate.

    :param rows: Prepared input records.
    :param metric: Temporal metric/model name.
    :param args: Reference mode and report paths.
    :returns: Canonical original and policy matching all prepared inputs.
    :raises ValueError: If checked evidence does not cover these exact inputs.
    """
    if args.reference == "generate":
        original, policy = generated_original(rows, metric, args)
    else:
        directory = args.oracle_dir / args.profile
        original = load_report(directory / f"{metric}.json")
        policy = json.loads((directory / f"{metric}-policy.json").read_text())
    if (original["metric"] != metric or policy.get("metric") != metric
            or original["revision"] != PINS[metric]
            or policy.get("reference_revision") != PINS[metric]):
        raise ValueError("original and policy must match the pinned reference revision")
    values = [{"score": 0} for _ in rows]
    expected = report(rows, values, metric, "input-verification", "prepared-inputs")
    if (expected["input_manifest_hash"] != original["input_manifest_hash"]
            or expected["configuration_hash"] != original["configuration_hash"]):
        raise ValueError("prepared clips do not match the frozen original corpus")
    for suffix, value in (("original", original), ("policy", policy)):
        (args.out_dir /
         f"{metric}-{suffix}.json").write_text(json.dumps(value, indent=2) + "\n")
    return original, policy


def candidate_scores(
    rows: list[dict[str, object]], metric: str, args: argparse.Namespace
) -> list[dict[str, object]]:
    """Score complete ordered clips in one candidate process.

    :param rows: Prepared input records.
    :param metric: Temporal metric/model name.
    :param args: Driver, repetition and output settings.
    :returns: Complete ordered candidate score/frame/timing records.
    :raises RuntimeError: If scoring fails, omits inputs or emits invalid scores.
    """
    pairs = args.out_dir / f"{metric}-pairs.tsv"
    pairs.write_text(
        "".join(
            f"{row['sample_id']}\t{_input_path(row, metric, 'original')}\t{_input_path(row, metric, 'distorted')}\n"
            for row in rows
        )
    )
    command = [
        str(args.binary.resolve()), "--metric", metric, "--json", "--pairs-file",
        str(pairs.resolve()), "--iterations",
        str(args.iterations), "--warmup",
        str(args.warmup)
    ]
    if metric in SPATIAL_METRICS:
        command.extend(("--input-transfer", "srgb"))
    elif metric == "cvvdp":
        # Match the frozen standard_4k reference at every input resolution.
        command.extend(("--cvvdp-ppd", "75.402449345163049"))
    started = time.perf_counter()
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=600 + 20 * len(rows),
        check=False
    )
    (args.out_dir / f"{metric}-candidate.log").write_text(
        json.dumps(command) + f"\nreturncode={result.returncode}\n" + result.stderr
    )
    (args.out_dir / f"{metric}-candidate.jsonl").write_text(result.stdout)
    values = [json.loads(line) for line in result.stdout.splitlines()]
    if result.returncode or len(values) != len(rows):
        raise RuntimeError(
            f"{metric}: candidate exit {result.returncode}, {len(values)}/{len(rows)} "
            "clip results; see candidate.log"
        )
    for row, value in zip(rows, values, strict=True):
        try:
            validate_cli_result(value, metric, row["sample_id"], row["frames"])
        except ValueError as error:
            raise RuntimeError(
                f"candidate returned mismatched or invalid clip results: {error}"
            ) from error
    timing = {
        "wall_seconds_including_warmup_decode_and_setup": time.perf_counter() - started,
        "timed_frames": sum(int(row["frames"]) for row in rows) * args.iterations,
        "warmup_frames": sum(int(row["frames"]) for row in rows) * args.warmup,
        "iterations_per_clip": args.iterations,
        "warmup_per_clip": args.warmup
    }
    (args.out_dir /
     f"{metric}-timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    return values


def _input_path(row: dict[str, object], metric: str, side: str) -> str:
    key = ("spatial_" if metric in SPATIAL_METRICS else "") + side
    if key not in row:
        raise ValueError("missing required spatial input path")
    return str(row[key])


def qualifies(result: dict[str, object], mode: str) -> bool:
    """Apply an explicit rollout mode without hiding either gate's result.

    :param result: Independent numerical and ranking statuses.
    :param mode: Both, ranking or numerical.
    :returns: Whether all requested gates passed.
    """
    numerical = result["numerical"]["status"] == "pass"
    ranking = result["proposed_status"] == "pass"
    return (mode == "ranking" or numerical) and (mode == "numerical" or ranking)


def run_metric(
    rows: list[dict[str, object]], metric: str, args: argparse.Namespace
) -> bool:
    """Record the paired old/new gate results for one complete temporal corpus.

    :param rows: Prepared input records.
    :param metric: Temporal metric/model name.
    :param args: Frozen reference, driver and rollout settings.
    :returns: Whether the requested acceptance mode passes.
    """
    original, policy = original_and_policy(rows, metric, args)
    if args.reference_only:
        return True
    _verify_metric_inputs(rows, metric)
    binary_hash = corpus.digest(args.binary)
    observed = report(
        rows, candidate_scores(rows, metric, args), metric, "vqm",
        "sha256:" + binary_hash
    )
    _verify_metric_inputs(rows, metric)
    if corpus.digest(args.binary) != binary_hash:
        raise ValueError("candidate binary changed during validation")
    (args.out_dir /
     f"{metric}-candidate.json").write_text(json.dumps(observed, indent=2) + "\n")
    result = evaluate_agreement(original, observed, policy)
    mode = args.gate_mode or default_gate_mode(original, policy)
    result.update(gate_mode=mode, selected_gate_passed=qualifies(result, mode))
    (args.out_dir / f"{metric}-agreement.json"
     ).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        f"{metric}: old numerical {result['numerical']['status']}; "
        f"new ranking {result['ranking']['status']}; subjective {result['subjective']['status']}",
        flush=True
    )
    return result["selected_gate_passed"]


def _verify_metric_inputs(rows: list[dict[str, object]], metric: str) -> None:
    selected = rows
    if metric in SPATIAL_METRICS:
        selected = [
            dict(
                row, original=row["spatial_original"],
                distorted=row["spatial_distorted"],
                original_hashes=row["spatial_original_hashes"],
                distorted_hashes=row["spatial_distorted_hashes"]
            ) for row in rows
        ]
    corpus.verify_inputs(selected)


def main() -> int:
    """Fetch, prepare and qualify the requested hosted temporal corpus.

    :returns: Zero when every requested gate passes, otherwise one.
    """
    args = arguments()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = args.only or (SMOKE_SOURCES if args.profile == "smoke" else None)
    fetch_sources(
        select_sources(load_manifest(MANIFEST), selected), args.download_dir, args.jobs,
        args.offline
    )
    metrics = args.metrics.split(",")
    rows = corpus.prepare(
        args.download_dir, args.cache_dir / args.profile, args.profile, args.only
    )
    if any(metric in SPATIAL_METRICS for metric in metrics):
        rows = ensure_spatial(rows)
    (args.out_dir / "inputs.json").write_text(json.dumps(rows, indent=2) + "\n")
    if args.reference == "generate":
        verify_references(metrics)
    passed = [run_metric(rows, metric, args) for metric in metrics]
    _summary(metrics, passed, args)
    return 0 if all(passed) else 1


def _summary(metrics: list[str], passed: list[bool], args: argparse.Namespace) -> None:
    results = {}
    if not args.reference_only:
        for metric in metrics:
            result = json.loads((args.out_dir / f"{metric}-agreement.json").read_text())
            results[metric] = {
                "gate_mode": result["gate_mode"],
                "selected_gate_passed": result["selected_gate_passed"],
                "ranking": result["ranking"]["status"],
                "subjective": result["subjective"]["status"],
                "numerical": result["numerical"]["status"],
                "numerical_diagnostics": result["numerical"],
                "srcc": result["ranking"]["global"]["srcc"],
                "tau_b": result["ranking"]["global"]["tau_b"],
                "pearson": result["numerical"]["pearson"],
                "samples": result["sample_count"],
                "policy_sha256": result["policy_sha256"]
            }
    summary = {
        "schema_version": 1,
        "profile": args.profile,
        "gate_mode": _summary_mode(results, args.gate_mode),
        "requested_gate_mode": args.gate_mode,
        "reference_only": args.reference_only,
        "selected_metrics": metrics,
        "selected_gate_passed": all(passed),
        "results": results
    }
    (args.out_dir /
     "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    _print_summary(results)


def _summary_mode(results: dict[str, dict[str, object]], requested: str | None) -> str:
    modes = {str(result["gate_mode"]) for result in results.values()}
    return next(iter(modes)) if len(modes) == 1 else requested or "domain-default"


def _print_summary(results: dict[str, dict[str, object]]) -> None:
    if not results:
        return
    print("\nmetric           SRCC       tau-b      Pearson    ranking  numerical")
    for metric, result in results.items():
        print(
            f"{metric:<16} {result['srcc']:.7f}  {result['tau_b']:.7f}  "
            f"{result['pearson']:.7f}  {result['ranking']:<8} {result['numerical']}",
            flush=True
        )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, ImportError,
            subprocess.TimeoutExpired) as error:
        print(f"temporal validation: {error}", file=sys.stderr)
        sys.exit(1)
