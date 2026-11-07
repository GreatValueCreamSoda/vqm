#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Validate every implementation against CID22 MCOS and against each other.

The manifest in validation/references.json lists, per metric, the
implementations to run and how to invoke each. Every implementation is scored
from one process over the whole pair list (its in-memory harness in batch mode)
so a fast metric does not dissolve into process startup: one Vship handler is
created per process, our pipeline is reused across equal-geometry pairs, and
the CPU references decode and score in memory. The manifest still supports
per-pair commands; `batch` selects the one-line-per-pair contract.

Reports land under artifacts/reports/validation/: one CSV per metric (the raw
scores next to MCOS) and a summary with the cross-implementation parity table
and the MOS correlation table.

Pass criteria: every selected implementation must score every selected pair,
and our metric's correlation with MCOS must be within --accuracy-tolerance of
the metric's reference correlation. The cross-implementation parity line is
reported for information and only gates the exit status with --strict-parity.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NotRequired
from typing import TextIO
from typing import TypedDict

if __package__ in (None, ""):
    # Keep direct script invocation compatible with package imports.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tools.build_paths import relocate_build_path
from validation import common
from validation.agreement import default_gate_mode
from validation.agreement import evaluate_agreement
from validation.agreement import report_from_rows
from validation.agreement_policy import AgreementPolicy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "validation", "references.json")


class Implementation(TypedDict):
    """Describe one scoring command and its process/worker configuration."""

    name: str
    command: list[str]
    batch: NotRequired[bool]
    jobs: NotRequired[int]
    threads: NotRequired[int]
    decode_threads: NotRequired[int]
    score_line: NotRequired[str]


class MetricSpec(TypedDict):
    """Describe the original arm and available commands for a metric."""

    direction: str
    reference: str
    implementations: list[Implementation]


Scores = dict[str, list[float | None]]
Correlations = dict[str, tuple[float, float, float]]
Timing = tuple[str, float, int]
ParityRow = tuple[str, int, float | None, float | None, float | None,
                  float | None, int | None, int | None]


def parse_args() -> argparse.Namespace:
    """Parse the metric, corpus selection and validation gates.

    :returns: Validated command-line options.
    :raises SystemExit: If an option is invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default="validation/data/CID22_validation_set"
    )
    parser.add_argument(
        "--csv", default=None, help="MCOS annotations (default: "
        "<data-root>/CID22_validation_set.csv)"
    )
    parser.add_argument(
        "--metric", default="all",
        choices=("all", "ssimulacra2", "butteraugli")
    )
    parser.add_argument(
        "--only", default=None, help="comma-separated implementation names"
    )
    parser.add_argument(
        "--jobs", type=int, default=os.cpu_count() or 1,
        help="parallel invocations for non-batch entries"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="only the first N pairs (0 = all)"
    )
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--out-dir", default="artifacts/reports/validation")
    parser.add_argument("--max-threshold", type=float, default=0.01)
    parser.add_argument("--mean-threshold", type=float, default=0.001)
    parser.add_argument(
        "--accuracy-tolerance", type=float, default=0.01,
        help="allowed shortfall of each MCOS correlation "
        "coefficient (default 0.01, about one standard "
        "error at n=4292)"
    )
    parser.add_argument(
        "--strict-parity", action="store_true",
        help="also require the max/mean cross-implementation "
        "thresholds to pass"
    )
    parser.add_argument(
        "--agreement-policy", type=Path,
        help="explicit frozen policy for canonical ranking qualification"
    )
    parser.add_argument(
        "--gate-mode", choices=("both", "ranking", "numerical"), help=(
            "agreement gates to require (default: ranking for qualified "
            "domains; both otherwise)"
        )
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="seconds per invocation; a batch invocation is "
        "allowed --timeout plus 2 s per pair"
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    for name in ("max_threshold", "mean_threshold", "accuracy_tolerance"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(
                f"--{name.replace('_', '-')} must be finite and non-negative"
            )
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if args.gate_mode and args.agreement_policy is None:
        parser.error("--gate-mode requires --agreement-policy")
    return args


def load_manifest(build_dir: str = "build") -> dict[str, MetricSpec]:
    """Load the local manifest of supported metric commands.

    :param build_dir: Directory containing the selected candidate and
        references.
    :returns: Metric names and their command configurations.
    """
    with open(MANIFEST) as f:
        manifest = json.load(f)
    for spec in manifest.values():
        for entry in spec["implementations"]:
            entry["command"] = [
                relocate_build_path(arg, build_dir) for arg in entry["command"]
            ]
    return manifest


def _digest_file(path: Path) -> str:
    """Identify one actual input or executable without retaining its bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest_json(value: object) -> str:
    """Identify deterministic finite JSON configuration or input metadata."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _reference_revision(spec: MetricSpec) -> str:
    """Verify the selected original checkout against the maintained lock pin."""
    name = {
        "cloudinary": "ssimulacra2"
    }.get(spec["reference"], spec["reference"])
    rows = [
        line.split()
        for line in (Path(REPO_ROOT) /
                     "tools/references.lock").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    pins = {row[0]: row[2] for row in rows if len(row) == 3}
    if name not in pins:
        raise ValueError(f"no pinned original checkout for {name}")
    subprocess.run(
        [
            sys.executable,
            str(Path(REPO_ROOT) / "tools/fetch_references.py"), "--check",
            "--only", name
        ], capture_output=True, text=True, check=True
    )
    return pins[name]


def _agreement_inputs(
    metric: str, pairs: list[tuple[str, str]], mcos: dict[tuple[str, str],
                                                          float], data_root: str
) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Identify raw source files and preserve complete pair/MOS membership."""
    files = {
        str((Path(data_root) / name).resolve())
        for pair in pairs
        for name in pair
    }
    hashes = {name: _digest_file(Path(name)) for name in sorted(files)}
    rows = []
    for original, distorted in pairs:
        first = hashes[str((Path(data_root) / original).resolve())]
        second = hashes[str((Path(data_root) / distorted).resolve())]
        rows.append(
            {
                "sample_id": _digest_json([original, distorted]),
                "source_id": first,
                "configuration_id": f"{metric}:cid22",
                "stratum": "cid22",
                "distortion_family": Path(distorted).parent.name,
                "input_hash": _digest_json([first, second]),
                "mos": mcos[(original, distorted)]
            }
        )
    return rows, hashes


def _agreement_context(
    metric: str, spec: MetricSpec, entries: list[Implementation],
    pairs: list[tuple[str, str]], mcos: dict[tuple[str, str],
                                             float], args: argparse.Namespace
) -> dict[str, object]:
    """Freeze input, command, binary and reference provenance before scoring."""
    names = {entry["name"] for entry in entries}
    if "ours" not in names or spec["reference"] not in names or spec[
            "reference"] == "ours":
        raise ValueError(
            "agreement qualification requires ours and the pinned original arm"
        )
    policy = json.loads(Path(args.agreement_policy).read_text(encoding="utf-8"))
    settings = AgreementPolicy.parse(policy)
    revision = _reference_revision(spec)
    if settings.metric != metric or settings.reference_revision != revision:
        raise ValueError(
            "agreement policy metric or original pin does not match this run"
        )
    inputs, fingerprints = _agreement_inputs(
        metric, pairs, mcos, args.data_root
    )
    commands = {}
    for entry in entries:
        binary = str(Path(entry["command"][0]).resolve())
        fingerprints[binary] = _digest_file(Path(binary))
        commands[entry["name"]] = {
            "command": entry["command"],
            "binary_sha256": fingerprints[binary]
        }
    configuration = {
        "metric": metric,
        "direction": spec["direction"],
        "reference_revision": revision,
        "input": "native-source-files",
        "implementations": commands,
        "manifest": spec
    }
    return {
        "metric": metric,
        "reference": spec["reference"],
        "revision": revision,
        "policy": policy,
        "inputs": inputs,
        "configuration": configuration,
        "fingerprints": fingerprints
    }


def _agreement_report(
    context: dict[str, object], scores: dict[str, list[float | None]], name: str
) -> dict[str, object]:
    """Build a canonical arm and reject missing or nonfinite scores."""
    values = scores[name]
    if any(value is None or not math.isfinite(value) for value in values):
        raise ValueError(
            f"{name}: agreement requires every requested finite score"
        )
    rows = [
        {
            **row, "score": score
        } for row, score in zip(context["inputs"], values, strict=True)
    ]
    configuration = context["configuration"]
    binary = configuration["implementations"][name]["binary_sha256"]
    revision = context["revision"] if name == context["reference"
                                                      ] else f"sha256:{binary}"
    return report_from_rows(
        rows, metric=context["metric"], implementation=name, revision=revision,
        configuration_hash=_digest_json(configuration),
        input_manifest_hash=_digest_json(context["inputs"])
    )


def _write_agreement(
    context: dict[str, object], scores: dict[str, list[float | None]],
    args: argparse.Namespace, numerical_passed: bool
) -> bool:
    """Retain separate gate results and apply the requested conjunction."""
    if any(_digest_file(Path(path)) != expected
           for path, expected in context["fingerprints"].items()):
        raise ValueError(
            "source inputs or measured executables changed during scoring"
        )
    original = _agreement_report(context, scores, context["reference"])
    candidate = _agreement_report(context, scores, "ours")
    result = evaluate_agreement(original, candidate, context["policy"])
    mode = getattr(args, "gate_mode",
                   None) or default_gate_mode(original, context["policy"])
    ranking_passed = result["proposed_status"] == "pass"
    numerical_gate = (
        numerical_passed and result["numerical"]["status"] != "fail"
    )
    passed = (
        ranking_passed if mode == "ranking" else numerical_gate
        if mode == "numerical" else ranking_passed and numerical_gate
    )
    result.update(
        gate_mode=mode, legacy_numerical_passed=numerical_passed,
        numerical_gate_passed=numerical_gate, selected_gate_passed=passed
    )
    records = {
        "original": original,
        "candidate": candidate,
        "agreement": result,
        "agreement-inputs": context["inputs"],
        "agreement-configuration": context["configuration"]
    }
    for name, record in records.items():
        path = Path(args.out_dir) / f"{context['metric']}-{name}.json"
        path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f"ranking agreement: {result['proposed_status'].upper()}")
    print(f"numerical parity: {'PASS' if numerical_gate else 'FAIL'}")
    print(f"selected gates ({mode}): {'PASS' if passed else 'FAIL'}")
    return passed


def _prepare_agreement(
    metric: str, spec: MetricSpec, entries: list[Implementation],
    pairs: list[tuple[str, str]], mcos: dict[tuple[str, str],
                                             float], args: argparse.Namespace
) -> dict[str, object] | bool | None:
    """Validate agreement options and distinguish rejection from no policy."""
    if getattr(args, "agreement_policy", None) is None:
        return None
    try:
        return _agreement_context(metric, spec, entries, pairs, mcos, args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"agreement qualification: FAIL ({error})", file=sys.stderr)
        return False


def _finish_metric(
    agreement: dict[str, object] | None, scores: dict[str, list[float | None]],
    args: argparse.Namespace, errors: int, accuracy_ok: bool, parity_ok: bool
) -> bool:
    """Preserve legacy acceptance unless an agreement policy is supplied."""
    if agreement is None:
        return errors == 0 and accuracy_ok and (
            parity_ok or not args.strict_parity
        )
    try:
        return _write_agreement(
            agreement, scores, args, errors == 0 and accuracy_ok and parity_ok
        )
    except (OSError, ValueError) as error:
        print(f"agreement qualification: FAIL ({error})", file=sys.stderr)
        return False


def _available_binaries(entries: list[Implementation]) -> bool:
    """Reject missing executables before starting any scoring process."""
    for entry in entries:
        binary = entry["command"][0]
        if not os.path.exists(binary):
            print(
                f"missing {binary} (build it first: "
                f"python3 tools/build_references.py --only {entry['name']}, "
                f"or cmake --build --preset release for ours)", file=sys.stderr
            )
            return False
    return True


def implementation_names(manifest: dict[str, MetricSpec],
                         metrics: list[str]) -> list[str]:
    """List unique implementations in manifest order by selected metric.

    :param manifest: All metric command specifications.
    :param metrics: Metric names to search.
    :returns: Implementation names, in first occurrence order.
    """
    return list(
        dict.fromkeys(
            entry["name"] for metric in metrics
            for entry in manifest[metric]["implementations"]
        )
    )


def _parse_batch_line(line: str) -> common.ScoreResult:
    if line == "ERROR":
        return None, "the driver reported ERROR"
    try:
        return common.parse_score(line), None
    except ValueError:
        return None, f"unparsable line {line[:80]!r}"


def _score_batch_command(
    argv: list[str], count: int, budget: float, log_path: str
) -> list[common.ScoreResult]:
    command = " ".join(argv) + "\n"
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=budget
        )
    except subprocess.TimeoutExpired:
        message = f"timeout after {budget:.0f}s"
        Path(log_path).write_text(command + message + "\n")
        return [(None, message)] * count
    except OSError as error:
        Path(log_path).write_text(command + f"{error}\n")
        return [(None, f"could not run {argv[0]}: {error}")] * count
    Path(log_path).write_text(command + result.stderr)
    if result.returncode != 0:
        return [
            (None, f"exit {result.returncode}: {result.stderr.strip()[:200]}")
        ] * count
    lines = [
        line.strip() for line in result.stdout.splitlines() if line.strip()
    ]
    if len(lines) != count:
        return [
            (None, f"expected {count} score lines, got {len(lines)}")
        ] * count
    return [_parse_batch_line(line) for line in lines]


def run_batch_chunk(
    entry: Implementation, pairs: list[common.Pair], data_root: str,
    timeout: float, log_path: str
) -> list[common.ScoreResult]:
    """Run one batch command over a contiguous slice of pairs.

    :param entry: Driver command and worker settings.
    :param pairs: Relative pair paths in scoring order.
    :param data_root: Extracted corpus directory.
    :param timeout: Fixed portion of the command timeout, in seconds.
    :param log_path: Destination for the command and its diagnostics.
    :returns: A score or error for each pair in input order.
    """
    absolute = [
        (os.path.join(data_root, o), os.path.join(data_root, d))
        for o, d in pairs
    ]
    directory = os.path.dirname(log_path)
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".pairs", delete=False,
                                     dir=directory) as handle:
        for original, distorted in absolute:
            handle.write(f"{original}\t{distorted}\n")
        pairs_path = handle.name
    try:
        argv = [
            part.replace("{pairs_file}", pairs_path)
            for part in entry["command"]
        ]
        # The driver parallelizes within one process, avoiding per-pair setup.
        argv += [
            "--iterations", "1", "--warmup", "0", "--threads",
            str(entry.get("threads") or 1), "--decode-threads",
            str(entry.get("decode_threads") or 4)
        ]
        return _score_batch_command(
            argv, len(pairs), timeout + 2.0 * len(pairs), log_path
        )
    finally:
        os.unlink(pairs_path)


def batch_scores(
    entry: Implementation, pairs: list[common.Pair], data_root: str,
    timeout: float, log_path: str, jobs: int
) -> list[common.ScoreResult]:
    """Score pairs in one batch process or in contiguous parallel slices.

    A batch harness takes `--threads`, so a corpus run is one process whose
    workers cover the machine: slices used to be the only way to parallelize,
    and they paid a device setup and a decode pool per slice. `jobs` therefore
    stays 1 for a batch entry unless the manifest asks otherwise; the slice
    path is kept for a single-threaded harness and for `--jobs` on entries that
    do not set `threads`.

    :param entry: Driver command and worker settings.
    :param pairs: Relative pair paths in scoring order.
    :param data_root: Extracted corpus directory.
    :param timeout: Fixed portion of each batch timeout, in seconds.
    :param log_path: Base path for command logs.
    :param jobs: Maximum number of concurrent batch processes.
    :returns: A score or error for each pair in input order.
    """
    jobs = max(1, min(jobs, len(pairs)))
    if jobs == 1:
        return run_batch_chunk(entry, pairs, data_root, timeout, log_path)
    bounds = []
    start = 0
    for index in range(jobs):
        stop = start + (len(pairs) - start) // (jobs - index)
        bounds.append((start, stop))
        start = stop
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(
                run_batch_chunk, entry, pairs[start:stop], data_root, timeout,
                f"{log_path}.{number}"
            ) for number, (start, stop) in enumerate(bounds)
        ]
        return [item for future in futures for item in future.result()]


def per_pair_scores(
    entry: Implementation, pairs: list[common.Pair], data_root: str, jobs: int,
    timeout: float
) -> list[common.ScoreResult]:
    """Run one command per pair and preserve input order.

    :param entry: Driver command and score line selection.
    :param pairs: Relative pair paths in scoring order.
    :param data_root: Extracted corpus directory.
    :param jobs: Maximum number of concurrent commands.
    :param timeout: Per-command timeout in seconds.
    :returns: A score or error for each pair in input order.
    """
    command = entry["command"]
    score_line = entry.get("score_line", "last")

    def one(index: int) -> tuple[int, float | None, str | None]:
        original, distorted = pairs[index]
        value, error = common.run_score(
            command, os.path.join(data_root, original),
            os.path.join(data_root, distorted), timeout=timeout,
            score_line=score_line
        )
        return index, value, error

    results = [(None, "not run")] * len(pairs)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for index, value, error in pool.map(one, range(len(pairs))):
            results[index] = (value, error)
    return results


def parity_table(scores: Scores, names: list[str],
                 reference: str) -> list[ParityRow]:
    """Summarize absolute score differences on pairs scored by both arms.

    :param scores: Score vectors indexed by implementation name.
    :param names: Implementations in report order.
    :param reference: Reference implementation name.
    :returns: One diagnostic row per non-reference implementation.
    """
    if reference not in scores:
        return []
    rows = []
    for name in names:
        if name == reference:
            continue
        diffs = [
            abs(scores[name][i] - scores[reference][i])
            for i in range(len(scores[name]))
            if scores[name][i] is not None and scores[reference][i] is not None
        ]
        if not diffs:
            rows.append((name, 0, None, None, None, None, None, None))
            continue
        diffs_array = np.asarray(diffs, dtype=np.float64)
        rows.append(
            (
                name, len(diffs), float(diffs_array.max()),
                float(diffs_array.mean()),
                float(np.percentile(diffs_array, 99)),
                float(math.sqrt(float(np.mean(diffs_array**2)))),
                int((diffs_array
                     > 0.001).sum()), int((diffs_array > 0.01).sum())
            )
        )
    return rows


def _write_timing_summary(f: TextIO, timings: list[Timing]) -> None:
    f.write("## Scoring time\n\n")
    f.write("| implementation | pairs | seconds | pairs/s |\n")
    f.write("| --- | --- | --- | --- |\n")
    for name, seconds, count in timings:
        rate = f"{count / seconds:.2f}" if seconds > 0 else "n/a"
        f.write(f"| {name} | {count} | {seconds:.2f} | {rate} |\n")
    f.write(
        "\nWall clock covers process startup, decode and scoring; "
        "pair counts include successfully scored pairs only.\n\n"
    )


def _write_parity_summary(
    f: TextIO, reference: str, parity_rows: list[ParityRow]
) -> None:
    f.write(f"## Parity vs {reference}\n\n")
    if not parity_rows:
        f.write("No reference comparison was selected.\n\n")
    f.write(
        "| implementation | pairs | max |diff| | mean |diff| | "
        "p99 |diff| | RMSE | >0.001 | >0.01 |\n"
    )
    f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for row in parity_rows:
        name, count, maximum, mean, p99, rmse, over1, over10 = row
        if count == 0:
            f.write(f"| {name} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |\n")
            continue
        f.write(
            f"| {name} | {count} | {maximum:.6f} | {mean:.6f} | "
            f"{p99:.6f} | {rmse:.6f} | {over1} | {over10} |\n"
        )


def write_summary(
    path: str, metric: str, spec: MetricSpec, pair_count: int, mcos_path: str,
    command: str, parity_rows: list[ParityRow], correlations: Correlations,
    timings: list[Timing]
) -> None:
    """Write the run's timing, reference parity and MCOS report.

    :param path: Destination Markdown file.
    :param metric: Metric name.
    :param spec: Metric direction and reference configuration.
    :param pair_count: Number of pairs with at least one score.
    :param mcos_path: Annotation file used for correlations.
    :param command: Command line that generated the report.
    :param parity_rows: Per-implementation reference comparison statistics.
    :param correlations: Per-implementation rank and linear correlations.
    :param timings: Elapsed time and successful pair count per implementation.
    """
    with open(path, "w") as f:
        f.write(f"# {metric} validation\n\n")
        f.write(
            f"generated: {datetime.datetime.now(datetime.UTC).isoformat()}\n"
        )
        f.write(f"pairs: {pair_count}\n")
        f.write(f"MCOS: {mcos_path}\n")
        f.write(f"command: {command}\n\n")
        if timings:
            _write_timing_summary(f, timings)
        _write_parity_summary(f, spec["reference"], parity_rows)
        f.write("\n## MCOS correlation (KRCC, SRCC, PCC)\n\n")
        f.write("| implementation | KRCC | SRCC | PCC |\n")
        f.write("| --- | --- | --- | --- |\n")
        for name in correlations:
            krcc, srcc, pcc = correlations[name]
            f.write(f"| {name} | {krcc:.4f} | {srcc:.4f} | {pcc:.4f} |\n")
        if spec["direction"] == "lower":
            f.write(
                "\nButteraugli is lower-is-better; the scores are negated "
                "before correlating, so a larger value always means "
                "better quality.\n"
            )


def _score_implementation(
    metric: str, entry: Implementation, pairs: list[common.Pair],
    args: argparse.Namespace
) -> tuple[list[common.ScoreResult], Timing]:
    name = entry["name"]
    if entry.get("batch"):
        # The harness spreads work over its own threads within one process.
        threads = int(entry.get("threads") or 1)
        print(f"  {name}: {threads} in-process thread(s), {len(pairs)} pairs")
        started = time.perf_counter()
        values = batch_scores(
            entry, pairs, args.data_root, args.timeout,
            os.path.join(args.out_dir, f"{metric}-{name}.log"), 1
        )
    else:
        jobs = entry.get("jobs") or args.jobs
        print(f"  {name}: {jobs} job(s), {len(pairs)} pairs")
        started = time.perf_counter()
        values = per_pair_scores(
            entry, pairs, args.data_root, jobs, args.timeout
        )
    elapsed = time.perf_counter() - started
    return values, (name, elapsed, sum(error is None for _, error in values))


def _record_scores(
    name: str, pairs: list[common.Pair], values: list[common.ScoreResult]
) -> tuple[list[float | None], int]:
    scores = [None] * len(pairs)
    errors = 0
    for index, (value, error) in enumerate(values):
        if error is not None:
            errors += 1
            original, distorted = pairs[index]
            print(
                f"error {name} {original} {distorted}: {error}", file=sys.stderr
            )
        else:
            scores[index] = value
    return scores, errors


def _score_pairs(
    metric: str, entries: list[Implementation], pairs: list[common.Pair],
    args: argparse.Namespace
) -> tuple[Scores, list[Timing], int]:
    names = [entry["name"] for entry in entries]
    print(f"\n== {metric} ({len(pairs)} pairs): {', '.join(names)}")
    scores = {name: [None] * len(pairs) for name in names}
    timings = []
    errors = 0
    for entry in entries:
        values, timing = _score_implementation(metric, entry, pairs, args)
        timings.append(timing)
        name = entry["name"]
        scores[name], count = _record_scores(name, pairs, values)
        errors += count
    return scores, timings, errors


def _write_scores(
    path: str, pairs: list[common.Pair], mcos: dict[common.Pair, float],
    scores: Scores
) -> int:
    names = list(scores)
    rows_written = 0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original", "distorted", "MCOS"] + names)
        for original, distorted, row_scores in _scored_pairs(pairs, scores):
            writer.writerow(
                [original, distorted, f"{mcos[(original, distorted)]:.8f}"] + [
                    "" if value is None else f"{value:.8f}"
                    for value in row_scores
                ]
            )
            rows_written += 1
    return rows_written


def _scored_pairs(
    pairs: list[common.Pair], scores: Scores
) -> Iterator[tuple[str, str, list[float | None]]]:
    for index, (original, distorted) in enumerate(pairs):
        row_scores = [values[index] for values in scores.values()]
        if any(value is not None for value in row_scores):
            yield original, distorted, row_scores


def _print_timings(timings: list[Timing]) -> None:
    if timings:
        print("\nscoring time (wall clock, successful pairs)")
        print(
            f"  {'implementation':<20} {'pairs':>6} {'seconds':>9} "
            f"{'pairs/s':>9}"
        )
        for name, seconds, count in timings:
            rate = f"{count / seconds:.2f}" if seconds > 0 else "n/a"
            print(f"  {name:<20} {count:>6} {seconds:>9.2f} {rate:>9}")


def _print_parity(
    parity_rows: list[ParityRow], parity_available: bool,
    args: argparse.Namespace
) -> bool:
    parity_ok = parity_available or not args.strict_parity
    print(
        f"  {'implementation':<20} {'pairs':>5} {'max |diff|':>10} "
        f"{'mean |diff|':>11} {'p99 |diff|':>10} {'RMSE':>9} "
        f"{'>0.001':>7} {'>0.01':>6}"
    )
    for row in parity_rows:
        name, count, maximum, mean, p99, rmse, over1, over10 = row
        if count == 0:
            continue
        print(
            f"  {name:<20} {count:>5} {maximum:>10.6f} {mean:>11.6f} "
            f"{p99:>10.6f} {rmse:>9.6f} {over1:>7} {over10:>6}"
        )
        if maximum > args.max_threshold or mean > args.mean_threshold:
            parity_ok = False
    if not parity_available:
        print(
            "parity vs reference: SKIP (reference or comparison not selected)"
        )
    else:
        parity_text = "PASS" if parity_ok else "FAIL"
        print(
            f"parity vs reference (max {args.max_threshold}, mean "
            f"{args.mean_threshold}): {parity_text}"
        )

    return parity_ok


def _correlate_scores(
    scores: Scores, mcos: dict[common.Pair, float], pairs: list[common.Pair],
    direction: str
) -> Correlations:
    names = list(scores)
    correlations = {}
    for name in names:
        paired = [
            (scores[name][i], mcos[pairs[i]]) for i in range(len(pairs))
            if scores[name][i] is not None
        ]
        if len(paired) < 2:
            continue
        values = [value for value, _ in paired]
        if direction == "lower":
            values = [-value for value in values]
        correlations[name] = common.correlation_triple(
            values, [m for _, m in paired]
        )

    return correlations


def _print_correlations(
    correlations: Correlations, names: list[str], reference: str
) -> None:
    print("\nMCOS correlation  (KRCC, SRCC, PCC)")
    order = [reference, "ours"
             ] + [name for name in names if name not in (reference, "ours")]
    for label in order:
        if label not in correlations:
            continue
        krcc, srcc, pcc = correlations[label]
        print(f"  {label:<20}{krcc:.4f}  {srcc:.4f}  {pcc:.4f}")


def _check_accuracy(
    correlations: Correlations, names: list[str], reference: str,
    args: argparse.Namespace
) -> bool:
    if "ours" not in correlations:
        reason = (
            "ours not selected"
            if "ours" not in names else "fewer than two scored pairs for ours"
        )
        print(f"accuracy vs MCOS: SKIP ({reason})")
        accuracy_ok = True
    elif reference not in correlations:
        reason = (
            "not selected"
            if reference not in names else "has fewer than two scored pairs"
        )
        print(f"accuracy vs MCOS: SKIP (reference {reference} {reason})")
        accuracy_ok = True
    else:
        accuracy_ok = all(
            correlations["ours"][i] >= correlations[reference][i] -
            args.accuracy_tolerance for i in range(3)
        )
        print(
            f"accuracy vs MCOS: {'PASS' if accuracy_ok else 'FAIL'}"
            f" (tolerance {args.accuracy_tolerance})"
        )

    return accuracy_ok


def run_metric(
    metric: str, spec: MetricSpec, pairs: list[common.Pair],
    mcos: dict[common.Pair,
               float], args: argparse.Namespace, only: set[str] | None
) -> bool:
    """Score selected implementations and evaluate the requested gates.

    :param metric: Metric name used for reports.
    :param spec: Implementation, reference and score direction settings.
    :param pairs: Relative corpus paths in scoring order.
    :param mcos: Subjective score for each selected pair.
    :param args: Runner paths, scoring settings and thresholds.
    :param only: Optional set of implementation names to run.
    :returns: Whether the requested metric checks passed.
    """
    entries = [
        entry for entry in spec["implementations"]
        if only is None or entry["name"] in only
    ]
    if not entries:
        print(f"\n== {metric}: no selected implementation")
        return getattr(args, "agreement_policy", None) is None
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    agreement = _prepare_agreement(metric, spec, entries, pairs, mcos, args)
    if agreement is False:
        return False

    if not _available_binaries(entries):
        return False

    scores, timings, errors = _score_pairs(metric, entries, pairs, args)
    csv_path = os.path.join(out_dir, f"{metric}.csv")
    rows_written = _write_scores(csv_path, pairs, mcos, scores)
    print(f"\npairs: {rows_written} ({errors} errors)")
    if rows_written == 0:
        print("no pair could be scored", file=sys.stderr)
        return False
    reference = spec["reference"]
    names = list(scores)
    _print_timings(timings)
    parity_rows = parity_table(scores, names, reference)
    parity_ok = _print_parity(
        parity_rows, reference in scores and len(names) > 1, args
    )
    correlations = _correlate_scores(scores, mcos, pairs, spec["direction"])
    _print_correlations(correlations, names, reference)
    accuracy_ok = _check_accuracy(correlations, names, reference, args)
    command = shlex.join([sys.executable, *sys.argv])
    write_summary(
        os.path.join(out_dir,
                     f"{metric}-summary.md"), metric, spec, rows_written,
        args.csv or os.path.join(args.data_root, "CID22_validation_set.csv"),
        command, parity_rows, correlations, timings
    )

    return _finish_metric(
        agreement, scores, args, errors, accuracy_ok, parity_ok
    )


def main() -> int:
    """Validate every selected metric against the CID22 corpus.

    :returns: Zero if every selected metric passes, otherwise one.
    :raises SystemExit: If the selection or corpus is invalid.
    """
    args = parse_args()
    data_root = args.data_root
    mcos_path = args.csv or os.path.join(data_root, "CID22_validation_set.csv")
    manifest = load_manifest(args.build_dir)
    metrics = (
        ["ssimulacra2", "butteraugli"]
        if args.metric == "all" else [args.metric]
    )

    only = None
    if args.only:
        only = set(
            part.strip() for part in args.only.split(",") if part.strip()
        )
        known = implementation_names(manifest, metrics)
        unknown = sorted(only - set(known))
        if unknown:
            sys.exit(
                f"unknown implementation(s) {', '.join(unknown)}; "
                f"choose from: {', '.join(known)}"
            )

    if not os.path.isdir(data_root):
        sys.exit(f"missing {data_root}; run python3 validation/fetch_cid22.py")
    pairs = common.enumerate_pairs(data_root)
    if args.limit:
        pairs = pairs[:args.limit]
    mcos = common.load_mcos(mcos_path)
    missing = [pair for pair in pairs if pair not in mcos]
    if missing:
        sys.exit(f"{len(missing)} pairs missing MCOS (first: {missing[0]})")

    ok = True
    for metric in metrics:
        ok = run_metric(
            metric, manifest[metric], pairs, mcos, args, only
        ) and ok
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
