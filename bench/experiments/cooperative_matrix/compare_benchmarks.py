#!/usr/bin/env python3
"""Interleave two vqm_bench binaries and retain every score, checksum and stage.

Example: compare_benchmarks.py half8 BASE CANDIDATE --pair 1080 REF DIST
All timing comes from the repository benchmark, never from this driver.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def digest(path: Path) -> str:
    """Hash a benchmark input or executable.

    :param path: File to read, hash or update.
    :returns: Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(
    binary: Path, metric: str, ref: Path, dist: Path, stem: Path, streaming: bool,
    chroma: str | None = None
) -> dict[str, object]:
    """Run one benchmark arm and collect its stage timings.

    :param binary: Benchmark executable.
    :param metric: Metric selector for the benchmark.
    :param ref: Reference image path.
    :param dist: Distorted image path.
    :param stem: Output path prefix for the command log and stage CSV.
    :param streaming: Whether to measure host streaming instead of resident inputs.
    :param chroma: VMAF chroma sampling override.
    :returns: Invocation, score, checksum and timing fields.
    :raises RuntimeError: If the benchmark exits unsuccessfully.
    :raises ValueError: If scores or stage timings are missing or invalid.
    """
    stage_file = Path(str(stem) + ".csv")
    command = [
        str(binary), "--metric", metric, "--pair",
        str(ref),
        str(dist), "--iterations", "8" if streaming else "20", "--warmup",
        "2" if streaming else "3", "--stages-csv",
        str(stage_file)
    ]
    if not streaming:
        command += ["--resident", "2"]
    if chroma is not None:
        command += ["--vmaf-chroma", chroma]
    run = subprocess.run(command, capture_output=True, text=True)
    Path(str(stem) + ".log").write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError(f"benchmark failed ({run.returncode}); see {stem}.log")
    score = re.search(r", score ([-+\d.eE]+)", run.stdout)
    checksum = re.search(r"checksum ([-+\d.eE]+)", run.stdout)
    core = re.search(r"core ([-+\d.eE]+) ms", run.stdout)
    if not all((score, checksum, core)):
        raise ValueError(f"missing benchmark result in {stem}.log")
    with stage_file.open() as file:
        stages = {
            row["stage"]: float(row["ms_per_iter"])
            for row in csv.DictReader(file)
        }
    values = [float(score[1]), float(checksum[1]), float(core[1]), *stages.values()]
    if not all(math.isfinite(value) for value in values) or not stages:
        raise ValueError(f"invalid score/timing in {stem}.log")
    return {
        "command": command,
        "score": score[1],
        "checksum": checksum[1],
        "core_ms": float(core[1]),
        "stages": stages
    }


def summarize(
    runs: list[dict[str, object]]
) -> dict[str, dict[str, float | dict[str, float]]]:
    """Summarize each stage across both benchmark arms.

    :param runs: Measurements collected in interleaved arm order.
    :returns: Per-arm statistics and percent change by stage.
    """
    summary = {}
    for stage in ["core_ms", *runs[0]["stages"]]:
        stats = {}
        for arm in ("A", "B"):
            values = [
                run["core_ms"] if stage == "core_ms" else run["stages"][stage]
                for run in runs if run["arm"] == arm
            ]
            stats[arm] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values)
            }
        stats["percent_change"
              ] = 100 * (stats["B"]["median"] / stats["A"]["median"] - 1)
        summary[stage] = stats
    return summary


def _measure_pair(
    args: argparse.Namespace, output: Path, binaries: dict[str, Path], name: str,
    ref: Path, dist: Path
) -> list[dict[str, object]]:
    """Run both arms five times in alternating order.

    :param args: Metric, streaming and chroma options.
    :param output: Directory for the run logs.
    :param binaries: Executable paths by arm label.
    :param name: Case label used in output filenames.
    :param ref: Reference image path.
    :param dist: Distorted image path.
    :returns: Runs in round-major, A/B order.
    """
    runs = []
    for round_index in range(1, 6):
        for arm, binary in binaries.items():
            stem = output / f"{name}-{arm}-{round_index}"
            run = measure(
                binary, args.metric, ref, dist, stem, args.streaming, args.vmaf_chroma
            )
            run.update({"round": round_index, "arm": arm})
            runs.append(run)
    return runs


def main() -> None:
    """Run alternating benchmark arms and write the paired report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--pair", nargs=3, action="append", required=True,
        metavar=("NAME", "REFERENCE", "DISTORTED")
    )
    parser.add_argument("--metric", choices=("vmaf-v0", "vmaf-v1"), default="vmaf-v1")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--vmaf-chroma", choices=("native", "420", "444"))
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    output = (
        args.out_dir or ROOT / "artifacts/reports/vmaf-cooperative-matrix" / args.label
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    binaries = {"A": args.baseline.resolve(), "B": args.candidate.resolve()}
    result = {
        "label": args.label,
        "metric": args.metric,
        "streaming": args.streaming,
        "vmaf_chroma": args.vmaf_chroma,
        "protocol": {
            "rounds": 5,
            "order": "A,B",
            "resident": 0 if args.streaming else 2,
            "iterations": 8 if args.streaming else 20,
            "warmup": 2 if args.streaming else 3
        },
        "binaries": {
            arm: {
                "path": str(path),
                "sha256": digest(path)
            }
            for arm, path in binaries.items()
        },
        "cases": []
    }
    for name, original, distorted in args.pair:
        ref, dist = Path(original).resolve(), Path(distorted).resolve()
        case = {
            "case": name,
            "inputs":
            [{
                "name": path.name,
                "sha256": digest(path)
            } for path in (ref, dist)],
            "runs": []
        }
        case["runs"] = _measure_pair(args, output, binaries, name, ref, dist)
        case["summary"] = summarize(case["runs"])
        case["scores_exact"] = all(
            a["score"] == b["score"] and a["checksum"] == b["checksum"]
            for a, b in zip(case["runs"][::2], case["runs"][1::2], strict=True)
        )
        result["cases"].append(case)
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            name, "scores_exact=", case["scores_exact"], {
                key: round(value["percent_change"], 2)
                for key, value in case["summary"].items()
            }, flush=True
        )


if __name__ == "__main__":
    main()
