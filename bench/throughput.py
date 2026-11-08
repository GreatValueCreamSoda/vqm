"""Run and optionally publish the complete fixed-count GPU throughput matrix."""

import argparse
import csv
import dataclasses
import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from bench import benchmark
from bench import inmemory_environment as environment
from bench import throughput_report as reports
from bench import throughput_workload as workload
from bench.inmemory_report import JsonObject
from tools.build_paths import build_directory


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the complete comparison's reproducible workload settings.

    :param argv: Explicit options or the process command line.
    :returns: Validated settings with absolute repository paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--build-dir", type=build_directory, default="build")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--input-dir", type=Path, default=Path(".cache/bench/throughput-v1")
    )
    parser.add_argument("--references", default=",".join(workload.PEERS))
    parser.add_argument("--rounds", type=benchmark.positive)
    parser.add_argument("--warmup", type=benchmark.positive, default=3)
    parser.add_argument("--workers", type=benchmark.positive, default=3)
    parser.add_argument(
        "--jobs", type=benchmark.positive, default=min(os.cpu_count() or 1, 8)
    )
    parser.add_argument("--timeout", type=benchmark.positive, default=180)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--list", action="store_true",
        help="print commands without setup or execution"
    )
    parser.add_argument(
        "--update-readme", action="store_true", help=
        "replace managed README tables only after a complete validated full run"
    )
    args = parser.parse_args(argv)
    _normalize(args, parser)
    return args


def _normalize(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    args.references = list(dict.fromkeys(args.references.split(",")))
    if not args.references or set(args.references) - set(workload.PEERS):
        parser.error("--references must select pinned Vship/Vship-fast peers")
    args.references = [
        name for name in workload.PEERS if name in args.references
    ]
    args.sizes = ["640x360"
                  ] if args.profile == "smoke" else list(workload.SIZES)
    args.rounds = args.rounds or (1 if args.profile == "smoke" else 5)
    stamp = datetime.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    args.out_dir = args.out_dir or Path(f"artifacts/reports/throughput-{stamp}")
    args.out_dir = (workload.ROOT / args.out_dir).resolve()
    args.input_dir = (workload.ROOT / args.input_dir).resolve()
    if args.update_readme and (args.profile != "full" or args.rounds < 5
                               or args.warmup < 3 or args.workers != 3
                               or set(args.references) != set(workload.PEERS)):
        parser.error(
            "README publication needs the full matrix, all peers, at least "
            "five rounds/three warmups and three spatial workers"
        )


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _setup(args: argparse.Namespace) -> None:
    if not args.skip_build:
        benchmark._build_references(args.references, args.jobs, args.build_dir)
        return
    checkouts = sorted({benchmark.CHECKOUTS[name] for name in args.references})
    benchmark.run(
        [
            sys.executable, "tools/fetch_references.py", "--check", "--only",
            ",".join(checkouts)
        ]
    )


def _machine(arm: workload.Arm, stdout: str) -> tuple[float, float, float]:
    lines = [
        line for line in stdout.splitlines() if line.startswith("throughput ")
    ]
    if len(lines) != 1:
        raise ValueError("expected one throughput receipt")
    fields = dict(field.split("=", 1) for field in lines[0].split()[1:])
    if (int(fields["calls"]) != arm.count
            or int(fields["threads"]) != arm.workers
            or fields["size"] != arm.size):
        raise ValueError("timed count, workers or geometry mismatch")
    return float(fields["seconds"]), float(fields["fps"]
                                           ), float(fields["checksum"])


def _fixed(arm: workload.Arm, stdout: str,
           csv_path: Path) -> tuple[float, float, float]:
    match = re.search(
        r"fixed-count wall clock: ([\d.]+) ms/(pair|image|frame), "
        r"([\d.]+) (pairs|images|frames)/s", stdout
    )
    if match is None:
        raise ValueError("missing fixed-count wall interval")
    expected_unit = {
        "psnr-hvs": "pair",
        "cvvdp-image": "image",
        "cvvdp-video": "frame"
    }[arm.metric]
    if match[2] != expected_unit or match[4] != expected_unit + "s":
        raise ValueError("incorrect throughput units")
    with csv_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1 or f"{rows[0]['width']}x{rows[0]['height']}" != arm.size:
        raise ValueError("CSV geometry or row count mismatch")
    if arm.metric == "cvvdp-video" and int(rows[0]["frames"]) != arm.frames:
        raise ValueError("incomplete temporal clip")
    checksum = re.search(r"^checksum ([^\s]+)$", stdout, re.MULTILINE)
    if checksum is None:
        raise ValueError("missing checksum")
    return arm.count * float(match[1]) / 1000, float(match[3]
                                                     ), float(checksum[1])


def parse_measurement(
    arm: workload.Arm, stdout: str, csv_path: Path, warmup: int
) -> JsonObject:
    """Validate one driver's counts, units, geometry, checksum and wall rate.

    :param arm: Expected command and count contract.
    :param stdout: Successful process output.
    :param csv_path: Fixed-count driver's row report.
    :param warmup: Excluded warmups per worker or complete clip.
    :returns: Checked timing and score fields.
    :raises ValueError: If any accounting or finite-value check fails.
    """
    seconds, fps, checksum = _machine(arm, stdout) if arm.machine else _fixed(
        arm, stdout, csv_path
    )
    checksum_calls = arm.count
    if not arm.machine:
        checksum_calls = (
            arm.iterations +
            warmup if arm.metric == "cvvdp-video" else arm.count +
            warmup * arm.workers
        )
    score = checksum / checksum_calls
    if not all(math.isfinite(value)
               for value in (seconds, fps, checksum, score)) or min(seconds,
                                                                    fps) <= 0:
        raise ValueError("non-finite score or nonpositive duration/rate")
    if not math.isclose(seconds * fps, arm.count, rel_tol=0.001):
        raise ValueError(
            "reported rate disagrees with completed count/wall time"
        )
    return {
        "seconds": seconds,
        "fps": fps,
        "checksum": checksum,
        "checksum_calls": checksum_calls,
        "score": score,
        "count": arm.count
    }


def _timeout_text(value: str | bytes | None) -> str:
    return value.decode(errors="replace"
                        ) if isinstance(value, bytes) else value or ""


def measure(
    arm: workload.Arm, index: int, args: argparse.Namespace
) -> JsonObject:
    """Run one process, preserving every outcome before validating its data.

    :param arm: Driver and accounting contract.
    :param index: One-based measurement round.
    :param args: Output directory, warmup count and per-process timeout.
    :returns: Successful checked measurement with exact command and exit status.
    :raises RuntimeError: If execution, timeout or result validation fails.
    """
    prefix = args.out_dir / f"{arm.key}-r{index}"
    csv_path = prefix.with_suffix(".csv")
    command = [*arm.command, *([] if arm.machine else ["--csv", str(csv_path)])]
    row = {**dataclasses.asdict(arm), "round": index, "command": command}
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command, cwd=workload.ROOT, capture_output=True, text=True,
            timeout=args.timeout, check=False
        )
    except subprocess.TimeoutExpired as error:
        prefix.with_suffix(".stdout").write_text(_timeout_text(error.stdout))
        prefix.with_suffix(".stderr").write_text(_timeout_text(error.stderr))
        _write_json(
            prefix.with_suffix(".process.json"), {
                **row, "status": "timeout",
                "process_seconds": time.perf_counter() - started
            }
        )
        raise RuntimeError(f"benchmark timed out: {prefix}") from error
    row.update(
        status=result.returncode, process_seconds=time.perf_counter() - started
    )
    prefix.with_suffix(".stdout").write_text(result.stdout)
    prefix.with_suffix(".stderr").write_text(result.stderr)
    _write_json(prefix.with_suffix(".process.json"), row)
    if result.returncode != 0:
        raise RuntimeError(
            f"benchmark failed ({result.returncode}): {prefix}\n{result.stderr}"
        )
    try:
        row.update(parse_measurement(arm, result.stdout, csv_path, args.warmup))
    except (KeyError, OSError, ValueError) as error:
        raise RuntimeError(
            f"invalid benchmark receipt: {prefix}: {error}"
        ) from error
    _write_json(prefix.with_suffix(".process.json"), row)
    return row


def _run_matrix(arms: list[workload.Arm],
                args: argparse.Namespace) -> list[JsonObject]:
    rows = []
    for size in args.sizes:
        rows.extend(
            _run_geometry([arm for arm in arms if arm.size == size], args)
        )
        _write_json(args.out_dir / "measurements.json", rows)
    return rows


def _run_geometry(arms: list[workload.Arm],
                  args: argparse.Namespace) -> list[JsonObject]:
    rows = []
    for index in range(args.rounds):
        for arm in workload.round_order(arms, index):
            print(
                f"round {index + 1}/{args.rounds}: {arm.key}, "
                f"{arm.count} timed pairs", flush=True
            )
            rows.append(measure(arm, index + 1, args))
            _write_json(args.out_dir / f"{arm.size}-measurements.json", rows)
    return rows


def _fingerprints(provenance: JsonObject, build_dir: Path) -> dict[str, str]:
    paths = list(provenance["inputs"])
    paths.extend(item["path"] for item in provenance["executables"].values())
    paths.extend(
        item["runtime_library"]["path"]
        for item in provenance["references"].values()
    )
    paths.append(str(build_dir / "libvqm_core_timing.a"))
    hashes = {path: environment.file_sha256(path) for path in paths}
    if None in hashes.values():
        raise RuntimeError(
            "could not fingerprint every input, executable and linked library"
        )
    return {path: str(digest) for path, digest in hashes.items()}


def _verify_fingerprints(hashes: dict[str, str]) -> None:
    if any(environment.file_sha256(path) != digest
           for path, digest in hashes.items()):
        raise RuntimeError(
            "input or binary changed during the benchmark; refusing publication"
        )


def _snapshot_sources(args: argparse.Namespace) -> dict[str, str]:
    changed = subprocess.run(
        [
            "git", "ls-files", "--modified", "--others", "--exclude-standard",
            "-z"
        ], cwd=workload.ROOT, capture_output=True, text=True, check=True
    ).stdout
    changed += subprocess.run(
        ["git", "diff", "HEAD", "--name-only", "-z"], cwd=workload.ROOT,
        capture_output=True, text=True, check=True
    ).stdout
    paths = {workload.ROOT / name for name in changed.split("\0") if name}
    paths.update(args.build_dir.glob("**/CMakeCache.txt"))
    hashes = {}
    for path in sorted(paths):
        if not path.is_file():
            continue
        relative = (
            path.relative_to(workload.ROOT)
            if path.is_relative_to(workload.ROOT) else Path("external-build") /
            path.relative_to(args.build_dir)
        )
        destination = args.out_dir / "source-snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        hashes[str(path)] = str(environment.file_sha256(path))
    return hashes


def _execute(args: argparse.Namespace, report: JsonObject) -> None:
    _setup(args)
    pairs = workload.prepare_inputs(args)
    drivers = workload.entries(args.build_dir, args.references)
    devices = environment.probe_devices(drivers)
    report["environment"] = environment.capture_environment(
        pairs, drivers, devices, args.build_dir
    )
    report["environment"]["cpu"] = environment.command_output(["lscpu"])
    report["environment"]["ffmpeg"] = environment.command_output(
        ["ffmpeg", "-version"]
    )
    hashes = _fingerprints(report["environment"], args.build_dir)
    hashes.update(_snapshot_sources(args))
    report["fingerprints"] = hashes
    arms = workload.matrix(args, drivers)
    report["matrix"] = [dataclasses.asdict(arm) for arm in arms]
    _write_json(args.out_dir / "summary.json", report)
    rows = _run_matrix(arms, args)
    _verify_fingerprints(hashes)
    report["summary"] = reports.summarize(arms, rows, args.rounds)
    report.update(status="complete", measurements=rows)
    _write_json(args.out_dir / "summary.json", report)
    (args.out_dir /
     "summary.md").write_text(reports.detailed(report), encoding="utf-8")
    if args.update_readme:
        reports.publish(report, workload.ROOT)


def main(argv: Sequence[str] | None = None) -> int:
    """Provision, measure, validate and optionally publish the entire matrix.

    :param argv: Explicit arguments or the process command line.
    :returns: Zero after a complete validated comparison.
    :raises RuntimeError: If execution fails, without publishing partial data.
    :raises FileExistsError: If the output directory already contains evidence.
    """
    args = arguments(argv)
    if args.list:
        arms = workload.matrix(
            args, workload.entries(args.build_dir, args.references)
        )
        print(json.dumps([dataclasses.asdict(arm) for arm in arms], indent=2))
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report: JsonObject = {
        "schema_version": 1,
        "status": "running",
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "protocol": vars(args).copy()
    }
    (args.out_dir / "working-tree.patch").write_text(
        environment.
        command_output(["git", "diff", "HEAD", "--binary"], workload.ROOT) or ""
    )
    _write_json(args.out_dir / "summary.json", report)
    try:
        _execute(args, report)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            SystemExit) as error:
        report.update(status="failed", error=str(error))
        _write_json(args.out_dir / "summary.json", report)
        raise
    print(f"Validated results: {args.out_dir / 'summary.md'}")
    return 0
