#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Bootstrap and benchmark the five qualified metrics on pinned hosted videos.

This module uses only the standard library until the managed reference Python
environment is available. Candidate clips are always scored by the normal vqm
CLI through the maintained temporal validation workflow.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from validation.temporal.cli_results import validate_cli_result  # noqa: E402

METRICS = ("ssimulacra2", "butteraugli", "vmaf-v0", "vmaf-v1", "cvvdp")
CHECKOUTS = "ssimulacra2,libjxl,vmaf,ColorVideoVDP"
BUILD_STEPS = "cloudinary,libjxl,vmaf,cvvdp"
SMOKE_SOURCES = "foreman_cif,bus_cif,mobile_cif"
RUNTIME = Path(".venv/bin/python")
PACKAGES = (
    "vulkan", "libpng", "libjpeg", "libhwy", "lcms2", "libbrotlienc", "libbrotlidec",
    "libbrotlicommon"
)
PACKAGE_VERSIONS = ("libhwy >= 0.15", "lcms2 >= 2.13")
BUILD_ENVIRONMENT = (
    "CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CMAKE_PREFIX_PATH",
    "CMAKE_TOOLCHAIN_FILE", "PKG_CONFIG_PATH", "PKG_CONFIG_LIBDIR", "VULKAN_SDK",
    "VK_ICD_FILENAMES", "VK_DRIVER_FILES", "VK_INSTANCE_LAYERS"
)


def positive(value: str) -> int:
    """Parse a positive benchmark count.

    :param value: Command-line integer.
    :returns: Positive count.
    :raises argparse.ArgumentTypeError: If the value is not positive.
    """
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return count


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse workload and bootstrap options before importing optional packages.

    :param argv: Explicit arguments or the process command line.
    :returns: Validated settings for one isolated benchmark run.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, epilog=
        "bench/benchmark.py defaults to --suite video; use --suite images for still-image peer comparisons."
    )
    parser.add_argument(
        "--profile", choices=("smoke", "qualification"), default="smoke"
    )
    parser.add_argument("--metrics", default=",".join(METRICS))
    parser.add_argument("--rounds", type=positive, default=5)
    parser.add_argument("--iterations", type=positive, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--jobs", type=positive, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--offline", action="store_true", help="require cached sources and packages"
    )
    parser.add_argument(
        "--setup-only", action="store_true", help="fetch and build without timing clips"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print commands without running or writing anything"
    )
    args = parser.parse_args(argv)
    selected = tuple(name.strip() for name in args.metrics.split(","))
    if not set(selected) <= set(METRICS) or len(selected) != len(set(selected)):
        parser.error("--metrics must select unique names from " + ",".join(METRICS))
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    args.metrics = selected
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    args.out_dir = (
        args.out_dir or ROOT / "artifacts/reports/video-benchmark" / timestamp
    ).resolve()
    return args


def setup_commands(args: argparse.Namespace) -> list[list[str]]:
    """Plan pinned reference, managed runtime, CLI and hosted-prefix setup.

    :param args: Profile, jobs and offline settings.
    :returns: Ordered commands; reference checkouts are never built in place.
    """
    fetch = [sys.executable, "tools/fetch_references.py", "--only", CHECKOUTS]
    hosted = [str(ROOT / RUNTIME), "-m", "validation.temporal.fetch"]
    if args.offline:
        fetch.append("--check")
        hosted.append("--check")
    if args.profile == "smoke":
        hosted.extend(("--only", SMOKE_SOURCES))
    return [
        fetch,
        [
            sys.executable, "tools/build_references.py", "--only", BUILD_STEPS,
            "--jobs",
            str(args.jobs)
        ], ["cmake", "--preset", "release"],
        [
            "cmake", "--build", "--preset", "release", "--target", "vqm", "vqm_bench",
            "--parallel",
            str(args.jobs)
        ], hosted
    ]


def metric_order(metrics: Sequence[str], round_index: int) -> tuple[str, ...]:
    """Rotate arm order between adjacent rounds.

    :param metrics: Selected metric names.
    :param round_index: Zero-based round index.
    :returns: One visit to each selected metric.
    """
    offset = round_index % len(metrics)
    return tuple(metrics[offset:]) + tuple(metrics[:offset])


def measurement_command(args: argparse.Namespace, metric: str,
                        output: Path) -> list[str]:
    """Use the public CLI workflow with checked originals for each measured arm.

    :param args: Workload and iteration settings.
    :param metric: One qualified metric.
    :param output: Unique arm report directory.
    :returns: Managed-Python validation command; no original score regeneration.
    """
    return [
        str(ROOT / RUNTIME), "-m", "validation.temporal.run", "--profile", args.profile,
        "--metrics", metric, "--iterations",
        str(args.iterations), "--warmup",
        str(args.warmup), "--jobs",
        str(args.jobs), "--offline", "--out-dir",
        str(output)
    ]


def preview(args: argparse.Namespace) -> int:
    """Print a complete dry-run without requiring dependencies or creating files.

    :param args: Validated command-line settings.
    :returns: Zero after printing the plan.
    """
    for command in setup_commands(args):
        print(shlex.join(command))
    print(shlex.join([str(ROOT / "build/vqm_bench"), "--device-info"]))
    if args.setup_only:
        return 0
    for index in range(args.rounds):
        for metric in metric_order(args.metrics, index):
            output = args.out_dir / f"round-{index + 1:02}" / metric
            print(shlex.join(measurement_command(args, metric, output)))
    return 0


def _capture(command: Sequence[str]) -> str:
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False, timeout=60
    )
    if result.returncode:
        raise RuntimeError(
            f"{shlex.join(command)}: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def _tool_commands() -> dict[str, str]:
    return {
        "git": "git",
        "cmake": "cmake",
        "ninja": "ninja",
        "uv": "uv",
        "meson": "meson",
        "pkg-config": "pkg-config",
        "glslc": "glslc",
        "cc": os.environ.get("CC", "cc"),
        "cxx": os.environ.get("CXX", "g++")
    }


def preflight() -> dict[str, object]:
    """Check system tools and development headers before downloads or builds.

    :returns: Exact tool paths/versions and development-package versions.
    :raises RuntimeError: If a required tool, header or compiler feature is missing.
    """
    commands = _tool_commands()
    missing = [value for value in commands.values() if shutil.which(value) is None]
    if missing:
        raise RuntimeError(
            "missing prerequisites: " + ", ".join(missing) +
            "; install the system tools listed in docs/temporal-validation.md"
        )
    tools = {
        name: {
            "path": shutil.which(command),
            "version": _capture([command, "--version"])
        }
        for name, command in commands.items()
    }
    version = re.search(r"cmake version (\d+)\.(\d+)", tools["cmake"]["version"])
    if version is None or tuple(map(int, version.groups())) < (3, 25):
        raise RuntimeError("CMake 3.25 or later is required")
    try:
        _capture(["pkg-config", "--exists", *PACKAGES, *PACKAGE_VERSIONS])
    except RuntimeError as error:
        raise RuntimeError(
            "install Vulkan, PNG, JPEG, Highway >=0.15, LCMS2 >=2.13 and Brotli development packages with pkg-config metadata"
        ) from error
    flags = shlex.split(_capture(["pkg-config", "--cflags", *PACKAGES]))
    _check_headers(commands["cxx"], flags)
    return {
        "tools": tools,
        "packages": {
            name: _capture(["pkg-config", "--modversion", name])
            for name in PACKAGES
        }
    }


def _check_headers(compiler: str, flags: list[str]) -> None:
    source = "#include <cstddef>\n#include <cstdio>\n#include <vulkan/vulkan.h>\n#include <png.h>\n#include <jpeglib.h>\n#include <hwy/highway.h>\n#include <lcms2.h>\n#include <brotli/encode.h>\n#include <brotli/decode.h>\nint main() { static_assert(__cplusplus >= 202002L); }\n"
    with tempfile.TemporaryDirectory(prefix="vqm-video-preflight-") as directory:
        path = Path(directory) / "headers.cc"
        path.write_text(source)
        try:
            _capture([compiler, "-std=c++20", "-fsyntax-only", *flags, str(path)])
        except RuntimeError as error:
            raise RuntimeError(
                f"C++20/Vulkan/PNG/JPEG development-header check failed: {error}"
            ) from error


def digest(path: Path) -> str:
    """Hash an existing artifact incrementally.

    :param path: Source, executable or report file.
    :returns: SHA-256 hexadecimal digest.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _run_command(command: list[str], log: Path,
                 environment: dict[str, str]) -> dict[str, object]:
    print("+ " + shlex.join(command), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("w") as stream:
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
            check=False
        )
    return {
        "command": command,
        "exit_code": result.returncode,
        "wall_seconds": time.perf_counter() - started,
        "log": str(log)
    }


def _setup(
    args: argparse.Namespace, report: dict[str, object], environment: dict[str, str]
) -> None:
    records = report["setup"]
    for index, command in enumerate(setup_commands(args)):
        record = _run_command(
            command, args.out_dir / f"setup-{index + 1:02}.log", environment
        )
        records.append(record)
        if record["exit_code"]:
            raise RuntimeError(
                f"setup command failed with exit {record['exit_code']}; see {record['log']}"
            )


def _source_provenance(output: Path) -> dict[str, object]:
    names = _capture(["git", "ls-files", "-co", "--exclude-standard", "-z"]).split("\0")
    hashes = {
        name: digest(ROOT / name)
        for name in sorted(set(names)) if name and (ROOT / name).is_file()
    }
    manifest = output / "source-files.json"
    manifest.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
        check=False
    )
    status = _capture(["git", "status", "--short"])
    (output / "working-tree.txt").write_text(status + "\n")
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status),
        "manifest": str(manifest),
        "manifest_sha256": digest(manifest),
        "files": len(hashes)
    }


def _runtime_provenance() -> dict[str, object]:
    code = "import importlib.metadata as m,json,sys;print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"
    return json.loads(_capture([str(ROOT / RUNTIME), "-c", code]))


def _reference_provenance(args: argparse.Namespace) -> dict[str, object]:
    directory = ROOT / "validation/temporal/oracles" / args.profile
    fixtures = {
        metric: {
            suffix: digest(directory / f"{metric}{suffix}.json")
            for suffix in ("", "-policy", "-provenance")
        }
        for metric in args.metrics
    }
    stamps = ROOT / "build/reference-fingerprints"
    builds = {
        name: json.loads((stamps / f"{name}.json").read_text())
        for name in BUILD_STEPS.split(",")
    }
    models = [
        ".reference/vmaf/model/vmaf_float_v0.6.1.json",
        ".reference/vmaf/model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json", *[
            str(path.relative_to(ROOT)) for path in sorted(
                (ROOT / ".reference/ColorVideoVDP/pycvvdp/vvdp_data").glob("*.json")
            )
        ]
    ]
    return {
        "checked_score_fixtures": fixtures,
        "reference_builds": builds,
        "models_sha256": {
            name: digest(ROOT / name)
            for name in models
        },
        "lock_sha256": digest(ROOT / "tools/references.lock"),
        "runtime_lock_sha256": digest(ROOT / "tools/cvvdp-reference-requirements.txt"),
        "sources_sha256": digest(ROOT / "validation/temporal/sources.json")
    }


def provenance(args: argparse.Namespace) -> dict[str, object]:
    """Fingerprint candidate, sources, originals, runtime and the shared device selector.

    :param args: Selected corpus and output path.
    :returns: Reproduction evidence, excluding any original timing interpretation.
    :raises RuntimeError: If the actual Vulkan device cannot be probed.
    """
    probe = ROOT / "build/vqm_bench"
    device = json.loads(_capture([str(probe), "--device-info"]))
    if not device.get("name") or not device.get("uuid"):
        raise RuntimeError("default Vulkan device probe omitted its name or UUID")
    return {
        "candidate": {
            "path": str(ROOT / "build/vqm"),
            "sha256": digest(ROOT / "build/vqm")
        },
        "device": device,
        "device_probe": {
            "command": [str(probe), "--device-info"],
            "sha256":
            digest(probe),
            "selection":
            "same default Vulkan selector as normal CLI; probe only, no clip scoring"
        },
        "source": _source_provenance(args.out_dir),
        "references": _reference_provenance(args),
        "runtime": _runtime_provenance(),
        "cmake_cache_sha256": digest(ROOT / "build/CMakeCache.txt"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "build_environment": {
            name: os.environ.get(name)
            for name in BUILD_ENVIRONMENT
        }
    }


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _validate_payload(
    values: list[dict[str, object]], metric: str, args: argparse.Namespace
) -> None:
    pairs, frames = (39, 12) if args.profile == "smoke" else (390, 40)
    if len(values) != pairs or len({value["sample_id"] for value in values}) != pairs:
        raise ValueError(f"{metric}: expected {pairs} distinct complete clip results")
    for value in values:
        validate_cli_result(value, metric, value["sample_id"], frames)


def _timings(values: list[dict[str, object]], timing: dict[str,
                                                           object]) -> dict[str, float]:
    core = [float(row["core_seconds"]) for row in values]
    decode = [float(row["decode_seconds"]) for row in values]
    frames = sum(int(row["frames"]) for row in values)
    wall = float(timing["wall_seconds_including_warmup_decode_and_setup"])
    if not math.isfinite(wall) or wall < 0:
        raise ValueError("candidate process wall time must be finite and nonnegative")
    return {
        "selected_core_seconds_sum": sum(core),
        "selected_core_clip_median_seconds": statistics.median(core),
        "selected_core_seconds_per_frame": sum(core) / frames,
        "selected_decode_seconds_sum": sum(decode),
        "selected_decode_clip_median_seconds": statistics.median(decode),
        "candidate_process_wall_seconds": wall
    }


def _metric_result(
    output: Path, metric: str, args: argparse.Namespace, binary_hash: str
) -> dict[str, object]:
    values = [
        json.loads(line)
        for line in (output / f"{metric}-candidate.jsonl").read_text().splitlines()
    ]
    _validate_payload(values, metric, args)
    agreement = _load(output / f"{metric}-agreement.json")
    timing = _load(output / f"{metric}-timing.json")
    if agreement["candidate"]["revision"] != "sha256:" + binary_hash:
        raise ValueError("candidate executable changed between benchmark arms")
    if (timing["iterations_per_clip"] != args.iterations
            or timing["warmup_per_clip"] != args.warmup):
        raise ValueError("candidate repetition counts differ from benchmark settings")
    files = (
        f"{metric}-candidate.jsonl", f"{metric}-agreement.json",
        f"{metric}-timing.json", "inputs.json", f"{metric}-pairs.tsv"
    )
    return {
        "samples": len(values),
        "frames_per_clip": values[0]["frames"],
        "timings": _timings(values, timing),
        "quality_passed": agreement["selected_gate_passed"],
        "gate_mode": agreement["gate_mode"],
        "ranking": agreement["ranking"]["status"],
        "numerical": agreement["numerical"]["status"],
        "numerical_diagnostics": agreement["numerical"],
        "srcc": agreement["ranking"]["global"]["srcc"],
        "tau_b": agreement["ranking"]["global"]["tau_b"],
        "policy_sha256": agreement["policy_sha256"],
        "configuration_hash": agreement["candidate"]["configuration_hash"],
        "input_manifest_hash": agreement["candidate"]["input_manifest_hash"],
        "report_hashes": {
            name: digest(output / name)
            for name in files
        }
    }


def _measure_metric(
    args: argparse.Namespace, metric: str, index: int, binary_hash: str,
    environment: dict[str, str]
) -> dict[str, object]:
    directory = args.out_dir / f"round-{index + 1:02}"
    output = directory / metric
    record = _run_command(
        measurement_command(args, metric, output), directory / f"{metric}-workflow.log",
        environment
    )
    record.update(metric=metric, round=index + 1, output=str(output), valid=False)
    try:
        record.update(_metric_result(output, metric, args, binary_hash))
        record["valid"] = record["exit_code"] == 0 and record["quality_passed"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        record["error"] = str(error)
    return record


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    middle = statistics.median(values)
    return {
        "median": middle,
        "min": min(values),
        "max": max(values),
        "mad": statistics.median(abs(value - middle) for value in values)
    }


def aggregate(records: list[dict[str, object]], metrics: Sequence[str],
              rounds: int) -> dict[str, object]:
    """Summarize repeated arms without treating failed or absent rounds as success.

    :param records: Every attempted arm with raw report references.
    :param metrics: Complete selected metric list.
    :param rounds: Required rounds per metric.
    :returns: Per-metric completion, quality and duration distributions.
    """
    return {
        metric:
        _aggregate_metric([row for row in records if row["metric"] == metric], rounds)
        for metric in metrics
    }


def _aggregate_metric(records: list[dict[str, object]],
                      rounds: int) -> dict[str, object]:
    timed = [record for record in records if record["valid"] and "timings" in record]
    names = tuple(timed[0]["timings"]) if timed else ()
    return {
        "expected_rounds":
        rounds,
        "attempted_rounds":
        len(records),
        "valid_rounds":
        sum(bool(record["valid"]) for record in records),
        "passed":
        len(records) == rounds and {record["round"]
                                    for record in records} == set(range(1, rounds + 1))
        and all(record["valid"] for record in records),
        "durations": {
            name: _distribution([float(row["timings"][name]) for row in timed])
            for name in names
        },
        "workflow_wall_seconds":
        _distribution([float(row["wall_seconds"]) for row in records])
    }


def _parameters(args: argparse.Namespace) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def _measure_all(
    args: argparse.Namespace, records: list[dict[str, object]], binary_hash: str,
    environment: dict[str, str]
) -> None:
    for index in range(args.rounds):
        for metric in metric_order(args.metrics, index):
            records.append(
                _measure_metric(args, metric, index, binary_hash, environment)
            )


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# Hosted video benchmark", "", f"Status: {report['status']}", "",
        "Candidate scores and timing come from the normal `vqm` CLI. Checked original scores are quality evidence; reference cache durations are not benchmark measurements.",
        "",
        "Core/decode durations describe each clip's selected median-core-time iteration. Candidate process wall includes all repetitions, warmups, decode and setup. Workflow wall additionally includes preparation, hashing and agreement evaluation.",
        "",
        "| Metric | Valid rounds | Core median ms/clip | Decode median ms/clip | Candidate wall median s |",
        "| --- | ---: | ---: | ---: | ---: |"
    ]
    for metric, value in report.get("metrics", {}).items():
        durations = value["durations"]
        names = (
            "selected_core_clip_median_seconds", "selected_decode_clip_median_seconds",
            "candidate_process_wall_seconds"
        )
        numbers = [
            durations[name]["median"] if name in durations else None for name in names
        ]
        cells = [
            "—" if number is None else f"{number * (1000 if index < 2 else 1):.6f}"
            for index, number in enumerate(numbers)
        ]
        lines.append(
            f"| {metric} | {value['valid_rounds']}/{value['expected_rounds']} | " +
            " | ".join(cells) + " |"
        )
    if "error" in report:
        lines.extend(("", "Error: " + str(report["error"])))
    lines.extend(
        (
            "",
            "All rounds, dispersion (minimum/maximum/MAD), parameters, fingerprints and raw report paths are retained in `summary.json`.",
            ""
        )
    )
    return "\n".join(lines)


def _write_summary(args: argparse.Namespace, report: dict[str, object]) -> None:
    report["metrics"] = aggregate(report["rounds"], args.metrics, args.rounds)
    (args.out_dir /
     "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "summary.md").write_text(_markdown(report))


def _execute(args: argparse.Namespace, report: dict[str, object]) -> int:
    environment = os.environ.copy()
    if args.offline:
        environment["UV_OFFLINE"] = "1"
    try:
        report["preflight"] = preflight()
        _setup(args, report, environment)
        report["provenance"] = provenance(args)
        if args.setup_only:
            report["status"] = "setup_only"
            return 0
        binary_hash = report["provenance"]["candidate"]["sha256"]
        _measure_all(args, report["rounds"], binary_hash, environment)
        metrics = aggregate(report["rounds"], args.metrics, args.rounds)
        report["status"] = "pass" if all(
            value["passed"] for value in metrics.values()
        ) else "fail"
        return 0 if report["status"] == "pass" else 1
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        report.update(status="fail", error=str(error))
        print(f"video benchmark: {error}", file=sys.stderr)
        return 1
    finally:
        _write_summary(args, report)
        print(f"Results: {args.out_dir / 'summary.md'}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Bootstrap a fresh checkout and record all requested hosted-video arms.

    :param argv: Explicit options or the process command line.
    :returns: Zero only for completed setup or a fully passing measured matrix.
    :raises RuntimeError: If the output directory already contains another run.
    """
    args = arguments(argv)
    if args.dry_run:
        return preview(args)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError(
            f"refusing to mix benchmark reports in nonempty {args.out_dir}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "suite": "video",
        "status": "incomplete",
        "parameters": _parameters(args),
        "setup": [],
        "rounds": []
    }
    return _execute(args, report)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as error:
        sys.exit(str(error))
