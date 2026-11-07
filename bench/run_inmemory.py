#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Compare still-image latency and fixed-window throughput.

Each implementation is driven through its in-memory harness (bench/inmemory/)
which decodes a pair outside the timer. The default metric set is SSIMULACRA 2
and Butteraugli. Select CVVDP explicitly for still-image latency.

* `latency` -- `--iterations` timed calls after `--warmup` warmups per pair, the
  median taken across `--rounds` interleaved rounds, grouped by geometry and
  reported as ms/pair and FPS = 1000 / ms.
* `throughput` -- a fixed `--throughput-seconds` window over a decoded pair.
  Compare one worker with the count in the manifest, subject to the CPU memory
  cap. Reports include FPS and the reciprocal of one-worker GPU stage time
  when available.

Rounds are interleaved one implementation at a time so GPU arms never overlap,
and inside a throughput row the one-thread and N-thread runs alternate, so both
columns share one clock sitting. Reports: <out-dir>/summary.md (latency),
throughput.md and their .json twins. Raw per-run CSVs and logs stay next to
them. Reports default to artifacts/reports/performance/; generated pair data
defaults to .cache/bench/.
"""

import argparse
import contextlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from typing import TypedDict

if __package__ in (None, ""):
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
from bench import inmemory_environment as probes
from bench import inmemory_measure as measurements
from bench import inmemory_report as reports
from bench.inmemory_measure import BenchEntry
from bench.inmemory_measure import Manifest
from tools.build_paths import relocate_build_path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "bench", "references.json")
DEFAULT_PAIRS_FILE = ".cache/bench/pairs/all.txt"


def parse_args() -> argparse.Namespace:
    """Parse the in-memory benchmark's selection and timing options.

    :returns: Requested benchmark settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-file", default=DEFAULT_PAIRS_FILE,
        help="latency-mode pair list"
    )
    parser.add_argument(
        "--metric", default="all",
        choices=("all", "ssimu2", "butteraugli", "cvvdp")
    )
    parser.add_argument(
        "--mode", default="all", choices=("all", "latency", "throughput")
    )
    parser.add_argument(
        "--only", default=None, help="comma-separated implementation names"
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--throughput-seconds", type=float, default=3.0)
    parser.add_argument("--throughput-runs", type=int, default=3)
    parser.add_argument(
        "--sizes", default="1920x1080,3840x2160",
        help="throughput geometries, e.g. 1920x1080,3840x2160"
    )
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--out-dir", default="artifacts/reports/performance")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument(
        "--allow-unverified-devices", action="store_true",
        help="allow comparisons when Vship device names cannot "
        "uniquely match our Vulkan device"
    )
    parser.add_argument(
        "--list", action="store_true", help="print the manifest and exit"
    )
    return parser.parse_args()


def load_manifest(build_dir: str = "build") -> Manifest:
    """Load the metric and implementation manifest.

    :param build_dir: Directory containing the selected candidate and
        references.
    :returns: Driver configurations grouped by metric.
    :raises OSError: If the manifest cannot be read.
    :raises json.JSONDecodeError: If the manifest contains malformed JSON.
    """
    with open(MANIFEST) as f:
        manifest = json.load(f)
    for entries in manifest.values():
        for entry in entries:
            entry["driver"] = relocate_build_path(entry["driver"], build_dir)
    return manifest


def capture_throughput_environment(
    sizes: list[str], entries: list[BenchEntry], devices: reports.JsonObject,
    build_dir: str = "build"
) -> reports.JsonObject:
    """Fingerprint the selected synthetic lists and the images they name.

    :param sizes: Measured geometries in benchmark order.
    :param entries: Selected comparison drivers.
    :param devices: Device matching report for those drivers.
    :param build_dir: Directory containing candidate outputs and reference
        builds.
    :returns: Provenance and any missing or malformed input lists.
    """
    pairs, pair_lists = [], {}
    for size in sizes:
        path = measurements.throughput_pairs_path(size)
        record = {"path": path, "sha256": probes.file_sha256(path)}
        try:
            selected = measurements.read_pairs(path)
            record["pairs"] = selected
            pairs.extend(selected)
        except (OSError, ValueError) as error:
            # Retain the measurement runner's input error alongside successful
            # geometries.
            record["error"] = str(error)
        pair_lists[size] = record
    environment = probes.capture_environment(pairs, entries, devices, build_dir)
    environment["pair_lists"] = pair_lists
    return environment


class _WindowState(TypedDict):
    threads: int
    one: list[reports.ThroughputFields]
    many: list[reports.ThroughputFields]
    error: str | None
    checksum_reference: reports.ThroughputFields | None


def _metric_entries(metric: str, manifest: Manifest,
                    only: set[str] | None) -> list[BenchEntry]:
    return [
        entry for entry in manifest[metric]
        if only is None or entry["name"] in only
    ]


def _report_missing_driver(entry: BenchEntry) -> None:
    print(
        f"missing {entry['driver']}; build it first "
        f"(cmake --build --preset release for ours-*, "
        f"python3 tools/build_references.py --only {entry['name']} otherwise)",
        file=sys.stderr
    )


def _throughput_entries(
    metric: str, entries: list[BenchEntry], failures: list[str]
) -> list[tuple[BenchEntry, reports.ThroughputRow]]:
    available = []
    for entry in entries:
        if not os.path.isfile(entry["driver"]):
            _report_missing_driver(entry)
            failures.append(f"{metric}/{entry['name']} (throughput)")
            continue
        row: reports.ThroughputRow = {
            "implementation": entry["name"],
            "metric": metric,
            "threads": int(entry.get("throughput_threads") or 1),
            "sizes": {}
        }
        available.append((entry, row))
    return available


def _window_state(
    entry: BenchEntry, row: reports.ThroughputRow, size: str
) -> _WindowState:
    requested = row["threads"]
    threads = measurements.cpu_worker_cap(entry["name"], size, requested)
    if threads != requested:
        print(
            f"{row['metric']}: {entry['name']} {size} capped to "
            f"{threads} of {requested} workers to stay inside "
            "this machine's memory"
        )
    return {
        "threads": threads,
        "one": [],
        "many": [],
        "error": None,
        "checksum_reference": None
    }


def _record_window(
    state: _WindowState, fields: reports.ThroughputFields | None,
    error: str | None, threads: int
) -> None:
    reference = state["checksum_reference"]
    if error is None and reference is not None:
        error = measurements.validate_throughput_checksum(fields, reference)
    if error is not None:
        state["error"] = error
        return
    if reference is None:
        state["checksum_reference"] = fields
    state["one" if threads == 1 else "many"].append(fields)


def _measure_windows(
    entry: BenchEntry, metric: str, size: str, state: _WindowState,
    args: argparse.Namespace, repeat: int
) -> None:
    if state["error"] is not None:
        return
    threads = state["threads"]
    for config in ([1] if threads == 1 else [1, threads]):
        print(
            f"{metric}: {entry['name']} {size} {config} thread(s) "
            f"run {repeat + 1}/{args.throughput_runs}"
        )
        fields, error = measurements.run_throughput(
            entry, metric, size, config, args, repeat, args.out_dir
        )
        _record_window(state, fields, error, config)
        if state["error"] is not None:
            return


def _publish_size(
    row: reports.ThroughputRow, size: str, state: _WindowState,
    failures: list[str]
) -> None:
    threads, error = state["threads"], state["error"]
    if error is not None:
        name = row["implementation"]
        print(f"  {name} {size}: {error}", file=sys.stderr)
        failures.append(f"{row['metric']}/{name}/{size}")
        row["sizes"][size] = {"error": error, "threads": threads}
        return
    one = state["one"]
    many = state["many"] or one
    per_call_us = [1e6 * fields["busy"] / fields["calls"] for fields in one]
    ceilings = [
        fields["ceiling_fps"] for fields in one if "ceiling_fps" in fields
    ]
    gpu_times = [fields["gpu_ms"] for fields in one if "gpu_ms" in fields]
    row["sizes"][size] = {
        "threads": threads,
        "fps": statistics.median(f["fps"] for f in many),
        "fps_one": statistics.median(f["fps"] for f in one),
        "us_per_call": statistics.median(per_call_us),
        "runs_one": [f["fps"] for f in one],
        "runs_many": [f["fps"] for f in many],
        "measurements_one": one,
        "measurements_many": many,
        "ceiling_fps": statistics.median(ceilings) if ceilings else None,
        "gpu_ms": statistics.median(gpu_times) if gpu_times else None,
    }


def _measure_throughput_size(
    args: argparse.Namespace, metric: str, size: str,
    available: list[tuple[BenchEntry,
                          reports.ThroughputRow]], failures: list[str]
) -> None:
    states = {
        entry["name"]: _window_state(entry, row, size)
        for entry, row in available
    }
    # Compare every arm in the same sitting: A1, AN, B1, BN, then repeat.
    for repeat in range(args.throughput_runs):
        for entry, _row in available:
            _measure_windows(
                entry, metric, size, states[entry["name"]], args, repeat
            )
    for entry, row in available:
        _publish_size(row, size, states[entry["name"]], failures)


def _throughput_sizes(args: argparse.Namespace) -> list[str]:
    return [part.strip() for part in args.sizes.split(",") if part.strip()]


def run_throughput_mode(
    args: argparse.Namespace, metrics: list[str], manifest: Manifest,
    only: set[str] | None, failures: list[str]
) -> tuple[list[reports.ThroughputRow], dict[reports.ArmKey,
                                             reports.ThroughputRow]]:
    """Measure interleaved fixed windows for every selected driver and size.

    :param args: Geometries, timing and output settings.
    :param metrics: Metrics to measure in order.
    :param manifest: Driver configurations grouped by metric.
    :param only: Optional implementation name filter.
    :param failures: Output list of failed arms or geometries.
    :returns: Throughput rows and their index by metric and implementation.
    """
    sizes = _throughput_sizes(args)
    rows = []
    details = {}
    for metric in metrics:
        entries = _metric_entries(metric, manifest, only)
        available = _throughput_entries(metric, entries, failures)
        for entry, row in available:
            rows.append(row)
            details[(metric, entry["name"])] = row
        for size in sizes:
            _measure_throughput_size(args, metric, size, available, failures)
    return rows, details


def _clear_reports(out_dir: str, filenames: tuple[str, str]) -> None:
    for name in filenames:
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(out_dir, name))


def prepare_latency_pairs(args: argparse.Namespace) -> list[reports.Pair]:
    """Invalidate earlier reports and prepare the requested pair list.

    :param args: Pair list and output directory settings.
    :returns: Ordered pairs to measure.
    :raises SystemExit: If the requested pair list cannot be prepared or read.
    """
    _clear_reports(args.out_dir, ("summary.md", "summary.json"))
    if not os.path.isfile(args.pairs_file):
        if args.pairs_file != DEFAULT_PAIRS_FILE:
            raise SystemExit(f"pair list {args.pairs_file} does not exist")
        print(f"{args.pairs_file} is missing; generating it")
        generated = subprocess.run(
            [
                sys.executable,
                os.path.join(REPO_ROOT, "bench", "make_bench_pairs.py")
            ], cwd=REPO_ROOT
        )
        if generated.returncode != 0 or not os.path.isfile(args.pairs_file):
            raise SystemExit(f"could not generate {args.pairs_file}")
    try:
        expected_pairs = measurements.read_pairs(args.pairs_file)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    return expected_pairs


def _latency_entries(
    metric: str, entries: list[BenchEntry], failures: list[str],
    failed: set[reports.ArmKey]
) -> list[BenchEntry]:
    available = []
    for entry in entries:
        if not os.path.isfile(entry["driver"]):
            _report_missing_driver(entry)
            failures.append(entry["name"])
            failed.add((metric, entry["name"]))
            continue
        available.append(entry)
    return available


def _measure_latency_round(
    args: argparse.Namespace, metric: str, entries: list[BenchEntry],
    round_index: int, expected_pairs: list[reports.Pair],
    measured: reports.MeasuredRounds, failed: set[reports.ArmKey],
    failures: list[str]
) -> None:
    for entry in entries:
        key = (metric, entry["name"])
        if key in failed:
            continue
        print(
            f"{metric}: {entry['name']} round {round_index + 1}/{args.rounds}"
        )
        round_rows, error = measurements.run_round(
            entry, metric, args, round_index, args.out_dir, expected_pairs
        )
        previous = measured.get(key)
        if error is None and previous:
            error = measurements.validate_repeated_round(
                round_rows, previous[0]
            )
        if error is not None:
            print(f"  {error}", file=sys.stderr)
            failures.append(entry["name"])
            failed.add(key)
            continue
        measured.setdefault(key, []).append(round_rows)


def _summarize_latency_metric(
    metric: str, entries: list[BenchEntry], measured: reports.MeasuredRounds,
    failed: set[reports.ArmKey]
) -> list[reports.LatencyRow]:
    ours = None
    ours_key = (metric, "ours-resident")
    ours_rounds = measured.get(ours_key)
    if ours_rounds and ours_key not in failed:
        ours = {
            (row["original"], row["distorted"]): row["score"]
            for row in ours_rounds[0]
        }
    rows = []
    for entry in entries:
        key = (metric, entry["name"])
        if key in failed or not measured.get(key):
            continue
        rows.extend(
            reports.aggregate(metric, entry["name"], measured[key], ours)
        )
    return rows


def measure_latency(
    args: argparse.Namespace, metrics: list[str], manifest: Manifest,
    only: set[str] | None, expected_pairs: list[reports.Pair],
    failures: list[str]
) -> tuple[list[reports.LatencyRow], reports.MeasuredRounds,
           set[reports.ArmKey]]:
    """Interleave rounds, reject unstable arms and aggregate results.

    :param args: Measurement protocol and output directory settings.
    :param metrics: Metrics to measure in order.
    :param manifest: Driver configurations grouped by metric.
    :param only: Optional implementation name filter.
    :param expected_pairs: Requested pair order.
    :param failures: Output list of failed implementation names.
    :returns: Latency rows, all measured rounds and the failed arm keys.
    """
    measured: reports.MeasuredRounds = {}
    failed: set[reports.ArmKey] = set()
    for metric in metrics:
        entries = _metric_entries(metric, manifest, only)
        available = _latency_entries(metric, entries, failures, failed)
        # Round-major order keeps all arms in one clock and thermal sitting.
        for round_index in range(args.rounds):
            _measure_latency_round(
                args, metric, available, round_index, expected_pairs, measured,
                failed, failures
            )
    rows = [
        row for metric in metrics for row in _summarize_latency_metric(
            metric, _metric_entries(metric, manifest, only), measured, failed
        )
    ]
    return rows, measured, failed


def _list_drivers(metrics: list[str], manifest: Manifest) -> None:
    for metric in metrics:
        for entry in manifest[metric]:
            extra = " ".join(entry.get("args", []))
            print(
                f"{metric:12} {entry['name']:20} {entry['driver']} "
                f"{extra} threads={entry.get('throughput_threads', 1)}".rstrip()
            )


def _validate_options(args: argparse.Namespace) -> list[str]:
    if args.rounds < 1 or args.iterations < 1 or args.warmup < 0:
        raise SystemExit("--rounds/--iterations must be >= 1 and --warmup >= 0")
    if (not math.isfinite(args.throughput_seconds)
            or args.throughput_seconds <= 0 or args.throughput_runs < 1):
        raise SystemExit(
            "--throughput-seconds must be > 0 and --throughput-runs >= 1"
        )
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("--timeout must be a positive finite number")
    if args.mode not in ("throughput", "all"):
        return []
    sizes = _throughput_sizes(args)
    if not sizes or any(not re.fullmatch(r"[1-9]\d*x[1-9]\d*", size)
                        for size in sizes):
        raise SystemExit(
            "--sizes must be a comma-separated list of positive WxH sizes"
        )
    return sizes


def _select_only(value: str | None, metrics: list[str],
                 manifest: Manifest) -> set[str] | None:
    if not value:
        return None
    only = {part.strip() for part in value.split(",") if part.strip()}
    known = dict.fromkeys(
        entry["name"] for metric in metrics for entry in manifest[metric]
    )
    unknown = sorted(only - known.keys())
    if unknown:
        raise SystemExit(
            f"unknown implementation(s) {', '.join(unknown)}; choose from: "
            f"{', '.join(known)}"
        )
    return only


def _protocol(
    args: argparse.Namespace, metrics: list[str], manifest: Manifest,
    only: set[str] | None
) -> reports.JsonObject:
    return {
        "iterations": args.iterations,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "mode": args.mode,
        "metric": args.metric,
        "arms": {
            f"{metric}/{entry['name']}": {
                "driver": entry["driver"],
                "args": entry.get("args", []),
                "throughput_threads": entry.get("throughput_threads", 1)
            }
            for metric in metrics
            for entry in _metric_entries(metric, manifest, only)
        }
    }


def _run_latency(
    args: argparse.Namespace, metrics: list[str], manifest: Manifest,
    only: set[str] | None, entries: list[BenchEntry],
    devices: reports.JsonObject, protocol: reports.JsonObject,
    failures: list[str]
) -> bool:
    expected_pairs = prepare_latency_pairs(args)
    environment = probes.capture_environment(
        expected_pairs, entries, devices, getattr(args, "build_dir", "build")
    )
    latency_protocol = {
        **protocol, "pairs_file_sha256": probes.file_sha256(args.pairs_file)
    }
    rows, measured, failed = measure_latency(
        args, metrics, manifest, only, expected_pairs, failures
    )
    if not rows:
        print("no implementation could be measured", file=sys.stderr)
        return False
    reports.write_latency_results(
        args, len(expected_pairs), rows, environment, latency_protocol,
        measured, failed
    )
    return True


def _run_fixed_windows(
    args: argparse.Namespace, metrics: list[str], manifest: Manifest,
    only: set[str] | None, entries: list[BenchEntry], sizes: list[str],
    devices: reports.JsonObject, protocol: reports.JsonObject,
    failures: list[str]
) -> bool:
    environment = capture_throughput_environment(
        sizes, entries, devices, getattr(args, "build_dir", "build")
    )
    _clear_reports(args.out_dir, ("throughput.md", "throughput.json"))
    rows, _details = run_throughput_mode(
        args, metrics, manifest, only, failures
    )
    if not rows:
        print("no throughput row could be measured", file=sys.stderr)
        return False
    reports.write_throughput_results(args, sizes, rows, environment, protocol)
    return True


def main() -> int:
    """Run the selected benchmark modes and publish validated reports.

    :returns: One if any selected arm fails, otherwise zero.
    :raises SystemExit: If options or required benchmark inputs are invalid.
    """
    args = parse_args()
    manifest = load_manifest(getattr(args, "build_dir", "build"))
    if args.metric == "cvvdp" and args.mode != "latency" and not args.list:
        raise SystemExit(
            "CVVDP still-image comparisons require --mode latency. "
            "Use build/vqm --metric cvvdp for complete clips."
        )
    metrics = ["ssimu2", "butteraugli"
               ] if args.metric == "all" else [args.metric]
    if args.list:
        _list_drivers(metrics, manifest)
        return 0
    sizes = _validate_options(args)
    only = _select_only(args.only, metrics, manifest)
    entries = [
        entry for metric in metrics
        for entry in _metric_entries(metric, manifest, only)
    ]
    devices = probes.probe_devices(
        entries, getattr(args, "allow_unverified_devices", False)
    )
    protocol = _protocol(args, metrics, manifest, only)
    os.makedirs(args.out_dir, exist_ok=True)
    failures = []
    latency_ok = True
    throughput_ok = True
    if args.mode in ("latency", "all"):
        latency_ok = _run_latency(
            args, metrics, manifest, only, entries, devices, protocol, failures
        )
    if args.mode in ("throughput", "all"):
        throughput_ok = _run_fixed_windows(
            args, metrics, manifest, only, entries, sizes, devices, protocol,
            failures
        )
    return 0 if not failures and latency_ok and throughput_ok else 1


if __name__ == "__main__":
    sys.exit(main())
