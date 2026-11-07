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
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from typing import NotRequired, TypedDict, cast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import inmemory_report as reports

# Keep the report helpers available to callers that import the driver.
CPU_ENTRIES = reports.CPU_ENTRIES
OURS_ENTRIES = reports.OURS_ENTRIES
RESIDENT_NOTICE = reports.RESIDENT_NOTICE
aggregate = reports.aggregate
ceiling_text = reports.ceiling_text
format_delta = reports.format_delta
json_safe = reports.json_safe
p95 = reports.p95
write_latency_results = reports.write_latency_results
write_summary = reports.write_summary
write_throughput_report = reports.write_throughput_report
write_throughput_results = reports.write_throughput_results

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "bench", "references.json")
DEFAULT_PAIRS_FILE = ".cache/bench/pairs/all.txt"

# A CPU metric holds several float planes of the frame per worker. Measured on
# this machine (peak RSS of the whole process, 4K: libjxl Butteraugli 2.0 GB at
# one worker, 7.5 GB at four -- about 245 bytes per pixel), so sixteen workers
# would ask for ~32 GB on a 30.5 GB machine and be killed. The cap keeps the
# arms inside 60% of MemTotal and never lowers a GPU arm: our pipelines put the
# pixels in VRAM.
CPU_BYTES_PER_PIXEL = 256
CPU_MEMORY_SHARE = 0.6


class BenchEntry(TypedDict):
    """Describe a comparison driver's executable and optional arguments."""

    name: str
    driver: str
    args: NotRequired[list[str]]
    throughput_threads: NotRequired[int]


Manifest = dict[str, list[BenchEntry]]
PathLike = str | os.PathLike[str]

REFERENCE_NAMES = {
    "cloudinary": "ssimulacra2",
    "libjxl": "libjxl",
    "fssimu2": "fssimu2",
    "vship-cuda": "Vship",
    "vship-vulkan": "Vship",
    "vship-fast-vulkan": "Vship-fast",
}
SHARED_LIBRARIES = {
    "fssimu2": "build/fssimu2-build/lib/libssimu2.so",
    "vship-cuda": "build/vship-build-cuda/libvship.so",
    "vship-vulkan": "build/vship-build-vulkan/libvship.so",
    "vship-fast-vulkan": "build/vship-fast-build-vulkan/libvship.so",
}


def command_output(argv: Sequence[str], cwd: PathLike | None = None) -> str | None:
    """Read trimmed stdout from an optional environment probe.

    :param argv: Probe executable and arguments.
    :param cwd: Optional working directory.
    :returns: Probe output, or None after an error or timeout.
    """
    try:
        result = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=15, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def file_sha256(path: PathLike) -> str | None:
    """Fingerprint a benchmark input or binary when it is readable.

    :param path: File to hash.
    :returns: SHA-256 in hexadecimal, or None if the file cannot be read.
    """
    try:
        with open(path, "rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    except OSError:
        return None


def _git_at(path: str, *args: str) -> str | None:
    return command_output(["git", "-C", path, *args])


def _reference_environment(name: str) -> reports.JsonObject:
    checkout = REFERENCE_NAMES[name]
    path = os.path.join(REPO_ROOT, ".reference", checkout)
    stamp = os.path.join(REPO_ROOT, "build", "reference-fingerprints", name + ".json")
    try:
        with open(stamp, encoding="utf-8") as source:
            fingerprint = json.load(source)
    except (OSError, ValueError):
        fingerprint = None
    source_status = _git_at(
        path, "status", "--porcelain", "--ignored", "--ignore-submodules=none"
    )
    record = {
        "checkout": checkout,
        "revision": _git_at(path, "rev-parse", "HEAD"),
        "dirty": None if source_status is None else bool(source_status),
        "submodules": _git_at(path, "submodule", "status", "--recursive"),
        "build_fingerprint": fingerprint,
    }
    if name in SHARED_LIBRARIES:
        library = os.path.join(REPO_ROOT, SHARED_LIBRARIES[name])
        record["runtime_library"] = {"path": library, "sha256": file_sha256(library)}
    return record


def _compiler_path() -> str | None:
    prefixes = ("CMAKE_CXX_COMPILER:FILEPATH=", "CMAKE_CXX_COMPILER:STRING=")
    try:
        with open(os.path.join(REPO_ROOT, "build", "CMakeCache.txt")) as source:
            return next(
                (
                    line.partition("=")[2].strip()
                    for line in source if line.startswith(prefixes)
                ), None
            )
    except OSError:
        return None


def capture_environment(
    pairs: list[reports.Pair], entries: list[BenchEntry], devices: reports.JsonObject
) -> reports.JsonObject:
    """Capture benchmark provenance using the environment data available locally.

    :param pairs: Ordered image pairs whose input files will be fingerprinted.
    :param entries: Selected comparison drivers.
    :param devices: Device matching report for those drivers.
    :returns: Revisions, input hashes, binaries, compiler and device metadata.
    """
    project_status = _git_at(
        REPO_ROOT, "status", "--porcelain", "--untracked-files=normal"
    )
    project = {
        "revision": _git_at(REPO_ROOT, "rev-parse", "HEAD"),
        "dirty": None if project_status is None else bool(project_status),
    }
    references = {}
    executables = {}
    for entry in entries:
        name, driver = entry["name"], entry["driver"]
        executables[name] = {
            "path": os.path.abspath(driver),
            "sha256": file_sha256(driver)
        }
        if name in REFERENCE_NAMES:
            references[name] = _reference_environment(name)
    paths = dict.fromkeys(path for pair in pairs for path in pair)
    images = {path: {"sha256": file_sha256(path)} for path in paths}
    compiler = _compiler_path()
    return {
        "project":
        project,
        "references":
        references,
        "devices":
        devices,
        "executables":
        executables,
        "inputs":
        images,
        "platform":
        platform.platform(),
        "python":
        sys.version,
        "compiler": {
            "path": compiler,
            "version": command_output([compiler, "--version"]) if compiler else None
        },
        # This is an inventory, not proof of which device an arm selected.
        "nvidia_inventory":
        command_output(
            [
                "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,pci.bus_id",
                "--format=csv,noheader"
            ]
        ),
    }


def _probe_device(entry: BenchEntry) -> object:
    output = command_output([entry["driver"], "--device-info"])
    try:
        return json.loads(output) if output else None
    except ValueError:
        return None


def _select_vship_device(
    entry: BenchEntry, device_name: str, allow_unverified: bool
) -> tuple[list[reports.JsonObject], int, bool]:
    available = _probe_device(entry)
    if not isinstance(available, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("name"), str)
            or not isinstance(item.get("id"), int) for item in available):
        raise SystemExit(f"cannot inspect {entry['name']} devices; rebuild its driver")
    matches = [item for item in available if item["name"] == device_name]
    if len(matches) == 1:
        return available, matches[0]["id"], True
    reason = (
        "no matching device name"
        if not matches else "multiple devices share the selected name"
    )
    if not allow_unverified:
        raise SystemExit(
            f"{entry['name']}: {reason}; device identity cannot be "
            "verified. Use --allow-unverified-devices only after "
            "checking the devices manually."
        )
    print(
        f"warning: {entry['name']}: {reason}; using device 0 without "
        "verified identity", file=sys.stderr
    )
    return available, 0, False


def _apply_vship_selection(entries: list[BenchEntry], name: str, selected: int) -> None:
    # Use the Vship backend's own index for every metric that uses this driver.
    for entry in entries:
        if entry["name"] == name:
            entry["args"] = [*entry.get("args", []), "--gpu", str(selected)]


def probe_devices(
    entries: list[BenchEntry], allow_unverified: bool = False
) -> reports.JsonObject:
    """Match Vship device IDs to the selected Vulkan device by name.

    :param entries: Selected drivers whose Vship arguments will be updated.
    :param allow_unverified: Whether to fall back to device zero on an ambiguous match.
    :returns: Report of the device names and IDs selected for each arm.
    :raises SystemExit: If a probe fails or a name match is required but unavailable.
    """
    entries_by_name = {entry["name"]: entry for entry in entries}
    ours = next(
        (entry for name, entry in entries_by_name.items() if name in OURS_ENTRIES), None
    )
    vships = [
        entry for name, entry in entries_by_name.items() if name.startswith("vship-")
    ]
    if not ours or not vships:
        return {"status": "not_compared"}
    device = _probe_device(ours)
    if not isinstance(device, dict) or not isinstance(device.get("name"), str):
        raise SystemExit("cannot inspect our benchmark device; rebuild the benchmark")
    report: reports.JsonObject = {
        "ours": device,
        "vship": {},
        "status": "matched_by_name_only"
    }
    for entry in vships:
        available, selected, verified = _select_vship_device(
            entry, device["name"], allow_unverified
        )
        if not verified:
            report["status"] = "unverified"
        _apply_vship_selection(entries, entry["name"], selected)
        report["vship"][entry["name"]] = {"devices": available, "selected_id": selected}
    print(
        "Device matching uses names: Vship does not expose UUIDs; physical "
        "device identity is not proven by matching names.", file=sys.stderr
    )
    return report


def mem_total_bytes() -> int | None:
    """Read total physical memory from Linux's memory inventory.

    :returns: Total bytes, or None when MemTotal cannot be read.
    """
    try:
        with open("/proc/meminfo") as source:
            return next(
                (
                    int(line.split()[1]) * 1024
                    for line in source if line.startswith("MemTotal:")
                ), None
            )
    except OSError:
        return None


def cpu_worker_cap(entry_name: str, size: str, requested: int) -> int:
    """Cap CPU workers using the estimated memory cost per geometry.

    :param entry_name: Benchmark implementation name.
    :param size: Image geometry in WxH form.
    :param requested: Requested maximum worker count.
    :returns: Allowed worker count for the available host memory.
    """
    if entry_name not in CPU_ENTRIES:
        return requested
    total = mem_total_bytes()
    if total is None:
        return requested
    width, height = (int(part) for part in size.split("x"))
    per_worker = width * height * CPU_BYTES_PER_PIXEL
    if per_worker <= 0:
        return requested
    fit = max(1, int(total * CPU_MEMORY_SHARE) // per_worker)
    return min(requested, fit)


def parse_args() -> argparse.Namespace:
    """Parse the in-memory benchmark's selection and timing options.

    :returns: Requested benchmark settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-file", default=DEFAULT_PAIRS_FILE, help="latency-mode pair list"
    )
    parser.add_argument(
        "--metric", default="all", choices=("all", "ssimu2", "butteraugli", "cvvdp")
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


def load_manifest() -> Manifest:
    """Load the metric and implementation manifest.

    :returns: Driver configurations grouped by metric.
    :raises OSError: If the manifest cannot be read.
    :raises json.JSONDecodeError: If the manifest contains malformed JSON.
    """
    with open(MANIFEST) as f:
        return json.load(f)


def read_round_csv(path: PathLike) -> list[reports.RoundRow]:
    """Read one driver's raw latency results.

    :param path: Driver CSV path.
    :returns: Typed measurement rows in input order.
    :raises OSError: If the CSV cannot be read.
    :raises ValueError: If a measurement cannot be converted.
    :raises KeyError: If a required column is absent.
    """
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "original": row["original"],
                    "distorted": row["distorted"],
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "core_ms": float(row["core_ms"]),
                    "score": float(row["score"]),
                }
            )
    return rows


def _parse_pair(line: str, path: PathLike, number: int) -> reports.Pair:
    original, separator, distorted = line.partition("\t")
    if not separator or not original or not distorted:
        raise ValueError(f"{path}:{number}: expected original<TAB>distorted")
    return original, distorted


def read_pairs(path: PathLike) -> list[reports.Pair]:
    """Read the same tab-separated pair list as the benchmark drivers.

    :param path: Pair list path.
    :returns: Original and distorted paths in file order.
    :raises OSError: If the list cannot be read.
    :raises ValueError: If a row is malformed or the list contains no pairs.
    """
    with open(path) as source:
        pairs = [
            _parse_pair(line, path, number) for number, text in enumerate(source, 1)
            if (line := text.rstrip("\r\n")) and not line.startswith("#")
        ]
    if not pairs:
        raise ValueError(f"{path}: no image pairs")
    return pairs


def throughput_pairs_path(size: str) -> str:
    """Locate the synthetic pair list for a throughput geometry.

    :param size: Image geometry in WxH form.
    :returns: Absolute path to the corresponding pair list.
    """
    return os.path.join(REPO_ROOT, ".cache", "bench", "pairs", f"synthetic-{size}.txt")


def capture_throughput_environment(
    sizes: list[str], entries: list[BenchEntry], devices: reports.JsonObject
) -> reports.JsonObject:
    """Fingerprint the selected synthetic lists and the images they name.

    :param sizes: Measured geometries in benchmark order.
    :param entries: Selected comparison drivers.
    :param devices: Device matching report for those drivers.
    :returns: Provenance and any missing or malformed input lists.
    """
    pairs, pair_lists = [], {}
    for size in sizes:
        path = throughput_pairs_path(size)
        record = {"path": path, "sha256": file_sha256(path)}
        try:
            selected = read_pairs(path)
            record["pairs"] = selected
            pairs.extend(selected)
        except (OSError, ValueError) as error:
            # run_throughput reports the corresponding arm failure. Keep its
            # input error in the report alongside the successful geometries.
            record["error"] = str(error)
        pair_lists[size] = record
    environment = capture_environment(pairs, entries, devices)
    environment["pair_lists"] = pair_lists
    return environment


def validate_round(
    rows: list[reports.RoundRow], metric: str, expected_pairs: list[reports.Pair] | None
) -> str | None:
    """Reject incomplete or malformed results before publishing a round.

    :param rows: Measured rows in driver output order.
    :param metric: Selected metric and its score domain.
    :param expected_pairs: Requested pair order, if available.
    :returns: Explanation of the first invalid measurement, otherwise None.
    """
    actual_pairs = [(row["original"], row["distorted"]) for row in rows]
    if expected_pairs is not None and actual_pairs != expected_pairs:
        return (
            f"driver returned {len(actual_pairs)} pairs that do not match "
            f"the requested {len(expected_pairs)} pairs in order"
        )
    for row in rows:
        if row["width"] <= 0 or row["height"] <= 0:
            return "driver returned a non-positive image size"
        if not math.isfinite(row["core_ms"]) or row["core_ms"] <= 0:
            return "driver returned a non-positive or non-finite core time"
        score = row["score"]
        if metric == "cvvdp" and not math.isfinite(score):
            return "driver returned an invalid CVVDP score"
        if math.isnan(score) or score == math.inf or (
                metric == "butteraugli" and (score < 0 or not math.isfinite(score))):
            return "driver returned an invalid score"
    return None


def validate_repeated_round(
    rows: list[reports.RoundRow], first: list[reports.RoundRow]
) -> str | None:
    """Check that a pair's geometry and score stay stable across rounds.

    :param rows: Current round in verified input order.
    :param first: Reference round in the same order.
    :returns: Explanation of any mismatch, otherwise None.
    """
    # Each round has already been checked against the requested pair order.
    for row, expected in zip(rows, first, strict=False):
        if (row["width"], row["height"]) != (expected["width"], expected["height"]):
            return "driver changed an image geometry between rounds"
        a, b = row["score"], expected["score"]
        if a != b and not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6):
            return (
                f"driver changed the score for {row['original']} vs "
                f"{row['distorted']} between rounds: {b} -> {a}"
            )
    return None


def run_logged(
    argv: list[str], log_path: str, timeout: float
) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
    """Run a driver and keep its command and output together in one log.

    :param argv: Driver executable and arguments.
    :param log_path: Destination for the command and its output.
    :param timeout: Process time limit in seconds.
    :returns: Completed process on success, or an explanatory error.
    :raises OSError: If the log cannot be written.
    """
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        with open(log_path, "w") as log:
            log.write(" ".join(argv) + f"\ntimeout after {timeout}s\n")
        return None, f"timeout after {timeout}s"
    except OSError as error:
        with open(log_path, "w") as log:
            log.write(" ".join(argv) + f"\n{error}\n")
        return None, f"could not run {argv[0]}: {error} (see {log_path})"
    with open(log_path, "w") as log:
        log.write(" ".join(argv) + "\n")
        log.write(result.stdout)
        log.write(result.stderr)
    if result.returncode != 0:
        return None, f"exit {result.returncode} (see {log_path})"
    return result, None


def run_round(
    entry: BenchEntry, metric: str, args: argparse.Namespace, round_index: int,
    log_dir: str, expected_pairs: list[reports.Pair] | None = None
) -> tuple[list[reports.RoundRow] | None, str | None]:
    """Measure one latency round and publish only a verified CSV.

    :param entry: Driver configuration.
    :param metric: Selected metric and its score domain.
    :param args: Pair list, repetition counts and process timeout.
    :param round_index: Zero-based round number for raw filenames.
    :param log_dir: Directory for raw CSV and logs.
    :param expected_pairs: Requested pair order, if available.
    :returns: Validated measurements on success, or an explanatory error.
    :raises OSError: If raw evidence cannot be saved.
    """
    out_csv = os.path.join(log_dir, f"{metric}-{entry['name']}-round{round_index}.csv")
    log_path = os.path.join(log_dir, f"{metric}-{entry['name']}-round{round_index}.log")
    with contextlib.suppress(FileNotFoundError):
        os.remove(out_csv)
    # The driver must create a fresh file: an earlier round's CSV is never a
    # valid result for a successful process that failed to write one.
    with tempfile.TemporaryDirectory(prefix=".benchmark-", dir=log_dir) as tmp:
        fresh_csv = os.path.join(tmp, "round.csv")
        argv = [
            entry["driver"], *entry.get("args", []), "--pairs-file", args.pairs_file,
            "--csv", fresh_csv, "--iterations",
            str(args.iterations), "--warmup",
            str(args.warmup)
        ]
        _, error = run_logged(argv, log_path, args.timeout)
        if error is not None:
            return None, error
        if not os.path.isfile(fresh_csv):
            return None, f"no CSV from driver (see {log_path})"
        try:
            rows = read_round_csv(fresh_csv)
        except (OSError, ValueError, KeyError, csv.Error) as error:
            return None, f"invalid CSV: {error} (see {log_path})"
        if not rows:
            return None, f"empty CSV from driver (see {log_path})"
        error = validate_round(rows, metric, expected_pairs)
        if error is not None:
            return None, f"{error} (see {log_path})"
        os.replace(fresh_csv, out_csv)
        return rows, None


def _convert_throughput_numbers(fields: reports.JsonObject) -> reports.ThroughputFields:
    for name, convert in (("fps", float), ("threads", int), ("calls", int),
                          ("seconds", float), ("busy", float), ("checksum", float),
                          ("ceiling_fps", float), ("gpu_ms", float)):
        if name in fields:
            fields[name] = convert(fields[name])
    return cast(reports.ThroughputFields, fields)


def _parse_throughput_fields(
    line: str, expected_size: str
) -> tuple[reports.ThroughputFields | None, str | None]:
    fields: reports.JsonObject = {}
    for token in line.split()[1:]:
        if "=" not in token:
            return None, f"unparsable field {token!r} in {line!r}"
        key, value = token.split("=", 1)
        if key in fields:
            return None, f"duplicate field {key} in {line!r}"
        fields[key] = value
    if fields.get("size") != expected_size:
        return None, f"size {fields.get('size')} in the line, {expected_size} requested"
    required = ("metric", "threads", "fps", "calls", "seconds", "busy", "checksum")
    missing = next((name for name in required if name not in fields), None)
    if missing is not None:
        return None, f"the line has no {missing}"
    try:
        return _convert_throughput_numbers(fields), None
    except ValueError as error:
        return None, f"unparsable number ({error}) in {line!r}"


def _check_throughput_measurement(
    fields: reports.ThroughputFields, metric: str
) -> str | None:
    if fields["calls"] <= 0 or fields["threads"] <= 0 or any(
            not math.isfinite(value) or value <= 0
            for value in (fields["fps"], fields["seconds"], fields["busy"])):
        return "non-positive or non-finite throughput measurement"
    # Both values are printed to six decimal places. Account for the
    # rounding of seconds, especially for a deliberately short window.
    seconds = fields["seconds"]
    lower = fields["calls"] / (seconds + 0.0000005) - 0.0000005
    upper = (
        fields["calls"] / (seconds - 0.0000005) +
        0.0000005 if seconds > 0.0000005 else math.inf
    )
    if not lower <= fields["fps"] <= upper:
        return "throughput FPS does not agree with calls / seconds"
    checksum = fields["checksum"]
    if math.isnan(checksum) or checksum == math.inf or (
            metric == "butteraugli" and (checksum < 0 or not math.isfinite(checksum))):
        return "invalid throughput checksum"
    for key in ("ceiling_fps", "gpu_ms"):
        if key in fields and (not math.isfinite(fields[key]) or fields[key] <= 0):
            return f"invalid {key} in throughput measurement"
    return None


def parse_throughput_line(
    stdout: str, expected_size: str, expected_metric: str | None = None,
    expected_threads: int | None = None
) -> tuple[reports.ThroughputFields | None, str | None]:
    """Parse and validate the driver's single fixed-window result line.

    :param stdout: Complete driver output.
    :param expected_size: Requested WxH geometry.
    :param expected_metric: Selected metric, if known.
    :param expected_threads: Requested worker count, if known.
    :returns: Validated measurements or an explanatory error.
    """
    lines = [line for line in stdout.splitlines() if line.startswith("throughput ")]
    if len(lines) != 1:
        return None, f"expected one throughput line, got {len(lines)}"
    fields, error = _parse_throughput_fields(lines[0], expected_size)
    if fields is None:
        return None, error
    # Our SSIMULACRA driver spells out the metric name while peers use the short name.
    metric = "ssimu2" if fields["metric"] == "ssimulacra2" else fields["metric"]
    expected = "ssimu2" if expected_metric == "ssimulacra2" else expected_metric
    if expected is not None and metric != expected:
        return None, f"metric {fields['metric']} in the line, {expected_metric} requested"
    if expected_threads is not None and fields["threads"] != expected_threads:
        return None, f"threads {fields['threads']} in the line, {expected_threads} requested"
    error = _check_throughput_measurement(fields, metric)
    return (None, error) if error is not None else (fields, None)


def run_throughput(
    entry: BenchEntry, metric: str, size: str, threads: int, args: argparse.Namespace,
    run_index: int, log_dir: str
) -> tuple[reports.ThroughputFields | None, str | None]:
    """Run a driver for one fixed-duration measurement window.

    :param entry: Driver configuration.
    :param metric: Selected metric.
    :param size: Requested WxH geometry.
    :param threads: Requested worker count.
    :param args: Window duration and process timeout.
    :param run_index: Zero-based repetition number for log filenames.
    :param log_dir: Directory for raw driver logs.
    :returns: Validated measurements or an explanatory error.
    """
    pairs_path = throughput_pairs_path(size)
    if not os.path.isfile(pairs_path):
        return None, (f"{pairs_path} is missing; run python3 bench/make_bench_pairs.py")
    try:
        pairs = read_pairs(pairs_path)
    except (OSError, ValueError) as error:
        return None, str(error)
    if len(pairs) != 1:
        return None, f"{pairs_path}: synthetic throughput workload must contain one pair"
    log_path = os.path.join(
        log_dir, f"throughput-{metric}-{entry['name']}-{size}-{threads}-{run_index}.log"
    )
    argv = [
        entry["driver"], *entry.get("args", []), "--throughput-seconds",
        f"{args.throughput_seconds:g}", "--threads",
        str(threads), "--pairs-file", pairs_path
    ]
    result, error = run_logged(argv, log_path, args.timeout)
    if error is not None:
        return None, error
    fields, error = parse_throughput_line(result.stdout, size, metric, threads)
    if error is not None:
        return None, f"{error} (see {log_path})"
    return fields, None


def validate_throughput_checksum(
    fields: reports.ThroughputFields, first: reports.ThroughputFields
) -> str | None:
    """Check that the per-call score stays stable across measurement windows.

    :param fields: Current fixed-window result.
    :param first: First result for the same driver and input pair.
    :returns: Explanation of any score mismatch, otherwise None.
    """
    score = fields["checksum"] / fields["calls"]
    expected = first["checksum"] / first["calls"]
    if score == expected:
        return None
    if math.isfinite(score) and math.isfinite(expected):
        # EmitThroughput prints the checksum to six decimal places. Divide
        # each half-digit rounding bound by its own exact integer call count.
        tolerance = 0.0000005 / fields["calls"] + 0.0000005 / first["calls"]
        # Workers sum the same score separately before combining their sums;
        # allow the floating-point accumulation error to grow with call count.
        tolerance += 2 * sys.float_info.epsilon * (
            abs(score) * fields["calls"] + abs(expected) * first["calls"]
        )
        if abs(score - expected) <= tolerance:
            return None
    return (
        "throughput checksum / calls changed between worker counts or repeats: "
        f"{expected:.12g} -> {score:.12g}"
    )


class _WindowState(TypedDict):
    threads: int
    one: list[reports.ThroughputFields]
    many: list[reports.ThroughputFields]
    error: str | None
    checksum_reference: reports.ThroughputFields | None


def _metric_entries(metric: str, manifest: Manifest,
                    only: set[str] | None) -> list[BenchEntry]:
    return [
        entry for entry in manifest[metric] if only is None or entry["name"] in only
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
    threads = cpu_worker_cap(entry["name"], size, requested)
    if threads != requested:
        print(
            f"{row['metric']}: {entry['name']} {size} capped to "
            f"{threads} of {requested} workers to stay inside this machine's memory"
        )
    return {
        "threads": threads,
        "one": [],
        "many": [],
        "error": None,
        "checksum_reference": None
    }


def _record_window(
    state: _WindowState, fields: reports.ThroughputFields | None, error: str | None,
    threads: int
) -> None:
    reference = state["checksum_reference"]
    if error is None and reference is not None:
        error = validate_throughput_checksum(fields, reference)
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
        fields, error = run_throughput(
            entry, metric, size, config, args, repeat, args.out_dir
        )
        _record_window(state, fields, error, config)
        if state["error"] is not None:
            return


def _publish_size(
    row: reports.ThroughputRow, size: str, state: _WindowState, failures: list[str]
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
    ceilings = [fields["ceiling_fps"] for fields in one if "ceiling_fps" in fields]
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
    available: list[tuple[BenchEntry, reports.ThroughputRow]], failures: list[str]
) -> None:
    states = {
        entry["name"]: _window_state(entry, row, size)
        for entry, row in available
    }
    # Compare every arm in the same sitting: A1, AN, B1, BN, then repeat.
    for repeat in range(args.throughput_runs):
        for entry, _row in available:
            _measure_windows(entry, metric, size, states[entry["name"]], args, repeat)
    for entry, row in available:
        _publish_size(row, size, states[entry["name"]], failures)


def _throughput_sizes(args: argparse.Namespace) -> list[str]:
    return [part.strip() for part in args.sizes.split(",") if part.strip()]


def run_throughput_mode(
    args: argparse.Namespace, metrics: list[str], manifest: Manifest,
    only: set[str] | None, failures: list[str]
) -> tuple[list[reports.ThroughputRow], dict[reports.ArmKey, reports.ThroughputRow]]:
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
    """Invalidate earlier reports and load or generate the requested pair list.

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
            [sys.executable,
             os.path.join(REPO_ROOT, "bench", "make_bench_pairs.py")], cwd=REPO_ROOT
        )
        if generated.returncode != 0 or not os.path.isfile(args.pairs_file):
            raise SystemExit(f"could not generate {args.pairs_file}")
    try:
        expected_pairs = read_pairs(args.pairs_file)
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
    args: argparse.Namespace, metric: str, entries: list[BenchEntry], round_index: int,
    expected_pairs: list[reports.Pair], measured: reports.MeasuredRounds,
    failed: set[reports.ArmKey], failures: list[str]
) -> None:
    for entry in entries:
        key = (metric, entry["name"])
        if key in failed:
            continue
        print(f"{metric}: {entry['name']} round {round_index + 1}/{args.rounds}")
        round_rows, error = run_round(
            entry, metric, args, round_index, args.out_dir, expected_pairs
        )
        previous = measured.get(key)
        if error is None and previous:
            error = validate_repeated_round(round_rows, previous[0])
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
        rows.extend(aggregate(metric, entry["name"], measured[key], ours))
    return rows


def measure_latency(
    args: argparse.Namespace, metrics: list[str], manifest: Manifest,
    only: set[str] | None, expected_pairs: list[reports.Pair], failures: list[str]
) -> tuple[list[reports.LatencyRow], reports.MeasuredRounds, set[reports.ArmKey]]:
    """Interleave complete rounds, reject unstable arms, and aggregate results.

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
                args, metric, available, round_index, expected_pairs, measured, failed,
                failures
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
    if (not math.isfinite(args.throughput_seconds) or args.throughput_seconds <= 0
            or args.throughput_runs < 1):
        raise SystemExit("--throughput-seconds must be > 0 and --throughput-runs >= 1")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("--timeout must be a positive finite number")
    if args.mode not in ("throughput", "all"):
        return []
    sizes = _throughput_sizes(args)
    if not sizes or any(not re.fullmatch(r"[1-9]\d*x[1-9]\d*", size) for size in sizes):
        raise SystemExit("--sizes must be a comma-separated list of positive WxH sizes")
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
            f"unknown implementation(s) {', '.join(unknown)}; choose from: {', '.join(known)}"
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
    only: set[str] | None, entries: list[BenchEntry], devices: reports.JsonObject,
    protocol: reports.JsonObject, failures: list[str]
) -> bool:
    expected_pairs = prepare_latency_pairs(args)
    environment = capture_environment(expected_pairs, entries, devices)
    latency_protocol = {**protocol, "pairs_file_sha256": file_sha256(args.pairs_file)}
    rows, measured, failed = measure_latency(
        args, metrics, manifest, only, expected_pairs, failures
    )
    if not rows:
        print("no implementation could be measured", file=sys.stderr)
        return False
    write_latency_results(
        args, len(expected_pairs), rows, environment, latency_protocol, measured, failed
    )
    return True


def _run_fixed_windows(
    args: argparse.Namespace, metrics: list[str], manifest: Manifest,
    only: set[str] | None, entries: list[BenchEntry], sizes: list[str],
    devices: reports.JsonObject, protocol: reports.JsonObject, failures: list[str]
) -> bool:
    environment = capture_throughput_environment(sizes, entries, devices)
    _clear_reports(args.out_dir, ("throughput.md", "throughput.json"))
    rows, _details = run_throughput_mode(args, metrics, manifest, only, failures)
    if not rows:
        print("no throughput row could be measured", file=sys.stderr)
        return False
    write_throughput_results(args, sizes, rows, environment, protocol)
    return True


def main() -> int:
    """Run the selected benchmark modes and publish validated reports.

    :returns: One if any selected arm fails, otherwise zero.
    :raises SystemExit: If options or required benchmark inputs are invalid.
    """
    args = parse_args()
    manifest = load_manifest()
    if args.metric == "cvvdp" and args.mode != "latency" and not args.list:
        raise SystemExit(
            "CVVDP still-image comparisons require --mode latency. "
            "Use build/vqm --metric cvvdp for complete clips."
        )
    metrics = ["ssimu2", "butteraugli"] if args.metric == "all" else [args.metric]
    if args.list:
        _list_drivers(metrics, manifest)
        return 0
    sizes = _validate_options(args)
    only = _select_only(args.only, metrics, manifest)
    entries = [
        entry for metric in metrics
        for entry in _metric_entries(metric, manifest, only)
    ]
    devices = probe_devices(entries, getattr(args, "allow_unverified_devices", False))
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
            args, metrics, manifest, only, entries, sizes, devices, protocol, failures
        )
    return 0 if not failures and latency_ok and throughput_ok else 1


if __name__ == "__main__":
    sys.exit(main())
