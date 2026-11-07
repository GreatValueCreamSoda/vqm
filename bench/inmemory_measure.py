# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Execute and validate individual still-image benchmark measurements."""

import argparse
import contextlib
import csv
import math
import os
import subprocess
import sys
import tempfile
from typing import NotRequired
from typing import TypedDict
from typing import cast

from bench import inmemory_report as reports

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    if entry_name not in reports.CPU_ENTRIES:
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
            _parse_pair(line, path, number)
            for number, text in enumerate(source, 1)
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
    return os.path.join(
        REPO_ROOT, ".cache", "bench", "pairs", f"synthetic-{size}.txt"
    )


def validate_round(
    rows: list[reports.RoundRow], metric: str,
    expected_pairs: list[reports.Pair] | None
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
                metric == "butteraugli" and
            (score < 0 or not math.isfinite(score))):
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
        if (row["width"], row["height"]) != (expected["width"],
                                             expected["height"]):
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
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout
        )
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
    out_csv = os.path.join(
        log_dir, f"{metric}-{entry['name']}-round{round_index}.csv"
    )
    log_path = os.path.join(
        log_dir, f"{metric}-{entry['name']}-round{round_index}.log"
    )
    with contextlib.suppress(FileNotFoundError):
        os.remove(out_csv)
    # The driver must create a fresh file: an earlier round's CSV is never a
    # valid result for a successful process that failed to write one.
    with tempfile.TemporaryDirectory(prefix=".benchmark-", dir=log_dir) as tmp:
        fresh_csv = os.path.join(tmp, "round.csv")
        argv = [
            entry["driver"], *entry.get("args", []), "--pairs-file",
            args.pairs_file, "--csv", fresh_csv, "--iterations",
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


def _convert_throughput_numbers(
    fields: reports.JsonObject
) -> reports.ThroughputFields:
    for name, convert in (("fps", float), ("threads", int), ("calls", int),
                          ("seconds", float), ("busy", float),
                          ("checksum", float), ("ceiling_fps",
                                                float), ("gpu_ms", float)):
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
        return None, (
            f"size {fields.get('size')} in the line, {expected_size} requested"
        )
    required = (
        "metric", "threads", "fps", "calls", "seconds", "busy", "checksum"
    )
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
            metric == "butteraugli" and
        (checksum < 0 or not math.isfinite(checksum))):
        return "invalid throughput checksum"
    for key in ("ceiling_fps", "gpu_ms"):
        if key in fields and (not math.isfinite(fields[key])
                              or fields[key] <= 0):
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
    lines = [
        line for line in stdout.splitlines() if line.startswith("throughput ")
    ]
    if len(lines) != 1:
        return None, f"expected one throughput line, got {len(lines)}"
    fields, error = _parse_throughput_fields(lines[0], expected_size)
    if fields is None:
        return None, error
    # Our SSIMULACRA driver spells out the metric name; peers abbreviate it.
    metric = "ssimu2" if fields["metric"] == "ssimulacra2" else fields["metric"]
    expected = "ssimu2" if expected_metric == "ssimulacra2" else expected_metric
    if expected is not None and metric != expected:
        return None, (
            f"metric {fields['metric']} in the line, {expected_metric} "
            f"requested"
        )
    if expected_threads is not None and fields["threads"] != expected_threads:
        return None, (
            f"threads {fields['threads']} in the line, {expected_threads} "
            f"requested"
        )
    error = _check_throughput_measurement(fields, metric)
    return (None, error) if error is not None else (fields, None)


def run_throughput(
    entry: BenchEntry, metric: str, size: str, threads: int,
    args: argparse.Namespace, run_index: int, log_dir: str
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
        return None, (
            f"{pairs_path} is missing; run python3 bench/make_bench_pairs.py"
        )
    try:
        pairs = read_pairs(pairs_path)
    except (OSError, ValueError) as error:
        return None, str(error)
    if len(pairs) != 1:
        return None, (
            f"{pairs_path}: synthetic throughput workload must contain one pair"
        )
    log_path = os.path.join(
        log_dir,
        f"throughput-{metric}-{entry['name']}-{size}-{threads}-{run_index}.log"
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
