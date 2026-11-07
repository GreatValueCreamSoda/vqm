# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Aggregate and render validated in-memory benchmark measurements."""

import argparse
import datetime
import json
import math
import os
import statistics
from typing import Any
from typing import NotRequired
from typing import TextIO
from typing import TypedDict

Pair = tuple[str, str]
ArmKey = tuple[str, str]
JsonObject = dict[str, Any]


class RoundRow(TypedDict):
    """Store a driver's measured score and duration for one image pair."""

    original: str
    distorted: str
    width: int
    height: int
    core_ms: float
    score: float


class ThroughputFields(TypedDict):
    """Store a driver's validated fixed-window measurement."""

    size: str
    metric: str
    threads: int
    fps: float
    calls: int
    seconds: float
    busy: float
    checksum: float
    ceiling_fps: NotRequired[float]
    gpu_ms: NotRequired[float]


class ThroughputSize(TypedDict, total=False):
    """Store a geometry's measurements or a description of its failure."""

    error: str
    threads: int
    fps: float
    fps_one: float
    us_per_call: float
    runs_one: list[float]
    runs_many: list[float]
    measurements_one: list[ThroughputFields]
    measurements_many: list[ThroughputFields]
    ceiling_fps: float | None
    gpu_ms: float | None


class ThroughputRow(TypedDict):
    """Group the fixed-window measurements of a metric and implementation."""

    implementation: str
    metric: str
    threads: int
    sizes: dict[str, ThroughputSize]


class LatencyRow(TypedDict):
    """Summarize the repeated timings and scores at one image geometry."""

    metric: str
    implementation: str
    size: str
    width: int
    height: int
    pairs: int
    median_ms: float
    round_min_ms: float
    round_max_ms: float
    mean_ms: float
    p95_ms: float
    fps: float
    median_score: float
    max_delta: float | None
    mean_delta: float | None


MeasuredRounds = dict[ArmKey, list[list[RoundRow]]]

# The CPU rows report `threads * FPS at one thread`, labelled ideal. Ours
# reports the reciprocal stage time; Vship has no stage timers.
CPU_ENTRIES = ("cloudinary", "libjxl", "fssimu2")
OURS_ENTRIES = ("ours-resident", "ours-streaming", "ours-half-planes")
RESIDENT_NOTICE = (
    "SSIMULACRA 2 resident arms prepare both images on every call using "
    "alternating reference slots. They reuse pixel contents, so GPU memory "
    "may stay hot. Butteraugli resident arms cache the reference. "
    "CVVDP latency arms score independent stills with reset history; Vship "
    "explicitly uses standard_4k and includes its reset API overhead. "
    "Vship CVVDP is a peer comparison, not the numerical oracle.\n\n"
)


def p95(values: list[float]) -> float:
    """Select the sample nearest the 95th percentile.

    :param values: Nonempty sample.
    :returns: Value at the rounded percentile index.
    """
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


_PairMedian = tuple[Pair, float, float]


def _group_pair_medians(
    rounds: list[list[RoundRow]],
) -> dict[tuple[int, int], list[_PairMedian]]:
    per_pair = {}
    for rows in rounds:
        for index, row in enumerate(rows):
            key = (row["original"], row["distorted"])
            entry = per_pair.setdefault(
                index, {
                    "pair": key,
                    "width": row["width"],
                    "height": row["height"],
                    "core_ms": [],
                    "score": row["score"],
                }
            )
            entry["core_ms"].append(row["core_ms"])
    groups = {}
    for entry in per_pair.values():
        median_ms = statistics.median(entry["core_ms"])
        size = (entry["width"], entry["height"])
        group = groups.setdefault(size, [])
        group.append((entry["pair"], median_ms, entry["score"]))
    return groups


def _summarize_group(
    metric: str, entry_name: str, size: tuple[int,
                                              int], values: list[_PairMedian],
    rounds: list[list[RoundRow]], ours: dict[Pair, float] | None
) -> LatencyRow:
    medians = [value[1] for value in values]
    round_medians = [
        statistics.median(
            row["core_ms"] for row in round_rows
            if (row["width"], row["height"]) == size
        ) for round_rows in rounds
    ]
    scores = [value[2] for value in values]
    deltas = [] if ours is None else [
        0.0 if score == ours[key] else score - ours[key]
        for key, _median, score in values if key in ours
    ]
    median_ms = statistics.median(medians)
    return {
        "metric":
        metric,
        "implementation":
        entry_name,
        "size":
        f"{size[0]}x{size[1]}",
        "width":
        size[0],
        "height":
        size[1],
        "pairs":
        len(values),
        "median_ms":
        median_ms,
        "round_min_ms":
        min(round_medians),
        "round_max_ms":
        max(round_medians),
        "mean_ms":
        statistics.fmean(medians),
        "p95_ms":
        p95(medians),
        "fps":
        1000.0 / median_ms if median_ms > 0.0 else 0.0,
        "median_score":
        statistics.median(scores),
        "max_delta":
        max((abs(value) for value in deltas), default=None),
        "mean_delta": (
            statistics.fmean(deltas)
            if deltas and all(math.isfinite(d) for d in deltas) else None
        ),
    }


def aggregate(
    metric: str, entry_name: str, rounds: list[list[RoundRow]],
    ours: dict[Pair, float] | None
) -> list[LatencyRow]:
    """Summarize validated rounds by image geometry.

    :param metric: Selected metric name.
    :param entry_name: Benchmark implementation name.
    :param rounds: Complete result rows grouped by round.
    :param ours: Optional resident scores indexed by input pair.
    :returns: Timing and score summaries in geometry order.
    """
    groups = _group_pair_medians(rounds)
    return [
        _summarize_group(metric, entry_name, size, groups[size], rounds, ours)
        for size in sorted(groups)
    ]


def format_delta(value: float | None) -> str:
    """Render an optional score difference for Markdown.

    :param value: Difference against the resident arm, if available.
    :returns: Four decimal places or the missing-value label.
    """
    return "n/a" if value is None else f"{value:.4f}"


def json_safe(value: object) -> object:
    """Encode nonfinite floats as strings for strict JSON output.

    :param value: A report value, list or dictionary.
    :returns: The equivalent value with nonfinite floats converted to strings.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


_LATENCY_HEADER = (
    "| implementation | metric | size | pairs | median ms/pair | round range "
    "ms | "
    "FPS | median score | max absolute score difference vs ours |"
)
_LATENCY_DIVIDER = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"


def _latency_line(row: LatencyRow) -> str:
    return (
        f"| {row['implementation']} | {row['metric']} | "
        f"{row['size']} | {row['pairs']} | "
        f"{row['median_ms']:.3f} | "
        f"{row['round_min_ms']:.3f}–{row['round_max_ms']:.3f} | "
        f"{row['fps']:.1f} | {row['median_score']:.4f} | "
        f"{format_delta(row['max_delta'])} |"
    )


def write_summary(
    path: str, args: argparse.Namespace, pair_count: int,
    rows: list[LatencyRow], environment: JsonObject | None = None
) -> None:
    """Write the Markdown latency report.

    :param path: Output report path.
    :param args: Pair list and measurement protocol settings.
    :param pair_count: Number of requested pairs.
    :param rows: Validated latency summaries.
    :param environment: Optional provenance and environment data.
    :raises OSError: If the report cannot be written.
    """
    with open(path, "w") as f:
        f.write("# In-memory benchmark\n\n")
        f.write(
            f"generated: "
            f"{datetime.datetime.now(datetime.UTC).isoformat()}\n"
        )
        f.write(f"pairs: {args.pairs_file} ({pair_count} pairs)\n")
        f.write(
            f"protocol: decode once per pair; median of {args.iterations} "
            f"timed calls after {args.warmup} warmups; {args.rounds} "
            f"interleaved rounds; FPS is 1000 / median ms/pair\n\n"
        )
        if environment is not None:
            project = environment["project"]
            f.write(
                f"project revision: {project['revision'] or 'unknown'}; "
                f"dirty: {project['dirty']}\n\n"
            )
            f.write(
                "Input hashes, devices, build fingerprints, and all rounds "
                "are recorded in `summary.json`. The round range is the "
                "min–max of per-round median pair times.\n\n"
            )
        f.write(RESIDENT_NOTICE)
        f.write(
            "For alpha inputs, serial latency arms time the dark-background "
            "call and verify the bright background outside the timer. "
            "libjxl Butteraugli handles alpha inside its comparator, so "
            "its alpha latency may include different work.\n\n"
        )
        f.write(_LATENCY_HEADER + "\n")
        f.write(_LATENCY_DIVIDER + "\n")
        f.writelines(_latency_line(row) + "\n" for row in rows)


def ceiling_text(
    entry_name: str, threads: int, fps_at_one: float, ceiling: float | None
) -> str:
    """Label a GPU stage rate or an ideal CPU scaling rate.

    :param entry_name: Benchmark implementation name.
    :param threads: Measured worker count at this geometry.
    :param fps_at_one: Measured one-worker throughput.
    :param ceiling: Optional reciprocal GPU stage duration.
    :returns: Formatted rate, or the missing-value label for unsupported arms.
    """
    if entry_name in OURS_ENTRIES:
        return "n/a" if ceiling is None else f"{ceiling:.0f} (gpu)"
    if entry_name in CPU_ENTRIES:
        return f"{threads * fps_at_one:.0f} (ideal)"
    return "n/a"


def _throughput_line(row: ThroughputRow, sizes: list[str]) -> str:
    entries = [row["sizes"].get(size, {}) for size in sizes]
    cells = [
        "n/a" if "error" in entry else f"{entry['fps']:.0f}"
        for entry in entries
    ]
    cells.extend(
        "n/a" if "error" in entry else f"{entry['fps_one']:.0f}"
        for entry in entries
    )
    cells.extend(
        "n/a" if "error" in entry else ceiling_text(
            row["implementation"], entry.get("threads", row["threads"]),
            entry["fps_one"], entry["ceiling_fps"]
        ) for entry in entries
    )
    threads_text = "/".join(
        str(entry.get("threads", row["threads"])) for entry in entries
    )
    return (f"| {row['implementation']} | {row['metric']} | {threads_text} | "
            ) + " | ".join(cells) + " |\n"


def _write_throughput_table(
    f: TextIO, sizes: list[str], rows: list[ThroughputRow]
) -> None:
    header = (
        "| implementation | metric | threads | " +
        " | ".join(f"{size} FPS" for size in sizes) + " | " +
        " | ".join(f"{size} FPS @1" for size in sizes) + " | " +
        " | ".join(f"{size} stage-rate/ideal FPS" for size in sizes) + " |\n"
    )
    f.write(header)
    f.write(
        "| --- | --- | --- | " +
        " | ".join("---" for _ in range(3 * len(sizes))) + " |\n"
    )
    f.writelines(_throughput_line(row, sizes) for row in rows)
    f.write(
        "\nThe GPU figure is the reciprocal of one-thread GPU stage "
        "time (`prepare` + `scale` for SSIMULACRA 2, all timed "
        "Butteraugli stages). It omits host and transfer costs and is "
        "neither measured end-to-end throughput nor a proven limit "
        "under concurrency. `ideal` for a CPU arm "
        "is `threads * FPS@1`, an ideal linear-scaling reference.\n"
    )


def _call_duration_cells(row: ThroughputRow, size: str) -> list[str]:
    entry = row["sizes"].get(size, {})
    us = entry.get("us_per_call")
    if "error" in entry or us is None:
        return ["n/a", "n/a", "n/a"]
    call_rate = 1e6 / us
    return [f"{us:.1f}", f"{call_rate:.0f}", f"{entry['fps'] / call_rate:.2f}x"]


def _write_call_durations(
    f: TextIO, sizes: list[str], rows: list[ThroughputRow]
) -> None:
    f.write("\n## Single-worker call duration\n\n")
    f.write(
        "Each driver reports the sum of its metric durations (`busy`) "
        "and the number of calls. `busy / calls` measures the whole "
        "single-worker scoring call, including host work and GPU "
        "waits, but excluding gaps between calls. Its reciprocal is "
        "a call-rate reference, not a ceiling on aggregate throughput. "
        "The ratio compares measured N-worker FPS with that reference; "
        "it is not a scaling-efficiency percentage.\n\n"
    )
    f.write(
        "| implementation | metric | " + " | ".join(
            f"{size} us/call | {size} call-rate FPS | "
            f"{size} achieved/call-rate" for size in sizes
        ) + " |\n"
    )
    f.write(
        "| --- | --- | " + " | ".join("---"
                                      for _ in range(3 * len(sizes))) + " |\n"
    )
    for row in rows:
        cells = [
            cell for size in sizes for cell in _call_duration_cells(row, size)
        ]
        f.write(
            f"| {row['implementation']} | {row['metric']} | " +
            " | ".join(cells) + " |\n"
        )


def _throughput_fps(row: ThroughputRow | None, size: str) -> float | None:
    if row is None:
        return None
    entry = row["sizes"].get(size, {})
    return None if "error" in entry else entry["fps"]


def _vship_cells(rows: dict[str, ThroughputRow], size: str) -> list[str]:
    names = ("ours-resident", "vship-cuda", "vship-vulkan", "vship-fast-vulkan")
    ours, cuda, vulkan, fast = [
        _throughput_fps(rows.get(name), size) for name in names
    ]
    cells = [
        "n/a" if fps is None else f"{fps:.0f}"
        for fps in (ours, cuda, vulkan, fast)
    ]
    cells.extend(
        "n/a" if ours is None or not fps else f"{ours / fps:.2f}x"
        for fps in (cuda, fast)
    )
    return cells


def _write_vship_comparisons(
    f: TextIO, sizes: list[str], rows: list[ThroughputRow]
) -> None:
    f.write("\n## Vship comparison\n\n")
    for metric in sorted({row["metric"] for row in rows}):
        f.write(f"\n### {metric}\n\n")
        f.write(
            "| size | ours-resident FPS | vship-cuda FPS | "
            "vship-vulkan FPS | vship-fast-vulkan FPS | "
            "ours/vship-cuda | ours/vship-fast |\n"
        )
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        # The original lookup selected the first row for each implementation.
        metric_rows = {
            row["implementation"]: row
            for row in reversed(rows) if row["metric"] == metric
        }
        for size in sizes:
            f.write(
                f"| {size} | " + " | ".join(_vship_cells(metric_rows, size)) +
                " |\n"
            )


def write_throughput_report(
    path: str, args: argparse.Namespace, sizes: list[str],
    rows: list[ThroughputRow]
) -> None:
    """Write throughput, call-rate and peer comparison tables.

    :param path: Output report path.
    :param args: Measurement protocol settings.
    :param sizes: Measured geometries in display order.
    :param rows: Throughput measurements grouped by implementation and metric.
    :raises OSError: If the report cannot be written.
    """
    with open(path, "w") as f:
        f.write("# Exploratory fixed-window throughput (FPS)\n\n")
        f.write(
            f"generated: "
            f"{datetime.datetime.now(datetime.UTC).isoformat()}\n"
        )
        f.write(
            f"protocol: --throughput-seconds {args.throughput_seconds:g}, "
            f"median of {args.throughput_runs} runs; implementations "
            f"interleaved by size and round, with one-thread and N-thread "
            f"runs alternating; decode outside the window\n\n"
        )
        f.write(
            "These are fixed-duration windows, not the fixed-pair-count "
            "wall-clock protocol required for aggregate throughput claims.\n\n"
        )
        f.write(RESIDENT_NOTICE)
        f.write(
            "The threads column is per size (left to right): a CPU arm "
            "is capped where the frame is large enough that the workers "
            "would not fit in this machine's memory.\n\n"
        )
        _write_throughput_table(f, sizes, rows)
        _write_call_durations(f, sizes, rows)
        _write_vship_comparisons(f, sizes, rows)


def write_latency_results(
    args: argparse.Namespace, pair_count: int, rows: list[LatencyRow],
    environment: JsonObject, protocol: JsonObject, measured: MeasuredRounds,
    failed: set[ArmKey]
) -> None:
    """Publish both report formats from the same validated measurements.

    :param args: Pair list, protocol and output directory settings.
    :param pair_count: Number of requested pairs.
    :param rows: Validated latency summaries.
    :param environment: Input, build and device provenance.
    :param protocol: Driver arguments and repetition settings.
    :param measured: Raw rounds indexed by metric and implementation.
    :param failed: Arms to omit from the published report.
    :raises OSError: If the report cannot be written.
    """
    summary_path = os.path.join(args.out_dir, "summary.md")
    write_summary(summary_path, args, pair_count, rows, environment)
    json_path = os.path.join(args.out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(
            json_safe(
                {
                    "schema_version": 2,
                    "generated":
                    datetime.datetime.now(datetime.UTC).isoformat(),
                    "pairs": args.pairs_file,
                    "pair_count": pair_count,
                    "rows": rows,
                    "protocol": protocol,
                    "environment": environment,
                    # Keep the original field for consumers of schema v1.
                    "raw": {
                        f"{metric}/{name}": per_round[0]
                        for (metric, name), per_round in measured.items()
                        if per_round and (metric, name) not in failed
                    },
                    "raw_rounds": {
                        f"{metric}/{name}": per_round
                        for (metric, name), per_round in measured.items()
                        if per_round and (metric, name) not in failed
                    },
                }
            ),
            f,
            indent=2,
            allow_nan=False
        )

    print()
    print(_LATENCY_HEADER)
    print(_LATENCY_DIVIDER)
    for row in rows:
        print(_latency_line(row))
    print()
    print(f"wrote {summary_path}")
    print(f"wrote {json_path}")


def write_throughput_results(
    args: argparse.Namespace, sizes: list[str], rows: list[ThroughputRow],
    environment: JsonObject, protocol: JsonObject
) -> None:
    """Publish exploratory throughput and reproducibility data.

    :param args: Measurement protocol and output directory settings.
    :param sizes: Measured geometries in display order.
    :param rows: Validated throughput measurements.
    :param environment: Input, build and device provenance.
    :param protocol: Driver arguments and repetition settings.
    :raises OSError: If the report cannot be written or read back.
    """
    report_path = os.path.join(args.out_dir, "throughput.md")
    write_throughput_report(report_path, args, sizes, rows)
    json_path = os.path.join(args.out_dir, "throughput.json")
    with open(json_path, "w") as f:
        json.dump(
            json_safe(
                {
                    "schema_version": 2,
                    "generated":
                    datetime.datetime.now(datetime.UTC).isoformat(),
                    "sizes": sizes,
                    "throughput_seconds": args.throughput_seconds,
                    "throughput_runs": args.throughput_runs,
                    "rows": rows,
                    "protocol": protocol,
                    "environment": environment,
                }
            ), f, indent=2, allow_nan=False
        )
    with open(report_path) as f:
        text = f.read()
    # The report is the table; print it rather than a second
    # rendering of the same numbers.
    print()
    print(text.rstrip())
    print()
    print(f"wrote {report_path}")
    print(f"wrote {json_path}")
