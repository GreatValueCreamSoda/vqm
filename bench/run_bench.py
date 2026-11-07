#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Compare our CLI's speed with reference binaries on CID22 pairs.

Each pair is scored by every binary, then every binary is timed on the same
pair inside one worker so the runs share cache and thermal conditions. Wall
time includes process start and decoding, so this is a tool-vs-tool
comparison; for a decode/core/stage split of our own binary use
`vqm_bench` (see `--stages`).
"""

import argparse
import contextlib
import csv
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from typing import NotRequired
from typing import TypedDict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, ""):
    sys.path.insert(0, REPO_ROOT)
from validation.common import enumerate_pairs  # noqa: E402
from validation.common import parse_score  # noqa: E402
from validation.common import run_score  # noqa: E402

DEFAULT_BASELINES = (
    ("reference-scalar", "build/reference-build/ssimulacra2"),
    # Derivative implementation, built by tools/build_references.py. Its
    # scores differ by a few points, so compare speed, not the score column.
    ("fssimu2", "build/fssimu2-build/bin/fssimu2"),
)

Pair = tuple[str, str]
Binary = tuple[str, str, list[str]]


class _Arm(TypedDict):
    binary: str
    command: list[str]
    score: float | None
    error: str | None
    times: list[float]


class _Measurement(TypedDict):
    binary: str
    original: str
    distorted: str
    score: NotRequired[float | None]
    median_ms: NotRequired[float]
    min_ms: NotRequired[float]
    pair_index: NotRequired[int]
    error: NotRequired[str]


def parse_args() -> argparse.Namespace:
    """Parse input selection and CLI timing options.

    :returns: Requested benchmark settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default="validation/data/CID22_validation_set"
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="number of pairs to use (0 = all)"
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--pairs-file", default=None,
        help="tab-separated original<TAB>distorted list instead "
        "of the data-root enumeration"
    )
    parser.add_argument("--ours", default="build/vqm")
    parser.add_argument(
        "--baseline", action="append", default=[], metavar="NAME=PATH",
        help="reference binary to compare against (repeatable); "
        "default: every existing reference baseline"
    )
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="parallel pairs (default 1; >1 introduces contention)"
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="seconds allowed per process invocation"
    )
    parser.add_argument("--out", default="artifacts/reports/bench/results.csv")
    parser.add_argument(
        "--stages", action="store_true",
        help="also run vqm_bench and echo its stage table"
    )
    return parser.parse_args()


def _read_pair(line: str, path: str, line_number: int) -> Pair:
    original, separator, distorted = line.partition("\t")
    if not separator or not original or not distorted:
        raise ValueError(
            f"{path}:{line_number}: expected original<TAB>distorted"
        )
    return original, distorted


def read_pairs(path: str) -> list[Pair]:
    """Read a benchmark pair list while allowing blank lines and comments.

    :param path: Tab-separated pair list path.
    :returns: Original and distorted path pairs in file order.
    :raises ValueError: If a non-comment row is missing either path.
    :raises OSError: If the file cannot be read.
    """
    with open(path) as source:
        return [
            _read_pair(line, path, index)
            for index, text in enumerate(source, 1)
            if (line := text.rstrip("\n\r")) and not line.startswith("#")
        ]


def parse_baselines(specs: list[str]) -> list[tuple[str, str]]:
    """Resolve explicit comparison binaries or use the existing defaults.

    :param specs: Explicit NAME=PATH entries.
    :returns: Names and paths in selection order.
    :raises SystemExit: If an explicit entry is malformed.
    """
    baselines = []
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"bad --baseline {spec!r}, expected NAME=PATH")
        baselines.append((name, path))
    if baselines:
        return baselines
    return [
        (name, path) for name, path in DEFAULT_BASELINES
        if os.path.exists(path)
    ]


def _score_arm(pair: Pair, spec: Binary, timeout: float) -> _Arm:
    name, binary, extra_args = spec
    original, distorted = pair
    try:
        value, error = run_score(
            [binary, *extra_args], original, distorted, timeout=timeout
        )
    except OSError as exc:
        value, error = None, f"could not run {binary}: {exc}"
    if error is None and (math.isnan(value) or value == math.inf):
        error = f"invalid score {value}"
    return {
        "binary": name,
        "command": [binary, *extra_args, original, distorted],
        "score": value,
        "error": error,
        "times": [],
    }


def _score_changed(value: float, previous: float) -> bool:
    return math.isnan(value) or value == math.inf or (
        value != previous
        and not math.isclose(value, previous, rel_tol=1e-6, abs_tol=1e-6)
    )


def _repeat_arm(state: _Arm, measured: bool, timeout: float) -> None:
    start = time.perf_counter()
    try:
        result = subprocess.run(
            state["command"], capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.perf_counter() - start
    except subprocess.TimeoutExpired:
        state["error"] = f"timeout after {timeout}s"
        return
    except OSError as exc:
        state["error"] = f"could not run {state['command'][0]}: {exc}"
        return
    if result.returncode != 0:
        state["error"] = f"exit {result.returncode}"
        return
    try:
        value = parse_score(result.stdout)
    except ValueError as exc:
        state["error"] = f"unparsable score: {exc}"
        return
    previous = state["score"]
    if _score_changed(value, previous):
        state["error"] = f"score changed from {previous} to {value}"
    elif measured:
        state["times"].append(elapsed)


def _repeat_arms(
    states: list[_Arm], repetitions: int, measured: bool, timeout: float
) -> None:
    for _ in range(repetitions):
        for state in (state for state in states if state["error"] is None):
            _repeat_arm(state, measured, timeout)


def _measurement_row(state: _Arm, pair: Pair) -> _Measurement:
    original, distorted = pair
    row: _Measurement = {
        "binary": state["binary"],
        "original": original,
        "distorted": distorted
    }
    if state["error"] is not None:
        row["error"] = state["error"]
    else:
        row.update(
            median_ms=1000.0 * statistics.median(state["times"]),
            min_ms=1000.0 * min(state["times"]), score=state["score"]
        )
    return row


def measure_pair(
    pair: Pair, binaries: list[Binary], reps: int, warmup: int,
    timeout: float = 120.0
) -> list[_Measurement]:
    """Score each arm, then alternate their warmups and timed invocations.

    :param pair: Original and distorted input paths.
    :param binaries: Comparison arm names, executables and extra arguments.
    :param reps: Timed calls per arm.
    :param warmup: Warmup calls per arm.
    :param timeout: Per-call time limit in seconds.
    :returns: Measurement or error row for every arm.
    """
    states = [_score_arm(pair, spec, timeout) for spec in binaries]
    _repeat_arms(states, warmup, False, timeout)
    _repeat_arms(states, reps, True, timeout)
    return [_measurement_row(state, pair) for state in states]


def p95(values: list[float]) -> float:
    """Select the sample nearest the 95th percentile.

    :param values: Nonempty sample.
    :returns: Value at the rounded percentile index.
    """
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def _profile_stages(
    bench: str, pairs: list[Pair], reps: int, stages_csv: str, warmup: int,
    timeout: float
) -> str | None:
    with contextlib.suppress(FileNotFoundError):
        os.remove(stages_csv)
    parent = os.path.dirname(stages_csv) or "."
    with tempfile.TemporaryDirectory(prefix=".vqm-bench-",
                                     dir=parent) as directory:
        pairs_path = os.path.join(directory, "pairs.txt")
        temp_csv = os.path.join(directory, "stages.csv")
        with open(pairs_path, "w") as target:
            target.writelines(
                f"{original}\t{distorted}\n" for original, distorted in pairs
            )
        result = subprocess.run(
            [
                bench, "--pairs-file", pairs_path, "--iterations",
                str(reps), "--warmup",
                str(warmup), "--stages-csv", temp_csv
            ], timeout=timeout
        )
        if result.returncode:
            return f"stage profiler exited {result.returncode}"
        if not os.path.isfile(temp_csv):
            return "stage profiler produced no CSV"
        os.replace(temp_csv, stages_csv)
    return None


def profile_stages(
    bench: str, pairs: list[Pair], reps: int, stages_csv: str, warmup: int = 1,
    timeout: float = 120.0
) -> str | None:
    """Run the stage profiler with a private temporary pair list.

    :param bench: Path to vqm_bench.
    :param pairs: Original and distorted path pairs in benchmark order.
    :param reps: Timed calls per pair.
    :param stages_csv: Destination for the stage timing CSV.
    :param warmup: Warmup calls per pair.
    :param timeout: Total profiling timeout in seconds.
    :returns: Error message on failure, otherwise None.
    """
    try:
        return _profile_stages(bench, pairs, reps, stages_csv, warmup, timeout)
    except subprocess.TimeoutExpired:
        return f"stage profiler timed out after {timeout}s"
    except OSError as error:
        return f"stage profiler failed: {error}"


def _validate_args(args: argparse.Namespace) -> None:
    if args.reps < 1 or args.warmup < 0:
        raise SystemExit("--reps must be >= 1 and --warmup >= 0")
    if args.jobs < 1 or args.limit < 0 or args.offset < 0:
        raise SystemExit("--jobs must be >= 1 and --limit/--offset >= 0")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("--timeout must be a positive finite number")
    if not os.path.exists(args.ours):
        raise SystemExit(f"missing our binary: {args.ours}")


def _select_pairs(args: argparse.Namespace) -> list[Pair]:
    try:
        if args.pairs_file:
            pairs = read_pairs(args.pairs_file)
        else:
            pairs = [
                (
                    os.path.join(args.data_root,
                                 o), os.path.join(args.data_root, d)
                ) for o, d in enumerate_pairs(args.data_root)
            ]
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    if args.limit:
        pairs = pairs[args.offset:args.offset + args.limit]
    elif args.offset:
        pairs = pairs[args.offset:]
    if not pairs:
        raise SystemExit("no pairs selected")

    return pairs


def _select_binaries(args: argparse.Namespace) -> list[Binary]:
    baselines = parse_baselines(args.baseline)
    missing = [path for _, path in baselines if not os.path.exists(path)]
    if missing:
        raise SystemExit(f"missing baseline binaries: {', '.join(missing)}")
    binaries = (
        [("ours", args.ours, [])] +
        [(name, path, []) for name, path in baselines]
    )
    names = [name for name, _, _ in binaries]
    if len(set(names)) != len(names):
        raise SystemExit("baseline names must be unique and cannot be 'ours'")
    return binaries


def _clear_outputs(args: argparse.Namespace) -> None:
    # An unsuccessful run must not leave an older CSV looking like its result.
    outputs = [args.out]
    if args.stages:
        outputs.append(f"{os.path.splitext(args.out)[0]}_stages.csv")
    for output in outputs:
        with contextlib.suppress(FileNotFoundError):
            os.remove(output)


def _record_pair_rows(
    rows: list[_Measurement], pair_rows: list[_Measurement], index: int
) -> int:
    errors = 0
    for row in pair_rows:
        if "error" in row:
            errors += 1
            print(
                f"error {row['binary']} {row['original']}: {row['error']}",
                file=sys.stderr
            )
            continue
        row["pair_index"] = index
        rows.append(row)
    return errors


def _collect_rows(
    futures: list[Future[list[_Measurement]]]
) -> tuple[list[_Measurement], int]:
    rows = []
    errors = 0
    for index, future in enumerate(futures, 1):
        errors += _record_pair_rows(rows, future.result(), index - 1)
        if index % 25 == 0:
            print(f"  {index}/{len(futures)} pairs", file=sys.stderr)
    return rows, errors


def _measure_pairs(
    args: argparse.Namespace, pairs: list[Pair], binaries: list[Binary]
) -> tuple[list[_Measurement], int]:
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(
                measure_pair, pair, binaries, args.reps, args.warmup,
                args.timeout
            ) for pair in pairs
        ]
        return _collect_rows(futures)


def _print_summary(
    args: argparse.Namespace, pair_count: int, binaries: list[Binary],
    rows: list[_Measurement]
) -> None:
    by_binary = {name: [] for name, *_ in binaries}
    for row in rows:
        by_binary[row["binary"]].append(row)
    if not by_binary["ours"]:
        raise SystemExit("no successful measurement for our binary")

    ours_rows = {r["pair_index"]: r for r in by_binary["ours"]}

    print(
        f"\nreps: {args.reps} (+{args.warmup} warmup), pairs: {pair_count}, "
        f"jobs: {args.jobs}"
    )
    print(
        "| binary | pairs | median ms | mean ms | p95 ms | speedup vs ours | "
        "max abs score diff |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for name, *_ in binaries:
        entries = by_binary[name]
        if not entries:
            continue
        medians = [r["median_ms"] for r in entries]
        median_ms = statistics.median(medians)
        matched = [
            (ours_rows[r["pair_index"]], r) for r in entries
            if r["pair_index"] in ours_rows
        ]
        speedup = _format_speedup(matched)
        diffs = [
            0.0 if a["score"] == b["score"] else abs(a["score"] - b["score"])
            for a, b in matched
        ]
        diff_text = f"{max(diffs):.4f}" if diffs else "n/a"
        print(
            f"| {name} | {len(entries)} | {median_ms:.1f} | "
            f"{statistics.fmean(medians):.1f} | {p95(medians):.1f} | "
            f"{speedup} | {diff_text} |"
        )


def _format_speedup(matched: list[tuple[_Measurement, _Measurement]]) -> str:
    if not matched:
        return "n/a"
    ours_median = statistics.median(a["median_ms"] for a, _ in matched)
    their_median = statistics.median(b["median_ms"] for _, b in matched)
    return f"{ours_median / their_median:.2f}x" if their_median > 0 else "n/a"


def _write_csv(args: argparse.Namespace, rows: list[_Measurement]) -> None:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["binary", "original", "distorted", "median_ms", "min_ms", "score"]
        )
        for row in rows:
            writer.writerow(
                [
                    row["binary"], row["original"], row["distorted"],
                    f"{row['median_ms']:.3f}", f"{row['min_ms']:.3f}",
                    f"{row['score']:.8f}"
                ]
            )
    print(f"\nwrote {args.out} ({len(rows)} rows)")


def _optional_stages(args: argparse.Namespace, pairs: list[Pair]) -> int:
    if not args.stages:
        return 0
    stem = os.path.splitext(args.out)[0]
    stages_csv = f"{stem}_stages.csv"
    bench = os.path.join("build", "vqm_bench")
    if not os.path.exists(bench):
        print(
            "vqm_bench not built: cmake --build build --target vqm_bench",
            file=sys.stderr
        )
        return 1
    print(f"\n== stage profile ({stages_csv})")
    error = profile_stages(
        bench, pairs, args.reps, stages_csv, args.warmup, args.timeout
    )
    if error is not None:
        print(error, file=sys.stderr)
        return 1
    return 0


def main() -> int:
    """Measure selected pairs and write the CLI timing report.

    :returns: One if a measurement or profile failed, otherwise zero.
    :raises SystemExit: If no valid pairs or benchmark arms are available.
    """
    args = parse_args()
    _validate_args(args)
    pairs = _select_pairs(args)
    binaries = _select_binaries(args)
    if args.jobs > 1:
        print(
            "warning: --jobs > 1 runs pairs concurrently; measurements include "
            "cross-process contention and are not isolated latency comparisons",
            file=sys.stderr
        )

    _clear_outputs(args)
    rows, errors = _measure_pairs(args, pairs, binaries)
    _print_summary(args, len(pairs), binaries, rows)
    _write_csv(args, rows)
    errors += _optional_stages(args, pairs)

    if errors:
        print(f"{errors} failure(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
