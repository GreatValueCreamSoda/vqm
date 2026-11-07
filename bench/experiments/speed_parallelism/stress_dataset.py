#!/usr/bin/env python3
"""Recreate and compare the historical seed-113, 10,000-matrix SpEED stress set.

Promoted from vmaf-v1-pooling-parallelism/stress-provenance/generate.py and
compare.py. Generator draws, acceptance and FP32 packing retain their order.
FP64 eigvalsh truth is approximate near ill-conditioned singular thresholds;
status agreement is a compatibility check, not a proof of exact classification.
"""

import argparse
import csv
import hashlib
import json
import struct
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / "artifacts/reports/vmaf-speed-parallelism/stress"
HISTORICAL_SHA256 = "8f94a20d04763bfaacc361160eb0bb1e2cae41b31cc0cc1713deb19a4d696062"

StressCase = tuple[bytes, str, float, float, float, float, float]


def _record_case(
    cases: list[StressCase], diagonal: NDArray[np.float64],
    off_diagonal: NDArray[np.float64], group: str, log_range: float, target: float
) -> bool:
    """Round one candidate matrix and retain it if positive definite.

    :param cases: Output list receiving accepted matrix cases.
    :param diagonal: Diagonal of the tridiagonal system.
    :param off_diagonal: Adjacent off-diagonal coefficients.
    :param group: Case family used for reporting.
    :param log_range: Range used to generate log-spaced operands.
    :param target: Target minimum eigenvalue before quantization.
    :returns: Whether the rounded matrix was retained.
    """
    diagonal = np.asarray(diagonal, dtype="<f4")
    off_diagonal = np.asarray(off_diagonal, dtype="<f4")
    matrix = (
        np.diag(diagonal.astype(np.float64)) +
        np.diag(off_diagonal.astype(np.float64), 1) +
        np.diag(off_diagonal.astype(np.float64), -1)
    )
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] <= 0 or not np.isfinite(eigenvalues).all():
        return False
    bits = np.concatenate((diagonal, off_diagonal)).astype("<f4", copy=False).tobytes()
    cases.append(
        (
            bits, group, float(log_range), float(target), float(eigenvalues[0]),
            float(eigenvalues[-1]), float(diagonal.max())
        )
    )
    return True


def _generate_cases() -> list[StressCase]:
    """Draw, quantize and filter the historical stress cases.

    :returns: Ordered input bytes and eigenvalue diagnostics.
    """
    rng = np.random.default_rng(113)
    cases = []

    threshold = np.float32(1e-6)
    for index in range(500):
        target = threshold
        for _ in range((index // 2) % 50):
            target = np.nextafter(
                target, np.float32(np.inf if index % 2 else 0), dtype=np.float32
            )
        diagonal = np.linspace(1.0, 25.0, 25)
        diagonal[0] = target
        assert _record_case(cases, diagonal, np.zeros(24), "diagonal", 0, target)
    while len(cases) < 10000:
        index = len(cases) - 500
        band = (index // 2375) % 4
        low, high = [(0.0, 1.5), (1.5, 3.0), (3.0, 4.5), (4.5, 6.0)][band]
        log_range = float(rng.uniform(low, high))
        ld = np.exp(rng.uniform(-log_range, log_range, 25))
        ls = rng.uniform(-1, 1, 24) * np.median(ld)
        diagonal = ld**2
        diagonal[1:] += ls**2
        off_diagonal = ld[:-1] * ls
        matrix = (
            np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
        )
        eigenvalues = np.linalg.eigvalsh(matrix)
        delta = 10**rng.uniform(-12, -3)
        target = 1e-6 + delta * (1 if rng.random() < 0.5 else -1)
        _record_case(
            cases, diagonal - eigenvalues[0] + target, off_diagonal, f"range{band}",
            log_range, target
        )
    return cases


def _write_dataset(output: Path, cases: list[StressCase]) -> None:
    """Write the binary matrix data, truth table and manifest.

    :param output: Destination directory.
    :param cases: Accepted matrices with truth metadata.
    """
    path = output / "matrices.bin"
    with path.open("wb") as file:
        file.write(b"VQMTRI01")
        file.write(struct.pack("<I", len(cases)))
        for row in cases:
            file.write(row[0])
    with (output / "truth.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "id", "input_sha256", "group", "log_range", "target_before_float32",
                "truth_min_eigen", "truth_max_eigen", "max_diagonal"
            ]
        )
        for index, (bits, group, width, target, minimum, maximum,
                    largest) in enumerate(cases):
            writer.writerow(
                [
                    index,
                    hashlib.sha256(bits).hexdigest(), group,
                    repr(width),
                    repr(target),
                    repr(minimum),
                    repr(maximum),
                    repr(largest)
                ]
            )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "count": len(cases),
        "seed": 113,
        "format":
        "VQMTRI01, u32le count, repeated 25 float32le diagonals then 24 float32le off-diagonals",
        "dataset_sha256": digest,
        "numpy_version": np.__version__,
        "truth":
        "FP64 eigvalsh of FP32-rounded tridiagonal; only strictly positive eigenvalues retained",
        "historical_dataset_sha256": HISTORICAL_SHA256,
        "matches_historical_dataset": digest == HISTORICAL_SHA256,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"cases={len(cases)} sha256={digest} historical_match={digest == HISTORICAL_SHA256}"
    )


def generate(output: Path) -> None:
    """Generate tridiagonal matrices and binary64 eigenvalue truth data.

    :param output: Destination for generated artifacts.
    """
    output.mkdir(parents=True, exist_ok=True)
    cases = _generate_cases()
    _write_dataset(output, cases)


def verified_inputs(
    directory: Path
) -> tuple[dict[str, object], list[dict[str, str]], list[str]]:
    """Load a dataset after validating its layout, hashes and truth row identities.

    :param directory: Directory containing the generated dataset.
    :returns: Manifest, truth records and verified input identities.
    :raises ValueError: If the dataset layout, hash or input identities do not match.
    """
    manifest = json.loads((directory / "manifest.json").read_text())
    data = (directory / "matrices.bin").read_bytes()
    if hashlib.sha256(
            data).hexdigest() != manifest["dataset_sha256"] or data[:8] != b"VQMTRI01":
        raise ValueError("dataset hash/header mismatch")
    count = struct.unpack("<I", data[8:12])[0]
    if count == 0 or count != manifest["count"] or len(data) != 12 + count * 196:
        raise ValueError("invalid dataset length/count")
    with (directory / "truth.csv").open() as file:
        truth = list(csv.DictReader(file))
    if len(truth) != count:
        raise ValueError("truth row count mismatch")
    identities = []
    for index, row in enumerate(truth):
        bits = data[12 + index * 196:12 + (index + 1) * 196]
        if int(row["id"]
               ) != index or hashlib.sha256(bits).hexdigest() != row["input_sha256"]:
            raise ValueError(f"truth input identity mismatch at {index}")
        hash_value = 14695981039346656037
        for value in bits:
            hash_value = ((hash_value ^ value) * 1099511628211) & 0xffffffffffffffff
        identities.append(f"{hash_value:016x}")
    return manifest, truth, identities


def _validate_eigenvalue_bits(name: str, index: int, row: dict[str, str]) -> None:
    """Check that all 25 eigenvalue bit strings are binary32 words.

    :param name: Benchmark arm label.
    :param index: Row index in the input data.
    :param row: Candidate result fields.
    :raises ValueError: If a word has invalid size or range.
    """
    for rank in range(25):
        bits = row[f"e{rank}"]
        if len(bits) != 8 or not 0 <= int(bits, 16) <= 0xffffffff:
            raise ValueError(f"invalid eigenvalue bits: {name}, row {index}")


def _encode_counts(counter: Counter[tuple[int, float]]) -> dict[str, int]:
    """Encode classification counts with readable transition keys.

    :param counter: Counts by expected and reported status.
    :returns: Sorted counts keyed by transition label.
    """
    return {f"{a}->{b:g}": value for (a, b), value in sorted(counter.items())}


def _read_arm(
    name: str, path: Path, truth: list[dict[str, str]], identities: list[str],
    threshold: float
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Validate one CSV against the dataset and report classification errors.

    :param name: Arm label used in diagnostics.
    :param path: Result CSV path.
    :param truth: Verified truth records in dataset order.
    :param identities: Expected hash of each raw input.
    :param threshold: Singular-value cutoff.
    :returns: Verified rows and the classification summary.
    :raises ValueError: If rows, identities, status or eigenvalue words are invalid.
    """
    with path.open() as file:
        rows = list(csv.DictReader(file))
    if len(rows) != len(truth):
        raise ValueError(f"CSV row count mismatch: {name}")
    counts, by_group, mismatches = Counter(), defaultdict(Counter), []
    for index, row in enumerate(rows):
        if int(row["id"]) != index or row["input_fnv1a64"] != identities[index]:
            raise ValueError(f"CSV input identity mismatch: {name}, row {index}")
        status = float(row["combined_status"])
        if status not in (0.0, 1.0, 2.0):
            raise ValueError(f"invalid combined status: {name}, row {index}")
        _validate_eigenvalue_bits(name, index, row)
        expected = int(float(truth[index]["truth_min_eigen"]) < threshold)
        counts[(expected, status)] += 1
        by_group[truth[index]["group"]][(expected, status)] += 1
        if status != expected and len(mismatches) < 30:
            mismatches.append(
                {
                    "id": index,
                    "expected": expected,
                    "status": status,
                    "truth_min_eigen": truth[index]["truth_min_eigen"],
                    "group": truth[index]["group"],
                    "log_range": truth[index]["log_range"]
                }
            )
    return rows, {
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "counts_expected_to_actual": _encode_counts(counts),
        "by_group": {
            group: _encode_counts(counter)
            for group, counter in by_group.items()
        },
        "first_mismatches": mismatches,
    }


def _compare_arms(arm_rows: dict[str, list[dict[str, str]]],
                  count: int) -> list[dict[str, object]]:
    """Compare status and eigenvalue bits across every pair of arms.

    :param arm_rows: Verified CSV rows by arm label.
    :param count: Number of verified dataset records.
    :returns: Pairwise difference counts and examples.
    """
    comparisons = []
    for a, b in combinations(arm_rows, 2):
        left, right = arm_rows[a], arm_rows[b]
        status = [
            i for i in range(count)
            if float(left[i]["combined_status"]) != float(right[i]["combined_status"])
        ]
        eigen = [
            i for i in range(count)
            if any(left[i][f"e{rank}"] != right[i][f"e{rank}"] for rank in range(25))
        ]
        comparisons.append(
            {
                "a": a,
                "b": b,
                "status_differences": len(status),
                "first_status_ids": status[:30],
                "eigen_bit_different_rows": len(eigen),
                "first_eigen_bit_ids": eigen[:30]
            }
        )
        print(
            f"{a} vs {b}: status differences={len(status)}, eigen bit rows={len(eigen)}"
        )
    return comparisons


def compare(directory: Path, inputs: list[list[str]], output: Path) -> None:
    """Compare stress-probe result CSVs with verified truth and each other.

    :param directory: Directory containing the generated dataset.
    :param inputs: Labels and paths of result CSVs.
    :param output: Destination for generated artifacts.
    :raises ValueError: If input identity checks fail or an arm label is repeated.
    """
    manifest, truth, identities = verified_inputs(directory)
    threshold = float(np.float32(1e-6))
    result = {
        "dataset_sha256": manifest["dataset_sha256"],
        "count": len(truth),
        "truth_limit":
        "FP64 eigvalsh near the cutoff is approximate, not a formal classification oracle",
        "arms": {},
        "pairwise": []
    }
    arm_rows = {}
    for name, path in inputs:
        if name in arm_rows:
            raise ValueError(f"duplicate arm label: {name}")
        rows, result["arms"][name] = _read_arm(
            name, Path(path), truth, identities, threshold
        )
        arm_rows[name] = rows
    result["pairwise"] = _compare_arms(arm_rows, len(truth))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {len(truth)} inputs in each of {len(inputs)} arms; wrote {output}")


def main() -> None:
    """Generate stress matrices or compare benchmark result CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generator = commands.add_parser("generate")
    generator.add_argument("--out-dir", type=Path, default=DEFAULT / "data")
    comparison = commands.add_parser("compare")
    comparison.add_argument("--data-dir", type=Path, default=DEFAULT / "data")
    comparison.add_argument(
        "--csv", nargs=2, action="append", required=True, metavar=("LABEL", "PATH")
    )
    comparison.add_argument("--out", type=Path, default=DEFAULT / "comparison.json")
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.out_dir)
    else:
        compare(args.data_dir, args.csv, args.out)


if __name__ == "__main__":
    main()
