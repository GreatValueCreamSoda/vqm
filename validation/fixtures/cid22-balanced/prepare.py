#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Verify the frozen CID22 ranking subset and cache source symlinks.

The original corpus remains untouched. Every selected source must match the
raw-file fingerprints in the retained independent original reports. No metric
scores are used to select or exclude inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
ROOT = FIXTURE.parents[2]


def _digest_file(path: Path) -> str:
    """Identify one raw corpus input or annotation file."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _digest_json(value: object) -> str:
    """Match the canonical adapter's stable pair/input identifiers."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _read_pairs() -> list[tuple[str, str]]:
    """Read the maintained balanced subset and reject unsafe relative paths."""
    with (FIXTURE / "subset-mcos.csv").open(newline="",
                                            encoding="utf-8") as handle:
        pairs = [
            (row["reference_img"], row["distorted_img"])
            for row in csv.DictReader(handle)
        ]
    paths = {name for pair in pairs for name in pair}
    if any(Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name
           for name in paths):
        raise ValueError("subset contains a nonrelative or unsafe path")
    if len(pairs) != 180 or len(set(pairs)) != len(pairs):
        raise ValueError("the frozen subset must contain 180 distinct pairs")
    return pairs


def _check_annotations(data_root: Path) -> dict[str, object]:
    """Require exact annotations and promoted subset membership."""
    selection = json.loads(
        (FIXTURE / "selection.json").read_text(encoding="utf-8")
    )
    full = data_root / "CID22_validation_set.csv"
    if _digest_file(full) != selection["mcos_sha256"]:
        raise ValueError(
            f"authoritative CID22 annotation SHA-256 mismatch: {full}"
        )
    if _digest_file(FIXTURE / "subset-mcos.csv"
                    ) != selection["subset_csv_sha256"]:
        raise ValueError(
            "maintained subset CSV differs from the frozen selection"
        )
    return selection


def _check_expected_sample(
    expected: dict[str, object], pair: tuple[str, str], hashes: dict[str, str]
) -> None:
    """Verify pair membership and bytes against an independent report."""
    original, distorted = pair
    if expected["source_id"] != hashes[original]:
        raise ValueError(f"raw original SHA-256 mismatch: {original}")
    if expected["input_hash"] != _digest_json([hashes[original],
                                               hashes[distorted]]):
        raise ValueError(f"raw pair SHA-256 mismatch: {original}, {distorted}")


def _check_report(
    metric: str, pairs: list[tuple[str, str]], hashes: dict[str, str]
) -> None:
    """Verify original arms cover the complete selected corpus."""
    report = json.loads(
        (FIXTURE / f"{metric}-original.json").read_text(encoding="utf-8")
    )
    policy = json.loads(
        (FIXTURE / f"{metric}-policy.json").read_text(encoding="utf-8")
    )
    if report["metric"] != metric or report["revision"] != policy[
            "reference_revision"]:
        raise ValueError(
            f"{metric}: retained original identity or pin mismatch"
        )
    expected = {sample["sample_id"]: sample for sample in report["samples"]}
    actual = {_digest_json(list(pair)): pair for pair in pairs}
    if expected.keys() != actual.keys():
        raise ValueError(f"{metric}: retained original pair membership differs")
    for sample_id, pair in actual.items():
        _check_expected_sample(expected[sample_id], pair, hashes)


def verify_inputs(data_root: Path) -> list[str]:
    """Verify annotations, source membership and all corpus bytes.

    :param data_root: Existing full CID22 validation corpus.
    :returns: Sorted relative paths needed by the frozen source-balanced subset.
    :raises OSError: If an authoritative input or maintained fixture is absent.
    :raises ValueError: If annotations, membership or raw-file hashes differ.
    """
    selection = _check_annotations(data_root)
    pairs = _read_pairs()
    sources = {Path(original).stem for original, _ in pairs}
    if sources != set(selection["sources"]):
        raise ValueError(
            "subset original sources differ from the frozen selection"
        )
    paths = sorted({name for pair in pairs for name in pair})
    hashes = {name: _digest_file(data_root / name) for name in paths}
    for metric in ("ssimulacra2", "butteraugli"):
        _check_report(metric, pairs, hashes)
    return paths


def _link_source(original: Path, destination: Path) -> None:
    """Reuse matching symlinks and preserve conflicting files."""
    if destination.is_symlink() and destination.resolve() == original.resolve():
        return
    if destination.exists() or destination.is_symlink():
        raise ValueError(
            f"preserve existing conflicting subset path: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(original.resolve())


def _prepare_inputs(args: argparse.Namespace) -> list[str]:
    """Validate all corpus selections before creating generated links."""
    paths = verify_inputs(args.data_root)
    if args.check:
        return paths
    for name in paths:
        _link_source(args.data_root / name, args.output_dir / name)
    return paths


def main(argv: list[str] | None = None) -> int:
    """Prepare the measured subset without copying or changing media.

    :param argv: Arguments, or None for process arguments.
    :returns: Zero for verified/prepared inputs, one for an
        unavailable/mismatched input.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "validation/data/CID22_validation_set"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / ".cache/validation/cid22-balanced"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="verify original corpus and fixture without creating symlinks"
    )
    args = parser.parse_args(argv)
    try:
        paths = _prepare_inputs(args)
    except (OSError, ValueError) as error:
        print(f"CID22 balanced subset: {error}", file=sys.stderr)
        return 1
    print(f"verified 30 originals, 180 pairs, {len(paths)} raw input files")
    if not args.check:
        print(f"subset data root: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
