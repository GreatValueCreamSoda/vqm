#!/usr/bin/env python3
"""Compare matched CSVs with optional provenance-bound ranking gates."""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):
    # Keep direct script invocation compatible with package imports.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.agreement_policy import ScoreSample
from validation.common import correlation_triple

PairKey = tuple[str, str]
ScoreMap = dict[PairKey, float]
HashMap = dict[PairKey, tuple[str, str]]


def pair_sample_id(pair: PairKey) -> str:
    """Encode both pair paths into an unambiguous canonical sample identity.

    :param pair: Original and distorted paths as written in the score CSV.
    :returns: A stable JSON array string.
    """
    return json.dumps(pair, separators=(",", ":"))


def pair_input_hash(hashes: tuple[str, str]) -> str:
    """Fingerprint ordered decoded hashes without changing inputs.

    :param hashes: Original and distorted YUV SHA-256 strings.
    :returns: A SHA-256 for the ordered pair of decoded inputs.
    """
    return hashlib.sha256(json.dumps(hashes, separators=(",", ":")).encode()
                          ).hexdigest()


def _score_row(path: Path, row: Mapping[str, str],
               column: str) -> tuple[PairKey, float]:
    pair = row["original"], row["distorted"]
    if not all(pair):
        raise ValueError(f"{path}: empty pair {pair}")
    try:
        score = float(row[column])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid score for {pair}") from exc
    if not math.isfinite(score):
        raise ValueError(f"{path}: non-finite score for {pair}")
    return pair, score


def _hash_columns(path: Path, fields: Sequence[str], column: str) -> bool:
    if not {"original", "distorted", column}.issubset(fields):
        raise ValueError(
            f"{path}: expected original, distorted and {column} columns"
        )
    hash_fields = {"original_yuv_sha256", "distorted_yuv_sha256"}
    present = hash_fields.intersection(fields)
    if present and present != hash_fields:
        raise ValueError(f"{path}: expected both YUV input hash columns")
    return bool(present)


def load_scores(path: Path, column: str) -> tuple[ScoreMap, HashMap | None]:
    """Load finite scores and optional provenance by exact pair.

    :param path: Score CSV to read.
    :param column: Numerical score column.
    :returns: Score and optional decoded-input hash dictionaries.
    :raises ValueError: If columns, pairs, hashes or scores are invalid.
    """
    scores: ScoreMap = {}
    hashes: HashMap = {}
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        present = _hash_columns(path, reader.fieldnames or (), column)
        rows = list(reader)
    for row in rows:
        pair, score = _score_row(path, row, column)
        if pair in scores:
            raise ValueError(f"{path}: duplicate pair {pair}")
        scores[pair] = score
        if present:
            hashes[pair] = row["original_yuv_sha256"], row[
                "distorted_yuv_sha256"]
    if not scores or any(not all(value) for value in hashes.values()):
        raise ValueError(f"{path}: no scores or missing YUV input hash")
    return scores, hashes if present else None


def direct_correlations(reference: Sequence[float],
                        candidate: Sequence[float]) -> dict[str, object]:
    """Report direct agreement as diagnostics rather than acceptance.

    :param reference: Reference scores in matched sample order.
    :param candidate: Candidate scores in the same order.
    :returns: Direct coefficients; undefined constant-data values are null.
    """
    values = correlation_triple(reference, candidate)
    coefficients = {
        name: value if math.isfinite(value) else None
        for name, value in
        zip(("tau_b", "srcc", "pearson"), values, strict=True)
    }
    return {"status": "diagnostic_only", **coefficients}


def compare(
    baseline: ScoreMap, candidate: ScoreMap,
    baseline_hashes: HashMap | None = None,
    candidate_hashes: HashMap | None = None
) -> dict[str, object]:
    """Compare exact pairs with separate ranking and numerical evidence.

    :param baseline: Original implementation's scores.
    :param candidate: Candidate implementation's scores.
    :param baseline_hashes: Optional original decoded input hashes.
    :param candidate_hashes: Optional candidate decoded input hashes.
    :returns: Numerical diagnostics and direct correlation statistics.
    :raises ValueError: If pair sets or available input hashes differ.
    """
    if set(baseline) != set(candidate):
        raise ValueError(
            f"pair mismatch: {len(set(baseline) - set(candidate))} "
            "missing from "
            f"candidate, {len(set(candidate) - set(baseline))} extra"
        )
    verified = baseline_hashes is not None and candidate_hashes is not None
    if verified and baseline_hashes != candidate_hashes:
        raise ValueError("YUV input hashes differ between runs")
    errors = [
        (abs(candidate[pair] - value), pair) for pair, value in baseline.items()
    ]
    ordered = sorted(error for error, _ in errors)
    position = 0.99 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    maximum, worst = max(errors)
    return {
        "pairs":
        len(errors),
        "input_hashes_verified":
        verified,
        "max_absolute":
        maximum,
        "mean_absolute":
        math.fsum(ordered) / len(ordered),
        "p99_absolute": (
            ordered[lower] * (upper - position) + ordered[upper] *
            (position - lower) if lower != upper else ordered[lower]
        ),
        "rmse":
        math.sqrt(math.fsum(value * value for value in ordered) / len(ordered)),
        "worst_pair": {
            "original": worst[0],
            "distorted": worst[1]
        },
        "direct_agreement":
        direct_correlations(
            list(baseline.values()), [candidate[pair] for pair in baseline]
        )
    }


def _check_bound_sample(
    pair: PairKey, score: float, hashes: tuple[str, str], sample: ScoreSample
) -> None:
    if sample.score != score or sample.source_id != pair[0]:
        raise ValueError("agreement report score or source does not match CSV")
    if any(re.fullmatch(r"[0-9a-fA-F]{64}", value) is None for value in hashes):
        raise ValueError(
            "ranking acceptance requires SHA-256 decoded-input hashes"
        )
    if sample.input_hash != pair_input_hash(hashes):
        raise ValueError("agreement report input hash does not match CSV")


def _bound_report(path: Path, scores: ScoreMap,
                  hashes: HashMap | None) -> dict[str, object]:
    from validation.agreement_policy import ScoreReport

    if hashes is None:
        raise ValueError(
            "ranking acceptance requires both YUV hash columns in each CSV"
        )
    raw = json.loads(path.read_text())
    report = ScoreReport.parse(raw)
    if report.revision == "unreported-external-scores":
        raise ValueError(
            "ranking acceptance requires a declared implementation revision"
        )
    samples = {sample.sample_id: sample for sample in report.samples}
    if set(samples) != {pair_sample_id(pair) for pair in scores}:
        raise ValueError("agreement report sample IDs do not match CSV pairs")
    for pair, score in scores.items():
        _check_bound_sample(
            pair, score, hashes[pair], samples[pair_sample_id(pair)]
        )
    return raw


def _policy_agreement(
    args: argparse.Namespace, baseline: ScoreMap, candidate: ScoreMap,
    baseline_hashes: HashMap | None, candidate_hashes: HashMap | None
) -> dict[str, object]:
    from validation.agreement import evaluate_agreement

    reference = _bound_report(args.reference_report, baseline, baseline_hashes)
    proposed = _bound_report(args.candidate_report, candidate, candidate_hashes)
    return evaluate_agreement(
        reference, proposed, json.loads(args.agreement_policy.read_text())
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--column", default="vqm_v0")
    parser.add_argument(
        "--reference-column", help="override --column for the baseline CSV"
    )
    parser.add_argument(
        "--candidate-column", help="override --column for the candidate CSV"
    )
    parser.add_argument(
        "--max-absolute", type=float,
        help="optional maximum permitted absolute drift"
    )
    parser.add_argument(
        "--mean-absolute", type=float,
        help="optional maximum permitted mean absolute drift"
    )
    parser.add_argument(
        "--agreement-policy", type=Path,
        help="frozen ranking policy; requires both canonical score reports"
    )
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--candidate-report", type=Path)
    args = parser.parse_args()
    for threshold in (args.max_absolute, args.mean_absolute):
        if threshold is not None and (not math.isfinite(threshold)
                                      or threshold < 0):
            parser.error("drift thresholds must be finite and non-negative")
    reports = (
        args.reference_report is not None and args.candidate_report is not None
    )
    if args.agreement_policy is not None and not reports:
        parser.error(
            "--agreement-policy requires --reference-report and "
            "--candidate-report"
        )
    return args


def main() -> int:
    """Run the CSV comparison and any explicitly requested acceptance policy.

    :returns: Zero for requested checks passing, one for failure, two for bad
        inputs.
    """
    args = _arguments()
    try:
        baseline, baseline_hashes = load_scores(
            args.baseline, args.reference_column or args.column
        )
        candidate, candidate_hashes = load_scores(
            args.candidate, args.candidate_column or args.column
        )
        result = compare(baseline, candidate, baseline_hashes, candidate_hashes)
        if args.agreement_policy:
            result["agreement"] = _policy_agreement(
                args, baseline, candidate, baseline_hashes, candidate_hashes
            )
    except (OSError, ValueError) as exc:
        print(f"score comparison: {exc}", file=sys.stderr)
        return 2
    passed = (
        (
            args.max_absolute is None
            or result["max_absolute"] <= args.max_absolute
        ) and (
            args.mean_absolute is None
            or result["mean_absolute"] <= args.mean_absolute
        )
    )
    result["within_thresholds"] = (
        passed if args.max_absolute is not None
        or args.mean_absolute is not None else None
    )
    ranking_passed = not args.agreement_policy or result["agreement"][
        "proposed_status"] == "pass"
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if passed and ranking_passed else 1


if __name__ == "__main__":
    sys.exit(main())
