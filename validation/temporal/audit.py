#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Challenge a frozen ranking policy with conceptual score mutations.

These mutations operate on recorded original score vectors, not on production
GPU implementations. Temporal ingestion and history faults require independent
public-API tests in addition to this policy sensitivity audit.
"""

import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path

from ..agreement import evaluate_agreement
from ..agreement import load_report


def _candidate(reference: Mapping[str, object], name: str) -> dict[str, object]:
    result = copy.deepcopy(dict(reference))
    result["implementation"] = "conceptual-output-mutation"
    result["revision"] = "mutation-audit-v1:" + name
    return result


def _mapped(
    reference: Mapping[str, object], name: str, transform: Callable[[float],
                                                                    float]
) -> dict[str, object]:
    result = _candidate(reference, name)
    for sample in result["samples"]:
        sample["score"] = transform(float(sample["score"]))
    return result


def _rotated(reference: Mapping[str, object]) -> dict[str, object]:
    result = _candidate(reference, "rotated-association")
    groups = defaultdict(list)
    for sample in result["samples"]:
        groups[(sample["source_id"], sample["configuration_id"])].append(sample)
    for samples in groups.values():
        ordered = sorted(samples, key=lambda sample: sample["sample_id"])
        scores = [sample["score"] for sample in ordered]
        for index, sample in enumerate(ordered):
            sample["score"] = scores[(index + 1) % len(scores)]
    return result


def _mutations(
    reference: Mapping[str, object]
) -> list[tuple[str, str, dict[str, object]]]:
    scores = [float(sample["score"]) for sample in reference["samples"]]
    minimum = min(scores)
    span = max(scores) - minimum
    if span <= 0:
        raise ValueError("mutation audit requires varying original scores")
    transforms = (
        ("identity", "pass", lambda score: score),
        ("offset-plus-three", "pass", lambda score: score + 3),
        ("positive-scale-quarter", "pass", lambda score: score * .25), (
            "increasing-cube", "pass", lambda score:
            (1 + (score - minimum) / span)**3
        ), ("reversed", "fail", lambda score: -score),
        ("constant-collapse", "fail", lambda _: minimum),
    )
    results = [
        (name, expected, _mapped(reference, name, transform))
        for name, expected, transform in transforms
    ]
    return [*results, ("rotated-association", "fail", _rotated(reference))]


def _summary(name: str, expected: str,
             evidence: Mapping[str, object]) -> dict[str, object]:
    ranking = evidence["ranking"]
    numerical = evidence["numerical"]
    return {
        "mutation": name,
        "expected_ranking": expected,
        "ranking": ranking["status"],
        "numerical": numerical["status"],
        "subjective": evidence["subjective"]["status"],
        "proposed": evidence["proposed_status"],
        "srcc": ranking["global"]["srcc"],
        "tau_b": ranking["global"]["tau_b"],
        "max_error": numerical["max_error"],
        "mae": numerical["mae"],
        "mean_offset": numerical["mean_offset"],
        "failed_domains": ranking["failures"]
    }


def run_audit(reference: Mapping[str, object],
              policy: Mapping[str, object]) -> dict[str, object]:
    """Check remappings and disrupted associations against a frozen policy.

    Numerical gates remain visible but their failure is not presumed: a policy
    may intentionally allow a large numerical range. Ranking behavior determines
    whether this mutation audit passes.

    :param reference: Complete canonical original score report.
    :param policy: Frozen agreement policy, including original
        input/configuration pins.
    :returns: Compact gate evidence and every unexpected ranking outcome.
    :raises ValueError: If original evidence, provenance or policy is invalid.
    """
    evaluate_agreement(reference, reference, policy)
    results = [
        _summary(
            name, expected, evaluate_agreement(reference, candidate, policy)
        ) for name, expected, candidate in _mutations(reference)
    ]
    discrepancies = [
        (
            f"{row['mutation']}: expected {row['expected_ranking']}, got "
            f"{row['ranking']}"
        ) for row in results if row["ranking"] != row["expected_ranking"]
    ]
    return {
        "schema_version": 1,
        "scope":
        "conceptual recorded-score mutations; no production fault injection",
        "metric": reference["metric"],
        "reference_revision": reference["revision"],
        "input_manifest_hash": reference["input_manifest_hash"],
        "configuration_hash": reference["configuration_hash"],
        "samples": len(reference["samples"]),
        "passed": not discrepancies,
        "discrepancies": discrepancies,
        "audits": results
    }


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a standalone policy sensitivity audit on checked score evidence.

    :param argv: Command-line arguments, or None for sys.argv.
    :returns: Zero for expected behavior, one for a missed mutation, two for
        invalid inputs.
    """
    args = _arguments(argv)
    try:
        reference = load_report(args.reference)
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
        result = run_audit(reference, policy)
        result["reference_report_sha256"] = hashlib.sha256(
            args.reference.read_bytes()
        ).hexdigest()
        result["policy_file_sha256"] = hashlib.sha256(args.policy.read_bytes()
                                                      ).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8"
        )
    except (OSError, ValueError, TypeError) as error:
        print(f"temporal policy audit: {error}", file=sys.stderr)
        return 2
    print(
        f"{result['metric']}: temporal policy mutation audit "
        f"{'PASS' if result['passed'] else 'FAIL'}"
    )
    for discrepancy in result["discrepancies"]:
        print(discrepancy)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
