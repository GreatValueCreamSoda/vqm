#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Qualify ranking agreement while retaining numerical parity diagnostics.

Canonical JSON reports contain schema_version, metric, implementation, revision,
configuration_hash, input_manifest_hash and samples. CSV reports repeat the five
provenance columns on every canonical sample row. Hashes describe real resolved
settings and input manifests; this tool never invents them from score values.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

try:
    from . import agreement_policy as schema
    from . import agreement_stats as stats
    from .gate_defaults import default_gate_mode
except ImportError:
    import agreement_policy as schema
    import agreement_stats as stats
    from gate_defaults import default_gate_mode

report_from_rows = schema.report_from_rows

__all__ = ["default_gate_mode", "evaluate_agreement", "load_report", "report_from_rows"]


def _matching_provenance(
    reference: schema.ScoreReport, candidate: schema.ScoreReport,
    policy: schema.AgreementPolicy
) -> None:
    if reference.metric != candidate.metric or reference.metric != policy.metric:
        raise ValueError("metric identities differ")
    if reference.revision != policy.reference_revision:
        raise ValueError("original revision does not match the frozen reference pin")
    for field in ("configuration_hash", "input_manifest_hash"):
        if getattr(reference, field) != getattr(candidate, field):
            raise ValueError(f"mismatched {field}")
        expected = getattr(policy, field)
        if expected is not None and getattr(reference, field) != expected:
            raise ValueError(f"{field} does not match the frozen policy pin")


def _aligned_samples(
    reference: schema.ScoreReport, candidate: schema.ScoreReport,
    policy: schema.AgreementPolicy
) -> list[tuple[schema.ScoreSample, schema.ScoreSample]]:
    _matching_provenance(reference, candidate, policy)
    first = {sample.sample_id: sample for sample in reference.samples}
    second = {sample.sample_id: sample for sample in candidate.samples}
    if first.keys() != second.keys():
        raise ValueError("original and candidate sample sets differ")
    aligned = [(first[name], second[name]) for name in sorted(first)]
    for pair in aligned:
        _matching_sample(*pair)
    _matching_domains(aligned, policy)
    return aligned


def _matching_sample(
    reference: schema.ScoreSample, candidate: schema.ScoreSample
) -> None:
    for field in ("source_id", "configuration_id", "stratum", "distortion_family",
                  "input_hash"):
        if getattr(reference, field) != getattr(candidate, field):
            raise ValueError(f"sample {reference.sample_id}: mismatched {field}")
    if candidate.mos is not None and candidate.mos != reference.mos:
        raise ValueError(f"sample {reference.sample_id}: mismatched MOS")


def _matching_domains(aligned: stats.Aligned, policy: schema.AgreementPolicy) -> None:
    configurations = {first.configuration_id for first, _ in aligned}
    strata = {first.stratum for first, _ in aligned}
    if configurations != policy.margins.keys():
        raise ValueError("observed configurations differ from required margin domains")
    if strata != policy.required_strata.keys():
        raise ValueError("observed strata differ from required policy strata")


def _coverage_failures(
    ranks: Mapping[str, object], orders: Mapping[str, object], coverage: schema.Coverage
) -> list[str]:
    checks = (
        ("samples", ranks["samples"], coverage.min_samples),
        ("informative_sources", orders["informative_sources"], coverage.min_sources),
        ("eligible_pairs", orders["eligible_pairs"], coverage.min_eligible_pairs), (
            "eligible_fraction", orders["eligible_fraction"],
            coverage.min_eligible_fraction
        )
    )
    return [
        f"{name} {actual} below required {minimum}" for name, actual, minimum in checks
        if actual < minimum
    ]


def _limit_failures(
    ranks: Mapping[str, object], orders: Mapping[str, object],
    policy: schema.AgreementPolicy, family: bool
) -> list[str]:
    failures = []
    checks = {} if family else {
        "srcc": policy.limits.srcc_min,
        "tau_b": policy.limits.tau_min
    }
    for name, minimum in checks.items():
        if ranks[name] is None or ranks[name] < minimum:
            failures.append(f"{name} {ranks[name]} below required {minimum}")
    maximum = policy.limits.family_error_max if family else policy.limits.macro_error_max
    for name, limit in (("macro_error", maximum), ("worst_source_error",
                                                   policy.limits.source_error_max)):
        if orders[name] is None or orders[name] > limit:
            failures.append(f"{name} {orders[name]} exceeds permitted {limit}")
    return failures


def _group_result(
    aligned: stats.Aligned, policy: schema.AgreementPolicy, coverage: schema.Coverage,
    *, family: bool = False
) -> dict[str, object]:
    ranks = stats.rank_statistics(aligned)
    orders = stats.order_statistics(aligned, policy.margins)
    failures = _coverage_failures(ranks, orders, coverage)
    failures.extend(_limit_failures(ranks, orders, policy, family))
    return {
        "status": "fail" if failures else "pass",
        "failures": failures,
        "coverage": asdict(coverage),
        "ranks": ranks,
        "ordering": orders
    }


def _grouped(
    aligned: stats.Aligned, field: str
) -> dict[str, list[tuple[schema.ScoreSample, schema.ScoreSample]]]:
    groups: dict[str, list[tuple[schema.ScoreSample,
                                 schema.ScoreSample]]] = defaultdict(list)
    for pair in aligned:
        groups[getattr(pair[0], field)].append(pair)
    return dict(sorted(groups.items()))


def _domain_results(
    aligned: stats.Aligned, policy: schema.AgreementPolicy, field: str, *,
    family: bool = False
) -> dict[str, object]:
    results = {}
    for name, samples in _grouped(aligned, field).items():
        coverage = policy.coverage if family else policy.required_strata[name]
        configurations = {
            configuration: _group_result(pairs, policy, coverage, family=family)
            for configuration, pairs in _grouped(samples, "configuration_id").items()
        }
        failed = any(result["status"] != "pass" for result in configurations.values())
        results[name] = {
            "status": "fail" if failed else "pass",
            "configurations": configurations
        }
    return results


def _ranking_result(aligned: stats.Aligned,
                    policy: schema.AgreementPolicy) -> dict[str, object]:
    configurations = {
        name: _group_result(pairs, policy, policy.coverage)
        for name, pairs in _grouped(aligned, "configuration_id").items()
    }
    strata = _domain_results(aligned, policy, "stratum")
    families = _domain_results(aligned, policy, "distortion_family", family=True)
    groups = {"configurations": configurations, "strata": strata, "families": families}
    failures = [
        f"{kind}/{name}" for kind, results in groups.items()
        for name, result in results.items() if result["status"] != "pass"
    ]
    total_sources = len({first.source_id for first, _ in aligned})
    if total_sources < policy.min_total_sources or len(aligned
                                                       ) < policy.min_total_samples:
        failures.append("total corpus coverage below required sources or samples")
    return {
        "status": "fail" if failures else "pass",
        "failures": failures,
        "global": stats.rank_statistics(aligned),
        "limits": asdict(policy.limits),
        "total_sources": total_sources,
        "min_total_sources": policy.min_total_sources,
        "min_total_samples": policy.min_total_samples,
        **groups
    }


def _numerical_result(aligned: stats.Aligned,
                      policy: schema.AgreementPolicy) -> dict[str, object]:
    diagnostics = stats.numerical_diagnostics(aligned)
    configured = {"max_error": policy.numerical_max, "mae": policy.numerical_mean}
    thresholds = {
        name: maximum
        for name, maximum in configured.items() if maximum is not None
    }
    failures = [
        name for name, maximum in thresholds.items()
        if diagnostics[name] is None or diagnostics[name] > maximum
    ]
    status = "not_requested" if not thresholds else ("fail" if failures else "pass")
    return {
        "status": status,
        "failures": failures,
        "thresholds": thresholds,
        **diagnostics
    }


def _provenance(report: schema.ScoreReport) -> dict[str, object]:
    values = asdict(report)
    del values["samples"]
    return values


def _proposed_status(
    ranking: Mapping[str, object], subjective: Mapping[str, object]
) -> str:
    if ranking["status"] == "fail" or subjective["status"] == "fail":
        return "fail"
    return "inconclusive" if subjective["status"] == "inconclusive" else "pass"


def evaluate_agreement(
    reference: Mapping[str, object], candidate: Mapping[str, object],
    policy: Mapping[str, object]
) -> dict[str, object]:
    """Evaluate direct agreement, subjective noninferiority and old numerical gates.

    Invalid provenance, missing samples, nonfinite scores and undeclared domains
    raise errors. Well-formed but insufficient or constant evidence cannot pass.
    Numerical parity is reported separately and does not change proposed_status.

    :param reference: Canonical report from the pinned original.
    :param candidate: Canonical report from the candidate implementation.
    :param policy: Frozen metric/configuration margins, coverage and thresholds.
    :returns: Strict JSON-compatible evidence, including separate gate statuses.
    :raises ValueError: If schemas, provenance or sample identities do not match.
    """
    original = schema.ScoreReport.parse(reference)
    observed = schema.ScoreReport.parse(candidate)
    settings = schema.AgreementPolicy.parse(policy)
    aligned = _aligned_samples(original, observed, settings)
    ranking = _ranking_result(aligned, settings)
    subjective = stats.subjective_statistics(aligned, settings)
    encoded_policy = json.dumps(dict(policy), sort_keys=True, allow_nan=False).encode()
    return {
        "schema_version": 1,
        "metric": settings.metric,
        "sample_count": len(aligned),
        "policy_sha256": hashlib.sha256(encoded_policy).hexdigest(),
        "policy": dict(policy),
        "reference": _provenance(original),
        "candidate": _provenance(observed),
        "ranking": ranking,
        "numerical": _numerical_result(aligned, settings),
        "subjective": subjective,
        "proposed_status": _proposed_status(ranking, subjective)
    }


def _csv_score(row: Mapping[str, str], sample_fields: set[str]) -> dict[str, object]:
    sample: dict[str, object] = {
        key: row[key]
        for key in sample_fields if key != "score"
    }
    sample["score"] = float(row["score"])
    if row.get("mos", "") != "":
        sample["mos"] = float(row["mos"])
    return sample


def _csv_report(path: Path) -> dict[str, object]:
    metadata = {
        "metric", "implementation", "revision", "configuration_hash",
        "input_manifest_hash"
    }
    sample_fields = {
        "sample_id", "source_id", "configuration_id", "stratum", "distortion_family",
        "input_hash", "score"
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)
                              ) or not metadata | sample_fields <= set(fields):
            raise ValueError("CSV needs unique canonical metadata and sample columns")
        if set(fields) - metadata - sample_fields - {"mos"}:
            raise ValueError("unknown CSV report columns")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV report must contain samples")
    if any(None in row or None in row.values() for row in rows):
        raise ValueError("CSV row has missing or extra cells")
    provenance = {key: rows[0][key] for key in metadata}
    if any(any(row[key] != value for key, value in provenance.items()) for row in rows):
        raise ValueError("CSV report provenance changes between rows")
    return report_from_rows(
        [_csv_score(row, sample_fields) for row in rows], **provenance
    )


def load_report(path: str | Path) -> dict[str, object]:
    """Load a canonical JSON or CSV report and validate its schema.

    :param path: JSON report or CSV with repeated provenance columns.
    :returns: A validated canonical report.
    :raises ValueError: If the report schema or CSV metadata is invalid.
    :raises OSError: If the file cannot be read.
    """
    source = Path(path)
    result = _csv_report(source) if source.suffix.lower() == ".csv" else schema.mapping(
        json.loads(source.read_text(encoding="utf-8")), "report"
    )
    schema.ScoreReport.parse(result)
    return result


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require", choices=("ranking", "numerical", "both"), help=
        "gate exit status (default: ranking for registered qualified domains, both otherwise); ranking also requires requested subjective noninferiority"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone agreement evaluator with explicit old/new gate output.

    :param argv: Command-line arguments, or None for sys.argv.
    :returns: Zero for required passing gates, one for qualification failure, two for bad input.
    """
    args = _arguments(argv)
    try:
        policy = schema.mapping(
            json.loads(args.policy.read_text(encoding="utf-8")), "policy"
        )
        original = load_report(args.reference)
        result = evaluate_agreement(original, load_report(args.candidate), policy)
        args.require = args.require or default_gate_mode(original, policy)
        result["gate_mode"] = args.require
        rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"agreement: {error}", file=sys.stderr)
        return 2
    print(f"reference ranking agreement: {result['ranking']['status'].upper()}")
    print(f"subjective noninferiority: {result['subjective']['status'].upper()}")
    print(f"old numerical parity: {result['numerical']['status'].upper()}")
    required = []
    if args.require in ("ranking", "both"):
        required.append(result["proposed_status"])
    if args.require in ("numerical", "both"):
        required.append(result["numerical"]["status"])
    return 0 if all(status == "pass" for status in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
