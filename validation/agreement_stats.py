#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Compute direct ranking, meaningful-order and paired subjective evidence."""

import math
from collections import Counter
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from validation.agreement_policy import AgreementPolicy
from validation.agreement_policy import ScoreSample
from validation.agreement_policy import SubjectivePolicy
from validation.common import kendall_tau_b
from validation.common import pearson
from validation.common import rankdata
from validation.common import spearman

Aligned = Sequence[tuple[ScoreSample, ScoreSample]]


def finite_or_none(value: object) -> float | None:
    """Convert a derived numerical diagnostic into strict JSON.

    :param value: A real scalar diagnostic.
    :returns: A finite float, or None when the diagnostic is undefined.
    """
    result = float(value)
    return result if math.isfinite(result) else None


def _tied_pairs(values: Sequence[object]) -> int:
    return sum(count * (count - 1) // 2 for count in Counter(values).values())


def rank_statistics(aligned: Aligned) -> dict[str, object]:
    """Describe ordinary exact-tie rank agreement without ambiguity margins.

    :param aligned: Original/candidate sample pairs with matching metadata.
    :returns: Rank coefficients and complete all-pair concordance/tie counts.
    """
    reference = [first.score for first, _ in aligned]
    candidate = [second.score for _, second in aligned]
    tau = finite_or_none(kendall_tau_b(reference, candidate))
    total = len(aligned) * (len(aligned) - 1) // 2
    reference_ties = _tied_pairs(reference)
    candidate_ties = _tied_pairs(candidate)
    both_ties = _tied_pairs(list(zip(reference, candidate, strict=True)))
    comparable = total - reference_ties - candidate_ties + both_ties
    signed = None if tau is None else tau * math.sqrt(
        (total - reference_ties) * (total - candidate_ties)
    )
    reversed_pairs = None if signed is None else round(
        (comparable - signed) / 2
    )
    return {
        "samples": len(aligned),
        "srcc": finite_or_none(spearman(reference, candidate)),
        "tau_b": tau,
        "total_pairs": total,
        "concordant_pairs":
        None if reversed_pairs is None else comparable - reversed_pairs,
        "reversed_pairs": reversed_pairs,
        "reference_tied_pairs": reference_ties,
        "candidate_tied_pairs": candidate_ties,
        "both_tied_pairs": both_ties
    }


def numerical_diagnostics(aligned: Aligned) -> dict[str, object]:
    """Report raw errors and affine calibration without correcting scores.

    :param aligned: Original/candidate sample pairs.
    :returns: Maximum/mean/RMS errors, signed offset and affine diagnostics.
    """
    reference = np.array(
        [first.score for first, _ in aligned], dtype=np.longdouble
    )
    candidate = np.array(
        [second.score for _, second in aligned], dtype=np.longdouble
    )
    error = candidate - reference
    centered = reference - reference.mean()
    variance = np.dot(centered, centered)
    slope = None if variance == 0 else np.dot(
        centered, candidate - candidate.mean()
    ) / variance
    intercept = None if slope is None else candidate.mean(
    ) - slope * reference.mean()
    scale_x = max(float(np.max(np.abs(reference))), 1)
    scale_y = max(float(np.max(np.abs(candidate))), 1)
    pcc = pearson(
        np.asarray(reference / scale_x, dtype=float),
        np.asarray(candidate / scale_y, dtype=float)
    )
    return {
        "max_error": finite_or_none(np.max(np.abs(error))),
        "mae": finite_or_none(np.mean(np.abs(error))),
        "rmse": finite_or_none(np.sqrt(np.mean(error * error))),
        "mean_offset": finite_or_none(error.mean()),
        "affine_slope": None if slope is None else finite_or_none(slope),
        "affine_intercept":
        None if intercept is None else finite_or_none(intercept),
        "pearson": finite_or_none(pcc)
    }


def _pair_counts(aligned: Aligned, margin: float) -> dict[str, int]:
    reference = np.array([first.score for first, _ in aligned])
    candidate = np.array([second.score for _, second in aligned])
    left, right = np.triu_indices(len(aligned), 1)
    difference = reference[left] - reference[right]
    candidate_sign = np.sign(candidate[left] - candidate[right])
    eligible = np.abs(difference) > margin
    ordered = np.sign(difference) * candidate_sign
    return {
        "pairs": len(left),
        "eligible_pairs": int(np.sum(eligible)),
        "excluded_pairs": int(np.sum(~eligible)),
        "failures": int(np.sum(eligible & (ordered <= 0))),
        "reversals": int(np.sum(eligible & (ordered < 0))),
        "collapsed_pairs": int(np.sum(eligible & (candidate_sign == 0))),
        "reference_tied_pairs": int(np.sum(difference == 0)),
        "candidate_tied_pairs": int(np.sum(candidate_sign == 0))
    }


def _source_counts(aligned: Aligned, margins: dict[str,
                                                   float]) -> dict[str, int]:
    configurations: dict[str, list[tuple[ScoreSample,
                                         ScoreSample]]] = defaultdict(list)
    for pair in aligned:
        configurations[pair[0].configuration_id].append(pair)
    counts = [
        _pair_counts(pairs, margins[name])
        for name, pairs in configurations.items()
    ]
    return {key: sum(count[key] for count in counts) for key in counts[0]}


def order_statistics(aligned: Aligned,
                     margins: dict[str, float]) -> dict[str, object]:
    """Count meaningful failures with equal weight per informative source.

    Comparisons never cross original sources or qualification configurations.
    A candidate tie means exact equality of returned scores.

    :param aligned: Original/candidate sample pairs.
    :param margins: Frozen original-score margin for every configuration.
    :returns: Aggregate counts, source rates, exclusions and worst-source rate.
    """
    sources: dict[str, list[tuple[ScoreSample,
                                  ScoreSample]]] = defaultdict(list)
    for pair in aligned:
        sources[pair[0].source_id].append(pair)
    counts = {
        source: _source_counts(pairs, margins)
        for source, pairs in sources.items()
    }
    rates = {
        source: count["failures"] / count["eligible_pairs"]
        for source, count in counts.items() if count["eligible_pairs"]
    }
    totals = {
        key: sum(count[key] for count in counts.values())
        for key in next(iter(counts.values()))
    }
    return {
        **totals, "sources":
        len(sources),
        "informative_sources":
        len(rates),
        "eligible_fraction":
        totals["eligible_pairs"] / totals["pairs"] if totals["pairs"] else 0.0,
        "macro_error":
        float(np.mean(list(rates.values()))) if rates else None,
        "worst_source_error":
        max(rates.values()) if rates else None,
        "source_errors":
        rates,
        "source_counts":
        counts,
        "uninformative_sources":
        sorted(set(sources) - rates.keys())
    }


def _subjective_arrays(aligned: Aligned,
                       policy: AgreementPolicy) -> tuple[np.ndarray, ...]:
    direction = 1 if policy.direction == "higher" else -1
    mos_direction = 1 if policy.mos_direction == "higher" else -1
    return (
        np.array([first.score * direction for first, _ in aligned]),
        np.array([second.score * direction for _, second in aligned]),
        np.array([first.mos * mos_direction for first, _ in aligned])
    )


def _subjective_correlations(
    arrays: tuple[np.ndarray, ...]
) -> tuple[float, float]:
    reference, candidate, mos = arrays
    mos_ranks = rankdata(mos)
    return pearson(rankdata(reference),
                   mos_ranks), pearson(rankdata(candidate), mos_ranks)


def _bootstrap_deltas(
    arrays: tuple[np.ndarray, ...], clusters: list[np.ndarray],
    policy: SubjectivePolicy
) -> tuple[list[float], int]:
    rng = np.random.default_rng(policy.seed)
    deltas = []
    undefined = 0
    for _ in range(policy.resamples):
        selected = rng.integers(0, len(clusters), len(clusters))
        indices = np.concatenate([clusters[index] for index in selected])
        original, candidate = _subjective_correlations(
            tuple(array[indices] for array in arrays)
        )
        if math.isfinite(original) and math.isfinite(candidate):
            deltas.append(candidate - original)
        else:
            undefined += 1
    return deltas, undefined


def _subjective_configuration(aligned: Aligned,
                              policy: AgreementPolicy) -> dict[str, object]:
    sources: dict[str, list[int]] = defaultdict(list)
    for index, (sample, _) in enumerate(aligned):
        sources[sample.source_id].append(index)
    if len(sources) < policy.subjective.min_sources:
        return {
            "status": "inconclusive",
            "reason": "too few independent sources",
            "sources": len(sources),
            "required_sources": policy.subjective.min_sources
        }
    if any(first.mos is None for first, _ in aligned):
        return {"status": "inconclusive", "reason": "missing MOS observations"}
    arrays = _subjective_arrays(aligned, policy)
    original, candidate = _subjective_correlations(arrays)
    if not math.isfinite(original) or not math.isfinite(candidate):
        return {
            "status": "inconclusive",
            "reason": "undefined observed subjective correlation"
        }
    clusters = [np.array(sources[name]) for name in sorted(sources)]
    deltas, undefined = _bootstrap_deltas(arrays, clusters, policy.subjective)
    lower = None if undefined else float(
        np.quantile(deltas, .05, method="linear")
    )
    status = "inconclusive" if lower is None else (
        "pass" if lower >= -policy.subjective.margin else "fail"
    )
    return {
        "status":
        status,
        "reference_srcc":
        original,
        "candidate_srcc":
        candidate,
        "delta_srcc":
        candidate - original,
        "lower_one_sided_95":
        lower,
        "margin":
        policy.subjective.margin,
        "sources":
        len(sources),
        "resamples":
        policy.subjective.resamples,
        "undefined_resamples":
        undefined,
        "seed":
        policy.subjective.seed,
        "statistic":
        "pooled sample SRCC within configuration",
        "interval": (
            "5th percentile of paired whole-source bootstrap deltas; linear "
            "quantile"
        )
    }


def subjective_statistics(aligned: Aligned,
                          policy: AgreementPolicy) -> dict[str, object]:
    """Evaluate paired MOS noninferiority separately for every configuration.

    Any undefined resample makes that configuration inconclusive; no resamples
    are silently discarded. Whole original sources, including all derivatives,
    are sampled with replacement using the same draw for both implementations.

    :param aligned: Original/candidate sample pairs with authoritative original
        MOS.
    :param policy: Frozen directions and bootstrap settings.
    :returns: Per-configuration evidence and pass/fail/inconclusive status.
    """
    if not policy.subjective.required:
        return {"status": "not_requested"}
    configurations: dict[str, list[tuple[ScoreSample,
                                         ScoreSample]]] = defaultdict(list)
    for pair in aligned:
        configurations[pair[0].configuration_id].append(pair)
    results = {
        name: _subjective_configuration(pairs, policy)
        for name, pairs in sorted(configurations.items())
    }
    statuses = {result["status"] for result in results.values()}
    status = "fail" if "fail" in statuses else (
        "inconclusive" if "inconclusive" in statuses else "pass"
    )
    return {"status": status, "configurations": results}
