#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Validate frozen policies and canonical score records for agreement tests."""

import math
import re
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass


def mapping(value: object, name: str) -> dict[str, object]:
    """Require a JSON object with string keys.

    :param value: Object to validate.
    :param name: Name included in errors.
    :returns: A shallow dictionary copy.
    :raises ValueError: If the value is not a string-keyed mapping.
    """
    if not isinstance(value, Mapping) or any(not isinstance(key, str)
                                             for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return dict(value)


def _keys(
    values: Mapping[str, object], required: set[str], optional: set[str]
) -> None:
    missing = required - values.keys()
    unknown = values.keys() - required - optional
    if missing or unknown:
        raise ValueError(
            f"schema keys: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def text_field(values: Mapping[str, object], key: str) -> str:
    """Read a required nonempty text field.

    :param values: Source mapping.
    :param key: Field name.
    :returns: The validated text.
    :raises ValueError: If the field is absent or empty.
    """
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be nonempty text")
    return value


def _number(values: Mapping[str, object], key: str, default: float) -> float:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{key} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{key} must be a finite number")
    return result


def _bounded(
    values: Mapping[str, object], key: str, default: float, low: float,
    high: float
) -> float:
    value = _number(values, key, default)
    if not low <= value <= high:
        raise ValueError(f"{key} must be in [{low}, {high}]")
    return value


def _integer(
    values: Mapping[str, object], key: str, default: int, minimum: int = 1
) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _hash(values: Mapping[str, object], key: str) -> str:
    value = text_field(values, key)
    if re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{key} must be a SHA-256 hexadecimal digest")
    return value.lower()


@dataclass(frozen=True)
class ScoreSample:
    """One scored image pair or whole clip with stable input provenance."""

    sample_id: str
    source_id: str
    configuration_id: str
    stratum: str
    distortion_family: str
    input_hash: str
    score: float
    mos: float | None = None

    @classmethod
    def parse(cls, value: object) -> "ScoreSample":
        """Validate one canonical sample.

        :param value: Sample mapping.
        :returns: The validated sample.
        :raises ValueError: If metadata or scores are invalid.
        """
        values = mapping(value, "sample")
        fields = {
            "sample_id", "source_id", "configuration_id", "stratum",
            "distortion_family", "input_hash", "score"
        }
        _keys(values, fields, {"mos"})
        mos = None if "mos" not in values else _number(values, "mos", 0)
        return cls(
            text_field(values, "sample_id"), text_field(values, "source_id"),
            text_field(values,
                       "configuration_id"), text_field(values, "stratum"),
            text_field(values, "distortion_family"),
            _hash(values, "input_hash"), _number(values, "score", 0), mos
        )


@dataclass(frozen=True)
class ScoreReport:
    """A complete scored manifest from one implementation revision."""

    metric: str
    implementation: str
    revision: str
    configuration_hash: str
    input_manifest_hash: str
    samples: tuple[ScoreSample, ...]

    @classmethod
    def parse(cls, value: object) -> "ScoreReport":
        """Validate a report and reject duplicate or absent samples.

        :param value: Canonical report mapping.
        :returns: The validated report.
        :raises ValueError: If schema, provenance or samples are invalid.
        """
        values = mapping(value, "report")
        _keys(
            values, {
                "schema_version", "metric", "implementation", "revision",
                "configuration_hash", "input_manifest_hash", "samples"
            }, set()
        )
        if (not isinstance(values["schema_version"], int)
                or isinstance(values["schema_version"], bool)
                or values["schema_version"] != 1):
            raise ValueError("report schema_version must be 1")
        raw = values["samples"]
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("report samples must be a nonempty array")
        samples = tuple(ScoreSample.parse(sample) for sample in raw)
        if len({sample.sample_id for sample in samples}) != len(samples):
            raise ValueError("duplicate sample_id")
        return cls(
            text_field(values, "metric"), text_field(values, "implementation"),
            text_field(values, "revision"), _hash(values, "configuration_hash"),
            _hash(values, "input_manifest_hash"), samples
        )


@dataclass(frozen=True)
class Limits:
    """Fixed ranking and meaningful-comparison acceptance limits."""

    srcc_min: float = .999
    tau_min: float = .99
    macro_error_max: float = .005
    family_error_max: float = .01
    source_error_max: float = .05

    @classmethod
    def parse(cls, value: object) -> "Limits":
        """Read optional overrides for the documented initial limits.

        :param value: Limit mapping.
        :returns: Validated limits.
        :raises ValueError: If keys or bounds are invalid.
        """
        values = mapping(value, "limits")
        _keys(values, set(), set(cls.__dataclass_fields__))
        return cls(
            **{
                key: _bounded(values, key, field.default, 0, 1)
                for key, field in cls.__dataclass_fields__.items()
            }
        )


@dataclass(frozen=True)
class Coverage:
    """Minimum fixed-corpus evidence for a configuration or input stratum."""

    min_sources: int = 2
    min_samples: int = 4
    min_eligible_pairs: int = 2
    min_eligible_fraction: float = .1

    @classmethod
    def parse(cls, value: object) -> "Coverage":
        """Validate coverage, including a minimum eligible-pair fraction.

        :param value: Coverage mapping.
        :returns: Validated coverage.
        :raises ValueError: If counts or fractions are invalid.
        """
        values = mapping(value, "coverage")
        _keys(values, set(), set(cls.__dataclass_fields__))
        return cls(
            _integer(values, "min_sources", 2),
            _integer(values, "min_samples", 4),
            _integer(values, "min_eligible_pairs", 2),
            _bounded(values, "min_eligible_fraction", .1, 0, 1)
        )


@dataclass(frozen=True)
class SubjectivePolicy:
    """Paired whole-source bootstrap noninferiority settings."""

    required: bool = False
    margin: float = .005
    min_sources: int = 30
    resamples: int = 10000
    seed: int = 0

    @classmethod
    def parse(cls, value: object) -> "SubjectivePolicy":
        """Validate subjective settings and forbid frame-level resampling.

        :param value: Settings mapping.
        :returns: Validated settings.
        :raises ValueError: If settings are invalid.
        """
        values = mapping(value, "subjective")
        _keys(values, set(), set(cls.__dataclass_fields__))
        required = values.get("required", False)
        if not isinstance(required, bool):
            raise ValueError("subjective.required must be boolean")
        return cls(
            required, _bounded(values, "margin", .005, 0, 2),
            _integer(values, "min_sources", 30, 2),
            _integer(values, "resamples", 10000, 100),
            _integer(values, "seed", 0, 0)
        )


@dataclass(frozen=True)
class AgreementPolicy:
    """Frozen policy for one metric and its declared qualification domains."""

    metric: str
    reference_revision: str
    margins: dict[str, float]
    required_strata: dict[str, Coverage]
    coverage: Coverage
    limits: Limits
    subjective: SubjectivePolicy
    direction: str = "higher"
    mos_direction: str = "higher"
    numerical_max: float | None = None
    numerical_mean: float | None = None
    min_total_sources: int = 2
    min_total_samples: int = 4
    configuration_hash: str | None = None
    input_manifest_hash: str | None = None

    @classmethod
    def parse(cls, value: object) -> "AgreementPolicy":
        """Read an explicit metric-specific policy.

        :param value: Policy mapping.
        :returns: The validated frozen policy.
        :raises ValueError: If required domains or policy fields are invalid.
        """
        values = mapping(value, "policy")
        _keys(
            values, {
                "schema_version", "metric", "reference_revision", "margins",
                "required_strata"
            }, {
                "coverage", "limits", "subjective", "direction",
                "mos_direction", "numerical_max", "numerical_mean",
                "min_total_sources", "min_total_samples", "configuration_hash",
                "input_manifest_hash"
            }
        )
        if (not isinstance(values["schema_version"], int)
                or isinstance(values["schema_version"], bool)
                or values["schema_version"] != 1):
            raise ValueError("policy schema_version must be 1")
        margins = _margins(values["margins"])
        strata = mapping(values["required_strata"], "required_strata")
        if not strata:
            raise ValueError("required_strata must not be empty")
        return cls(
            text_field(values, "metric"),
            text_field(values, "reference_revision"), margins, {
                name: Coverage.parse(raw)
                for name, raw in strata.items()
            }, Coverage.parse(values.get("coverage", {})),
            Limits.parse(values.get("limits", {})),
            SubjectivePolicy.parse(values.get("subjective", {})),
            _direction(values, "direction"),
            _direction(values, "mos_direction"),
            _optional_nonnegative(values, "numerical_max"),
            _optional_nonnegative(values, "numerical_mean"),
            _integer(values, "min_total_sources", 2),
            _integer(values, "min_total_samples", 4),
            _hash(values, "configuration_hash")
            if "configuration_hash" in values else None,
            _hash(values, "input_manifest_hash")
            if "input_manifest_hash" in values else None
        )


def _direction(values: Mapping[str, object], key: str) -> str:
    result = values.get(key, "higher")
    if result not in ("higher", "lower"):
        raise ValueError(f"{key} must be higher or lower")
    return str(result)


def _optional_nonnegative(
    values: Mapping[str, object], key: str
) -> float | None:
    if key not in values:
        return None
    return _bounded(values, key, 0, 0, float("inf"))


def _margins(value: object) -> dict[str, float]:
    values = mapping(value, "margins")
    if not values or any(not key.strip() for key in values):
        raise ValueError("margins must declare nonempty configuration IDs")
    return {key: _bounded(values, key, 0, 0, float("inf")) for key in values}


def report_from_rows(
    rows: Sequence[Mapping[str, object]], *, metric: str, implementation: str,
    revision: str, configuration_hash: str, input_manifest_hash: str
) -> dict[str, object]:
    """Build and validate a canonical report without inventing provenance.

    :param rows: Canonical sample dictionaries, including input hashes and
        scores.
    :param metric: Metric/model identity shared with the policy.
    :param implementation: Implementation name.
    :param revision: Exact implementation revision or fingerprint.
    :param configuration_hash: SHA-256 of resolved input/metric/decoder
        settings.
    :param input_manifest_hash: SHA-256 of the frozen input manifest.
    :returns: A validated JSON-compatible report dictionary.
    :raises ValueError: If any supplied field or sample is invalid.
    """
    report: dict[str, object] = {
        "schema_version": 1,
        "metric": metric,
        "implementation": implementation,
        "revision": revision,
        "configuration_hash": configuration_hash,
        "input_manifest_hash": input_manifest_hash,
        "samples": [dict(row) for row in rows]
    }
    ScoreReport.parse(report)
    return report
