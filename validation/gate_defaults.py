#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Select ranking defaults only for independently qualified frozen domains."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

try:
    from . import agreement_policy as schema
except ImportError:
    import agreement_policy as schema

REGISTRY = Path(__file__).with_name("qualified_domains.json")


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _identity(reference: Mapping[str, object],
              policy: Mapping[str, object]) -> dict[str, object]:
    original = schema.ScoreReport.parse(reference)
    settings = schema.AgreementPolicy.parse(policy)
    samples = sorted(reference["samples"], key=lambda row: row["sample_id"])
    return {
        "metric":
        original.metric,
        "reference_revision":
        original.revision,
        "input_manifest_hash":
        original.input_manifest_hash,
        "configuration_hash":
        original.configuration_hash
        if settings.configuration_hash is not None else None,
        "reference_samples_sha256":
        _digest(samples),
        "policy_sha256":
        _digest(dict(policy)),
        "effective_policy_sha256":
        _digest(asdict(settings))
    }


def default_gate_mode(
    reference: Mapping[str, object], policy: Mapping[str, object]
) -> str:
    """Use ranking for a registered domain and both gates for other domains.

    The registry binds the complete original scores and sample identities as
    well as the frozen policy. Editing thresholds or reusing a declared input
    hash with different samples cannot silently retire numerical acceptance.
    CID22 policies permit a changing global configuration fingerprint because
    it includes candidate binary hashes; their original samples stay pinned.
    Temporal policies additionally bind their resolved configuration hash.
    Callers still evaluate provenance, coverage, ranking and subjective gates.

    :param reference: Canonical report from the pinned original.
    :param policy: Frozen metric/configuration policy.
    :returns: Ranking for an exact qualified identity, otherwise both.
    :raises ValueError: If report, policy or registry schemas are invalid.
    :raises OSError: If the maintained qualification registry cannot be read.
    """
    identity = _identity(reference, policy)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1 or not isinstance(registry.get("domains"),
                                                             list):
        raise ValueError("qualified-domain registry must have schema version 1")
    approved = any(entry["identity"] == identity for entry in registry["domains"])
    return "ranking" if approved else "both"
