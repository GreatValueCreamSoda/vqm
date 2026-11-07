#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Keep ranking defaults confined to the qualified original/policy domains."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from itertools import product
from pathlib import Path
from unittest import mock

from validation.agreement import default_gate_mode
from validation.agreement_policy import AgreementPolicy

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("ssimulacra2", "butteraugli", "vmaf-v0", "vmaf-v1", "cvvdp")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _cases() -> list[tuple[str, dict[str, object], dict[str, object]]]:
    result = []
    base = ROOT / "validation/temporal/oracles"
    for profile in ("smoke", "qualification"):
        for metric in METRICS:
            directory = base / profile
            result.append(
                (
                    f"{profile}/{metric}", _read(directory / f"{metric}.json"),
                    _read(directory / f"{metric}-policy.json")
                )
            )
    base = ROOT / "validation/fixtures/cid22-balanced"
    return result + [
        (
            f"cid22/{metric}", _read(base / f"{metric}-original.json"
                                     ), _read(base / f"{metric}-policy.json")
        ) for metric in ("ssimulacra2", "butteraugli")
    ]


class QualifiedDefaultTests(unittest.TestCase):
    """Resolve complete frozen domains without depending on their filesystem paths."""

    def test_all_twelve_qualified_profiles_default_to_ranking(self) -> None:
        """Both hosted profiles and both balanced still policies are promoted."""
        cases = _cases()
        self.assertEqual(len(cases), 12)
        for name, reference, policy in cases:
            with self.subTest(domain=name):
                self.assertEqual(default_gate_mode(reference, policy), "ranking")

    def test_implicit_policy_defaults_cannot_change_under_an_approved_policy(
        self
    ) -> None:
        """A schema default change requires qualification even when JSON stays the same."""
        _, reference, policy = _cases()[0]
        settings = AgreementPolicy.parse(policy)
        changed = replace(settings, limits=replace(settings.limits, srcc_min=.9))
        with mock.patch.object(AgreementPolicy, "parse", return_value=changed):
            self.assertEqual(default_gate_mode(reference, policy), "both")

    def test_key_order_and_reference_sample_order_do_not_change_qualification(
        self
    ) -> None:
        """Canonical identity survives JSON presentation and whole-report sorting."""
        for name, reference, policy in _cases():
            with self.subTest(domain=name):
                reordered = dict(reversed(tuple(reference.items())))
                reordered["samples"] = list(reversed(reference["samples"]))
                settings = dict(reversed(tuple(policy.items())))
                self.assertEqual(default_gate_mode(reordered, settings), "ranking")

    def test_policy_changes_do_not_inherit_the_qualified_default(self) -> None:
        """New margins, provenance, thresholds and subjective rules need new evidence."""
        for name, reference, policy in _cases():
            self._check_policy_changes(name, reference, policy)

    def _check_policy_changes(
        self, name: str, reference: dict[str, object], policy: dict[str, object]
    ) -> None:
        changes = {
            "metric": "unqualified-metric",
            "reference_revision": "unqualified-original",
            "input_manifest_hash": _digest("unqualified-inputs"),
            "numerical_max": float(policy["numerical_max"]) + 1,
            "min_total_samples": int(policy["min_total_samples"]) - 1,
            "subjective": {
                "required": False,
                "margin": .5
            },
            "margins": {
                key: float(value) * 2
                for key, value in policy["margins"].items()
            }
        }
        for field, value in changes.items():
            with self.subTest(domain=name, field=field):
                changed = dict(policy, **{field: value})
                self.assertEqual(default_gate_mode(reference, changed), "both")

    def test_reference_provenance_cannot_reuse_a_registered_policy(self) -> None:
        """Copying an approved policy cannot qualify a different metric or input manifest."""
        changes = {
            "metric": "unqualified-metric",
            "revision": "unqualified-original",
            "input_manifest_hash": _digest("different-samples")
        }
        for (name, reference, policy), (field, value) in product(_cases(),
                                                                 changes.items()):
            with self.subTest(domain=name, field=field):
                changed = dict(reference, **{field: value})
                self.assertEqual(default_gate_mode(changed, policy), "both")

    def test_original_scores_cannot_be_replaced_under_the_same_declared_hash(
        self
    ) -> None:
        """Claiming the old manifest does not approve a different original score vector."""
        for name, reference, policy in _cases():
            with self.subTest(domain=name):
                changed = copy.deepcopy(reference)
                changed["samples"][0]["score"] += .5
                self.assertEqual(default_gate_mode(changed, policy), "both")

    def test_sample_metadata_cannot_be_replaced_under_the_same_declared_hash(
        self
    ) -> None:
        """Input IDs, domains, families and per-sample hashes remain independently bound."""
        fields = {
            "sample_id": "different-sample",
            "source_id": "different-source",
            "configuration_id": "different-configuration",
            "stratum": "different-stratum",
            "distortion_family": "different-family",
            "input_hash": _digest("different-input-bytes")
        }
        for name, reference, policy in _cases():
            self._check_sample_changes(name, reference, policy, fields)

    def _check_sample_changes(
        self, name: str, reference: dict[str, object], policy: dict[str, object],
        changes: dict[str, str]
    ) -> None:
        for field, value in changes.items():
            with self.subTest(domain=name, field=field):
                changed = copy.deepcopy(reference)
                changed["samples"][0][field] = value
                self.assertEqual(default_gate_mode(changed, policy), "both")

    def test_missing_or_additional_samples_need_new_qualification(self) -> None:
        """A partial or enlarged corpus cannot inherit the complete frozen-domain default."""
        for name, reference, policy in _cases():
            with self.subTest(domain=name):
                changed = copy.deepcopy(reference)
                changed["samples"] = changed["samples"][:-1]
                self.assertEqual(default_gate_mode(changed, policy), "both")
                extra = copy.deepcopy(reference["samples"][0])
                extra["sample_id"] += "/additional"
                changed["samples"] = [*reference["samples"], extra]
                self.assertEqual(default_gate_mode(changed, policy), "both")

    def test_cid22_configuration_fingerprint_may_change_with_a_binary_rebuild(
        self
    ) -> None:
        """Balanced still policies bind logical settings and inputs rather than one executable."""
        for name, reference, policy in _cases():
            if not name.startswith("cid22/"):
                continue
            with self.subTest(domain=name):
                self.assertNotIn("configuration_hash", policy)
                changed = dict(reference, configuration_hash=_digest("rebuilt-binary"))
                self.assertEqual(default_gate_mode(changed, policy), "ranking")

    def test_temporal_configuration_fingerprint_cannot_change(self) -> None:
        """The hosted qualification binds exact conversion, pooling, model and timing settings."""
        for name, reference, policy in _cases():
            if name.startswith("cid22/"):
                continue
            with self.subTest(domain=name):
                changed = dict(
                    reference, configuration_hash=_digest("different-settings")
                )
                self.assertEqual(default_gate_mode(changed, policy), "both")

    def test_changed_still_mos_labels_cannot_reuse_the_qualified_default(self) -> None:
        """The approved human-label noninferiority contract keeps its original annotations."""
        for name, reference, policy in _cases():
            if not name.startswith("cid22/"):
                continue
            with self.subTest(domain=name):
                changed = copy.deepcopy(reference)
                changed["samples"][0]["mos"] += 1
                self.assertEqual(default_gate_mode(changed, policy), "both")

    def test_unknown_policy_stays_on_both_even_with_a_known_metric(self) -> None:
        """Metric names alone never qualify alternate models, displays or content."""
        _, reference, policy = _cases()[0]
        changed = dict(policy, input_manifest_hash=_digest("new-domain"))
        observed = dict(reference, input_manifest_hash=changed["input_manifest_hash"])
        self.assertEqual(default_gate_mode(observed, changed), "both")

    def test_default_qualified_cli_accepts_offset_and_keeps_numerical_failure(
        self
    ) -> None:
        """A real command defaults to ranking while explicit numerical requirements still fail."""
        with tempfile.TemporaryDirectory() as temporary:
            command, output = _qualified_command(Path(temporary))
            for mode, expected in ((None, 0), ("ranking", 0), ("both", 1), ("numerical",
                                                                            1)):
                self._check_cli_result(command, output, mode, expected)

    def _check_cli_result(
        self, command: list[str], output: Path, mode: str | None, expected: int
    ) -> None:
        options = [] if mode is None else ["--require", mode]
        with self.subTest(mode=mode):
            result = subprocess.run(
                command + options, capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, expected, result.stderr)
            evidence = _read(output)
            self.assertEqual(evidence["ranking"]["status"], "pass")
            self.assertEqual(evidence["numerical"]["status"], "fail")
            self.assertAlmostEqual(evidence["numerical"]["mean_offset"], 3)
            self.assertEqual(evidence["gate_mode"], mode or "ranking")
            self.assertIn("old numerical parity: FAIL", result.stdout)

    def test_default_qualified_cli_rejects_reversed_order(self) -> None:
        """Retiring point-score acceptance does not accept a broken ordering."""
        with tempfile.TemporaryDirectory() as temporary:
            command, output = _qualified_command(Path(temporary), reverse=True)
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            evidence = _read(output)
            self.assertEqual(evidence["gate_mode"], "ranking")
            self.assertEqual(evidence["ranking"]["status"], "fail")
            self.assertEqual(evidence["numerical"]["status"], "fail")

    def test_default_unqualified_cli_keeps_the_numerical_conjunction(self) -> None:
        """An altered policy does not inherit ranking acceptance from a metric name."""
        with tempfile.TemporaryDirectory() as temporary:
            command, output = _qualified_command(Path(temporary), unqualified=True)
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            evidence = _read(output)
            self.assertEqual(evidence["gate_mode"], "both")
            self.assertEqual(evidence["ranking"]["status"], "pass")
            self.assertEqual(evidence["numerical"]["status"], "fail")


def _qualified_command(
    directory: Path, *, reverse: bool = False, unqualified: bool = False
) -> tuple[list[str], Path]:
    _, reference, policy = _cases()[0]
    if unqualified:
        policy["numerical_max"] *= 2
    candidate = copy.deepcopy(reference)
    candidate.update(implementation="offset-test-candidate", revision="test-build")
    for sample in candidate["samples"]:
        sample["score"] = -sample["score"] if reverse else sample["score"] + 3
    paths = {
        name: directory / f"{name}.json"
        for name in ("reference", "candidate", "policy", "result")
    }
    for name, value in (("reference", reference), ("candidate", candidate), ("policy",
                                                                             policy)):
        paths[name].write_text(json.dumps(value))
    return [
        sys.executable, "-m", "validation.agreement",
        str(paths["reference"]),
        str(paths["candidate"]), "--policy",
        str(paths["policy"]), "--output",
        str(paths["result"])
    ], paths["result"]


if __name__ == "__main__":
    unittest.main()
