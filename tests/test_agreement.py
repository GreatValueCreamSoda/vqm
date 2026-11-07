#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Test ranking with harmless remappings and known failures."""

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

from validation.agreement import evaluate_agreement
from validation.agreement import load_report
from validation.agreement import report_from_rows


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _rows(sources: int = 3) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"source-{source:04}/variant-{variant}",
            "source_id": f"source-{source:04}",
            "configuration_id": "standard",
            "stratum": "native",
            "distortion_family": "noise",
            "input_hash": _digest(f"test input {source}/{variant}"),
            "score": float(1 + 10 * source + 2 * variant)
        } for source in range(sources) for variant in range(5)
    ]


def _report(rows: list[dict[str, object]], *,
            original: bool = False) -> dict[str, object]:
    return report_from_rows(
        rows, metric="test-metric",
        implementation="original" if original else "candidate",
        revision="pinned-original" if original else "candidate-revision",
        configuration_hash=_digest("resolved settings"),
        input_manifest_hash=_digest("frozen manifest")
    )


def _policy() -> dict[str, object]:
    return {
        "schema_version": 1,
        "metric": "test-metric",
        "reference_revision": "pinned-original",
        "margins": {
            "standard": .01
        },
        "required_strata": {
            "native": {}
        },
        "numerical_max": .001,
        "numerical_mean": .001
    }


def _evaluate(
    rows: list[dict[str, object]], transform: Callable[[float], float],
    policy: dict[str, object] | None = None
) -> dict[str, object]:
    candidate = [{**row, "score": transform(row["score"])} for row in rows]
    return evaluate_agreement(
        _report(rows, original=True), _report(candidate),
        _policy() if policy is None else policy
    )


class AgreementInvarianceTests(unittest.TestCase):
    """Accept common monotonic recalibrations and record numerical drift."""

    def test_offset_passes_ranking_and_fails_old_numerical_gate(self) -> None:
        """The requested uniform three-point shift preserves every decision."""
        result = _evaluate(_rows(), lambda value: value + 3)
        self.assertEqual(result["ranking"]["status"], "pass")
        self.assertEqual(result["proposed_status"], "pass")
        self.assertEqual(result["numerical"]["status"], "fail")
        self.assertEqual(result["numerical"]["mean_offset"], 3)
        self.assertEqual(result["numerical"]["affine_slope"], 1)

    def test_positive_scales_and_nonlinear_maps_preserve_rank(self) -> None:
        """Exact output distinctions survive even a very small score scale."""
        transforms = (
            lambda value: value * 1e-12, lambda value: value * 5,
            lambda value: value**3
        )
        for transform in transforms:
            with self.subTest(transform=transform):
                result = _evaluate(_rows(), transform)
                self.assertEqual(result["ranking"]["status"], "pass")
                self.assertEqual(result["ranking"]["global"]["srcc"], 1)

    def test_score_clipping_cannot_hide_collapsed_order(self) -> None:
        """Final returned scores determine ties after remapping and clipping."""
        result = _evaluate(_rows(), lambda value: min(value + 3, 23))
        group = result["ranking"]["configurations"]["standard"]
        self.assertEqual(result["ranking"]["status"], "fail")
        self.assertGreater(group["ordering"]["collapsed_pairs"], 0)
        self.assertGreater(group["ranks"]["candidate_tied_pairs"], 0)

    def test_constant_candidate_or_original_is_uninformative(self) -> None:
        """Undefined correlations never become a perfect-agreement result."""
        for original_constant in (False, True):
            rows = [{
                **row, "score": 1.0
            } for row in _rows()] if original_constant else _rows()
            result = _evaluate(rows, lambda _: 1.0)
            self.assertEqual(result["proposed_status"], "fail")
            self.assertIsNone(result["ranking"]["global"]["srcc"])

    def test_reference_ties_are_reported_without_invented_order(self) -> None:
        """A tied original pair is excluded from clearly ordered comparisons."""
        rows = _rows()
        rows[0]["score"] = rows[1]["score"]
        result = _evaluate(rows, lambda value: value + 3)
        counts = result["ranking"]["configurations"]["standard"]["ordering"]
        self.assertEqual(counts["reference_tied_pairs"], 1)
        self.assertEqual(counts["failures"], 0)
        self.assertEqual(result["ranking"]["status"], "pass")


class AgreementFailureTests(unittest.TestCase):
    """Expose local failures that can disappear in aggregate correlations."""

    def test_catastrophic_source_cannot_hide_in_global_correlation(
        self
    ) -> None:
        """Catch one inverted source among 200 with per-source caps."""
        rows = _rows(200)
        candidate = [
            {
                **row, "score": 10 - row["score"]
            } if row["source_id"] == "source-0000" else dict(row)
            for row in rows
        ]
        result = evaluate_agreement(
            _report(rows, original=True), _report(candidate), _policy()
        )
        self.assertGreater(result["ranking"]["global"]["srcc"], .999)
        group = result["ranking"]["configurations"]["standard"]
        self.assertEqual(group["ordering"]["worst_source_error"], 1)
        self.assertEqual(result["ranking"]["status"], "fail")

    def test_required_stratum_has_its_own_correlation_gate(self) -> None:
        """Reject rare broken paths despite agreement on dominant paths."""
        rows = [
            {
                **row, "stratum":
                "rare" if row["source_id"] == "source-0000" else "native"
            } for row in _rows(200)
        ]
        candidate = [
            {
                **row, "score": 10 - row["score"]
            } if row["stratum"] == "rare" else dict(row) for row in rows
        ]
        policy = _policy()
        policy["required_strata"] = {"native": {}, "rare": {"min_sources": 1}}
        policy["limits"] = {
            "macro_error_max": 1,
            "family_error_max": 1,
            "source_error_max": 1
        }
        result = evaluate_agreement(
            _report(rows, original=True), _report(candidate), policy
        )
        self.assertEqual(
            result["ranking"]["configurations"]["standard"]["status"], "pass"
        )
        self.assertEqual(result["ranking"]["strata"]["rare"]["status"], "fail")

    def test_broad_margin_fails_coverage(self) -> None:
        """Ambiguity exclusion cannot discard the entire decision population."""
        policy = _policy()
        policy["margins"] = {"standard": 1000}
        result = _evaluate(_rows(), lambda value: value, policy)
        group = result["ranking"]["configurations"]["standard"]
        self.assertEqual(group["ordering"]["eligible_pairs"], 0)
        self.assertEqual(result["ranking"]["status"], "fail")

    def test_frozen_reference_margin_does_not_move_with_candidate(self) -> None:
        """Preserve eligible original pairs across calibration changes."""
        first = _evaluate(_rows(), lambda value: value)
        shifted = _evaluate(_rows(), lambda value: value * .0001 + 100)
        key = "standard"
        first_counts = first["ranking"]["configurations"][key]["ordering"]
        shifted_counts = shifted["ranking"]["configurations"][key]["ordering"]
        self.assertEqual(
            first_counts["eligible_pairs"], shifted_counts["eligible_pairs"]
        )


class AgreementSchemaTests(unittest.TestCase):
    """Reject incomplete evidence and mismatched provenance."""

    def test_mismatched_sample_sets_and_duplicates_are_rejected(self) -> None:
        """Reject dropped, renamed or duplicate samples."""
        original = _report(_rows(), original=True)
        for samples in (_rows()[:-1], _rows() + [_rows()[0]]):
            with self.subTest(samples=len(samples)
                              ), self.assertRaises(ValueError):
                evaluate_agreement(
                    original, {
                        **_report(_rows()), "samples": samples
                    }, _policy()
                )

    def test_nonfinite_boolean_and_absent_scores_are_rejected(self) -> None:
        """All compared scores must be present, real and finite."""
        original = _report(_rows(), original=True)
        for value in (float("nan"), float("inf"), -float("inf"), True, None,
                      10**1000):
            with self.subTest(score=value), self.assertRaises(ValueError):
                rows = _rows()
                rows[0]["score"] = value
                evaluate_agreement(
                    original, {
                        **_report(_rows()), "samples": rows
                    }, _policy()
                )
        rows = _rows()
        del rows[0]["score"]
        with self.assertRaises(ValueError):
            evaluate_agreement(
                original, {
                    **_report(_rows()), "samples": rows
                }, _policy()
            )

    def test_input_metadata_mismatch_is_rejected(self) -> None:
        """Require matching inputs and domains as well as sample names."""
        replacements = {
            "input_hash": _digest("wrong pixels"),
            "source_id": "other-source",
            "configuration_id": "other-config",
            "stratum": "other-input",
            "distortion_family": "other-artifact",
            "mos": 3.0
        }
        for field, value in replacements.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                rows = _rows()
                rows[0][field] = value
                evaluate_agreement(
                    _report(_rows(), original=True), _report(rows), _policy()
                )

    def test_report_provenance_and_reference_pin_are_enforced(self) -> None:
        """Reject mismatched settings, manifests, metrics and revisions."""
        original = _report(_rows(), original=True)
        for field in ("configuration_hash", "input_manifest_hash", "metric"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                value = "other-metric" if field == "metric" else _digest(
                    "other provenance"
                )
                evaluate_agreement(
                    original, {
                        **_report(_rows()), field: value
                    }, _policy()
                )
        with self.assertRaisesRegex(ValueError, "revision"):
            evaluate_agreement(
                {
                    **original, "revision": "different-original"
                }, _report(_rows()), _policy()
            )

    def test_required_domains_cannot_be_dropped(self) -> None:
        """Require all frozen configurations and input strata."""
        for field, value in (("margins", {"standard": .01, "other": .01}),
                             ("required_strata", {"native": {}, "missing":
                                                  {}})):
            with self.subTest(field=field), self.assertRaises(ValueError):
                policy = {**_policy(), field: value}
                _evaluate(_rows(), lambda score: score, policy)

    def test_expected_manifest_and_settings_are_policy_pins(self) -> None:
        """Matching reports cannot bypass frozen input provenance."""
        for field in ("configuration_hash", "input_manifest_hash"):
            with self.subTest(field=field
                              ), self.assertRaisesRegex(ValueError,
                                                        "policy pin"):
                policy = {
                    **_policy(), field: _digest("frozen different inputs")
                }
                _evaluate(_rows(), lambda value: value, policy)

    def test_global_source_coverage_is_distinct_from_configuration_coverage(
        self
    ) -> None:
        """Enforce corpus coverage beyond per-domain minima."""
        policy = {
            **_policy(), "min_total_sources": 30,
            "min_total_samples": 150
        }
        result = _evaluate(_rows(), lambda value: value, policy)
        self.assertEqual(
            result["ranking"]["configurations"]["standard"]["status"], "pass"
        )
        self.assertEqual(result["ranking"]["status"], "fail")

    def test_schema_versions_require_integers(self) -> None:
        """Boolean or floating JSON values are not valid schema identifiers."""
        for version in (True, 1.0, "1", 2):
            with self.subTest(version=version), self.assertRaises(ValueError):
                original = {
                    **_report(_rows(), original=True), "schema_version": version
                }
                evaluate_agreement(original, _report(_rows()), _policy())

    def test_policy_schema_and_margin_values_are_validated(self) -> None:
        """Reject misspelled fields and invalid thresholds."""
        invalid = (
            {
                "unknown": 1
            }, {
                "margins": {
                    "standard": -1
                }
            }, {
                "margins": {
                    "standard": float("nan")
                }
            }, {
                "limits": {
                    "srcc_min": 2
                }
            }, {
                "direction": "automatic"
            }
        )
        for replacement in invalid:
            with self.subTest(replacement=replacement
                              ), self.assertRaises(ValueError):
                _evaluate(
                    _rows(), lambda score: score, {
                        **_policy(),
                        **replacement
                    }
                )


class AgreementSubjectiveTests(unittest.TestCase):
    """Keep subjective noninferiority separate from original agreement."""

    def _subjective_policy(self) -> dict[str, object]:
        policy = _policy()
        policy["subjective"] = {"required": True, "resamples": 100, "seed": 137}
        return policy

    def test_offset_has_identical_subjective_agreement(self) -> None:
        """A paired bootstrap preserves the zero delta under an offset."""
        rows = [{**row, "mos": row["score"]} for row in _rows(30)]
        first = _evaluate(
            rows, lambda value: value + 3, self._subjective_policy()
        )
        second = _evaluate(
            rows, lambda value: value + 3, self._subjective_policy()
        )
        self.assertEqual(first["subjective"], second["subjective"])
        result = first["subjective"]["configurations"]["standard"]
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["lower_one_sided_95"], 0)

    def test_lower_is_better_direction_is_applied(self) -> None:
        """Orient distance metrics consistently with higher MOS."""
        rows = [
            {
                **row, "mos": row["score"],
                "score": -row["score"]
            } for row in _rows(30)
        ]
        policy = {**self._subjective_policy(), "direction": "lower"}
        result = _evaluate(rows, lambda value: value + 3, policy)
        observed = result["subjective"]["configurations"]["standard"]
        self.assertEqual(observed["reference_srcc"], 1)
        self.assertEqual(observed["candidate_srcc"], 1)

    def test_subjective_regression_fails(self) -> None:
        """A large paired loss cannot be accepted as nonsignificant change."""
        rows = [{**row, "mos": row["score"]} for row in _rows(30)]
        result = _evaluate(
            rows, lambda value: -value, self._subjective_policy()
        )
        self.assertEqual(result["subjective"]["status"], "fail")

    def test_too_few_sources_or_missing_mos_is_inconclusive(self) -> None:
        """Require source coverage beyond frame or derivative counts."""
        variants = (
            [{
                **row, "mos": row["score"]
            } for row in _rows()], _rows(30)
        )
        for rows in variants:
            with self.subTest(samples=len(rows)):
                result = _evaluate(
                    rows, lambda value: value, self._subjective_policy()
                )
                self.assertEqual(result["subjective"]["status"], "inconclusive")
                self.assertEqual(result["proposed_status"], "inconclusive")

    def test_undefined_resamples_are_not_silently_discarded(self) -> None:
        """Expose single-source uncertainty through source resampling."""
        rows = [
            {
                **row, "mos": float(row["source_id"] == "source-0029")
            } for row in _rows(30)
        ]
        result = _evaluate(
            rows, lambda value: value + 3, self._subjective_policy()
        )
        subjective = result["subjective"]["configurations"]["standard"]
        self.assertGreater(subjective["undefined_resamples"], 0)
        self.assertEqual(subjective["status"], "inconclusive")
        self.assertIsNone(subjective["lower_one_sided_95"])


class AgreementCommandTests(unittest.TestCase):
    """Test JSON/CSV loading and direct/package command execution."""

    def test_canonical_csv_round_trip(self) -> None:
        """CSV reports retain exact sample identity and constant provenance."""
        report = _report(_rows(), original=True)
        metadata = {
            key: value
            for key, value in report.items()
            if key not in ("schema_version", "samples")
        }
        rows = [{**metadata, **sample} for sample in report["samples"]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(load_report(path), report)

    def test_cli_reports_old_and_new_gates(self) -> None:
        """Show raw parity failures beside successful shadow rankings."""
        with tempfile.TemporaryDirectory() as directory:
            paths = {
                name: Path(directory) / f"{name}.json"
                for name in ("original", "candidate", "policy", "result")
            }
            original = _report(_rows(), original=True)
            candidate = _report(
                [{
                    **row, "score": row["score"] + 3
                } for row in _rows()]
            )
            for name, value in (("original", original),
                                ("candidate", candidate), ("policy",
                                                           _policy())):
                paths[name].write_text(json.dumps(value), encoding="utf-8")
            command = [
                sys.executable, "-m", "validation.agreement",
                str(paths["original"]),
                str(paths["candidate"]), "--policy",
                str(paths["policy"]), "--output",
                str(paths["result"])
            ]
            result = subprocess.run(
                command + ["--require", "ranking"], capture_output=True,
                text=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("reference ranking agreement: PASS", result.stdout)
            self.assertIn("old numerical parity: FAIL", result.stdout)
            default = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(default.returncode, 1, default.stderr)
            strict = subprocess.run(
                command + ["--require", "both"], capture_output=True, text=True,
                check=False
            )
            self.assertEqual(strict.returncode, 1, strict.stderr)
            self.assertEqual(
                json.loads(paths["result"].read_text())["proposed_status"],
                "pass"
            )


if __name__ == "__main__":
    unittest.main()
