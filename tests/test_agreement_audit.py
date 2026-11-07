#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Check frozen-policy sensitivity audits independently of production faults."""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from validation.agreement import report_from_rows
from validation.temporal.audit import main, run_audit


def _report() -> dict[str, object]:
    rows = [
        {
            "sample_id": f"source-{source}/sample-{severity}",
            "source_id": f"source-{source}",
            "configuration_id": "standard",
            "stratum": "native",
            "distortion_family": "noise",
            "input_hash": hashlib.sha256(f"{source}/{severity}".encode()).hexdigest(),
            "score": float(1 + source * 10 + severity)
        } for source in range(3) for severity in range(4)
    ]
    return report_from_rows(
        rows, metric="test", implementation="original", revision="pinned",
        configuration_hash="a" * 64, input_manifest_hash="b" * 64
    )


def _policy() -> dict[str, object]:
    return {
        "schema_version": 1,
        "metric": "test",
        "reference_revision": "pinned",
        "margins": {
            "standard": .01
        },
        "required_strata": {
            "native": {}
        },
        "numerical_max": .001,
        "numerical_mean": .001
    }


class AgreementAuditTests(unittest.TestCase):
    """Verify mutation expectations, failure reporting and portable audit artifacts."""

    def test_remaps_pass_and_wrong_associations_fail(self) -> None:
        """A common score offset and nonlinear scale survive the ranking policy."""
        report = _report()
        before = json.dumps(report, sort_keys=True)
        result = run_audit(report, _policy())
        self.assertTrue(result["passed"])
        audits = {row["mutation"]: row for row in result["audits"]}
        self.assertEqual(audits["offset-plus-three"]["ranking"], "pass")
        self.assertEqual(audits["offset-plus-three"]["numerical"], "fail")
        self.assertEqual(audits["offset-plus-three"]["mean_offset"], 3)
        self.assertEqual(audits["increasing-cube"]["ranking"], "pass")
        self.assertEqual(audits["rotated-association"]["ranking"], "fail")
        self.assertEqual(audits["constant-collapse"]["ranking"], "fail")
        self.assertEqual(json.dumps(report, sort_keys=True), before)

    def test_a_missed_fault_is_an_explicit_discrepancy(self) -> None:
        """An overly permissive policy cannot mark its own sensitivity audit passed."""
        policy = _policy()
        policy["limits"] = {
            "srcc_min": 0,
            "tau_min": 0,
            "macro_error_max": 1,
            "family_error_max": 1,
            "source_error_max": 1
        }
        result = run_audit(_report(), policy)
        self.assertFalse(result["passed"])
        self.assertTrue(
            any("rotated-association" in item for item in result["discrepancies"])
        )

    def test_numerical_remapping_failure_is_not_assumed(self) -> None:
        """Large old tolerances remain visible without changing ranking expectations."""
        policy = {**_policy(), "numerical_max": 1e6, "numerical_mean": 1e6}
        result = run_audit(_report(), policy)
        self.assertTrue(result["passed"])
        self.assertTrue(all(row["numerical"] == "pass" for row in result["audits"]))

    def test_cli_writes_hashes_and_all_mutation_statuses(self) -> None:
        """The standalone artifact identifies the exact original and policy files."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, policy, output = (
                root / name for name in ("original.json", "policy.json", "audit.json")
            )
            original.write_text(json.dumps(_report()))
            policy.write_text(json.dumps(_policy()))
            with redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "--reference",
                        str(original), "--policy",
                        str(policy), "--output",
                        str(output)
                    ]
                )
            self.assertEqual(status, 0)
            result = json.loads(output.read_text())
            self.assertTrue(result["passed"])
            self.assertEqual(len(result["audits"]), 7)
            self.assertEqual(
                result["reference_report_sha256"],
                hashlib.sha256(original.read_bytes()).hexdigest()
            )


if __name__ == "__main__":
    unittest.main()
