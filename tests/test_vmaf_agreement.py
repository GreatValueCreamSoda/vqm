#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Exercise ranking acceptance, CSV provenance binding and CID22 integration."""

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from validation import compare_vmaf_scores as comparison
from validation import vmaf_cid22 as cid

ROOT = Path(__file__).resolve().parent.parent


def _rows(candidate: list[float] | None = None) -> list[dict[str, object]]:
    values = candidate or [13.0, 23.0, 33.0, 43.0]
    return [
        {
            "original": f"original/source{index // 2}.png",
            "distorted":
            f"compressed/source{index // 2}/codec/level{index % 2}.png",
            "original_yuv_sha256": f"{index // 2 + 16:064x}",
            "distorted_yuv_sha256": f"{index + 1:064x}",
            "libvmaf_v0": float(10 * (index + 1)),
            "vqm_v0": values[index],
            "mcos": float(10 * (index + 1))
        } for index in range(4)
    ]


def _policy() -> dict[str, object]:
    return {
        "schema_version": 1,
        "metric": "vmaf-v0",
        "reference_revision": cid.REFERENCE_REVISION,
        "margins": {
            cid.AGREEMENT_CONFIGURATION: .01
        },
        "required_strata": {
            "still-sdr": {
                "min_sources": 2,
                "min_samples": 4,
                "min_eligible_pairs": 2
            }
        }
    }


def _fixture(directory: Path,
             candidate: list[float] | None = None) -> list[str]:
    rows = _rows(candidate)
    target = directory / "scores.csv"
    with target.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    reference, proposed = cid.build_agreement_reports(
        rows, "v0", "a" * 64, "b" * 64, "candidate-revision"
    )
    for name, value in (("reference", reference), ("candidate", proposed),
                        ("policy", _policy())):
        (directory / f"{name}.json").write_text(json.dumps(value))
    return [
        "compare_vmaf_scores.py",
        str(target),
        str(target), "--reference-column", "libvmaf_v0", "--candidate-column",
        "vqm_v0", "--agreement-policy",
        str(directory / "policy.json"), "--reference-report",
        str(directory / "reference.json"), "--candidate-report",
        str(directory / "candidate.json")
    ]


def _run(argv: list[str]) -> tuple[int, dict[str, object], str]:
    output, error = io.StringIO(), io.StringIO()
    with (mock.patch.object(sys, "argv",
                            argv), contextlib.redirect_stdout(output),
          contextlib.redirect_stderr(error)):
        status = comparison.main()
    report = json.loads(output.getvalue()) if output.getvalue() else {}
    return status, report, error.getvalue()


class VmafAgreementTests(unittest.TestCase):
    """Check independently requested ranking and numerical CSV gates."""

    def test_uniform_offset_preserves_ranking_but_old_absolute_gate_remains(
        self
    ) -> None:
        """Accept offsets unless explicit numerical bounds reject them."""
        with tempfile.TemporaryDirectory() as temporary:
            argv = _fixture(Path(temporary))
            status, report, _ = _run(argv)
            self.assertEqual(status, 0)
            self.assertEqual(report["agreement"]["ranking"]["status"], "pass")
            self.assertEqual(report["max_absolute"], 3)
            self.assertIsNone(report["within_thresholds"])
            status, report, _ = _run([*argv, "--max-absolute", "2"])
            self.assertEqual(status, 1)
            self.assertFalse(report["within_thresholds"])
            self.assertEqual(report["agreement"]["ranking"]["status"], "pass")

    def test_reordered_quality_fails_required_ranking(self) -> None:
        """Reject score order changes even with a small complete score table."""
        with tempfile.TemporaryDirectory() as temporary:
            status, report, _ = _run(
                _fixture(Path(temporary), [10, 30, 20, 40])
            )
            self.assertEqual(status, 1)
            self.assertEqual(report["agreement"]["ranking"]["status"], "fail")

    def test_canonical_sidecar_is_bound_to_csv_score_and_source(self) -> None:
        """Reject sidecars describing different scores or sources."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = _fixture(root)
            target = root / "candidate.json"
            report = json.loads(target.read_text())
            report["samples"][0]["score"] += 1
            target.write_text(json.dumps(report))
            status, _, error = _run(argv)
            self.assertEqual(status, 2)
            self.assertIn("does not match CSV", error)

    def test_canonical_sidecar_is_bound_to_decoded_input_hashes(self) -> None:
        """Reject a sidecar whose input hash differs from the matched CSV."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = _fixture(root)
            target = root / "candidate.json"
            report = json.loads(target.read_text())
            report["samples"][0]["input_hash"] = "f" * 64
            target.write_text(json.dumps(report))
            status, _, error = _run(argv)
            self.assertEqual(status, 2)
            self.assertIn("input hash does not match CSV", error)

    def test_diagnostic_sidecar_without_revision_cannot_qualify_later(
        self
    ) -> None:
        """Require provenance when reusing diagnostic sidecars."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = _fixture(root)
            target = root / "candidate.json"
            report = json.loads(target.read_text())
            report["revision"] = "unreported-external-scores"
            target.write_text(json.dumps(report))
            status, _, error = _run(argv)
            self.assertEqual(status, 2)
            self.assertIn("declared implementation revision", error)

    def test_diagnostic_correlations_do_not_claim_acceptance(self) -> None:
        """Report undefined constant ranks as null with diagnostic status."""
        result = comparison.direct_correlations([1, 1, 1], [4, 4, 4])
        self.assertEqual(result["status"], "diagnostic_only")
        self.assertIsNone(result["srcc"])
        self.assertIsNone(result["tau_b"])
        json.dumps(result, allow_nan=False)

    def test_policy_requires_both_score_sidecars(self) -> None:
        """Reject required policies with missing provenance."""
        argv = [
            "compare_vmaf_scores.py", "a.csv", "b.csv", "--agreement-policy",
            "policy.json"
        ]
        with (mock.patch.object(sys, "argv", argv),
              contextlib.redirect_stderr(io.StringIO()
                                         ), self.assertRaises(SystemExit)):
            comparison.main()


class Cid22AgreementTests(unittest.TestCase):
    """Check canonical CID22 reports without a corpus or GPU dependency."""

    def test_report_preserves_model_input_hashes_and_source_groups(
        self
    ) -> None:
        """Keep each derivative with its original source and decoded inputs."""
        rows = _rows()
        reference, candidate = cid.build_agreement_reports(
            rows, "v0", "a" * 64, "b" * 64, "revision"
        )
        self.assertEqual(reference["metric"], "vmaf-v0")
        self.assertEqual(
            reference["configuration_hash"], candidate["configuration_hash"]
        )
        self.assertEqual(reference["input_manifest_hash"], "b" * 64)
        self.assertEqual(candidate["revision"], "revision")
        self.assertEqual(
            reference["samples"][0]["source_id"], "original/source0.png"
        )
        self.assertEqual(
            reference["samples"][1]["source_id"], "original/source0.png"
        )
        self.assertEqual(reference["samples"][0]["distortion_family"], "codec")
        self.assertEqual(reference["samples"][0]["score"], 10)
        self.assertEqual(candidate["samples"][0]["score"], 13)

    def test_supplied_acceptance_requires_matching_decoded_input_hashes(
        self
    ) -> None:
        """Reject score imports without prepared input identity."""
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "scores.csv"
            target.write_text("original,distorted,v0\na,b,10\n")
            with self.assertRaisesRegex(ValueError, "decoded-input hashes"):
                cid._supplied_rows(target, True)
            target.write_text(
                "original,distorted,v0,original_yuv_sha256,"
                "distorted_yuv_sha256\n"
                f"a,b,10,{'a' * 64},{'b' * 64}\n"
            )
            args = mock.Mock(
                vqm_scores=target, agreement_policy=Path("policy.json")
            )
            rows = [
                {
                    "original": "a",
                    "distorted": "b",
                    "original_yuv_sha256": "c" * 64,
                    "distorted_yuv_sha256": "b" * 64
                }
            ]
            with self.assertRaisesRegex(ValueError, "differ from prepared"):
                cid._apply_supplied(args, rows, [("a", "b")])

    def test_policy_model_must_be_scored(self) -> None:
        """Reject policy checks for unmeasured models."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = root / "policy.json"
            policy.write_text(json.dumps(_policy()))
            args = mock.Mock(agreement_policy=policy)
            with self.assertRaisesRegex(ValueError,
                                        "matching selected VQM model"):
                cid._record_agreement(args, _rows(), ["v1"], {})

    def test_recorded_cid22_policy_controls_status_and_writes_sidecars(
        self
    ) -> None:
        """Accept offsets and reject reordered quality in CID22 gates."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = root / "policy.json"
            policy.write_text(json.dumps(_policy()))
            args = mock.Mock(
                agreement_policy=policy,
                candidate_revision="candidate-revision",
                vqm_scores=Path("supplied.csv"), vqm_v0=False, vqm_v1=False,
                out_dir=root
            )
            info = {
                "v0_model_sha256": "a" * 64,
                "input_manifest_sha256": "b" * 64
            }
            self.assertTrue(cid._record_agreement(args, _rows(), ["v0"], info))
            self.assertEqual(
                info["agreement"]["v0"]["ranking"]["status"], "pass"
            )
            self.assertTrue((root / "v0-reference-agreement.json").is_file())
            self.assertEqual(
                json.loads((root / "v0-candidate-agreement.json").read_text()
                           )["revision"], "candidate-revision"
            )
            self.assertFalse(
                cid._record_agreement(
                    args, _rows([10, 30, 20, 40]), ["v0"], info
                )
            )
            self.assertEqual(
                info["agreement"]["v0"]["ranking"]["status"], "fail"
            )

    def test_no_policy_records_diagnostics_without_qualifying(self) -> None:
        """Keep ordinary CID22 reports distinct from ranking acceptance."""
        with tempfile.TemporaryDirectory() as temporary:
            args = mock.Mock(
                agreement_policy=None, candidate_revision=None,
                vqm_scores=Path("supplied.csv"), vqm_v0=False, vqm_v1=False,
                out_dir=Path(temporary)
            )
            info = {
                "v0_model_sha256": "a" * 64,
                "input_manifest_sha256": "b" * 64
            }
            self.assertTrue(cid._record_agreement(args, _rows(), ["v0"], info))
            self.assertEqual(info["agreement"], {"status": "not_requested"})

    def test_cid22_policy_rejects_unidentified_supplied_revision(self) -> None:
        """Require declared revisions for external score imports."""
        argv = [
            "vmaf_cid22.py", "--agreement-policy", "policy.json",
            "--vqm-scores", "scores.csv"
        ]
        with (mock.patch.object(sys, "argv", argv),
              contextlib.redirect_stderr(io.StringIO()
                                         ), self.assertRaises(SystemExit)):
            cid.parse_args()


if __name__ == "__main__":
    unittest.main()
