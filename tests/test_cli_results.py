# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Keep incomplete CLI frame evidence out of validation and benchmark reports."""

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from validation.temporal import run as temporal_run
from validation.temporal.cli_results import FEATURE_COUNTS, validate_cli_result


def _result(metric: str = "ssimulacra2") -> dict[str, object]:
    """Build complete finite output for a three-frame qualified clip."""
    frames = [
        {
            "index": index,
            "score": 80.0,
            "features": [1.0] * FEATURE_COUNTS.get(metric, 0)
        } for index in range(3)
    ]
    return {
        "sample_id": "clip",
        "metric": metric,
        "score": 80.0,
        "frames": 3,
        "core_seconds": .01,
        "decode_seconds": .001,
        "frame_results": [] if metric == "cvvdp" else frames
    }


class CliResultTests(unittest.TestCase):
    """Validate output shape, ordered frames, feature numbers and pooling."""

    def test_complete_results_cover_every_qualified_metric(self) -> None:
        """CVVDP emits only its clip score; other metrics expose every frame."""
        for metric in (*FEATURE_COUNTS, "cvvdp"):
            with self.subTest(metric=metric):
                validate_cli_result(_result(metric), metric, "clip", 3)

    def test_invalid_clip_fields_are_rejected(self) -> None:
        """Missing, nonnumeric and inconsistent scalar fields cannot qualify."""
        cases = (
            ("score", True), ("score", "80"), ("score", float("nan")),
            ("score", 10**400), ("frames", 3.0), ("frames", True), ("frames", 2),
            ("sample_id", "other"), ("metric", "cvvdp"), ("core_seconds", 0),
            ("core_seconds", "1"), ("decode_seconds", -1),
            ("decode_seconds", float("inf")), ("frame_results",
                                               None), ("frame_results", [])
        )
        for field, invalid in cases:
            value = _result()
            value[field] = invalid
            with self.subTest(field=field,
                              invalid=invalid), self.assertRaises(ValueError):
                validate_cli_result(value, "ssimulacra2", "clip", 3)

    def test_frame_indices_features_and_mean_are_required(self) -> None:
        """A correct declared count cannot conceal reordered or malformed frames."""
        cases = (
            ("index", 2), ("index", True), ("score", float("inf")), ("score", "80"),
            ("score", 79.0), ("features", []), ("features", [1.0] * 5 + [float("nan")]),
            ("features", [1.0] * 5 + [True])
        )
        for field, invalid in cases:
            value = _result("vmaf-v0")
            value["frame_results"][1][field] = invalid
            with self.subTest(field=field,
                              invalid=invalid), self.assertRaises(ValueError):
                validate_cli_result(value, "vmaf-v0", "clip", 3)

    def test_cvvdp_cannot_misrepresent_prefix_scores_as_frames(self) -> None:
        """Whole-clip CVVDP output requires an explicitly empty frame array."""
        value = _result("cvvdp")
        value["frame_results"] = _result()["frame_results"]
        with self.assertRaisesRegex(ValueError, "without frame scores"):
            validate_cli_result(value, "cvvdp", "clip", 3)


class CandidateOutputTests(unittest.TestCase):
    """Exercise the normal validation ingestion path without launching a GPU."""

    def test_video_entry_point_needs_only_the_standard_library(self) -> None:
        """The direct bootstrap command can inspect its plan before setup."""
        script = Path(__file__).resolve().parents[1] / "bench/video.py"
        result = subprocess.run(
            [sys.executable, "-I", "-S",
             str(script), "--dry-run", "--rounds", "1"], capture_output=True, text=True,
            check=False, timeout=10
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("validation.temporal.run", result.stdout)

    def test_complete_frame_evidence_is_required_before_agreement(self) -> None:
        """A successful process with incomplete stdout cannot create timing evidence."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "sample_id": "clip",
                    "frames": 3,
                    "original": "original.y4m",
                    "distorted": "distorted.y4m"
                }
            ]
            args = argparse.Namespace(
                out_dir=root, binary=root / "binary", iterations=1, warmup=0
            )
            value = _result("vmaf-v0")
            incomplete = copy.deepcopy(value)
            incomplete["frame_results"].pop()
            for payload, accepted in ((incomplete, False), (value, True)):
                self._check_output(rows, args, payload, accepted)

    def _check_output(
        self, rows: list[dict[str, object]], args: argparse.Namespace,
        payload: dict[str, object], accepted: bool
    ) -> None:
        """Supply controlled CLI stdout and check the validation result."""
        completed = subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(payload) + "\n", stderr=""
        )
        with mock.patch.object(temporal_run.subprocess, "run", return_value=completed):
            if accepted:
                observed = temporal_run.candidate_scores(rows, "vmaf-v0", args)
                self.assertEqual(observed, [payload])
                return
            with self.assertRaisesRegex(RuntimeError, "incomplete frame results"):
                temporal_run.candidate_scores(rows, "vmaf-v0", args)
            self.assertFalse((args.out_dir / "vmaf-v0-timing.json").exists())
