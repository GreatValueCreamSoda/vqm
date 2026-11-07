#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Verify cold video bootstrap, timing labels and complete benchmark matrices."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from bench import benchmark, video


def _values() -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"clip-{index}",
            "metric": "ssimulacra2",
            "score": 80.0,
            "frames": 12,
            "core_seconds": .012,
            "decode_seconds": .003,
            "frame_results": [{
                "index": frame,
                "score": 80.0
            } for frame in range(12)]
        } for index in range(39)
    ]


def _record(metric: str, index: int = 1, valid: bool = True) -> dict[str, object]:
    return {
        "metric": metric,
        "round": index,
        "valid": valid,
        "wall_seconds": 50.0,
        "timings": {
            "selected_core_seconds_sum": .468,
            "selected_core_clip_median_seconds": .012,
            "selected_decode_clip_median_seconds": .003,
            "candidate_process_wall_seconds": 10.0
        }
    }


def _fixture(output: Path) -> argparse.Namespace:
    output.mkdir(parents=True, exist_ok=True)
    metric = "ssimulacra2"
    values = _values()
    agreement = {
        "candidate": {
            "revision": "sha256:expected",
            "configuration_hash": "config",
            "input_manifest_hash": "inputs"
        },
        "selected_gate_passed": True,
        "gate_mode": "ranking",
        "ranking": {
            "status": "pass",
            "global": {
                "srcc": 1.0,
                "tau_b": 1.0
            }
        },
        "numerical": {
            "status": "fail",
            "max_error": 3.0,
            "mae": 3.0,
            "mean_offset": 3.0
        },
        "policy_sha256": "policy"
    }
    timing = {
        "iterations_per_clip": 20,
        "warmup_per_clip": 3,
        "wall_seconds_including_warmup_decode_and_setup": 999.0,
        "timed_frames": 39 * 12 * 20,
        "warmup_frames": 39 * 12 * 3
    }
    (output / f"{metric}-candidate.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in values)
    )
    (output / f"{metric}-agreement.json").write_text(json.dumps(agreement))
    (output / f"{metric}-timing.json").write_text(json.dumps(timing))
    (output / "inputs.json").write_text("[]\n")
    (output / f"{metric}-pairs.tsv").write_text("clip-0\toriginal.y4m\tdistorted.y4m\n")
    (output /
     f"{metric}-original-details.json").write_text(json.dumps({"seconds": 1000000}))
    return video.arguments(["--out-dir", str(output)])


class VideoBenchmarkPlanningTests(unittest.TestCase):
    """Keep the default one-command path independent of optional Python imports."""

    def test_defaults_cover_five_hosted_metrics_and_repeated_rounds(self) -> None:
        """The default is the prescribed smoke matrix and latency protocol."""
        args = video.arguments([])
        self.assertEqual(args.metrics, video.METRICS)
        self.assertEqual(args.profile, "smoke")
        self.assertEqual((args.rounds, args.iterations, args.warmup), (5, 20, 3))
        self.assertIn("artifacts/reports/video-benchmark", str(args.out_dir))

    def test_invalid_counts_and_metric_selections_fail_before_setup(self) -> None:
        """An empty, duplicate or unknown metric cannot silently shrink the matrix."""
        invalid = (
            ["--metrics", ""], ["--metrics",
                                "ssimulacra2,ssimulacra2"], ["--metrics", "psnr-hvs"],
            ["--rounds", "0"], ["--iterations", "0"], ["--jobs",
                                                       "0"], ["--warmup", "-1"]
        )
        for options in invalid:
            with self.subTest(options=options), redirect_stderr(io.StringIO()):
                self.assertRaises(SystemExit, video.arguments, options)

    def test_dry_run_never_launches_processes_or_creates_output(self) -> None:
        """Inspect bootstrap and scoring commands on a machine with no dependencies."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            stream = io.StringIO()
            with mock.patch.object(video.subprocess,
                                   "run") as launched, redirect_stdout(stream):
                self.assertEqual(
                    video.main(
                        ["--dry-run", "--rounds", "1", "--out-dir",
                         str(output)]
                    ), 0
                )
            launched.assert_not_called()
            self.assertFalse(output.exists())
            plan = stream.getvalue()
            self.assertIn(".venv/bin/python -m validation.temporal.run", plan)
            self.assertEqual(plan.count("-m validation.temporal.run"), 5)
            self.assertNotIn("Vship", plan)
            self.assertNotIn("cid22", plan)

    def test_suite_dispatch_defaults_to_video_and_preserves_images(self) -> None:
        """The CLI defaults to videos while explicit legacy image flags keep their parser."""
        with mock.patch.object(video, "main", return_value=7) as selected:
            self.assertEqual(benchmark.command_line(["--rounds", "1"]), 7)
            selected.assert_called_once_with(["--rounds", "1"])
        with mock.patch.object(benchmark, "main", return_value=9) as selected:
            self.assertEqual(
                benchmark.command_line(
                    ["--suite", "images", "--metric", "butteraugli"]
                ), 9
            )
            selected.assert_called_once_with(["--metric", "butteraugli"])

    def test_setup_builds_only_the_needed_original_families(self) -> None:
        """Fetch four pinned trees and build the managed runtime plus normal CLI."""
        args = video.arguments(["--offline"])
        commands = video.setup_commands(args)
        self.assertEqual(
            commands[0][commands[0].index("--only") + 1],
            "ssimulacra2,libjxl,vmaf,ColorVideoVDP"
        )
        self.assertIn("--check", commands[0])
        self.assertEqual(
            commands[1][commands[1].index("--only") + 1], "cloudinary,libjxl,vmaf,cvvdp"
        )
        self.assertIn("vqm", commands[3])
        self.assertIn("vqm_bench", commands[3])
        self.assertIn("--check", commands[-1])
        self.assertEqual(commands[-1][-1], video.SMOKE_SOURCES)

    def test_rotating_order_balances_first_position(self) -> None:
        """Every arm occurs once per round and leads once in the default five rounds."""
        orders = [video.metric_order(video.METRICS, index) for index in range(5)]
        self.assertEqual(tuple(order[0] for order in orders), video.METRICS)
        self.assertTrue(all(set(order) == set(video.METRICS) for order in orders))
        self.assertEqual(video.metric_order(("cvvdp", ), 50), ("cvvdp", ))

    def test_round_commands_use_checked_scores_and_the_normal_cli_workflow(
        self
    ) -> None:
        """Reference score generation cannot masquerade as a measured CPU arm."""
        args = video.arguments(
            [
                "--profile", "qualification", "--metrics", "cvvdp", "--iterations", "2",
                "--warmup", "0"
            ]
        )
        command = video.measurement_command(args, "cvvdp", Path("arm"))
        self.assertIn("validation.temporal.run", command)
        self.assertIn("--offline", command)
        self.assertNotIn("generate", command)
        self.assertEqual(command[command.index("--iterations") + 1], "2")
        self.assertEqual(command[command.index("--profile") + 1], "qualification")


class VideoBenchmarkTimingTests(unittest.TestCase):
    """Verify complete clips and distinguish selected core/decode from process wall."""

    def test_timing_aggregation_does_not_use_original_cache_seconds(self) -> None:
        """Only candidate JSON and candidate process timing enter the duration report."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = _fixture(output)
            result = video._metric_result(output, "ssimulacra2", args, "expected")
            timing = result["timings"]
            self.assertAlmostEqual(timing["selected_core_seconds_sum"], 39 * .012)
            self.assertAlmostEqual(timing["selected_core_clip_median_seconds"], .012)
            self.assertAlmostEqual(timing["selected_core_seconds_per_frame"], .001)
            self.assertAlmostEqual(timing["selected_decode_seconds_sum"], 39 * .003)
            self.assertEqual(timing["candidate_process_wall_seconds"], 999)
            self.assertEqual(result["numerical"], "fail")
            self.assertEqual(result["numerical_diagnostics"]["max_error"], 3.0)
            self.assertEqual(result["numerical_diagnostics"]["mae"], 3.0)
            self.assertTrue(result["quality_passed"])
            self.assertNotIn("reference_seconds", timing)
            self.assertNotIn("speedup", result)
            self.assertIn("inputs.json", result["report_hashes"])

    def test_truncated_duplicate_and_mismatched_payloads_are_rejected(self) -> None:
        """A missing clip or wrong association cannot become a shorter successful benchmark."""
        args = video.arguments([])
        malformed = [_values()[:-1], _values(), _values(), _values()]
        malformed[1][1]["sample_id"] = malformed[1][0]["sample_id"]
        malformed[2][0]["metric"] = "cvvdp"
        malformed[3][0]["frames"] = 11
        for values in malformed:
            with self.subTest(values=values[0]), self.assertRaises(ValueError):
                video._validate_payload(values, "ssimulacra2", args)

    def test_invalid_candidate_durations_never_enter_statistics(self) -> None:
        """Zero core time, negative values and nonfinite durations are invalid measurements."""
        args = video.arguments([])
        cases = (
            ("core_seconds", 0), ("core_seconds", -1), ("core_seconds", float("nan")),
            ("decode_seconds", -1), ("decode_seconds", float("inf"))
        )
        for name, value in cases:
            values = _values()
            values[0][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                video._validate_payload(values, "ssimulacra2", args)

    def test_changed_binary_and_repetition_flags_are_rejected(self) -> None:
        """All arms must use the same candidate and requested repetition protocol."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = _fixture(output)
            self.assertRaises(
                ValueError, video._metric_result, output, "ssimulacra2", args,
                "other-binary"
            )
            args.iterations = 21
            self.assertRaises(
                ValueError, video._metric_result, output, "ssimulacra2", args,
                "expected"
            )

    def test_failed_process_and_quality_gate_both_invalidate_an_arm(self) -> None:
        """Complete stdout does not qualify a failed process or failed ranking gate."""
        args = video.arguments([])
        for exit_code, quality in ((1, True), (0, False)):
            completed = {"exit_code": exit_code, "wall_seconds": 10, "log": "log"}
            evidence = {"quality_passed": quality, "timings": {}}
            with mock.patch.object(video, "_run_command", return_value=completed), \
                    mock.patch.object(video, "_metric_result", return_value=evidence):
                record = video._measure_metric(args, "cvvdp", 0, "binary", {})
            self.assertFalse(record["valid"])

    def test_round_dispersion_and_missing_arms_remain_visible(self) -> None:
        """A partial matrix records failed completion alongside medians and spread."""
        first, second = _record("ssimulacra2"), _record("ssimulacra2", 2)
        rejected = _record("ssimulacra2", 3, False)
        second["timings"]["selected_core_seconds_sum"] = .668
        rejected["timings"]["selected_core_seconds_sum"] = 999
        result = video.aggregate([first, second, rejected], ("ssimulacra2", "cvvdp"), 3)
        self.assertFalse(result["ssimulacra2"]["passed"])
        self.assertFalse(result["cvvdp"]["passed"])
        self.assertEqual(result["cvvdp"]["attempted_rounds"], 0)
        values = result["ssimulacra2"]["durations"]["selected_core_seconds_sum"]
        self.assertAlmostEqual(values["median"], .568)
        self.assertAlmostEqual(values["mad"], .1)
        self.assertEqual(values["min"], .468)
        self.assertEqual(values["max"], .668)

    def test_invalid_process_wall_cannot_enter_statistics(self) -> None:
        """Process elapsed time must be finite even when per-clip times are valid."""
        for wall in (-1, float("nan"), float("inf")):
            with self.subTest(wall=wall), self.assertRaises(ValueError):
                video._timings(
                    _values(), {"wall_seconds_including_warmup_decode_and_setup": wall}
                )

    def test_duplicate_rounds_cannot_cover_a_missing_round(self) -> None:
        """The completion contract includes round identity rather than only record count."""
        records = [_record("cvvdp"), _record("cvvdp")]
        self.assertFalse(video.aggregate(records, ("cvvdp", ), 2)["cvvdp"]["passed"])


class VideoBenchmarkFailureTests(unittest.TestCase):
    """Retain clear setup and measurement failures without overwriting past runs."""

    def test_missing_system_tools_fail_before_any_download(self) -> None:
        """Report actionable missing prerequisites without starting subprocesses."""
        with mock.patch.object(video.shutil, "which", return_value=None), \
                mock.patch.object(video, "_capture") as captured:
            self.assertRaisesRegex(
                RuntimeError, "missing prerequisites", video.preflight
            )
        captured.assert_not_called()

    def test_preflight_checks_required_development_packages_and_versions(self) -> None:
        """The bootstrap checks native original dependencies as well as Vulkan image headers."""

        def captured(command: list[str]) -> str:
            return "cmake version 3.25.0" if command[0] == "cmake" else "1.0"
        with mock.patch.object(video.shutil, "which", side_effect=lambda name: "/bin/" + name), \
                mock.patch.object(video, "_capture", side_effect=captured) as called, \
                mock.patch.object(video, "_check_headers"):
            result = video.preflight()
        self.assertIn("libhwy", result["packages"])
        self.assertIn("lcms2", result["packages"])
        package_check = [
            call.args[0] for call in called.call_args_list if "--exists" in call.args[0]
        ][0]
        self.assertIn("libhwy >= 0.15", package_check)
        self.assertIn("lcms2 >= 2.13", package_check)

    def test_nonempty_output_is_preserved(self) -> None:
        """A repeated explicit output path cannot mix or replace earlier evidence."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            prior = output / "summary.json"
            prior.write_text("prior evidence")
            with mock.patch.object(video, "preflight") as checked:
                self.assertRaisesRegex(
                    RuntimeError, "nonempty", video.main, ["--out-dir", directory]
                )
            checked.assert_not_called()
            self.assertEqual(prior.read_text(), "prior evidence")

    def test_setup_only_never_claims_a_timed_matrix(self) -> None:
        """A successful bootstrap records its phase without launching score arms."""
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(video, "preflight", return_value={}), \
                    mock.patch.object(video, "_setup"), \
                    mock.patch.object(video, "provenance", return_value={}), \
                    mock.patch.object(video, "_measure_metric") as measured, \
                    redirect_stdout(io.StringIO()):
                status = video.main(["--setup-only", "--out-dir", directory])
            result = json.loads((Path(directory) / "summary.json").read_text())
            measured.assert_not_called()
            self.assertEqual(status, 0)
            self.assertEqual(result["status"], "setup_only")
            self.assertEqual(result["rounds"], [])
            self.assertFalse(any(row["passed"] for row in result["metrics"].values()))

    def test_interrupted_matrix_preserves_completed_arms(self) -> None:
        """A launch failure retains earlier round records and returns failure."""
        with tempfile.TemporaryDirectory() as directory:
            first = _record("ssimulacra2")
            with mock.patch.object(video, "preflight", return_value={}), \
                    mock.patch.object(video, "_setup"), \
                    mock.patch.object(video, "provenance", return_value={"candidate": {"sha256": "binary"}}), \
                    mock.patch.object(video, "_measure_metric", side_effect=[first, OSError("launch failed")]), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = video.main(
                    [
                        "--rounds", "1", "--metrics", "ssimulacra2,cvvdp", "--out-dir",
                        directory
                    ]
                )
            result = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(status, 1)
            self.assertEqual(result["status"], "fail")
            self.assertEqual(len(result["rounds"]), 1)
            self.assertEqual(result["metrics"]["cvvdp"]["attempted_rounds"], 0)
            self.assertIn("launch failed", result["error"])

    def test_empty_measurement_cannot_be_reported_as_success(self) -> None:
        """An accidentally skipped matrix cannot pass through all(empty) semantics."""
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(video, "preflight", return_value={}), \
                    mock.patch.object(video, "_setup"), \
                    mock.patch.object(video, "provenance", return_value={"candidate": {"sha256": "binary"}}), \
                    mock.patch.object(video, "_measure_all"), redirect_stdout(io.StringIO()):
                status = video.main(
                    ["--rounds", "1", "--metrics", "cvvdp", "--out-dir", directory]
                )
            result = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(status, 1)
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["metrics"]["cvvdp"]["valid_rounds"], 0)


if __name__ == "__main__":
    unittest.main()
