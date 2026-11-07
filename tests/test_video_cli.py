#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Exercise the public CLI's ordered video, batch and machine-output contracts."""

import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import unittest
from pathlib import Path

import make_goldens


def _binary() -> str:
    path = os.environ.get("VQM_CLI_BINARY")
    if not path:
        raise unittest.SkipTest("CLI binary not configured")
    return path


def _require_gpu() -> None:
    path = os.environ.get("VQM_TEST_BINARY")
    if not path:
        raise unittest.SkipTest("GPU test binary not configured")
    probe = subprocess.run(
        [path, "--probe-gpu"], capture_output=True, text=True, timeout=120, check=False
    )
    if probe.returncode == 0:
        return
    if "no usable Vulkan device" in probe.stderr:
        raise unittest.SkipTest("no usable Vulkan device")
    raise AssertionError(f"GPU probe failed: {probe.stderr}")


def _run(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_binary(), *(str(arg) for arg in args)], capture_output=True, text=True,
        timeout=120, check=False
    )


def _write_y4m(
    path: Path, *, frames: int = 3, distorted: bool = False, width: int = 32,
    height: int = 32, fps: str = "30:1", distorted_frame: int | None = None
) -> None:
    pixels = bytearray(f"YUV4MPEG2 W{width} H{height} F{fps} Ip C444\n".encode())
    offsets = (0, 7, 5, 17)
    if distorted and distorted_frame is None:
        distorted_frame = frames - 1
    for frame in range(frames):
        pixels.extend(b"FRAME\n")
        for plane in range(3):
            pixels.extend(
                _plane(
                    width, height, offsets[frame % 4], plane, frame == distorted_frame
                )
            )
    path.write_bytes(pixels)


def _plane(width: int, height: int, offset: int, plane: int, distorted: bool) -> bytes:
    values = bytearray()
    for y in range(height):
        for x in range(width):
            value = 64 + (3 * x + 5 * y + 37 * plane) % 96 + offset
            changed = 224 - value if plane == 0 else 220
            values.append(changed if distorted and x >= width // 2 else value)
    return bytes(values)


class VideoCliArguments(unittest.TestCase):
    """Reject malformed batch, timing and repetition inputs before GPU work."""

    def test_rejects_invalid_repetition_and_output_options(self) -> None:
        """Keep iteration counts unsigned and output modes unambiguous."""
        cases = (
            ("--iterations", "0"), ("--iterations", "-1"), ("--warmup", "-1"),
            ("--iterations", "1x"), ("--iterations", "4294967296"), ("--json", "--raw"),
            ("--metric", "all", "--raw"), ("--verbose", "--raw")
        )
        for flags in cases:
            with self.subTest(flags=flags):
                result = _run(*flags, "a", "b")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(result.stdout, "")

    def test_rejects_empty_or_unknown_metric_list_items(self) -> None:
        """Validate every list item, including items after the all alias."""
        for value in ("", ",cvvdp", "cvvdp,", "cvvdp,,butteraugli", "all,unknown"):
            with self.subTest(value=value):
                result = _run("--metric", value, "a", "b")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("--metric", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_rejects_vmaf_geometry_options_without_vmaf(self) -> None:
        """Require a selected VMAF metric when scaling or padding."""
        for flag in ("--vmaf-scale", "--vmaf-pad"):
            with self.subTest(flag=flag):
                result = _run("--metric", "cvvdp", flag, "a", "b")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(flag, result.stderr)
                self.assertEqual(result.stdout, "")

    def test_rejects_vmaf_v1_model_with_wrong_metric_or_name(self) -> None:
        """Require a VMAF v1 selection and a supported preset."""
        for metric, model in (("vmaf-v0", "4k-3h"), ("vmaf-v1", "unknown")):
            result = _run("--metric", metric, "--vmaf-v1-model", model, "a", "b")
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("--vmaf-v1-model", result.stderr)

    def test_vmaf_v1_temporal_and_hdr_option_validation(self) -> None:
        """Require VMAF v1 and let its HDR model control geometry."""
        for flags, flag in ((("--metric", "vmaf-v0", "--vmaf-v1-hfr"), "--vmaf-v1-hfr"),
                            (("--metric", "vmaf-v1", "--vmaf-v1-model", "hdr-4k-1.5h",
                              "--vmaf-pad"), "--vmaf-pad")):
            result = _run(*flags, "a", "b")
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn(flag, result.stderr)

    def test_pair_list_requires_machine_output_and_unique_ids(self) -> None:
        """Validate complete TSV records before opening any clip or GPU."""
        with tempfile.TemporaryDirectory() as temporary:
            pairs = Path(temporary) / "pairs.tsv"
            pairs.write_text("one\ta\tb\n")
            result = _run("--pairs-file", pairs)
            self.assertEqual(result.returncode, 2)
            self.assertIn("requires --json", result.stderr)
            pairs.write_text("one\ta\tb\none\tc\td\n")
            result = _run("--json", "--pairs-file", pairs)
            self.assertEqual(result.returncode, 1)
            self.assertIn("duplicate sample ID", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_video_timing_and_geometry_must_match(self) -> None:
        """Reject incompatible video headers before constructing a pipeline."""
        with tempfile.TemporaryDirectory() as temporary:
            original, distorted = Path(temporary) / "a.y4m", Path(temporary) / "b.y4m"
            _write_y4m(original)
            _write_y4m(distorted, fps="30000:1001")
            result = _run("--json", "--metric", "vmaf-v0", original, distorted)
            self.assertEqual(result.returncode, 1)
            self.assertIn("unequal frame rates", result.stderr)
            self.assertEqual(result.stdout, "")
            _write_y4m(distorted, width=34)
            result = _run("--json", original, distorted)
            self.assertEqual(result.returncode, 1)
            self.assertIn("size mismatch", result.stderr)
            self.assertEqual(result.stdout, "")


class VideoCliGpu(unittest.TestCase):
    """Check complete ordered clips through the shipped executable."""

    def test_all_qualified_metrics_emit_complete_clip_results(self) -> None:
        """Expose ordered frame results or a whole-clip JOD for qualification."""
        _require_gpu()
        metrics = ("ssimulacra2", "butteraugli", "vmaf-v0", "vmaf-v1", "cvvdp")
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        original, distorted = Path(temporary) / "a.y4m", Path(temporary) / "b.y4m"
        for metric in metrics:
            with self.subTest(metric=metric):
                width, height = (216, 80) if metric == "vmaf-v1" else (32, 32)
                _write_y4m(original, width=width, height=height)
                _write_y4m(distorted, width=width, height=height, distorted=True)
                result = _run("--json", "--metric", metric, original, distorted)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["metric"], metric)
                self.assertEqual(report["frames"], 3)
                self.assertTrue(math.isfinite(report["score"]))
                self.assertGreater(report["core_seconds"], 0)
                self.assertGreater(report["decode_seconds"], 0)
                self._check_frames(report, metric)

    def test_all_selection_matches_independent_reports(self) -> None:
        """Score all six metrics on one complete clip in canonical order."""
        _require_gpu()
        metrics = (
            "ssimulacra2", "butteraugli", "vmaf-v0", "vmaf-v1", "cvvdp", "psnr-hvs"
        )
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        original, distorted = Path(temporary) / "a.y4m", Path(temporary) / "b.y4m"
        _write_y4m(original, width=216, height=80)
        _write_y4m(distorted, width=216, height=80, distorted=True)
        combined = _run("--json", "--metric", "all", original, distorted)
        self.assertEqual(combined.returncode, 0, combined.stderr)
        reports = [json.loads(line) for line in combined.stdout.splitlines()]
        self.assertEqual([report["metric"] for report in reports], list(metrics))
        self.assertEqual(len({report["decode_seconds"] for report in reports}), 1)
        for report in reports:
            with self.subTest(metric=report["metric"]):
                independent = _run(
                    "--json", "--metric", report["metric"], original, distorted
                )
                self.assertEqual(independent.returncode, 0, independent.stderr)
                expected = json.loads(independent.stdout)
                self.assertEqual(report["frames"], expected["frames"])
                self.assertEqual(report["score"], expected["score"])
                self.assertEqual(report["frame_results"], expected["frame_results"])

    def test_vmaf_scale_and_pad_are_independent_for_both_versions(self) -> None:
        """Exercise native, scale, pad, and combined modes on a 4:3 clip."""
        _require_gpu()
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        original, distorted = root / "a.y4m", root / "b.y4m"
        _write_y4m(original, width=640, height=480)
        _write_y4m(distorted, width=640, height=480, distorted=True)
        metrics = ("vmaf-v0", "vmaf-v1")
        features = {metric: set() for metric in metrics}
        modes = (
            (), ("--vmaf-scale", ), ("--vmaf-pad", ), ("--vmaf-scale", "--vmaf-pad")
        )
        for flags in modes:
            result = _run(
                "--json", "--metric", ",".join(metrics), *flags, original, distorted
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            reports = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual([report["metric"] for report in reports], list(metrics))
            for report in reports:
                self.assertTrue(math.isfinite(report["score"]))
                frames = report["frame_results"]
                self.assertEqual([frame["index"] for frame in frames], [0, 1, 2])
                features[report["metric"]].add(
                    tuple(tuple(frame["features"]) for frame in frames)
                )
        for metric, results in features.items():
            with self.subTest(metric=metric):
                self.assertEqual(len(results), len(modes))

    def test_vmaf_v1_hfr_drains_both_frames(self) -> None:
        """Keep both delayed HFR scores in the ordered CLI report."""
        _require_gpu()
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        original = root / "hfr-original.y4m"
        distorted = root / "hfr-distorted.y4m"
        _write_y4m(original, width=216, height=80, fps="60000:1001")
        _write_y4m(distorted, width=216, height=80, fps="60000:1001", distorted=True)
        result = _run(
            "--json", "--metric", "vmaf-v0,vmaf-v1", "--vmaf-v1-hfr", original,
            distorted
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for line in result.stdout.splitlines():
            clip = json.loads(line)
            self.assertEqual(clip["frames"], 3)
            self.assertEqual(
                [frame["index"] for frame in clip["frame_results"]], [0, 1, 2]
            )

    def test_vmaf_v1_4k_3h_can_score_above_100(self) -> None:
        """Pass the selected predictor through to clip scoring and reporting."""
        _require_gpu()
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        original = root / "clip.y4m"
        _write_y4m(original, width=320, height=240)
        result = _run(
            "--json", "--verbose", "--metric", "vmaf-v1", "--vmaf-v1-model", "4k-3h",
            original, original
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertGreater(report["score"], 100)
        self.assertLessEqual(report["score"], 110)
        scores = [frame["score"] for frame in report["frame_results"]]
        self.assertEqual(report["frame_stats"]["perfect_frames"], scores.count(110))

    def test_psnr_hvs_pools_frame_errors_before_db_conversion(self) -> None:
        """Zero-error frames contribute to the clip mean without dominating it."""
        _require_gpu()
        temporary = Path(self.enterContext(tempfile.TemporaryDirectory()))
        original, distorted = temporary / "a.y4m", temporary / "b.y4m"
        _write_y4m(original)
        _write_y4m(distorted, distorted=True)
        result = _run("--json", "--metric", "psnr-hvs", original, distorted)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        frames = report["frame_results"]
        self.assertEqual([frame["index"] for frame in frames], [0, 1, 2])
        self.assertEqual([frame["score"] for frame in frames[:2]], ["inf", "inf"])
        errors = [frame["normalized_error"] for frame in frames]
        weighted_errors = [0.8 * y + 0.1 * cb + 0.1 * cr for y, cb, cr in errors]
        mean_error = sum(weighted_errors) / len(frames)
        self.assertIsInstance(report["score"], float)
        self.assertAlmostEqual(
            report["score"], -10 * math.log10(mean_error), delta=1e-6
        )

        identity = _run("--json", "--metric", "psnr-hvs", original, original)
        self.assertEqual(identity.returncode, 0, identity.stderr)
        self.assertEqual(json.loads(identity.stdout)["score"], "inf")

    def test_verbose_json_reports_temporal_scores_and_infinite_frames(self) -> None:
        """Expose filtered frame JOD and summarize finite scores in batch records."""
        _require_gpu()
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        original, distorted, single = root / "a.y4m", root / "b.y4m", root / "c.y4m"
        _write_y4m(original)
        _write_y4m(distorted, distorted_frame=0)
        _write_y4m(single, frames=1)
        pairs = root / "pairs.tsv"
        pairs.write_text(f"clip\t{original}\t{distorted}\nsingle\t{single}\t{single}\n")
        flags = ("--json", "--metric", "cvvdp,psnr-hvs", "--pairs-file", pairs)
        ordinary = _run(*flags)
        verbose = _run("--verbose", *flags)
        self.assertEqual(ordinary.returncode, 0, ordinary.stderr)
        self.assertEqual(verbose.returncode, 0, verbose.stderr)
        baseline = [json.loads(line) for line in ordinary.stdout.splitlines()]
        reports = [json.loads(line) for line in verbose.stdout.splitlines()]
        self.assertEqual(
            [(r["sample_id"], r["metric"]) for r in reports], [
                ("clip", "cvvdp"), ("clip", "psnr-hvs"), ("single", "cvvdp"),
                ("single", "psnr-hvs")
            ]
        )
        for expected, actual in zip(baseline, reports, strict=True):
            self.assertNotIn("frame_stats", expected)
            self.assertNotIn("scoring_fps", expected)
            self.assertEqual(actual["score"], expected["score"])
            self._check_frame_stats(actual)
        self._check_scoring_fps(reports[:2])
        self._check_scoring_fps(reports[2:])
        self.assertEqual(baseline[0]["frame_results"], [])
        self.assertLess(reports[0]["frame_results"][1]["score"], 10)
        self.assertGreater(reports[0]["frame_stats"]["finite_stddev"], 0)
        self.assertEqual(
            [f["score"] for f in reports[1]["frame_results"][1:]], ["inf", "inf"]
        )
        self.assertEqual(reports[1]["frame_results"], baseline[1]["frame_results"])
        self.assertEqual(reports[2]["frame_results"], [{"index": 0, "score": 10}])
        self.assertEqual(reports[3]["frame_stats"]["infinite_frames"], 1)

    def _check_scoring_fps(self, reports: list[dict[str, object]]) -> None:
        fps = reports[0]["scoring_fps"]
        expected = reports[0]["frames"] / sum(
            report["core_seconds"] for report in reports
        )
        self.assertGreater(fps, 0)
        self.assertTrue(math.isclose(fps, expected, rel_tol=1e-12))
        self.assertEqual({report["scoring_fps"] for report in reports}, {fps})

    def _check_frame_stats(self, report: dict[str, object]) -> None:
        frames = report["frame_results"]
        self.assertEqual(
            [frame["index"] for frame in frames], list(range(report["frames"]))
        )
        scores = [float(frame["score"]) for frame in frames]
        finite = [score for score in scores if math.isfinite(score)]
        stats = report["frame_stats"]
        perfect_score = {
            "ssimulacra2": 100,
            "butteraugli": 0,
            "vmaf-v0": 100,
            "vmaf-v1": 100,
            "cvvdp": 10,
            "psnr-hvs": math.inf,
        }[report["metric"]]
        infinite = len(scores) - len(finite)
        self.assertEqual(stats["min"], min(finite, default="inf"))
        self.assertEqual(stats["max"], "inf" if infinite else max(finite))
        self.assertEqual(stats["perfect_frames"], scores.count(perfect_score))
        self.assertEqual(stats["infinite_frames"], infinite)
        if not finite:
            self.assertIsNone(stats["finite_mean"])
            self.assertIsNone(stats["finite_stddev"])
            return
        self.assertAlmostEqual(stats["finite_mean"], statistics.mean(finite))
        self.assertAlmostEqual(stats["finite_stddev"], statistics.pstdev(finite))

    def test_verbose_text_aligns_selected_metrics(self) -> None:
        """Align each metric's pooled score in selection order."""
        _require_gpu()
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        original, distorted = root / "a.y4m", root / "b.y4m"
        _write_y4m(original, width=216, height=80)
        _write_y4m(distorted, width=216, height=80, distorted=True)
        metrics = (
            "psnr-hvs", "butteraugli", "cvvdp", "ssimulacra2", "vmaf-v1", "vmaf-v0"
        )
        flags = ("-v", "--metric", ",".join(metrics), original, distorted)
        result = _run(*flags)
        self.assertEqual(result.returncode, 0, result.stderr)
        count, fps, header, *lines = result.stdout.splitlines()
        self.assertEqual(count, "Frames: 3")
        self.assertRegex(fps, r"^Scoring FPS: \d+\.\d{2}$")
        self.assertGreater(float(fps.removeprefix("Scoring FPS: ")), 0)
        self.assertEqual(
            " ".join(header.split()),
            "Metric Aggregate Frame min Frame max Finite stddev Perfect"
        )
        rows = [list(re.finditer(r"\S+", line)) for line in lines]
        self.assertEqual([row[0].group() for row in rows], list(metrics))
        self.assertEqual([len(row) for row in rows], [6] * len(metrics))
        self.assertEqual(
            len({tuple(cell.end() for cell in row[1:])
                 for row in rows}), 1
        )
        verbose_json = _run("--json", *flags)
        self.assertEqual(verbose_json.returncode, 0, verbose_json.stderr)
        reports = [json.loads(line) for line in verbose_json.stdout.splitlines()]
        self._check_scoring_fps(reports)
        for row, report in zip(rows, reports, strict=True):
            self.assertEqual(row[0].group(), report["metric"])
            self._check_frame_stats(report)
            aggregate = float(row[1].group())
            self.assertTrue(
                math.isclose(aggregate, float(report["score"]), rel_tol=5e-8)
            )
            if report["metric"] in {"cvvdp", "psnr-hvs"}:
                self.assertNotAlmostEqual(
                    aggregate, report["frame_stats"]["finite_mean"]
                )
            self.assertEqual(
                int(row[-1].group()), report["frame_stats"]["perfect_frames"]
            )

    def _check_frames(self, report: dict[str, object], metric: str) -> None:
        frames = report["frame_results"]
        if metric == "cvvdp":
            self.assertEqual(frames, [])
            self.assertLess(report["score"], 10)
            return
        self.assertEqual([frame["index"] for frame in frames], [0, 1, 2])
        self.assertAlmostEqual(
            report["score"],
            sum(frame["score"] for frame in frames) / 3, delta=1e-12
        )
        if metric.startswith("vmaf"):
            count = 6 if metric == "vmaf-v0" else 4
            self.assertEqual([len(frame["features"]) for frame in frames], [count] * 3)
        elif metric == "ssimulacra2":
            self.assertEqual(frames[0]["score"], 100)
            self.assertLess(frames[-1]["score"], frames[0]["score"])
        else:
            self.assertEqual(frames[0]["score"], 0)
            self.assertGreater(frames[-1]["score"], 0)

    def test_spatial_text_output_uses_all_video_frames(self) -> None:
        """A distortion confined to the last frame affects the ordinary CLI."""
        _require_gpu()
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        original, distorted = Path(temporary) / "a.y4m", Path(temporary) / "b.y4m"
        _write_y4m(original)
        _write_y4m(distorted, distorted=True)
        metrics = ("butteraugli", "ssimulacra2")
        expected = []
        for metric in metrics:
            with self.subTest(metric=metric):
                plain = _run("--metric", metric, original, distorted)
                machine = _run("--json", "--metric", metric, original, distorted)
                self.assertEqual(plain.returncode, 0, plain.stderr)
                self.assertEqual(machine.returncode, 0, machine.stderr)
                self.assertAlmostEqual(
                    float(plain.stdout),
                    json.loads(machine.stdout)["score"], delta=1e-7
                )
                expected.append(f"{metric}: {plain.stdout.strip()}")
        combined = _run("--metric", ",".join(metrics), original, distorted)
        self.assertEqual(combined.returncode, 0, combined.stderr)
        self.assertEqual(combined.stdout.splitlines(), expected)

    def test_batch_and_repeated_clips_reset_pending_history(self) -> None:
        """Keep metric order, escaped IDs, and isolated sequence tails in a batch."""
        _require_gpu()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, distorted, still = root / "a.y4m", root / "b.y4m", root / "c.y4m"
            _write_y4m(original)
            _write_y4m(distorted, distorted=True)
            _write_y4m(still, frames=1)
            png = root / "still.png"
            make_goldens.write_png(png, 32, 32, 3, make_goldens.gradient())
            pairs = root / "pairs.tsv"
            sample_id = 'clip"with\\quotes'
            pairs.write_text(
                f"{sample_id}\t{original}\t{distorted}\n"
                f"last\t{still}\t{still}\nimage\t{png}\t{png}\n"
            )
            batch = _run(
                "--json", "--metric", "vmaf-v0", "--metric", "butteraugli,vmaf-v0",
                "--pairs-file", pairs, "--iterations", "2", "--warmup", "1"
            )
            self.assertEqual(batch.returncode, 0, batch.stderr)
            reports = [json.loads(line) for line in batch.stdout.splitlines()]
            self.assertEqual(
                [report["sample_id"] for report in reports],
                [sample_id, sample_id, "last", "last", "image", "image"]
            )
            self.assertEqual(
                [report["metric"] for report in reports], ["vmaf-v0", "butteraugli"] * 3
            )
            self.assertEqual(
                [report["frames"] for report in reports], [3, 3, 1, 1, 1, 1]
            )
            independent = _run("--json", "--metric", "vmaf-v0", still, still)
            self.assertEqual(independent.returncode, 0, independent.stderr)
            self.assertEqual(
                reports[2]["score"],
                json.loads(independent.stdout)["score"]
            )
            self.assertEqual(reports[2]["frame_results"][0]["features"][1], 0)

    def test_late_malformed_or_unequal_frames_emit_no_partial_result(self) -> None:
        """A successful prefix must never become a complete clip result."""
        _require_gpu()
        with tempfile.TemporaryDirectory() as temporary:
            original, distorted = Path(temporary) / "a.y4m", Path(temporary) / "b.y4m"
            _write_y4m(original)
            _write_y4m(distorted, frames=2)
            result = _run("--json", original, distorted)
            self.assertEqual(result.returncode, 1)
            self.assertIn("unequal frame counts", result.stderr)
            self.assertEqual(result.stdout, "")
            _write_y4m(distorted)
            distorted.write_bytes(distorted.read_bytes()[:-1])
            result = _run(
                "--json", "--metric", "ssimulacra2,butteraugli", original, distorted
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("truncated y4m frame", result.stderr)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
