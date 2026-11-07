#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Exercise the public CLI's ordered video, batch and machine-output contracts."""

import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


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
    height: int = 32, fps: str = "30:1"
) -> None:
    pixels = bytearray(f"YUV4MPEG2 W{width} H{height} F{fps} Ip C444\n".encode())
    offsets = (0, 7, 5, 17)
    for frame in range(frames):
        pixels.extend(b"FRAME\n")
        for plane in range(3):
            pixels.extend(
                _plane(
                    width, height, offsets[frame % 4], plane, distorted
                    and frame == frames - 1
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
            ("--iterations", "1x"), ("--iterations", "4294967296"), ("--json", "--raw")
        )
        for flags in cases:
            with self.subTest(flags=flags):
                result = _run(*flags, "a", "b")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(result.stdout, "")

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
        for metric in ("ssimulacra2", "butteraugli"):
            with self.subTest(metric=metric):
                plain = _run("--metric", metric, original, distorted)
                machine = _run("--json", "--metric", metric, original, distorted)
                self.assertEqual(plain.returncode, 0, plain.stderr)
                self.assertEqual(machine.returncode, 0, machine.stderr)
                self.assertAlmostEqual(
                    float(plain.stdout),
                    json.loads(machine.stdout)["score"], delta=1e-7
                )

    def test_batch_and_repeated_clips_reset_pending_history(self) -> None:
        """Emit one object per clip with escaped IDs and isolated sequence tails."""
        _require_gpu()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, distorted, still = root / "a.y4m", root / "b.y4m", root / "c.y4m"
            _write_y4m(original)
            _write_y4m(distorted, distorted=True)
            _write_y4m(still, frames=1)
            pairs = root / "pairs.tsv"
            sample_id = 'clip"with\\quotes'
            pairs.write_text(
                f"{sample_id}\t{original}\t{distorted}\nlast\t{still}\t{still}\n"
            )
            batch = _run(
                "--json", "--metric", "vmaf-v0", "--pairs-file", pairs, "--iterations",
                "2", "--warmup", "1"
            )
            self.assertEqual(batch.returncode, 0, batch.stderr)
            reports = [json.loads(line) for line in batch.stdout.splitlines()]
            self.assertEqual(
                [report["sample_id"] for report in reports], [sample_id, "last"]
            )
            self.assertEqual([report["frames"] for report in reports], [3, 1])
            independent = _run("--json", "--metric", "vmaf-v0", still, still)
            self.assertEqual(independent.returncode, 0, independent.stderr)
            self.assertEqual(
                reports[-1]["score"],
                json.loads(independent.stdout)["score"]
            )
            self.assertEqual(reports[-1]["frame_results"][0]["features"][1], 0)

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
            result = _run("--json", "--metric", "butteraugli", original, distorted)
            self.assertEqual(result.returncode, 1)
            self.assertIn("truncated y4m frame", result.stderr)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
