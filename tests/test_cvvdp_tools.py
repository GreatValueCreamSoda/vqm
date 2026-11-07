#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Check CVVDP command-line contracts and complete-clip benchmark behavior."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def binary(name: str) -> str:
    """Find a configured test executable.

    :param name: Environment variable naming the executable.
    :returns: The configured executable path.
    :raises unittest.SkipTest: If the executable is not configured.
    """
    value = os.environ.get(name)
    if value is None:
        raise unittest.SkipTest(f"{name} is not configured")
    return value


def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one tool with a bounded timeout.

    :param arguments: Executable and its arguments.
    :returns: Captured process output and status.
    """
    return subprocess.run(arguments, capture_output=True, text=True, timeout=120)


def write_y4m(path: Path, fps: str = "30:1", frames: int = 3) -> None:
    """Write a small progressive monochrome sequence.

    :param path: Destination file.
    :param fps: Rational Y4M frame-rate token.
    :param frames: Number of complete frames.
    """
    header = f"YUV4MPEG2 W32 H32 F{fps} Ip Cmono\n".encode()
    contents = b"".join(
        b"FRAME\n" + bytes([32 + 48 * index]) * 1024 for index in range(frames)
    )
    path.write_bytes(header + contents)


def write_late_distortion(path: Path, distorted: bool) -> None:
    """Write four changing frames with an optional final-frame spatial error.

    :param path: Destination file.
    :param distorted: Whether to alter only the final frame.
    """
    contents = bytearray(b"YUV4MPEG2 W32 H32 F30:1 Ip Cmono\n")
    for frame in range(4):
        values = [
            32 + (5 * x + 3 * y + 19 * frame) % 192 for y in range(32)
            for x in range(32)
        ]
        if distorted and frame == 3:
            values = [255 - value for value in values]
        contents.extend(b"FRAME\n" + bytes(values))
    path.write_bytes(contents)


def require_gpu() -> None:
    """Skip GPU-dependent cases when no device is available.

    :raises unittest.SkipTest: If no usable Vulkan device is available.
    :raises AssertionError: If probing fails for another reason.
    """
    probe = run([binary("VQM_TEST_BINARY"), "--probe-gpu"])
    if probe.returncode == 0:
        return
    if "GPU required, but no usable Vulkan device" in probe.stderr:
        raise unittest.SkipTest("no usable Vulkan device")
    raise AssertionError(probe.stderr)


class CvvdpToolTests(unittest.TestCase):
    """Exercise physical settings, timing validation and clip reporting."""

    def test_comparison_runner_lists_cvvdp_latency_arms(self) -> None:
        """Expose CVVDP explicitly without changing the default metric matrix."""
        root = Path(__file__).resolve().parent.parent
        result = run(
            [
                sys.executable,
                str(root / "bench/run_inmemory.py"), "--metric", "cvvdp", "--list"
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 3)
        self.assertIn("ours-resident", result.stdout)
        self.assertIn("ours-streaming", result.stdout)
        self.assertIn("vship-vulkan", result.stdout)

    def test_comparison_runner_rejects_unsupported_modes(self) -> None:
        """Reject throughput and unqualified references before any build/run."""
        root = Path(__file__).resolve().parent.parent
        commands = (
            [
                sys.executable,
                str(root / "bench/run_inmemory.py"), "--metric", "cvvdp", "--mode",
                "throughput"
            ], [
                sys.executable,
                str(root / "bench/benchmark.py"), "--suite", "images", "--metric",
                "cvvdp", "--references", "libjxl"
            ]
        )
        for command in commands:
            result = run(command)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CVVDP", result.stderr)

    def test_rejects_invalid_display_options(self) -> None:
        """Reject invalid display values before opening input files."""
        cli = binary("VQM_CLI_BINARY")
        for flag, value in (("--cvvdp-peak", "nan"), ("--cvvdp-ppd", "0"),
                            ("--cvvdp-contrast", "-1"), ("--cvvdp-ambient", "-1"),
                            ("--cvvdp-display", "automatic")):
            result = run([cli, "--metric", "cvvdp", flag, value, "a", "b"])
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn(flag, result.stderr)

    def test_missing_timing_and_interlace_fail_before_gpu(self) -> None:
        """A video metric must not infer timing or score interlaced fields."""
        cli = binary("VQM_CLI_BINARY")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.y4m"
            for header, diagnostic in ((b"YUV4MPEG2 W32 H32 Ip Cmono\n",
                                        "y4m requires a valid frame rate"),
                                       (b"YUV4MPEG2 W32 H32 F30:1 It Cmono\n",
                                        "only progressive y4m is supported")):
                path.write_bytes(header + b"FRAME\n" + bytes(1024))
                result = run([cli, "--metric", "cvvdp", str(path), str(path)])
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(diagnostic, result.stderr)
                self.assertEqual(result.stdout, "")

    def test_unequal_timing_fails_before_gpu(self) -> None:
        """Reject input frame-rate mismatches before metric construction."""
        cli = binary("VQM_CLI_BINARY")
        with tempfile.TemporaryDirectory() as directory:
            original, distorted = Path(directory) / "a.y4m", Path(directory) / "b.y4m"
            write_y4m(original, "30000:1001")
            write_y4m(distorted, "30:1")
            result = run([cli, "--metric", "cvvdp", str(original), str(distorted)])
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("unequal frame rates", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_cli_consumes_all_frames_and_rejects_truncation(self) -> None:
        """Verify final clip identity and errors after the first valid frame."""
        require_gpu()
        cli = binary("VQM_CLI_BINARY")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.y4m"
            write_y4m(path)
            result = run([cli, "--metric", "cvvdp", "--raw", str(path), str(path)])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), ["frames 3", "10.00000000"])
            path.write_bytes(path.read_bytes()[:-1])
            truncated = run([cli, "--metric", "cvvdp", str(path), str(path)])
            self.assertEqual(truncated.returncode, 1, truncated.stderr)
            self.assertIn("truncated y4m frame", truncated.stderr)
            self.assertEqual(truncated.stdout, "")

    def test_benchmark_reports_complete_clips_and_frame_units(self) -> None:
        """A video iteration scores a whole reset clip, not one repeated pair."""
        bench = binary("VQM_BENCH_BINARY")
        require_gpu()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.y4m"
            write_y4m(path, "24000:1001")
            result = run(
                [
                    bench, "--metric", "cvvdp", "--cvvdp-video", "--pair",
                    str(path),
                    str(path), "--iterations", "2", "--warmup", "1"
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("3 frames", result.stdout)
            self.assertIn("ms/clip", result.stdout)
            self.assertIn("ms/frame", result.stdout)
            self.assertIn("frame rate 24000/1001", result.stdout)
            self.assertIn("score 10.00000000", result.stdout)
            self.assertIn("checksum 30.00000000", result.stdout)

    def test_cli_scores_nonidentical_final_frame(self) -> None:
        """A late spatial error must affect the complete clip score."""
        require_gpu()
        cli = binary("VQM_CLI_BINARY")
        with tempfile.TemporaryDirectory() as directory:
            original, distorted = Path(directory) / "a.y4m", Path(directory) / "b.y4m"
            write_late_distortion(original, distorted=False)
            write_late_distortion(distorted, distorted=True)
            identity = run(
                [cli, "--metric", "cvvdp", "--raw",
                 str(original),
                 str(original)]
            )
            result = run(
                [cli, "--metric", "cvvdp", "--raw",
                 str(original),
                 str(distorted)]
            )
            self.assertEqual(identity.returncode, 0, identity.stderr)
            self.assertEqual(identity.stdout.splitlines(), ["frames 4", "10.00000000"])
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.splitlines()
            self.assertEqual(len(lines), 2, result.stdout)
            self.assertEqual(lines[0], "frames 4")
            self.assertLess(float(lines[1]), 10.0)


if __name__ == "__main__":
    unittest.main()
