#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Check CVVDP command-line contracts and complete-clip benchmark behavior."""

import os
import re
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
    return subprocess.run(
        arguments, capture_output=True, text=True, timeout=120
    )


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


def write_late_distortion(
    path: Path, distorted: bool, *, width: int = 32, height: int = 32
) -> None:
    """Write four changing frames with an optional final-frame spatial error.

    :param path: Destination file.
    :param distorted: Whether to alter only the final frame.
    :param width: Frame width in pixels.
    :param height: Frame height in pixels.
    """
    contents = bytearray(
        f"YUV4MPEG2 W{width} H{height} F30:1 Ip Cmono\n".encode()
    )
    for frame in range(4):
        values = [
            32 + (5 * x + 3 * y + 19 * frame) % 192 for y in range(height)
            for x in range(width)
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
        """Expose CVVDP without changing the default metric matrix."""
        root = Path(__file__).resolve().parent.parent
        result = run(
            [
                sys.executable,
                str(root / "bench/run_inmemory.py"), "--metric", "cvvdp",
                "--list"
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 4)
        self.assertIn("ours-resident", result.stdout)
        self.assertIn("ours-streaming", result.stdout)
        self.assertIn("vship-vulkan", result.stdout)
        self.assertIn("vship-fast-vulkan", result.stdout)

    def test_comparison_runner_accepts_both_cvvdp_vulkan_peers(self) -> None:
        """Accept the pinned Vulkan peers before validating warmup counts."""
        root = Path(__file__).resolve().parent.parent
        for references in ("vship-fast-vulkan",
                           "vship-vulkan,vship-fast-vulkan"):
            result = run(
                [
                    sys.executable,
                    str(root / "bench/benchmark.py"), "--suite", "images",
                    "--metric", "cvvdp", "--references", references, "--warmup",
                    "-1"
                ]
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--warmup must be non-negative", result.stderr)

    def test_comparison_runner_rejects_unsupported_modes(self) -> None:
        """Reject throughput and unqualified references before any build/run."""
        root = Path(__file__).resolve().parent.parent
        commands = (
            [
                sys.executable,
                str(root / "bench/run_inmemory.py"), "--metric", "cvvdp",
                "--mode", "throughput"
            ], [
                sys.executable,
                str(root / "bench/benchmark.py"), "--suite", "images",
                "--metric", "cvvdp", "--references", "libjxl"
            ]
        )
        for command in commands:
            result = run(command)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CVVDP", result.stderr)

    def test_rejects_invalid_display_options(self) -> None:
        """Reject invalid display values before opening input files."""
        cli = binary("VQM_CLI_BINARY")
        cases = (
            ("--cvvdp-peak", "nan"), ("--cvvdp-ppd",
                                      "0"), ("--cvvdp-contrast",
                                             "-1"), ("--cvvdp-ambient", "-1"),
            ("--cvvdp-display",
             "automatic"), ("--cvvdp-display",
                            "standard-4k"), ("--cvvdp-display", "standard-fhd"),
            ("--cvvdp-display",
             "standard-sdr"), ("--cvvdp-display", "compact-sdr"),
            ("--cvvdp-display", "standard-hdr"), ("--cvvdp-aspect", "16"),
            ("--cvvdp-aspect", "16:0"), ("--cvvdp-aspect", "16:9:1"),
            ("--cvvdp-diagonal", "inf"), ("--cvvdp-distance", "0"),
        )
        for flag, value in cases:
            result = run([cli, "--metric", "cvvdp", flag, value, "a", "b"])
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn(flag, result.stderr.splitlines()[0])

    def test_rejects_conflicting_display_geometry(self) -> None:
        """Reject a geometry override combined with explicit PPD."""
        cli = binary("VQM_CLI_BINARY")
        for flag, value in (("--cvvdp-aspect", "16:9"),
                            ("--cvvdp-diagonal", "27"), ("--cvvdp-distance",
                                                         "0.7")):
            with self.subTest(flag=flag):
                result = run(
                    [
                        cli, "--metric", "cvvdp", "--cvvdp-ppd", "30", flag,
                        value, "a", "b"
                    ]
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("--cvvdp-ppd", result.stderr.splitlines()[0])
                self.assertEqual(result.stdout, "")

    def test_display_geometry_scores_with_aspect_fitted_ppd(self) -> None:
        """Match PPD for full-screen, letterbox and pillarbox content."""
        require_gpu()
        cli = binary("VQM_CLI_BINARY")
        with tempfile.TemporaryDirectory() as directory:
            original, distorted = Path(directory
                                       ) / "a.y4m", Path(directory) / "b.y4m"
            write_late_distortion(
                original, distorted=False, width=320, height=180
            )
            write_late_distortion(
                distorted, distorted=True, width=320, height=180
            )
            # Values from the central-pixel formula for an 8-inch panel at
            # 0.7 m, using its width for 4:3 and height for 21:9.
            cases = (
                ("16:9", "22.07479467324551"), ("4:3", "24.049813582989522"),
                ("21:9", "27.473688241145847"),
            )
            scores = []
            for aspect, ppd in cases:
                inputs = [str(original), str(distorted)]
                command = [cli, "--metric", "cvvdp", "--raw"]
                result = run(
                    [
                        *command, "--cvvdp-aspect", aspect, "--cvvdp-diagonal",
                        "8", "--cvvdp-distance", "0.7", *inputs
                    ]
                )
                explicit = run([*command, "--cvvdp-ppd", ppd, *inputs])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(explicit.returncode, 0, explicit.stderr)
                frames, score = result.stdout.splitlines()
                self.assertEqual(frames, "frames 4")
                self.assertLess(float(score), 10.0)
                self.assertEqual(result.stdout, explicit.stdout)
                scores.append(score)
            self.assertEqual(len(set(scores)), len(cases))

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
            original, distorted = Path(directory
                                       ) / "a.y4m", Path(directory) / "b.y4m"
            write_y4m(original, "30000:1001")
            write_y4m(distorted, "30:1")
            result = run(
                [cli, "--metric", "cvvdp",
                 str(original),
                 str(distorted)]
            )
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
            result = run(
                [cli, "--metric", "cvvdp", "--raw",
                 str(path),
                 str(path)]
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                result.stdout.splitlines(), ["frames 3", "10.00000000"]
            )
            path.write_bytes(path.read_bytes()[:-1])
            truncated = run([cli, "--metric", "cvvdp", str(path), str(path)])
            self.assertEqual(truncated.returncode, 1, truncated.stderr)
            self.assertIn("truncated y4m frame", truncated.stderr)
            self.assertEqual(truncated.stdout, "")

    def test_benchmark_reports_complete_clips_and_frame_units(self) -> None:
        """A benchmark scores a reset clip using a partial display override."""
        bench = binary("VQM_BENCH_BINARY")
        require_gpu()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.y4m"
            write_y4m(path, "24000:1001")
            result = run(
                [
                    bench, "--metric", "cvvdp", "--cvvdp-video", "--pair",
                    str(path),
                    str(path), "--iterations", "2", "--warmup", "1",
                    "--cvvdp-diagonal", "8"
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("3 frames", result.stdout)
            self.assertIn("ms/clip", result.stdout)
            self.assertIn("ms/frame", result.stdout)
            self.assertIn("frame rate 24000/1001", result.stdout)
            # A diagonal override keeps the default 16:9 aspect and 0.7472 m.
            self.assertIn("ppd 4.189031,", result.stdout)
            self.assertIn("score 10.00000000", result.stdout)
            self.assertIn("checksum 30.00000000", result.stdout)

    def test_benchmark_resident_video_matches_streaming(self) -> None:
        """Preserve temporal scores and resets across both input paths."""
        bench = binary("VQM_BENCH_BINARY")
        require_gpu()
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original.y4m"
            distorted = Path(directory) / "distorted.y4m"
            write_late_distortion(original, distorted=False)
            write_late_distortion(distorted, distorted=True)
            checksums = []
            for flags in ((), ("--vulkan-buffers", )):
                result = run(
                    [
                        bench, "--metric", "cvvdp", "--cvvdp-video", "--pair",
                        str(original),
                        str(distorted), "--iterations", "2", "--warmup", "1",
                        *flags
                    ]
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("4 frames", result.stdout)
                self.assertIn("frames/s", result.stdout)
                checksums.append(
                    float(re.search(r"checksum ([\d.]+)", result.stdout)[1])
                )
            self.assertAlmostEqual(*checksums, delta=1e-6)
            self.assertLess(checksums[0], 30.0)
            distorted.write_bytes(distorted.read_bytes()[:-1])
            truncated = run(
                [
                    bench, "--metric", "cvvdp", "--cvvdp-video",
                    "--vulkan-buffers", "--pair",
                    str(original),
                    str(distorted)
                ]
            )
            self.assertNotEqual(truncated.returncode, 0)
            self.assertIn("truncated y4m frame", truncated.stderr)

    def test_physical_presets_resolve_ppd_at_each_input_size(self) -> None:
        """Resolve PPD by geometry and apply peak brightness independently."""
        bench = binary("VQM_BENCH_BINARY")
        require_gpu()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.y4m"
            # Standard and compact are both 16:9, at 30 in and 0.7472 m or
            # 24 in and 0.6 m respectively.
            cases = (
                (320, 180, (), "6.28354146", 200),
                (640, 360, ("--cvvdp-display", "standard"), "12.5670769", 200),
                (320, 180, ("--cvvdp-display", "compact"), "6.30708789", 200), (
                    640, 360,
                    ("--cvvdp-peak", "1500", "--cvvdp-display",
                     "compact"), "12.6141697", 1500
                ),
            )
            for width, height, flags, ppd, peak in cases:
                write_late_distortion(
                    path, distorted=False, width=width, height=height
                )
                result = run(
                    [
                        bench, "--metric", "cvvdp", "--cvvdp-video", "--pair",
                        str(path),
                        str(path), "--iterations", "1", "--warmup", "0", *flags
                    ]
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    (
                        f"ppd {ppd}, peak {peak} nit, contrast 1000, ambient "
                        f"250 lux"
                    ), result.stdout
                )

    def test_cli_scores_nonidentical_final_frame(self) -> None:
        """A late spatial error must affect the complete clip score."""
        require_gpu()
        cli = binary("VQM_CLI_BINARY")
        with tempfile.TemporaryDirectory() as directory:
            original, distorted = Path(directory
                                       ) / "a.y4m", Path(directory) / "b.y4m"
            write_late_distortion(original, distorted=False)
            write_late_distortion(distorted, distorted=True)
            identity = run(
                [
                    cli, "--metric", "cvvdp", "--raw",
                    str(original),
                    str(original)
                ]
            )
            result = run(
                [
                    cli, "--metric", "cvvdp", "--raw",
                    str(original),
                    str(distorted)
                ]
            )
            self.assertEqual(identity.returncode, 0, identity.stderr)
            self.assertEqual(
                identity.stdout.splitlines(), ["frames 4", "10.00000000"]
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.splitlines()
            self.assertEqual(len(lines), 2, result.stdout)
            self.assertEqual(lines[0], "frames 4")
            self.assertLess(float(lines[1]), 10.0)


if __name__ == "__main__":
    unittest.main()
