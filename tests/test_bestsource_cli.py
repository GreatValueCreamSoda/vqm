#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Check automatic BestSource decoding and indexed frame ranges."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_video_cli import _binary, _require_gpu

_START = 174
_COUNT = 24
_METRICS = ("ssimulacra2", "butteraugli", "vmaf-v0", "vmaf-v1", "cvvdp", "psnr-hvs")
_UNSUPPORTED = (
    "GPU decoding unavailable:", "GPU can't decode this track",
    "BestSource requires Vulkan opaque FD memory and semaphore sharing"
)


class BestSourceCliGpu(unittest.TestCase):
    """Exercise BestSource seeking and device-local planar export."""

    @classmethod
    def setUpClass(cls) -> None:
        """Check fixture and GPU prerequisites."""
        _require_gpu()
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise unittest.SkipTest("FFmpeg is required to generate fixtures")
        cls.ffmpeg = ffmpeg
        if "libx264" not in cls._ffmpeg("-encoders").split():
            raise unittest.SkipTest("FFmpeg with libx264 is required for this test")

    @classmethod
    def _ffmpeg(cls, *args: str | Path) -> str:
        result = subprocess.run(
            [cls.ffmpeg, "-hide_banner", "-nostdin", *(str(arg) for arg in args)],
            capture_output=True, text=True, timeout=120, check=False
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout

    def _fixture(self, path: Path, *, distortion: str = "", frames: int = 216) -> None:
        self._ffmpeg(
            "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x192:rate=30",
            "-vf", f"format=yuv420p{distortion}", "-frames:v", str(frames), "-c:v",
            "libx264", "-threads", "2", "-g", "24", "-sc_threshold", "0", "-bf", "3",
            "-x264-params", "open-gop=1", "-color_range", "tv", "-colorspace",
            "bt470bg", "-color_trc", "bt709", "-color_primaries", "bt709",
            "-chroma_sample_location", "left", path
        )

    def _software_decode(self, path: Path, *args: str) -> Path:
        raw = path.with_suffix(".y4m")
        self._ffmpeg("-loglevel", "error", "-hwaccel", "none", "-i", path, *args, raw)
        return raw

    def _run(self, directory: Path, *args:
             str | Path) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [_binary(), *(str(arg) for arg in args)], env={
                **os.environ, "XDG_CACHE_HOME": str(directory)
            }, capture_output=True, text=True, timeout=120, check=False
        )
        if result.returncode and any(reason in result.stderr
                                     for reason in _UNSUPPORTED):
            self.skipTest(result.stderr.strip())
        return result

    def _reports(self, directory: Path, *args: str | Path) -> list[dict[str, object]]:
        result = self._run(directory, "--json", *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines()]

    def test_indexed_range_matches_software_decode(self) -> None:
        """Check frame order, both chroma planes and fresh temporal history."""
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        original, distorted = directory / "original.mp4", directory / "distorted.mp4"
        self._fixture(original)
        self._fixture(distorted, distortion=",gblur=sigma=1.2")
        # This range crosses an open GOP boundary and has a usable seek point
        # after BestSource's 100-frame linear threshold.
        actual = self._reports(
            directory, "--metric", "all", "--start-frame", str(_START), "--frame-count",
            str(_COUNT), original, distorted
        )
        trim = ("-vf", f"trim=start_frame={_START}:end_frame={_START + _COUNT}")
        expected = self._reports(
            directory, "--metric", "all", self._software_decode(original, *trim),
            self._software_decode(distorted, *trim)
        )
        self.assertEqual(tuple(report["metric"] for report in actual), _METRICS)
        for got, want in zip(actual, expected, strict=True):
            self.assertEqual(got["metric"], want["metric"])
            self.assertEqual(got["frames"], _COUNT)
            self.assertEqual(got["score"], want["score"])
            self.assertEqual(got["frame_results"], want["frame_results"])

    def test_default_command_detects_video_by_content(self) -> None:
        """Score a whole compressed clip automatically with the default metric."""
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        original, distorted = directory / "original.mp4", directory / "distorted.mp4"
        self._fixture(original, frames=8)
        self._fixture(distorted, frames=8, distortion=",gblur=sigma=1.2")
        renamed = original.rename(original.with_suffix(".data"))
        expected, = self._reports(directory, renamed, distorted)
        result = self._run(directory, renamed, distorted)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(expected["frames"], 8)
        self.assertAlmostEqual(float(result.stdout), expected["score"], delta=1e-8)

    def test_mixed_batch_keeps_temporal_history_separate(self) -> None:
        """Run compressed and Y4M pairs on the same explicitly selected device."""
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        listing = self._run(directory, "--list-devices")
        self.assertEqual(listing.returncode, 0, listing.stderr)
        device = next(
            (
                line.split()[0]
                for line in listing.stdout.splitlines() if " GPU)" in line
            ), None,
        )
        if device is None:
            self.skipTest("no Vulkan GPU listed for hardware decoding")
        original, distorted = directory / "original.mp4", directory / "distorted.mp4"
        self._fixture(original, frames=8)
        self._fixture(distorted, frames=8, distortion=",gblur=sigma=1.2")
        raw_original = self._software_decode(original)
        raw_distorted = self._software_decode(distorted)
        pairs = directory / "pairs.tsv"
        pairs.write_text(
            f"encoded\t{original}\t{distorted}\nraw\t{raw_original}\t{raw_distorted}\n",
            encoding="utf-8"
        )
        encoded, raw = self._reports(
            directory, "--device", device, "--metric", "vmaf-v0", "--pairs-file", pairs,
            "--iterations", "2", "--warmup", "1"
        )
        self.assertEqual((encoded["sample_id"], raw["sample_id"]), ("encoded", "raw"))
        self.assertEqual(encoded["frames"], 8)
        self.assertEqual(encoded["score"], raw["score"])
        self.assertEqual(encoded["frame_results"], raw["frame_results"])

    def test_frame_range_requires_compressed_video(self) -> None:
        """Reject ranges on Y4M without initializing a video decoder."""
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        raw = directory / "frames.y4m"
        raw.write_bytes(b"YUV4MPEG2 W32 H32 F30:1 Ip C420\n")
        result = self._run(directory, "--start-frame", "0", raw, raw)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("require two compressed video inputs", result.stderr)
        self.assertEqual(result.stdout, "")
