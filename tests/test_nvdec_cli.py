#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Compare NVIDIA-decoded clips with software-decoded Y4M through the CLI."""

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_video_cli import _require_gpu, _run

_METRICS = ("ssimulacra2", "butteraugli", "vmaf-v0", "vmaf-v1", "cvvdp", "psnr-hvs")
_FRAMES = 8
_ENCODER_OPTIONS = {
    "libx265": (
        "-preset", "ultrafast", "-x265-params",
        "lossless=1:pools=1:frame-threads=1:log-level=error:bframes=2:b-adapt=0"
    ),
    "av1_nvenc": ("-preset", "p1", "-cq", "20", "-b:v", "0")
}


class NvdecCliGpu(unittest.TestCase):
    """Exercise compressed video with the opt-in NVDEC CLI backend."""

    @classmethod
    def setUpClass(cls) -> None:
        """Check prerequisites and encode the shared 8-bit 4:2:0 pair."""
        _require_gpu()
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise unittest.SkipTest(
                "FFmpeg is required to generate compressed fixtures"
            )
        cls.ffmpeg = ffmpeg
        cls.encoders = cls._ffmpeg("-encoders").stdout.split()
        if "libx265" not in cls.encoders:
            raise unittest.SkipTest("FFmpeg with libx265 is required for HEVC fixtures")
        cls.encoder_formats = cls._ffmpeg("-h", "encoder=libx265").stdout.split()
        cls.directory = Path(cls.enterClassContext(tempfile.TemporaryDirectory()))
        cls.original, cls.distorted = cls._encode_pair("yuv420p")
        cls._require_cuda_format(cls.original)

    @classmethod
    def _ffmpeg(cls, *args: str | Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [cls.ffmpeg, "-hide_banner", "-nostdin", *(str(arg) for arg in args)],
            capture_output=True, text=True, timeout=120, check=False
        )

    @classmethod
    def _encode_pair(cls, pixel_format: str, *,
                     encoder: str = "libx265") -> tuple[Path, Path]:
        return cls._encode(pixel_format, "original", encoder=encoder), cls._encode(
            pixel_format, "distorted", encoder=encoder, distortion=",gblur=sigma=1.2"
        )

    @classmethod
    def _encode(
        cls, pixel_format: str, label: str, *, distortion: str = "",
        frames: int = _FRAMES, encoder: str = "libx265"
    ) -> Path:
        # Stay above NVDEC's minimum HEVC picture size and VMAF v1's width floor.
        path = cls.directory / f"{encoder}-{pixel_format}-{label}.mkv"
        result = cls._ffmpeg(
            "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=224x160:rate=25",
            "-vf", f"format={pixel_format}{distortion}", "-frames:v", str(frames),
            "-fps_mode", "passthrough", "-c:v", encoder, *_ENCODER_OPTIONS[encoder],
            "-color_range", "tv", "-colorspace", "bt470bg", "-color_trc", "bt709",
            "-color_primaries", "bt709", "-chroma_sample_location", "left", path
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        return path

    def _software_decode(self, path: Path, *, expect_b_frames: bool = True) -> Path:
        raw = path.with_suffix(".y4m")
        result = self._ffmpeg(
            "-y", "-loglevel", "info", "-hwaccel", "none", "-i", path, "-vf",
            "showinfo", "-strict", "-1", raw
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        if expect_b_frames:
            self.assertIn(
                "type:B", result.stderr, "fixture must exercise frame reordering"
            )
        return raw

    @classmethod
    def _require_cuda_format(cls, path: Path) -> None:
        # Format support varies by GPU generation. FFmpeg can fall back to
        # software, so check its reported output pixel format as well.
        result = cls._ffmpeg(
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", path, "-f",
            "null", "-"
        )
        if result.returncode or not re.search(r"Video: wrapped_avframe,.*\bcuda\b",
                                              result.stderr):
            raise unittest.SkipTest(
                f"NVDEC does not expose CUDA frames for {path.stem}"
            )

    def _report(self, *args: str | Path) -> dict[str, object]:
        result = _run("--json", *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def _compare(
        self, metric: str, compressed: tuple[Path, Path], raw: tuple[Path, Path]
    ) -> None:
        expected = self._report("--metric", metric, *raw)
        actual = self._report("--metric", metric, *compressed)
        self.assertEqual(actual["frames"], _FRAMES)
        self.assertEqual(actual["metric"], metric)
        self.assertAlmostEqual(actual["score"], expected["score"], delta=1e-6)
        self.assertEqual(actual["frame_results"], expected["frame_results"])
        if metric == "ssimulacra2":
            scores = {frame["score"] for frame in actual["frame_results"]}
            self.assertGreater(len(scores), 1)
            self.assertLess(max(scores), 100)

    def test_all_metrics_match_software_decoded_frames(self) -> None:
        """Compare complete 8-bit 4:2:0 clips, including reordered B frames."""
        pair = self.original, self.distorted
        raw = self._software_decode(pair[0]), self._software_decode(pair[1])
        for metric in _METRICS:
            with self.subTest(metric=metric):
                self._compare(metric, pair, raw)

    def test_default_text_output_uses_nvdec(self) -> None:
        """Detect compressed video content and score it without extra flags."""
        expected = self._report(self.original, self.distorted)
        renamed = self.original.with_suffix(".data")
        shutil.copyfile(self.original, renamed)
        result = _run(renamed, self.distorted)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertAlmostEqual(float(result.stdout), expected["score"], delta=1e-8)

    def test_main10_planes_match_software_decode(self) -> None:
        """Check the high-bit-depth luma and both chroma planes."""
        self._check_optional_format("yuv420p10le")

    def test_main12_planes_match_software_decode(self) -> None:
        """Check the 12-bit sample shift in decoded word planes."""
        self._check_optional_format("yuv420p12le")

    def test_422_planes_match_software_decode(self) -> None:
        """Check half-width, full-height chroma planes where supported."""
        self._check_optional_format("yuv422p")

    def test_444_planes_match_software_decode(self) -> None:
        """Check GPU decode with full-size, independently varying chroma planes."""
        self._check_optional_format("yuv444p")

    def test_av1_nvenc_matches_software_decode(self) -> None:
        """Select a CUDA-capable AV1 decoder when FFmpeg also registers libdav1d."""
        if "av1_nvenc" not in self.encoders:
            self.skipTest("FFmpeg was built without the AV1 NVENC encoder")
        try:
            pair = self._encode_pair("yuv420p", encoder="av1_nvenc")
        except RuntimeError as error:
            self.skipTest(f"AV1 NVENC fixture encoding is unavailable: {error}")
        self._require_cuda_format(pair[0])
        raw = tuple(self._software_decode(path, expect_b_frames=False) for path in pair)
        self._compare("psnr-hvs", pair, raw)

    def _check_optional_format(self, pixel_format: str) -> None:
        if pixel_format not in self.encoder_formats:
            self.skipTest(f"FFmpeg's libx265 encoder does not support {pixel_format}")
        pair = self._encode_pair(pixel_format)
        self._require_cuda_format(pair[0])
        raw = self._software_decode(pair[0]), self._software_decode(pair[1])
        for metric in ("ssimulacra2", "psnr-hvs"):
            with self.subTest(metric=metric):
                self._compare(metric, pair, raw)

    def test_mixed_batch_resets_decoder_and_temporal_history(self) -> None:
        """Switch between raw and compressed pairs with fresh temporal history."""
        raw_original = self._software_decode(self.original)
        raw_distorted = self._software_decode(self.distorted)
        clips = (
            ("raw-changed", raw_original,
             raw_distorted), ("encoded-identity", self.original, self.original),
            ("encoded-changed", self.original,
             self.distorted), ("raw-identity", raw_original, raw_original)
        )
        pairs = self.directory / "clips.tsv"
        pairs.write_text(
            "".join(
                f"{label}\t{original}\t{distorted}\n"
                for label, original, distorted in clips
            ), encoding="utf-8"
        )
        result = _run(
            "--json", "--metric", "vmaf-v0", "--pairs-file", pairs, "--iterations", "2",
            "--warmup", "1"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        reports = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(
            [report["sample_id"] for report in reports],
            [label for label, _, _ in clips]
        )
        self.assertEqual(
            [report["frames"] for report in reports], [_FRAMES] * len(clips)
        )
        for report, (_, original, distorted) in zip(reports, clips, strict=True):
            expected = self._report("--metric", "vmaf-v0", original, distorted)
            self.assertEqual(report["score"], expected["score"])
            self.assertEqual(report["frame_results"], expected["frame_results"])

    def test_unequal_frame_counts_emit_no_result(self) -> None:
        """Reject a complete prefix paired with a longer compressed clip."""
        shorter = self._encode("yuv420p", "shorter", frames=_FRAMES - 1)
        result = _run("--json", self.original, shorter)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("unequal frame counts", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_nvdec_requires_two_compressed_inputs(self) -> None:
        """Reject explicit NVDEC on Y4M and automatic mixed-format pairs."""
        raw = self._software_decode(self.original)
        for args in (("--nvdec", raw, raw), (raw, self.original)):
            with self.subTest(args=args):
                result = _run("--json", *args)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(
                    "NVDEC requires two compressed video inputs", result.stderr
                )
                self.assertEqual(result.stdout, "")

    def test_variable_frame_rate_emits_no_result(self) -> None:
        """Reject a timestamp discontinuity after a successfully decoded prefix."""
        variable = self._encode(
            "yuv420p", "variable", distortion=r",setpts=PTS+2*gte(N\,3)"
        )
        result = _run("--json", variable, variable)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("constant frame rate", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
