#!/usr/bin/env python3
"""Checks for generated VMAF fixtures and the optional pinned CPU oracle."""

import csv
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import make_vmaf_fixtures as fixtures

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build/vmaf-build"
SOURCE = ROOT / "build/vmaf-src"
ORACLE_AVAILABLE = (BUILD / "vmaf_reference").is_file() and (BUILD /
                                                             "tools/vmaf").is_file()


class FixtureTests(unittest.TestCase):
    """Verify deterministic fixture generation and sample precision."""

    def test_low_bit_cases_keep_all_upper_eight_bits(self) -> None:
        """Confirm low-bit distortions preserve the highest eight luma bits."""
        width, height, frames = 16, 16, 3
        for depth in (10, 12):
            with self.subTest(depth=depth):
                self._check_low_bits(depth, width, height, frames)

    def _check_low_bits(self, depth: int, width: int, height: int, frames: int) -> None:
        """Compare luma and unchanged chroma across one complete fixture.

        :param depth: Input bit depth.
        :param width: Width in pixels.
        :param height: Height in pixels.
        :param frames: Number of frames to compare.
        """
        ref, dist = fixtures.make_pair(width, height, depth, frames, True)
        self.assertEqual(len(ref), width * height * 3 // 2 * 2 * frames)
        self.assertEqual(len(ref), len(dist))
        frame_size = width * height * 3 // 2 * 2
        for frame in range(frames):
            start = frame * frame_size
            n = width * height
            a = struct.unpack_from(f"<{n}H", ref, start)
            b = struct.unpack_from(f"<{n}H", dist, start)
            self.assertTrue(
                all(
                    (x >> (depth - 8)) == (y >> (depth - 8))
                    for x, y in zip(a, b, strict=True)
                )
            )
            self.assertTrue(all(x != y for x, y in zip(a, b, strict=True)))
            self.assertEqual(
                ref[start + n * 2:start + frame_size],
                dist[start + n * 2:start + frame_size]
            )

    def test_fixture_generation_is_deterministic(self) -> None:
        """Generate identical manifests and fixtures in separate directories."""
        with tempfile.TemporaryDirectory() as directory:
            cmd = [
                sys.executable,
                str(ROOT / "tests/make_vmaf_fixtures.py"), "--out-dir", directory,
                "--width", "32", "--height", "32", "--frames", "2"
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            manifest = Path(directory) / "manifest.json"
            first = json.loads(manifest.read_text())
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.assertEqual(first, json.loads(manifest.read_text()))
            self.assertEqual(len(first["cases"]), 5)


@unittest.skipUnless(ORACLE_AVAILABLE, "pinned libvmaf oracle not built")
class LibvmafOracleTests(unittest.TestCase):
    """Compare the reference harness with the pinned native libvmaf CLI."""

    def test_joint_444_models_match_native_cli(self) -> None:
        """Compare both YUV444 model scores against the native CLI."""
        width, height = 512, 512
        n = width * height
        reference = [
            64 + ((x * 11 + y * 7) % 877) for y in range(height) for x in range(width)
        ]
        distorted = [
            max(64, min(940, value + (i % 15) - 7))
            for i, value in enumerate(reference)
        ]
        chroma = [512 + (i % 31) - 15 for i in range(n)]
        models = [
            SOURCE / "model/vmaf_float_v0.6.1.json",
            SOURCE / "model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json"
        ]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name, luma in [("ref", reference), ("dist", distorted)]:
                (base / f"{name}.yuv").write_bytes(
                    struct.pack(f"<{n * 3}H", *(luma + chroma + chroma))
                )
            output = subprocess.check_output(
                [
                    str(BUILD / "vmaf_reference"), "models",
                    str(models[0]),
                    str(base / "ref.yuv"),
                    str(base / "dist.yuv"),
                    str(width),
                    str(height), "10",
                    str(models[1])
                ], text=True
            )
            rows = list(csv.DictReader(io.StringIO(output)))
            self.assertEqual(len(rows), 1)
            result = base / "cli.json"
            subprocess.run(
                [
                    str(BUILD / "tools/vmaf"), "--reference",
                    str(base / "ref.yuv"), "--distorted",
                    str(base / "dist.yuv"), "--width",
                    str(width), "--height",
                    str(height), "--pixel_format", "444", "--bitdepth", "10", "--model",
                    f"path={models[0]}:name=v0", "--model", f"path={models[1]}:name=v1",
                    "--json", "--output",
                    str(result), "--threads", "1", "--quiet",
                ], check=True, capture_output=True, text=True
            )
            frame = json.loads(result.read_text())["frames"][0]
            for name in ("v0", "v1"):
                self.assertAlmostEqual(
                    float(rows[0][name]), frame["metrics"][name], delta=5.1e-7
                )

    def test_full_precision_driver_agrees_with_reference_cli(self) -> None:
        """Compare float and integer drivers at every supported bit depth."""
        base = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for depth, low_bits in ((8, False), (10, True), (12, True)):
            ref, dist = fixtures.make_pair(64, 48, depth, 4, low_bits)
            (base / "ref.yuv").write_bytes(ref)
            (base / "dist.yuv").write_bytes(dist)
            for kind in ("float", "integer"):
                self.check_reference_cli(base, kind, depth)

    def check_reference_cli(self, base: Path, kind: str, depth: int) -> None:
        """Compare the reference harness with a native CLI run for one input.

        :param base: Directory containing the reference and distorted input.
        :param kind: Float or integer VMAF feature mode.
        :param depth: Input bit depth.
        """
        with self.subTest(kind=kind, depth=depth):
            model = SOURCE / "model" / (
                "vmaf_float_v0.6.1.json" if kind == "float" else "vmaf_v0.6.1.json"
            )
            output = subprocess.check_output(
                [
                    str(BUILD / "vmaf_reference"), kind,
                    str(model),
                    str(base / "ref.yuv"),
                    str(base / "dist.yuv"), "64", "48",
                    str(depth),
                ], text=True
            )
            rows = list(csv.DictReader(io.StringIO(output)))
            self.assertEqual(len(rows), 4)
            result = base / "cli.json"
            subprocess.run(
                [
                    str(BUILD / "tools/vmaf"), "--reference",
                    str(base / "ref.yuv"), "--distorted",
                    str(base / "dist.yuv"), "--width", "64", "--height", "48",
                    "--pixel_format", "420", "--bitdepth",
                    str(depth), "--model", f"path={model}:name=test_score", "--json",
                    "--output",
                    str(result), "--threads", "1", "--quiet",
                ], check=True, capture_output=True, text=True
            )
            upstream = json.loads(result.read_text())["frames"]
            self.assertEqual(len(upstream), len(rows))
            for row, frame in zip(rows, upstream, strict=True):
                self._check_reference_frame(row, frame, kind)

    def _check_reference_frame(
        self, row: dict[str, str], frame: dict[str, object], kind: str
    ) -> None:
        """Compare one frame's index, model score and feature values.

        :param row: CSV frame result from the harness.
        :param frame: JSON frame result from the native CLI.
        :param kind: Float or integer feature extractor.
        """
        self.assertEqual(int(row["frame"]), frame["frameNum"])
        for feature in ("adm2", "motion2", "vif_scale0", "vif_scale1", "vif_scale2",
                        "vif_scale3", "vmaf"):
            key = "test_score" if feature == "vmaf" else (
                ("integer_" if kind == "integer" else "") + feature
            )
            self.assertAlmostEqual(
                float(row[feature]), frame["metrics"][key], delta=5.1e-7,
                msg=f"{kind} {key}"
            )

    def test_low_bits_are_measured_and_truncated_sequences_rejected(self) -> None:
        """Measure low-bit distortion and reject truncated frame payloads."""
        ref, dist = fixtures.make_pair(64, 48, 10, 1, True)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            ref_path, dist_path = base / "ref.yuv", base / "dist.yuv"
            ref_path.write_bytes(ref)
            dist_path.write_bytes(dist)
            model = SOURCE / "model/vmaf_float_v0.6.1.json"
            command = [
                str(BUILD / "vmaf_reference"), "float",
                str(model),
                str(ref_path),
                str(dist_path), "64", "48", "10"
            ]
            changed = next(
                csv.DictReader(
                    io.StringIO(subprocess.check_output(command, text=True))
                )
            )
            command[4] = str(ref_path)
            identical = next(
                csv.DictReader(
                    io.StringIO(subprocess.check_output(command, text=True))
                )
            )
            self.assertEqual(float(changed["motion2"]), 0.0)
            self.assertLess(
                float(changed["vif_scale0"]), float(identical["vif_scale0"])
            )
            command[4] = str(dist_path)
            dist_path.write_bytes(dist[:-1])
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("complete frames", failed.stderr)

    def test_adm_debug_scale_components_match_native_cli(self) -> None:
        """Compare ADM debug scale components with native CLI features."""
        base = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for depth, low_bits in ((8, False), (10, True), (12, True)):
            with self.subTest(depth=depth):
                self._check_adm_debug(base, depth, low_bits)

    def _check_adm_debug(self, base: Path, depth: int, low_bits: bool) -> None:
        """Compare ADM diagnostic output for a single sample format.

        :param base: Directory for reference and distorted clips.
        :param depth: Input bit depth.
        :param low_bits: Distort only the lowest sample bits.
        """
        ref, dist = fixtures.make_pair(64, 48, depth, 4, low_bits)
        (base / "ref.yuv").write_bytes(ref)
        (base / "dist.yuv").write_bytes(dist)
        output = subprocess.check_output(
            [
                str(BUILD / "vmaf_reference"), "adm-debug", "-",
                str(base / "ref.yuv"),
                str(base / "dist.yuv"), "64", "48",
                str(depth),
            ], text=True
        )
        rows = list(csv.DictReader(io.StringIO(output)))
        result = base / "adm.json"
        subprocess.run(
            [
                str(BUILD / "tools/vmaf"), "--reference",
                str(base / "ref.yuv"), "--distorted",
                str(base / "dist.yuv"), "--width", "64", "--height", "48",
                "--pixel_format", "420", "--bitdepth",
                str(depth), "--no_prediction", "--feature", "float_adm=debug=true",
                "--json", "--output",
                str(result), "--threads", "1", "--quiet",
            ], check=True, capture_output=True, text=True
        )
        upstream = json.loads(result.read_text())["frames"]
        self.assertEqual(len(rows), len(upstream))
        for row, frame in zip(rows, upstream, strict=True):
            self._check_adm_frame(row, frame)

    def _check_adm_frame(self, row: dict[str, str], frame: dict[str, object]) -> None:
        """Compare the ADM score and per-scale terms for one frame.

        :param row: CSV frame result from the diagnostic harness.
        :param frame: JSON frame result from the native CLI.
        """
        self.assertEqual(int(row["frame"]), frame["frameNum"])
        for scale in range(4):
            for kind in ("num", "den"):
                self.assertAlmostEqual(
                    float(row[f"{kind}{scale}"]),
                    frame["metrics"][f"adm_{kind}_scale{scale}"], delta=5.1e-7,
                )
        self.assertAlmostEqual(
            float(row["adm2"]), frame["metrics"]["adm2"], delta=5.1e-7
        )


if __name__ == "__main__":
    unittest.main()
