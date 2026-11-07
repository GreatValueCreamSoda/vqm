#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""CPU checks for reproducible CID22 VMAF inputs and reference parsing."""

import importlib.util
import json
import shutil
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))
try:
    import numpy as np
    from PIL import Image
except ImportError:
    cid = None
else:
    # Only unavailable optional dependencies skip these tests. Import errors
    # in the modules under test must fail collection and remain visible.
    spec = importlib.util.spec_from_file_location(
        "vmaf_cid22", ROOT / "validation/vmaf_cid22.py"
    )
    cid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cid)
    import vmaf_sdr_cid22 as sdr


@unittest.skipIf(cid is None, "CID22 VMAF preparation needs NumPy and Pillow")
class VmafCid22Tests(unittest.TestCase):
    """Exercise CID22 preparation and reference scoring for the VMAF adapters."""

    @unittest.skipUnless(shutil.which("ffmpeg"), "RGB16 PNG decode requires ffmpeg")
    def test_sdr_source_preserves_16_bit_rgb_low_bits(self) -> None:
        """Preserve the lowest bits when decoding a sixteen-bit source PNG."""

        # A tiny RGB16 PNG written independently of Pillow, which truncates RGB16.
        def chunk(name: str, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data)) + name + data +
                struct.pack(">I", zlib.crc32(name + data))
            )

        pixels = np.array(
            [[[0x0102, 0xABCD, 0xFFFF], [0xFF01, 0x1234, 0x4567]]], dtype=">u2"
        )
        png = (
            b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, 16, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(b"\x00" + pixels.tobytes())) +
            chunk(b"IEND", b"")
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb16.png"
            path.write_bytes(png)
            decoded = sdr.read_source(path)
            self.assertEqual(decoded.dtype, np.uint16)
            np.testing.assert_array_equal(decoded, pixels.astype(np.uint16))
            self.assertNotEqual(int(decoded[0, 0, 0]) & 255, 0)

    def test_sdr_conversion_is_display_referred_16_bit_planar(self) -> None:
        """Convert SDR sources into display-referred sixteen-bit planar input."""
        rgb = np.array(
            [[[0, 0, 0], [255, 255, 255], [255, 0, 0], [128, 128, 128]]], dtype=np.uint8
        )
        actual = np.frombuffer(sdr.convert_rgb(rgb), dtype="<u2").reshape(3, 4)
        np.testing.assert_array_equal(actual[:, 0], [4096, 32768, 32768])
        np.testing.assert_array_equal(actual[:, 1], [60160, 32768, 32768])
        np.testing.assert_array_equal(actual[:, 2], [16015, 26198, 61440])
        linear = ((128 / 255 + 0.055) / 1.055)**2.4
        expected = round((16 + 219 * linear**(1 / 2.4)) * 256)
        self.assertEqual(actual[0, 3], expected)
        self.assertNotEqual(actual[0, 3], round((16 + 219 * 128 / 255) * 256))

    def test_sdr_reference_uses_16_bit_and_checks_scores(self) -> None:
        """Verify the sixteen-bit reference command and its score handling."""
        row = {
            "original_yuv": "ref",
            "distorted_yuv": "dist",
            "width": 17,
            "height": 19,
            "distorted": "dist.png"
        }
        response = SimpleNamespace(
            returncode=0, stdout="frame,v0,v1\n0,42.5,9.25\n", stderr=""
        )
        with mock.patch.object(sdr.subprocess, "run", return_value=response) as run:
            self.assertEqual(
                sdr.reference_score_16(row, "exe", "m0", "m1"), [42.5, 9.25]
            )
            self.assertEqual(run.call_args.args[0][7], "16")
        response.stdout = "frame,v0,v1\n0,nan,2\n"
        with (mock.patch.object(sdr.subprocess, "run", return_value=response),
              self.assertRaisesRegex(ValueError, "non-finite"),
              ):
            sdr.reference_score_16(row, "exe", "m0", "m1")

    def test_sdr_baseline_joins_by_pair_and_rejects_duplicates(self) -> None:
        """Join SDR baseline results by pair identity and reject duplicates."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            path.write_text("original,distorted,v0\na,b,1\nc,d,2\n")
            self.assertEqual(
                sdr.read_score_csv(path, "v0", [("c", "d"), ("a", "b")]), [2, 1]
            )
            path.write_text("original,distorted,v0\na,b,1\na,b,2\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                sdr.read_score_csv(path, "v0", [("a", "b")])

    def test_sdr_baseline_checks_selected_source_hashes(self) -> None:
        """Bind an SDR baseline to the selected sources' hashes."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            report = root / "report"
            report.mkdir()
            for name in ("ref", "dist", "other"):
                (data / name).write_text(name)
            mcos = data / "mcos.csv"
            mcos.write_text("annotations")
            manifest = report / "inputs.csv"
            manifest.write_text(
                "original,distorted,original_source_sha256,distorted_source_sha256\n"
                f"ref,dist,{cid.digest(data / 'ref')},{cid.digest(data / 'dist')}\n"
                f"ref,other,{cid.digest(data / 'ref')},{cid.digest(data / 'other')}\n"
            )
            (report / "summary.json").write_text(
                json.dumps(
                    {
                        "conversion": cid.CONVERSION,
                        "mcos_sha256": cid.digest(mcos),
                        "input_manifest_sha256": cid.digest(manifest),
                    }
                )
            )
            (data / "other").write_text("changed, but outside subset")
            result = sdr.validate_baseline(report, [("ref", "dist")], data, mcos, 1)
            self.assertEqual(result["source_files_verified"], 2)
            self.assertTrue(result["manifest_verified"])
            (data / "dist").write_text("changed")
            with self.assertRaisesRegex(ValueError, "source hash differs"):
                sdr.validate_baseline(report, [("ref", "dist")], data, mcos, 1)

    def test_black_white_and_red_planar_samples(self) -> None:
        """Pin the planar codes for black, white and red samples."""
        rgb = np.array([[[0, 0, 0], [255, 255, 255], [255, 0, 0]]], dtype=np.uint8)
        actual = np.frombuffer(cid.convert_rgb(rgb), dtype="<u2").reshape(3, 3)
        np.testing.assert_array_equal(
            actual, [[64, 940, 250], [512, 512, 409], [512, 512, 960]]
        )

    def test_cache_is_repaired_and_transparency_rejected(self) -> None:
        """Repair a damaged conversion cache and reject transparent input."""
        with tempfile.TemporaryDirectory() as directory:
            path, cache = Path(directory) / "image.png", Path(directory) / "cache"
            cache.mkdir()
            Image.new("RGB", (17, 19), (12, 34, 56)).save(path)
            first = cid.convert_file(path, cache)
            target = Path(first["file"])
            original = target.read_bytes()
            target.write_bytes(b"corrupt")
            second = cid.convert_file(path, cache)
            self.assertEqual(first, second)
            self.assertEqual(target.read_bytes(), original)
            Image.new("RGBA", (17, 19), (12, 34, 56, 0)).save(path)
            with self.assertRaisesRegex(ValueError, "non-opaque"):
                cid.convert_file(path, cache)

    def test_reference_requires_one_finite_frame(self) -> None:
        """Require one finite reference result for a still-image comparison."""
        row = {
            "original_yuv": "ref",
            "distorted_yuv": "dist",
            "width": 17,
            "height": 19,
            "distorted": "dist.png"
        }
        with mock.patch.object(cid.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout="frame,v0,v1\n0,42.5,9.25\n", stderr="")):
            self.assertEqual(cid.reference_score(row, "exe", "m0", "m1"), [42.5, 9.25])
        for output in ("frame,v0,v1\n0,nan,2\n", "frame,v0,v1\n0,1,2\n1,3,4\n"):
            with (mock.patch.object(cid.subprocess, "run",
                                    return_value=SimpleNamespace(returncode=0,
                                                                 stdout=output,
                                                                 stderr="")),
                  self.assertRaises(ValueError),
                  ):
                cid.reference_score(row, "exe", "m0", "m1")

    def test_y4m_wrapper_keeps_raw_little_endian_samples(self) -> None:
        """Preserve the little-endian sample payload in the Y4M wrapper."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "picture.yuv"
            payload = np.array([64, 255, 512, 940, 960, 1023], dtype="<u2").tobytes()
            raw.write_bytes(payload)
            target = Path(cid.write_y4m(str(raw), root, 2, 1))
            header = b"YUV4MPEG2 W2 H1 F1:1 Ip A0:0 C444p10\nFRAME\n"
            self.assertEqual(target.read_bytes(), header + payload)
            target.write_bytes(b"corrupt")
            cid.write_y4m(str(raw), root, 2, 1)
            self.assertEqual(target.read_bytes(), header + payload)

    def test_vqm_model_selection_and_score_output(self) -> None:
        """Select the requested VMAF model and parse its emitted score."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "build").mkdir()
            (root / "build/vqm_bench").touch()
            output = root / "report"
            output.mkdir()
            raw = root / "picture.yuv"
            raw.write_bytes(
                np.array([64, 255, 512, 940, 960, 1023], dtype="<u2").tobytes()
            )
            rows = [
                {
                    "original": "a",
                    "distorted": "b",
                    "original_yuv": str(raw),
                    "distorted_yuv": str(raw),
                    "width": 2,
                    "height": 1
                }
            ]
            response = SimpleNamespace(returncode=0, stdout="42.5\n", stderr="")
            with mock.patch.object(cid, "ROOT", root), \
                    mock.patch.object(cid.subprocess, "run", return_value=response) as run:
                self.assertEqual(cid.vqm_scores(rows, output, 1, "v1"), [42.5])
                argv = run.call_args.args[0]
                self.assertEqual(argv[argv.index("--metric") + 1], "vmaf-v1")
                self.assertEqual(
                    (output / "vqm-v1.csv").read_text(),
                    "original,distorted,v1\na,b,42.5\n"
                )


if __name__ == "__main__":
    unittest.main()
