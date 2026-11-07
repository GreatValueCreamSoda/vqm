#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""CLI argument validation and scoring integration tests."""

import io
import os
import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path

import make_goldens


class CliTests(unittest.TestCase):
    """Exercise argument handling and decoding through the installed CLI."""

    @classmethod
    def setUpClass(cls) -> None:
        """Locate the configured CLI executable."""
        path = os.environ.get("VQM_CLI_BINARY")
        if not path:
            raise unittest.SkipTest("CLI binary not configured")
        cls.cli = Path(path)

    def test_malformed_images_fail_without_a_gpu(self) -> None:
        """Reject malformed image data before constructing a GPU pipeline."""
        # Decoding must reject malformed files before pipeline construction.
        # The very large dimension fits size_t but its plane layout does not.
        max_size = sys.maxsize * 2 + 1
        cases = {
            "wrapped-width": (
                f"YUV4MPEG2 W{max_size + 9} H8 C444\nFRAME\n".encode() +
                bytes([128]) * 192, "invalid y4m width"
            ),
            "huge-layout": (
                f"YUV4MPEG2 W{max_size} H2 C444\nFRAME\n".encode(),
                "image layout exceeds size_t"
            ),
            "truncated-y4m": (b"YUV4MPEG2 W8 H8 C444\nFRAME\n", "truncated y4m frame"),
            "invalid-frame": (
                b"YUV4MPEG2 W8 H8 C444\nFRAMEoops\n" + bytes(192),
                "y4m file has no frame"
            ),
            "truncated-png": (b"\x89PNG\r\n\x1a\n", "Could not load original image"),
            "truncated-jpeg": (b"\xff\xd8", "Could not load original image"),
        }
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for name, (content, expected) in cases.items():
            with self.subTest(name=name):
                image = directory / name
                image.write_bytes(content)
                result = subprocess.run(
                    [str(self.cli), str(image), str(image)], capture_output=True,
                    text=True, timeout=5,
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertIn(expected, result.stderr)

    def test_jpeg_scan_truncation_is_rejected_before_gpu_creation(self) -> None:
        """Reject truncated JPEG entropy scans while accepting complete image data."""
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        env = os.environ.copy()
        # Accepted controls stop at device creation without using a GPU.
        missing_driver = str(directory / "no-such-driver.json")
        env["VK_DRIVER_FILES"] = missing_driver
        env["VK_ICD_FILENAMES"] = missing_driver
        for name, (content, accepted) in self._jpeg_cases().items():
            with self.subTest(name=name):
                image = directory / f"{name}.jpg"
                image.write_bytes(content)
                self._check_jpeg_decoding(image, env, accepted)

    def _jpeg_cases(self) -> dict[str, tuple[bytes | bytearray, bool]]:
        """Generate complete JPEG controls and truncated entropy scans.

        :returns: Case names mapped to JPEG bytes and expected acceptance.
        :raises unittest.SkipTest: If Pillow is unavailable.
        """
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is required to generate JPEG fixtures")
        pixels = bytes(
            (17 * x + 29 * y + 71 * channel) % 256 for y in range(64)
            for x in range(64) for channel in range(3)
        )
        encoded = io.BytesIO()
        Image.frombytes("RGB", (64, 64), pixels).save(
            encoded, format="JPEG", quality=95, subsampling=0,
        )
        complete = encoded.getvalue()
        sos = complete.index(b"\xff\xda")
        scan_start = sos + 2 + int.from_bytes(complete[sos + 2:sos + 4], "big")
        mid_scan = scan_start + (len(complete) - scan_start - 2) // 2
        metadata_warning = bytearray(complete)
        metadata_warning[complete.index(b"JFIF\x00") + 5] = 2
        return {
            "no-scan": (complete[:scan_start], False),
            "partial-scan": (complete[:mid_scan], False),
            "complete": (complete, True),
            "missing-eoi-only": (complete[:-2], True),
            "metadata-warning": (metadata_warning, True),
        }

    def _check_jpeg_decoding(
        self, image: Path, env: dict[str, str], accepted: bool
    ) -> None:
        """Distinguish JPEG decode failure from the expected missing device.

        :param image: Encoded image fixture.
        :param env: Environment selecting a nonexistent Vulkan driver.
        :param accepted: Whether decoding should succeed.
        """
        result = subprocess.run(
            [str(self.cli), str(image), str(image)], env=env, capture_output=True,
            text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, "")
        if accepted:
            self.assertNotIn("Could not load original image", result.stderr)
            self.assertIn("no usable compute device", result.stderr)
        else:
            self.assertIn("Could not load original image", result.stderr)

    def test_rejects_invalid_butteraugli_numeric_arguments(self) -> None:
        """Reject invalid numeric Butteraugli options before scoring."""
        for flag, value in product(("--intensity-target", "--hf-asymmetry"),
                                   ("0", "-1", "nan", "inf", "1oops")):
            with self.subTest(flag=flag, value=value):
                result = subprocess.run(
                    [str(self.cli), "--metric", "butteraugli", flag, value],
                    capture_output=True, text=True, timeout=5,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("invalid value", result.stderr)
                self.assertIn(flag, result.stderr)

    def test_device_listing_and_selection(self) -> None:
        """Score a still image using an index reported by the device listing."""
        listing = subprocess.run(
            [str(self.cli), "--list-devices"], check=True, capture_output=True,
            text=True, timeout=30,
        )
        if listing.stdout.startswith("No compatible Vulkan compute devices"):
            self.skipTest("no compatible Vulkan compute device")
        devices = listing.stdout.splitlines()
        for index, device in enumerate(devices):
            self.assertTrue(device.startswith(f"{index}  "), device)
            self.assertRegex(device, r"  [\da-f]{8}(?:-[\da-f]{4}){3}-[\da-f]{12}$")
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        image = directory / "identity.png"
        make_goldens.write_png(image, 32, 32, 3, [[128] * 96] * 32)
        result = subprocess.run(
            [str(self.cli), "--device",
             str(len(devices) - 1),
             str(image),
             str(image)], check=True, capture_output=True, text=True, timeout=120,
        )
        self.assertAlmostEqual(float(result.stdout), 100.0, delta=0.5)

    def test_unavailable_device_index_fails_before_scoring(self) -> None:
        """Report an unavailable index without reading either input."""
        result = subprocess.run(
            [str(self.cli), "--device", "999999", "original.png", "distorted.png"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("--device 999999 is unavailable", result.stderr)
        self.assertIn("--list-devices", result.stderr)

    def test_alpha_on_either_image_uses_worst_background(self) -> None:
        """Use the worse background score when either image contains alpha."""
        test_binary = os.environ.get("VQM_TEST_BINARY")
        if not test_binary:
            self.skipTest("GPU test binary not configured")
        probe = subprocess.run(
            [test_binary, "--probe-gpu"], capture_output=True, text=True, timeout=120
        )
        if probe.returncode:
            if "GPU required, but no usable Vulkan device" in probe.stderr:
                self.skipTest("no usable Vulkan device")
            self.fail(f"GPU test probe failed: {probe.stderr}")
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        opaque = directory / "opaque.png"
        transparent = directory / "transparent.png"
        make_goldens.write_png(opaque, 32, 32, 3, [[128, 128, 128] * 32] * 32)
        make_goldens.write_png(transparent, 32, 32, 4, [[128, 128, 128, 0] * 32] * 32)
        thresholds = (
            ("ssimulacra2", self.assertLess, 90.0),
            ("butteraugli", self.assertGreater, 0.0)
        )
        for (metric, check, threshold), (original,
                                         distorted) in product(thresholds,
                                                               ((opaque, transparent),
                                                                (transparent, opaque))):
            with self.subTest(metric=metric, original=original.name):
                result = subprocess.run(
                    [str(self.cli), "--metric", metric,
                     str(original),
                     str(distorted)], check=True, capture_output=True, text=True,
                    timeout=120,
                )
                check(float(result.stdout.strip()), threshold)


if __name__ == "__main__":
    unittest.main()
