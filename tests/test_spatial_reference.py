#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Verify exact float transport and independent spatial scoring."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from validation.temporal import spatial_reference as reference


def _inputs(root: Path, *,
            frames: int = 2) -> tuple[dict[str, object], np.ndarray]:
    values = np.linspace(.05, .95, frames * 3 * 8 * 8,
                         dtype=np.float32).reshape(frames, 3, 8, 8)
    row: dict[str, object] = {"frames": frames, "width": 8, "height": 8}
    for side, data in (("original", values), ("distorted",
                                              values * np.float32(.9))):
        path = root / f"{side}.y4m"
        raw = data.astype("<f4").tobytes()
        Path(str(path) + ".rgb").write_bytes(raw)
        row[side] = str(path)
        row[side + "_hashes"] = {"rgb": hashlib.sha256(raw).hexdigest()}
    return row, values


def _decode_pfm(path: Path) -> np.ndarray:
    # This byte decoder is independent of the writer's transpose operation.
    header, dimensions, scale, payload = path.read_bytes().split(b"\n", 3)
    if header != b"PF" or scale != b"-1.0":
        raise AssertionError("unexpected PFM transport header")
    width, height = (int(value) for value in dimensions.split())
    samples = np.frombuffer(payload, dtype="<f4")
    result = np.empty((3, height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            result[:, height - 1 - y,
                   x] = samples[3 * (y * width + x):3 * (y * width + x + 1)]
    return result


def _run_response(stdout: str, code: int = 0,
                  stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, stdout, stderr)


class SpatialReferenceTests(unittest.TestCase):
    """Exercise transport, clip boundaries, command selection and provenance."""

    def test_pfm_keeps_float32_precision_channel_order_and_orientation(
        self
    ) -> None:
        """Round-trip all rows and unequal channels without quantization."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, values = _inputs(root)
            path = root / "frame.pfm"
            fingerprint = reference.write_pfm(path, values[0])
            np.testing.assert_array_equal(_decode_pfm(path), values[0])
            self.assertEqual(
                fingerprint,
                hashlib.sha256(path.read_bytes()).hexdigest()
            )
            self.assertFalse(
                np.all(values[0] * 65535 == np.rint(values[0] * 65535))
            )

    def test_original_batch_scores_every_frame_then_pools(self) -> None:
        """Run one deterministic original and retain ordered evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, values = _inputs(root)
            observed: dict[str, object] = {}

            def original(command: list[str],
                         **kwargs: object) -> subprocess.CompletedProcess[str]:
                pairs = Path(command[command.index("--pairs-file") + 1])
                lines = pairs.read_text().splitlines()
                observed["inputs"] = [
                    [_decode_pfm(Path(path)) for path in line.split("\t")]
                    for line in lines
                ]
                observed["kwargs"] = kwargs
                return _run_response("10.25\n20.75\n")

            with mock.patch.object(reference.subprocess, "run",
                                   side_effect=original) as run:
                result = reference.spatial_score(
                    row, "ssimulacra2", root / "report"
                )
            self.assertEqual(run.call_count, 1)
            np.testing.assert_array_equal(observed["inputs"][0][0], values[0])
            np.testing.assert_array_equal(observed["inputs"][1][0], values[1])
            self.assertEqual(result["score"], 15.5)
            self.assertEqual(result["frames"], 2)
            self.assertEqual(
                result["frame_results"],
                [{
                    "index": 0,
                    "score": 10.25
                }, {
                    "index": 1,
                    "score": 20.75
                }]
            )
            self.assertEqual(len(result["frame_inputs"]), 2)
            command = run.call_args.args[0]
            self.assertTrue(command[0].endswith("ss2_cloudinary_bench"))
            self.assertEqual(command[command.index("--metric") + 1], "ssimu2")
            self.assertEqual(command[command.index("--threads") + 1], "1")
            self.assertEqual(
                command[command.index("--decode-threads") + 1], "1"
            )
            self.assertEqual(command[command.index("--iterations") + 1], "1")
            self.assertEqual(command[command.index("--warmup") + 1], "0")
            self.assertFalse(
                any(path.is_dir() for path in (root / "report").iterdir())
            )

    def test_spatial_specific_sidecars_override_generic_inputs(self) -> None:
        """Use separate canonical spatial YUV/RGB metadata when present."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, _ = _inputs(root)
            for side in ("original", "distorted"):
                row["spatial_" + side] = row[side]
                row["spatial_" + side + "_hashes"] = row[side + "_hashes"]
                row[side] = "unavailable-generic-input"
            with mock.patch.object(reference.subprocess, "run",
                                   return_value=_run_response("1\n3\n")) as run:
                result = reference.spatial_score(
                    row, "butteraugli", root / "report"
                )
            self.assertEqual(result["score"], 2)
            self.assertTrue(
                run.call_args.args[0][0].endswith("ss2_libjxl_bench")
            )
            self.assertEqual(run.call_args.args[0][2], "butteraugli")

    def test_incomplete_spatial_metadata_fails_without_fallback(self) -> None:
        """Reject partial specialized inputs instead of scoring other bytes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, _ = _inputs(root)
            row["spatial_original"] = row["original"]
            with self.assertRaisesRegex(ValueError, "incomplete"):
                reference.spatial_score(row, "ssimulacra2", root / "report")

    def test_truncated_extra_or_unpinned_rgb_bytes_fail(self) -> None:
        """Require exactly the declared frames and their recorded RGB hash."""
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(temporary)
        for label in ("truncated", "extra", "hash"):
            row, _ = _inputs(root)
            path = Path(str(row["original"]) + ".rgb")
            _damage(path, row, label)
            with (self.subTest(case=label
                               ), mock.patch.object(reference.subprocess, "run")
                  as run, self.assertRaises(ValueError)):
                reference.spatial_score(row, "ssimulacra2", root / label)
            run.assert_not_called()

    def test_nonfinite_or_out_of_gamut_rgb_is_never_clipped(self) -> None:
        """Reject invalid samples without altering original inputs."""
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(temporary)
        for value in (np.nan, np.inf, -0.01, 1.01):
            with self.subTest(value=value
                              ), self.assertRaisesRegex(ValueError,
                                                        "finite encoded RGB"):
                reference.write_pfm(
                    root / "bad.pfm",
                    np.full((3, 8, 8), value, dtype=np.float32)
                )

    def test_missing_nonfinite_or_failed_original_frames_are_rejected(
        self
    ) -> None:
        """Reject pooled scores from partial or failed originals."""
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(temporary)
        row, _ = _inputs(root)
        for output in ("1\n", "1\nnan\n", "1\ninf\n", "1\nERROR\n"):
            with (self.subTest(output=output),
                  mock.patch.object(reference.subprocess, "run",
                                    return_value=_run_response(output)),
                  self.assertRaises(ValueError)):
                reference.spatial_score(row, "ssimulacra2", root / "report")
        with (mock.patch.object(reference.subprocess, "run",
                                return_value=_run_response("", 1,
                                                           "original failed")),
              self.assertRaisesRegex(RuntimeError, "original failed")):
            reference.spatial_score(row, "butteraugli", root / "report")

    def test_provenance_fingerprints_binary_build_sources_and_parameters(
        self
    ) -> None:
        """Retain original source pins and Butteraugli configuration."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fake_build(root, "butteraugli")
            with mock.patch.object(reference, "ROOT", root):
                info = reference.spatial_provenance("butteraugli")
            self.assertEqual(info["revision"], reference.PINS["butteraugli"])
            self.assertEqual(
                info["parameters"], {
                    "hf_asymmetry": 1,
                    "intensity_target": 80,
                    "xmul": 1
                }
            )
            self.assertEqual(
                info["executable_sha256"],
                hashlib.sha256(b"binary").hexdigest()
            )
            self.assertEqual(info["threads"], 1)
            self.assertIn("encoded-sRGB", info["input"])
            self.assertIn(
                "bench/inmemory/libjxl_bench.cc", info["source_sha256"]
            )

    def test_provenance_rejects_a_missing_or_wrong_revision_build(self) -> None:
        """Require matching build pins beyond reference labels."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fake_build(root, "ssimulacra2")
            stamp = root / "build/reference-fingerprints/cloudinary.json"
            stamp.write_text(json.dumps({"revision": "wrong-revision"}))
            with (mock.patch.object(reference, "ROOT", root),
                  self.assertRaisesRegex(RuntimeError,
                                         "wrong pinned revision")):
                reference.spatial_provenance("ssimulacra2")
            stamp.unlink()
            with (mock.patch.object(reference, "ROOT", root),
                  self.assertRaisesRegex(RuntimeError,
                                         "build pinned spatial reference")):
                reference.spatial_provenance("ssimulacra2")


def _damage(path: Path, row: dict[str, object], kind: str) -> None:
    payload = path.read_bytes()
    if kind == "truncated":
        path.write_bytes(payload[:-1])
    elif kind == "extra":
        path.write_bytes(payload + b"x")
    else:
        row["original_hashes"] = {"rgb": "0" * 64}


def _fake_build(root: Path, metric: str) -> None:
    binary = root / reference.BINARIES[metric]
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary")
    stamp = root / "build/reference-fingerprints" / (
        f"{reference.STEPS[metric]}.json"
    )
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps({"revision": reference.PINS[metric]}))
    paths = [
        reference.HARNESSES[metric], "bench/inmemory/driver.h",
        "bench/inmemory/pairs.h", "bench/inmemory/prefetch.h",
        *reference._reference_sources(metric)
    ]
    for name in paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)


@unittest.skipUnless(
    all(
        (reference.ROOT / path).is_file()
        for path in reference.BINARIES.values()
    ), "pinned spatial originals not built"
)
class OriginalSpatialIntegration(unittest.TestCase):
    """Check pinned readers and metric entry points when available."""

    def test_float_pfm_identity_and_nonidentity_through_both_originals(
        self
    ) -> None:
        """Run real originals on exact floats and pool complete frames."""
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(temporary)
        row, _ = _inputs(root)
        for metric, identity in (("ssimulacra2", 100), ("butteraugli", 0)):
            with self.subTest(metric=metric):
                same = {
                    **row, "distorted": row["original"],
                    "distorted_hashes": row["original_hashes"]
                }
                identical = reference.spatial_score(
                    same, metric, root / metric / "identity"
                )
                self.assertEqual(identical["score"], identity)
                changed = reference.spatial_score(
                    row, metric, root / metric / "changed"
                )
                self.assertEqual(changed["frames"], 2)
                self.assertNotEqual(changed["score"], identity)
                self.assertEqual(
                    [frame["index"] for frame in changed["frame_results"]],
                    [0, 1]
                )


if __name__ == "__main__":
    unittest.main()
