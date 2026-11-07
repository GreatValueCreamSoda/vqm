#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Check offline temporal preparation, frozen policies and reports."""

import argparse
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from itertools import product
from pathlib import Path
from unittest import mock

import numpy as np

from validation.agreement import evaluate_agreement, load_report
from validation.temporal import corpus, reference
from validation.temporal import run as temporal_run
from validation.temporal.fetch import Source


def _source(
    directory: Path, *, name: str = "fixture", width: int = 8, height: int = 8,
    frames: int = 4
) -> tuple[Source, np.ndarray]:
    pixels = width * height
    values = np.full((frames, pixels * 3 // 2), 128, dtype=np.uint8)
    grid = np.arange(pixels, dtype=np.uint16).reshape(1, -1)
    times = np.arange(frames, dtype=np.uint16).reshape(-1, 1)
    values[:, :pixels] = ((grid * 7 + times * 29) % 200 + 20).astype(np.uint8)
    header = f"YUV4MPEG2 W{width} H{height} F30:1 Ip A1:1 C420jpeg"
    data = header.encode() + b"\n" + b"".join(
        b"FRAME\n" + frame.tobytes() for frame in values
    )
    path = directory / (name + ".y4m")
    path.write_bytes(data)
    source = Source(
        name, name, "https://invalid.example/fixture.y4m", path.name, len(data),
        corpus.digest(path), width, height, frames, "30:1", "420jpeg", header,
        len(data), "https://invalid.example/catalogue"
    )
    return source, values


def _prepared(directory: Path) -> list[dict[str, object]]:
    rows = []
    for index in range(3):
        source, _ = _source(directory, name=f"fixture_{index}")
        rows.extend(
            corpus.prepare_source(
                source, directory, directory / "derived", 4, corpus.VARIANTS
            )
        )
    return rows


def _fake_values(rows: list[dict[str, object]],
                 offset: float = 0) -> list[dict[str, object]]:
    return [
        {
            "score": float(index) + offset,
            "frames": row["frames"]
        } for index, row in enumerate(rows)
    ]


class TemporalPreparationTests(unittest.TestCase):
    """Verify integer recipes, shared samples and source content hashes offline."""

    def test_recipes_preserve_inputs_and_exact_temporal_positions(self) -> None:
        """Freeze, flicker and impulse errors occur at their declared frames."""
        values = np.full((4, 12), 128, dtype=np.uint8)
        values[:, :8] += np.arange(4, dtype=np.uint8)[:, None]
        original = values.copy()
        frozen = corpus.distort(values, corpus.Variant("freeze", "freeze", 3), 8)
        np.testing.assert_array_equal(frozen, values[[0, 0, 0, 3]])
        flicker = corpus.distort(values, corpus.Variant("flicker", "flicker", 8), 8)
        np.testing.assert_array_equal(
            flicker[:, :8], values[:, :8] + np.array([-8, 8, -8, 8])[:, None]
        )
        np.testing.assert_array_equal(flicker[:, 8:], values[:, 8:])
        pulse = corpus.distort(values, corpus.Variant("pulse", "impulse", 12), 8)
        np.testing.assert_array_equal(pulse[:-1], values[:-1])
        np.testing.assert_array_equal(pulse[-1, :8], values[-1, :8] + 12)
        np.testing.assert_array_equal(pulse[-1, 8:], values[-1, 8:])
        np.testing.assert_array_equal(values, original)

    def test_quantization_clips_instead_of_wrapping(self) -> None:
        """High code values remain high after rounded integer quantization."""
        values = np.array(
            [[0, 7, 16, 31, 32, 249, 255, 128, 128, 128, 128, 128]], dtype=np.uint8
        )
        result = corpus.distort(
            values, corpus.Variant("quantize", "quantization", 32), 8
        )
        np.testing.assert_array_equal(result[0, :7], [0, 0, 32, 32, 32, 255, 255])
        self.assertFalse(np.shares_memory(values, result))

    def test_canonical_rgb_has_limited_range_neutral_endpoints(self) -> None:
        """Neutral legal black and white decode to exact shared RGB endpoints."""
        for code, expected in ((16, 0.0), (235, 1.0)):
            with self.subTest(code=code):
                frame = np.array([code] * 4 + [128] * 8, dtype=np.uint8)
                decoded = corpus.rgb_frame(frame, 2, 2)
                np.testing.assert_array_equal(decoded, np.full((3, 2, 2), expected))
                self.assertEqual(decoded.dtype, np.dtype("<f4"))

    def test_serialized_representations_are_deterministic(self) -> None:
        """The raw reference bytes and candidate RGB bytes have reproducible hashes."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, values = _source(root)
            canonical = corpus.expand_chroma(values, source.width, source.height)
            first, second = root / "first.y4m", root / "second.y4m"
            hashes = corpus.write_clip(first, canonical, source)
            self.assertEqual(hashes, corpus.write_clip(second, canonical, source))
            self.assertEqual(
                first.with_suffix(".yuv").read_bytes(), canonical.tobytes()
            )
            self.assertEqual(
                Path(str(first) + ".rgb").stat().st_size, 4 * 3 * 8 * 8 * 4
            )
            np.testing.assert_array_equal(corpus.read_source(source, root, 4), values)
            self.assertEqual(hashes["rgb"], corpus.digest(Path(str(first) + ".rgb")))

    def test_source_checksum_and_frame_limits_are_enforced(self) -> None:
        """Preparation cannot use altered downloads or an unpinned extra frame."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _source(root)
            with self.assertRaisesRegex(ValueError, "frame count"):
                corpus.read_source(source, root, 5)
            path = root / source.filename
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "checksum"):
                corpus.read_source(source, root, 4)
            changed = replace(source, sha256=corpus.digest(path))
            with self.assertRaisesRegex(ValueError, "truncated"):
                corpus.read_source(changed, root, 4)

    def test_unsupported_original_chroma_cannot_be_misread_as_420(self) -> None:
        """Preparation rejects layouts outside its explicitly supported decoder."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _source(root)
            with self.assertRaisesRegex(ValueError, "YUV420"):
                corpus.read_source(replace(source, chroma="444"), root, 4)

    def test_prepared_reports_are_independent_of_cache_location(self) -> None:
        """Identical serialized inputs produce portable manifest and sample identities."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _source(root)
            first = corpus.prepare_source(source, root, root / "a", 4, corpus.VARIANTS)
            second = corpus.prepare_source(source, root, root / "b", 4, corpus.VARIANTS)
            a = temporal_run.report(
                first, _fake_values(first), "cvvdp", "test", "revision"
            )
            b = temporal_run.report(
                second, _fake_values(second), "cvvdp", "test", "revision"
            )
            self.assertEqual(a, b)
            changed = copy.deepcopy(second)
            changed[0]["distorted_hashes"]["rgb"] = "0" * 64
            self.assertNotEqual(
                a["input_manifest_hash"],
                temporal_run.report(
                    changed, _fake_values(changed), "cvvdp", "test", "revision"
                )["input_manifest_hash"]
            )


class TemporalRolloutTests(unittest.TestCase):
    """Check old/new gate selection and freeze original evidence before candidates."""

    def test_default_gate_mode_is_resolved_from_the_frozen_domain(self) -> None:
        """The parser defers default selection until the original and policy are known."""
        with mock.patch.object(sys, "argv", ["temporal.run"]):
            self.assertIsNone(temporal_run.arguments().gate_mode)

    def test_gate_modes_do_not_hide_one_failed_endpoint(self) -> None:
        """An offset can pass ranking mode while still failing both mode."""
        result = {"numerical": {"status": "fail"}, "proposed_status": "pass"}
        self.assertTrue(temporal_run.qualifies(result, "ranking"))
        self.assertFalse(temporal_run.qualifies(result, "numerical"))
        self.assertFalse(temporal_run.qualifies(result, "both"))
        result = {"numerical": {"status": "pass"}, "proposed_status": "inconclusive"}
        self.assertTrue(temporal_run.qualifies(result, "numerical"))
        self.assertFalse(temporal_run.qualifies(result, "ranking"))
        self.assertFalse(temporal_run.qualifies(result, "both"))

    def test_invalid_gate_mode_is_rejected_by_cli(self) -> None:
        """Unknown rollout modes cannot silently choose an acceptance policy."""
        with mock.patch.object(sys, "argv", ["temporal.run", "--gate-mode", "automatic"]), \
                self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            temporal_run.arguments()

    def test_generated_policy_is_saved_before_candidate_scoring(self) -> None:
        """The candidate cannot influence margins, while a positive offset is accepted."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = _prepared(root)
            original = temporal_run.report(
                rows, _fake_values(rows), "vmaf-v0", "original",
                reference.PINS["vmaf-v0"]
            )
            policy = temporal_run.frozen_policy(original, "smoke")
            args = argparse.Namespace(
                out_dir=root, reference="generate", profile="smoke",
                gate_mode="ranking", binary=root / "binary", reference_only=False
            )
            args.binary.write_bytes(b"test-only candidate executable fingerprint")
            with mock.patch.object(temporal_run, "generated_original", return_value=(original, policy)), \
                    mock.patch.object(temporal_run, "candidate_scores", side_effect=self._saved_policy_candidate), \
                    redirect_stdout(io.StringIO()):
                self.assertTrue(temporal_run.run_metric(rows, "vmaf-v0", args))
                args.gate_mode = "both"
                self.assertFalse(temporal_run.run_metric(rows, "vmaf-v0", args))
            result = json.loads((root / "vmaf-v0-agreement.json").read_text())
            self.assertEqual(result["ranking"]["status"], "pass")
            self.assertEqual(result["numerical"]["status"], "fail")
            self.assertEqual(result["numerical"]["mean_offset"], 3)
            with redirect_stdout(io.StringIO()):
                temporal_run._summary(["vmaf-v0"], [False], args)
            summary = json.loads((root / "summary.json").read_text())
            diagnostics = summary["results"]["vmaf-v0"]["numerical_diagnostics"]
            self.assertEqual(diagnostics, result["numerical"])

    def _saved_policy_candidate(
        self, rows: list[dict[str, object]], metric: str, args: argparse.Namespace
    ) -> list[dict[str, object]]:
        self.assertTrue((args.out_dir / f"{metric}-original.json").is_file())
        policy = json.loads((args.out_dir / f"{metric}-policy.json").read_text())
        self.assertEqual(policy["reference_revision"], reference.PINS[metric])
        self.assertIn("input_manifest_hash", policy)
        return _fake_values(rows, 3)

    def test_checked_original_requires_matching_actual_input_manifest(self) -> None:
        """Changing prepared bytes invalidates checked original scores before use."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = _prepared(root)
            original = temporal_run.report(
                rows, _fake_values(rows), "vmaf-v0", "original",
                reference.PINS["vmaf-v0"]
            )
            oracle = root / "oracles" / "smoke"
            oracle.mkdir(parents=True)
            (oracle / "vmaf-v0.json").write_text(json.dumps(original))
            (oracle / "vmaf-v0-policy.json").write_text(
                json.dumps(temporal_run.frozen_policy(original, "smoke"))
            )
            args = argparse.Namespace(
                reference="checked", profile="smoke", oracle_dir=root / "oracles",
                out_dir=root
            )
            changed = copy.deepcopy(rows)
            changed[0]["original_hashes"]["yuv"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "frozen original corpus"):
                temporal_run.original_and_policy(changed, "vmaf-v0", args)

    def test_batch_output_frame_mismatch_fails_before_agreement(self) -> None:
        """A complete-looking batch cannot claim an omitted late clip frame."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = _prepared(root)
            values = [
                dict(
                    value, sample_id=row["sample_id"], metric="vmaf-v0",
                    core_seconds=.01, decode_seconds=.001, frame_results=[
                        {
                            "index": index,
                            "score": value["score"],
                            "features": [1.] * 6
                        } for index in range(row["frames"])
                    ]
                ) for row, value in zip(rows, _fake_values(rows), strict=True)
            ]
            values[-1]["frames"] -= 1
            output = "".join(json.dumps(value) + "\n" for value in values)
            args = argparse.Namespace(
                out_dir=root, binary=root / "binary", iterations=1, warmup=0
            )
            completed = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
            with mock.patch.object(temporal_run.subprocess, "run", return_value=completed), \
                    self.assertRaisesRegex(RuntimeError, "mismatched"):
                temporal_run.candidate_scores(rows, "vmaf-v0", args)


class TemporalReferenceCacheTests(unittest.TestCase):
    """Verify cached independent results without importing optional reference runtimes."""

    def _cached(self, root: Path, *, score: float = 1,
                frames: int = 4) -> tuple[dict[str, object], dict[str, object], Path]:
        row = _prepared(root)[0]
        identity = {"revision": "test-reference-pin", "runtime": "test-runtime"}
        inputs = {
            key: value
            for key, value in row.items() if key not in ("original", "distorted")
        }
        fingerprint = corpus.json_digest(
            {
                "inputs": inputs,
                "reference": identity,
                "protocol": 1
            }
        )
        path = root / "cache" / "cvvdp" / fingerprint / "result.json"
        path.parent.mkdir(parents=True)
        value = {
            "score": score,
            "frames": frames,
            "fingerprint": fingerprint,
            "seconds": 1
        }
        value["result_sha256"] = corpus.json_digest(value)
        path.write_text(json.dumps(value))
        return row, identity, path

    def test_valid_cached_result_requires_no_reference_runtime(self) -> None:
        """Content-addressed original evidence can be reused entirely offline."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, identity, _ = self._cached(root)
            with mock.patch.object(reference, "cvvdp_score",
                                   side_effect=AssertionError("must use cache")):
                result = reference.score_reference(
                    row, "cvvdp", root / "cache", identity
                )
            self.assertEqual(result["score"], 1)
            self.assertEqual(result["frames"], 4)

    def test_incomplete_or_nonfinite_cached_original_is_rejected(self) -> None:
        """A matching cache key cannot bless incomplete or invalid reference scores."""
        for score, frames in ((float("nan"), 4), (1, 3)):
            with self.subTest(
                    score=score,
                    frames=frames), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                row, identity, _ = self._cached(root, score=score, frames=frames)
                self.assertRaisesRegex(
                    ValueError, "corrupt reference cache", reference.score_reference,
                    row, "cvvdp", root / "cache", identity
                )

    def test_cached_result_bytes_are_authenticated(self) -> None:
        """An edited score is rejected even when the input fingerprint still matches."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, identity, path = self._cached(root)
            value = json.loads(path.read_text())
            value["score"] = 100
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "corrupt reference cache"):
                reference.score_reference(row, "cvvdp", root / "cache", identity)


class TemporalFixtureIntegrityTests(unittest.TestCase):
    """Validate maintained originals and frozen policies without external inputs."""

    def test_checked_originals_satisfy_their_frozen_complete_policies(self) -> None:
        """Every maintained profile has full evidence and exact self-agreement."""
        profiles = (("smoke", 39, 3), ("qualification", 390, 30))
        for (profile, samples, sources), metric in product(profiles,
                                                           temporal_run.METRICS):
            with self.subTest(profile=profile, metric=metric):
                directory = temporal_run.ORACLES / profile
                original = load_report(directory / f"{metric}.json")
                policy = json.loads((directory / f"{metric}-policy.json").read_text())
                self.assertEqual(len(original["samples"]), samples)
                self.assertEqual(policy["min_total_samples"], samples)
                self.assertEqual(policy["min_total_sources"], sources)
                result = evaluate_agreement(original, original, policy)
                self.assertEqual(result["sample_count"], samples)
                self.assertEqual(result["ranking"]["total_sources"], sources)
                self.assertEqual(
                    result["ranking"]["status"], "pass", result["ranking"]["failures"]
                )
                self.assertEqual(result["numerical"]["status"], "pass")
                self.assertEqual(result["proposed_status"], "pass")


if __name__ == "__main__":
    unittest.main()
