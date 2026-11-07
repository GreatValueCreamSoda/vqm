#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Keep every supported metric in the same pinned CLI correlation workflow."""

import argparse
import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from itertools import product
from pathlib import Path
from unittest import mock

import numpy as np

from validation.agreement import evaluate_agreement
from validation.temporal import corpus, reference, spatial_inputs
from validation.temporal import run as workflow
from validation.temporal.fetch import Source

METRICS = ("ssimulacra2", "butteraugli", "vmaf-v0", "vmaf-v1", "cvvdp")
REFERENCE_NAMES = {
    "ssimulacra2": "ssimulacra2",
    "butteraugli": "libjxl",
    "vmaf-v0": "vmaf",
    "vmaf-v1": "vmaf",
    "cvvdp": "ColorVideoVDP"
}


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _rows(sources: int = 3) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"source-{source}/variant-{severity:02}",
            "source_id": f"source-{source}",
            "distortion_family": "noise",
            "stratum": "sdr-native-444",
            "original": f"/test/source-{source}.y4m",
            "distorted": f"/test/source-{source}-variant-{severity}.y4m",
            "width": 352,
            "height": 288,
            "fps": "30000:1001",
            "frames": 40,
            "original_hashes": {
                kind: _hash(f"source-{source}/{kind}")
                for kind in ("y4m", "yuv", "rgb")
            },
            "distorted_hashes": {
                kind: _hash(f"source-{source}/{severity}/{kind}")
                for kind in ("y4m", "yuv", "rgb")
            },
            "spatial_original": f"/test/source-{source}-spatial.y4m",
            "spatial_distorted":
            f"/test/source-{source}-variant-{severity}-spatial.y4m",
            "spatial_original_hashes": {
                kind: _hash(f"source-{source}/spatial/{kind}")
                for kind in ("y4m", "yuv", "rgb")
            },
            "spatial_distorted_hashes": {
                kind: _hash(f"source-{source}/{severity}/spatial/{kind}")
                for kind in ("y4m", "yuv", "rgb")
            },
        } for source in range(sources) for severity in range(13)
    ]


def _values(rows: list[dict[str, object]],
            offset: float = 0) -> list[dict[str, object]]:
    return [
        {
            "score": float(index) + offset,
            "frames": row["frames"],
            "sample_id": row["sample_id"]
        } for index, row in enumerate(rows)
    ]


def _cli_values(rows: list[dict[str, object]], metric: str) -> list[dict[str, object]]:
    """Build complete per-metric CLI evidence for routing subprocess mocks.

    :param rows: Expected complete-clip identities and frame counts.
    :param metric: Selected algorithm arm.
    :returns: Complete JSON-compatible clip results in input order.
    """
    features = {"vmaf-v0": 6, "vmaf-v1": 4}.get(metric, 0)
    return [
        dict(
            value, metric=metric, core_seconds=.01, decode_seconds=.001,
            frame_results=[] if metric == "cvvdp" else [
                {
                    "index": index,
                    "score": value["score"],
                    "features": [1.] * features
                } for index in range(value["frames"])
            ]
        ) for value in _values(rows)
    ]


def _mislabelled_values(rows: list[dict[str, object]],
                        label: str | None) -> list[dict[str, object]]:
    """Corrupt only metric identity in otherwise complete SSIMULACRA 2 output.

    :param rows: Expected clip identities and frame counts.
    :param label: Incorrect metric name, or None to omit the field.
    :returns: Candidate output with deliberately invalid metric identity.
    """
    values = _cli_values(rows, "ssimulacra2")
    for value in values:
        value["metric"] = label
        if label is None:
            del value["metric"]
    return values


def _locked_revisions() -> dict[str, str]:
    fields = [
        line.split()
        for line in (workflow.ROOT / "tools/references.lock").read_text().splitlines()
    ]
    return {
        row[0]: row[2]
        for row in fields if len(row) == 3 and not row[0].startswith("#")
    }


def _media(root: Path) -> tuple[Source, dict[str, object]]:
    header = "YUV4MPEG2 W4 H4 F30:1 Ip A1:1 C420jpeg"
    values = np.full((2, 24), 128, dtype=np.uint8)
    values[:, :16] = np.arange(16, dtype=np.uint8) * 13 + 16
    values[1, 16:20], values[1, 20:] = 200, 56
    contents = header.encode() + b"\n" + b"".join(
        b"FRAME\n" + frame.tobytes() for frame in values
    )
    path = root / "fixture.y4m"
    path.write_bytes(contents)
    source = Source(
        "fixture", "fixture", "https://invalid.example/fixture.y4m", path.name,
        len(contents), corpus.digest(path), 4, 4, 2, "30:1", "420jpeg", header,
        len(contents), "https://invalid.example/catalogue"
    )
    expanded = corpus.expand_chroma(values, 4, 4)
    original, distorted = root / "original.y4m", root / "distorted.y4m"
    first = corpus.write_clip(original, expanded, source)
    second = corpus.write_clip(distorted, expanded[::-1], source)
    row = {
        "sample_id": "fixture/variant",
        "source_id": "fixture",
        "original": str(original),
        "distorted": str(distorted),
        "original_hashes": first,
        "distorted_hashes": second,
        "width": 4,
        "height": 4,
        "fps": "30:1",
        "frames": 2,
        "stratum": "sdr-native-444",
        "distortion_family": "freeze"
    }
    return source, row


class AllMetricRoutingTests(unittest.TestCase):
    """Require all metric defaults, independent pins and the normal CLI batch path."""

    def test_default_and_explicit_metric_selection_cover_all_five(self) -> None:
        """Adding spatial metrics cannot leave them outside ordinary validation."""
        with mock.patch.object(sys, "argv", ["temporal.run"]):
            default = workflow.arguments()
        self.assertEqual(set(default.metrics.split(",")), set(METRICS))
        self.assertIsNone(default.gate_mode)
        with mock.patch.object(sys, "argv",
                               ["temporal.run", "--metrics", ",".join(METRICS)]):
            selected = workflow.arguments()
        self.assertEqual(selected.metrics.split(","), list(METRICS))

    def test_unknown_metric_cannot_select_an_unpinned_reference(self) -> None:
        """Unsupported names fail before fetching media or scoring any candidate."""
        with mock.patch.object(sys, "argv", ["temporal.run", "--metrics", "ssimulacra2,unknown"]), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            workflow.arguments()

    def test_every_metric_reference_revision_matches_the_repository_lock(self) -> None:
        """Cloudinary, libjxl, libvmaf and original CVVDP retain independent pins."""
        locked = _locked_revisions()
        for metric in METRICS:
            with self.subTest(metric=metric):
                self.assertEqual(
                    reference.PINS[metric], locked[REFERENCE_NAMES[metric]]
                )

    def test_reference_verification_selects_each_correct_original(self) -> None:
        """A spatial metric must not accidentally verify libvmaf instead."""
        completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        for metric in METRICS:
            with self.subTest(metric=metric
                              ), mock.patch.object(reference.subprocess, "run",
                                                   return_value=completed) as called:
                reference.verify_references([metric])
                command = called.call_args.args[0]
                self.assertIn("--check", command)
                self.assertEqual(
                    command[command.index("--only") + 1], REFERENCE_NAMES[metric]
                )

    def test_reference_verification_failure_is_not_a_skip(self) -> None:
        """A missing, dirty or wrong-pinned original cannot become qualification."""
        completed = subprocess.CompletedProcess(
            [], 1, stdout="wrong revision", stderr="dirty original"
        )
        with mock.patch.object(reference.subprocess, "run", return_value=completed), \
                self.assertRaises(RuntimeError):
            reference.verify_references(list(METRICS))

    def test_each_metric_routes_batch_scoring_through_the_public_cli(self) -> None:
        """Every arm consumes complete clips using JSON and stable sample IDs."""
        rows = _rows()
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                out_dir=Path(directory), binary=Path(directory) / "vqm", iterations=2,
                warmup=1
            )
            for metric in METRICS:
                self._check_cli_route(rows, metric, args)

    def _check_cli_route(
        self, rows: list[dict[str, object]], metric: str, args: argparse.Namespace
    ) -> None:
        output = "".join(
            json.dumps(value) + "\n" for value in _cli_values(rows, metric)
        )
        completed = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
        with mock.patch.object(workflow.subprocess, "run",
                               return_value=completed) as called:
            values = workflow.candidate_scores(rows, metric, args)
        command = called.call_args.args[0]
        self.assertEqual(Path(command[0]), args.binary)
        self.assertIn("--json", command)
        self.assertEqual(command[command.index("--metric") + 1], metric)
        self.assertEqual(command[command.index("--iterations") + 1], "2")
        self.assertEqual(len(values), len(rows))
        spatial = metric in ("ssimulacra2", "butteraugli")
        prefix = "spatial_" if spatial else ""
        pair_file = Path(command[command.index("--pairs-file") + 1])
        first = pair_file.read_text().splitlines()[0].split("\t")
        self.assertEqual(
            first[1:], [rows[0][prefix + "original"], rows[0][prefix + "distorted"]]
        )
        if spatial:
            self.assertEqual(command[command.index("--input-transfer") + 1], "srgb")

    def test_wrong_or_absent_metric_label_cannot_be_rebound_to_requested_metric(
        self
    ) -> None:
        """Matching sample counts and scores cannot conceal a wrong algorithm arm."""
        for label in ("butteraugli", None):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                rows = _rows()
                values = _mislabelled_values(rows, label)
                completed = subprocess.CompletedProcess(
                    [], 0, stdout="".join(json.dumps(value) + "\n" for value in values),
                    stderr=""
                )
                args = argparse.Namespace(
                    out_dir=Path(temporary), binary=Path(temporary) / "vqm",
                    iterations=1, warmup=0
                )
                self._expect_wrong_metric_failure(rows, args, completed)

    def _expect_wrong_metric_failure(
        self, rows: list[dict[str, object]], args: argparse.Namespace,
        completed: subprocess.CompletedProcess[str]
    ) -> None:
        with mock.patch.object(workflow.subprocess, "run", return_value=completed), \
                self.assertRaisesRegex(RuntimeError, "mismatched|metric"):
            workflow.candidate_scores(rows, "ssimulacra2", args)


class AllMetricFrozenReportTests(unittest.TestCase):
    """Check profile/report binding using temporary evidence, never online runtimes."""

    def test_configuration_identity_separates_metrics_and_video_timing(self) -> None:
        """Only metrics with a frame-rate parameter create distinct timing domains."""
        row = _rows()[0]
        reports = [
            workflow.report(
                [row], _values([row]), metric, "test", reference.PINS[metric]
            ) for metric in METRICS
        ]
        self.assertEqual(
            len({report["configuration_hash"]
                 for report in reports}), len(METRICS)
        )
        for metric in METRICS:
            with self.subTest(metric=metric):
                changed = dict(row, fps="60:1")
                first = workflow.configuration_id(row, metric)
                second = workflow.configuration_id(changed, metric)
                self.assertEqual(first == second, metric != "cvvdp")

    def test_frozen_policies_bind_each_original_and_score_direction(self) -> None:
        """Lower-is-better Butteraugli remains correctly oriented for future MOS."""
        rows = _rows()
        for metric in METRICS:
            with self.subTest(metric=metric):
                original = workflow.report(
                    rows, _values(rows), metric, "original", reference.PINS[metric]
                )
                policy = workflow.frozen_policy(original, "smoke")
                self.assertEqual(policy["reference_revision"], reference.PINS[metric])
                self.assertEqual(
                    policy["configuration_hash"], original["configuration_hash"]
                )
                self.assertEqual(
                    policy["input_manifest_hash"], original["input_manifest_hash"]
                )
                self.assertEqual(
                    policy.get("direction", "higher"),
                    "lower" if metric == "butteraugli" else "higher"
                )
                result = evaluate_agreement(original, original, policy)
                self.assertEqual(result["proposed_status"], "pass")
                self.assertEqual(result["numerical"]["status"], "pass")

    def test_spatial_inputs_do_not_rebind_existing_temporal_oracles(self) -> None:
        """Separate spatial byte representations cannot silently change the other arms."""
        rows = _rows()
        changed = copy.deepcopy(rows)
        changed[0]["spatial_distorted_hashes"]["rgb"] = "0" * 64
        for metric in METRICS:
            with self.subTest(metric=metric):
                first = workflow.report(
                    rows, _values(rows), metric, "original", reference.PINS[metric]
                )
                second = workflow.report(
                    changed, _values(changed), metric, "original",
                    reference.PINS[metric]
                )
                self.assertEqual(
                    first["input_manifest_hash"] == second["input_manifest_hash"],
                    metric not in ("ssimulacra2", "butteraugli")
                )

    def test_spatial_reports_require_their_own_decoded_input_hashes(self) -> None:
        """Spatial arms cannot fall back to hashes for differently decoded inputs."""
        rows = _rows()
        del rows[0]["spatial_original_hashes"]
        for metric in ("ssimulacra2", "butteraugli"):
            with self.subTest(metric=metric), self.assertRaises((KeyError, ValueError)):
                workflow.report(
                    rows, _values(rows), metric, "original", reference.PINS[metric]
                )

    def test_checked_profile_selection_is_metric_specific(self) -> None:
        """Smoke and qualification read the requested metric's own frozen evidence."""
        cases = product(("smoke", "qualification"), METRICS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for profile, metric in cases:
                self._check_profile(root, profile, metric)

    def _check_profile(self, root: Path, profile: str, metric: str) -> None:
        rows = _rows(3 if profile == "smoke" else 30)
        original = workflow.report(
            rows, _values(rows), metric, "original", reference.PINS[metric]
        )
        policy = workflow.frozen_policy(original, profile)
        directory = root / "oracles" / profile
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{metric}.json").write_text(json.dumps(original))
        (directory / f"{metric}-policy.json").write_text(json.dumps(policy))
        args = argparse.Namespace(
            reference="checked", profile=profile, oracle_dir=root / "oracles",
            out_dir=root
        )
        actual, actual_policy = workflow.original_and_policy(rows, metric, args)
        self.assertEqual(actual, original)
        self.assertEqual(actual_policy, policy)

    def test_absent_checked_original_never_falls_back_to_candidate(self) -> None:
        """Every selected metric needs its own independent original report."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(
                reference="checked", profile="smoke", oracle_dir=root, out_dir=root
            )
            for metric in METRICS:
                self.assertRaises(
                    OSError, workflow.original_and_policy, _rows(), metric, args
                )

    def test_checked_original_cannot_redefine_the_pinned_reference_revision(
        self
    ) -> None:
        """A matching stale oracle and policy cannot replace the repository's original pin."""
        for metric in METRICS:
            with self.subTest(metric=metric
                              ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                rows = _rows()
                original = workflow.report(
                    rows, _values(rows), metric, "original", "wrong-original-revision"
                )
                policy = workflow.frozen_policy(original, "smoke")
                directory = root / "smoke"
                directory.mkdir()
                (directory / f"{metric}.json").write_text(json.dumps(original))
                (directory / f"{metric}-policy.json").write_text(json.dumps(policy))
                args = argparse.Namespace(
                    reference="checked", profile="smoke", oracle_dir=root, out_dir=root
                )
                self.assertRaisesRegex(
                    ValueError, "pin|revision", workflow.original_and_policy, rows,
                    metric, args
                )


class SharedSpatialInputTests(unittest.TestCase):
    """Protect the decoded sample contract for the two added spatial reference arms."""

    def test_encoded_rgb_corners_retain_headroom_after_integer_yuv(self) -> None:
        """Quantization of extreme colors cannot force different reference clipping."""
        rgb = np.array(list(product((0.0, 1.0), repeat=3)), dtype=np.float32).T[:,
                                                                                None, :]
        encoded = spatial_inputs.encode_frame(rgb)
        decoded = corpus.rgb_frame(encoded.reshape(-1), 8, 1)
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertGreater(float(decoded.min()), 0)
        self.assertLess(float(decoded.max()), 1)
        np.testing.assert_allclose(decoded, rgb * .96 + .02, atol=.01, rtol=0)

    def test_invalid_rgb_is_rejected_before_serialization(self) -> None:
        """Nonfinite, unnormalized and wrongly shaped pixels are not valid inputs."""
        cases = (
            np.full((3, 2, 2),
                    np.nan), np.full((3, 2, 2),
                                     -0.1), np.full((3, 2, 2),
                                                    1.1), np.zeros((2, 2, 2))
        )
        for rgb in cases:
            with self.subTest(shape=rgb.shape,
                              value=rgb.flat[0]), self.assertRaises(ValueError):
                spatial_inputs.encode_frame(rgb)

    def test_spatial_files_preserve_original_media_and_bind_actual_bytes(self) -> None:
        """New representations stay separate and report their real serialized hashes."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, row = _media(root)
            before = copy.deepcopy(row)
            prepared = spatial_inputs.ensure_spatial([row], [source])[0]
            self.assertEqual(row, before)
            corpus.verify_inputs([row])
            target = Path(prepared["spatial_distorted"])
            self.assertNotEqual(target, Path(row["distorted"]))
            self.assertEqual(
                prepared["spatial_distorted_hashes"]["y4m"], corpus.digest(target)
            )
            self.assertEqual(
                prepared["spatial_distorted_hashes"]["rgb"],
                corpus.digest(Path(str(target) + ".rgb"))
            )
            decoded = np.fromfile(str(target) + ".rgb", dtype="<f4")
            self.assertEqual(decoded.size, 2 * 3 * 4 * 4)
            self.assertTrue(np.all((decoded > 0) & (decoded < 1)))

    def test_changed_original_rgb_cannot_reuse_spatial_cache(self) -> None:
        """Cached converted files cannot conceal a changed canonical source."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, row = _media(root)
            spatial_inputs.ensure_spatial([row], [source])
            rgb = Path(row["original"] + ".rgb")
            data = bytearray(rgb.read_bytes())
            data[0] ^= 1
            rgb.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "RGB hash mismatch"):
                spatial_inputs.ensure_spatial([row], [source])

    def test_invalid_derived_cache_is_rebuilt_from_verified_rgb(self) -> None:
        """Disposable conversions are regenerated when their content hash changes."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, row = _media(root)
            first = spatial_inputs.ensure_spatial([row], [source])[0]
            target = Path(first["spatial_distorted"])
            target.write_bytes(b"corrupt derived media")
            rebuilt = spatial_inputs.ensure_spatial([row], [source])[0]
            self.assertEqual(
                first["spatial_distorted_hashes"], rebuilt["spatial_distorted_hashes"]
            )
            self.assertEqual(
                corpus.digest(target), first["spatial_distorted_hashes"]["y4m"]
            )


if __name__ == "__main__":
    unittest.main()
