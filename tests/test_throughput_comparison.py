"""Check reproducible throughput accounting and fail-closed publication."""

import contextlib
import dataclasses
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench import benchmark
from bench import throughput
from bench import throughput_report as reports
from bench import throughput_workload as workload


class ThroughputTests(unittest.TestCase):
    """Exercise comparison contracts without requiring a GPU or references."""

    def setUp(self) -> None:
        """Prepare isolated reports and the complete default matrix."""
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.args = throughput.arguments(["--out-dir", str(self.directory)])
        self.arms = workload.matrix(
            self.args,
            workload.entries(self.args.build_dir, self.args.references)
        )

    def _rows(self) -> list[dict[str, object]]:
        return [
            {
                **dataclasses.asdict(arm), "round": index,
                "status": 0,
                "count": arm.count,
                "fps": 100.0 + index,
                "score": 10.0
            } for arm in self.arms for index in range(1, self.args.rounds + 1)
        ]

    def _report(self) -> dict[str, object]:
        rows = self._rows()
        return {
            "status": "complete",
            "created_at": "2026-11-07T12:00:00-05:00",
            "protocol": vars(self.args).copy(),
            "matrix": [dataclasses.asdict(arm) for arm in self.arms],
            "measurements": rows,
            "summary": reports.summarize(self.arms, rows, self.args.rounds),
            "environment": {
                "devices": {
                    "status": "matched_by_name_only",
                    "ours": {
                        "name": "Test GPU"
                    }
                },
                "project": {
                    "revision": "fixture"
                },
                "references": {
                    name: {
                        "revision": "pin"
                    }
                    for name in workload.PEERS
                }
            }
        }

    def _documents(self) -> None:
        (self.directory / "bench").mkdir()
        self.document = f"prefix\n{reports.START}\nold\n{reports.END}\nsuffix\n"
        for name in ("README.md", "bench/README.md"):
            (self.directory / name).write_text(self.document)

    def test_full_matrix_counts_and_workers(self) -> None:
        """Cover the full matrix with total, not per-worker counts."""
        self.assertEqual(len(self.arms), 36)
        self.assertEqual(len(self._rows()), 180)
        self.assertEqual(len({arm.key for arm in self.arms}), 36)
        for arm in self.arms:
            expected = 1 if arm.metric.startswith("cvvdp") else 3
            self.assertEqual(arm.workers, expected)
            self.assertEqual(arm.count, arm.iterations * arm.frames)
            if arm.metric == "cvvdp-video":
                self.assertEqual(arm.implementation, "vqm")
                self.assertIn("--cvvdp-video", arm.command)
                self.assertEqual(arm.frames, workload.clip_frames(arm.size))

    def test_rotating_order_preserves_coverage(self) -> None:
        """Rotate serial rounds without changing the workload."""
        first = self.arms[:18]
        second = workload.round_order(first, 1)
        self.assertEqual(set(first), set(second))
        self.assertNotEqual(first[0], second[0])

    def test_peer_selection_does_not_invent_temporal_arms(self) -> None:
        """Allow targeted diagnostics without implying unmeasured peer rates."""
        drivers = workload.entries(self.args.build_dir, ["vship-cuda"])
        arms = workload.matrix(self.args, drivers)
        self.assertEqual(len(arms), 24)
        peers = [arm.metric for arm in arms if arm.implementation != "vqm"]
        self.assertEqual(set(peers), workload.SPATIAL)

    def test_list_dispatch_does_not_build_or_probe(self) -> None:
        """List the entire matrix offline through the public entry point."""
        output = io.StringIO()
        setup = self.enterContext(mock.patch.object(throughput, "_setup"))
        with contextlib.redirect_stdout(output):
            status = benchmark.command_line(["--suite", "throughput", "--list"])
        self.assertEqual(status, 0)
        setup.assert_not_called()
        self.assertEqual(len(json.loads(output.getvalue())), 36)

    def test_reduced_protocol_cannot_request_publication(self) -> None:
        """Reject smoke, reduced rounds, missing peers and different workers."""
        variants = (
            ["--profile", "smoke"], ["--rounds", "4"], ["--warmup", "2"],
            ["--workers", "1"], ["--references", "vship-cuda"]
        )
        for options in variants:
            with self.subTest(options=options), contextlib.redirect_stderr(
                    io.StringIO()), self.assertRaises(SystemExit):
                throughput.arguments([*options, "--update-readme"])

    def test_peer_order_is_canonical(self) -> None:
        """Allow any peer order without altering publication coverage."""
        args = throughput.arguments(
            ["--references", ",".join(reversed(workload.PEERS))]
        )
        self.assertEqual(args.references, list(workload.PEERS))

    def test_missing_receipt_or_wrong_units_are_rejected(self) -> None:
        """Do not silently interpret images/s as temporal frames/s."""
        arm = next(arm for arm in self.arms if arm.metric == "cvvdp-video")
        for receipt in ("no wall clock",
                        "fixed-count wall clock: 2 ms/image, 500 images/s"):
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                throughput.parse_measurement(arm, receipt, Path("unused"), 3)

    def test_machine_receipts_validate_counts_and_time(self) -> None:
        """Require exact total count, worker count, geometry and finite rate."""
        arm = self.arms[0]
        receipt = (
            f"throughput calls={arm.count} threads=3 size={arm.size} "
            f"seconds=2 fps={arm.count / 2} checksum={arm.count * 10}\n"
        )
        row = throughput.parse_measurement(arm, receipt, Path("unused"), 3)
        self.assertEqual(row["score"], 10)
        invalid = (
            receipt.replace("threads=3", "threads=2"),
            receipt.replace(f"calls={arm.count}",
                            "calls=1"), receipt.replace(arm.size, "64x64"),
            receipt.replace("seconds=2", "seconds=1"),
            receipt.replace(f"fps={arm.count / 2}",
                            "fps=nan"), receipt + receipt
        )
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                throughput.parse_measurement(arm, text, Path("unused"), 3)

    def test_fixed_checksums_include_warmups_not_extra_frames(self) -> None:
        """Distinguish worker warmups, still calls and complete-clip JOD."""
        selected = [
            arm for arm in self.arms if arm.size == workload.SIZES[0]
            and arm.mode == "streaming" and not arm.machine
        ]
        for arm in selected:
            unit = {
                "psnr-hvs": "pair",
                "cvvdp-image": "image",
                "cvvdp-video": "frame"
            }[arm.metric]
            calls = (
                arm.iterations + 3 if arm.frames > 1 else arm.count +
                3 * arm.workers
            )
            csv_path = self.directory / "receipt.csv"
            csv_path.write_text(
                "width,height,frames\n1920,1080," + str(arm.frames) + "\n"
            )
            receipt = (
                f"fixed-count wall clock: 2.000000 ms/{unit}, "
                f"500.000 {unit}s/s\nchecksum {calls * 10}\n"
            )
            row = throughput.parse_measurement(arm, receipt, csv_path, 3)
            self.assertEqual(row["score"], 10)
            self.assertEqual(row["count"], arm.count)
        csv_path.write_text("width,height,frames\n1920,1080,1\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            throughput.parse_measurement(arm, receipt, csv_path, 3)

    def test_failed_processes_preserve_receipts(self) -> None:
        """Never accept plausible output from a crashing process."""
        result = subprocess.CompletedProcess([], -11, "partial", "crash")
        self.enterContext(
            mock.patch.object(subprocess, "run", return_value=result)
        )
        with self.assertRaisesRegex(RuntimeError, "benchmark failed"):
            throughput.measure(self.arms[0], 1, self.args)
        path = self.directory / f"{self.arms[0].key}-r1.process.json"
        self.assertEqual(json.loads(path.read_text())["status"], -11)

    def test_timeouts_preserve_partial_output(self) -> None:
        """Retain timed-out evidence without returning a measurement."""
        error = subprocess.TimeoutExpired([], 1, output=b"partial")
        self.enterContext(
            mock.patch.object(subprocess, "run", side_effect=error)
        )
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            throughput.measure(self.arms[0], 1, self.args)
        path = self.directory / f"{self.arms[0].key}-r1.stdout"
        self.assertEqual(path.read_text(), "partial")

    def test_summary_rejects_bad_coverage_and_scores(self) -> None:
        """Reject missing/duplicate rounds, failures and unstable scores."""
        rows = self._rows()
        invalid = (
            rows[:-1], [*rows, rows[-1]], [{
                **rows[0], "status": 1
            }, *rows[1:]], [{
                **rows[0], "score": 11
            }, *rows[1:]], [{
                **rows[0], "fps": float("inf")
            }, *rows[1:]]
        )
        for values in invalid:
            with self.subTest(first=values[0]), self.assertRaises(ValueError):
                reports.summarize(self.arms, values, 5)

    def test_summary_rejects_streaming_resident_mismatch(self) -> None:
        """Require identical candidate scores for both input paths."""
        rows = self._rows()
        for row in rows:
            if row["mode"] == "resident":
                row["score"] = 11
        with self.assertRaisesRegex(ValueError, "scores disagree"):
            reports.summarize(self.arms, rows, 5)

    def test_cache_and_binary_changes_are_rejected(self) -> None:
        """Refuse modified inputs and binaries rather than mixing evidence."""
        target = self.directory / "binary"
        target.write_text("initial")
        hashes = {str(target): workload.file_sha256(target)}
        throughput._verify_fingerprints(hashes)
        target.write_text("changed")
        with self.assertRaises(RuntimeError):
            throughput._verify_fingerprints(hashes)
        with self.assertRaises(RuntimeError):
            workload._verify_inputs(self.directory, {"version": 1})

    def test_publication_preserves_surrounding_document(self) -> None:
        """Generate both README tables from validated complete evidence."""
        self._documents()
        reports.publish(self._report(), self.directory)
        text = (self.directory / "README.md").read_text()
        self.assertTrue(text.startswith("prefix\n"))
        self.assertTrue(text.endswith("\nsuffix\n"))
        for required in ("Vship-fast Vulkan", "CVVDP video", "resident"):
            self.assertIn(required, text)
        self.assertNotIn("\nold\n", text)

    def test_both_documents_are_validated_before_writing(self) -> None:
        """Leave the first README alone if the second lacks markers."""
        self._documents()
        (self.directory / "bench/README.md").write_text("no markers")
        with self.assertRaises(ValueError):
            reports.publish(self._report(), self.directory)
        self.assertEqual(
            (self.directory / "README.md").read_text(), self.document
        )

    def test_headline_labels_streaming_and_resident_results(self) -> None:
        """Keep both input modes and identify peer timing boundaries."""
        text = reports.headline(self._report())
        for label in ("1080p streaming", "1080p resident", "4K streaming",
                      "4K resident", "Competitor comparison: streaming inputs",
                      "per-call uploads"):
            self.assertIn(label, text)
        self.assertNotIn("Do not compare", text)

    def test_invalid_markers_are_rejected(self) -> None:
        """Reject missing, duplicate and out-of-order markers."""
        for text in ("none", reports.START * 2 + reports.END,
                     reports.END + reports.START):
            with self.subTest(text=text), self.assertRaises(ValueError):
                reports.replace_section(text, "replacement")

    def test_publication_revalidates_evidence(self) -> None:
        """Do not trust a complete label or forged summary alone."""
        self._documents()
        report = self._report()
        report["measurements"].pop()
        with self.assertRaises(ValueError):
            reports.publish(report, self.directory)
        report = self._report()
        report["matrix"].pop()
        with self.assertRaises(ValueError):
            reports.publish(report, self.directory)
        report = self._report()
        report["summary"][0]["median"] = 9999
        with self.assertRaises(ValueError):
            reports.publish(report, self.directory)


if __name__ == "__main__":
    unittest.main()
