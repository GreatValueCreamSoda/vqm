#!/usr/bin/env python3
"""CPU-only tests for validation score parsing and gate reporting."""

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))
common = importlib.import_module("common")
compare_vmaf_scores = importlib.import_module("compare_vmaf_scores")

spec = importlib.util.spec_from_file_location(
    "run_corpus", ROOT / "validation" / "run_corpus.py"
)
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)
spec = importlib.util.spec_from_file_location(
    "build_references", ROOT / "tools" / "build_references.py"
)
references = importlib.util.module_from_spec(spec)
spec.loader.exec_module(references)


class ScoreParsingTests(unittest.TestCase):
    """Check validation input matching and finite score parsing."""

    def test_vmaf_comparison_joins_by_pair_and_reports_drift(self) -> None:
        """Compare score drift by pair identity regardless of input order."""
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory, "baseline.csv")
            candidate = Path(directory, "candidate.csv")
            baseline.write_text("original,distorted,vqm_v0\na,b,10\nc,d,20\n")
            candidate.write_text("original,distorted,vqm_v0\nc,d,20.2\na,b,9.9\n")
            left, _ = compare_vmaf_scores.load_scores(baseline, "vqm_v0")
            right, _ = compare_vmaf_scores.load_scores(candidate, "vqm_v0")
            result = compare_vmaf_scores.compare(left, right)
            self.assertEqual(result["pairs"], 2)
            self.assertFalse(result["input_hashes_verified"])
            self.assertAlmostEqual(result["max_absolute"], 0.2)
            self.assertAlmostEqual(result["mean_absolute"], 0.15)
            self.assertAlmostEqual(result["p99_absolute"], 0.199)
            self.assertEqual(result["worst_pair"], {"original": "c", "distorted": "d"})
            candidate.write_text("original,distorted,vqm_v0\na,b,nan\n")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                compare_vmaf_scores.load_scores(candidate, "vqm_v0")
            candidate.write_text("original,distorted,vqm_v0\na,b,10\na,b,10\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                compare_vmaf_scores.load_scores(candidate, "vqm_v0")
            with self.assertRaisesRegex(ValueError, "pair mismatch"):
                compare_vmaf_scores.compare({("a", "b"): 1}, {("c", "d"): 1})
            with self.assertRaisesRegex(ValueError, "input hashes differ"):
                compare_vmaf_scores.compare(
                    left, right, {("a", "b"): ("x", "y")}, {("a", "b"): ("x", "z")}
                )

    def test_non_finite_scores_are_rejected(self) -> None:
        """Reject nonfinite scalar score output."""
        for text in ("nan", "inf", "-inf"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                common.parse_score(text)
        self.assertEqual(common.parse_score("warning\n-42.5\n"), -42.5)

    def test_batch_rejects_non_finite_and_cleans_pair_file(self) -> None:
        """Reject nonfinite batch scores and remove the private pair list."""
        result = SimpleNamespace(returncode=0, stdout="1.5\nnan\n-inf\n", stderr="")
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(corpus.subprocess, "run", return_value=result):
            values = corpus.run_batch_chunk(
                {"command": ["fake", "{pairs_file}"]},
                [("a", "b"), ("c", "d"), ("e", "f")], directory, 1,
                os.path.join(directory, "run.log")
            )
            self.assertEqual(values[0], (1.5, None))
            self.assertIsNone(values[1][0])
            self.assertIsNotNone(values[1][1])
            self.assertIsNone(values[2][0])
            self.assertEqual(list(Path(directory).glob("*.pairs")), [])

    def test_launch_failure_is_reported_for_batch_and_single_pair(self) -> None:
        """Report process launch failures in both scoring paths."""
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(corpus.subprocess, "run",
                               side_effect=PermissionError("not executable")):
            batch = corpus.run_batch_chunk(
                {"command": ["fake", "{pairs_file}"]}, [("a", "b")], directory, 1,
                os.path.join(directory, "run.log")
            )
            self.assertIsNone(batch[0][0])
            self.assertIn("could not run fake", batch[0][1])
            self.assertEqual(list(Path(directory).glob("*.pairs")), [])
            value, error = common.run_score(["fake"], "a", "b")
            self.assertIsNone(value)
            self.assertIn("could not run fake", error)

    def test_non_finite_mcos_is_rejected(self) -> None:
        """Reject a nonfinite subjective score in the corpus CSV."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mcos.csv"
            path.write_text("reference_img,distorted_img,MCOS\na,b,nan\n")
            with self.assertRaises(ValueError):
                common.load_mcos(path)


class GateReportingTests(unittest.TestCase):
    """Exercise validation gate selection and coverage reporting."""

    def test_missing_reference_is_skipped_without_crashing(self) -> None:
        """Report an unavailable reference without crashing."""
        pairs = [("a", "b"), ("c", "d")]
        mcos = {pairs[0]: 1.0, pairs[1]: 2.0}
        impl = {"name": "ours", "command": [sys.executable], "batch": True}
        metric = {
            "reference": "cloudinary",
            "direction": "higher",
            "implementations": [impl]
        }
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                data_root=directory, csv=None, out_dir=directory, timeout=1, jobs=1,
                max_threshold=0.01, mean_threshold=0.001, accuracy_tolerance=0.005,
                strict_parity=False
            )
            output = io.StringIO()
            with mock.patch.object(corpus, "batch_scores",
                                   return_value=[(1.0, None), (2.0, None)]), \
                 mock.patch.object(sys, "argv", ["run_corpus.py", "--limit", "2",
                                                  "--only", "ours"]), \
                 contextlib.redirect_stdout(output):
                ok = corpus.run_metric(
                    "ssimulacra2", metric, pairs, mcos, args, {"ours"}
                )
            self.assertTrue(ok)
            self.assertIn("parity vs reference: SKIP", output.getvalue())
            self.assertIn("accuracy vs MCOS: SKIP", output.getvalue())
            summary = Path(directory, "ssimulacra2-summary.md").read_text()
            self.assertIn("--limit 2 --only ours", summary)
            self.assertIn("No reference comparison was selected", summary)
            args.strict_parity = True
            with mock.patch.object(corpus, "batch_scores",
                                   return_value=[(1.0, None), (2.0, None)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(
                    corpus.run_metric(
                        "ssimulacra2", metric, pairs, mcos, args, {"ours"}
                    )
                )

    def test_summary_counts_successes_instead_of_attempts(self) -> None:
        """Count successful pair scores rather than attempted pairs."""
        pairs = [("a", "b"), ("c", "d")]
        mcos = {pairs[0]: 1.0, pairs[1]: 2.0}
        metric = {
            "reference":
            "ours",
            "direction":
            "higher",
            "implementations":
            [{
                "name": "ours",
                "command": [sys.executable],
                "batch": True
            }]
        }
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                data_root=directory, csv=None, out_dir=directory, timeout=1, jobs=1,
                max_threshold=0.01, mean_threshold=0.001, accuracy_tolerance=0.005,
                strict_parity=False
            )
            with mock.patch.object(corpus, "batch_scores",
                                   return_value=[(1.0, None), (None, "bad")]), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertFalse(
                    corpus.run_metric(
                        "ssimulacra2", metric, pairs, mcos, args, {"ours"}
                    )
                )
            summary = Path(directory, "ssimulacra2-summary.md").read_text()
            self.assertRegex(summary, r"\| ours \| 1 \| [0-9.]+ \|")
            self.assertIn("successfully scored pairs only", summary)

    def test_invalid_limits_and_non_finite_thresholds_are_rejected(self) -> None:
        """Reject invalid pair limits and numerical thresholds."""
        for argv in (("--limit", "-1"), ("--jobs", "0"), ("--accuracy-tolerance",
                                                          "nan"), ("--timeout", "inf"),
                     ("--max-threshold", "-1")):
            with (self.subTest(argv=argv), mock.patch.object(sys, "argv",
                                                             ["run_corpus.py", *argv]),
                  contextlib.redirect_stderr(io.StringIO()
                                             ), self.assertRaises(SystemExit),
                  ):
                corpus.parse_args()


class ReferenceRebuildTests(unittest.TestCase):
    """Exercise dependency checks for reference harness rebuilds."""

    def test_existing_cmake_harnesses_use_incremental_build(self) -> None:
        """Use incremental CMake builds for existing reference harnesses."""
        for build, target in ((references.build_cloudinary, "ss2_cloudinary_bench"),
                              (references.build_libjxl, "ss2_libjxl_bench")):
            with self.subTest(target=target), \
                 mock.patch.object(references, "require_reference"), \
                 mock.patch.object(references, "need_build", return_value=False), \
                 mock.patch.object(references, "run") as run:
                build(3, False)
                command = run.call_args.args[0]
                self.assertEqual(command[:2], ["cmake", "--build"])
                self.assertIn(target, command)

    def test_existing_direct_harnesses_are_recompiled(self) -> None:
        """Recompile harnesses built directly by the compiler."""
        with mock.patch.object(references, "require_reference"), \
             mock.patch.object(references, "need_build", return_value=False), \
             mock.patch.object(references.os.path, "isfile", return_value=True), \
             mock.patch.object(references, "copy_reference"), \
             mock.patch.object(references, "zig_version") as zig, \
             mock.patch.object(references, "run") as run:
            references.build_fssimu2(2, False)
            self.assertIn("fssimu2_bench.cc", " ".join(run.call_args.args[0]))
            self.assertTrue(
                any(arg.endswith("/libvqm_core.a") for arg in run.call_args.args[0])
            )
            self.assertTrue(
                any(arg.endswith("/libvqm_io.a") for arg in run.call_args.args[0])
            )
            zig.assert_not_called()
            run.reset_mock()
            references.build_vship_reference(
                "Vship", "Vulkan", "vship-build-vulkan", 2, False
            )
            self.assertIn("vship_bench.cc", " ".join(run.call_args.args[0]))
            self.assertEqual(run.call_count, 1)

    def test_existing_core_archives_are_checked_by_build_system(self) -> None:
        """Check core archive freshness through the build system."""
        with mock.patch.object(references.os.path, "isfile", return_value=True), \
             mock.patch.object(references, "run") as run:
            references.preflight(2)
            self.assertEqual(run.call_args.args[0][:3], ["cmake", "--build", "build"])
            self.assertIn("vqm_core", run.call_args.args[0])
            self.assertIn("vqm_io", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
