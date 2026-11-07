#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""GPU integration checks for benchmark checksums, scoring and reports."""

import csv
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path

import make_goldens

ROOT = Path(__file__).resolve().parent.parent
IMAGE_PAIRS = (("gradient.png", "gradient_B.png"), ("alpha_A.png", "alpha_B.png"))
SPATIAL_METRICS = ("ssimulacra2", "butteraugli")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    """Read all rows of a benchmark CSV report.

    :param path: Report path.
    :returns: Parsed rows in report order.
    :raises OSError: If the report cannot be opened.
    """
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class BenchmarkArguments(unittest.TestCase):
    """Exercise benchmark argument parsing without initializing Vulkan."""

    @classmethod
    def setUpClass(cls) -> None:
        """Locate the configured benchmark executable."""
        binary = os.environ.get("VQM_BENCH_BINARY")
        if not binary:
            raise unittest.SkipTest("benchmark binary not configured")
        cls.bench = Path(binary)

    def test_last_metric_selection_replaces_the_previous_metric(self) -> None:
        """Check that the final metric option determines option compatibility."""
        # Missing pairs reach usage only after parsing and validation succeed,
        # and never create a GPU device. VMAF v1 previously left a stale flag.
        for metric in ("ssimulacra2", "butteraugli", "vmaf-v0"):
            with self.subTest(metric=metric):
                result = subprocess.run(
                    [
                        str(self.bench), "--metric", "vmaf-v1", "--metric", metric,
                        "--vmaf-neg"
                    ], capture_output=True, text=True, timeout=5,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertTrue(result.stderr.startswith("Usage:"), result.stderr)
        result = subprocess.run(
            [
                str(self.bench), "--metric", "butteraugli", "--metric", "vmaf-v1",
                "--vmaf-neg"
            ], capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("require VMAF v0", result.stderr)


class BenchmarkIntegration(unittest.TestCase):
    """Exercise benchmark scoring, timings and data paths on a Vulkan device."""

    @classmethod
    def setUpClass(cls) -> None:
        """Probe Vulkan support and create the benchmark image fixtures."""
        bench = os.environ.get("VQM_BENCH_BINARY")
        core = os.environ.get("VQM_TEST_BINARY")
        if not bench or not core:
            raise unittest.SkipTest("benchmark and GPU test binaries not configured")
        cls.bench = Path(bench)
        probe = subprocess.run(
            [core, "--probe-gpu"], capture_output=True, text=True, timeout=120
        )
        if probe.returncode != 0:
            if "GPU required, but no usable Vulkan device" in probe.stderr:
                raise unittest.SkipTest("no usable Vulkan device")
            raise RuntimeError(f"GPU test probe failed: {probe.stderr}")
        cls.directory = tempfile.TemporaryDirectory(prefix="vqm-bench-test-")
        cls.fixtures = Path(cls.directory.name)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tests/make_goldens.py"), "--out-dir",
                str(cls.fixtures)
            ], check=True, capture_output=True, text=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the temporary image and report directory."""
        cls.directory.cleanup()

    def run_bench(
        self, metric: str, original: Path, distorted: Path, *args: str
    ) -> str:
        """Run the benchmark for one image pair and return its stdout.

        :param metric: Benchmark metric name.
        :param original: Original input path.
        :param distorted: Distorted input path.
        :param args: Additional command-line flags and values.
        :returns: Benchmark stdout.
        :raises subprocess.CalledProcessError: If the benchmark exits unsuccessfully.
        :raises subprocess.TimeoutExpired: If the benchmark exceeds its timeout.
        """
        return subprocess.run(
            [
                str(self.bench), "--metric", metric, "--pair",
                str(original),
                str(distorted), *args
            ], check=True, capture_output=True, text=True, timeout=60,
        ).stdout

    @staticmethod
    def score_and_checksum(output: str, name: str) -> tuple[float, float]:
        """Extract the score row and aggregate checksum from benchmark output.

        :param output: Benchmark stdout.
        :param name: Name identifying the requested fixture or output row.
        :returns: Parsed score and checksum.
        :raises StopIteration: If no score row matches the name.
        :raises ValueError: If a score or checksum is not numeric.
        """
        row = next(line for line in output.splitlines() if name in line)
        score = float(row.rsplit(maxsplit=1)[1])
        checksum = float(re.search(r"^checksum ([^\s]+)$", output, re.M).group(1))
        return score, checksum

    def test_metric_pair_scores_checksums_and_stages(self) -> None:
        """Compare independent, shared and unified scoring and stage reports."""
        for input_flags, (original, distorted) in product(((), ("--vulkan-buffers", )),
                                                          IMAGE_PAIRS):
            with self.subTest(input_flags=input_flags, original=original):
                self._check_metric_pair(original, distorted, input_flags)

    def _check_metric_pair(
        self, original: str, distorted: str, input_flags: tuple[str, ...]
    ) -> None:
        """Check paired modes for one fixture pair and input path.

        :param original: Original fixture filename.
        :param distorted: Distorted fixture filename.
        :param input_flags: Extra input path arguments.
        """
        values = {
            mode: self._metric_pair_fields(original, distorted, mode, input_flags)
            for mode in ("independent", "shared", "unified")
        }
        for field in ("ssimulacra2_score", "butteraugli_score", "ssimulacra2_checksum",
                      "butteraugli_checksum"):
            self.assertAlmostEqual(
                values["independent"][field], values["shared"][field], delta=1e-5
            )
            self.assertAlmostEqual(
                values["independent"][field], values["unified"][field], delta=1e-5
            )
        if original == "gradient.png":
            for metric in SPATIAL_METRICS:
                self.assertAlmostEqual(
                    values["shared"][metric + "_checksum"],
                    3 * values["shared"][metric + "_score"], delta=1e-5
                )

    def _metric_pair_fields(
        self, original: str, distorted: str, mode: str, input_flags: tuple[str, ...]
    ) -> dict[str, float]:
        """Measure one paired mode and check its timing and stage fields.

        :param original: Original fixture filename.
        :param distorted: Distorted fixture filename.
        :param mode: Benchmark metric-pair mode.
        :param input_flags: Extra input path arguments.
        :returns: Scores, checksums and wall-clock measures.
        """
        stages = self.fixtures / f"metric-pair-{mode}.csv"
        output = self.run_bench(
            "ssimulacra2", self.fixtures / original, self.fixtures / distorted,
            "--metric-pair", mode, *input_flags, "--warmup", "1", "--iterations", "2",
            "--stages-csv", str(stages)
        )
        fields = {}
        for field in ("ssimulacra2_score", "butteraugli_score", "ssimulacra2_checksum",
                      "butteraugli_checksum", "wall_seconds", "ms_per_pair",
                      "pairs_per_second"):
            match = re.search(rf"(?m)^{field} ([^\s]+)$", output)
            self.assertIsNotNone(match, output)
            fields[field] = float(match.group(1))
        self.assertGreater(fields["wall_seconds"], 0)
        self.assertGreater(fields["ms_per_pair"], 0)
        names = {row["stage"] for row in _csv_rows(stages)}
        self.assertTrue(
            {
                "prepare", "scale", "score", "opsin", "separate", "malta", "diff",
                "mask", "combine", "reduce"
            } <= names
        )
        return fields

    def test_device_info_is_machine_readable(self) -> None:
        """Require device identity in machine-readable output."""
        output = subprocess.run(
            [str(self.bench), "--device-info"], check=True, capture_output=True,
            text=True, timeout=60
        ).stdout
        device = json.loads(output)
        self.assertTrue(device["name"])
        self.assertRegex(device["uuid"], r"^[0-9a-f]{32}$")
        self.assertIsInstance(device["driver_version"], int)

    def test_vmaf_sequence_benchmark_checksum_and_stages(self) -> None:
        """Check VMAF v0 stages, resident checksum and score-only behavior."""
        paths = []
        for name, change in (("vmaf-ref.y4m", 0), ("vmaf-dist.y4m", 2)):
            path = self.fixtures / name
            width = height = 32
            luma = [
                64 + ((x * 11 + y * 7) % 800) + change for y in range(height)
                for x in range(width)
            ]
            plane = struct.pack("<" + "H" * (width * height), *luma)
            chroma = struct.pack("<H", 512) * (width * height)
            path.write_bytes(
                b"YUV4MPEG2 W32 H32 F25:1 Ip A1:1 C444p10\nFRAME\n" + plane + chroma +
                chroma
            )
            paths.append(path)
        reference, distorted = paths
        output = self.run_bench(
            "vmaf-v0", reference, distorted, "--resident", "2", "--warmup", "3",
            "--iterations", "2"
        )
        score, checksum = self.score_and_checksum(output, reference.name)
        self.assertAlmostEqual(checksum, 5 * score, delta=1e-4)
        for stage in ("vmaf_luma", "vmaf_vif", "vmaf_adm", "vmaf_motion"):
            self.assertRegex(output, rf"(?m)^\s+{stage}\s")
        streamed = float(
            self.run_bench("vmaf-v0", reference, distorted, "--score-only").strip()
        )
        self.assertAlmostEqual(score, streamed, delta=1e-5)
        invalid = subprocess.run(
            [
                str(self.bench), "--metric", "vmaf-v0", "--pair",
                str(reference),
                str(distorted), "--score-only", "--stages-csv",
                str(self.fixtures / "invalid-vmaf-stages.csv")
            ], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("cannot report timed stages", invalid.stderr)

    def test_vmaf_v1_benchmark_checksum_and_stages(self) -> None:
        """Check VMAF v1 stages, checksums and streamed scores."""
        paths = []
        width = height = 224
        for name, delta in (("vmaf1-ref.y4m", 0), ("vmaf1-dist.y4m", 3)):
            path = self.fixtures / name
            data = bytearray(b"YUV4MPEG2 W224 H224 F25:1 Ip A1:1 C444p10\nFRAME\n")
            for plane, y, x in product(range(3), range(height), range(width)):
                value = 64 + (
                    (x * 37 + y * 19 + x * y * (plane + 3) + delta * (plane + 1)) % 876
                )
                data.extend(struct.pack("<H", value))
            path.write_bytes(data)
            paths.append(path)
        reference, distorted = paths
        output = self.run_bench(
            "vmaf-v1", reference, distorted, "--resident", "2", "--warmup", "2",
            "--iterations", "2"
        )
        score, checksum = self.score_and_checksum(output, reference.name)
        self.assertAlmostEqual(checksum, 4 * score, delta=1e-4)
        for stage in ("vmaf_v1_luma", "vmaf_v1_adm", "vmaf_v1_speed", "vmaf_v1_cambi",
                      "vmaf_v1_motion"):
            self.assertRegex(output, rf"(?m)^\s+{stage}\s")
        streamed = float(
            self.run_bench("vmaf-v1", reference, distorted, "--score-only").strip()
        )
        self.assertAlmostEqual(score, streamed, delta=1e-4)

    def test_resident_ring_cycles_pairs_and_scores(self) -> None:
        """Verify resident rings cycle through complete pairs with stable scores."""
        a = self.fixtures / "gradient.png"
        b = self.fixtures / "gradient_B.png"
        c = self.fixtures / "gradient_C.png"
        score0, _ = self.score_and_checksum(
            self.run_bench(
                "ssimulacra2", a, b, "--resident", "2", "--warmup", "0", "--iterations",
                "1"
            ), a.name
        )
        score1, _ = self.score_and_checksum(
            self.run_bench(
                "ssimulacra2", b, c, "--resident", "2", "--warmup", "0", "--iterations",
                "1"
            ), b.name
        )
        stages = self.fixtures / "ring-stages.csv"
        output = subprocess.run(
            [
                str(self.bench), "--resident-ring", "2", "--pair",
                str(a),
                str(b), "--pair",
                str(b),
                str(c), "--iterations", "3", "--warmup", "1", "--stages-csv",
                str(stages)
            ], check=True, capture_output=True, text=True, timeout=60
        ).stdout
        scores = [
            float(value)
            for value in re.findall(r"^pair\[\d+\].* score ([^\s]+)$", output, re.M)
        ]
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0], score0, delta=1e-5)
        self.assertAlmostEqual(scores[1], score1, delta=1e-5)
        checksum = float(re.search(r"^checksum ([^\s]+)$", output, re.M).group(1))
        self.assertAlmostEqual(checksum, 2 * (score0 + score1), delta=1e-5)
        self.assertIn("2 pairs, 4 slots, 32x32, 65536 resident input bytes", output)
        self.assertIn("fixed-count wall clock", output)
        with stages.open(newline="") as f:
            self.assertIn("prepare", [row["stage"] for row in csv.DictReader(f)])

    def test_resident_ring_rejects_invalid_workloads(self) -> None:
        """Reject resident ring workloads with incompatible options."""
        a = self.fixtures / "gradient.png"
        b = self.fixtures / "gradient_B.png"
        small = self.fixtures / "ring-small.png"
        make_goldens.write_png(small, 16, 16, 3, [[0] * 48 for _ in range(16)])
        pairs = ["--pair", str(a), str(b), "--pair", str(a), str(small)]
        for options, message in ((["--resident-ring", "0"], "maximum pair count"),
                                 (["--resident-ring", "2", "--workers",
                                   "2"], "one worker"), (["--resident-ring", "2",
                                                          "--iterations", "1",
                                                          *pairs], "cover every"),
                                 (["--resident-ring", "2", "--iterations", "2",
                                   *pairs], "matching image geometries"),
                                 ):
            with self.subTest(options=options):
                result = subprocess.run(
                    [str(self.bench), *options], capture_output=True, text=True,
                    timeout=60
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_distorted_only_alpha_uses_both_backgrounds(self) -> None:
        """Use both alpha backgrounds when only the distorted input has alpha."""
        opaque = self.fixtures / "ring-opaque.png"
        transparent = self.fixtures / "ring-transparent.png"
        make_goldens.write_png(opaque, 32, 32, 3, [[128] * 96 for _ in range(32)])
        make_goldens.write_png(
            transparent, 32, 32, 4, [[128, 128, 128, 0] * 32 for _ in range(32)]
        )
        paths = (
            (), ("--resident", "2"), ("--resident-reference", ), ("--host-views", ),
            ("--vulkan-buffers", ), ("--pool", "--workers", "2")
        )
        thresholds = (
            ("ssimulacra2", self.assertLess, 95.0),
            ("butteraugli", self.assertGreater, 0.1)
        )
        for (metric, check, threshold), options in product(thresholds, paths):
            with self.subTest(metric=metric, options=options):
                output = self.run_bench(
                    metric, opaque, transparent, "--warmup", "0", "--iterations", "1",
                    *options
                )
                score, _ = self.score_and_checksum(output, opaque.name)
                check(score, threshold)
        output = subprocess.run(
            [
                str(self.bench), "--resident-ring", "2", "--pair",
                str(opaque),
                str(transparent), "--pair",
                str(transparent),
                str(opaque), "--warmup", "0", "--iterations", "2"
            ], check=True, capture_output=True, text=True, timeout=60
        ).stdout
        scores = [
            float(value)
            for value in re.findall(r"^pair\[\d+\].* score ([^\s]+)$", output, re.M)
        ]
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(score < 95.0 for score in scores))

    def test_butter_parameters_reject_nonpositive(self) -> None:
        """Reject nonpositive Butteraugli configuration values."""
        for option, value in product(("--hf-asymmetry", "--intensity-target"),
                                     ("0", "-1", "nan", "1junk")):
            with self.subTest(option=option, value=value):
                result = subprocess.run(
                    [str(self.bench), option, value], capture_output=True, text=True,
                    timeout=10
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("invalid", result.stderr)

    def test_checksum_includes_warmup_for_both_metrics(self) -> None:
        """Account for both timed iterations and warmups in checksums."""
        for metric, (a_name, b_name) in product(SPATIAL_METRICS, IMAGE_PAIRS):
            with self.subTest(metric=metric, original=a_name):
                a = self.fixtures / a_name
                b = self.fixtures / b_name
                one = self.run_bench(
                    metric, a, b, "--resident", "2", "--warmup", "0", "--iterations",
                    "1"
                )
                _, dark = self.score_and_checksum(one, a.name)
                output = self.run_bench(
                    metric, a, b, "--resident", "2", "--warmup", "3", "--iterations",
                    "2"
                )
                _, checksum = self.score_and_checksum(output, a.name)
                self.assertAlmostEqual(checksum, dark * 5, delta=1e-5)

    def test_more_workers_than_iterations_preserves_alpha_score(self) -> None:
        """Keep the alpha score stable when some workers have no timed calls."""
        a = self.fixtures / "alpha_A.png"
        b = self.fixtures / "alpha_B.png"
        for metric in ("ssimulacra2", "butteraugli"):
            with self.subTest(metric=metric):
                self._check_alpha_worker_count(metric, a, b)

    def _check_alpha_worker_count(self, metric: str, a: Path, b: Path) -> None:
        """Compare alpha score and checksums for a single worker and two workers.

        :param metric: Metric to measure.
        :param a: Original alpha fixture.
        :param b: Distorted alpha fixture.
        """
        single = self.run_bench(
            metric, a, b, "--resident", "2", "--warmup", "1", "--iterations", "1"
        )
        expected, single_checksum = self.score_and_checksum(single, a.name)
        for iterations in (1, 5):
            output = self.run_bench(
                metric, a, b, "--resident", "2", "--workers", "2", "--warmup", "1",
                "--iterations", str(iterations),
            )
            score, checksum = self.score_and_checksum(output, a.name)
            self.assertAlmostEqual(score, expected, delta=1e-5)
            self.assertAlmostEqual(
                checksum,
                single_checksum * (iterations + 2) / 2, delta=1e-5,
            )

    def test_csv_preserves_commas_and_quotes_in_paths(self) -> None:
        """Round-trip commas and quotes in CSV paths."""
        source = self.fixtures / 'original, "quoted".png'
        distorted = self.fixtures / 'distorted, "quoted".png'
        shutil.copyfile(self.fixtures / "gradient.png", source)
        shutil.copyfile(self.fixtures / "gradient_B.png", distorted)
        for metric in ("ssimulacra2", "butteraugli"):
            with self.subTest(metric=metric):
                report = self.fixtures / f"{metric}.csv"
                self.run_bench(
                    metric, source, distorted, "--iterations", "1", "--warmup", "0",
                    "--csv", str(report)
                )
                rows = _csv_rows(report)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["original"], str(source))
                self.assertEqual(rows[0]["distorted"], str(distorted))

    def test_throughput_rejects_mixed_geometries(self) -> None:
        """Reject mixed image sizes in a throughput workload."""
        small = self.fixtures / "small.png"
        make_goldens.write_png(small, 16, 16, 3, [[0] * 48 for _ in range(16)])
        result = subprocess.run(
            [
                str(self.bench), "--pair",
                str(self.fixtures / "gradient.png"),
                str(self.fixtures / "gradient_B.png"), "--pair",
                str(small),
                str(small), "--workers", "2", "--throughput-seconds", "0.01"
            ], capture_output=True, text=True, timeout=60,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("same geometry", result.stderr)

    def test_multiworker_stages_do_not_claim_wall_shares(self) -> None:
        """Keep aggregate stage durations separate from process wall time."""
        a = self.fixtures / "gradient.png"
        b = self.fixtures / "gradient_B.png"
        for metric in ("ssimulacra2", "butteraugli"):
            with self.subTest(metric=metric):
                report = self.fixtures / f"{metric}-stages.csv"
                output = self.run_bench(
                    metric, a, b, "--resident", "2", "--workers", "2", "--warmup", "1",
                    "--iterations", "2", "--stages-csv", str(report),
                )
                self.assertIn("wall-clock shares and untimed are unavailable", output)
                self.assertIn("ms/iter (cumulative)", output)
                rows = _csv_rows(report)
                self.assertTrue(all(row["share"] == "" for row in rows))
                self.assertEqual(rows[-1]["stage"], "untimed")
                self.assertEqual(rows[-1]["ms_per_iter"], "")
                self.assertTrue(all(row["ms_per_iter"] != "" for row in rows[:-1]))

    def test_ssimu_resident_rebuild_scores_and_fixed_count_rate(self) -> None:
        """Check SSIMULACRA 2 resident scores and fixed-count throughput."""
        for a_name, b_name in (("gradient.png", "gradient_B.png"), ("alpha_A.png",
                                                                    "alpha_B.png")):
            with self.subTest(original=a_name):
                self._check_resident_rebuild(a_name, b_name)

    def _check_resident_rebuild(self, a_name: str, b_name: str) -> None:
        """Compare resident rebuild, multiworker and resident reference results.

        :param a_name: Original fixture filename.
        :param b_name: Distorted fixture filename.
        """
        a = self.fixtures / a_name
        b = self.fixtures / b_name
        expected_output = self.run_bench(
            "ssimulacra2", a, b, "--resident", "2", "--warmup", "0", "--iterations",
            "1",
        )
        expected_score, dark = self.score_and_checksum(expected_output, a.name)
        for options, warm_calls in ((["--resident", "2"], 2), (["--resident", "2",
                                                                "--workers", "2"], 4),
                                    (["--resident-reference"], 2)):
            output = self.run_bench(
                "ssimulacra2", a, b, *options, "--warmup", "2", "--iterations", "3",
            )
            score, checksum = self.score_and_checksum(output, a.name)
            self.assertAlmostEqual(score, expected_score, delta=1e-5)
            self.assertAlmostEqual(checksum, dark * (warm_calls + 3), delta=1e-5)
            if "--resident" in options:
                self.assertIn("fixed-count wall clock", output)

    def test_ssimu_resident_window_scores_with_multiple_workers(self) -> None:
        """Compare resident window scores across worker counts."""
        a = self.fixtures / "gradient.png"
        b = self.fixtures / "gradient_B.png"
        expected_output = self.run_bench(
            "ssimulacra2", a, b, "--resident", "2", "--warmup", "0", "--iterations",
            "1",
        )
        score, _ = self.score_and_checksum(expected_output, a.name)
        for workers in (1, 2):
            with self.subTest(workers=workers):
                output = self.run_bench(
                    "ssimulacra2", a, b, "--resident", "2", "--workers", str(workers),
                    "--throughput-seconds", "0.02",
                )
                fields = dict(token.split("=", 1) for token in output.split()[1:])
                self.assertGreater(int(fields["calls"]), 0)
                self.assertAlmostEqual(
                    float(fields["checksum"]) / int(fields["calls"]), score, delta=1e-5
                )

    def test_host_view_mode_matches_streaming_scores(self) -> None:
        """Compare borrowed host view scores with streamed scores."""
        for metric, (original, distorted) in product(SPATIAL_METRICS, IMAGE_PAIRS):
            with self.subTest(metric=metric, original=original):
                a, b = self.fixtures / original, self.fixtures / distorted
                options = ("--iterations", "2", "--warmup", "1")
                ordinary = self.run_bench(metric, a, b, *options)
                views = self.run_bench(metric, a, b, *options, "--host-views")
                self.assertEqual(
                    self.score_and_checksum(views, original),
                    self.score_and_checksum(ordinary, original)
                )

    def test_host_view_mode_rejects_resident_measurement(self) -> None:
        """Reject resident measurement with borrowed host views."""
        result = subprocess.run(
            [str(self.bench), "--host-views", "--resident", "2"], capture_output=True,
            text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("streaming latency mode", result.stderr)

    def test_public_pool_benchmark_uses_fixed_count_and_warmups(self) -> None:
        """Verify public pool call counts and per-worker warmup accounting."""
        for metric, (a_name, b_name) in product(SPATIAL_METRICS, IMAGE_PAIRS):
            with self.subTest(metric=metric, original=a_name):
                a = self.fixtures / a_name
                b = self.fixtures / b_name
                single = self.run_bench(
                    metric, a, b, "--iterations", "1", "--warmup", "0"
                )
                expected, dark = self.score_and_checksum(single, a.name)
                output = self.run_bench(
                    metric, a, b, "--pool", "--workers", "2", "--iterations", "3",
                    "--warmup", "1"
                )
                score, checksum = self.score_and_checksum(output, a.name)
                self.assertAlmostEqual(score, expected, delta=1e-5)
                self.assertAlmostEqual(checksum, dark * 5, delta=1e-5)
                self.assertIn("pool:", output)
                self.assertIn("fixed-count wall clock", output)

        invalid = subprocess.run(
            [
                str(self.bench), "--pair",
                str(self.fixtures / "gradient.png"),
                str(self.fixtures / "gradient_B.png"), "--pool", "--resident", "2"
            ], capture_output=True, text=True, timeout=60,
        )
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("fixed-count streaming", invalid.stderr)

        invalid_views = subprocess.run(
            [str(self.bench), "--pool", "--host-views"], capture_output=True, text=True,
            timeout=5,
        )
        self.assertNotEqual(invalid_views.returncode, 0)
        self.assertIn("--host-views", invalid_views.stderr)

        unsupported_vulkan = subprocess.run(
            [str(self.bench), "--pool", "--vulkan-buffers"], capture_output=True,
            text=True, timeout=5,
        )
        self.assertNotEqual(unsupported_vulkan.returncode, 0)

    def test_vulkan_buffer_mode_matches_streaming_scores(self) -> None:
        """Compare caller-owned Vulkan buffer scores with streamed scores."""
        for metric, (original, distorted) in product(SPATIAL_METRICS, IMAGE_PAIRS):
            with self.subTest(metric=metric, original=original):
                a, b = self.fixtures / original, self.fixtures / distorted
                options = ("--iterations", "2", "--warmup", "1")
                ordinary = self.run_bench(metric, a, b, *options)
                external = self.run_bench(metric, a, b, *options, "--vulkan-buffers")
                self.assertEqual(
                    self.score_and_checksum(external, original),
                    self.score_and_checksum(ordinary, original)
                )

    def test_vulkan_buffer_mode_rejects_incompatible_options(self) -> None:
        """Reject incompatible Vulkan buffer modes before scoring."""
        for option in ("--resident", "--resident-reference", "--host-views", "--pool"):
            args = [str(self.bench), "--vulkan-buffers", option]
            if option == "--resident":
                args.append("2")
            result = subprocess.run(args, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--vulkan-buffers", result.stderr)

    def test_unsupported_half_planes_are_rejected(self) -> None:
        """Reject half-plane storage for a metric that does not support it."""
        result = subprocess.run(
            [str(self.bench), "--metric", "butteraugli", "--half-planes"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("only for ssimulacra2", result.stderr)

    def test_invalid_numeric_arguments_are_rejected(self) -> None:
        """Reject invalid benchmark counts and numeric parameters."""
        for flag, value in (("--intensity-target", "80oops"),
                            ("--intensity-target", "nan"), ("--intensity-target", "0"),
                            ("--hf-asymmetry", "inf"), ("--throughput-seconds", "inf"),
                            ("--throughput-seconds", "1e300")):
            with self.subTest(flag=flag, value=value):
                result = subprocess.run(
                    [str(self.bench), flag, value], capture_output=True, text=True,
                    timeout=5,
                )
                self.assertEqual(result.returncode, 2)

    def test_vmaf_chroma_requires_a_supported_mode_and_metric(self) -> None:
        """Validate VMAF chroma mode names and metric compatibility."""
        for args in (("--metric", "vmaf-v1", "--vmaf-chroma", "422"), ("--vmaf-chroma",
                                                                       "420")):
            with self.subTest(args=args):
                result = subprocess.run(
                    [str(self.bench), *args], capture_output=True, text=True, timeout=5
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("--vmaf-chroma", result.stderr)


class MetricPairArgumentValidation(unittest.TestCase):
    """Check metric-pair option errors independently of Vulkan."""

    def test_unsupported_modes_fail_before_gpu_initialization(self) -> None:
        """Reject unsupported metric-pair modes before device initialization."""
        bench = os.environ.get("VQM_BENCH_BINARY")
        if not bench:
            self.skipTest("benchmark binary not configured")
        for flags in (("--metric-pair",
                       "invalid"), ("--metric-pair", "shared", "--resident",
                                    "2"), ("--metric-pair", "shared", "--workers", "2"),
                      ("--metric-pair", "shared", "--score-only"),
                      ("--metric-pair", "shared", "--csv", "unused.csv")):
            with self.subTest(flags=flags):
                result = subprocess.run(
                    [bench, *flags], capture_output=True, text=True, timeout=5
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("--metric-pair", result.stderr)


if __name__ == "__main__":
    unittest.main()
