#!/usr/bin/env python3
"""CPU-only checks for benchmark and validation tooling."""

import contextlib
import importlib.util
import io
import itertools
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest import mock

from bench import inmemory_environment as probes
from bench import inmemory_measure as measurements
from bench import inmemory_report as reports
from bench import run_inmemory as inmemory

ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, path: str | Path) -> ModuleType:
    """Load a repository script by its path for offline tests.

    :param name: Name identifying the requested fixture or output row.
    :param path: Destination or source filesystem path.
    :returns: Loaded script module.
    """
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load_module("run_bench", "bench/run_bench.py")
fetch_cid22 = load_module("fetch_cid22", "validation/fetch_cid22.py")
bootstrap = load_module("benchmark_bootstrap", "bench/benchmark.py")


class BenchmarkBootstrapTests(unittest.TestCase):
    """Check reference bootstrap, benchmark plans and provenance."""

    def test_synthetic_only_does_not_import_numpy(self) -> None:
        """Generate synthetic pairs without importing NumPy."""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable, "-S",
                    str(ROOT / "bench/make_bench_pairs.py"), "--synthetic-only",
                    "--sizes", "", "--out-dir", directory
                ], capture_output=True, text=True
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("no pair list could be written", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_synthetic_dimensions_are_checked_before_writing(self) -> None:
        """Reject invalid geometries before generating any synthetic pair."""
        target = Path(
            self.enterContext(tempfile.TemporaryDirectory())
        ) / "pairs"
        for invalid in ("17x20", "20x19", "0x16", "-2x16"):
            with self.subTest(invalid=invalid):
                result = subprocess.run(
                    [
                        sys.executable, "-S",
                        str(ROOT / "bench/make_bench_pairs.py"
                            ), "--synthetic-only", "--sizes",
                        f"32x32,{invalid}", "--out-dir",
                        str(target)
                    ], capture_output=True, text=True, check=False, timeout=10
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("positive and even", result.stderr)
                self.assertFalse(target.exists())

    def test_custom_pair_runs_selected_references(self) -> None:
        """Forward custom pairs and the selected reference arms."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.png"
            distorted = root / "distorted.png"
            original.touch()
            distorted.touch()
            commands = []
            arguments = [
                "benchmark.py", "--pair",
                str(original),
                str(distorted), "--references", "vship-vulkan,vship-cuda",
                "--iterations", "2", "--rounds", "1"
            ]
            with (mock.patch.object(bootstrap, "ROOT", root),
                  mock.patch.object(bootstrap, "run",
                                    side_effect=commands.append),
                  mock.patch.object(sys, "argv", arguments)):
                self.assertEqual(bootstrap.main(), 0)
            self.assertEqual(commands[0][-1], "Vship")
            self.assertIn("vship-vulkan,vship-cuda", commands[3])
            self.assertIn(
                "ours-resident,ours-streaming,vship-vulkan,vship-cuda",
                commands[4]
            )
            self.assertEqual(
                (root / ".cache/bench/quick-pairs/custom.txt").read_text(),
                f"{original}\t{distorted}\n"
            )
            self.assertEqual(
                commands[4][commands[4].index("--out-dir") + 1],
                str(root / "artifacts/reports/quick-benchmark")
            )

    def test_report_header_has_nine_columns(self) -> None:
        """Keep the benchmark report header aligned with its table columns."""
        args = SimpleNamespace(
            pairs_file="pairs.txt", iterations=1, warmup=0, rounds=1
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.md"
            reports.write_summary(path, args, 1, [])
            header = next(
                line for line in path.read_text().splitlines()
                if line.startswith("| implementation")
            )
            self.assertEqual(len(header.split("|")) - 2, 9)

    def test_device_matching_uses_vship_enumeration_and_flags_ambiguity(
        self
    ) -> None:
        """Match the benchmark device uniquely against Vship's enumeration."""
        ours = {"name": "ours-resident", "driver": "ours"}
        vship = {"name": "vship-cuda", "driver": "vship"}
        duplicate_metric = {"name": "vship-cuda", "driver": "vship"}

        def probe(command: Sequence[str], cwd: str | Path | None = None) -> str:
            if command[0] == "ours":
                return json.dumps({"name": "Target GPU", "uuid": "our-uuid"})
            return json.dumps(
                [
                    {
                        "id": 0,
                        "name": "Other GPU"
                    }, {
                        "id": 1,
                        "name": "Target GPU"
                    }
                ]
            )

        with (mock.patch.object(probes, "command_output", side_effect=probe),
              contextlib.redirect_stderr(io.StringIO())):
            report = probes.probe_devices([ours, vship, duplicate_metric])
        self.assertEqual(vship["args"], ["--gpu", "1"])
        self.assertEqual(duplicate_metric["args"], ["--gpu", "1"])
        self.assertEqual(report["status"], "matched_by_name_only")

        def ambiguous(
            command: Sequence[str], cwd: str | Path | None = None
        ) -> str:
            if command[0] == "ours":
                return json.dumps({"name": "Target GPU", "uuid": "our-uuid"})
            return json.dumps(
                [
                    {
                        "id": 0,
                        "name": "Target GPU"
                    }, {
                        "id": 1,
                        "name": "Target GPU"
                    }
                ]
            )

        with (mock.patch.object(probes, "command_output",
                                side_effect=ambiguous),
              contextlib.redirect_stderr(io.StringIO())):
            with self.assertRaisesRegex(SystemExit, "multiple devices"):
                probes.probe_devices(
                    [ours, {
                        "name": "vship-cuda",
                        "driver": "vship"
                    }]
                )
            report = probes.probe_devices(
                [ours, {
                    "name": "vship-cuda",
                    "driver": "vship"
                }], True
            )
        self.assertEqual(report["status"], "unverified")

    def test_environment_records_input_hashes_and_cmake_compiler(self) -> None:
        """Record input hashes and the configured CMake compiler."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "build").mkdir()
            (root / "build/CMakeCache.txt"
             ).write_text("CMAKE_CXX_COMPILER:STRING=/custom/bin/c++\n")
            image = root / "image.png"
            image.write_bytes(b"test image")
            executable = root / "driver"
            executable.write_bytes(b"driver")

            def output(
                command: Sequence[str], cwd: str | Path | None = None
            ) -> str | None:
                if command[:1] == ["/custom/bin/c++"]:
                    return "compiler version"
                if "status" in command:
                    return ""
                if "rev-parse" in command:
                    return "deadbeef"
                return None

            with (mock.patch.object(probes, "REPO_ROOT", str(root)),
                  mock.patch.object(measurements, "REPO_ROOT",
                                    str(root)), mock.patch.object(inmemory,
                                                                  "REPO_ROOT",
                                                                  str(root)),
                  mock.patch.object(probes, "command_output",
                                    side_effect=output)):
                environment = probes.capture_environment(
                    [(str(image), str(image))],
                    [{
                        "name": "ours-resident",
                        "driver": str(executable)
                    }], {"status": "not_compared"}
                )
            self.assertEqual(
                environment["project"], {
                    "revision": "deadbeef",
                    "dirty": False
                }
            )
            self.assertEqual(
                environment["compiler"]["version"], "compiler version"
            )
            self.assertEqual(
                len(environment["inputs"][str(image)]["sha256"]), 64
            )
            self.assertEqual(
                len(environment["executables"]["ours-resident"]["sha256"]), 64
            )


def _write_pair_lists(root: Path) -> tuple[list[Path], list[Path], Path, Path]:
    """Write the latency and throughput workload fixtures.

    :param root: Isolated temporary benchmark repository.
    :returns: Latency images, throughput images and their respective pair lists.
    """
    latency_images = [root / "latency-orig.png", root / "latency-dist.png"]
    throughput_images = [root / "window-orig.png", root / "window-dist.png"]
    for image in [*latency_images, *throughput_images]:
        image.write_bytes(image.name.encode())
    pairs = root / "pairs.txt"
    pairs.write_text(
        "\t".join(map(str, latency_images)) + "\n" +
        "\t".join(map(str, latency_images)) + "\n"
    )
    synthetic = root / ".cache/bench/pairs/synthetic-8x8.txt"
    synthetic.parent.mkdir(parents=True)
    synthetic.write_text("\t".join(map(str, throughput_images)) + "\n")
    return latency_images, throughput_images, pairs, synthetic


def _write_benchmark_driver(root: Path, calls: Path) -> Path:
    """Write a stand-in driver that logs calls and emits valid scores.

    :param root: Isolated temporary benchmark repository.
    :param calls: Log that receives one line per invocation.
    :returns: Path of the generated Python driver.
    """
    driver = root / "driver.py"
    driver.write_text(
        "import csv, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "name = args[args.index('--name') + 1]\n"
        f"calls = pathlib.Path({str(calls)!r})\n"
        "if '--throughput-seconds' in args:\n"
        "    threads = int(args[args.index('--threads') + 1])\n"
        "    with calls.open('a') as f: f.write(f'{name}:{threads}\\n')\n"
        "    print(f'throughput metric=ssimu2 size=8x8 threads={threads} ' "
        "f'calls={threads * 10} seconds=1 fps={threads * 10} busy=1 ' "
        "f'checksum={threads * 10 * 42:.6f}')\n"
        "else:\n"
        "    with calls.open('a') as f: f.write(f'{name}:latency\\n')\n"
        "    out = args[args.index('--csv') + 1]\n"
        "    with open(out, 'w', newline='') as f:\n"
        "        writer = csv.writer(f)\n"
        "        writer.writerow(['original','distorted','width','height',"
        "'core_ms','score'])\n"
        "        pairs_file = pathlib.Path(args[args.index('--pairs-file') + "
        "1])\n"
        "        for index, pair in "
        "enumerate(pairs_file.read_text().splitlines()):\n"
        "            writer.writerow([*pair.split('\\t'),8,8,index + 1,42])\n"
        "    print('checksum 84')\n"
    )
    return driver


class BenchmarkFailureTests(unittest.TestCase):
    """Exercise benchmark failures and report integrity with offline drivers."""

    def test_round_rejects_stale_csv_when_driver_writes_nothing(self) -> None:
        """Reject an old CSV when the driver publishes no current result."""
        args = SimpleNamespace(
            pairs_file="pairs.txt", iterations=1, warmup=0, timeout=1.0
        )
        entry = {"name": "fake", "driver": "driver"}
        with tempfile.TemporaryDirectory() as directory:
            stale = Path(directory, "ssimu2-fake-round0.csv")
            stale.write_text(
                "original,distorted,width,height,core_ms,score\n"
                "a,b,8,8,1,42\n"
            )
            with mock.patch.object(inmemory.subprocess, "run",
                                   return_value=SimpleNamespace(returncode=0,
                                                                stdout="",
                                                                stderr="")):
                rows, error = measurements.run_round(
                    entry, "ssimu2", args, 0, directory
                )
            self.assertIsNone(rows)
            self.assertIn("no CSV from driver", error)
            self.assertFalse(stale.exists())
            self.assertEqual(
                sorted(p.name for p in Path(directory).iterdir()),
                ["ssimu2-fake-round0.log"]
            )

    def test_round_publishes_fresh_csv_and_logs_output(self) -> None:
        """Publish the fresh CSV and retain driver output."""
        args = SimpleNamespace(
            pairs_file="pairs.txt", iterations=1, warmup=0, timeout=1.0
        )
        entry = {"name": "fake", "driver": "driver"}
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory, "ssimu2-fake-round0.csv")
            final.write_text("stale")

            def write_csv(
                command: Sequence[str], **kwargs: object
            ) -> SimpleNamespace:
                path = Path(command[command.index("--csv") + 1])
                self.assertNotEqual(path, final)
                self.assertFalse(path.exists())
                path.write_text(
                    "original,distorted,width,height,core_ms,score\n"
                    "a,b,8,8,1.5,42\n"
                )
                return SimpleNamespace(returncode=0, stdout="done\n", stderr="")

            with mock.patch.object(inmemory.subprocess, "run",
                                   side_effect=write_csv):
                rows, error = measurements.run_round(
                    entry, "ssimu2", args, 0, directory
                )
            self.assertIsNone(error)
            self.assertEqual(rows[0]["score"], 42.0)
            self.assertEqual(rows[0]["core_ms"], 1.5)
            self.assertIn("a,b,8,8,1.5,42", final.read_text())
            self.assertIn(
                "done",
                Path(directory, "ssimu2-fake-round0.log").read_text()
            )
            self.assertEqual(len(list(Path(directory).iterdir())), 2)

    def test_logged_driver_failure_is_reported(self) -> None:
        """Include the failed driver exit in the benchmark diagnostic."""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory, "driver.log")
            with mock.patch.object(inmemory.subprocess, "run",
                                   return_value=SimpleNamespace(
                                       returncode=7, stdout="out\n",
                                       stderr="bad\n")):
                result, error = measurements.run_logged(["driver"], log, 1.0)
            self.assertIsNone(result)
            self.assertIn("exit 7", error)
            self.assertIn("out\nbad\n", log.read_text())

    def test_throughput_timeout_reports_error(self) -> None:
        """Report a timed-out throughput driver."""
        args = SimpleNamespace(throughput_seconds=1.0, timeout=0.1)
        with tempfile.TemporaryDirectory() as directory:
            with (mock.patch.object(inmemory.os.path, "isfile",
                                    return_value=True),
                  mock.patch.object(measurements, "read_pairs", return_value=[
                      ("a", "b")
                  ]), mock.patch.object(inmemory.subprocess, "run",
                                        side_effect=subprocess.TimeoutExpired(
                                            "driver", 0.1))):
                result = measurements.run_throughput(
                    {
                        "name": "fake",
                        "driver": "driver"
                    }, "ssimulacra2", "4k", 1, args, 0, directory
                )
            self.assertEqual(result, (None, "timeout after 0.1s"))
            self.assertIn(
                "timeout after 0.1s",
                Path(directory,
                     "throughput-ssimulacra2-fake-4k-1-0.log").read_text()
            )

    def test_stage_failure_removes_private_pair_file(self) -> None:
        """Clean the temporary pair list after a stage driver failure."""
        paths = []

        def fail(command: Sequence[str], **_kwargs: object) -> SimpleNamespace:
            path = Path(command[command.index("--pairs-file") + 1])
            paths.append(path)
            self.assertEqual(path.read_text(), "original\tdistorted\n")
            return SimpleNamespace(returncode=42)

        with (tempfile.TemporaryDirectory() as
              directory, mock.patch.object(bench.subprocess, "run",
                                           side_effect=fail)):
            output = os.path.join(directory, "out.csv")
            for _ in range(2):
                self.assertEqual(
                    bench.profile_stages(
                        "bench", [("original", "distorted")], 1, output
                    ), "stage profiler exited 42"
                )
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(all(not path.exists() for path in paths))

    def test_stage_success_without_csv_does_not_reuse_stale_file(self) -> None:
        """Reject a successful stage driver that omits its output file."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "stages.csv")
            output.write_text("stale")
            with mock.patch.object(bench.subprocess, "run",
                                   return_value=SimpleNamespace(returncode=0)):
                error = bench.profile_stages(
                    "bench", [("original", "distorted")], 1, str(output)
                )
            self.assertEqual(error, "stage profiler produced no CSV")
            self.assertFalse(output.exists())

    def test_round_validation_rejects_missing_reordered_and_invalid_rows(
        self
    ) -> None:
        """Reject incomplete, reordered and malformed benchmark results."""
        first = {
            "original": "a",
            "distorted": "b",
            "width": 8,
            "height": 8,
            "core_ms": 1.0,
            "score": 42.0
        }
        second = {**first, "distorted": "c"}
        expected = [("a", "b"), ("a", "c")]
        self.assertIsNone(
            measurements.validate_round([first, second], "ssimu2", expected)
        )
        self.assertIn(
            "do not match",
            measurements.validate_round([first], "ssimu2", expected)
        )
        self.assertIn(
            "do not match",
            measurements.validate_round([second, first], "ssimu2", expected)
        )
        self.assertIn(
            "core time",
            measurements.validate_round(
                [{
                    **first, "core_ms": float("nan")
                }], "ssimu2", None
            )
        )
        self.assertIn(
            "invalid score",
            measurements.validate_round(
                [{
                    **first, "score": float("nan")
                }], "ssimu2", None
            )
        )
        self.assertIsNone(
            measurements.validate_round(
                [{
                    **first, "score": float("-inf")
                }], "ssimu2", None
            )
        )
        self.assertIn(
            "invalid score",
            measurements.validate_round(
                [{
                    **first, "score": -1.0
                }], "butteraugli", None
            )
        )
        self.assertIn(
            "changed the score",
            measurements.validate_repeated_round(
                [{
                    **first, "score": 41.0
                }], [first]
            )
        )

    def test_duplicate_pairs_keep_their_weight(self) -> None:
        """Preserve repeated pairs when computing the aggregate."""
        first = {
            "original": "a",
            "distorted": "b",
            "width": 8,
            "height": 8,
            "core_ms": 1.0,
            "score": 42.0
        }
        second = {**first, "core_ms": 3.0}
        row = reports.aggregate("ssimu2", "ours", [[first, second]], None)[0]
        self.assertEqual(row["pairs"], 2)
        self.assertEqual(row["median_ms"], 2.0)

    def test_nonfinite_score_is_encoded_as_valid_json(self) -> None:
        """Serialize infinite scores as strings in valid JSON."""
        payload = reports.json_safe(
            {
                "score": float("-inf"),
                "values": [float("inf")]
            }
        )
        self.assertEqual(
            json.loads(json.dumps(payload, allow_nan=False)), {
                "score": "-inf",
                "values": ["inf"]
            }
        )

    def test_throughput_line_checks_size_metric_threads_and_finite_times(
        self
    ) -> None:
        """Reject throughput with inconsistent counts, timing or metadata."""
        line = (
            "throughput metric=ssimulacra2 size=8x8 threads=2 calls=10 "
            "seconds=1 fps=10 busy=1 checksum=20"
        )
        fields, error = measurements.parse_throughput_line(
            line, "8x8", "ssimu2", 2
        )
        self.assertIsNone(error)
        self.assertEqual(fields["calls"], 10)
        self.assertIn(
            "metric",
            measurements.parse_throughput_line(line, "8x8", "butteraugli", 2)[1]
        )
        self.assertIn(
            "threads",
            measurements.parse_throughput_line(line, "8x8", "ssimu2", 3)[1]
        )
        self.assertIn(
            "non-finite",
            measurements.parse_throughput_line(
                line.replace("fps=10", "fps=nan"), "8x8", "ssimu2", 2
            )[1]
        )
        self.assertIn(
            "non-positive",
            measurements.parse_throughput_line(
                line.replace("calls=10", "calls=0"), "8x8", "ssimu2", 2
            )[1]
        )
        self.assertIn(
            "calls / seconds",
            measurements.parse_throughput_line(
                line.replace("fps=10", "fps=12"), "8x8", "ssimu2", 2
            )[1]
        )
        self.assertIn(
            "expected one",
            measurements.parse_throughput_line(
                line + "\n" + line, "8x8", "ssimu2", 2
            )[1]
        )

    def test_throughput_checksum_accounts_for_print_rounding_and_accumulation(
        self
    ) -> None:
        """Allow checksum rounding while rejecting a changed per-call score."""
        first = {"calls": 1, "checksum": 1.234568}
        self.assertIsNone(
            measurements.validate_throughput_checksum(
                {
                    "calls": 3,
                    "checksum": 3.703703
                }, first
            )
        )
        self.assertIn(
            "checksum / calls changed",
            measurements.validate_throughput_checksum(
                {
                    "calls": 3,
                    "checksum": 3.703730
                }, first
            )
        )
        self.assertIsNone(
            measurements.validate_throughput_checksum(
                {
                    "calls": 1000000,
                    "checksum": 42000000.0001
                }, {
                    "calls": 10,
                    "checksum": 420.0
                }
            )
        )
        self.assertIsNone(
            measurements.validate_throughput_checksum(
                {
                    "calls": 10,
                    "checksum": float("-inf")
                }, {
                    "calls": 1,
                    "checksum": float("-inf")
                }
            )
        )
        self.assertIn(
            "checksum / calls changed",
            measurements.validate_throughput_checksum(
                {
                    "calls": 10,
                    "checksum": float("-inf")
                }, first
            )
        )

    def test_throughput_rejects_multi_pair_synthetic_workload(self) -> None:
        """Reject more than one synthetic throughput pair."""
        args = SimpleNamespace(throughput_seconds=1.0, timeout=1.0)
        with tempfile.TemporaryDirectory() as directory:
            pairs = Path(directory, "synthetic-8x8.txt")
            pairs.write_text("a\tb\nc\td\n")
            with (mock.patch.object(measurements, "throughput_pairs_path",
                                    return_value=str(pairs)),
                  mock.patch.object(measurements, "run_logged") as run):
                _, error = measurements.run_throughput(
                    {
                        "name": "fake",
                        "driver": "driver"
                    }, "ssimu2", "8x8", 2, args, 0, directory
                )
            self.assertIn("must contain one pair", error)
            run.assert_not_called()

    def test_throughput_arms_interleave_and_keep_each_measurement(self) -> None:
        """Interleave benchmark arms and retain their per-run measurements."""
        args = SimpleNamespace(sizes="8x8", throughput_runs=2, out_dir="out")
        manifest = {
            "ssimu2": [
                {
                    "name": "a",
                    "driver": "a",
                    "throughput_threads": 3
                }, {
                    "name": "b",
                    "driver": "b",
                    "throughput_threads": 2
                }
            ]
        }
        calls = []

        def run(
            entry: dict[str, str | int], metric: str, size: str, threads: int,
            _args: SimpleNamespace, repeat: int, _out: str
        ) -> tuple[dict[str, int | float], None]:
            calls.append((entry["name"], threads, repeat))
            # Implementations may have different scores; only each arm's
            # own per-call checksum must remain stable across worker counts.
            score = 2.0 if entry["name"] == "a" else 3.0
            return {
                "fps": 10.0 * threads,
                "calls": 10 * threads,
                "busy": 1.0,
                "checksum": score * 10 * threads,
                "seconds": 1.0
            }, None

        with (mock.patch.object(inmemory.os.path, "isfile", return_value=True),
              mock.patch.object(measurements, "run_throughput",
                                side_effect=run),
              contextlib.redirect_stdout(io.StringIO())):
            failures = []
            rows, _ = inmemory.run_throughput_mode(
                args, ["ssimu2"], manifest, None, failures
            )
        self.assertEqual(
            calls, [
                ("a", 1, 0), ("a", 3, 0), ("b", 1, 0), ("b", 2, 0), ("a", 1, 1),
                ("a", 3, 1), ("b", 1, 1), ("b", 2, 1)
            ]
        )
        self.assertEqual(failures, [])
        self.assertEqual(len(rows[0]["sizes"]["8x8"]["measurements_one"]), 2)
        self.assertEqual(
            rows[0]["sizes"]["8x8"]["measurements_one"][0]["checksum"], 20.0
        )

    def test_throughput_rejects_checksum_changes_in_workers_and_repeats(
        self
    ) -> None:
        """Reject checksum changes across worker counts and repeated runs."""
        args = SimpleNamespace(sizes="8x8", throughput_runs=2, out_dir="out")
        manifest = {
            "ssimu2": [{
                "name": "a",
                "driver": "a",
                "throughput_threads": 3
            }]
        }
        for changed in ((3, 0), (1, 1)):
            with self.subTest(changed=changed):
                self._check_throughput_checksum_change(changed, args, manifest)

    def _check_throughput_checksum_change(
        self, changed: tuple[int, int], args: SimpleNamespace,
        manifest: dict[str, list[dict[str, str | int]]]
    ) -> None:
        """Reject a changed checksum in one throughput measurement.

        :param changed: Worker count and round where the score changes.
        :param args: Throughput settings for the fake driver.
        :param manifest: Selected metric and arm configurations.
        """

        def run(
            _entry: dict[str, str | int], _metric: str, _size: str,
            threads: int, _args: SimpleNamespace, repeat: int, _out: str
        ) -> tuple[dict[str, int | float], None]:
            score = 43.0 if (threads, repeat) == changed else 42.0
            return {
                "fps": 10.0 * threads,
                "calls": 10 * threads,
                "busy": 1.0,
                "seconds": 1.0,
                "checksum": score * 10 * threads
            }, None

        with (mock.patch.object(inmemory.os.path, "isfile", return_value=True),
              mock.patch.object(measurements, "run_throughput",
                                side_effect=run),
              contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(io.StringIO())):
            failures = []
            rows, _ = inmemory.run_throughput_mode(
                args, ["ssimu2"], manifest, None, failures
            )
        self.assertEqual(failures, ["ssimu2/a/8x8"])
        self.assertIn(
            "checksum / calls changed", rows[0]["sizes"]["8x8"]["error"]
        )

    def test_cli_benchmark_alternates_arms_and_detects_warmup_failures(
        self
    ) -> None:
        """Alternate CLI arms and fail on bad warmup results."""
        calls = []
        clock = itertools.count()

        def run(command: Sequence[str], **_kwargs: object) -> SimpleNamespace:
            calls.append(command[0])
            return SimpleNamespace(returncode=0, stdout="42\n")

        with (mock.patch.object(bench, "run_score", return_value=(42.0, None)),
              mock.patch.object(bench.subprocess, "run", side_effect=run),
              mock.patch.object(bench.time, "perf_counter",
                                side_effect=lambda: next(clock))):
            rows = bench.measure_pair(
                ("a", "b"), [("x", "x", []), ("y", "y", [])], 2, 1
            )
        self.assertEqual(calls, ["x", "y", "x", "y", "x", "y"])
        self.assertTrue(all("error" not in row for row in rows))

        with (mock.patch.object(bench, "run_score", return_value=(42.0, None)),
              mock.patch.object(bench.subprocess, "run",
                                return_value=SimpleNamespace(returncode=7))):
            rows = bench.measure_pair(("a", "b"), [("x", "x", [])], 1, 1)
        self.assertEqual(rows[0]["error"], "exit 7")

        with (mock.patch.object(bench, "run_score", return_value=(42.0, None)),
              mock.patch.object(bench.subprocess, "run",
                                return_value=SimpleNamespace(returncode=0,
                                                             stdout="43\n"))):
            rows = bench.measure_pair(("a", "b"), [("x", "x", [])], 1, 1)
        self.assertIn("score changed", rows[0]["error"])

    def test_pair_readers_reject_malformed_lines(self) -> None:
        """Reject malformed pair-list rows in both benchmark readers."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "pairs.txt")
            path.write_text("# comment\na\tb\na\tb\n")
            self.assertEqual(
                measurements.read_pairs(path), [("a", "b"), ("a", "b")]
            )
            self.assertEqual(bench.read_pairs(path), [("a", "b"), ("a", "b")])
            path.write_text("broken\n")
            with self.assertRaisesRegex(ValueError, "expected original"):
                measurements.read_pairs(path)
            with self.assertRaisesRegex(ValueError, "expected original"):
                bench.read_pairs(path)

    def test_inmemory_runner_end_to_end_with_fake_drivers(self) -> None:
        """Exercise the in-memory workflow with executable stand-ins."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (latency_images, throughput_images, pairs,
             synthetic) = _write_pair_lists(root)
            calls = root / "calls.txt"
            driver = _write_benchmark_driver(root, calls)
            manifest = {
                "ssimu2": [
                    {
                        "name": "a",
                        "driver": sys.executable,
                        "args": [str(driver), "--name", "a"],
                        "throughput_threads": 3
                    }, {
                        "name": "b",
                        "driver": sys.executable,
                        "args": [str(driver), "--name", "b"],
                        "throughput_threads": 2
                    }
                ]
            }
            args = SimpleNamespace(
                pairs_file=str(pairs), metric="ssimu2", mode="all", only=None,
                rounds=2, iterations=2, warmup=1, throughput_seconds=1.0,
                throughput_runs=2, sizes="8x8", out_dir=str(root / "report"),
                timeout=3.0, list=False
            )
            with (mock.patch.object(probes, "REPO_ROOT", str(root)),
                  mock.patch.object(measurements, "REPO_ROOT",
                                    str(root)), mock.patch.object(inmemory,
                                                                  "REPO_ROOT",
                                                                  str(root)),
                  mock.patch.object(inmemory, "parse_args", return_value=args),
                  mock.patch.object(inmemory, "load_manifest",
                                    return_value=manifest),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(inmemory.main(), 0)
            self.assertEqual(
                calls.read_text().splitlines(), [
                    "a:latency", "b:latency", "a:latency", "b:latency", "a:1",
                    "a:3", "b:1", "b:2", "a:1", "a:3", "b:1", "b:2"
                ]
            )
            latency = json.loads((root / "report/summary.json").read_text())
            self.assertEqual(latency["rows"][0]["pairs"], 2)
            self.assertEqual(latency["rows"][0]["median_ms"], 1.5)
            self.assertEqual(latency["rows"][0]["round_min_ms"], 1.5)
            self.assertEqual(len(latency["raw_rounds"]["ssimu2/a"]), 2)
            self.assertEqual(latency["schema_version"], 2)
            self.assertEqual(latency["protocol"]["iterations"], 2)
            self.assertEqual(latency["protocol"]["rounds"], 2)
            self.assertEqual(len(latency["protocol"]["pairs_file_sha256"]), 64)
            self.assertEqual(
                set(latency["environment"]["inputs"]),
                set(map(str, latency_images))
            )
            throughput = self._check_throughput_report(
                root, throughput_images, synthetic
            )
            # Throughput-only runs must also fingerprint their actual inputs,
            # without requiring a latency pair list to exist.
            args.mode = "throughput"
            args.pairs_file = str(root / "missing-latency.txt")
            args.out_dir = str(root / "throughput-only")
            with (mock.patch.object(probes, "REPO_ROOT", str(root)),
                  mock.patch.object(measurements, "REPO_ROOT",
                                    str(root)), mock.patch.object(inmemory,
                                                                  "REPO_ROOT",
                                                                  str(root)),
                  mock.patch.object(inmemory, "parse_args", return_value=args),
                  mock.patch.object(inmemory, "load_manifest",
                                    return_value=manifest),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(inmemory.main(), 0)
            only = json.loads(
                (root / "throughput-only/throughput.json").read_text()
            )
            self.assertEqual(
                only["environment"]["inputs"],
                throughput["environment"]["inputs"]
            )
            self.assertEqual(
                only["environment"]["pair_lists"],
                throughput["environment"]["pair_lists"]
            )

    def _check_throughput_report(
        self, root: Path, throughput_images: list[Path], synthetic: Path
    ) -> dict[str, object]:
        """Check the throughput workload provenance and report wording.

        :param root: Isolated temporary benchmark repository.
        :param throughput_images: Input paths for the throughput arm.
        :param synthetic: Pair list selected by the throughput arm.
        :returns: The validated throughput report.
        """
        throughput = json.loads((root / "report/throughput.json").read_text())
        self.assertEqual(
            set(throughput["environment"]["inputs"]),
            set(map(str, throughput_images))
        )
        self.assertNotIn("pairs_file_sha256", throughput["protocol"])
        selected = throughput["environment"]["pair_lists"]["8x8"]
        self.assertEqual(selected["path"], str(synthetic))
        self.assertEqual(selected["sha256"], probes.file_sha256(synthetic))
        for image in throughput_images:
            self.assertEqual(
                throughput["environment"]["inputs"][str(image)]["sha256"],
                probes.file_sha256(image)
            )
        self.assertEqual(
            len(throughput["rows"][0]["sizes"]["8x8"]["measurements_one"]), 2
        )
        report = (root / "report/throughput.md").read_text()
        self.assertIn("not the fixed-pair-count", report)
        self.assertIn("not a ceiling on aggregate throughput", report)
        return throughput

    def test_cli_comparison_end_to_end_with_fake_binaries(self) -> None:
        """Exercise CLI comparison with deterministic fake binaries."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = root / "calls.txt"
            binaries = []
            for name in ("ours", "peer"):
                path = root / name
                path.write_text(
                    "#!/usr/bin/env python3\n"
                    "from pathlib import Path\n"
                    f"with Path({str(calls)!r}).open('a') as f: "
                    f"f.write({name!r} + '\\n')\n"
                    "print('42')\n"
                )
                path.chmod(0o755)
                binaries.append(path)
            pairs = root / "pairs.txt"
            pairs.write_text("a\tb\na\tb\n")
            out = root / "artifacts/reports/bench/results.csv"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "bench/run_bench.py"), "--ours",
                    str(binaries[0]), "--baseline", f"peer={binaries[1]}",
                    "--pairs-file",
                    str(pairs), "--reps", "2", "--warmup", "1"
                ], cwd=root, capture_output=True, text=True, timeout=10
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                calls.read_text().splitlines(), ["ours", "peer"] * 8
            )
            self.assertEqual(len(out.read_text().splitlines()), 5)
            self.assertIn("jobs: 1", result.stdout)


class FetchCid22ArgumentTests(unittest.TestCase):
    """Exercise argument handling for the CID22 downloader."""

    def test_help_and_unknown_arguments_do_not_start_work(self) -> None:
        """Return help and option errors without touching the archive."""
        for option, status in (("--help", 0), ("--unknown", 2)):
            with self.subTest(option=option):
                self._check_fetch_argument(option, status)

    def _check_fetch_argument(self, option: str, status: int) -> None:
        """Check a downloader exit that must precede archive access.

        :param option: Command-line option under test.
        :param status: Expected process exit status.
        """
        with (mock.patch.object(sys, "argv", ["fetch_cid22.py", option]),
              mock.patch.object(fetch_cid22, "ensure_archive") as
              ensure_archive, contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(io.StringIO()),
              self.assertRaises(SystemExit) as exit_code,
              ):
            fetch_cid22.main()
        self.assertEqual(exit_code.exception.code, status)
        ensure_archive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
