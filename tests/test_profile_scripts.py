#!/usr/bin/env python3
"""CPU-only checks for profiling wrapper argument and fallback handling."""

import os
import shutil
import subprocess
import tempfile
import unittest
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class ProfileScriptTests(unittest.TestCase):
    """Exercise profiling wrappers with local stub commands."""

    def make_command(self, directory: Path, name: str, body: str) -> Path:
        """Write an executable shell command for a profiling fixture.

        :param directory: Directory for the temporary command.
        :param name: Name identifying the requested fixture or output row.
        :param body: Shell script contents.
        :returns: Executable fixture path.
        """
        path = Path(directory, name)
        path.write_text("#!/bin/bash\nset -eu\n" + body + "\n")
        path.chmod(0o755)
        return path

    def make_environment(self, root: Path) -> tuple[Path, Path]:
        """Set up isolated profiler and benchmark command directories.

        :param root: Temporary workspace.
        :returns: Stub command and benchmark directories.
        """
        bin_dir = root / "bin"
        build = root / "custom build"
        bin_dir.mkdir()
        build.mkdir()
        for name in ("basename", "find", "mkdir", "mktemp", "rm", "sed", "sort"):
            (bin_dir / name).symlink_to(shutil.which(name))
        self.make_command(build, "vqm_bench", 'printf "STAGES %s\\n" "$*"')
        return bin_dir, build

    def run_profile(
        self, root: Path, bin_dir: Path, build: Path, tool: str, *extra: str,
        pair: tuple[str, str] | None = ("original.png", "distorted.png")
    ) -> subprocess.CompletedProcess[str]:
        """Invoke the profiler wrapper in the isolated fixture environment.

        :param root: Temporary workspace.
        :param bin_dir: Path of stub profiler executables.
        :param build: Path of the fixture benchmark directory.
        :param tool: Requested profiler mode.
        :param extra: Additional profiler flags and values.
        :param pair: Explicit inputs, or None to exercise fixture selection.
        :returns: Captured profiler invocation.
        :raises subprocess.CalledProcessError: If the profiler wrapper fails.
        """
        pair_options = ("--pair", *pair) if pair else ()
        return subprocess.run(
            [
                "/bin/bash",
                str(ROOT / "bench/profile.sh"), "--build-dir",
                str(build), "--tool", tool, "--iterations", "7", *pair_options, *extra
            ], cwd=root, env={
                **os.environ, "PATH": str(bin_dir)
            }, capture_output=True, text=True, check=True
        )

    def test_auto_falls_back_when_perf_sampling_is_denied(self) -> None:
        """Use built-in stages if perf cannot open a sampling event."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir, build = self.make_environment(root)
            self.make_command(bin_dir, "perf", "exit 1")
            result = self.run_profile(root, bin_dir, build, "auto")
            self.assertIn("built-in stage timing", result.stdout)
            self.assertIn(
                "STAGES --iterations 7 --pair original.png distorted.png", result.stdout
            )
            self.assertFalse((root / "artifacts").exists())
            self.assertFalse((build / "bench").exists())

    def test_partial_cid22_falls_back_to_generated_fixtures(self) -> None:
        """Use generated fixtures until the selected CID22 pair is complete."""
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        bin_dir, build = self.make_environment(root)
        corpus = Path("validation/data/CID22_validation_set")
        original = corpus / "original/source.png"
        distorted = corpus / "compressed/source/variant.jpg"
        (root / original).parent.mkdir(parents=True)
        (root / original).touch()
        (root / corpus / "compressed").mkdir()
        fixtures = build / "fixtures"
        fixtures.mkdir()
        first, second = fixtures / "gradient.png", fixtures / "gradient_B.png"
        first.touch()
        second.touch()

        fallback = self.run_profile(root, bin_dir, build, "stages", pair=None)
        self.assertIn(f"--pair {first} {second}", fallback.stdout)
        (root / distorted).parent.mkdir()
        (root / distorted).touch()
        complete = self.run_profile(root, bin_dir, build, "stages", pair=None)
        self.assertIn(f"--pair {original} {distorted}", complete.stdout)

    def test_profiler_outputs_have_independent_default_and_override(self) -> None:
        """Put profiler output in the requested or documented default directory."""
        fake_perf = r"""printf "PERF %s\n" "$*"
if [ "$1" = record ]; then
  shift
  while [ "$1" != -- ]; do
    if [ "$1" = -o ]; then
      printf 'fake perf data\n' > "$2"
      shift 2
    else
      shift
    fi
  done
fi"""
        fake_valgrind = r"""printf "CALLGRIND %s\n" "$*"
for argument in "$@"; do
  case "$argument" in
    --callgrind-out-file=*)
      printf 'fake callgrind data\n' > "${argument#--callgrind-out-file=}"
      ;;
  esac
done"""
        tools = (("perf", "perf.data"), ("callgrind", "callgrind.out"))
        for (tool, filename), override in product(tools, (False, True)):
            with self.subTest(tool=tool, override=override):
                self._check_profile_output(
                    tool, filename, override, fake_perf, fake_valgrind
                )

    def _check_profile_output(
        self, tool: str, filename: str, override: bool, fake_perf: str,
        fake_valgrind: str
    ) -> None:
        """Check one profiler's output path and report command.

        :param tool: Selected profiling wrapper.
        :param filename: Expected profile data filename.
        :param override: Use the explicit output directory.
        :param fake_perf: Stub perf command body.
        :param fake_valgrind: Stub Valgrind command body.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir, build = self.make_environment(root)
            self.make_command(bin_dir, "perf", fake_perf)
            self.make_command(bin_dir, "valgrind", fake_valgrind)
            self.make_command(
                bin_dir, "callgrind_annotate", 'printf "ANNOTATE %s\\n" "$*"'
            )
            output = Path(
                "custom profiles" if override else "artifacts/reports/profile"
            )
            extra = ("--out-dir", str(output)) if override else ()
            result = self.run_profile(root, bin_dir, build, tool, *extra)
            self.assertTrue((root / output / filename).is_file())
            self.assertIn(
                "--iterations 7 --pair original.png distorted.png", result.stdout
            )
            self.assertIn(str(build / "vqm_bench"), result.stdout)
            report = (
                "PERF report --stdio --percent-limit 1 -i"
                if tool == "perf" else "ANNOTATE --threshold=95"
            )
            self.assertIn(f"{report} {output / filename}", result.stdout)
            self.assertFalse((build / "bench").exists())
            if override:
                self.assertFalse((root / "artifacts").exists())

    def test_nsys_rejects_unsupported_metric_mode_before_capture(self) -> None:
        """Reject unsupported metric and mode combinations before capturing."""
        result = subprocess.run(
            [
                "/bin/bash",
                str(ROOT / "bench/profile_nsys.sh"), "--metric", "butteraugli",
                "--mode", "half-planes"
            ], cwd=ROOT, capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("available only for ssimu2", result.stderr)

    def test_nsys_capture_matrix_includes_butteraugli_workers(self) -> None:
        """List the Butteraugli multiworker capture in the Nsight matrix."""
        script = str(ROOT / "bench/profile_nsys.sh")
        result = subprocess.run(
            ["/bin/bash", script, "--metric", "all", "--mode", "all", "--list"],
            cwd=ROOT, capture_output=True, text=True, check=True
        )
        self.assertEqual(
            result.stdout.splitlines(), [
                "ssimu2 resident", "ssimu2 streaming", "ssimu2 half-planes",
                "ssimu2 workers", "butteraugli resident", "butteraugli streaming",
                "butteraugli workers"
            ]
        )
        result = subprocess.run(
            [
                "/bin/bash", script, "--metric", "butteraugli", "--mode", "workers",
                "--list"
            ], cwd=ROOT, capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "butteraugli workers")

    def test_nsys_rejects_invalid_iterations(self) -> None:
        """Reject a nonpositive Nsight capture iteration count."""
        result = subprocess.run(
            [
                "/bin/bash",
                str(ROOT / "bench/profile_nsys.sh"), "--iterations", "0", "--list"
            ], cwd=ROOT, capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("positive integer", result.stderr)


if __name__ == "__main__":
    unittest.main()
