#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Check isolated original runtime bootstrap and reference build selection."""

import argparse
import hashlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reference_build_tests", ROOT / "tools/build_references.py"
)
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class ReferenceBuildSelectionTests(unittest.TestCase):
    """Fail invalid bootstrap requests before touching sources or build caches."""

    def test_cvvdp_step_maps_to_the_pinned_original_checkout(self) -> None:
        """The CPU runtime is a first-class step with no peer substitute."""
        self.assertIn("cvvdp", BUILD.ALL_STEPS)
        self.assertEqual(BUILD.STEP_CHECKOUTS["cvvdp"], "ColorVideoVDP")

    def test_empty_selection_and_nonpositive_jobs_are_rejected(self) -> None:
        """An empty subset never expands into all reference builds."""
        for only, jobs in (("", 1), (", ,", 1), (None, 0), ("cvvdp", -1)):
            args = argparse.Namespace(only=only, jobs=jobs)
            with self.subTest(only=only, jobs=jobs), redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                BUILD._selected_steps(args, argparse.ArgumentParser())

    def test_known_subsets_are_deduplicated_and_unknown_steps_fail(self) -> None:
        """Only named known steps participate in the bootstrap transaction."""
        args = argparse.Namespace(only="cvvdp,libjxl,cvvdp", jobs=2)
        self.assertEqual(
            BUILD._selected_steps(args, argparse.ArgumentParser()), ["cvvdp", "libjxl"]
        )
        args.only = "wrong-original"
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            BUILD._selected_steps(args, argparse.ArgumentParser())

    def test_cpu_originals_do_not_require_a_core_build(self) -> None:
        """Cloudinary, libjxl, libvmaf and Python originals stand alone."""
        with mock.patch.object(BUILD, "run") as run:
            BUILD.preflight(2, ["cloudinary", "libjxl", "vmaf", "cvvdp"])
        run.assert_not_called()

    def test_core_linked_peers_restore_io_after_a_core_only_configure(self) -> None:
        """An existing no-CLI cache cannot hide the archives needed by peer drivers."""
        with mock.patch.object(BUILD, "run") as run:
            BUILD.preflight(2, ["vship-vulkan"])
        self.assertEqual(
            run.call_args_list[0].args[0],
            ["cmake", "--preset", "release", "-DVQM_BUILD_CLI=ON"]
        )
        self.assertIn("vqm_io", run.call_args_list[1].args[0])
        self.assertIn("vqm_core", run.call_args_list[1].args[0])


class CvvdpBootstrapTests(unittest.TestCase):
    """Keep Python installation isolated and pinned without modifying the original."""

    def test_new_runtime_uses_python312_and_the_locked_cpu_requirements(self) -> None:
        """Only the generated environment is populated, using the explicit CPU wheel index."""
        with mock.patch.object(BUILD, "require_reference") as require, \
                mock.patch.object(BUILD.shutil, "which", return_value="uv"), \
                mock.patch.object(BUILD.os.path, "isfile", return_value=False), \
                mock.patch.object(BUILD, "_validate_cvvdp_python"), \
                mock.patch.object(BUILD, "_runtime_matches", side_effect=[False, True]), \
                mock.patch.object(BUILD, "run") as run, \
                mock.patch.object(BUILD, "_probe_cvvdp") as probe, \
                redirect_stdout(io.StringIO()):
            BUILD.build_cvvdp(2, False)
        require.assert_called_once_with("ColorVideoVDP")
        create, install = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            create, ["uv", "venv", "--no-project", "--python", "3.12", BUILD.CVVDP_ENV]
        )
        self.assertEqual(install[:3], ["uv", "pip", "install"])
        self.assertEqual(install[-2:], ["-r", BUILD.CVVDP_REQUIREMENTS])
        self.assertIn("https://download.pytorch.org/whl/cpu", install)
        self.assertNotIn("--system", install)
        self.assertEqual(
            install[install.index("--python") + 1], probe.call_args.args[0]
        )
        self.assertIn(".venv/bin/python", probe.call_args.args[0])
        self.assertIn("--index", install)

    def test_existing_matching_runtime_is_probed_without_reinstallation(self) -> None:
        """A reusable environment still proves it can load and run the CPU original."""
        with mock.patch.object(BUILD, "require_reference"), \
                mock.patch.object(BUILD.shutil, "which", return_value="uv"), \
                mock.patch.object(BUILD.os.path, "isfile", return_value=True), \
                mock.patch.object(BUILD, "_validate_cvvdp_python"), \
                mock.patch.object(BUILD, "_runtime_matches", return_value=True), \
                mock.patch.object(BUILD, "run") as run, \
                mock.patch.object(BUILD, "_probe_cvvdp") as probe, \
                redirect_stdout(io.StringIO()):
            BUILD.build_cvvdp(2, False)
        run.assert_not_called()
        probe.assert_called_once()

    def test_missing_uv_is_actionable_and_preserves_existing_files(self) -> None:
        """A missing prerequisite fails before deleting or creating an environment."""
        with mock.patch.object(BUILD, "require_reference"), \
                mock.patch.object(BUILD.shutil, "which", return_value=None), \
                mock.patch.object(BUILD.shutil, "rmtree") as remove, \
                mock.patch.object(BUILD, "run") as run, \
                self.assertRaisesRegex(SystemExit, "uv is required"):
            BUILD.build_cvvdp(2, True)
        remove.assert_not_called()
        run.assert_not_called()

    def test_force_reapplies_dependencies_without_removing_the_environment(
        self
    ) -> None:
        """A forced reference refresh preserves unrelated project packages and files."""
        with mock.patch.object(BUILD, "require_reference"), \
                mock.patch.object(BUILD.shutil, "which", return_value="uv"), \
                mock.patch.object(BUILD.os.path, "isfile", return_value=True), \
                mock.patch.object(BUILD.shutil, "rmtree") as remove, \
                mock.patch.object(BUILD, "_validate_cvvdp_python"), \
                mock.patch.object(BUILD, "_runtime_matches", return_value=True), \
                mock.patch.object(BUILD, "run") as run, \
                mock.patch.object(BUILD, "_probe_cvvdp"), \
                redirect_stdout(io.StringIO()):
            BUILD.build_cvvdp(2, True)
        remove.assert_not_called()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][:3], ["uv", "pip", "install"])

    def test_probe_disables_original_bytecode_writes_and_runs_on_cpu(self) -> None:
        """The import/prediction probe retains machine-readable runtime evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "build").mkdir()
            completed = subprocess.CompletedProcess(
                [], 0, stdout=json.dumps({
                    "device": "cpu",
                    "identity_score": 10
                }), stderr=""
            )
            with mock.patch.object(BUILD, "REPO_ROOT", str(root)), \
                    mock.patch.object(BUILD.subprocess, "run", return_value=completed) as run:
                BUILD._probe_cvvdp("isolated-python")
            command = run.call_args.args[0]
            self.assertIn("-B", command)
            self.assertIn("sys.dont_write_bytecode = True", command[3])
            self.assertIn('torch.device("cpu")', command[3])
            self.assertIn("metric.predict", command[3])
            self.assertEqual(command[-1], str(root / ".reference/ColorVideoVDP"))
            self.assertEqual(
                json.loads((root / "build/cvvdp-reference-probe.json").read_text()), {
                    "device": "cpu",
                    "identity_score": 10
                }
            )

    def test_failed_cpu_probe_is_not_accepted_as_a_runtime(self) -> None:
        """Missing imports and model/prediction failures stop bootstrap."""
        completed = subprocess.CompletedProcess(
            [], 1, stdout="", stderr="missing torch"
        )
        with mock.patch.object(BUILD.subprocess, "run", return_value=completed), \
                self.assertRaisesRegex(SystemExit, "CPU original probe failed"):
            BUILD._probe_cvvdp("isolated-python")

    def test_runtime_validation_allows_extras_but_checks_every_locked_version(
        self
    ) -> None:
        """A working torch import cannot hide a changed NumPy or another locked dependency."""
        with tempfile.TemporaryDirectory() as temporary:
            requirements = Path(temporary) / "requirements.txt"
            requirements.write_text("torch==2.15.0+cpu\nnumpy==2.5.4\n")
            installed = {"torch": "2.15.0+cpu", "numpy": "2.5.4", "extra-tool": "1"}
            with mock.patch.object(BUILD, "CVVDP_REQUIREMENTS", str(requirements)), \
                    mock.patch.object(BUILD, "capture", side_effect=lambda _: json.dumps(installed)):
                self.assertTrue(BUILD._runtime_matches("python"))
                installed["numpy"] = "different"
                self.assertFalse(BUILD._runtime_matches("python"))
                del installed["numpy"]
                self.assertFalse(BUILD._runtime_matches("python"))

    def test_an_existing_wrong_python_is_reported_without_replacement(self) -> None:
        """A shared project environment is not discarded to obtain a different interpreter."""
        with mock.patch.object(BUILD, "capture", return_value="3.14"), \
                self.assertRaisesRegex(SystemExit, "files were preserved"):
            BUILD._validate_cvvdp_python(".venv/bin/python")

    def test_runtime_fingerprint_binds_interpreter_packages_and_requirements(
        self
    ) -> None:
        """Lock changes invalidate reuse even when the original source pin is stable."""
        with tempfile.TemporaryDirectory() as temporary:
            requirements = Path(temporary) / "requirements.txt"
            requirements.write_text("torch==2.15.0+cpu\n")
            with mock.patch.object(BUILD, "CVVDP_REQUIREMENTS", str(requirements)), \
                    mock.patch.object(BUILD, "capture", return_value="runtime-version"):
                first = BUILD._cvvdp_fingerprint()
                self.assertEqual(first["requested_python"], "3.12")
                self.assertEqual(first["python_version"], "runtime-version")
                self.assertEqual(first["installed_packages"], "runtime-version")
                self.assertEqual(
                    first["requirements_sha256"],
                    hashlib.sha256(requirements.read_bytes()).hexdigest()
                )
                requirements.write_text("torch==different\n")
                second = BUILD._cvvdp_fingerprint()
                self.assertNotEqual(
                    first["requirements_sha256"], second["requirements_sha256"]
                )

    def test_created_runtime_versions_are_fingerprinted_after_installation(
        self
    ) -> None:
        """The successful stamp describes the real interpreter rather than its absent predecessor."""
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(BUILD, "FINGERPRINT_DIR", temporary), \
                    mock.patch.object(BUILD, "reference_fingerprint", side_effect=[{"runtime": None}, {"runtime": "installed"}]), \
                    redirect_stdout(io.StringIO()):
                builder = mock.Mock()
                BUILD.build_fingerprinted("cvvdp", 2, False, builder)
            builder.assert_called_once_with(2, True)
            self.assertEqual(
                json.loads((Path(temporary) / "cvvdp.json").read_text()),
                {"runtime": "installed"}
            )


if __name__ == "__main__":
    unittest.main()
