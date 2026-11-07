# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Keep selected builds consistent across commands, manifests and provenance."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench import inmemory_environment
from bench import run_inmemory
from bench import video
from tools.build_paths import ROOT
from tools.build_paths import relocate_build_path
from validation import run_corpus


class SelectedBuildTests(unittest.TestCase):
    """Exercise build selection at the tools' public boundaries."""

    def test_manifest_commands_select_candidate_and_reference_builds(
        self
    ) -> None:
        """Relocate default executables and preserve explicit commands."""
        with tempfile.TemporaryDirectory(prefix="vqm build ") as directory:
            selected = Path(directory)
            drivers = run_inmemory.load_manifest(directory)
            corpus = run_corpus.load_manifest(directory)
            paths = [
                entry["driver"] for entries in drivers.values()
                for entry in entries
            ]
            paths += [
                entry["command"][0] for spec in corpus.values()
                for entry in spec["implementations"]
            ]
            self.assertTrue(
                all(Path(path).is_relative_to(selected) for path in paths)
            )
            self.assertEqual(
                relocate_build_path("/explicit/vqm", directory), "/explicit/vqm"
            )
            result = subprocess.run(
                [
                    sys.executable, "bench/run_inmemory.py", "--list",
                    "--build-dir", directory
                ], cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(selected / "vqm_bench"), result.stdout)

    def test_video_setup_and_measurement_use_the_same_build(self) -> None:
        """Build the binaries used by the public clip workflow."""
        args = video.arguments(["--build-dir", "build-selected"])
        commands = video.setup_commands(args)
        selected = str(ROOT / "build-selected")
        self.assertIn(selected, commands[1])
        self.assertIn(selected, commands[2])
        self.assertIn(selected, commands[3])
        scoring = video.measurement_command(args, "cvvdp", args.out_dir)
        self.assertEqual(
            scoring[scoring.index("--binary") + 1], str(args.build_dir / "vqm")
        )

    def test_provenance_uses_selected_cache_fingerprint_and_runtime_library(
        self
    ) -> None:
        """Fingerprint the selected build's binaries."""
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory)
            stamps = selected / "reference-fingerprints"
            stamps.mkdir()
            (stamps / "vship-vulkan.json").write_text(
                json.dumps({"revision": "selected"})
            )
            (selected / "CMakeCache.txt"
             ).write_text("CMAKE_CXX_COMPILER:FILEPATH=/selected/cxx\n")
            entry = {
                "name": "vship-vulkan",
                "driver": str(selected / "vship_bench")
            }
            with mock.patch.object(inmemory_environment, "command_output",
                                   return_value=None):
                result = inmemory_environment.capture_environment(
                    [], [entry], {}, selected
                )
            reference = result["references"]["vship-vulkan"]
            self.assertEqual(result["compiler"]["path"], "/selected/cxx")
            self.assertEqual(
                reference["build_fingerprint"]["revision"], "selected"
            )
            self.assertEqual(
                reference["runtime_library"]["path"],
                str(selected / "vship-build-vulkan/libvship.so")
            )


if __name__ == "__main__":
    unittest.main()
