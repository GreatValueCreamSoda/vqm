#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""CPU-only checks for selecting reference checkouts and the CID22 corpus."""

import contextlib
import importlib.util
import io
import sys
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "fetch_references", ROOT / "tools/fetch_references.py"
)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class ReferenceSelectionTests(unittest.TestCase):
    """Exercise reference and corpus selection without network access."""

    def run_setup(
        self, arguments: Sequence[str], failure: SystemExit | None = None
    ) -> tuple[int, Sequence[tuple[object, ...]], list[list[str]]]:
        """Run reference setup with mock Git and corpus operations.

        :param arguments: Reference setup command-line arguments.
        :param failure: Optional simulated corpus command failure.
        :returns: Status, checkout calls and corpus commands.
        """
        rows = [("Vship", "https://example.invalid/vship", "commit")]
        calls = []
        with mock.patch.object(sys, "argv", ["fetch_references.py", *arguments]), \
             mock.patch.object(fetch, "parse_lock", return_value=rows), \
             mock.patch.object(fetch, "check_state") as check_state, \
             mock.patch.object(fetch, "run", side_effect=failure or calls.append), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            result = fetch.main()
        return result, check_state.call_args_list, calls

    def test_default_includes_reference_and_corpus(self) -> None:
        """Check that the default selects both the references and CID22."""
        status, checkouts, calls = self.run_setup([])
        self.assertEqual(status, 0)
        self.assertEqual(len(checkouts), 1)
        self.assertEqual(checkouts[0].args[0], "Vship")
        self.assertEqual(
            calls,
            [[sys.executable, str(ROOT / "validation/fetch_cid22.py")]]
        )

    def test_selected_reference_does_not_fetch_corpus(self) -> None:
        """Fetch only a requested checkout."""
        status, checkouts, calls = self.run_setup(["--only", "Vship"])
        self.assertEqual(status, 0)
        self.assertEqual(len(checkouts), 1)
        self.assertEqual(calls, [])

    def test_corpus_only_fetch_does_not_clone_references(self) -> None:
        """Fetch CID22 without cloning Git references."""
        status, checkouts, calls = self.run_setup(["--only", "cid22"])
        self.assertEqual(status, 0)
        self.assertEqual(checkouts, [])
        self.assertEqual(
            calls,
            [[sys.executable, str(ROOT / "validation/fetch_cid22.py")]]
        )

    def test_default_check_checks_every_selected_input_offline(self) -> None:
        """Check both selected input kinds offline."""
        status, checkouts, calls = self.run_setup(["--check"])
        self.assertEqual(status, 0)
        self.assertEqual(len(checkouts), 1)
        self.assertTrue(checkouts[0].args[4])
        self.assertEqual(calls[0][-1], "--check")

    def test_corpus_only_check_is_offline(self) -> None:
        """Forward offline mode to a CID22-only check."""
        status, checkouts, calls = self.run_setup(["--only", "cid22", "--check"])
        self.assertEqual(status, 0)
        self.assertEqual(checkouts, [])
        self.assertEqual(
            calls,
            [[sys.executable,
              str(ROOT / "validation/fetch_cid22.py"), "--check"]]
        )

    def test_combined_check_and_force_preserve_selection(self) -> None:
        """Preserve explicit input selection with both check and force flags."""
        status, checkouts, calls = self.run_setup(
            ["--only", "cid22,Vship", "--check", "--force"]
        )
        self.assertEqual(status, 0)
        self.assertEqual(len(checkouts), 1)
        self.assertEqual(checkouts[0].args[4:6], (True, True))
        self.assertEqual(calls[0][-1], "--check")
        self.assertNotIn("--force", calls[0])

    def test_failed_corpus_fails_setup(self) -> None:
        """Return failure if the corpus command fails."""
        status, _, _ = self.run_setup(
            ["--only", "cid22", "--check"], failure=SystemExit(1)
        )
        self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
