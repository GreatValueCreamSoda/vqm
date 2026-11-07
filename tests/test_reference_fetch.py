#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""CPU-only checks for selecting reference checkouts and the CID22 corpus."""

import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
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
        with (mock.patch.object(sys, "argv",
                                ["fetch_references.py", *arguments]),
              mock.patch.object(fetch, "parse_lock", return_value=rows),
              mock.patch.object(fetch, "check_state") as
              check_state, mock.patch.object(fetch, "run", side_effect=failure
                                             or calls.append),
              contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(io.StringIO())):
            result = fetch.main()
        return result, check_state.call_args_list, calls

    def test_default_includes_reference_and_corpus(self) -> None:
        """Check that the default selects both the references and CID22."""
        status, checkouts, calls = self.run_setup([])
        self.assertEqual(status, 0)
        self.assertEqual(len(checkouts), 1)
        self.assertEqual(checkouts[0].args[0], "Vship")
        self.assertEqual(
            calls, [[sys.executable,
                     str(ROOT / "validation/fetch_cid22.py")]]
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
            calls, [[sys.executable,
                     str(ROOT / "validation/fetch_cid22.py")]]
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
        status, checkouts, calls = self.run_setup(
            ["--only", "cid22", "--check"]
        )
        self.assertEqual(status, 0)
        self.assertEqual(checkouts, [])
        self.assertEqual(
            calls, [
                [
                    sys.executable,
                    str(ROOT / "validation/fetch_cid22.py"), "--check"
                ]
            ]
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


class ReferenceCheckoutTests(unittest.TestCase):
    """Preserve authored changes while verifying pinned reference checkouts."""

    @staticmethod
    def make_git_repo(path: Path) -> None:
        """Create a temporary Git checkout with tracked and ignored files.

        :param path: Directory for the temporary Git repository.
        :raises subprocess.CalledProcessError: If the temporary checkout cannot
            be initialized.
        """
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(
            ["git", "-C", str(path), "config", "user.name", "Test"], check=True
        )
        subprocess.run(
            [
                "git", "-C",
                str(path), "config", "user.email", "test@example.com"
            ], check=True
        )
        (path / "tracked").write_text("original\n")
        (path / ".gitignore").write_text("ignored\n")
        subprocess.run(["git", "-C", str(path), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(path), "commit", "-qm", "initial"], check=True
        )

    def test_reference_detects_edits_before_force_or_check(self) -> None:
        """Reject dirty reference checkouts in both verification modes."""
        directory = self.enterContext(tempfile.TemporaryDirectory())
        repo = Path(directory)
        self.make_git_repo(repo)
        self.assertFalse(fetch.checkout_dirty(str(repo)))
        pin = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        status, mismatches = set(), []
        with contextlib.redirect_stdout(io.StringIO()):
            fetch.check_state(
                "reference", "unused", str(repo), pin, True, False, status,
                mismatches
            )
        self.assertFalse(status)
        with contextlib.redirect_stdout(io.StringIO()):
            fetch.check_state(
                "reference", "unused", str(repo), "different-pin", True, False,
                status, mismatches
            )
        self.assertEqual(status, {"failed"})
        for name in ("tracked", "untracked", "ignored"):
            path = repo / name
            path.write_text("local work\n")
            self.assertTrue(fetch.checkout_dirty(str(repo)), name)
            status, mismatches = set(), []
            with contextlib.redirect_stdout(io.StringIO()):
                fetch.check_state(
                    "reference", "unused", str(repo), "different-pin", False,
                    True, status, mismatches
                )
            self.assertEqual(status, {"failed"})
            self.assertEqual(mismatches, ["reference"])
            self.assertEqual(path.read_text(), "local work\n")
            if name == "tracked":
                path.write_text("original\n")
            else:
                path.unlink()

    def test_force_checks_out_pin_only_when_checkout_is_clean(self) -> None:
        """Restore the pinned revision only for a clean reference checkout."""
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            self.make_git_repo(repo)
            pin = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
            (repo / "tracked").write_text("second commit\n")
            subprocess.run(
                ["git", "-C",
                 str(repo), "commit", "-qam", "second"], check=True
            )
            subprocess.run(
                ["git", "-C",
                 str(repo), "remote", "add", "origin",
                 str(repo)], check=True
            )
            status, mismatches = set(), []
            with (contextlib.redirect_stdout(io.StringIO()),
                  contextlib.redirect_stderr(io.StringIO())):
                fetch.check_state(
                    "reference", "unused", str(repo), pin, False, True, status,
                    mismatches
                )
            self.assertFalse(status)
            head = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
            self.assertEqual(head, pin)
            self.assertEqual((repo / "tracked").read_text(), "original\n")

    def test_reference_detects_ignored_file_in_submodule(self) -> None:
        """Detect ignored files in reference submodules."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child, parent = root / "child", root / "parent"
            child.mkdir()
            parent.mkdir()
            self.make_git_repo(child)
            self.make_git_repo(parent)
            subprocess.run(
                [
                    "git", "-C",
                    str(parent), "-c", "protocol.file.allow=always",
                    "submodule", "add", "-q",
                    str(child), "nested"
                ], check=True
            )
            subprocess.run(
                ["git", "-C",
                 str(parent), "commit", "-qam", "submodule"], check=True
            )
            self.assertFalse(fetch.checkout_dirty(str(parent)))
            (parent / "nested/ignored").write_text("local work\n")
            self.assertTrue(fetch.checkout_dirty(str(parent)))

    def test_fetch_rejects_unknown_reference(self) -> None:
        """Reject an unknown reference selection before fetching."""
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/fetch_references.py"), "--only",
                "not-a-reference", "--check"
            ], capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("known checkout names", result.stderr)


if __name__ == "__main__":
    unittest.main()
