#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Offline tests for pinned CID22 archive and dataset handling."""

import hashlib
import importlib.util
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "fetch_cid22", ROOT / "validation/fetch_cid22.py"
)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class FetchCid22Tests(unittest.TestCase):
    """Test CID22 fetching and publication with a local archive."""

    def setUp(self) -> None:
        """Create a pinned archive and patch expected metadata."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.archive = root / "download" / "cid22.zip"
        self.data = root / "data"
        self.dataset = self.data / fetch.DATASET_NAME
        self.csv = (
            "reference_img,distorted_img,MCOS\n"
            "original/a.png,original/a.png,99\n"
            "original/a.png,compressed/a/codec/q.jpg,42\n"
        )
        self.members = {
            "LICENSE": b"official license\n",
            f"{fetch.DATASET_NAME}/original/a.png": b"original",
            f"{fetch.DATASET_NAME}/compressed/a/codec/q.jpg": b"distorted",
            f"{fetch.DATASET_NAME}/{fetch.DATASET_NAME}.csv": self.csv.encode(),
        }
        self.patches = [
            mock.patch.object(fetch, "ARCHIVE", self.archive),
            mock.patch.object(fetch, "DATA_ROOT", self.data),
            mock.patch.object(fetch, "EXPECTED_SIZE", 0),
            mock.patch.object(fetch, "EXPECTED_SHA256", ""),
            mock.patch.object(fetch, "EXPECTED_ORIGINALS", 1),
            mock.patch.object(fetch, "EXPECTED_DISTORTED", 1)
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.create_archive()

    def create_archive(self) -> None:
        """Serialize the current fixture members and update the pinned hash."""
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            for name, content in self.members.items():
                archive.writestr(name, content)
        self.bytes = output.getvalue()
        fetch.EXPECTED_SIZE = len(self.bytes)
        fetch.EXPECTED_SHA256 = hashlib.sha256(self.bytes).hexdigest()

    def cache(self, data: bytes | None = None) -> None:
        """Write a fixture archive into the download cache.

        :param data: Override bytes, or None for the valid archive.
        """
        self.archive.parent.mkdir(parents=True, exist_ok=True)
        self.archive.write_bytes(self.bytes if data is None else data)

    def fetch_download(self, data: bytes | None = None) -> int:
        """Run the downloader with a mocked archive response.

        :param data: Override response bytes, or None for the valid archive.
        :returns: Downloader exit status.
        """
        with mock.patch.object(
                fetch.urllib.request, "urlopen",
                return_value=io.BytesIO(self.bytes if data is None else data)):
            return fetch.main([])

    def test_download_extract_check_and_rerun(self) -> None:
        """Extract a valid download and verify subsequent offline runs."""
        self.assertEqual(self.fetch_download(), 0)
        self.assertEqual(
            (self.data / "LICENSE").read_bytes(), b"official license\n"
        )
        self.assertEqual(
            (self.dataset / "original/a.png").read_bytes(), b"original"
        )
        with mock.patch.object(fetch.urllib.request, "urlopen",
                               side_effect=AssertionError("network used")):
            self.assertEqual(fetch.main(["--check"]), 0)
            self.assertEqual(fetch.main([]), 0)

    def test_check_is_offline_and_does_not_create_directories(self) -> None:
        """Report a missing archive offline without creating directories."""
        with (mock.patch.object(fetch.urllib.request, "urlopen",
                                side_effect=AssertionError("network used")),
              self.assertRaisesRegex(SystemExit, "archive is missing"),
              ):
            fetch.main(["--check"])
        self.assertFalse(self.archive.parent.exists())
        self.assertFalse(self.data.exists())

    def test_cached_corruption_is_never_replaced(self) -> None:
        """Preserve a corrupt cache for diagnosis."""
        self.cache(b"X" * len(self.bytes))
        with (mock.patch.object(fetch.urllib.request, "urlopen",
                                side_effect=AssertionError("network used")),
              self.assertRaisesRegex(SystemExit, "SHA-256"),
              ):
            fetch.main([])
        self.assertEqual(self.archive.read_bytes(), b"X" * len(self.bytes))
        self.assertFalse(self.data.exists())

    def test_bad_download_is_not_published(self) -> None:
        """Keep invalid downloads out of the cache and dataset."""
        with self.assertRaisesRegex(SystemExit, "SHA-256"):
            self.fetch_download(b"wrong")
        self.assertFalse(self.archive.exists())
        self.assertFalse(self.data.exists())
        self.assertEqual(list(self.archive.parent.iterdir()), [])

    def test_existing_unmanaged_dataset_is_not_changed(self) -> None:
        """Preserve an unmanaged dataset directory and its contents."""
        self.cache()
        self.dataset.mkdir(parents=True)
        sentinel = self.dataset / "private"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(SystemExit, "missing or unsafe"):
            fetch.main([])
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertEqual(list(self.dataset.iterdir()), [sentinel])
        self.assertFalse((self.data / "LICENSE").exists())

    def test_existing_license_is_never_overwritten(self) -> None:
        """Reject a conflicting existing license without overwriting it."""
        self.cache()
        self.data.mkdir()
        license_path = self.data / "LICENSE"
        license_path.write_text("mine")
        with self.assertRaisesRegex(SystemExit, "does not match"):
            fetch.main([])
        self.assertEqual(license_path.read_text(), "mine")
        self.assertFalse(self.dataset.exists())

    def test_tampered_extracted_file_and_missing_license_fail_check(
        self
    ) -> None:
        """Detect changed images and missing licenses."""
        self.cache()
        self.assertEqual(fetch.main([]), 0)
        image = self.dataset / "original/a.png"
        image.write_bytes(b"different")
        with self.assertRaisesRegex(SystemExit, "does not match"):
            fetch.main(["--check"])
        self.assertEqual(image.read_bytes(), b"different")
        image.write_bytes(b"original")
        (self.data / "LICENSE").unlink()
        with self.assertRaisesRegex(SystemExit, "missing or unsafe"):
            fetch.main(["--check"])

    def test_csv_missing_pair_rejected_before_publication(self) -> None:
        """Reject a CSV that omits a distortion pair before publication."""
        key = f"{fetch.DATASET_NAME}/{fetch.DATASET_NAME}.csv"
        self.members[key] = (
            b"reference_img,distorted_img,MCOS\n"
            b"original/a.png,original/a.png,99\n"
        )
        self.create_archive()
        self.cache()
        with self.assertRaisesRegex(SystemExit, "does not cover"):
            fetch.main([])
        self.assertFalse(self.dataset.exists())
        self.assertFalse((self.data / "LICENSE").exists())

    @unittest.skipIf(
        fetch.os.name == "nt", "directory publication differs on Windows"
    )
    def test_failed_publication_leaves_no_partial_dataset_and_can_retry(
        self
    ) -> None:
        """Clean a failed dataset reservation and permit a retry."""
        self.cache()
        with (mock.patch.object(fetch.os, "replace",
                                side_effect=OSError("interrupted")),
              self.assertRaisesRegex(SystemExit, "interrupted"),
              ):
            fetch.main([])
        self.assertFalse(self.dataset.exists())
        self.assertEqual(fetch.main([]), 0)
        self.assertEqual(fetch.main(["--check"]), 0)

    def test_zip_path_traversal_rejected(self) -> None:
        """Reject paths that escape the private archive extraction directory."""
        self.members[f"{fetch.DATASET_NAME}/../../escape"] = b"bad"
        self.create_archive()
        self.cache()
        with self.assertRaisesRegex(SystemExit, "unsafe CID22 archive"):
            fetch.main([])
        self.assertFalse((self.data.parent / "escape").exists())
        self.assertFalse(self.dataset.exists())


if __name__ == "__main__":
    unittest.main()
