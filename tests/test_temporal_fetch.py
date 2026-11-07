#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Verify pinned temporal-prefix downloads without contacting the network."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.client import IncompleteRead
from pathlib import Path
from unittest.mock import patch

from validation.temporal import fetch

HEADER = "YUV4MPEG2 W2 H2 F30000:1001 Ip A128:117"
PAYLOAD = HEADER.encode("ascii") + b"\nFRAME\n" + bytes(range(6))
PAYLOAD += b"FRAME\n" + bytes(range(6, 12))
UPSTREAM_BYTES = 1000
SOURCE_URL = "https://example.test/source.y4m"


def manifest_document() -> dict[str, object]:
    """Return a tiny complete-frame source manifest.

    :returns: Mutable valid manifest with the same schema as the hosted corpus.
    """
    return {
        "schema_version":
        1,
        "name":
        "test-corpus",
        "sources": [
            {
                "id": "source",
                "source_id": "source",
                "url": SOURCE_URL,
                "filename": "source.y4m",
                "prefix_bytes": len(PAYLOAD),
                "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
                "width": 2,
                "height": 2,
                "frames": 2,
                "fps": "30000:1001",
                "chroma": "420jpeg",
                "header": HEADER,
                "upstream_bytes": UPSTREAM_BYTES,
                "catalogue": "https://example.test/"
            }
        ]
    }


class Response(io.BytesIO):
    """Represent an HTTPS byte-range response in memory."""

    def __init__(
        self, payload: bytes = PAYLOAD, status: int = 206,
        headers: dict[str, str] | None = None, url: str = SOURCE_URL
    ) -> None:
        """Build a response with optional protocol and payload faults.

        :param payload: Bytes returned by the response stream.
        :param status: HTTP response status.
        :param headers: Header overrides, including deliberately invalid values.
        :param url: Final URL after any redirects.
        """
        super().__init__(payload)
        self.status = status
        self.headers = {
            "Content-Range": f"bytes 0-{len(PAYLOAD) - 1}/{UPSTREAM_BYTES}",
            "Content-Length": str(len(PAYLOAD))
        }
        self.headers.update(headers or {})
        self.url = url

    def geturl(self) -> str:
        """Return the simulated final URL.

        :returns: Final source URL.
        """
        return self.url


class TemporalFetchTests(unittest.TestCase):
    """Exercise manifest contracts, atomic publication and offline verification."""

    def setUp(self) -> None:
        """Create an isolated manifest and initially absent cache directory."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.manifest = self.root / "sources.json"
        self.cache = self.root / "cache"
        self.write_manifest(manifest_document())
        self.sources = fetch.load_manifest(self.manifest)

    def write_manifest(self, document: dict[str, object]) -> None:
        """Write one temporary test manifest.

        :param document: JSON-serializable source manifest.
        """
        self.manifest.write_text(json.dumps(document), encoding="utf-8")

    def test_checked_in_manifest_is_self_consistent(self) -> None:
        """Check the real hosted matrix without opening source URLs."""
        sources = fetch.load_manifest()
        self.assertEqual(len(sources), 30)
        self.assertEqual(len({source.source_id for source in sources}), 30)
        self.assertTrue(all(source.frames == 40 for source in sources))

    def test_manifest_rejects_invalid_source_fields(self) -> None:
        """Reject traversal, unpinned content and inconsistent frame metadata."""
        invalid = [
            ("id", "../source"), ("source_id", "source/elsewhere"),
            ("filename", "../source.y4m"), ("filename", "nested\\source.y4m"),
            ("filename", "/source.y4m"), ("url", "http://example.test/a"),
            ("url", "https://user:secret@example.test/a"), ("sha256", "abc"),
            ("sha256", "g" * 64), ("width", True), ("width", 0), ("height", "2"),
            ("frames", 3), ("prefix_bytes", len(PAYLOAD) - 1), ("upstream_bytes", 1),
            ("fps", "0:1"), ("fps", "30:1"), ("header", HEADER + "\n"),
            ("header", HEADER + " W2"), ("chroma", "420p10")
        ]
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                document = manifest_document()
                document["sources"][0][field] = value
                self.write_manifest(document)
                self.assertRaises(fetch.FetchError, fetch.load_manifest, self.manifest)

    def test_manifest_rejects_duplicate_ids_and_destinations(self) -> None:
        """Never assign two download workers to the same logical source or file."""
        for field in ("id", "filename"):
            document = manifest_document()
            second = copy.deepcopy(document["sources"][0])
            second.update(id="other", filename="other.y4m")
            second[field] = document["sources"][0][field]
            document["sources"].append(second)
            self.write_manifest(document)
            with self.subTest(field=field):
                self.assertRaises(fetch.FetchError, fetch.load_manifest, self.manifest)

    def test_manifest_requires_version_and_source_rows(self) -> None:
        """Reject unsupported schema versions and missing source descriptions."""
        for document in ({"schema_version": True, "sources":
                          []}, {"schema_version": 2, "sources":
                                []}, {"schema_version": 1, "sources":
                                      []}, {"schema_version": 1, "sources": [None]}):
            self.write_manifest(document)
            self.assertRaises(fetch.FetchError, fetch.load_manifest, self.manifest)

    def test_selects_known_sources(self) -> None:
        """Reject accidental empty/unknown selections before any network work."""
        self.assertEqual(fetch.select_sources(self.sources, None), self.sources)
        self.assertEqual(fetch.select_sources(self.sources, " source "), self.sources)
        self.assertRaises(fetch.FetchError, fetch.select_sources, self.sources, "")
        self.assertRaises(
            fetch.FetchError, fetch.select_sources, self.sources, "missing"
        )

    def test_downloads_exact_range_then_replays_offline(self) -> None:
        """Publish the complete verified prefix and reuse it without network calls."""
        with (patch.object(fetch, "urlopen", return_value=Response()) as
              opened, redirect_stdout(io.StringIO())):
            fetch.fetch_sources(self.sources, self.cache)
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, SOURCE_URL)
        self.assertEqual(request.get_header("Range"), f"bytes=0-{len(PAYLOAD) - 1}")
        self.assertEqual(request.get_header("Accept-encoding"), "identity")
        self.assertEqual(opened.call_args.kwargs["timeout"], 60)
        self.assertEqual((self.cache / "source.y4m").read_bytes(), PAYLOAD)
        self.assertEqual(
            sorted(path.name for path in self.cache.iterdir()), ["source.y4m"]
        )
        with patch.object(fetch, "urlopen") as opened, redirect_stdout(io.StringIO()):
            fetch.fetch_sources(self.sources, self.cache)
            fetch.fetch_sources(self.sources, self.cache, check=True)
        opened.assert_not_called()

    def test_preserves_corrupt_cache_without_network(self) -> None:
        """Leave both wrong-sized and same-sized corrupt cache files untouched."""
        self.cache.mkdir()
        path = self.cache / "source.y4m"
        for contents in (b"incomplete", PAYLOAD[:-1] + b"x"):
            path.write_bytes(contents)
            with patch.object(fetch, "urlopen") as opened:
                self.assertRaises(
                    fetch.FetchError, fetch.fetch_sources, self.sources, self.cache
                )
            opened.assert_not_called()
            self.assertEqual(path.read_bytes(), contents)

    def test_offline_check_never_creates_directory_or_opens_url(self) -> None:
        """Missing inputs in --check fail entirely offline without mutations."""
        with patch.object(fetch, "urlopen") as opened, redirect_stderr(io.StringIO()):
            result = fetch.main(
                [
                    "--manifest",
                    str(self.manifest), "--output-dir",
                    str(self.cache), "--check"
                ]
            )
        self.assertEqual(result, 1)
        self.assertFalse(self.cache.exists())
        opened.assert_not_called()

    def test_rejects_ignored_or_wrong_range_responses(self) -> None:
        """Check range bounds/total, byte encoding and final HTTPS after redirects."""
        responses = [
            Response(status=200),
            Response(headers={"Content-Range": "bytes 1-9/1000"}),
            Response(headers={"Content-Range": f"bytes 0-{len(PAYLOAD) - 1}/2000"}),
            Response(headers={"Content-Length": "999"}),
            Response(headers={"Content-Encoding": "gzip"}),
            Response(url="http://example.test/redirected.y4m")
        ]
        for response in responses:
            with (patch.object(fetch, "urlopen",
                               return_value=response), redirect_stdout(io.StringIO())):
                self.assertRaises(
                    fetch.FetchError, fetch.fetch_sources, self.sources, self.cache
                )
            self.assertEqual(list(self.cache.iterdir()), [])

    def test_rejects_truncated_and_oversized_bodies(self) -> None:
        """Do not publish partial content or trust the server's claimed size."""
        for payload in (PAYLOAD[:-1], PAYLOAD + b"extra"):
            response = Response(payload)
            with (patch.object(fetch, "urlopen",
                               return_value=response), redirect_stdout(io.StringIO())):
                self.assertRaises(
                    fetch.FetchError, fetch.fetch_sources, self.sources, self.cache
                )
            self.assertEqual(list(self.cache.iterdir()), [])

    def test_rejects_checksum_mismatch_after_download(self) -> None:
        """A same-sized corrupt response must fail before atomic publication."""
        with (patch.object(fetch, "urlopen",
                           return_value=Response(PAYLOAD[:-1] + b"x")),
              redirect_stdout(io.StringIO())):
            self.assertRaises(
                fetch.FetchError, fetch.fetch_sources, self.sources, self.cache
            )
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_rejects_hash_pinned_invalid_frame_structure(self) -> None:
        """The manifest hash cannot substitute for complete Y4M frame boundaries."""
        malformed = PAYLOAD.replace(b"FRAME\n", b"BROKE\n", 1)
        document = manifest_document()
        document["sources"][0]["sha256"] = hashlib.sha256(malformed).hexdigest()
        self.write_manifest(document)
        sources = fetch.load_manifest(self.manifest)
        with (patch.object(fetch, "urlopen", return_value=Response(malformed)),
              redirect_stdout(io.StringIO())):
            self.assertRaises(
                fetch.FetchError, fetch.fetch_sources, sources, self.cache
            )
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_accepts_https_redirect_and_preserves_original_aspect(self) -> None:
        """Keep hosted source bytes intact even when original pixel aspect is non-square."""
        response = Response(url="https://cdn.example.test/redirected.y4m")
        with (patch.object(fetch, "urlopen",
                           return_value=response), redirect_stdout(io.StringIO())):
            fetch.fetch_sources(self.sources, self.cache)
        self.assertEqual((self.cache / "source.y4m").read_bytes(), PAYLOAD)

    def test_refuses_symlink_cache_destination(self) -> None:
        """Never follow a cache destination into a maintained or external file."""
        target = self.root / "maintained.y4m"
        target.write_bytes(PAYLOAD)
        self.cache.mkdir()
        (self.cache / "source.y4m").symlink_to(target)
        with patch.object(fetch, "urlopen") as opened:
            self.assertRaises(
                fetch.FetchError, fetch.fetch_sources, self.sources, self.cache
            )
        opened.assert_not_called()
        self.assertEqual(target.read_bytes(), PAYLOAD)

    def test_incomplete_http_stream_is_reported_and_cleaned(self) -> None:
        """Treat HTTP-level truncation as a named fetch failure without partial files."""
        with (patch.object(fetch, "urlopen", side_effect=IncompleteRead(b"partial",
                                                                        99)),
              redirect_stdout(io.StringIO())):
            self.assertRaises(
                fetch.FetchError, fetch.fetch_sources, self.sources, self.cache
            )
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_publication_never_replaces_a_conflicting_concurrent_file(self) -> None:
        """Atomic publication must preserve an existing destination created during transfer."""

        def conflict(_temporary: Path, destination: Path) -> None:
            """Simulate another writer creating unique work immediately before publication."""
            destination.write_bytes(b"preserve concurrent file")
            raise FileExistsError(destination)

        with (patch.object(fetch, "urlopen", return_value=Response()),
              patch.object(fetch.os, "link",
                           side_effect=conflict), redirect_stdout(io.StringIO())):
            self.assertRaises(
                fetch.FetchError, fetch.fetch_sources, self.sources, self.cache
            )
        self.assertEqual(
            (self.cache / "source.y4m").read_bytes(), b"preserve concurrent file"
        )
        self.assertEqual(
            sorted(path.name for path in self.cache.iterdir()), ["source.y4m"]
        )


if __name__ == "__main__":
    unittest.main()
