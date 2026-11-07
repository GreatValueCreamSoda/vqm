#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Fetch checksum-pinned, complete-frame prefixes of hosted temporal sources.

Run ``python3 -m validation.temporal.fetch`` from the repository root. Existing
files are verified and preserved; remove an invalid cache file explicitly before
retrying. ``--check`` is entirely offline and never creates directories or
files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from http.client import HTTPException
from http.client import HTTPResponse
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlsplit
from urllib.request import Request
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("sources.json")
DEFAULT_OUTPUT = ROOT / "validation/download/temporal"
CHUNK_BYTES = 1024 * 1024


class FetchError(ValueError):
    """Report a manifest, cache or HTTP response that cannot be verified."""


@dataclass(frozen=True)
class Source:
    """One pinned prefix, with original Y4M metadata and source provenance."""

    id: str
    source_id: str
    url: str
    filename: str
    prefix_bytes: int
    sha256: str
    width: int
    height: int
    frames: int
    fps: str
    chroma: str
    header: str
    upstream_bytes: int
    catalogue: str


def _mapping(value: object, description: str) -> dict[str, object]:
    """Require an object in the JSON schema."""
    if not isinstance(value, dict):
        raise FetchError(f"{description}: expected an object")
    return value


def _text(row: dict[str, object], name: str) -> str:
    """Require a nonempty string field."""
    value = row.get(name)
    if not isinstance(value, str) or not value or value.strip() != value:
        raise FetchError(
            f"{name}: expected a nonempty string without outer whitespace"
        )
    return value


def _positive_int(row: dict[str, object], name: str) -> int:
    """Require a positive JSON integer, excluding booleans."""
    value = row.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FetchError(f"{name}: expected a positive integer")
    return value


def _https_url(value: str) -> None:
    """Reject unsupported, ambiguous or credential-bearing source URLs."""
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.hostname:
        raise FetchError(f"source URL must use HTTPS: {value}")
    if parts.username or parts.password or parts.fragment:
        raise FetchError(
            f"source URL must omit credentials and fragments: {value}"
        )


def _validate_names(source: Source) -> None:
    """Keep identities and output filenames independent of path traversal."""
    for name in (source.id, source.source_id):
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) is None:
            raise FetchError(f"invalid source identity: {name}")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.y4m",
                    source.filename) is None:
        raise FetchError(f"invalid Y4M filename: {source.filename}")
    if re.fullmatch(r"[a-fA-F0-9]{64}", source.sha256) is None:
        raise FetchError(
            f"{source.id}: sha256 must contain exactly 64 hexadecimal digits"
        )
    _https_url(source.url)
    _https_url(source.catalogue)


def _header_fields(header: str) -> dict[str, str]:
    """Read single-valued Y4M header tokens without changing source bytes."""
    if not header.isascii() or any(ord(character) < 32 for character in header):
        raise FetchError("Y4M header must be one printable ASCII line")
    tokens = header.split()
    if not tokens or tokens[0] != "YUV4MPEG2":
        raise FetchError("invalid Y4M header magic")
    fields: dict[str, str] = {}
    for token in tokens[1:]:
        if token[0] in fields:
            raise FetchError(f"duplicate Y4M header token: {token[0]}")
        fields[token[0]] = token[1:]
    return fields


def frame_bytes(source: Source) -> int:
    """Return the byte size of one supported eight-bit raw Y4M frame.

    :param source: Validated geometry and chroma coding.
    :returns: Payload bytes, excluding the frame marker.
    :raises FetchError: If the chroma coding is unsupported.
    """
    luma = source.width * source.height
    if source.chroma in {"420jpeg", "420mpeg2", "420paldv"}:
        return luma + 2 * ((source.width + 1) // 2) * ((source.height + 1) // 2)
    if source.chroma == "422":
        return luma + 2 * ((source.width + 1) // 2) * source.height
    if source.chroma == "444":
        return 3 * luma
    if source.chroma == "mono":
        return luma
    raise FetchError(
        f"{source.id}: unsupported eight-bit chroma: {source.chroma}"
    )


def _validate_layout(source: Source) -> None:
    """Match prefix bounds, geometry and timing to the pinned header."""
    fields = _header_fields(source.header)
    expected = {
        "W": str(source.width),
        "H": str(source.height),
        "F": source.fps,
        "I": "p"
    }
    if any(fields.get(name) != value for name, value in expected.items()):
        raise FetchError(
            f"{source.id}: header geometry, timing or interlace mismatch"
        )
    if fields.get("C", "420jpeg") != source.chroma:
        raise FetchError(f"{source.id}: header chroma mismatch")
    if re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", source.fps) is None:
        raise FetchError(f"{source.id}: fps must be a positive integer ratio")
    expected_bytes = len(source.header
                         ) + 1 + source.frames * (6 + frame_bytes(source))
    if source.prefix_bytes != expected_bytes:
        raise FetchError(
            f"{source.id}: prefix_bytes does not contain exactly frames frames"
        )
    if source.upstream_bytes < source.prefix_bytes:
        raise FetchError(f"{source.id}: prefix exceeds upstream_bytes")


def _source(row: object) -> Source:
    """Parse one source row and validate its complete-frame prefix contract."""
    data = _mapping(row, "source")
    source = Source(
        id=_text(data, "id"), source_id=_text(data, "source_id"),
        url=_text(data, "url"), filename=_text(data, "filename"),
        prefix_bytes=_positive_int(data, "prefix_bytes"),
        sha256=_text(data, "sha256"), width=_positive_int(data, "width"),
        height=_positive_int(data,
                             "height"), frames=_positive_int(data, "frames"),
        fps=_text(data,
                  "fps"), chroma=_text(data,
                                       "chroma"), header=_text(data, "header"),
        upstream_bytes=_positive_int(data, "upstream_bytes"),
        catalogue=_text(data, "catalogue")
    )
    _validate_names(source)
    _validate_layout(source)
    return source


def load_manifest(path: Path = DEFAULT_MANIFEST) -> list[Source]:
    """Load pinned sources and reject ambiguous cache destinations.

    :param path: JSON manifest to read.
    :returns: Validated sources in manifest order.
    :raises FetchError: If the manifest schema or source metadata is invalid.
    :raises OSError: If the manifest cannot be read.
    """
    document = _mapping(
        json.loads(path.read_text(encoding="utf-8")), "manifest"
    )
    if type(document.get("schema_version")
            ) is not int or document["schema_version"] != 1:
        raise FetchError("manifest schema_version must be 1")
    rows = document.get("sources")
    if not isinstance(rows, list) or not rows:
        raise FetchError("manifest sources must be a nonempty list")
    sources = [_source(row) for row in rows]
    if len({source.id for source in sources}) != len(sources):
        raise FetchError("manifest contains duplicate source IDs")
    if len({source.filename for source in sources}) != len(sources):
        raise FetchError("manifest contains duplicate source filenames")
    return sources


def select_sources(sources: list[Source], only: str | None) -> list[Source]:
    """Select known source IDs while retaining deterministic manifest order.

    :param sources: Complete validated source list.
    :param only: Comma-separated IDs, or None for all sources.
    :returns: Selected sources.
    :raises FetchError: If the selection is empty or contains unknown IDs.
    """
    if only is None:
        return sources
    selected = {name.strip() for name in only.split(",") if name.strip()}
    unknown = selected - {source.id for source in sources}
    if not selected or unknown:
        raise FetchError(
            f"--only requires known source IDs; unknown: {sorted(unknown)}"
        )
    return [source for source in sources if source.id in selected]


def _check_frame_markers(handle: BinaryIO, source: Source) -> None:
    """Verify the header and every untagged frame in a pinned prefix."""
    if handle.readline() != source.header.encode("ascii") + b"\n":
        raise FetchError(
            "cached original Y4M header does not match the manifest"
        )
    for _ in range(source.frames):
        if handle.read(6) != b"FRAME\n":
            raise FetchError(
                "cached source has an invalid or incomplete frame marker"
            )
        handle.seek(frame_bytes(source), os.SEEK_CUR)


def _verify_file(path: Path, source: Source) -> None:
    """Check size, SHA-256 and frame bounds without rewriting the cache."""
    if path.is_symlink() or not path.is_file():
        raise FetchError(
            f"refusing a symlink or non-regular cache file: {path}"
        )
    if path.stat().st_size != source.prefix_bytes:
        raise FetchError(
            f"cache size mismatch; preserve or remove before retrying: {path}"
        )
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != source.sha256.lower():
            raise FetchError(
                f"cache SHA-256 mismatch; preserve or remove before "
                f"retrying: {path}"
            )
        handle.seek(0)
        _check_frame_markers(handle, source)


def _validate_response(response: HTTPResponse, source: Source) -> None:
    """Require the exact pinned HTTP byte range and an HTTPS final URL."""
    _https_url(response.geturl())
    if response.status != 206:
        raise FetchError(
            f"server returned HTTP {response.status}; expected 206 byte range"
        )
    expected = f"bytes 0-{source.prefix_bytes - 1}/{source.upstream_bytes}"
    if response.headers.get("Content-Range") != expected:
        raise FetchError(f"server Content-Range must be {expected}")
    length = response.headers.get("Content-Length")
    if length is not None and length != str(source.prefix_bytes):
        raise FetchError(
            "server Content-Length does not match the pinned prefix"
        )
    if response.headers.get("Content-Encoding",
                            "identity").lower() != "identity":
        raise FetchError(
            "server encoded the byte range; original bytes are required"
        )


def _copy_response(
    response: HTTPResponse, destination: BinaryIO, expected: int
) -> None:
    """Stream at most the pinned byte count plus one overflow-detection byte."""
    received = 0
    while chunk := response.read(min(CHUNK_BYTES, expected - received + 1)):
        received += len(chunk)
        if received > expected:
            raise FetchError("server sent an oversized byte range")
        destination.write(chunk)
    if received != expected:
        raise FetchError(
            f"truncated byte range: received {received}, expected {expected}"
        )


def _publish_file(temporary: Path, destination: Path, source: Source) -> None:
    """Publish atomically without replacing an existing file, even in a race."""
    try:
        os.link(temporary, destination)
    except FileExistsError:
        _verify_file(destination, source)


def _download(source: Source, destination: Path) -> None:
    """Fetch a pinned range before publishing without overwriting files."""
    request = Request(
        source.url, headers={
            "Range": f"bytes=0-{source.prefix_bytes - 1}",
            "Accept-Encoding": "identity",
            "User-Agent": "VQM-temporal-corpus/1"
        }
    )
    with tempfile.NamedTemporaryFile(prefix=f".{source.id}.", suffix=".part",
                                     dir=destination.parent,
                                     delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with urlopen(request,
                     timeout=60) as response, temporary.open("wb") as handle:
            _validate_response(response, source)
            _copy_response(response, handle, source.prefix_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        _verify_file(temporary, source)
        _publish_file(temporary, destination, source)
    finally:
        temporary.unlink(missing_ok=True)


def _fetch_one(source: Source, output_dir: Path, check: bool) -> None:
    """Verify one cached source, or fetch a missing source when authorized."""
    destination = output_dir / source.filename
    if destination.exists() or destination.is_symlink():
        _verify_file(destination, source)
        print(
            f"{source.id}: verified {source.prefix_bytes} cached bytes",
            flush=True
        )
        return
    if check:
        raise FetchError(f"missing cache: {destination}")
    print(f"{source.id}: downloading {source.prefix_bytes} bytes", flush=True)
    _download(source, destination)
    print(f"{source.id}: verified download", flush=True)


def _result_error(future: Future[None], source_id: str) -> str:
    """Collect a named worker failure while allowing other sources to finish."""
    try:
        future.result()
    except (OSError, ValueError, HTTPException) as error:
        return f"{source_id}: {error}"
    return ""


def fetch_sources(
    sources: list[Source], output_dir: Path, jobs: int = 4, check: bool = False
) -> None:
    """Verify sources and fetch only missing complete-frame prefixes.

    :param sources: Validated source descriptions.
    :param output_dir: Cache destination; existing invalid files are preserved.
    :param jobs: Maximum simultaneous verification/download workers, 1 through
        32.
    :param check: Verify entirely offline without creating directories or files.
    :raises FetchError: If a selected source cannot be verified or downloaded.
    :raises OSError: If the cache directory cannot be created.
    """
    if not 1 <= jobs <= 32:
        raise FetchError("--jobs must be between 1 and 32")
    if not check:
        output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        pending = {
            executor.submit(_fetch_one, source, output_dir, check): source.id
            for source in sources
        }
        failures = [
            _result_error(future, pending[future])
            for future in as_completed(pending)
        ]
    failures = [failure for failure in failures if failure]
    if failures:
        raise FetchError("\n".join(sorted(failures)))


def main(argv: list[str] | None = None) -> int:
    """Run the bounded hosted-source fetch or offline verification command.

    :param argv: Command arguments, or None for the process arguments.
    :returns: Zero on success, one on a manifest, cache or network error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated source IDs")
    parser.add_argument(
        "--check", action="store_true", help="verify offline only"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        sources = select_sources(load_manifest(args.manifest), args.only)
        fetch_sources(sources, args.output_dir, args.jobs, args.check)
    except (OSError, ValueError) as error:
        print(f"temporal fetch: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
