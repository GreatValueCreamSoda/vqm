#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Fetch the pinned CID22 validation set, or verify it offline with --check.

The official archive is cached under validation/download/; the extracted data
is kept under validation/data/. Neither is tracked in Git. Existing invalid
files or datasets are reported and never overwritten or removed.
"""

import argparse
import csv
import hashlib
import math
import os
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path
from pathlib import PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent
URL = (
    "https://cloudinary-marketing-res.cloudinary.com/raw/upload/"
    "v1682006242/CID22_validation_set.zip"
)
ARCHIVE = REPO_ROOT / "validation/download/CID22_validation_set.zip"
DATA_ROOT = REPO_ROOT / "validation/data"
DATASET_NAME = "CID22_validation_set"
EXPECTED_SIZE = 1467356735
EXPECTED_SHA256 = (
    "ab358e9b3589002d51f27e9d89475af445fd2edf545ec58312c99da9433d1381"
)
EXPECTED_ORIGINALS = 49
EXPECTED_DISTORTED = 4292


def verify_archive(path: Path) -> None:
    """Check the entire cached or freshly downloaded archive before use.

    :param path: Archive to verify.
    :raises ValueError: If the archive differs from its pinned size or digest.
    """
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            size += len(chunk)
            digest.update(chunk)
    if size != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError(
            f"CID22 archive {path} failed its size or SHA-256 check "
            f"(size {size}, SHA-256 {digest.hexdigest()}); "
            "inspect or remove it before retrying"
        )


def _download_archive(temporary: Path) -> None:
    with urllib.request.urlopen(
            URL, timeout=60) as response, temporary.open("wb") as out:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            out.write(chunk)


def ensure_archive(check: bool = False) -> None:
    """Fetch into a private temporary file and publish without overwriting.

    :param check: Verify the cached archive without fetching it.
    :raises ValueError: If a required archive is missing or fails verification.
    """
    if os.path.lexists(ARCHIVE):
        verify_archive(ARCHIVE)
        return
    if check:
        raise ValueError(f"CID22 archive is missing: {ARCHIVE}")
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    with tempfile.TemporaryDirectory(prefix="cid22-download-",
                                     dir=ARCHIVE.parent) as temp:
        temporary = Path(temp) / "archive.zip"
        _download_archive(temporary)
        verify_archive(temporary)
        # A concurrent downloader or an unmanaged cache must never be replaced.
        try:
            os.link(temporary, ARCHIVE)
        except FileExistsError:
            verify_archive(ARCHIVE)


def archive_manifest(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """List members after checking paths and types for safe extraction.

    :param archive: Verified CID22 ZIP archive.
    :returns: Dataset-relative files and their ZIP metadata.
    :raises ValueError: If an archive member is unsafe or the license is
        missing.
    """
    files = {}
    seen = set()
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        if (not name or not path.parts or "\\" in name or path.is_absolute()
                or any(part in (".", "..") for part in path.parts)
                or (path.parts[0] != DATASET_NAME and name != "LICENSE")):
            raise ValueError(f"unsafe CID22 archive member: {name!r}")
        kind = (info.external_attr >> 16) & 0o170000
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(f"unsupported CID22 archive member: {name!r}")
        normalized = str(path)
        if normalized in seen:
            raise ValueError(f"duplicate CID22 archive member: {name!r}")
        seen.add(normalized)
        if not info.is_dir() and path.parts[0] == DATASET_NAME:
            relative = str(path.relative_to(DATASET_NAME))
            files[relative] = info
    if "LICENSE" not in seen or archive.getinfo("LICENSE").is_dir():
        raise ValueError("CID22 archive is missing its license")
    return files


def verify_file(path: Path, info: zipfile.ZipInfo) -> None:
    """Match an extracted regular file to its archive member.

    :param path: Extracted path.
    :param info: Archive size and CRC record.
    :raises ValueError: If the file is missing, unsafe or altered.
    """
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"CID22 file is missing or unsafe: {path}")
    crc, size = 0, 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            crc = zlib.crc32(chunk, crc)
            size += len(chunk)
    if size != info.file_size or crc != info.CRC:
        raise ValueError(
            f"CID22 file does not match the verified archive: {path}"
        )


def verify_dataset(dataset: Path, files: dict[str, zipfile.ZipInfo]) -> None:
    """Verify extracted files against the pinned archive and CSV pairs.

    :param dataset: Extracted dataset directory.
    :param files: Verified archive member metadata keyed by relative path.
    :raises ValueError: If layout, extracted files or annotation coverage
        differ.
    """
    if dataset.is_symlink() or not dataset.is_dir():
        raise ValueError(
            f"CID22 dataset is missing or not a directory: {dataset}"
        )
    csv_name = f"{DATASET_NAME}.csv"
    originals = {
        p
        for p in files
        if p.startswith("original/") and p.lower().endswith(".png")
    }
    distorted = {
        p
        for p in files
        if p.startswith("compressed/") and p.lower().endswith((".png", ".jpg"))
    }
    if (len(originals) != EXPECTED_ORIGINALS
            or len(distorted) != EXPECTED_DISTORTED
            or set(files) != originals | distorted | {csv_name}):
        raise ValueError("CID22 archive has an unexpected dataset layout")
    root = dataset.resolve()
    for name, info in files.items():
        path = dataset / name
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"CID22 file is missing or unsafe: {path}")
        verify_file(path, info)
    image_files = {
        path.relative_to(dataset).as_posix()
        for path in dataset.rglob("*")
        if path.is_file() and path.suffix.lower() in (".png", ".jpg")
    }
    if image_files != originals | distorted:
        raise ValueError("CID22 dataset has unexpected or missing images")
    with (dataset / csv_name).open(newline="", encoding="utf-8") as stream:
        _verify_csv_pairs(csv.DictReader(stream), originals, distorted)


def _finite_mcos(value: str | None) -> bool:
    try:
        return math.isfinite(float(value)) if value is not None else False
    except ValueError:
        return False


def _verify_csv_pairs(
    reader: csv.DictReader, originals: set[str], distorted: set[str]
) -> None:
    if not reader.fieldnames or not {"reference_img", "distorted_img", "MCOS"
                                     } <= set(reader.fieldnames):
        raise ValueError("CID22 CSV is missing required columns")
    pairs, identities, seen_distorted = set(), set(), set()
    for row in reader:
        ref, dist = row["reference_img"], row["distorted_img"]
        if (ref not in originals or dist not in originals | distorted
                or not _finite_mcos(row["MCOS"]) or (ref, dist) in pairs):
            raise ValueError("CID22 CSV contains an invalid or duplicate pair")
        pairs.add((ref, dist))
        if dist == ref:
            identities.add(ref)
        elif (dist in distorted and dist not in seen_distorted
              and len(PurePosixPath(dist).parts) == 4
              and PurePosixPath(dist).parts[1] == PurePosixPath(ref).stem):
            seen_distorted.add(dist)
        else:
            raise ValueError(
                "CID22 CSV has an unexpected reference or duplicate distortion"
            )
    if identities != originals or seen_distorted != distorted:
        raise ValueError(
            "CID22 CSV does not cover every original and distortion"
        )


def _publish_license(
    staged: Path, license_path: Path, info: zipfile.ZipInfo
) -> None:
    if os.path.lexists(license_path):
        verify_file(license_path, info)
        return
    try:
        os.link(staged, license_path)
    except FileExistsError:
        verify_file(license_path, info)


def _move_dataset(staged: Path, dataset: Path) -> None:
    if os.name == "nt":
        # Windows does not replace an existing directory.
        for child in staged.iterdir():
            child.rename(dataset / child.name)
    else:
        os.replace(staged, dataset)


def _remove_reservation(dataset: Path, reservation: os.stat_result) -> None:
    # Leave a failed install retryable if the directory is still ours and empty.
    try:
        current = dataset.stat()
        if (current.st_dev, current.st_ino) == (reservation.st_dev,
                                                reservation.st_ino):
            dataset.rmdir()
    except OSError:
        pass


def _install_dataset(staged: Path, dataset: Path) -> None:
    # Reserve the destination exclusively before publishing.
    dataset.mkdir()
    reservation = dataset.stat()
    try:
        _move_dataset(staged, dataset)
    except OSError:
        _remove_reservation(dataset, reservation)
        raise


def _ensure_dataset(archive: zipfile.ZipFile, check: bool) -> Path:
    files = archive_manifest(archive)
    dataset = DATA_ROOT / DATASET_NAME
    license_path = DATA_ROOT / "LICENSE"
    if os.path.lexists(dataset):
        verify_dataset(dataset, files)
        verify_file(license_path, archive.getinfo("LICENSE"))
        return dataset
    if check:
        raise ValueError(f"CID22 dataset is missing: {dataset}")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"extracting {ARCHIVE}")
    with tempfile.TemporaryDirectory(prefix="cid22-extract-",
                                     dir=DATA_ROOT) as temp:
        archive.extractall(temp)
        staged = Path(temp) / DATASET_NAME
        staged_license = Path(temp) / "LICENSE"
        verify_dataset(staged, files)
        info = archive.getinfo("LICENSE")
        verify_file(staged_license, info)
        _publish_license(staged_license, license_path, info)
        _install_dataset(staged, dataset)
    return dataset


def main(argv: list[str] | None = None) -> int:
    """Fetch or verify the pinned CID22 archive and extracted dataset.

    :param argv: Optional command-line arguments.
    :returns: Zero after verification succeeds.
    :raises SystemExit: If arguments or corpus verification fail.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify offline without writing"
    )
    args = parser.parse_args(argv)
    try:
        ensure_archive(check=args.check)
        with zipfile.ZipFile(ARCHIVE) as archive:
            dataset = _ensure_dataset(archive, args.check)
        print(f"CID22 validation set verified in {dataset}")
        return 0
    except (OSError, ValueError, csv.Error, zipfile.BadZipFile,
            urllib.error.URLError) as error:
        raise SystemExit(f"CID22 verification failed: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
