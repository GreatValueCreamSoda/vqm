#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Fetch the pinned reference checkouts and CID22 validation corpus.

Each reference is fetched into .reference/<name> at the commit pinned in
tools/references.lock. libjxl is a blobless clone with a sparse checkout of
lib/, third_party/, tools/ and cmake/, and its build submodules are fetched;
it is the one tree large enough that the full history is not worth having.

--check verifies selected checkouts and the corpus without downloading. --force
re-checks out the pinned commit in a clean existing clone. Modified, untracked,
or ignored files are rejected (including in submodules): the build copies the
checkout, so even ignored files could affect the result. A directory that holds
a non-git copy cannot be verified and is reported, never overwritten.

The pinned trees are never built in place: tools/build_references.py copies
them under build/ first. CID22 is the official 1.4 GiB archive, fetched by
validation/fetch_cid22.py; it is not a Git checkout. Use --only to select
specific checkout names and/or cid22. The default fetches all of them.
"""

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(REPO_ROOT, "tools", "references.lock")
REFERENCE_DIR = os.path.join(REPO_ROOT, ".reference")

# libjxl bundles the dependencies its build consumes; the sparse checkout
# leaves them empty. Only the ones the reference harness build needs are
# fetched: highway and brotli are required, lcms and skcms are its colour
# management backends, sjpeg is only referenced by its CMake rules.
LIBJXL_SUBMODULES = (
    "third_party/highway", "third_party/brotli", "third_party/lcms",
    "third_party/skcms", "third_party/sjpeg",
)
LIBJXL_SPARSE = ("lib", "third_party", "tools", "cmake")


def parse_lock(path: str) -> list[tuple[str, str, str]]:
    """Read the reference names, repository URLs and revisions.

    :param path: Lock file path.
    :returns: Pinned entries in file order.
    :raises SystemExit: If an entry lacks any required column.
    """
    rows = []
    with open(path) as source:
        lines = source.readlines()
    for number, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise SystemExit(f"{path}:{number}: expected <name> <url> <pin>")
        name, url, pin = parts
        rows.append((name, url, pin))
    return rows


def run(argv: Sequence[str], cwd: str | None = None) -> None:
    """Run a reference setup command.

    :param argv: Executable and arguments.
    :param cwd: Optional working directory.
    :raises SystemExit: If the executable is absent or fails.
    """
    try:
        subprocess.run(argv, cwd=cwd, check=True)
    except FileNotFoundError:
        raise SystemExit(f"{argv[0]} is required") from None
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None


def capture(argv: Sequence[str]) -> str:
    """Capture the output of a Git query.

    :param argv: Executable and arguments.
    :returns: Trimmed stdout, empty if the command fails.
    """
    result = subprocess.run(argv, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def is_git(directory: str) -> bool:
    """Check whether a directory contains a Git checkout.

    :param directory: Candidate checkout path.
    :returns: Whether a Git metadata directory exists.
    """
    return os.path.isdir(os.path.join(directory, ".git"))


def checkout_dirty(directory: str) -> bool:
    """Report local files that a reference build could otherwise copy.

    :param directory: Reference checkout path.
    :returns: Whether the checkout or its submodules contain local changes.
    :raises SystemExit: If Git cannot inspect the checkout.
    """
    status_args = [
        "status", "--porcelain=v1", "--untracked-files=normal", "--ignored",
        "--ignore-submodules=none"
    ]
    commands = [
        ["git", "-C", directory, *status_args],
        [
            "git", "-C", directory, "submodule", "foreach", "--quiet",
            "--recursive", "git", *status_args
        ]
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(
                f"cannot inspect reference {directory}: "
                f"{result.stderr.strip() or 'git status failed'}"
            )
        if result.stdout.strip():
            return True
    return False


def sparse_checkout_file(directory: str) -> str:
    """Locate the sparse checkout patterns.

    :param directory: Reference checkout path.
    :returns: Path to Git's sparse checkout pattern file.
    """
    return os.path.join(directory, ".git", "info", "sparse-checkout")


def ensure_sparse_cmake(directory: str) -> None:
    """Add cmake/ to a sparse checkout that predates it, idempotently.

    :param directory: libjxl checkout path.
    :raises SystemExit: If Git fails to update the patterns.
    """
    path = sparse_checkout_file(directory)
    if not os.path.isfile(path):
        return
    with open(path) as f:
        patterns = f.read().split()
    if "/cmake/" in patterns:
        return
    run(["git", "-C", directory, "sparse-checkout", "set", *LIBJXL_SPARSE])


def missing_submodules(directory: str) -> list[str]:
    """List uninitialized submodules required by the libjxl build.

    :param directory: libjxl checkout path.
    :returns: Uninitialized submodule paths.
    """
    output = capture(
        [
            "git", "-C", directory, "submodule", "status", "--",
            *LIBJXL_SUBMODULES
        ]
    )
    return [
        line.split()[1] for line in output.splitlines() if line.startswith("-")
    ]


def init_submodules(directory: str) -> None:
    """Fetch the pinned libjxl submodules required by the build.

    :param directory: libjxl checkout path.
    :raises SystemExit: If Git fails to initialize the submodules.
    """
    run(
        [
            "git", "-C", directory, "submodule", "update", "--init", "--depth",
            "1", "--", *LIBJXL_SUBMODULES
        ]
    )


def _check_libjxl(directory: str, check: bool) -> bool:
    if check and not os.path.isfile(os.path.join(directory, "cmake",
                                                 "FindAtomics.cmake")):
        print("libjxl: sparse checkout lacks cmake/; run again")
        return False
    if not check:
        ensure_sparse_cmake(directory)
    uninitialized = missing_submodules(directory)
    if not uninitialized:
        return True
    if check:
        print(
            f"libjxl: uninitialized submodules: {' '.join(uninitialized)}; "
            f"run again"
        )
        return False
    init_submodules(directory)
    print("libjxl: initialized submodules")
    return True


def _check_existing(
    name: str, directory: str, pin: str, check: bool, force: bool
) -> bool:
    if checkout_dirty(directory):
        print(
            f"{name}: checkout has modified, untracked, or ignored files; "
            f"inspect {directory} and its submodules before proceeding"
        )
        return False
    head = capture(["git", "-C", directory, "rev-parse", "HEAD"])
    if head == pin:
        if name == "libjxl" and not _check_libjxl(directory, check):
            return False
        print(f"{name}: ok")
        return True
    if check or not force:
        print(
            f"{name}: HEAD {head or 'unknown'} != {pin}; "
            "run with --force to re-checkout"
        )
        return False
    run(["git", "-C", directory, "fetch", "--tags", "origin"])
    run(["git", "-C", directory, "checkout", pin])
    if name == "libjxl":
        init_submodules(directory)
    print(f"{name}: checked out {pin}")
    return True


def _checkout(
    name: str, url: str, directory: str, pin: str, check: bool, force: bool
) -> bool:
    if os.path.isdir(directory) and is_git(directory):
        return _check_existing(name, directory, pin, check, force)
    if os.path.exists(directory):
        print(f"{name}: unmanaged checkout at {directory}; remove it and rerun")
        return False
    if check:
        print(f"{name}: missing; run without --check to clone it")
        return False
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    if name == "libjxl":
        run(
            [
                "git", "clone", "--filter=blob:none", "--no-checkout", url,
                directory
            ]
        )
        run(["git", "-C", directory, "sparse-checkout", "set", *LIBJXL_SPARSE])
        run(["git", "-C", directory, "checkout", pin])
        init_submodules(directory)
    else:
        run(["git", "-c", "advice.detachedHead=false", "clone", url, directory])
        run(["git", "-C", directory, "checkout", pin])
    print(f"{name}: cloned {pin}")
    return True


def check_state(
    name: str, url: str, directory: str, pin: str, check: bool, force: bool,
    status: set[str], mismatches: list[str]
) -> None:
    """Verify or prepare one checkout and record any reference mismatch.

    :param name: Reference name from the lock file.
    :param url: Git repository URL.
    :param directory: Expected local checkout directory.
    :param pin: Required Git revision.
    :param check: Whether to verify without making changes.
    :param force: Whether to update a clean checkout at the wrong revision.
    :param status: Shared set of failure markers.
    :param mismatches: Ordered names of references that fail verification.
    :raises SystemExit: If a Git command cannot be completed.
    """
    if not _checkout(name, url, directory, pin, check, force):
        mismatches.append(name)
        status.add("failed")


def _parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="verify selected inputs offline without downloading"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-checkout pinned commits in existing Git clones"
    )
    parser.add_argument(
        "--only", default=None, metavar="NAMES",
        help="comma-separated checkout names and/or cid22"
    )
    return parser, parser.parse_args()


def _select(
    parser: argparse.ArgumentParser, only: str | None,
    rows: list[tuple[str, str, str]]
) -> tuple[list[tuple[str, str, str]], bool]:
    if only is None:
        return rows, True
    selected = {name.strip() for name in only.split(",") if name.strip()}
    known = {name for name, _, _ in rows} | {"cid22"}
    if not selected or selected - known:
        parser.error(
            "--only needs known checkout names or cid22; choose from: " +
            ", ".join([name for name, _, _ in rows] + ["cid22"])
        )
    return [row for row in rows if row[0] in selected], "cid22" in selected


def _check_cid22(check: bool, status: set[str], mismatches: list[str]) -> None:
    command = [
        sys.executable,
        os.path.join(REPO_ROOT, "validation", "fetch_cid22.py")
    ]
    if check:
        command.append("--check")
    try:
        run(command)
    except SystemExit:
        mismatches.append("cid22")
        status.add("failed")


def main() -> int:
    """Verify or fetch the selected reference inputs.

    :returns: One if any selected input fails verification, otherwise zero.
    :raises SystemExit: If the lock file or required Git commands are
        unavailable.
    """
    parser, args = _parse_args()
    if not os.path.isfile(LOCK):
        raise SystemExit(f"missing {LOCK}")

    rows, include_cid22 = _select(parser, args.only, parse_lock(LOCK))
    status = set()
    mismatches = []
    for name, url, pin in rows:
        check_state(
            name, url, os.path.join(REFERENCE_DIR, name), pin, args.check,
            args.force, status, mismatches
        )
    if include_cid22:
        _check_cid22(args.check, status, mismatches)
    if status:
        print(f"reference mismatch: {' '.join(mismatches)}", file=sys.stderr)
        return 1
    print(
        "all selected references and CID22 ok"
        if include_cid22 else "all references ok"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
