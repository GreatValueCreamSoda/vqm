#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Check maintained formatting and optionally lint compiled production C++."""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("src", "include", "cli", "tests", "bench", "tools")
FORMAT_SUFFIXES = {".cc", ".h", ".cu", ".comp", ".glsl"}
FORMAT_MAJOR = 23


def maintained(path: Path) -> bool:
    """Report whether a resolved source path belongs to maintained code.

    :param path: Absolute source path.
    :returns: Whether the path is under a maintained source directory.
    """
    return any(path.is_relative_to(ROOT / directory) for directory in SOURCE_DIRS)


def check_format(executable: str) -> None:
    """Check all maintained C++, headers and shaders with the pinned formatter.

    :param executable: clang-format executable name or path.
    :raises ValueError: If the executable has the wrong major version.
    :raises subprocess.CalledProcessError: If any maintained file is unformatted.
    """
    version = subprocess.check_output([executable, "--version"], text=True)
    if not re.search(rf"clang-format version {FORMAT_MAJOR}\.", version):
        raise ValueError(
            f"clang-format {FORMAT_MAJOR} is required; found {version.strip()}"
        )
    sources = sorted(
        path for directory in SOURCE_DIRS for path in (ROOT / directory).rglob("*")
        if path.suffix in FORMAT_SUFFIXES and path.is_file()
    )
    # Bound the command length on platforms with smaller process limits.
    for start in range(0, len(sources), 50):
        subprocess.run(
            [
                executable, "--dry-run", "--Werror",
                *(str(path) for path in sources[start:start + 50])
            ], cwd=ROOT, check=True
        )
    print(f"formatting: {len(sources)} maintained files checked", flush=True)


def compiled_sources(database: Path, include_tests: bool) -> set[Path]:
    """Select distinct maintained C++ files from one compile database.

    :param database: CMake compile_commands.json path.
    :param include_tests: Whether to include test and oracle translation units.
    :returns: Absolute compiled source paths.
    :raises ValueError: If the database contains no selected source files.
    """
    entries = json.loads(database.read_text())
    sources = set()
    for entry in entries:
        path = Path(entry["file"])
        if not path.is_absolute():
            path = Path(entry["directory"]) / path
        path = path.resolve()
        is_test = path.is_relative_to(ROOT / "tests")
        if maintained(path) and path.suffix == ".cc" and (include_tests or not is_test):
            sources.add(path)
    if not sources:
        raise ValueError(f"no maintained C++ files in {database}")
    return sources


def print_lint_result(result: subprocess.CompletedProcess[str]) -> bool:
    """Print one lint process's captured output in source order.

    :param result: Completed clang-tidy subprocess.
    :returns: Whether the process failed.
    """
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    return result.returncode != 0


def check_lint(
    executable: str, build_dir: Path, jobs: int, include_tests: bool = False
) -> None:
    """Lint selected compiled C++ files with bounded parallel workers.

    :param executable: clang-tidy executable name or path.
    :param build_dir: Configured CMake build directory.
    :param jobs: Maximum concurrent lint processes.
    :param include_tests: Whether to include test and oracle translation units.
    :raises ValueError: If source selection is empty or linting fails.
    """
    sources = compiled_sources(build_dir / "compile_commands.json", include_tests)

    def run(path: Path) -> tuple[Path, subprocess.CompletedProcess[str]]:
        result = subprocess.run(
            [
                executable, "-p",
                str(build_dir),
                str(path), "--quiet", "--warnings-as-errors=readability-function-size"
            ], cwd=ROOT, capture_output=True, text=True,
        )
        return path, result

    with ThreadPoolExecutor(max_workers=jobs) as workers:
        failures = [
            path for path, result in workers.map(run, sorted(sources))
            if print_lint_result(result)
        ]
    if failures:
        raise ValueError(
            f"clang-tidy failed for {len(failures)} of {len(sources)} C++ files"
        )
    print(f"lint: {len(sources)} compiled C++ files checked", flush=True)


def main() -> int:
    """Run the requested maintained formatting and compiled C++ lint gates.

    :returns: Zero after every selected quality check passes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    lint = parser.add_mutually_exclusive_group()
    lint.add_argument(
        "--lint", action="store_true", help="also run recursive clang-tidy"
    )
    lint.add_argument(
        "--lint-only", action="store_true", help="run only recursive clang-tidy"
    )
    parser.add_argument(
        "--include-tests", action="store_true",
        help="also lint test matrices and reference oracles"
    )
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--format-tool", default="clang-format")
    parser.add_argument("--lint-tool", default="clang-tidy")
    parser.add_argument(
        "--jobs", type=int, default=min(4,
                                        os.cpu_count() or 1),
        help="concurrent clang-tidy processes (default: at most 4)"
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.include_tests and not (args.lint or args.lint_only):
        parser.error("--include-tests requires --lint or --lint-only")
    try:
        if not args.lint_only:
            check_format(args.format_tool)
        if args.lint or args.lint_only:
            check_lint(
                args.lint_tool, args.build_dir.resolve(), args.jobs, args.include_tests
            )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"quality check failed: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
