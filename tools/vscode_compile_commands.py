#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Write one compile_commands.json for clangd, covering every C++ file.

CMake's own database (build/compile_commands.json) describes the library, the
CLI, the tests and every source under bench/driver/. It cannot describe the reference harnesses:
they are compiled inside each reference's CMake project, and bench/vship_bench.cc
and bench/inmemory/fssimu2_bench.cc are compiled by hand. clangd then guesses a
command from a sibling file, which is wrong enough to paint the editor red.

This script merges the reference builds' own databases (they are configured
with CMAKE_EXPORT_COMPILE_COMMANDS=ON by tools/build_references.py) for the
harness sources only, synthesizes the two hand-compiled entries, and writes
compile_commands.json at the repository root, where clangd looks. Re-run it
after configure or after building references; tools/build_references.py does.
"""

import json
import os
import sys
from typing import NotRequired, TypedDict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(REPO_ROOT, "compile_commands.json")
MAIN_DATABASE = os.path.join(REPO_ROOT, "build", "compile_commands.json")
REFERENCE_DATABASES = (
    os.path.join(REPO_ROOT, "build", "reference-build", "compile_commands.json"),
    os.path.join(REPO_ROOT, "build", "libjxl-build", "compile_commands.json"),
)
CXX = os.environ.get("CXX", "g++")


class CompileEntry(TypedDict):
    """Describe one translation unit in a CMake compilation database."""

    directory: str
    file: str
    arguments: NotRequired[list[str]]
    command: NotRequired[str]
    output: NotRequired[str]


def load(path: str) -> list[CompileEntry]:
    """Load entries from a CMake compilation database.

    :param path: Database path.
    :returns: Compilation entries in database order.
    :raises OSError: If the file cannot be opened.
    :raises json.JSONDecodeError: If the database is malformed.
    """
    with open(path) as f:
        return json.load(f)


def entry(file: str, arguments: list[str]) -> CompileEntry:
    """Describe a manually compiled driver.

    :param file: Source file relative to the repository root.
    :param arguments: Compiler executable and arguments.
    :returns: Entry with an absolute source path.
    """
    return {
        "directory": REPO_ROOT,
        "file": os.path.join(REPO_ROOT, file),
        "arguments": arguments,
    }


def synthesized() -> list[CompileEntry]:
    """Describe the hand-compiled drivers with the flags that build them.

    :returns: The fssimu2 entry and the Vship entry when its sources are available.
    """
    entries = [
        entry(
            "bench/inmemory/fssimu2_bench.cc", [
                CXX, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic",
                f"-I{REPO_ROOT}/src", f"-I{REPO_ROOT}/tools/image_io",
                f"-I{REPO_ROOT}/include",
                f"-I{REPO_ROOT}/.reference/fssimu2/src/include", "-c",
            ]
        )
    ]
    for backend in ("Vulkan", "Cuda"):
        source = os.path.join(REPO_ROOT, "build", f"Vship-src-{backend}", "src")
        if os.path.isdir(source):
            entries.append(
                entry(
                    "bench/vship_bench.cc", [
                        CXX, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic",
                        f"-I{source}", f"-I{REPO_ROOT}/src",
                        f"-I{REPO_ROOT}/tools/image_io", f"-I{REPO_ROOT}/include", "-c",
                    ]
                )
            )
            break
    return entries


def merge() -> int:
    """Merge project and reference entries into the root clangd database.

    :returns: Zero after writing the database.
    :raises OSError: If a database cannot be read or written.
    :raises json.JSONDecodeError: If an input database is malformed.
    """
    entries = []
    if os.path.isfile(MAIN_DATABASE):
        entries.extend(load(MAIN_DATABASE))
    for path in REFERENCE_DATABASES:
        if not os.path.isfile(path):
            continue
        entries.extend(
            item for item in load(path) if "/bench/inmemory/" in item.get("file", "")
        )
    entries.extend(synthesized())

    deduplicated = {}
    for item in entries:
        deduplicated[os.path.realpath(item["file"])] = item
    ordered = list(deduplicated.values())
    with open(OUTPUT, "w") as f:
        json.dump(ordered, f, indent=1)
    print(f"wrote {OUTPUT} ({len(ordered)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(merge())
