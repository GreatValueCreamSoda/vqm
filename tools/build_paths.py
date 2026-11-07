# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Resolve generated build outputs consistently across repository tools."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_directory(value: str | Path = "build") -> Path:
    """Resolve a build directory relative to the repository root.

    :param value: Absolute path or repository-relative directory.
    :returns: Absolute directory for generated build outputs.
    """
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def relocate_build_path(value: str, directory: str | Path = "build") -> str:
    """Relocate default build paths while preserving explicit paths.

    :param value: Manifest path or command argument.
    :param directory: Selected build directory.
    :returns: Relocated build path, or the original non-build argument.
    """
    path = Path(value)
    if not path.parts or path.parts[0] != "build":
        return value
    return str(build_directory(directory).joinpath(*path.parts[1:]))
