# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Capture benchmark input, build and device provenance."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Sequence

from bench import inmemory_report as reports
from bench.inmemory_measure import BenchEntry
from bench.inmemory_measure import PathLike
from tools.build_paths import relocate_build_path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REFERENCE_NAMES = {
    "cloudinary": "ssimulacra2",
    "libjxl": "libjxl",
    "fssimu2": "fssimu2",
    "vship-cuda": "Vship",
    "vship-vulkan": "Vship",
    "vship-fast-vulkan": "Vship-fast",
}
SHARED_LIBRARIES = {
    "fssimu2": "build/fssimu2-build/lib/libssimu2.so",
    "vship-cuda": "build/vship-build-cuda/libvship.so",
    "vship-vulkan": "build/vship-build-vulkan/libvship.so",
    "vship-fast-vulkan": "build/vship-fast-build-vulkan/libvship.so",
}


def command_output(
    argv: Sequence[str], cwd: PathLike | None = None
) -> str | None:
    """Read trimmed stdout from an optional environment probe.

    :param argv: Probe executable and arguments.
    :param cwd: Optional working directory.
    :returns: Probe output, or None after an error or timeout.
    """
    try:
        result = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=15,
            check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def file_sha256(path: PathLike) -> str | None:
    """Fingerprint a benchmark input or binary when it is readable.

    :param path: File to hash.
    :returns: SHA-256 in hexadecimal, or None if the file cannot be read.
    """
    try:
        with open(path, "rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    except OSError:
        return None


def _git_at(path: str, *args: str) -> str | None:
    return command_output(["git", "-C", path, *args])


def _reference_environment(
    name: str, build_dir: PathLike
) -> reports.JsonObject:
    checkout = REFERENCE_NAMES[name]
    path = os.path.join(REPO_ROOT, ".reference", checkout)
    stamp = os.path.join(
        REPO_ROOT, build_dir, "reference-fingerprints", name + ".json"
    )
    try:
        with open(stamp, encoding="utf-8") as source:
            fingerprint = json.load(source)
    except (OSError, ValueError):
        fingerprint = None
    source_status = _git_at(
        path, "status", "--porcelain", "--ignored", "--ignore-submodules=none"
    )
    record = {
        "checkout": checkout,
        "revision": _git_at(path, "rev-parse", "HEAD"),
        "dirty": None if source_status is None else bool(source_status),
        "submodules": _git_at(path, "submodule", "status", "--recursive"),
        "build_fingerprint": fingerprint,
    }
    if name in SHARED_LIBRARIES:
        library = relocate_build_path(
            SHARED_LIBRARIES[name], os.path.join(REPO_ROOT, build_dir)
        )
        record["runtime_library"] = {
            "path": library,
            "sha256": file_sha256(library)
        }
    return record


def _compiler_path(build_dir: PathLike) -> str | None:
    prefixes = ("CMAKE_CXX_COMPILER:FILEPATH=", "CMAKE_CXX_COMPILER:STRING=")
    try:
        with open(os.path.join(REPO_ROOT, build_dir,
                               "CMakeCache.txt")) as source:
            return next(
                (
                    line.partition("=")[2].strip()
                    for line in source if line.startswith(prefixes)
                ), None
            )
    except OSError:
        return None


def capture_environment(
    pairs: list[reports.Pair], entries: list[BenchEntry],
    devices: reports.JsonObject, build_dir: PathLike = "build"
) -> reports.JsonObject:
    """Capture benchmark provenance from locally available environment data.

    :param pairs: Ordered image pairs whose input files will be fingerprinted.
    :param entries: Selected comparison drivers.
    :param devices: Device matching report for those drivers.
    :param build_dir: Directory containing candidate outputs and reference
        builds.
    :returns: Revisions, input hashes, binaries, compiler and device metadata.
    """
    project_status = _git_at(
        REPO_ROOT, "status", "--porcelain", "--untracked-files=normal"
    )
    project = {
        "revision": _git_at(REPO_ROOT, "rev-parse", "HEAD"),
        "dirty": None if project_status is None else bool(project_status),
    }
    references = {}
    executables = {}
    for entry in entries:
        name, driver = entry["name"], entry["driver"]
        executables[name] = {
            "path": os.path.abspath(driver),
            "sha256": file_sha256(driver)
        }
        if name in REFERENCE_NAMES:
            references[name] = _reference_environment(name, build_dir)
    paths = dict.fromkeys(path for pair in pairs for path in pair)
    images = {path: {"sha256": file_sha256(path)} for path in paths}
    compiler = _compiler_path(build_dir)
    return {
        "project":
        project,
        "references":
        references,
        "devices":
        devices,
        "executables":
        executables,
        "inputs":
        images,
        "platform":
        platform.platform(),
        "python":
        sys.version,
        "compiler": {
            "path": compiler,
            "version":
            command_output([compiler, "--version"]) if compiler else None
        },
        # This is an inventory, not proof of which device an arm selected.
        "nvidia_inventory":
        command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,pci.bus_id",
                "--format=csv,noheader"
            ]
        ),
    }


def _probe_device(entry: BenchEntry) -> object:
    output = command_output([entry["driver"], "--device-info"])
    try:
        return json.loads(output) if output else None
    except ValueError:
        return None


def _select_vship_device(
    entry: BenchEntry, device_name: str, allow_unverified: bool
) -> tuple[list[reports.JsonObject], int, bool]:
    available = _probe_device(entry)
    if not isinstance(available, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("name"), str)
            or not isinstance(item.get("id"), int) for item in available):
        raise SystemExit(
            f"cannot inspect {entry['name']} devices; rebuild its driver"
        )
    matches = [item for item in available if item["name"] == device_name]
    if len(matches) == 1:
        return available, matches[0]["id"], True
    reason = (
        "no matching device name"
        if not matches else "multiple devices share the selected name"
    )
    if not allow_unverified:
        raise SystemExit(
            f"{entry['name']}: {reason}; device identity cannot be "
            "verified. Use --allow-unverified-devices only after "
            "checking the devices manually."
        )
    print(
        f"warning: {entry['name']}: {reason}; using device 0 without "
        "verified identity", file=sys.stderr
    )
    return available, 0, False


def _apply_vship_selection(
    entries: list[BenchEntry], name: str, selected: int
) -> None:
    # Use the Vship backend's own index for every metric that uses this driver.
    for entry in entries:
        if entry["name"] == name:
            entry["args"] = [*entry.get("args", []), "--gpu", str(selected)]


def probe_devices(
    entries: list[BenchEntry], allow_unverified: bool = False
) -> reports.JsonObject:
    """Match Vship device IDs to the selected Vulkan device by name.

    :param entries: Selected drivers whose Vship arguments will be updated.
    :param allow_unverified: Whether to fall back to device zero on an ambiguous
        match.
    :returns: Report of the device names and IDs selected for each arm.
    :raises SystemExit: If a probe fails or a name match is required but
        unavailable.
    """
    entries_by_name = {entry["name"]: entry for entry in entries}
    ours = next(
        (
            entry for name, entry in entries_by_name.items()
            if name in reports.OURS_ENTRIES
        ), None
    )
    vships = [
        entry for name, entry in entries_by_name.items()
        if name.startswith("vship-")
    ]
    if not ours or not vships:
        return {"status": "not_compared"}
    device = _probe_device(ours)
    if not isinstance(device, dict) or not isinstance(device.get("name"), str):
        raise SystemExit(
            "cannot inspect our benchmark device; rebuild the benchmark"
        )
    report: reports.JsonObject = {
        "ours": device,
        "vship": {},
        "status": "matched_by_name_only"
    }
    for entry in vships:
        available, selected, verified = _select_vship_device(
            entry, device["name"], allow_unverified
        )
        if not verified:
            report["status"] = "unverified"
        _apply_vship_selection(entries, entry["name"], selected)
        report["vship"][entry["name"]] = {
            "devices": available,
            "selected_id": selected
        }
    print(
        "Device matching uses names: Vship does not expose UUIDs; physical "
        "device identity is not proven by matching names.", file=sys.stderr
    )
    return report
