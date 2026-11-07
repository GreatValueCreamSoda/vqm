#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Build and optionally run the standalone cooperative-matrix correctness probe.

There are no performance timers. The default three shape variants match the
FP16-input/FP32-result shapes observed on the study's RTX 5070. The executable
runs every matching shape advertised by its selected device; unsupported probe
requirements fail explicitly rather than silently substituting scalar math.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
SHAPES = ((16, 16, 16), (16, 8, 16), (16, 8, 8))


def sha256(path: Path) -> str:
    """Hash a probe input or build output.

    :param path: File to read, hash or update.
    :returns: Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(command: list[str | Path], commands: list[list[str]]) -> None:
    """Record and execute one build command.

    :param command: Executable and arguments in invocation order.
    :param commands: Provenance log receiving the command.
    :raises subprocess.CalledProcessError: If the command exits unsuccessfully.
    """
    commands.append([str(part) for part in command])
    subprocess.run(command, cwd=ROOT, check=True)


def _run_probe(
    executable: Path, build: Path, out: Path, validation: bool,
    metadata: dict[str, object], commands: list[list[str]]
) -> None:
    """Execute the probe and require a valid result after saving diagnostics.

    :param executable: Built probe executable.
    :param build: Directory containing its SPIR-V modules.
    :param out: Report directory.
    :param validation: Whether to enable Khronos synchronization validation.
    :param metadata: Build and test provenance to write.
    :param commands: Command log stored in the provenance.
    :raises RuntimeError: If correctness or requested validation fails.
    """
    env = os.environ.copy()
    if validation:
        env.update(
            VK_INSTANCE_LAYERS="VK_LAYER_KHRONOS_validation", VK_LAYER_VALIDATE_SYNC="1"
        )
    command = [str(executable), str(build)]
    commands.append(command)
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    (out / "results.json").write_text(result.stdout)
    (out / "stderr.log").write_text(result.stderr)
    metadata["exit_code"] = result.returncode
    (out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result.check_returncode()
    if re.search(r"VUID-|Validation Error|SYNC-HAZARD", result.stderr, re.IGNORECASE):
        raise RuntimeError("validation reported an error; inspect stderr.log")
    data = json.loads(result.stdout)
    if not data["passed"] or not data["tested_cases"]:
        raise RuntimeError("matrix correctness probe did not pass")
    print(f"{data['tested_cases']} matrix cases passed on {data['device']}")


def main() -> None:
    """Build the matrix probe and optionally run its correctness checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir", type=Path, default=ROOT / "build/cooperative-matrix-probe"
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "artifacts/reports/vmaf-cooperative-matrix/platform"
    )
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument(
        "--validation", action="store_true",
        help="require a clean run with Khronos validation and sync validation"
    )
    parser.add_argument(
        "--build-only", action="store_true",
        help="compile and validate SPIR-V without creating a Vulkan device"
    )
    args = parser.parse_args()
    build = args.build_dir.resolve()
    out = args.out_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    commands = []
    modules = []
    for m, n, k in SHAPES:
        target = build / f"{m}-{n}-{k}.spv"
        execute(
            [
                "glslc", "--target-env=vulkan1.1", "--target-spv=spv1.3", "-O",
                f"-DMATRIX_M={m}", f"-DMATRIX_N={n}", f"-DMATRIX_K={k}",
                SOURCE / "platform.comp", "-o", target
            ], commands
        )
        execute(["spirv-val", "--target-env", "vulkan1.1", target], commands)
        disassembly = subprocess.check_output(["spirv-dis", target], text=True)
        (out / f"{m}-{n}-{k}.spvasm").write_text(disassembly)
        modules.append(
            {
                "shape": [m, n, k],
                "sha256": sha256(target),
                "has_cooperative_matrix_multiply": "OpCooperativeMatrixMulAddKHR"
                in disassembly,
                "has_vulkan_memory_model": "OpMemoryModel Logical Vulkan" in disassembly
            }
        )
    executable = build / "platform_probe"
    execute(
        [
            args.cxx, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic",
            "-Wno-missing-field-initializers", SOURCE / "platform_probe.cc", "-lvulkan",
            "-o", executable
        ], commands
    )
    metadata = {
        "kind":
        "correctness-only; no performance timers",
        "sources": {
            name: sha256(SOURCE / name)
            for name in ("platform_probe.cc", "platform.comp", "run_platform.py")
        },
        "compiler":
        subprocess.check_output([args.cxx, "--version"], text=True).splitlines()[0],
        "glslc":
        subprocess.check_output(["glslc", "--version"], text=True).splitlines(),
        "commands":
        commands,
        "executable_sha256":
        sha256(executable),
        "modules":
        modules,
        "validation_requested":
        args.validation,
        "executed":
        not args.build_only,
    }
    if not args.build_only:
        _run_probe(executable, build, out, args.validation, metadata, commands)
    else:
        (out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("Compiled probe and validated three matrix SPIR-V modules; GPU not used")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
