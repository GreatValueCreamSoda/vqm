#!/usr/bin/env python3
"""Compile and validate the exact CAMBI scan shortcut without GPU execution."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from pool_zero_sum import POOL, ROOT, install


def run(command: list[str | Path]) -> str:
    """Run one compiler or SPIR-V inspection command.

    :param command: Executable and arguments in invocation order.
    :returns: Standard output from the successful command.
    :raises subprocess.CalledProcessError: If the command exits unsuccessfully.
    """
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def sha256(path: Path) -> str:
    """Hash a shader or compiled module.

    :param path: File to read, hash or update.
    :returns: Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_guards(source: Path) -> dict[str, bool]:
    """Check installer rejection for a repeated install and the maintained tree.

    :param source: Isolated copy after one successful install.
    :returns: Success flags for both rejection cases.
    :raises AssertionError: If the installer accepts an invalid target.
    """
    guards = {}
    for label, rejected_source in (("reapply", source), ("maintained_checkout", ROOT)):
        try:
            install(rejected_source)
        except ValueError:
            guards[f"{label}_rejected"] = True
        else:
            raise AssertionError(f"installer accepted {label}")
    return guards


def check(out: Path) -> dict[str, object]:
    """Compile and compare the baseline and candidate shader contracts.

    :param out: Directory for SPIR-V outputs and the report.
    :returns: Shader provenance, SPIR-V validation and installer-guard outcomes.
    """
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": "compiler/CPU only; no GPU correctness or timing",
        "tools": {},
        "modules": []
    }
    for tool in ("glslc", "spirv-val", "clang-format"):
        result["tools"][tool] = run([tool, "--version"]).strip()
    with tempfile.TemporaryDirectory(prefix="vqm-pool-zero-") as temporary:
        source = Path(temporary)
        shader_dir = source / POOL
        shutil.copytree(ROOT / POOL, shader_dir)
        baseline = {
            name: sha256(shader_dir / f"{name}.comp")
            for name in ("vmaf_cambi_pool_select", "vmaf_cambi_pool_partial")
        }
        install(source)
        for name, baseline_sha in baseline.items():
            shader = shader_dir / f"{name}.comp"
            run(["clang-format", "-i", str(shader)])
            shutil.copy2(shader, out / shader.name)
            binary = out / f"{name}.spv"
            command = [
                "glslc", "-O", "--target-env=vulkan1.1", "-I",
                str(ROOT / "src/vulkan/shaders"), "-I",
                str(shader_dir),
                str(shader), "-o",
                str(binary)
            ]
            run(command)
            run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
            result["modules"].append(
                {
                    "name": name,
                    "baseline_source_sha256": baseline_sha,
                    "source_sha256": sha256(shader),
                    "spv_sha256": sha256(binary),
                    "compile_command": command,
                    "spirv_val_passed": True
                }
            )
        result.update(_check_guards(source))
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Compile the shortcut shader variants and verify installer guards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "artifacts/reports/vmaf-v1-headroom/pool-zero-sum/compiler"
    )
    args = parser.parse_args()
    check(args.out.resolve())
    print(
        "Both changed shaders compile and pass spirv-val; installer guards pass; no GPU work."
    )


if __name__ == "__main__":
    main()
