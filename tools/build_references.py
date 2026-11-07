#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Build every reference implementation this repository measures against.

Each pinned checkout in .reference/ is copied under the selected --build-dir
(default: build/) first (libjxl
excepted only in that its copy is the sparse tree), so nothing is ever written
into .reference/. The in-memory harness for each reference is compiled here as
well.

  cloudinary         .reference/ssimulacra2 -> build/reference-build/
                     ssimulacra2, ss2_cloudinary_bench
  libjxl             .reference/libjxl      -> build/libjxl-build/
                     ssimulacra2, butteraugli_main, ss2_libjxl_bench
  fssimu2            .reference/fssimu2     -> build/fssimu2-build/
                     fssimu2 (Zig), ssimu2_bench (C API driver)
  vship-cuda         .reference/Vship       -> build/vship-build-cuda/
  vship-vulkan       .reference/Vship       -> build/vship-build-vulkan/
  vship-fast-vulkan  .reference/Vship-fast  -> build/vship-fast-build-vulkan/
  vmaf               .reference/vmaf        -> build/vmaf-build/ (CPU oracle)
  cvvdp              .reference/ColorVideoVDP -> .venv/
                     isolated Python 3.12 CPU original runtime (requires uv)

Reference build trees are reused only while their source and toolchain
fingerprints match. --force removes a step's build trees first. Missing
reference checkouts are reported with the fetch command.
"""

import argparse
import contextlib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.build_paths import build_directory

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL_STEPS = (
    "cloudinary", "libjxl", "fssimu2", "vship-cuda", "vship-vulkan",
    "vship-fast-vulkan", "vmaf", "cvvdp"
)
STEP_CHECKOUTS = {
    "cloudinary": "ssimulacra2",
    "libjxl": "libjxl",
    "fssimu2": "fssimu2",
    "vship-cuda": "Vship",
    "vship-vulkan": "Vship",
    "vship-fast-vulkan": "Vship-fast",
    "vmaf": "vmaf",
    "cvvdp": "ColorVideoVDP"
}
BUILD_DIR = build_directory()
FINGERPRINT_DIR = str(BUILD_DIR / "reference-fingerprints")
CVVDP_ENV = ".venv"
CVVDP_REQUIREMENTS = "tools/cvvdp-reference-requirements.txt"
BUILD_ENV = (
    "PATH", "CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "NVCC",
    "CUDA_HOME", "CUDA_PATH", "CMAKE_PREFIX_PATH", "CMAKE_TOOLCHAIN_FILE",
    "PKG_CONFIG_PATH", "PKG_CONFIG_LIBDIR", "VULKAN_SDK", "ZIG_GLOBAL_CACHE_DIR"
)


def capture(argv: Sequence[str]) -> str | None:
    """Capture trimmed stdout from an optional build tool.

    :param argv: Executable and arguments.
    :returns: Tool output, or None if the command cannot run successfully.
    """
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def reference_fingerprint(step: str) -> dict[str, object]:
    """Describe inputs known to affect the generated reference build.

    :param step: Selected reference build step.
    :returns: Source revision, recipe and toolchain provenance.
    :raises SystemExit: If Git cannot inspect the pinned source.
    """
    checkout = os.path.join(".reference", STEP_CHECKOUTS[step])
    revision = capture(["git", "-C", checkout, "rev-parse", "HEAD"])
    submodules = capture(
        ["git", "-C", checkout, "submodule", "status", "--recursive"]
    )
    if revision is None or submodules is None:
        raise SystemExit(f"cannot fingerprint reference {checkout}")
    compiler = os.environ.get("CXX", "g++")
    c_compiler = os.environ.get("CC", "cc")
    commands = {
        "cxx": compiler,
        "cc": c_compiler,
        "cmake": "cmake",
        "make": "make",
        "ninja": "ninja"
    }
    if step == "fssimu2":
        commands["zig"] = "zig"
    if step == "vship-cuda":
        commands["nvcc"] = os.environ.get("NVCC", "nvcc")
    if step == "vmaf":
        commands["meson"] = "meson"
    if step == "cvvdp":
        commands["uv"] = "uv"
    with open(__file__, "rb") as script:
        recipe_hash = hashlib.sha256(script.read()).hexdigest()
    fingerprint = {
        "schema": 1,
        "step": step,
        "revision": revision,
        "submodules": submodules,
        "recipe_sha256": recipe_hash,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "environment": {
            name: os.environ.get(name)
            for name in BUILD_ENV
        },
        "tools": {
            name: {
                "command": command,
                "path": shutil.which(command),
                "version": capture([command, "--version"])
            }
            for name, command in commands.items()
        }
    }
    if step == "cvvdp":
        fingerprint["runtime"] = _cvvdp_fingerprint()
    return fingerprint


def _cvvdp_fingerprint() -> dict[str, object]:
    """Fingerprint the locked CPU environment without importing the original.

    :returns: Requested interpreter, actual environment versions and lock
        digest.
    """
    python = os.path.join(REPO_ROOT, CVVDP_ENV, "bin", "python")
    with open(CVVDP_REQUIREMENTS, "rb") as source:
        requirements_hash = hashlib.sha256(source.read()).hexdigest()
    return {
        "requested_python": "3.12",
        "python": python,
        "python_version": capture([python, "--version"]),
        "requirements_sha256": requirements_hash,
        "installed_packages":
        capture(["uv", "pip", "freeze", "--python", python]),
    }


def build_fingerprinted(
    step: str, jobs: int, force: bool, builder: Callable[[int, bool], None]
) -> None:
    """Invalidate generated trees when a reference's known inputs change.

    :param step: Selected reference build step.
    :param jobs: Build concurrency passed to the builder.
    :param force: Whether to rebuild regardless of the fingerprint.
    :param builder: Build function receiving concurrency and the rebuild flag.
    :raises OSError: If the fingerprint cannot be saved.
    :raises SystemExit: If the pinned source cannot be verified or built.
    """
    fingerprint = reference_fingerprint(step)
    os.makedirs(FINGERPRINT_DIR, exist_ok=True)
    stamp = os.path.join(FINGERPRINT_DIR, f"{step}.json")
    try:
        with open(stamp, encoding="utf-8") as source:
            previous = json.load(source)
    except (OSError, ValueError):
        previous = None
    rebuild = force or previous != fingerprint
    if rebuild:
        print(
            f"{step}: build fingerprint changed or missing; rebuilding",
            flush=True
        )
    # The stamp must never survive a failed rebuild, even an incremental one.
    with contextlib.suppress(FileNotFoundError):
        os.remove(stamp)
    builder(jobs, rebuild)
    if step == "cvvdp":
        # The first call creates the interpreter whose actual versions we pin.
        fingerprint = reference_fingerprint(step)
    fd, temporary = tempfile.mkstemp(prefix=f".{step}-", dir=FINGERPRINT_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            json.dump(fingerprint, target, indent=2, sort_keys=True)
            target.write("\n")
        os.replace(temporary, stamp)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def run(argv: Sequence[str], cwd: str | None = None) -> None:
    """Run a required build command.

    :param argv: Executable and arguments.
    :param cwd: Optional working directory.
    :raises SystemExit: If the executable is absent or exits unsuccessfully.
    """
    try:
        subprocess.run(argv, cwd=cwd, check=True)
    except FileNotFoundError:
        raise SystemExit(f"{argv[0]} is required") from None
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None


def need_build(artifact: str, force: bool) -> bool:
    """Check whether a requested build output is missing or obsolete.

    :param artifact: Expected output path.
    :param force: Whether to ignore the existing output.
    :returns: Whether a rebuild is needed.
    """
    return force or not os.path.isfile(artifact)


def require_reference(name: str) -> None:
    """Require the pinned checkout for a build.

    :param name: Reference name in the lock file.
    :raises SystemExit: If the checkout is absent.
    """
    if not os.path.isdir(os.path.join(".reference", name)):
        raise SystemExit(
            f"missing .reference/{name}; run "
            f"python3 tools/fetch_references.py"
        )


def copy_reference(source: str, destination: str) -> None:
    """Copy a reference under build/ without its Git object store.

    :param source: Pinned checkout directory.
    :param destination: Scratch source tree for the build.
    :raises OSError: If the checkout cannot be copied.
    """
    if os.path.isdir(destination):
        return
    os.makedirs(destination, exist_ok=True)
    if shutil.which("rsync"):
        run(["rsync", "-a", "--exclude=.git", source + "/", destination + "/"])
    else:
        shutil.copytree(
            source, destination, dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git")
        )


def cmake_configure(source: str, build: str, extra: Sequence[str] = ()) -> None:
    """Configure a copied reference with Ninja in release mode.

    :param source: Copied source tree.
    :param build: Destination build directory.
    :param extra: Additional CMake options.
    :raises SystemExit: If CMake fails.
    """
    run(
        [
            "cmake", "-S", source, "-B", build, "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            *extra
        ]
    )


def append_harness_target(path: str, block: str) -> None:
    """Add this project's benchmark target to a copied CMake file once.

    :param path: CMakeLists.txt in the copied reference source tree.
    :param block: Target definition to append.
    :raises OSError: If the build file cannot be read or written.
    """
    with open(path) as f:
        text = f.read()
    if "# ssimulacra2 reference bench" in text:
        return
    with open(path, "a") as f:
        f.write(
            "\n# ssimulacra2 reference bench "
            "(appended by tools/build_references.py)\n"
        )
        f.write(block + "\n")


def cmake_literal(value: str) -> str:
    """Quote a path without CMake expanding spaces, backslashes, or dollars.

    :param value: Literal text to quote.
    :returns: A CMake bracket argument containing the original value.
    """
    equals = ""
    while f"]{equals}]" in value:
        equals += "="
    return f"[{equals}[{value}]{equals}]"


def zig_version() -> tuple[int, ...] | None:
    """Read the numeric components of the installed Zig version.

    :returns: Version components, or None if Zig is unavailable.
    """
    try:
        result = subprocess.run(
            ["zig", "version"], capture_output=True, text=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    parts = []
    for piece in result.stdout.strip().split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def build_cloudinary(jobs: int, force: bool) -> None:
    """Build the Cloudinary scalar metric and its in-memory driver.

    :param jobs: Parallel build job limit.
    :param force: Whether to recreate the copied source and build directories.
    :raises SystemExit: If a required tool or build step fails.
    """
    require_reference("ssimulacra2")
    out = str(BUILD_DIR / "reference-build")
    if not need_build(f"{out}/ss2_cloudinary_bench", force):
        # Let the existing build system track our harness and its headers.
        run(
            [
                "cmake", "--build", out, "--target", "ssimulacra2",
                "ss2_cloudinary_bench", "-j",
                str(jobs)
            ]
        )
        return
    if force:
        shutil.rmtree(str(BUILD_DIR / "reference-src"), ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
    copy_reference(".reference/ssimulacra2", str(BUILD_DIR / "reference-src"))
    append_harness_target(
        str(BUILD_DIR / "reference-src/src/CMakeLists.txt"), f"""\
add_executable(ss2_cloudinary_bench
  {cmake_literal(os.path.join(REPO_ROOT, 'bench/inmemory/cloudinary_bench.cc'))}
  ssimulacra2.cc)
target_compile_features(ss2_cloudinary_bench PRIVATE cxx_std_20)
target_link_libraries(ss2_cloudinary_bench jxl jxl_extras-static hwy)"""
    )
    cmake_configure(
        str(BUILD_DIR / "reference-src/src"), out, [
            "-DJPEGXL_ENABLE_OPENEXR=OFF",
            "-DCMAKE_CXX_FLAGS=-DHWY_COMPILE_ONLY_SCALAR",
        ]
    )
    run(
        [
            "cmake", "--build", out, "--target", "ssimulacra2",
            "ss2_cloudinary_bench", "-j",
            str(jobs)
        ]
    )
    print(
        f"cloudinary: built {out}/ssimulacra2 and "
        f"{out}/ss2_cloudinary_bench"
    )


def build_libjxl(jobs: int, force: bool) -> None:
    """Build the libjxl metric tools and the in-memory comparison driver.

    :param jobs: Parallel build job limit.
    :param force: Whether to recreate the copied source and build directories.
    :raises SystemExit: If a required tool or build step fails.
    """
    require_reference("libjxl")
    out = str(BUILD_DIR / "libjxl-build")
    if not need_build(f"{out}/tools/ss2_libjxl_bench", force):
        run(
            [
                "cmake", "--build", out, "--target", "ssimulacra2",
                "butteraugli_main", "ss2_libjxl_bench", "-j",
                str(jobs)
            ]
        )
        return
    if force:
        shutil.rmtree(str(BUILD_DIR / "libjxl-src"), ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
    copy_reference(".reference/libjxl", str(BUILD_DIR / "libjxl-src"))
    append_harness_target(
        str(BUILD_DIR / "libjxl-src/tools/CMakeLists.txt"), f"""\
add_executable(ss2_libjxl_bench
  {cmake_literal(os.path.join(REPO_ROOT, 'bench/inmemory/libjxl_bench.cc'))}
  ssimulacra2.cc)
target_compile_features(ss2_libjxl_bench PRIVATE cxx_std_20)
target_link_libraries(ss2_libjxl_bench PRIVATE jxl_extras-internal
  jxl_threads jxl_tool jxl_gauss_blur)"""
    )
    cmake_configure(
        str(BUILD_DIR / "libjxl-src"), out, [
            "-DJPEGXL_ENABLE_TOOLS=ON", "-DJPEGXL_ENABLE_DEVTOOLS=ON",
            "-DJPEGXL_ENABLE_BENCHMARK=OFF", "-DJPEGXL_ENABLE_EXAMPLES=OFF",
            "-DJPEGXL_ENABLE_VIEWERS=OFF", "-DBUILD_TESTING=OFF",
            "-DJPEGXL_ENABLE_MANPAGES=OFF", "-DJPEGXL_ENABLE_DOXYGEN=OFF",
            "-DJPEGXL_ENABLE_JNI=OFF", "-DJPEGXL_ENABLE_PLUGINS=OFF",
            "-DJPEGXL_ENABLE_FUZZERS=OFF", "-DJPEGXL_ENABLE_SJPEG=OFF",
            "-DJPEGXL_ENABLE_OPENEXR=OFF",
        ]
    )
    run(
        [
            "cmake", "--build", out, "--target", "ssimulacra2",
            "butteraugli_main", "ss2_libjxl_bench", "-j",
            str(jobs)
        ]
    )
    print(
        f"libjxl: built {out}/tools/{{ssimulacra2,butteraugli_main,"
        f"ss2_libjxl_bench}}"
    )


def build_fssimu2(jobs: int, force: bool) -> None:
    """Build the Zig library and compile its in-memory driver.

    :param jobs: Parallel build job limit.
    :param force: Whether to recreate the copied source and build directories.
    :raises SystemExit: If a required tool or build step fails.
    """
    require_reference("fssimu2")
    out = str(BUILD_DIR / "fssimu2-build")
    source = str(BUILD_DIR / "fssimu2-src")
    have_library = (
        not need_build(f"{out}/bin/ssimu2_bench", force)
        and os.path.isfile(f"{out}/lib/libssimu2.so")
    )
    if force:
        shutil.rmtree(out, ignore_errors=True)
        shutil.rmtree(source, ignore_errors=True)
    copy_reference(".reference/fssimu2", source)
    if force or not have_library:
        version = zig_version()
        if version is None or version < (0, 16):
            raise SystemExit(
                "fssimu2 needs Zig >= 0.16 "
                "(https://ziglang.org/download/)"
            )
        os.makedirs(out, exist_ok=True)
        run(
            [
                "zig", "build", "-Doptimize=ReleaseFast", "--prefix",
                os.path.join(REPO_ROOT, out), "--cache-dir",
                os.path.join(REPO_ROOT, out, "zig-cache")
            ], cwd=source
        )
    # A direct compiler invocation has no dependency graph. Recompile the
    # small driver even when the upstream library already exists.
    compiler = os.environ.get("CXX", "g++")
    run(
        [
            compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic",
            f"{REPO_ROOT}/bench/inmemory/fssimu2_bench.cc",
            f"-I{REPO_ROOT}/src", f"-I{REPO_ROOT}/tools/image_io",
            f"-I{REPO_ROOT}/include", f"-I{source}/src/include",
            f"{BUILD_DIR}/libvqm_io.a", f"{BUILD_DIR}/libvqm_core.a",
            f"-L{out}/lib", "-l:libssimu2.so", f"-Wl,-rpath,{out}/lib", "-lpng",
            "-ljpeg", "-lvulkan", "-o", f"{out}/bin/ssimu2_bench"
        ]
    )
    print(f"fssimu2: built {out}/bin/fssimu2 and {out}/bin/ssimu2_bench")


def build_vship_reference(
    reference: str, backend: str, out: str, jobs: int, force: bool
) -> None:
    """Build one Vship backend and link its in-memory driver.

    :param reference: Pinned Vship checkout name.
    :param backend: Upstream build backend name.
    :param out: Output directory name under the selected build directory.
    :param jobs: Parallel build job limit.
    :param force: Whether to recreate the copied source and build directories.
    :raises SystemExit: If a required tool or build step fails.
    """
    require_reference(reference)
    source = str(BUILD_DIR / f"{reference}-src-{backend}")
    out = str(BUILD_DIR / out)
    library = f"{out}/libvship.so"
    have_library = os.path.isfile(library)
    if force:
        shutil.rmtree(source, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
    copy_reference(f".reference/{reference}", source)
    if force or not have_library:
        run(
            [
                "make", "-C", source, f"BACKEND={backend}", "-j",
                str(jobs), "build"
            ]
        )
        os.makedirs(out, exist_ok=True)
        shutil.copyfile(f"{source}/libvship.so", library)
    # The checkout's own src/ comes first on the include path: a system Vship
    # installation puts a VshipAPI.h in /usr/include, and mixing that header
    # with this library reads garbage (the struct layouts are not guaranteed
    # to match).
    compiler = os.environ.get("CXX", "g++")
    run(
        [
            compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic",
            f"{REPO_ROOT}/bench/vship_bench.cc", f"-I{source}/src",
            f"-I{REPO_ROOT}/src", f"-I{REPO_ROOT}/tools/image_io",
            f"-I{REPO_ROOT}/include", f"{BUILD_DIR}/libvqm_io.a",
            f"{BUILD_DIR}/libvqm_core.a", f"-L{out}", "-l:libvship.so",
            f"-Wl,-rpath,{out}", "-lpng", "-ljpeg", "-lvulkan", "-o",
            f"{out}/vship_bench"
        ]
    )
    print(f"vship: built {out}/vship_bench")


def build_vmaf(jobs: int, force: bool) -> None:
    """Build Netflix's CPU scalar reference with integer and float features.

    Models stay in the copied checkout and are selected by explicit path.
    Disable embedded models to avoid requiring xxd, and SIMD assembly to keep
    the oracle's arithmetic independent of runtime CPU feature detection.

    :param jobs: Parallel build job limit.
    :param force: Whether to recreate the copied source and build directories.
    :raises SystemExit: If a required tool or build step fails.
    """
    require_reference("vmaf")
    source, out = str(BUILD_DIR / "vmaf-src"), str(BUILD_DIR / "vmaf-build")
    if force:
        shutil.rmtree(source, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
    copy_reference(".reference/vmaf", source)
    if not os.path.isfile(f"{out}/build.ninja"):
        run(
            [
                "meson", "setup", out, f"{source}/libvmaf",
                "--buildtype=release", "-Ddefault_library=static",
                "-Dbuilt_in_models=false", "-Denable_float=true",
                "-Denable_asm=false", "-Denable_tests=false",
                "-Denable_docs=false", "-Denable_tools=true"
            ]
        )
    run(["meson", "compile", "-C", out, "-j", str(jobs)])
    compiler = os.environ.get("CXX", "g++")
    run(
        [
            compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic",
            f"{REPO_ROOT}/validation/vmaf_reference.cc",
            f"-I{source}/libvmaf/include", f"{out}/src/libvmaf.a", "-pthread",
            "-lm", "-o", f"{out}/vmaf_reference"
        ]
    )
    print(f"vmaf: built {out}/tools/vmaf and {out}/vmaf_reference")


def build_cvvdp(jobs: int, force: bool) -> None:
    """Create and validate a locked, isolated CPU ColorVideoVDP runtime.

    :param jobs: Shared build concurrency setting, unused for this Python
        runtime.
    :param force: Whether to reapply the original's locked dependency versions.
    :raises SystemExit: If uv, dependency installation or the CPU probe fails.
    """
    del jobs
    require_reference("ColorVideoVDP")
    if shutil.which("uv") is None:
        raise SystemExit(
            "uv is required for the isolated CVVDP reference runtime"
        )
    python = os.path.join(REPO_ROOT, CVVDP_ENV, "bin", "python")
    if not os.path.isfile(python):
        run(["uv", "venv", "--no-project", "--python", "3.12", CVVDP_ENV])
    _validate_cvvdp_python(python)
    if force or not _runtime_matches(python):
        run(
            [
                "uv", "pip", "install", "--python", python, "--index",
                "https://download.pytorch.org/whl/cpu", "--index-strategy",
                "unsafe-best-match", "-r", CVVDP_REQUIREMENTS
            ]
        )
    if not _runtime_matches(python):
        raise SystemExit(
            "CVVDP runtime does not match its locked package versions"
        )
    _probe_cvvdp(python)
    print(f"cvvdp: validated CPU original runtime {python}")


def _validate_cvvdp_python(python: str) -> None:
    """Preserve the environment and require the qualified Python version.

    :param python: Project environment interpreter.
    :raises SystemExit: If the existing interpreter is not Python 3.12.
    """
    version = capture(
        [python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"]
    )
    if version != "3.12":
        raise SystemExit(
            f"CVVDP's locked runtime requires Python 3.12; existing {python} "
            f"reports {version or 'unavailable'}. "
            "Existing .venv files were preserved."
        )


def _runtime_matches(python: str) -> bool:
    """Check locked dependencies while allowing unrelated project packages.

    :param python: Project environment interpreter.
    :returns: Whether all exact original dependency versions are installed.
    """
    query = (
        "import importlib.metadata, json; "
        "print(json.dumps({d.metadata['Name'].lower().replace('_', '-'): "
        "d.version "
        "for d in importlib.metadata.distributions()}))"
    )
    recorded = capture([python, "-c", query])
    if recorded is None:
        return False
    installed = json.loads(recorded)
    with open(CVVDP_REQUIREMENTS, encoding="utf-8") as source:
        expected = dict(
            line.strip().split("==", 1) for line in source
            if line.strip() and not line.lstrip().startswith("#")
        )
    return all(
        installed.get(name.lower().replace("_", "-")) == version
        for name, version in expected.items()
    )


def _probe_cvvdp(python: str) -> None:
    """Run one CPU prediction without creating files in the pinned checkout.

    :param python: Isolated locked original interpreter.
    :raises SystemExit: If import, original model loading or CPU prediction
        fails.
    """
    checkout = os.path.join(REPO_ROOT, ".reference", "ColorVideoVDP")
    probe = """
import json
import math
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import numpy
import torch
import pycvvdp
torch.set_num_threads(1)
torch.set_float32_matmul_precision("highest")
metric = pycvvdp.cvvdp(display_name="standard_4k", device=torch.device("cpu"),
                     quiet=True, temp_padding="replicate")
pixels = torch.full((1, 3, 2, 64, 64), .5, dtype=torch.float32)
with torch.inference_mode():
    score, _ = metric.predict(
        pixels, pixels, dim_order="BCFHW", frames_per_second=30
    )
assert math.isfinite(float(score))
assert torch.version.cuda is None
print(json.dumps({"python": sys.version, "torch": str(torch.__version__),
                  "numpy": numpy.__version__, "device": "cpu",
                  "precision": "FP32-highest", "identity_score": float(score)}))
"""
    result = subprocess.run(
        [python, "-B", "-c", probe, checkout], capture_output=True, text=True,
        check=False
    )
    if result.returncode:
        raise SystemExit(
            "CVVDP CPU original probe failed:\n" +
            (result.stderr or result.stdout)
        )
    target = str(BUILD_DIR / "cvvdp-reference-probe.json")
    with open(target, "w", encoding="utf-8") as output:
        json.dump(json.loads(result.stdout), output, indent=2)
        output.write("\n")


def preflight(jobs: int, selected: list[str] | None = None) -> None:
    """Build the project archives required by selected reference drivers.

    :param jobs: Parallel build job limit.
    :param selected: Build steps that may require the core and image IO
        archives.
    :raises SystemExit: If CMake cannot configure or build the archives.
    """
    core_steps = {"fssimu2", "vship-cuda", "vship-vulkan", "vship-fast-vulkan"}
    if selected is not None and not core_steps.intersection(selected):
        return
    # Restore the IO target after an earlier core-only configure of this cache.
    run(
        [
            "cmake", "--preset", "release", "-B",
            str(BUILD_DIR), "-DVQM_BUILD_CLI=ON"
        ]
    )
    # Source archives can exist but predate the current working tree.
    run(
        [
            "cmake", "--build",
            str(BUILD_DIR), "--target", "vqm_io", "vqm_core", "-j",
            str(jobs)
        ]
    )


def _selected_steps(args: argparse.Namespace,
                    parser: argparse.ArgumentParser) -> list[str]:
    """Reject invalid selection or concurrency before fetching or building.

    :param args: Requested reference selection and concurrency.
    :param parser: Parser used for actionable command-line errors.
    :returns: Requested known reference steps, or the complete default matrix.
    """
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.only is None:
        return list(ALL_STEPS)
    selected = [name.strip() for name in args.only.split(",") if name.strip()]
    if not selected:
        parser.error("--only must select at least one reference step")
    unknown = [name for name in selected if name not in ALL_STEPS]
    if unknown:
        parser.error(
            f"unknown reference(s) {', '.join(unknown)}; choose "
            f"from: {', '.join(ALL_STEPS)}"
        )
    return list(dict.fromkeys(selected))


def main() -> int:
    """Verify sources, build references and refresh clangd's database.

    :returns: Zero after all selected builds succeed.
    :raises SystemExit: If selection, verification or a build command fails.
    """
    global BUILD_DIR, FINGERPRINT_DIR

    # The command owns its working directory; importing build helpers does not.
    os.chdir(REPO_ROOT)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only", default=None, help="comma-separated steps (default: all)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="remove the step's build trees and rebuild"
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    args = parser.parse_args()

    selected = _selected_steps(args, parser)
    BUILD_DIR = build_directory(args.build_dir)
    FINGERPRINT_DIR = str(BUILD_DIR / "reference-fingerprints")

    # A matching HEAD alone is not enough: uncommitted and ignored files can
    # also be copied into the reference build, including inside submodules.
    checkouts = sorted({STEP_CHECKOUTS[name] for name in selected})
    run(
        [
            sys.executable, "tools/fetch_references.py", "--check", "--only",
            ",".join(checkouts)
        ]
    )
    preflight(args.jobs, selected)
    jobs = args.jobs
    built = []
    builders = {
        "cloudinary":
        build_cloudinary,
        "libjxl":
        build_libjxl,
        "fssimu2":
        build_fssimu2,
        "vship-cuda":
        lambda j, f:
        build_vship_reference("Vship", "Cuda", "vship-build-cuda", j, f),
        "vship-vulkan":
        lambda j, f:
        build_vship_reference("Vship", "Vulkan", "vship-build-vulkan", j, f),
        "vship-fast-vulkan":
        lambda j, f: build_vship_reference(
            "Vship-fast", "Vulkan", "vship-fast-build-vulkan", j, f
        ),
        "vmaf":
        build_vmaf,
        "cvvdp":
        build_cvvdp,
    }
    for step in ALL_STEPS:
        if step in selected:
            build_fingerprinted(step, jobs, args.force, builders[step])
            built.append(step)
    # Refresh the clangd database so the editor sees the harness entries.
    from tools import vscode_compile_commands
    vscode_compile_commands.merge(BUILD_DIR)
    print(f"built: {' '.join(built)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
