#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Benchmark hosted videos by default; --suite images selects still-image peers."""

import argparse
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKOUTS = {
    "cloudinary": "ssimulacra2",
    "libjxl": "libjxl",
    "fssimu2": "fssimu2",
    "vship-cuda": "Vship",
    "vship-vulkan": "Vship",
    "vship-fast-vulkan": "Vship-fast",
}


def positive(value: str) -> int:
    """Parse a positive integer option.

    :param value: Command-line value.
    :returns: Parsed count.
    :raises argparse.ArgumentTypeError: If the count is less than one.
    """
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def run(command: Sequence[str]) -> None:
    """Print and run a benchmark step from the repository root.

    :param command: Executable and argument list.
    :raises subprocess.CalledProcessError: If the step exits unsuccessfully.
    """
    print("+ " + shlex.join(str(arg) for arg in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _image_parser() -> argparse.ArgumentParser:
    """Define the still-image comparison options."""
    parser = argparse.ArgumentParser(
        description=
        "Compare still-image latency against pinned reference implementations."
    )
    parser.add_argument(
        "--references", default="vship-vulkan", metavar="NAMES",
        help="comma-separated reference implementations; default: "
        "vship-vulkan (choices: " + ", ".join(CHECKOUTS) + ")"
    )
    parser.add_argument(
        "--metric", choices=("ssimu2", "butteraugli", "cvvdp"), default="ssimu2"
    )
    parser.add_argument(
        "--pair", nargs=2, metavar=("ORIGINAL", "DISTORTED"),
        help="input images; default: generate a synthetic pair"
    )
    parser.add_argument(
        "--size", default="1920x1080", metavar="WxH",
        help="synthetic pair size (default: 1920x1080)"
    )
    parser.add_argument("--rounds", type=positive, default=5)
    parser.add_argument("--iterations", type=positive, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--jobs", type=positive, default=min(os.cpu_count() or 1, 8))
    parser.add_argument(
        "--out-dir", default=str(ROOT / "artifacts/reports/quick-benchmark")
    )
    parser.add_argument(
        "--allow-unverified-devices", action="store_true",
        help="allow comparisons where device names cannot be uniquely matched"
    )
    return parser


def _references(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[str]:
    """Check selected reference implementations and repetition settings."""
    references = list(
        dict.fromkeys(
            name.strip() for name in args.references.split(",") if name.strip()
        )
    )
    unknown = set(references) - CHECKOUTS.keys()
    if not references or unknown:
        parser.error(
            "--references must contain known implementations: " + ", ".join(CHECKOUTS)
        )
    if args.metric == "cvvdp" and references != ["vship-vulkan"]:
        parser.error("CVVDP currently supports the vship-vulkan comparison arm only")
    if args.metric == "butteraugli" and ({"cloudinary", "fssimu2"} & set(references)):
        parser.error("cloudinary and fssimu2 benchmark SSIMULACRA 2 only")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return references


def _input_pair(parser: argparse.ArgumentParser,
                args: argparse.Namespace) -> list[Path] | None:
    """Resolve custom inputs or validate the requested synthetic geometry."""
    if not re.fullmatch(r"[1-9]\d*x[1-9]\d*", args.size):
        parser.error("--size must be a positive WxH geometry, for example 1920x1080")
    if not args.pair and any(int(dimension) % 2 for dimension in args.size.split("x")):
        parser.error(
            "synthetic --size dimensions must be even (FFmpeg testsrc2 "
            "rounds odd sizes down)"
        )
    if not args.pair:
        return None
    pair = [Path(path).resolve() for path in args.pair]
    if any(not path.is_file() for path in pair):
        parser.error("both input images must exist")
    if any(any(c in str(path) for c in "\t\r\n") for path in pair):
        parser.error("input paths cannot contain tabs or newlines")
    return pair


def _build_references(references: list[str], jobs: int) -> None:
    """Fetch pinned sources and build the selected comparison binaries."""
    run(
        [
            sys.executable, "tools/fetch_references.py", "--only",
            ",".join(sorted({CHECKOUTS[name]
                             for name in references}))
        ]
    )
    run(["cmake", "--preset", "release"])
    run(["cmake", "--build", "build", "--target", "vqm_bench", "--parallel", str(jobs)])
    run(
        [
            sys.executable, "tools/build_references.py", "--only", ",".join(references),
            "--jobs",
            str(jobs)
        ]
    )


def _pairs_file(size: str, pair: list[Path] | None) -> Path:
    """Prepare the pair list consumed by the in-memory driver."""
    pairs_dir = ROOT / ".cache/bench/quick-pairs"
    if pair is None:
        run(
            [
                sys.executable, "bench/make_bench_pairs.py", "--sizes", size,
                "--synthetic-only", "--out-dir",
                str(pairs_dir)
            ]
        )
        pairs_file = pairs_dir / f"synthetic-{size}.txt"
    else:
        pairs_dir.mkdir(parents=True, exist_ok=True)
        pairs_file = pairs_dir / "custom.txt"
        pairs_file.write_text(f"{pair[0]}\t{pair[1]}\n", encoding="utf-8")
    return pairs_file


def main(argv: Sequence[str] | None = None) -> int:
    """Build and benchmark the selected still-image implementations.

    :param argv: Explicit options or the process command line.
    :returns: Zero after the benchmark finishes.
    :raises subprocess.CalledProcessError: If a benchmark step fails.
    """
    parser = _image_parser()
    args = parser.parse_args(argv)
    references = _references(parser, args)
    pair = _input_pair(parser, args)
    _build_references(references, args.jobs)
    pairs_file = _pairs_file(args.size, pair)
    out_dir = Path(args.out_dir).resolve()
    benchmark = [
        sys.executable, "bench/run_inmemory.py", "--mode", "latency", "--metric",
        args.metric, "--pairs-file",
        str(pairs_file), "--only",
        ",".join(["ours-resident", "ours-streaming", *references]), "--rounds",
        str(args.rounds), "--iterations",
        str(args.iterations), "--warmup",
        str(args.warmup), "--out-dir",
        str(out_dir)
    ]
    if args.allow_unverified_devices:
        benchmark.append("--allow-unverified-devices")
    run(benchmark)
    print(f"Results: {out_dir / 'summary.md'}")
    return 0


def command_line(argv: Sequence[str] | None = None) -> int:
    """Dispatch the default hosted-video suite or explicit still-image suite.

    :param argv: Explicit options or the process command line.
    :returns: Selected suite's exit status.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--suite", choices=("video", "images"), default="video")
    selection, remaining = parser.parse_known_args(argv)
    if selection.suite == "images":
        return main(remaining)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bench import video
    return video.main(remaining)


if __name__ == "__main__":
    try:
        sys.exit(command_line())
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
    except FileNotFoundError as error:
        sys.exit(f"missing executable: {error.filename}")
    except RuntimeError as error:
        sys.exit(str(error))
