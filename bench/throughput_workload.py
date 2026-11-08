"""Define and provision the fixed-count GPU throughput comparison matrix."""

import argparse
import dataclasses
import json
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from bench.inmemory_environment import file_sha256
from bench.inmemory_measure import BenchEntry

ROOT = Path(__file__).resolve().parent.parent
METRICS = (
    "ssimulacra2", "butteraugli", "psnr-hvs", "cvvdp-image", "cvvdp-video"
)
PEERS = ("vship-cuda", "vship-vulkan", "vship-fast-vulkan")
SIZES = ("1920x1080", "3840x2160")
PPD = "75.402449345163049"
SPATIAL = frozenset(("ssimulacra2", "butteraugli"))
FILES = ("orig.png", "dist.png", "orig.y4m", "dist.y4m")
COUNTS = {
    "ssimulacra2": (2400, 600),
    "butteraugli": (600, 120),
    "psnr-hvs": (12000, 2400),
    "cvvdp-image": (120, 30),
    "cvvdp-video": (10, 10)
}


@dataclasses.dataclass(frozen=True)
class Arm:
    """Describe one fixed-count process and its accounting contract."""

    metric: str
    size: str
    implementation: str
    mode: str
    workers: int
    iterations: int
    frames: int
    command: tuple[str, ...]
    machine: bool

    @property
    def count(self) -> int:
        """Return timed frame pairs, excluding warmups."""
        return self.iterations * self.frames

    @property
    def key(self) -> str:
        """Return a stable unique arm identifier."""
        return f"{self.size}-{self.implementation}-{self.metric}-{self.mode}"


def entries(build_dir: Path, references: Sequence[str]) -> list[BenchEntry]:
    """Select executable paths before matching peer devices.

    :param build_dir: Absolute candidate and peer build directory.
    :param references: Selected peer names.
    :returns: Device-probe and provenance entries.
    """
    directories = {
        "vship-cuda": "vship-build-cuda",
        "vship-vulkan": "vship-build-vulkan",
        "vship-fast-vulkan": "vship-fast-build-vulkan"
    }
    peers: list[BenchEntry] = [
        {
            "name": name,
            "driver": str(build_dir / directories[name] / "vship_bench"),
            "args": []
        } for name in references
    ]
    return [
        {
            "name": "ours-streaming",
            "driver": str(build_dir / "vqm_bench"),
            "args": []
        }, *peers
    ]


def input_directory(args: argparse.Namespace, size: str) -> Path:
    """Locate one versioned workload's cached inputs.

    :param args: Input root and profile settings.
    :param size: Pixel geometry.
    :returns: Geometry-specific input directory.
    """
    return args.input_dir / args.profile / size


def clip_frames(size: str) -> int:
    """Return the complete clip length used at each geometry.

    :param size: Pixel geometry.
    :returns: Number of sequential frames per clip.
    """
    return {"1920x1080": 32, "3840x2160": 16}.get(size, 4)


def _resident_flags(metric: str, mode: str) -> tuple[str, ...]:
    if mode == "streaming":
        return ()
    return ("--vulkan-buffers",
            ) if metric == "cvvdp-video" else ("--resident", "2")


def _iterations(args: argparse.Namespace, metric: str, size: str) -> int:
    if args.profile == "smoke":
        return 2 if metric == "cvvdp-video" else 7
    return COUNTS[metric][SIZES.index(size)]


def _arm(
    args: argparse.Namespace, size: str, metric: str, entry: BenchEntry,
    mode: str
) -> Arm:
    ours = entry["name"] == "ours-streaming"
    implementation = "vqm" if ours else entry["name"]
    video = metric == "cvvdp-video"
    workers = 1 if metric.startswith("cvvdp") else args.workers
    iterations = _iterations(args, metric, size)
    machine = not ours or metric in SPATIAL
    pair = input_directory(args, size)
    suffix = "y4m" if video or metric == "psnr-hvs" else "png"
    driver_metric = "cvvdp" if metric.startswith("cvvdp") else metric
    if not ours and metric == "ssimulacra2":
        driver_metric = "ssimu2"
    command = [
        entry["driver"], *entry.get("args", []), "--metric", driver_metric,
        "--pair",
        str(pair / f"orig.{suffix}"),
        str(pair / f"dist.{suffix}"), "--workers" if ours else "--threads",
        str(workers), "--warmup",
        str(args.warmup),
        "--throughput-iterations" if machine else "--iterations",
        str(iterations)
    ]
    command.extend(_resident_flags(metric, mode))
    if ours and metric.startswith("cvvdp"):
        command.extend(("--cvvdp-ppd", PPD))
    if video:
        command.append("--cvvdp-video")
    return Arm(
        metric, size, implementation, mode, workers, iterations,
        clip_frames(size) if video else 1, tuple(command), machine
    )


def _metric_arms(
    args: argparse.Namespace, size: str, metric: str, drivers: list[BenchEntry]
) -> list[Arm]:
    ours = [
        _arm(args, size, metric, drivers[0], mode)
        for mode in ("streaming", "resident")
    ]
    peers = [
        entry for entry in drivers[1:] if metric in SPATIAL or
        (metric == "cvvdp-image" and entry["name"] != "vship-cuda")
    ]
    return ours + [
        _arm(args, size, metric, entry, "streaming") for entry in peers
    ]


def matrix(args: argparse.Namespace, drivers: list[BenchEntry]) -> list[Arm]:
    """Build every requested spatial, independent-image and temporal arm.

    :param args: Validated protocol settings.
    :param drivers: Candidate first, followed by device-matched peer entries.
    :returns: Complete ordered matrix, without executing any command.
    """
    return [
        arm for size in args.sizes for metric in METRICS
        for arm in _metric_arms(args, size, metric, drivers)
    ]


def round_order(arms: Sequence[Arm], index: int) -> list[Arm]:
    """Rotate complete arm order without running implementations concurrently.

    :param arms: One geometry's matrix.
    :param index: Zero-based measurement round.
    :returns: Rotated arm sequence.
    """
    offset = (index * 5) % len(arms)
    return [*arms[offset:], *arms[:offset]]


def _ffmpeg(arguments: Sequence[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *arguments],
        cwd=ROOT, check=True, timeout=180
    )


def _images(directory: Path, size: str) -> None:
    source = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate=1"]
    distortion = ("scale=iw/2:ih/2:flags=bicubic,scale=iw*2:ih*2:flags=bicubic")
    _ffmpeg(
        [
            *source, "-frames:v", "1", "-pix_fmt", "rgb24",
            str(directory / "orig.png")
        ]
    )
    _ffmpeg(
        [
            *source, "-vf", distortion, "-frames:v", "1", "-pix_fmt", "rgb24",
            str(directory / "dist.png")
        ]
    )


def _video(directory: Path, size: str) -> None:
    width, height = (int(value) for value in size.split("x"))
    original = directory / "orig.y4m"
    distortion = (
        f"scale={width // 2}:{height // 2}:flags=bicubic,"
        f"scale={width}:{height}:flags=bicubic"
    )
    _ffmpeg(
        [
            "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=24", "-frames:v",
            str(clip_frames(size)), "-pix_fmt", "yuv420p",
            str(original)
        ]
    )
    _ffmpeg(
        [
            "-i",
            str(original), "-vf", distortion, "-frames:v",
            str(clip_frames(size)), "-pix_fmt", "yuv420p",
            str(directory / "dist.y4m")
        ]
    )


def _input_hashes(directory: Path) -> dict[str, str | None]:
    return {name: file_sha256(directory / name) for name in FILES}


def prepare_inputs(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Generate deterministic inputs or verify a previously generated cache.

    :param args: Input root, profile and selected geometry settings.
    :returns: All image and video pairs for provenance hashing.
    :raises RuntimeError: If cached inputs are incomplete or have changed.
    """
    pairs = []
    for size in args.sizes:
        directory = input_directory(args, size)
        _prepare_geometry(directory, size)
        pairs.extend(
            (str(directory / original), str(directory / distorted))
            for original, distorted in
            (("orig.png", "dist.png"), ("orig.y4m", "dist.y4m"))
        )
    return pairs


def _prepare_geometry(directory: Path, size: str) -> None:
    recipe = {
        "version": 1,
        "size": size,
        "fps": 24,
        "frames": clip_frames(size)
    }
    if directory.exists():
        _verify_inputs(directory, recipe)
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
        staging = Path(temporary) / "inputs"
        staging.mkdir()
        _images(staging, size)
        _video(staging, size)
        manifest = {"recipe": recipe, "sha256": _input_hashes(staging)}
        (staging / "manifest.json"
         ).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(directory)


def _verify_inputs(directory: Path, recipe: dict[str, str | int]) -> None:
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"unverified input cache: {directory}") from error
    hashes = _input_hashes(directory)
    if manifest != {"recipe": recipe, "sha256": hashes
                    } or None in hashes.values():
        raise RuntimeError(
            f"input cache changed; select a new --input-dir: {directory}"
        )
