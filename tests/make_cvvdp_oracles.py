#!/usr/bin/env python3
# Copyright (c) 2024 Graphics and Displays group.
# MIT; see licenses/COLORVIDEOVDP-LICENSE.
"""Generate independent raw-tensor CVVDP oracle scores and stage arrays.

Run with the optional environment in docs/cvvdp-development.md. The upstream
checkout is imported read-only, bytecode output is disabled, and all operations
use CPU FP32 with one thread. No file loader or video decoder participates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / ".reference/ColorVideoVDP"
DEFAULT_OUTPUT = ROOT / "tests/metrics/cvvdp/cvvdp_reference.inc"
DEFAULT_REPORT = ROOT / "bench/notes/cvvdp/oracle-characterization.json"
PIN = "2a268bce8d56e2f3abde46df3927d8a633707a24"
sys.path.insert(0, str(REFERENCE))

import pycvvdp  # noqa: E402
import torch  # noqa: E402
from pycvvdp.display_model import vvdp_display_photo_eotf  # noqa: E402
from pycvvdp.lpyr_dec import lpyr_dec, weber_contrast_pyr  # noqa: E402


@dataclass(frozen=True)
class Case:
    """One deterministic input pair and its physical interpretation."""

    name: str
    width: int = 32
    height: int = 32
    frames: int = 1
    fps_num: int = 0
    fps_den: int = 1
    display: str = "standard_4k"
    color_space: str = "sRGB"
    artifact: int = 0


CASES = (
    Case("image_pattern"), Case("image_identity",
                                artifact=1), Case("image_tiny_error", artifact=2),
    Case("image_black_white", artifact=3), Case("image_opponent", artifact=5),
    Case("image_odd_odd", width=33,
         height=17), Case("image_odd_even", width=33,
                          height=18), Case("image_even_odd", width=34, height=17),
    Case("image_small", width=8, height=8), Case("image_fhd", display="standard_fhd"),
    Case("image_p3", color_space="Display P3 Apple"),
    Case("image_pq", display="standard_hdr_pq", color_space="BT.2020-PQ"),
    Case("image_hlg", display="standard_hdr_hlg", color_space="BT.2020-HLG"),
    Case("image_linear", display="standard_hdr_linear",
         color_space="BT.709-linear"), Case("video_one_frame", fps_num=30),
    Case("video_two_frames", frames=2,
         fps_num=30), Case("video_short", frames=4,
                           fps_num=24), Case("video_ring_wrap", frames=20, fps_num=60),
    Case("video_identity", frames=3, fps_num=30,
         artifact=1), Case("video_minimum_fps", frames=2, fps_num=1),
    Case("video_120fps_ring", width=32, height=24, frames=40, fps_num=120),
    Case("video_240fps_ring", width=32, height=24, frames=62, fps_num=240),
    Case("video_fractional_double_wrap", frames=37, fps_num=60000,
         fps_den=1001), Case("video_taps_minus_one", frames=8, fps_num=30, artifact=4),
    Case("video_taps_exact", frames=9, fps_num=30,
         artifact=4), Case("video_taps_plus_one", frames=10, fps_num=30, artifact=4),
    Case("video_flicker", frames=12, fps_num=30000, fps_den=1001,
         artifact=4), Case("video_negative", frames=30, fps_num=30, artifact=6),
    Case(
        "video_pq", frames=4, fps_num=24, display="standard_hdr_pq",
        color_space="BT.2020-PQ"
    ),
)


def sample(
    x: int, y: int, channel: int, frame: int, artifact: int, distorted: bool
) -> float:
    """Return a deterministic normalized sample before optional nits scaling.

    :param x: Horizontal coordinate.
    :param y: Vertical coordinate.
    :param channel: RGB channel, zero through two.
    :param frame: Zero-based frame index.
    :param artifact: Perturbation identifier from ``CASES``.
    :param distorted: Whether this is the test image.
    :returns: Encoded sample in [0, 1].
    """
    if artifact == 2:
        return 0.5001 if distorted else 0.5
    if artifact == 3:
        return 1.0 if distorted else 0.0
    if artifact == 5:
        return 1.0 if channel == (0 if distorted else 1) else 0.0
    if artifact == 6:
        active = (2 if frame % 2 else 0) if distorted else 1
        return 1.0 if channel == active else 0.0
    code = 20 + (
        x * (3 + 2 * channel) + y * (5 + 2 * channel) + frame *
        (11 + 4 * channel) + channel * 37
    ) % 211
    if distorted and artifact == 0:
        code = ((code + 3) // 8) * 8
        code += 5 if (x // 7 + y // 5 + frame + channel) % 3 == 0 else 0
    value = float32(min(255, max(0, code)) / 255)
    if distorted and artifact == 4:
        offset = float32(0.08 if frame % 2 else -0.08)
        value = min(1.0, max(0.0, float32(value + offset)))
    return value


def float32(value: float) -> float:
    """Round one input-fixture operation to the C++ float32 sample contract.

    :param value: Intermediate arithmetic result.
    :returns: Value rounded to IEEE binary32.
    """
    return struct.unpack("f", struct.pack("f", value))[0]


def tensor(case: Case, distorted: bool) -> torch.Tensor:
    """Construct an explicitly FP32 [B,C,F,H,W] raw input tensor.

    :param case: Input geometry and interpretation.
    :param distorted: Whether to include distortion.
    :returns: Contiguous raw RGB samples; linear inputs are absolute nits.
    """
    values = [
        sample(x, y, c, f, case.artifact, distorted) for c in range(3)
        for f in range(case.frames) for y in range(case.height)
        for x in range(case.width)
    ]
    result = torch.tensor(values, dtype=torch.float32
                          ).reshape(1, 3, case.frames, case.height, case.width)
    return result * 1000 if case.color_space.endswith("-linear") else result


def metric_for(case: Case) -> pycvvdp.cvvdp:
    """Create the pinned original metric with explicit source colorimetry.

    :param case: Physical display and source color space.
    :returns: CPU-only FP32 reference metric.
    """
    display = pycvvdp.vvdp_display_photometry.load(case.display, [])
    photo = vvdp_display_photo_eotf(
        display.Y_peak, contrast=display.contrast, source_colorspace=case.color_space,
        E_ambient=display.E_ambient, k_refl=display.k_refl, exposure=display.exposure
    )
    return pycvvdp.cvvdp(
        display_name=case.display, display_photometry=photo, device=torch.device("cpu"),
        quiet=True, temp_padding="replicate"
    )


def evaluate(case: Case) -> dict[str, Any]:
    """Score one case repeatedly and preserve independent latent features.

    :param case: Deterministic pair and configuration.
    :returns: Scores, repeat variation, features, source hashes and DKL probes.
    """
    metric = metric_for(case)
    original, distorted = tensor(case, False), tensor(case, True)
    fps = case.fps_num / case.fps_den
    scores = []
    with torch.inference_mode():
        for _ in range(3):
            score, stats = metric.predict(distorted, original, frames_per_second=fps)
            scores.append(float(score))
    reference_dkl = metric.display_photometry.linear_2_target_colorspace(
        metric.display_photometry.forward(original[:, :, :1]), "DKLd65"
    )
    return {
        "name": case.name,
        "score": scores[0],
        "repeat_scores": scores,
        "repeat_spread": max(scores) - min(scores),
        "Q_per_ch": stats["Q_per_ch"].tolist(),
        "rho_band": stats["rho_band"].tolist(),
        "ppd": metric.pix_per_deg,
        "reference_dkl_first_pixel": reference_dkl[0, :, 0, 0, 0].tolist(),
        "original_sha256": hashlib.sha256(original.numpy().tobytes()).hexdigest(),
        "distorted_sha256": hashlib.sha256(distorted.numpy().tobytes()).hexdigest()
    }


def temporal_oracles() -> list[dict[str, Any]]:
    """Generate coefficient arrays directly with the original irFFT path.

    :returns: Four filters per rational frame rate.
    """
    metric = metric_for(CASES[0])
    rows = []
    for numerator, denominator in ((24, 1), (30, 1), (60, 1), (120, 1), (24000, 1001),
                                   (30000, 1001), (60000, 1001)):
        filters, _ = metric.get_temporal_filters(numerator / denominator)
        rows.append(
            {
                "fps_num": numerator,
                "fps_den": denominator,
                "taps": len(filters[0]),
                "coefficients": [item.tolist() for item in filters]
            }
        )
    return rows


def odd_boundary_characterization() -> list[dict[str, Any]]:
    """Exercise the original horizontal parity branch without editing it.

    :returns: Original reductions and a mathematical symmetric-edge comparison.
    """
    rows = []
    for width, height in ((4, 4), (5, 5), (4, 5), (5, 4)):
        image = torch.arange(width * height,
                             dtype=torch.float32).reshape(1, 1, height, width)
        pyr = lpyr_dec(width, height, 75.0, torch.device("cpu"))
        reduced = pyr.gausspyr_reduce(image)
        expected = symmetric_reduction(image)
        rows.append(
            {
                "width": width,
                "height": height,
                "original": reduced.flatten().tolist(),
                "symmetric": expected.flatten().tolist(),
                "max_difference": float((reduced - expected).abs().max())
            }
        )
    return rows


def symmetric_reduction(image: torch.Tensor) -> torch.Tensor:
    """Compute a small independent symmetric five-tap reduction.

    :param image: One 2D plane wrapped in [1,1,H,W].
    :returns: Reduced plane with mathematically mirrored column boundaries.
    """
    height, width = image.shape[-2:]
    kernel = torch.tensor([0.05, 0.25, 0.4, 0.25, 0.05])
    rows = []
    for y in range(0, height, 2):
        rows.append([symmetric_pixel(image, kernel, x, y) for x in range(0, width, 2)])
    return torch.tensor(rows).reshape(1, 1, (height + 1) // 2, (width + 1) // 2)


def symmetric_pixel(image: torch.Tensor, kernel: torch.Tensor, x: int, y: int) -> float:
    """Evaluate one separable reflected tap neighborhood.

    :param image: Input plane.
    :param kernel: Five symmetric weights.
    :param x: Output center in source coordinates.
    :param y: Output center in source coordinates.
    :returns: Independently filtered scalar.
    """
    height, width = image.shape[-2:]
    vertical = []
    for dx in range(-2, 3):
        horizontal = [
            image[0, 0, mirror(y + dy, height),
                  mirror(x + dx, width)] * kernel[dy + 2] for dy in range(-2, 3)
        ]
        vertical.append(torch.stack(horizontal).sum() * kernel[dx + 2])
    return float(torch.stack(vertical).sum())


def mirror(index: int, length: int) -> int:
    """Reflect one coordinate, repeating the edge sample.

    :param index: Unbounded coordinate.
    :param length: Positive axis length.
    :returns: In-bounds symmetric coordinate.
    """
    if index < 0:
        return -index - 1
    if index >= length:
        return 2 * length - index - 1
    return index


def spatial_input(width: int, height: int, channels: int) -> torch.Tensor:
    """Construct direct DKL planes with stepwise FP32 arithmetic.

    :param width: Plane width.
    :param height: Plane height.
    :param channels: Three image or four video channels.
    :returns: Interleaved [T0,R0,T1,R1,...] planes, [1,2*C,1,H,W].
    """
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    reference = [
        10.0 + (x % 7).float() + 2.0 * (y % 5).float(),
        0.3 + ((x + 3 * y) % 11).float() / 50.0,
        0.2 + ((3 * x + y) % 13).float() / 60.0, ((x % 3) - (y % 3)).float() / 10.0
    ]
    distorted = [
        reference[0] + (((x + 2 * y) % 5) - 2).float() / 4.0,
        reference[1] + torch.where((x + y) % 2 != 0, 0.02, -0.01), reference[2] * 0.93,
        reference[3] + 0.1 * (x % 2).float()
    ]
    pairs = [plane for c in range(channels) for plane in (distorted[c], reference[c])]
    return torch.stack(pairs).reshape(1, 2 * channels, 1, height, width)


def spatial_oracles() -> list[dict[str, Any]]:
    """Evaluate the full spatial metric independently of input photometry/FIR.

    :returns: Channel-major band features for ten DKL fixtures.
    """
    metric = metric_for(CASES[0])
    rows = []
    for width, height, channels in ((w, h, c) for w, h in ((32, 24), (33, 24), (32, 25),
                                                           (17, 17), (2, 2))
                                    for c in (3, 4)):
        pyramid = weber_contrast_pyr(
            width, height, 60.0, torch.device("cpu"), "weber_g1"
        )
        planes = spatial_input(width, height, channels)
        quality, _ = metric.process_block_of_frames(
            planes, (height, width, 1), channels - 2, pyramid, channels == 3
        )
        rows.append(
            {
                "width": width,
                "height": height,
                "channels": channels,
                "features": quality.flatten().tolist(),
                "input_sha256": hashlib.sha256(planes.numpy().tobytes()).hexdigest()
            }
        )
    return rows


def emit_spatial(rows: list[dict[str, Any]]) -> str:
    """Render the independently generated direct spatial stage fixtures.

    :param rows: Original per-channel/band feature arrays.
    :returns: Self-contained C++ declarations.
    """
    lines = [
        "// Copyright (c) 2024 Graphics and Displays group.",
        "// MIT; see licenses/COLORVIDEOVDP-LICENSE.",
        "// Generated by tests/make_cvvdp_oracles.py; original v0.5.7 CPU FP32.",
        "// Input formula: spatial_input(); ppd=60; features [channel][band].",
        "#include <array>", "#include <cstddef>", "#include <cstdint>",
        "#include <span>", "// clang-format off"
    ]
    for index, row in enumerate(rows):
        features = row["features"]
        lines.append(
            f"constexpr std::array<double, {len(features)}> kCvvdpSpatial{index} = {{"
        )
        lines.extend(
            "    " + ", ".join(f"{v:.17g}" for v in features[i:i + 4]) + ","
            for i in range(0, len(features), 4)
        )
        lines.append("};")
    lines.extend(
        [
            "struct CvvdpSpatialReference {", "    std::size_t width, height;",
            "    std::uint32_t channels;", "    std::span<const double> features;",
            "};",
            f"constexpr std::array<CvvdpSpatialReference, {len(rows)}> kCvvdpSpatialReferences = {{{{"
        ]
    )
    lines.extend(
        f'    {{{row["width"]}, {row["height"]}, {row["channels"]}, kCvvdpSpatial{i}}},'
        for i, row in enumerate(rows)
    )
    lines.extend(["}};", "// clang-format on", ""])
    return "\n".join(lines)


def emit_header(results: list[dict[str, Any]], filters: list[dict[str, Any]]) -> str:
    """Render compact independent C++ oracle data.

    :param results: Original scores for ``CASES``.
    :param filters: Original temporal coefficients.
    :returns: Include text with self-contained oracle structures.
    """
    lines = [
        "// Copyright (c) 2024 Graphics and Displays group.",
        "// MIT; see licenses/COLORVIDEOVDP-LICENSE.",
        "// Generated by tests/make_cvvdp_oracles.py; do not edit.",
        f"// Original ColorVideoVDP v0.5.7, {PIN}; CPU FP32, one thread.",
        "// Input formula: tests/make_cvvdp_oracles.py:sample; no image loader.",
        "#include <array>", "#include <cstddef>", "#include <cstdint>",
        "struct CvvdpReferenceCase {", "    const char* name;",
        "    std::size_t width, height, frames;", "    std::uint32_t fps_num, fps_den;",
        "    const char* display;", "    const char* color_space;", "    int artifact;",
        "    double score;", "};", "// clang-format off",
        f"constexpr std::array<CvvdpReferenceCase, {len(CASES)}> kCvvdpReferences = {{{{"
    ]
    for case, result in zip(CASES, results, strict=True):
        lines.append(
            f'    {{"{case.name}", {case.width}, {case.height}, {case.frames}, '
            f'{case.fps_num}, {case.fps_den}, "{case.display}", '
            f'"{case.color_space}", {case.artifact}, {result["score"]:.17g}}},'
        )
    lines.extend(
        [
            "}};", *probe_lines(results), *filter_lines(filters), "// clang-format on",
            ""
        ]
    )
    return "\n".join(lines)


def probe_lines(results: list[dict[str, Any]]) -> list[str]:
    """Render compact independent DKL and first-image pooling probes.

    :param results: Original reference outputs.
    :returns: C++ probe arrays.
    """
    lines = [
        f"constexpr std::array<std::array<double, 3>, {len(results)}> kCvvdpReferenceDkl = {{{{"
    ]
    for result in results:
        values = ", ".join(f"{v:.17g}" for v in result["reference_dkl_first_pixel"])
        lines.append("    {" + values + "},")
    lines.append("}};")
    features = [
        v for channel in results[0]["Q_per_ch"][0] for frame in channel for v in frame
    ]
    lines.append(
        f"constexpr std::array<double, {len(features)}> kCvvdpPatternFeatures = {{"
    )
    lines.extend(
        "    " + ", ".join(f"{v:.17g}" for v in features[i:i + 4]) + ","
        for i in range(0, len(features), 4)
    )
    lines.append("};")
    return lines


def filter_lines(filters: list[dict[str, Any]]) -> list[str]:
    """Render each original filter as a separate flattened constexpr array.

    :param filters: Frame-rate coefficient records.
    :returns: C++ declaration lines.
    """
    lines = []
    for row in filters:
        values = [value for channel in row["coefficients"] for value in channel]
        label = f'{row["fps_num"]}_{row["fps_den"]}'
        lines.append(
            f"constexpr std::array<double, {len(values)}> kCvvdpFilters{label} = {{"
        )
        lines.extend(
            "    " + ", ".join(f"{v:.17g}" for v in values[i:i + 4]) + ","
            for i in range(0, len(values), 4)
        )
        lines.append("};")
    return lines


def main() -> None:
    """Generate checked-in fixtures and compact characterization evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--spatial-output", type=Path,
        default=DEFAULT_OUTPUT.with_name("cvvdp_spatial_reference.inc")
    )
    args = parser.parse_args()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/fetch_references.py"), "--check", "--only",
            "ColorVideoVDP"
        ], check=True
    )
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    results = [evaluate(case) for case in CASES]
    filters = temporal_oracles()
    spatial = spatial_oracles()
    report = {
        "upstream_commit": PIN,
        "torch": torch.__version__,
        "numpy": __import__("numpy").__version__,
        "threads": 1,
        "matmul_precision": "highest",
        "results": results,
        "temporal_filters": filters,
        "spatial": spatial,
        "odd_boundary": odd_boundary_characterization()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(emit_header(results, filters))
    args.spatial_output.parent.mkdir(parents=True, exist_ok=True)
    args.spatial_output.write_text(emit_spatial(spatial))
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for row in results:
        print(
            f'{row["name"]}: {row["score"]:.9f} (repeat spread {row["repeat_spread"]:.3g})'
        )


if __name__ == "__main__":
    main()
