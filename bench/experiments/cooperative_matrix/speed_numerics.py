"""Illustrate SpEED Gram precision; not a GPU oracle or timing benchmark.

The crop, 5x5 patch order, means and 64-lane reduction tree follow the maintained
SpEED covariance shader. Products use binary64 temporaries before binary32
accumulation to model full products of binary16 inputs, rather than products
rounded to binary16. Matrix hardware may choose a different addition order or
handling of subnormals; actual shader tests remain necessary.
"""

import argparse
import hashlib
import json
import math
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[3]
SHADERS = ROOT / "src/metrics/vmaf/speed/shaders"
DEFAULT_OUTPUT = ROOT / "artifacts/reports/vmaf-cooperative-matrix/numerics/speed.json"
RESIDUAL_SCALE = np.float32(2048)
SINGULAR_THRESHOLD = 1e-6


def coefficients(name: str, count: int) -> NDArray[np.float32]:
    """Read a literal SpEED filter array from the maintained shader.

    :param name: Name of the coefficient array, stage or case.
    :param count: Expected number of coefficients or samples.
    :returns: Coefficient array in binary32 tap order.
    :raises ValueError: If the named shader array is missing or its count differs.
    """
    source = (SHADERS / "vmaf_speed_filter_common.glsl").read_text()
    match = re.search(rf"{name}\[{count}\].*?\((.*?)\);", source, re.S)
    if match is None:
        raise ValueError(f"Cannot read shader array {name}")
    values = [float(value) for value in match.group(1).split(",")]
    if len(values) != count:
        raise ValueError(f"Wrong shader coefficient count for {name}")
    return np.asarray(values, dtype=np.float32)


def mirror(indices: NDArray[np.integer], extent: int) -> NDArray[np.integer]:
    """Reflect sample indices about the filter boundary.

    :param indices: Input sample indices before reflection.
    :param extent: Number of addressable samples on the axis.
    :returns: Reflected indices within the axis extent.
    """
    return np.where(
        indices < 0, -indices,
        np.where(indices >= extent, 2 * extent - indices - 2, indices)
    )


def filter_plane(source: NDArray[np.float32]) -> NDArray[np.float32]:
    """Separate binary32 multiply/add taps in the shader's original order.

    :param source: Input sample plane.
    :returns: Filtered binary32 plane at the downsampled size.
    """
    antialias, mean = coefficients("kAntialias", 9), coefficients("kMean", 3)
    height, width = source.shape
    down_height, down_width = height // 16, width // 16
    x = mirror(
        16 * (np.arange(9 * down_width) // 9) + np.arange(9 * down_width) % 9 - 4, width
    )
    vertical = np.zeros((down_height, 9 * down_width), dtype=np.float32)
    for tap, weight in enumerate(antialias):
        y = mirror(16 * np.arange(down_height) + tap - 4, height)
        vertical += np.float32(weight * source[np.ix_(y, x)])
    down = np.zeros((down_height, down_width), dtype=np.float32)
    for tap, weight in enumerate(antialias):
        down += np.float32(weight * vertical[:, tap::9])
    vertical_mean = np.zeros_like(down)
    for tap, weight in enumerate(mean):
        y = mirror(np.arange(down_height) + tap - 1, down_height)
        vertical_mean += np.float32(weight * down[y])
    smooth = np.zeros_like(down)
    for tap, weight in enumerate(mean):
        x = mirror(np.arange(down_width) + tap - 1, down_width)
        smooth += np.float32(weight * vertical_mean[:, x])
    return np.float32(down - smooth)


def lane_sum(values: NDArray[np.float32]) -> NDArray[np.float32] | np.float32:
    """Preserve the shader's strided per-lane order and halving add tree.

    :param values: Values in the input operand array.
    :returns: Binary32 mean reduced across the sample axis.
    """
    samples = len(values)
    padded = np.zeros(((samples + 63) // 64 * 64, *values.shape[1:]), dtype=np.float32)
    padded[:samples] = values
    lanes = np.zeros((64, *values.shape[1:]), dtype=np.float32)
    for batch in padded.reshape((-1, *lanes.shape)):
        lanes += batch
    for stride in (32, 16, 8, 4, 2, 1):
        lanes[:stride] += lanes[stride:2 * stride]
    return np.float32(lanes[0] / np.float32(samples))


def patch_samples(plane: NDArray[np.float32]) -> NDArray[np.float32]:
    """Crop to complete 5x5 blocks first, then enumerate overlapping patches.

    :param plane: Input image plane.
    :returns: Contiguous array of 25 samples per overlapping patch.
    :raises ValueError: If either cropped dimension is smaller than five.
    """
    height, width = plane.shape
    cropped = plane[:height // 5 * 5, :width // 5 * 5]
    if min(cropped.shape) < 5:
        raise ValueError("SpEED covariance needs both dimensions at least five")
    patches = np.lib.stride_tricks.sliding_window_view(cropped, (5, 5))
    return np.ascontiguousarray(patches.reshape((-1, 25)), dtype=np.float32)


def centered_samples(
    plane: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Subtract the shader-order column means from patch samples.

    :param plane: Input image plane.
    :returns: Centered patch matrix and its column means.
    """
    samples = patch_samples(plane)
    means = lane_sum(samples)
    return np.float32(samples - means), means


def gram(
    left: NDArray[np.float32], right: NDArray[np.float32] | None = None,
    half_products: bool = False
) -> NDArray[np.float32]:
    """Compute a Gram matrix using the shader's strided reduction order.

    A product of two finite binary16 operands is exactly representable in
    binary32 when its exponent is in range. Binary64 product/add temporaries
    model one binary32-rounded FMA step for such operands. For general binary32
    inputs binary64 temporaries are only a diagnostic approximation to FMA.

    :param left: Left matrix operand.
    :param right: Right matrix operand, or the left operand by default.
    :param half_products: Whether to round individual products to binary16.
    :returns: Binary32 Gram matrix.
    """
    if right is None:
        right = left
    samples, columns = left.shape
    padded = (samples + 63) // 64 * 64
    first = np.zeros((padded, columns), dtype=np.float32)
    second = np.zeros_like(first)
    first[:samples], second[:samples] = left, right
    lanes = np.zeros((64, columns, columns), dtype=np.float32)
    for start in range(0, padded, 64):
        product = (
            first[start:start + 64, :, None].astype(np.float64) *
            second[start:start + 64, None, :].astype(np.float64)
        )
        if half_products:
            product = product.astype(np.float16).astype(np.float64)
        lanes = (lanes.astype(np.float64) + product).astype(np.float32)
    for stride in (32, 16, 8, 4, 2, 1):
        lanes[:stride] += lanes[stride:2 * stride]
    return np.float32(lanes[0] / np.float32(samples))


def power_scale(values: NDArray[np.float32]) -> np.float32:
    """Choose a power-of-two scale from the largest absolute operand.

    :param values: Values in the input operand array.
    :returns: Binary32 power-of-two scale factor.
    """
    peak = float(np.max(np.abs(values)))
    return np.float32(
        1 if peak == 0 else math.ldexp(1.0, 7 - math.floor(math.log2(peak)))
    )


def half_round(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Round operands to binary16 and widen them back to binary32.

    :param values: Values in the input operand array.
    :returns: Binary32 array containing the rounded binary16 operands.
    """
    return values.astype(np.float16).astype(np.float32)


def encoding_bound(
    original: NDArray[np.floating], approximation: NDArray[np.floating]
) -> float:
    """Frobenius-norm upper bound on exact-Gram operand-quantization error.

    :param original: Original unquantized operands.
    :param approximation: Operands after the candidate encoding.
    :returns: Frobenius-norm bound on the exact Gram error.
    """
    difference = original.astype(np.float64) - approximation
    original_norm = np.linalg.norm(approximation)
    difference_norm = np.linalg.norm(difference)
    return float(
        (2 * original_norm * difference_norm + difference_norm**2) / len(original)
    )


def candidates(
    values: NDArray[np.float32]
) -> tuple[dict[str, NDArray[np.float32]], dict[str, object]]:
    """Evaluate the candidate Gram mappings and operand error bounds.

    :param values: Values in the input operand array.
    :returns: Gram matrices by encoding mode and operand diagnostics.
    """
    scale = power_scale(values)
    scaled = np.float32(values * scale)
    high = half_round(scaled)
    low = half_round(np.float32((scaled - high) * RESIDUAL_SCALE))
    high_gram, cross = gram(high), gram(high, low)
    cross_term = np.float32(np.float32(cross + cross.T) / RESIDUAL_SCALE)
    split_two = np.float32(high_gram + cross_term)
    split_three = np.float32(split_two + np.float32(gram(low) / RESIDUAL_SCALE**2))
    squared_scale = np.float32(scale * scale)
    plain = half_round(values)
    reconstructed = np.float32(high + np.float32(low / RESIDUAL_SCALE))
    exact_high = high.astype(np.float64) / float(scale)
    exact_low = low.astype(np.float64) / (float(scale) * float(RESIDUAL_SCALE))
    omitted_bound = float(np.linalg.norm(exact_low)**2 / len(values))
    reconstruction_bound = encoding_bound(values, exact_high + exact_low)
    return {
        "half_inputs_wide": gram(plain),
        "half_inputs_wide_power_scaled": np.float32(high_gram / squared_scale),
        "split_two_gram_products": np.float32(split_two / squared_scale),
        "split_three_gram_products": np.float32(split_three / squared_scale),
        "prior_half_products": gram(plain, half_products=True),
    }, {
        "power_scale":
        float(scale),
        "residual_scale":
        float(RESIDUAL_SCALE),
        "half_input_max_error":
        float(np.max(np.abs(values - plain))),
        "reconstruction_max_error":
        float(np.max(np.abs(values - reconstructed / scale))),
        "reconstruction_exact_values":
        int(np.count_nonzero(values == reconstructed / scale)),
        "value_count":
        int(values.size),
        "scaled_low_nonzero_subnormals":
        int(np.count_nonzero((low != 0) & (np.abs(low) < np.finfo(np.float16).tiny))),
        "operand_only_spectral_error_bounds": {
            "half_inputs_wide": encoding_bound(values, plain.astype(np.float64)),
            "half_inputs_wide_power_scaled": encoding_bound(values, exact_high),
            "split_two_gram_products": reconstruction_bound + omitted_bound,
            "split_three_gram_products": reconstruction_bound,
        },
        "bounds_scope":
        "Exact real-arithmetic Gram bounds for operand rounding; "
        "exclude FP32 accumulation, reduction, rescaling and addition rounding.",
    }


def covariance_diagnostics(
    candidate: NDArray[np.float32], reference: NDArray[np.float32]
) -> dict[str, object]:
    """Compare candidate and reference covariance entries and spectra.

    :param candidate: Candidate values to compare with the reference.
    :param reference: Reference values used for comparison.
    :returns: Entry, spectrum and classification error diagnostics.
    """
    reference64, candidate64 = reference.astype(np.float64
                                                ), candidate.astype(np.float64)
    reference_eigen = np.linalg.eigvalsh(reference64)
    candidate_eigen = np.linalg.eigvalsh(candidate64)
    error = candidate64 - reference64
    reference_norm = float(np.linalg.norm(reference64, 2))
    error_norm = float(np.linalg.norm(error, 2))
    reference_min, candidate_min = float(reference_eigen[0]), float(candidate_eigen[0])
    return {
        "max_absolute_entry_error":
        float(np.max(np.abs(error))),
        "relative_spectral_error":
        error_norm / max(reference_norm, 1e-300),
        "spectral_error":
        error_norm,
        "reference_minimum_eigenvalue":
        reference_min,
        "candidate_minimum_eigenvalue":
        candidate_min,
        "reference_singular":
        reference_min < SINGULAR_THRESHOLD,
        "candidate_singular":
        candidate_min < SINGULAR_THRESHOLD,
        "singular_classification_changed":
        ((reference_min < SINGULAR_THRESHOLD) != (candidate_min < SINGULAR_THRESHOLD)),
        "nonpositive_eigenvalue_count":
        int(np.count_nonzero(candidate_eigen <= 0)),
        "relative_minimum_eigenvalue_error":
        abs(candidate_min - reference_min) / max(abs(reference_min), 1e-30),
        "observed_error_condition_ratio":
        error_norm / max(abs(candidate_min), 1e-30),
        "threshold_separated_by_observed_error":
        bool(np.min(np.abs(candidate_eigen - SINGULAR_THRESHOLD)) > error_norm),
    }


def plane_cases() -> Iterator[tuple[str, NDArray[np.float32]]]:
    """Yield synthetic image planes with representative sample patterns.

    :returns: Pattern labels and synthetic filtered sample planes.
    """
    for height, width in ((320, 320), (465, 593), (81, 81)):
        y, x = np.indices((height, width), dtype=np.int64)
        val = 16 + ((x * 7 + y * 13 + (x // 7) * 23 + (y // 11) * 31 + 19) % 220)
        yield f"filtered-test-pattern-{width}x{height}", filter_plane(
            np.float32(val - 128)
        )
    height, width = 593, 465
    y, x = np.indices((height, width), dtype=np.float32)
    cases = {
        "flat-coded": np.full((height, width), 36.25, dtype=np.float32),
        "lowbit-ripple": np.float32(127 + ((x + y) % 4) / 256),
        "smooth-chroma": np.float32(80 * np.sin(x / 33) + 35 * np.cos(y / 19)),
        "highcontrast-stripes": np.float32(((x // 17 + y // 29) % 2) * 255 - 128),
    }
    for name, plane in cases.items():
        yield "filtered-" + name, filter_plane(plane)
    y, x = np.indices((29, 37), dtype=np.float32)
    yield "stats-low-rank-gradient", np.float32(7 * x + 3 * y)
    yield "stats-near-low-rank-gradient", np.float32(
        7 * x + 3 * y + .0002 * np.sin(x * 13 + y * 7)
    )
    yield "stats-smooth-rank", np.float32(np.sin(x / 3) + np.cos(y / 5))
    yield "stats-tiny-variation", np.float32(1e-8 * np.sin(x * 13 + y * 7))


def synthetic_matrix_cases() -> Iterator[tuple[str, NDArray[np.float32]]]:
    """Direct matrix stress cases; not asserted to be realizable image patches.

    :returns: Case labels and operand matrices with prescribed spectra.
    """
    rng = np.random.default_rng(85917)
    samples = 1024
    orthogonal, _ = np.linalg.qr(rng.normal(size=(samples, 25)))
    rotation, _ = np.linalg.qr(rng.normal(size=(25, 25)))
    for relative in (.95, .999, 1.001, 1.05, 4, 100):
        spectrum = np.linspace(1, 8, 25)
        spectrum[0] = SINGULAR_THRESHOLD * relative
        values = np.float32((orthogonal * np.sqrt(samples * spectrum)) @ rotation)
        yield f"direct-matrix-min-eigen-{relative:g}e-6", values


def scaled_patch_cases() -> Iterator[tuple[str, NDArray[np.float32]]]:
    """Patch-realizable conditioning stress, without integer-source claims.

    :returns: Case labels and scaled source planes.
    """
    y, x = np.indices((29, 37), dtype=np.int64)
    plane = np.float32(((x * 7 + y * 13 + x // 7 * 23 + y // 11 * 31) % 220) - 110)
    values, _ = centered_samples(plane)
    minimum = float(np.linalg.eigvalsh(gram(values).astype(np.float64))[0])
    for ratio in (.999, 1.001, 1.05):
        scaled = np.float32(
            plane * np.float32(math.sqrt(ratio * SINGULAR_THRESHOLD / minimum))
        )
        yield f"stats-scaled-patches-min-eigen-{ratio:g}e-6", scaled


def analyze(
    name: str, values: NDArray[np.float32], geometry: tuple[int, ...] | None = None,
    means: NDArray[np.float32] | None = None
) -> dict[str, object]:
    """Measure the covariance precision of one input case.

    :param name: Name of the coefficient array, stage or case.
    :param values: Values in the input operand array.
    :param geometry: Source plane geometry, if applicable.
    :param means: Patch means, if applicable.
    :returns: Covariance comparisons and operand diagnostics for the case.
    """
    baseline = gram(values)
    variants, encoding = candidates(values)
    fp64 = values.astype(np.float64).T @ values.astype(np.float64) / len(values)
    result = {
        "name":
        name,
        "samples":
        len(values),
        "zero_pad_rows_to_k16": (16 - len(values) % 16) % 16,
        "sample_sha256":
        hashlib.sha256(values.tobytes()).hexdigest(),
        "centering":
        "shader-order FP32 means before operand quantization"
        if means is not None else "direct matrix, no centering",
        "encoding":
        encoding,
        "fp32_baseline_vs_fp64_max_entry_error":
        float(np.max(np.abs(baseline - fp64))),
        "variants": {
            key: covariance_diagnostics(value, baseline)
            for key, value in variants.items()
        },
    }
    if geometry is not None:
        result["filtered_geometry"] = list(geometry)
        result["means_sha256"] = hashlib.sha256(means.tobytes()).hexdigest()
    return result


def main() -> None:
    """Write covariance precision diagnostics as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    results = []
    for name, plane in [*plane_cases(), *scaled_patch_cases()]:
        values, means = centered_samples(plane)
        results.append(analyze(name, values, plane.shape, means))
    for name, values in synthetic_matrix_cases():
        results.append(analyze(name, values))
    summary = {}
    for variant in results[0]["variants"]:
        entries = [case["variants"][variant] for case in results]
        summary[variant] = {
            "maximum_absolute_entry_error":
            max(e["max_absolute_entry_error"] for e in entries),
            "maximum_relative_spectral_error":
            max(e["relative_spectral_error"] for e in entries),
            "singular_classification_changes":
            sum(e["singular_classification_changed"] for e in entries),
            "classification_change_cases": [
                case["name"] for case in results
                if case["variants"][variant]["singular_classification_changed"]
            ],
        }
    output = {
        "schema_version": 1,
        "scope": "CPU numerical diagnostic, not GPU oracle or performance evidence",
        "numpy": np.__version__,
        "seed": 85917,
        "source_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__), SHADERS / "vmaf_speed_stats_cov.comp", SHADERS /
                "vmaf_speed_filter_common.glsl"
            ]
        },
        "singular_threshold": SINGULAR_THRESHOLD,
        "summary": summary,
        "cases": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
