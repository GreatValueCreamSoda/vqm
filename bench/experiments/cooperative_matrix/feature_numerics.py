"""Illustrate wide-product operand limits in VMAF filters; no GPU or timers.

Read coefficients from maintained shaders and retain explicit FP32 product/add
boundaries. This is a numerical illustration, not an independent metric oracle
or an emulation of a driver's unspecified matrix summation order.
"""

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "artifacts/reports/vmaf-cooperative-matrix/feature-mapping/numerics.json"
SHADERS = ROOT / "src/metrics/vmaf/features/shaders"
SPLIT_SCALE = np.float32(4096.0)
MODES = ("fp32", "half_products", "wide_half", "split_three", "split_four")


def coefficients(path: Path, name: str, count: int) -> NDArray[np.float32]:
    """Fail visibly if a shader coefficient declaration changes form.

    :param path: File to read, hash or update.
    :param name: Name of the coefficient array, stage or case.
    :param count: Expected number of coefficients or samples.
    :returns: Coefficient array in binary32 tap order.
    :raises ValueError: If the shader array is missing or its count differs.
    """
    match = re.search(rf"{name}\[{count}\].*?\((.*?)\);", path.read_text(), re.S)
    if match is None:
        raise ValueError(f"Could not read {name}[{count}] in {path}")
    values = [float(value) for value in match.group(1).split(",")]
    if len(values) != count:
        raise ValueError(f"Unexpected coefficient count for {name}")
    return np.asarray(values, np.float32)


def split(
    value: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Split operands into binary16 high parts and scaled residuals.

    :param value: Input sample value.
    :returns: High operands and scaled residual operands.
    """
    value = np.asarray(value, np.float32)
    high = value.astype(np.float16).astype(np.float32)
    low = np.float32(np.float32(value - high) * SPLIT_SCALE)
    return high, low.astype(np.float16).astype(np.float32)


def sum32(products: NDArray[np.float32]) -> NDArray[np.float32] | np.float32:
    """Accumulate products in order with binary32 additions.

    :param products: Ordered array of products to accumulate.
    :returns: Sum in binary32 precision.
    """
    total = np.zeros(products.shape[1:], np.float32)
    for product in products:
        total = np.float32(total + product)
    return total


def dot(weights: NDArray[np.float32], samples: NDArray[np.float32],
        mode: str) -> NDArray[np.float32] | np.float32:
    """Reduce axis zero; one FP32 summation order, not a GPU implementation.

    :param weights: Filter coefficients in tap order.
    :param samples: Sample values grouped by tap.
    :param mode: Numerical or installation mode.
    :returns: Dot products reduced across the sample axis.
    :raises ValueError: If the precision mode is unsupported.
    """
    weights = np.asarray(weights, np.float32
                         ).reshape((len(weights), ) + (1, ) * (samples.ndim - 1))
    samples = np.asarray(samples, np.float32)
    if mode == "fp32":
        return sum32(np.float32(weights * samples))
    high_w, low_w = split(weights)
    high_x, low_x = split(samples)
    if mode == "half_products":
        with np.errstate(over="raise", invalid="raise"):
            products = np.multiply(
                high_w.astype(np.float16), high_x.astype(np.float16), dtype=np.float16
            )
        return sum32(products.astype(np.float32))
    total = sum32(np.float32(high_w * high_x))
    if mode == "wide_half":
        return total
    cross = np.float32(
        sum32(np.float32(high_w * low_x)) + sum32(np.float32(low_w * high_x))
    )
    total = np.float32(total + np.float32(cross / SPLIT_SCALE))
    if mode == "split_four":
        residual = sum32(np.float32(low_w * low_x))
        total = np.float32(total + np.float32(residual / (SPLIT_SCALE * SPLIT_SCALE)))
    elif mode != "split_three":
        raise ValueError(mode)
    return total


def errors(reference: NDArray[np.floating],
           candidate: NDArray[np.floating]) -> dict[str, float | int]:
    """Measure absolute differences between two results.

    :param reference: Reference values used for comparison.
    :param candidate: Candidate values to compare with the reference.
    :returns: Absolute error statistics and changed sample count.
    """
    difference = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    return {
        "maximum": float(np.max(difference)),
        "mean": float(np.mean(difference)),
        "changed": int(np.count_nonzero(difference)),
        "count": int(difference.size)
    }


def operand_report() -> dict[str, object]:
    """Measure binary16 encoding error across native sample depths.

    :returns: Operand encoding errors indexed by source bit depth.
    """
    result = {}
    for depth in (8, 10, 12, 16):
        values = np.float32(
            np.arange(1 << depth, dtype=np.float32) * np.float32(2.0**(8 - depth)) -
            np.float32(128.0)
        )
        high, low = split(values)
        result[str(depth)] = {
            "half":
            errors(values, high),
            "high_plus_scaled_residual":
            errors(values, np.float32(high + low / SPLIT_SCALE)),
        }
    return result


def coefficient_report(filters: dict[str, NDArray[np.float32]]) -> dict[str, object]:
    """Compare binary32 filter weights with their binary16 encodings.

    :param filters: Named arrays of filter coefficients.
    :returns: Encoded weights, sums and errors by filter name.
    """
    result = {}
    for name, weights in filters.items():
        high, low = split(weights)
        result[name] = {
            "fp32": weights.astype(float).tolist(),
            "half": high.astype(float).tolist(),
            "sum_fp32_values_in_fp64": float(np.sum(weights.astype(np.float64))),
            "sum_half_values_in_fp64": float(np.sum(high.astype(np.float64))),
            "half_error": errors(weights, high),
            "split_error": errors(weights, np.float32(high + low / SPLIT_SCALE)),
        }
    return result


def dot_report(filters: dict[str, NDArray[np.float32]]) -> dict[str, object]:
    """Compare fixed-seed dot products with binary32 and binary64 results.

    :param filters: Named arrays of filter coefficients.
    :returns: Dot-product errors by filter and precision mode.
    """
    random = np.random.default_rng(20261105)
    result = {}
    for name, weights in filters.items():
        samples = np.float32(random.uniform(-128.0, 128.0, (len(weights), 4096)))
        reference = dot(weights, samples, "fp32")
        ideal = np.sum(
            weights.astype(np.float64)[:, None] * samples.astype(np.float64), axis=0
        )
        result[name] = {}
        for mode in MODES:
            candidate = dot(weights, samples, mode)
            result[name][mode] = {
                "vs_fp32_order": errors(reference, candidate),
                "vs_fp64_dot": errors(ideal, candidate),
            }
    return result


def constant_moments(value: np.float32, weights: NDArray[np.float32],
                     mode: str) -> dict[str, float | bool]:
    """Filter the first and second moments of a constant field.

    :param value: Input sample value.
    :param weights: Filter coefficients in tap order.
    :param mode: Numerical or installation mode.
    :returns: Filtered mean, second moment, variance and branch decision.
    """
    quantities = np.asarray([value, np.float32(value * value)], np.float32)
    samples = np.repeat(quantities[None, :], len(weights), axis=0)
    vertical = dot(weights, samples, mode)
    samples = np.repeat(vertical[None, :], len(weights), axis=0)
    horizontal = dot(weights, samples, mode)
    variance = np.float32(horizontal[1] - np.float32(horizontal[0] * horizontal[0]))
    return {
        "mean": float(horizontal[0]),
        "second_moment": float(horizontal[1]),
        "variance_before_clamp": float(variance),
        "low_variance_branch": bool(max(variance, np.float32(0.0)) < 2.0)
    }


def vif_report(weights: NDArray[np.float32]) -> dict[str, object]:
    """Report VIF constant-field moments for representative sample values.

    :param weights: Filter coefficients in tap order.
    :returns: Moment diagnostics by constant sample value and precision mode.
    """
    return {
        str(value): {
            mode: constant_moments(np.float32(value), weights, mode)
            for mode in MODES
        }
        for value in (100.5, 100.25, 127.0, 127.00390625, -128.0)
    }


def weighted_gram_report(weights: NDArray[np.float32]) -> dict[str, object]:
    """Evaluate direct weighted-Gram moments without precomputed squares.

    :param weights: Filter coefficients in tap order.
    :returns: Moment and weight-normalization diagnostics by case and mode.
    """
    weights_2d = np.float32(weights[:, None] * weights[None, :]).reshape(-1)
    roots = np.sqrt(weights_2d, dtype=np.float32)
    result = {}
    checker = ((np.indices((len(weights), len(weights))).sum(axis=0) % 2) * 2 -
               1).reshape(-1)
    cases = {
        str(value): np.full(len(roots), value, np.float32)
        for value in (100.5, 100.25, 127.0, -128.0)
    }
    cases["checker100.5+-1.5"] = np.float32(100.5 + checker * 1.5)
    for name, values in cases.items():
        columns = np.float32(
            roots[:, None] * np.stack((values, np.ones_like(values)), axis=1)
        )
        quantized = columns.astype(np.float16).astype(np.float32)
        moments = {}
        for mode, operands in (("fp32_operands", columns), ("half_operands",
                                                            quantized)):
            square = sum32(np.float32(operands[:, 0] * operands[:, 0]))
            mean = sum32(np.float32(operands[:, 0] * operands[:, 1]))
            norm = sum32(np.float32(operands[:, 1] * operands[:, 1]))
            variance = np.float32(square - np.float32(mean * mean))
            normalized_mean = np.float32(mean / norm)
            normalized_variance = np.float32(
                square / norm - np.float32(normalized_mean * normalized_mean)
            )
            moments[mode] = {
                "mean": float(mean),
                "second_moment": float(square),
                "variance_before_clamp": float(variance),
                "weight_norm": float(norm),
                "variance_with_extra_weight_normalization": float(normalized_variance),
                "low_variance_branch": bool(max(variance, np.float32(0.0)) < 2.0)
            }
        result[name] = moments
    return result


def _motion_step(
    weights: NDArray[np.float32], mode: str, base: float, step: float
) -> float:
    """Filter both constants and measure the binary32 transition.

    :param weights: Motion filter coefficients.
    :param mode: Precision mode for the dot product.
    :param base: First field's sample value.
    :param step: Difference in the second field's sample value.
    :returns: Filtered difference rounded to binary32.
    """
    blurred = []
    for value in (base, base + step):
        vertical = dot(weights, np.full((5, ), value, np.float32), mode)
        horizontal = dot(weights, np.full((5, ), vertical, np.float32), mode)
        blurred.append(horizontal)
    return float(np.float32(blurred[1] - blurred[0]))


def motion_report(weights: NDArray[np.float32]) -> dict[str, dict[str, float]]:
    """Measure constant-field steps after the motion filter.

    :param weights: Filter coefficients in tap order.
    :returns: Filtered transitions by initial value, step and precision mode.
    """
    result = {}
    for base, step in ((127.0, 1.0 / 256.0), (127.0, 1.0 / 16.0), (100.5, 0.25)):
        result[f"{base}+{step}"] = {}
        for mode in MODES:
            result[f"{base}+{step}"][mode] = _motion_step(weights, mode, base, step)
    return result


def wavelet(
    source: NDArray[np.float32], low_weights: NDArray[np.float32],
    high_weights: NDArray[np.float32], mode: str
) -> tuple[NDArray[np.float32], ...]:
    """Apply a separable wavelet level with the selected dot-product precision.

    :param source: Input sample plane.
    :param low_weights: Lowpass coefficients in tap order.
    :param high_weights: Highpass coefficients in tap order.
    :param mode: Numerical or installation mode.
    :returns: Lowpass and three detail bands in DWT order.
    """
    height, width = source.shape
    out_h, out_w = (height + 1) // 2, (width + 1) // 2
    low_rows, high_rows = [], []
    for x in range(4):
        columns = 2 * np.arange(out_w) - 1 + x
        columns = np.where(
            columns < 0, -columns,
            np.where(columns >= width, 2 * width - columns - 1, columns)
        )
        taps = []
        for y in range(4):
            rows = 2 * np.arange(out_h) - 1 + y
            rows = np.where(
                rows < 0, -rows, np.where(rows >= height, 2 * height - rows - 1, rows)
            )
            taps.append(source[np.ix_(rows, columns)])
        samples = np.asarray(taps, np.float32)
        low_rows.append(dot(low_weights, samples, mode))
        high_rows.append(dot(high_weights, samples, mode))
    low_rows, high_rows = np.asarray(low_rows), np.asarray(high_rows)
    return (
        dot(low_weights, low_rows, mode), dot(low_weights, high_rows, mode),
        dot(high_weights, low_rows, mode), dot(high_weights, high_rows, mode)
    )


def _wavelet_levels(
    source: NDArray[np.float32], low: NDArray[np.float32], high: NDArray[np.float32]
) -> dict[str, object]:
    """Compare four DWT levels for one source plane.

    :param source: Initial sample plane.
    :param low: Lowpass tap coefficients.
    :param high: Highpass tap coefficients.
    :returns: Error summaries by level and precision mode.
    """
    levels = {}
    reference = source
    states = {mode: source for mode in MODES[1:]}
    for level in range(4):
        bands = wavelet(reference, low, high, "fp32")
        levels[str(level)] = {}
        for mode in MODES[1:]:
            candidate = wavelet(states[mode], low, high, mode)
            levels[str(level)][mode] = {
                "low": errors(bands[0], candidate[0]),
                "details": errors(np.stack(bands[1:]), np.stack(candidate[1:])),
            }
            states[mode] = candidate[0]
        reference = bands[0]
    return levels


def wavelet_report(low: NDArray[np.float32],
                   high: NDArray[np.float32]) -> dict[str, object]:
    """Measure four wavelet levels against their binary32 results.

    :param low: Lowpass coefficients in tap order.
    :param high: Highpass coefficients in tap order.
    :returns: Per-level error reports by source pattern and precision mode.
    """
    random = np.random.default_rng(20261105)
    sources = {
        "native10": np.float32(random.integers(0, 1024, (65, 67)) * 0.25 - 128),
        "native16": np.float32(random.integers(0, 65536, (65, 67)) / 256.0 - 128)
    }
    result = {}
    for name, source in sources.items():
        result[name] = _wavelet_levels(source, low, high)
    return result


def historical_stage_shares(path: Path) -> dict[str, object]:
    """Read stage shares from a retained benchmark artifact if available.

    :param path: File to read, hash or update.
    :returns: Timing shares by case, or a reason the artifact was unavailable.
    """
    if not path.is_file():
        return {
            "available": False,
            "reason": "optional original raw timing artifact absent"
        }
    record = json.loads(path.read_text())
    cases = {}
    for case in record["cases"]:
        if not case["case"].endswith(("4k", "1080")):
            continue
        runs = [run for run in case["runs"] if run["arm"] == "B"]
        frame = statistics.median(run["core_ms"] for run in runs)
        cases[case["case"]] = {
            "frame_ms": frame,
            "stages": {
                stage: {
                    "ms":
                    statistics.median(run["stages"][stage] for run in runs),
                    "fraction_of_frame":
                    statistics.median(run["stages"][stage] for run in runs) / frame
                }
                for stage in runs[0]["stages"]
            }
        }
    return {
        "available": True,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope":
        "historical retained arm; not a new measurement or covariance-only share",
        "cases": cases
    }


def main() -> None:
    """Write filter precision diagnostics as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = {
        "vif": SHADERS / "vif/vmaf_vif_common.glsl",
        "adm": SHADERS / "adm/vmaf_adm_dwt.comp",
        "motion": SHADERS / "preprocess/vmaf_motion_common.glsl",
        "speed": ROOT / "src/metrics/vmaf/speed/shaders/vmaf_speed_filter_common.glsl"
    }
    filters = {
        f"vif{count}": coefficients(paths["vif"], f"kVif{count}", count)
        for count in (17, 9, 5, 3)
    }
    filters.update(
        {
            "adm_low": coefficients(paths["adm"], "kLow", 4),
            "adm_high": coefficients(paths["adm"], "kHigh", 4),
            "motion5": coefficients(paths["motion"], "kMotionFilter", 5),
            "speed9": coefficients(paths["speed"], "kAntialias", 9),
            "speed3": coefficients(paths["speed"], "kMean", 3)
        }
    )
    result = {
        "schema_version":
        1,
        "numpy":
        np.__version__,
        "seed":
        20261105,
        "arithmetic":
        "explicit separate FP32 products/additions; binary16 operands; no driver summation emulation",
        "split_scale":
        float(SPLIT_SCALE),
        "shader_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths.values()
        },
        "native_operands":
        operand_report(),
        "coefficients":
        coefficient_report(filters),
        "dots":
        dot_report(filters),
        "vif_constant_precomputed_moments":
        vif_report(filters["vif17"]),
        "vif_constant_direct_weighted_gram":
        weighted_gram_report(filters["vif17"]),
        "motion_constant_steps":
        motion_report(filters["motion5"]),
        "adm_four_levels":
        wavelet_report(filters["adm_low"], filters["adm_high"]),
        "historical_stages":
        historical_stage_shares(
            ROOT / "artifacts/reports/vmaf-fp16/production-native-resident/results.json"
        )
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.out}")
    for depth, report in result["native_operands"].items():
        print(
            f"depth {depth}: half max error {report['half']['maximum']}; "
            f"split max error {report['high_plus_scaled_residual']['maximum']}"
        )


if __name__ == "__main__":
    main()
