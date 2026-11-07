"""Illustrate VIF/ADM binary16 limits; not a GPU oracle or timing benchmark.

Promoted from the FP16 research report's ``experiments/vif-adm-numerics.py``.
The explicit casts and operation order retain the original diagnostic arithmetic.
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts/reports/precision-examples/vmaf-precision.json"

# Illustrative axial/diagonal CSF weights used in the original range experiment.
CSF_WEIGHTS = np.asarray(
    [
        [0.01183, 0.004302], [0.025026, 0.011778], [0.04295, 0.023918],
        [0.058621, 0.035901],
    ], dtype=np.float32,
)


def shader_coefficients(path: Path, name: str, count: int) -> NDArray[np.float32]:
    """Read a literal GLSL coefficient array, failing visibly if its form changes.

    :param path: File to read, hash or update.
    :param name: Name of the coefficient array, stage or case.
    :param count: Expected number of coefficients or samples.
    :returns: Coefficient array in binary32 tap order.
    :raises ValueError: If the array is missing or has the wrong number of coefficients.
    """
    source = path.read_text()
    pattern = rf"{name}\[{count}\].*?\((.*?)\);"
    match = re.search(pattern, source, re.S)
    if match is None:
        raise ValueError(f"Could not find {name}[{count}] in {path}")
    values = [float(value) for value in match.group(1).split(",")]
    if len(values) != count:
        raise ValueError(f"Expected {count} coefficients for {name} in {path}")
    return np.asarray(values, dtype=np.float32)


def add32(first: NDArray[np.float32],
          second: NDArray[np.float32]) -> NDArray[np.float32]:
    """Add two arrays with binary32 rounding.

    :param first: First addend.
    :param second: Second addend.
    :returns: Rounded sum in binary32 precision.
    """
    return np.float32(first + second)


def _constant_products(values: NDArray[np.float32],
                       half_products: bool) -> NDArray[np.float32]:
    """Form both squares and the cross product at the selected precision.

    :param values: Reference and distorted constants.
    :param half_products: Whether products round to binary16.
    :returns: Two square terms and one cross term widened to binary32.
    """
    reference_value, distorted_value = values
    if half_products:
        half = values.astype(np.float16)
        products = np.asarray(
            [
                np.float16(half[0] * half[0]),
                np.float16(half[1] * half[1]),
                np.float16(half[0] * half[1]),
            ], dtype=np.float32,
        )
    else:
        products = np.asarray(
            [
                np.float32(reference_value * reference_value),
                np.float32(distorted_value * distorted_value),
                np.float32(reference_value * distorted_value),
            ], dtype=np.float32,
        )

    return products


def _vif_ratio(
    variance_reference: float, variance_distorted: float, covariance: float
) -> float:
    """Form the VIF ratio using the original low-variance and gain rules.

    :param variance_reference: Clamped reference variance.
    :param variance_distorted: Clamped distorted variance.
    :param covariance: Reference/distorted covariance.
    :returns: VIF numerator divided by denominator.
    """
    if variance_reference < 2.0:
        numerator = 1.0 - variance_distorted * (4.0 / (255.0 * 255.0))
        denominator = 1.0
    else:
        gain = covariance / (variance_reference + 1e-10)
        residual = variance_distorted - gain * covariance
        if variance_distorted < 1e-10:
            gain, residual = 0.0, 0.0
        if gain < 0.0:
            residual, gain = variance_distorted, 0.0
        gain = min(gain, 100.0)
        residual = max(residual, 1e-10)
        numerator = math.log2(
            1.0 + (gain * gain * variance_reference) / (residual + 2.0)
        )
        denominator = math.log2(1.0 + variance_reference / 2.0)
        if covariance < 0.0:
            numerator = 0.0
    return numerator / denominator


def vif_constant(
    reference: float, distorted: float, weights: NDArray[np.float32],
    half_products: bool = False, center: tuple[float,
                                               float] = (0.0, 0.0), scale: float = 1.0
) -> dict[str, object]:
    """Filter one constant field with separate FP32 additions, then form VIF.

    :param reference: Reference values used for comparison.
    :param distorted: Distorted sample value.
    :param weights: Filter coefficients in tap order.
    :param half_products: Whether to round individual products to binary16.
    :param center: Reference and distorted recentering values.
    :param scale: Operand scaling factor.
    :returns: Filtered moments, branch selection and VIF ratio.
    """
    values = np.asarray(
        [(reference - center[0]) / scale, (distorted - center[1]) / scale],
        dtype=np.float32,
    )
    products = _constant_products(values, half_products)

    means = np.zeros(2, dtype=np.float32)
    moments = np.zeros(3, dtype=np.float32)
    for weight in weights:
        means = add32(means, np.float32(weight * values))
        moments = add32(moments, np.float32(weight * products))
    horizontal_means = np.zeros(2, dtype=np.float32)
    horizontal_moments = np.zeros(3, dtype=np.float32)
    for weight in weights:
        horizontal_means = add32(horizontal_means, np.float32(weight * means))
        horizontal_moments = add32(horizontal_moments, np.float32(weight * moments))

    mean = np.float32(horizontal_means * np.float32(scale))
    moment = np.float32(horizontal_moments * np.float32(scale * scale))
    variance_reference = max(
        float(np.float32(moment[0] - np.float32(mean[0] * mean[0]))), 0.0
    )
    variance_distorted = max(
        float(np.float32(moment[1] - np.float32(mean[1] * mean[1]))), 0.0
    )
    covariance = float(np.float32(moment[2] - np.float32(mean[0] * mean[1])))
    ratio = _vif_ratio(variance_reference, variance_distorted, covariance)
    return {
        "means": mean.tolist(),
        "moments": moment.tolist(),
        "variance_reference": variance_reference,
        "variance_distorted": variance_distorted,
        "covariance": covariance,
        "low_variance_branch": variance_reference < 2.0,
        "ratio": ratio,
    }


def _wavelet_product(weight: np.float32, value: NDArray[np.floating],
                     mode: str) -> NDArray[np.floating]:
    """Multiply one DWT tap using the configured rounding.

    :param weight: Tap coefficient.
    :param value: Values sampled for the tap.
    :param mode: Product precision mode.
    :returns: Product rounded to binary32 or binary16.
    """
    if mode == "fp32":
        return np.float32(weight * value)
    return np.float16(np.float16(weight) * np.float16(value))


def wavelet(
    source: NDArray[np.float32], mode: str, low_weights: NDArray[np.float32],
    high_weights: NDArray[np.float32]
) -> list[NDArray[np.float32]]:
    """Apply one separable DWT level with the selected product/sum precision.

    :param source: Input sample plane.
    :param mode: Numerical or installation mode.
    :param low_weights: Lowpass coefficients in tap order.
    :param high_weights: Highpass coefficients in tap order.
    :returns: Lowpass and three detail bands in DWT order.
    """
    height, width = source.shape
    output_height, output_width = (height + 1) // 2, (width + 1) // 2
    dtype = np.float16 if mode == "fp16_all" else np.float32
    rows_low, rows_high = [], []
    for x in range(4):
        columns = 2 * np.arange(output_width) - 1 + x
        columns = np.where(
            columns < 0, -columns,
            np.where(columns >= width, 2 * width - columns - 1, columns),
        )
        low = np.zeros((output_height, output_width), dtype=dtype)
        high = low.copy()
        for y in range(4):
            rows = 2 * np.arange(output_height) - 1 + y
            rows = np.where(
                rows < 0, -rows, np.where(rows >= height, 2 * height - rows - 1, rows)
            )
            value = source[np.ix_(rows, columns)]
            low_product = _wavelet_product(low_weights[y], value, mode)
            high_product = _wavelet_product(high_weights[y], value, mode)
            low = dtype(low + low_product.astype(dtype))
            high = dtype(high + high_product.astype(dtype))
        rows_low.append(low)
        rows_high.append(high)

    bands = [np.zeros((output_height, output_width), dtype=dtype) for _ in range(4)]
    for x in range(4):
        for band, weight, value in [
            (0, low_weights[x], rows_low[x]), (1, low_weights[x], rows_high[x]),
            (2, high_weights[x], rows_low[x]), (3, high_weights[x], rows_high[x]),
        ]:
            product = _wavelet_product(weight, value, mode)
            bands[band] = dtype(bands[band] + product.astype(dtype))
    return [np.asarray(band, dtype=np.float32) for band in bands]


def _adm_levels(
    source: NDArray[np.float32], mode: str, low_weights: NDArray[np.float32],
    high_weights: NDArray[np.float32]
) -> list[dict[str, float | bool]]:
    """Measure the range of coefficients over four DWT levels.

    :param source: Initial sample plane.
    :param mode: DWT product and sum precision.
    :param low_weights: Lowpass coefficients.
    :param high_weights: Highpass coefficients.
    :returns: Coefficient and weighted-cube bounds by level.
    """
    current = source
    levels = []
    for level in range(4):
        bands = wavelet(current, mode, low_weights, high_weights)
        weighted = np.asarray(
            [
                abs(bands[1]) * CSF_WEIGHTS[level, 0],
                abs(bands[2]) * CSF_WEIGHTS[level, 0],
                abs(bands[3]) * CSF_WEIGHTS[level, 1],
            ], dtype=np.float32,
        )
        maximum = float(weighted.max())
        levels.append(
            {
                "max_abs_lowpass": float(abs(bands[0]).max()),
                "max_abs_detail": float(max(abs(band).max() for band in bands[1:])),
                "max_csf_weighted": maximum,
                "max_cube_fp32": maximum**3,
                "cube_exceeds_binary16": maximum**3 > 65504,
            }
        )
        current = bands[0]
    return levels


def generate_examples() -> dict[str, object]:
    """Measure representative binary16 operand limits for VIF and ADM.

    :returns: Binary16 range, coded-source, VIF and ADM diagnostics.
    """
    shader_root = ROOT / "src/metrics/vmaf/features/shaders"
    vif_weights = shader_coefficients(
        shader_root / "vif/vmaf_vif_common.glsl", "kVif17", 17
    )
    adm_path = shader_root / "adm/vmaf_adm_dwt_decouple.comp"
    low_weights = shader_coefficients(adm_path, "kLow", 4)
    high_weights = shader_coefficients(adm_path, "kHigh", 4)
    results = {
        "note":
        "IEEE binary16 CPU diagnostics; dot additions are separate FP32 operations, not a claim of exact GPU driver contraction behavior.",
        "finfo": {
            "max": float(np.finfo(np.float16).max),
            "smallest_normal": float(np.finfo(np.float16).smallest_normal),
            "smallest_subnormal": float(np.nextafter(np.float16(0), np.float16(1))),
        },
        "guards": {
            str(value): float(np.float16(value))
            for value in [1e-10, 1e-30, 0.9996954135]
        },
        "coded_source": {},
        "vif": {},
        "adm": {},
    }

    for depth in [8, 10, 12, 16]:
        values = (
            np.arange(1 << depth, dtype=np.float32) * np.float32(2.0**(8 - depth)) -
            np.float32(128.0)
        )
        errors = np.abs(values.astype(np.float16).astype(np.float32) - values)
        results["coded_source"][str(depth)] = {
            "all_codes_exact": bool(np.all(errors == 0)),
            "maximum_error": float(errors.max()),
            "maximum_native_codes": float(errors.max() / 2.0**(8 - depth)),
        }
    for name, half, center, scale in [
        ("fp32", False, (0, 0), 1), ("fp16_products_fp32_sums", True, (0, 0), 1),
        ("recentered_uniform_field", True, (100.5, 100.25), 2),
        ("recentered_heterogeneous_tile_center_zero", True, (0, 0), 2),
    ]:
        results["vif"][name] = vif_constant(
            100.5, 100.25, vif_weights, half, center, scale
        )
    results["vif"]["exact_unfiltered_products"] = {
        "reference_square": 100.5**2,
        "distorted_square": 100.25**2,
        "cross": 100.5 * 100.25,
    }
    results["vif"]["half_unfiltered_products"] = {
        "reference_square": float(np.float16(np.float16(100.5) * np.float16(100.5))),
        "distorted_square": float(np.float16(np.float16(100.25) * np.float16(100.25))),
        "cross": float(np.float16(np.float16(100.5) * np.float16(100.25))),
    }

    for name in ["constant", "checker16", "stripes16"]:
        y, x = np.indices((256, 256))
        if name == "constant":
            source = np.full((256, 256), 100.5, dtype=np.float32)
        else:
            pattern = x // 16 + y // 16 if name == "checker16" else x // 16
            source = np.where(pattern % 2, 127.75, -128.0).astype(np.float32)
        by_mode = {}
        for mode in ["fp32", "fp16_products", "fp16_all"]:
            by_mode[mode] = _adm_levels(source, mode, low_weights, high_weights)
        results["adm"][name] = by_mode
    return results


def main() -> None:
    """Write VIF and ADM precision examples and print selected results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path"
    )
    args = parser.parse_args()
    results = generate_examples()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                "vif": results["vif"],
                "coded_source": results["coded_source"],
                "guards": results["guards"],
                "adm_coarse": {
                    name: {
                        mode: levels[-1]
                        for mode, levels in modes.items()
                    }
                    for name, modes in results["adm"].items()
                },
            }, indent=2,
        )
    )


if __name__ == "__main__":
    main()
