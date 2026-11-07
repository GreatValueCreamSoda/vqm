"""Illustrate motion quantization; not a GPU oracle or performance benchmark.

Promoted from the FP16 research report's ``motion/cpu_precision_examples.py``.
The explicit casts and operation order retain the original diagnostic arithmetic.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts/reports/precision-examples/vmaf-motion-precision.json"
WEIGHTS = np.array(
    [0.054488685, 0.244201342, 0.402619947, 0.244201342, 0.054488685], np.float32
)


def blur(source: NDArray[np.float32], kind: str) -> NDArray[np.float32]:
    """Apply a reflected five-tap blur with explicit precision boundaries.

    :param source: Input sample plane.
    :param kind: Precision mode or pattern label.
    :returns: Filtered image plane widened to binary32.
    """
    data = source if kind == "fp32" else source.astype(np.float16).astype(np.float32)
    height, width = data.shape
    accumulator = np.float16 if kind == "half_alu" else np.float32
    vertical_source = np.pad(data, ((2, 2), (0, 0)), mode="reflect")
    vertical = np.zeros((height, width), accumulator)
    for tap in range(5):
        sample = vertical_source[tap:tap + height]
        if kind in ("half_alu", "half_products"):
            product = np.multiply(
                np.float16(WEIGHTS[tap]), sample.astype(np.float16), dtype=np.float16
            )
        else:
            product = np.multiply(WEIGHTS[tap], sample, dtype=np.float32)
        vertical = np.add(vertical, product.astype(accumulator), dtype=accumulator)
    if kind in ("half_shared", "half_alu", "half_products"):
        vertical = vertical.astype(np.float16).astype(np.float32)

    horizontal_source = np.pad(vertical, ((0, 0), (2, 2)), mode="reflect")
    result = np.zeros((height, width), accumulator)
    for tap in range(5):
        sample = horizontal_source[:, tap:tap + width]
        if kind in ("half_alu", "half_products"):
            product = np.multiply(
                np.float16(WEIGHTS[tap]), sample.astype(np.float16), dtype=np.float16
            )
        else:
            product = np.multiply(WEIGHTS[tap], sample, dtype=np.float32)
        result = np.add(result, product.astype(accumulator), dtype=accumulator)
    return result.astype(np.float32)


def _compare_blurs(
    depth: int, content: str, original: NDArray[np.float32],
    changed: NDArray[np.float32], start: float, scale: float
) -> dict[str, object]:
    """Compare the candidate blurs with binary32 for one image pair.

    :param depth: Native source bit depth.
    :param content: Pair pattern label.
    :param original: Original sample plane.
    :param changed: Changed sample plane.
    :param start: Original code for the near-flat pattern.
    :param scale: Difference in the near-flat pattern.
    :returns: Reference transition and error summaries for each precision mode.
    """
    reference = [blur(image, "fp32") for image in (original, changed)]
    expected = float(np.mean(np.abs(reference[1] - reference[0]), dtype=np.float64))
    row = {
        "depth": depth,
        "content": content,
        "fp32_transition": expected,
        "variants": {},
    }
    if content == "one_code_near_high":
        row.update(first_luma=start, second_luma=start + scale)
    for kind in ("half_input", "half_shared", "half_alu", "half_products"):
        candidate = [blur(image, kind) for image in (original, changed)]
        value = float(np.mean(np.abs(candidate[1] - candidate[0]), dtype=np.float64))
        row["variants"][kind] = {
            "transition":
            value,
            "absolute_transition_error":
            abs(value - expected),
            "maximum_blur_error":
            max(
                float(np.max(np.abs(first - second)))
                for first, second in zip(candidate, reference, strict=True)
            ),
        }
    return row


def generate_examples() -> dict[str, object]:
    """Compare motion blur precision for translated and near-flat inputs.

    :returns: Filter weights and per-case transition and blur errors.
    """
    random = np.random.default_rng(46324)
    shape = (17, 33)
    results = []
    for depth in (8, 10, 12, 16):
        scale = 2.0**(8 - depth)
        codes = random.integers(0, 1 << depth, size=shape)
        first = (codes * scale - 128).astype(np.float32)
        start = 126.0 if depth == 8 else 127.0
        pairs = [
            ("translated_random", first, np.roll(first, 1, axis=1)),
            (
                "one_code_near_high", np.full(shape, start, np.float32),
                np.full(shape, start + scale, np.float32),
            ),
        ]
        for content, original, changed in pairs:
            results.append(
                _compare_blurs(depth, content, original, changed, start, scale)
            )
    return {
        "note":
        "CPU illustrative sequential FP32/FP16 simulation; not a GPU oracle or performance result. GPU FMA contraction can change values.",
        "numpy_version": np.__version__,
        "weights_fp32": WEIGHTS.astype(float).tolist(),
        "weights_fp16": WEIGHTS.astype(np.float16).astype(float).tolist(),
        "weights_fp16_sum_fp64":
        float(np.sum(WEIGHTS.astype(np.float16), dtype=np.float64)),
        "results": results,
    }


def main() -> None:
    """Write motion precision examples and print the near-flat results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path"
    )
    args = parser.parse_args()
    result = generate_examples()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["results"]:
        if row["content"] == "one_code_near_high":
            print(
                row["depth"], row["fp32_transition"], {
                    kind: value["transition"]
                    for kind, value in row["variants"].items()
                },
            )


if __name__ == "__main__":
    main()
