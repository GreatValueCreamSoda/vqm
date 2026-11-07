#!/usr/bin/env python3
# Copyright (c) 2020 Netflix, Inc.
# BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
"""Generate the VMAF v0.6.1 float predictor tables from the pinned models.

The models are from Netflix/vmaf commit 4611e47a7330cd342498dcc130e1008efbeab6e4.
The checked-in output is buildable without an upstream checkout. To verify it:

    python3 tools/generate_vmaf_model.py --check
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / ".reference/vmaf/model"
DEFAULT_OUTPUT = ROOT / "src/metrics/vmaf/v0/vmaf_model_data.inc"
FEATURE_NAMES = [
    "VMAF_feature_adm2_score", "VMAF_feature_motion2_score",
    "VMAF_feature_vif_scale0_score", "VMAF_feature_vif_scale1_score",
    "VMAF_feature_vif_scale2_score", "VMAF_feature_vif_scale3_score",
]


class ModelSpec:
    """Pin the expected configuration and checksum of one upstream model."""

    def __init__(
        self, label: str, filename: str, sha256: str, total_sv: int, rho: str,
        transform: bool
    ) -> None:
        """Describe the pinned predictor and its emitted constants.

        :param label: Suffix used in the C++ constant names.
        :param filename: Model filename in the pinned checkout.
        :param sha256: Expected digest of the full model file.
        :param total_sv: Expected support vector count.
        :param rho: Expected SVM intercept literal.
        :param transform: Whether the model has a phone transform.
        """
        self.label = label
        self.path = MODEL_DIR / filename
        self.filename = filename
        self.sha256 = sha256
        self.total_sv = total_sv
        self.rho = rho
        self.transform = transform


MODELS = [
    ModelSpec(
        "1080", "vmaf_float_v0.6.1.json",
        "26573aa333346f1109226e7283969e8b3581119a1ab7f18897767351dd963ff8", 211,
        "-1.33133", True
    ),
    ModelSpec(
        "4K", "vmaf_float_4k_v0.6.1.json",
        "2af34f3068174720a35126807dbe8d0b19c5143fee328a9fc2331c15c54e4433", 262,
        "-2.30449", False
    ),
]


def number(value: float) -> str:
    """Render a finite double with round-trip precision.

    :param value: Model constant.
    :returns: Decimal literal for the generated C++ table.
    :raises ValueError: If the constant is not finite.
    """
    if not math.isfinite(value):
        raise ValueError("model values must be finite")
    return format(value, ".17g")


def _load_metadata(
    spec: ModelSpec,
) -> tuple[list[float], list[float], dict[str, float] | None, list[float], str]:
    content = spec.path.read_bytes()
    if hashlib.sha256(content).hexdigest() != spec.sha256:
        raise ValueError(f"{spec.path}: not the pinned {spec.label} model")
    model = json.loads(content)["model_dict"]
    if model["feature_names"] != FEATURE_NAMES:
        raise ValueError(f"{spec.path}: unexpected feature names")
    if model["norm_type"] != "linear_rescale":
        raise ValueError(f"{spec.path}: unexpected normalization")
    slopes, intercepts = model["slopes"], model["intercepts"]
    if len(slopes) != 7 or len(intercepts) != 7:
        raise ValueError(f"{spec.path}: unexpected normalization size")
    clip = model["score_clip"]
    if len(clip) != 2:
        raise ValueError(f"{spec.path}: unexpected score clip")
    transform = model.get("score_transform")
    if spec.transform != (transform
                          is not None) or (transform is not None
                                           and transform.get("out_gte_in") != "true"):
        raise ValueError(f"{spec.path}: unexpected score transform")
    return slopes, intercepts, transform, clip, model["model"]


def _vector_row(line: str, spec: ModelSpec) -> str:
    fields = line.split()
    coefficient = float(fields[0])
    vector = [0.0] * 6
    previous = 0
    for entry in fields[1:]:
        index_text, value = entry.split(":", 1)
        index = int(index_text)
        if index <= previous or index > 6:
            raise ValueError(f"{spec.path}: invalid support vector index")
        vector[index - 1] = float(value)
        previous = index
    return "    {" + number(coefficient) + ", {" + ", ".join(
        map(number, vector)
    ) + "}},"


def load_model(
    spec: ModelSpec,
) -> tuple[list[float], list[float], dict[str, float] | None, list[float], list[str]]:
    """Load and validate the constants of a pinned v0 model.

    :param spec: Expected model configuration and digest.
    :returns: Slopes, intercepts, score transform, clip bounds and C++ vector rows.
    :raises OSError: If the source model is unavailable.
    :raises ValueError: If the model differs from its pinned configuration.
    :raises KeyError: If a required model field is missing.
    """
    slopes, intercepts, transform, clip, svm = _load_metadata(spec)
    header, points = svm.split("\nSV\n", 1)
    fields = dict(line.split(maxsplit=1) for line in header.splitlines())
    expected = {
        "svm_type": "nu_svr",
        "kernel_type": "rbf",
        "gamma": "0.04",
        "nr_class": "2",
        "total_sv": str(spec.total_sv),
        "rho": spec.rho
    }
    if fields != expected:
        raise ValueError(f"{spec.path}: unexpected SVM configuration")
    rows = [_vector_row(line, spec) for line in points.splitlines()]
    if len(rows) != spec.total_sv:
        raise ValueError(f"{spec.path}: incorrect support vector count")
    return slopes, intercepts, transform, clip, rows


def emit_model(
    spec: ModelSpec, slopes: list[float], intercepts: list[float],
    transform: dict[str, float] | None, clip: list[float], rows: list[str]
) -> list[str]:
    """Emit the C++ constants of one predictor.

    :param spec: Expected model configuration.
    :param slopes: Feature and score normalization slopes.
    :param intercepts: Feature and score normalization intercepts.
    :param transform: Optional phone score transform coefficients.
    :param clip: Lower and upper score bounds.
    :param rows: Rendered support vector rows.
    :returns: C++ declaration lines.
    """
    suffix = spec.label
    out = [
        f"inline constexpr std::array<double, 7> kSlopes{suffix} = {{" +
        ", ".join(number(x) for x in slopes) + "};",
        f"inline constexpr std::array<double, 7> kIntercepts{suffix} = {{" +
        ", ".join(number(x) for x in intercepts) + "};",
        f"inline constexpr std::array<double, 2> kClip{suffix} = {{" +
        ", ".join(number(x) for x in clip) + "};",
        f"inline constexpr double kGamma{suffix} = 0.04;",
        f"inline constexpr double kRho{suffix} = {spec.rho};",
    ]
    if transform is not None:
        out.append(
            f"inline constexpr std::array<double, 3> kTransform{suffix} = {{" +
            ", ".join(number(transform[name]) for name in ("p0", "p1", "p2")) + "};"
        )
    out.append(
        f"inline constexpr std::array<SupportVector, {spec.total_sv}> "
        f"kSupportVectors{suffix} = " + "{{"
    )
    out.extend(rows)
    out.append("}};")
    out.append("")
    return out


def generate() -> str:
    """Generate the checked-in C++ header from both pinned predictors.

    :returns: Complete generated header text.
    :raises OSError: If a source file is unavailable.
    :raises ValueError: If a model does not match its pin.
    """
    lines = [
        "// Copyright (c) 2020 Netflix, Inc.",
        "// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.",
        "// Generated by tools/generate_vmaf_model.py from Netflix/vmaf:",
    ]
    for spec in MODELS:
        lines.append(
            f"//   {spec.filename} at 4611e47a7330cd342498dcc130e1008efbeab6e4"
        )
        lines.append(f"//   SHA-256: {spec.sha256}")
    lines.extend(["// Do not edit by hand.", ""])
    for spec in MODELS:
        lines.extend(emit_model(spec, *load_model(spec)))
    return "\n".join(lines)


def main() -> None:
    """Write the generated model table or check the existing table.

    :raises SystemExit: If generation or verification fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        result = generate()
        if not args.check:
            args.output.write_text(result)
            print(f"wrote {args.output}")
            return
        if args.output.read_text() != result:
            raise ValueError(f"{args.output}: generated constants differ")
        print(f"{args.output}: matches pinned models")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
