#!/usr/bin/env python3
# Copyright (c) 2020 Netflix, Inc.
# BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
"""Generate v1 SDR and HDR predictor constants from the pinned model files.

Verify the checked-in output with `python3 tools/generate_vmaf_v1_model.py --check`.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / ".reference/vmaf/model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json"
DEFAULT_OUTPUT = ROOT / "src/metrics/vmaf/v1/vmaf_v1_model_data.inc"
MODEL_SHA256 = "e4cf8c147e1368b35497d772920bc92f98c1ad7853c1033d8a836947f427140e"
# Order matches VmafV1Model. Only the preset parameters differ between files.
OTHER_MODELS = (
    ("1d5h_2160", "634d36e0502cb797384726cdd1a106a2c207827595a9e4958d3dae1bd517b139"),
    ("3d0h_2160", "77f552fd8d5b6c6a2166bfc4470c9368ec68d2425ed40f1cc235b0cc500f33c2"),
    ("5d0h", "64e12d9c1e4ed5d79da7fa515b82fac7247a9e2ccc91e0ebce31d05489500385"),
)
HDR_NAME = "hdr_vmaf_v1.1.8_1d5h_2160.json"
HDR_SHA256 = "53d36d3173012efa6eb65ada92aa3967d3caceee68d088275b2a51a320948510"
FEATURE_NAMES = [
    "Cambi_feature_cambi_score", "Speed_chroma_feature_speed_chroma_uv_score",
    "VMAF_integer_feature_adm3_score", "VMAF_integer_feature_motion3_score"
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


def _load_model(source: Path, digest: str) -> dict[str, Any]:
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"{source}: not the pinned VMAF v1 model")
    model = json.loads(content)["model_dict"]
    if model["feature_names"] != FEATURE_NAMES or model["norm_type"] != "linear_rescale":
        raise ValueError("unexpected VMAF model features or normalization")
    slopes, intercepts = model["slopes"], model["intercepts"]
    if len(slopes) != 5 or len(intercepts) != 5:
        raise ValueError("unexpected normalization size")
    transform, clip = model["score_transform"], model["score_clip"]
    if transform.get("enabled") is not True or len(clip) != 2:
        raise ValueError("unexpected score transform or clipping")
    return model


def _model_row(model: dict[str, Any]) -> str:
    options = model["feature_opts_dicts"]
    height = options[2]["adm_ref_display_height"]
    prescale = number(options[1].get("speed_prescale", 1.0))
    transform = ", ".join(
        number(model["score_transform"][name]) for name in ("p0", "p1", "p2")
    )
    clip = ", ".join(map(number, model["score_clip"]))
    return f"    {{{{{height * 16 // 9}, {height}}}, {prescale}, {{{transform}}}, {{{clip}}}}},"


def _model_rows(source: Path, standard: dict[str, Any]) -> list[str]:
    shared = ("model", "slopes", "intercepts", "chroma_correction_parameter")
    rows = [_model_row(standard)]
    for suffix, digest in OTHER_MODELS:
        variant = _load_model(source.with_name(f"vmaf_v1.0.16_{suffix}.json"), digest)
        if any(variant[key] != standard[key] for key in shared):
            raise ValueError(f"{suffix}: predictor differs from the standard model")
        rows.append(_model_row(variant))
    return rows


def _vector_row(line: str) -> str:
    fields = line.split()
    coefficient = float(fields[0])
    vector = [0.0] * 4
    previous = 0
    for entry in fields[1:]:
        index_text, value = entry.split(":", 1)
        index = int(index_text)
        if index <= previous or index > 4:
            raise ValueError("invalid support vector index")
        vector[index - 1] = float(value)
        previous = index
    return "    {" + number(coefficient) + ", {" + ", ".join(
        map(number, vector)
    ) + "}},"


def _predictor(model: dict[str, Any], prefix: str) -> list[str]:
    header, points = model["model"].split("\nSV\n", 1)
    fields = dict(line.split(maxsplit=1) for line in header.splitlines())
    if (fields["svm_type"], fields["kernel_type"], fields["nr_class"]) != ("nu_svr",
                                                                           "rbf", "2"):
        raise ValueError("unexpected VMAF SVM configuration")
    rows = [_vector_row(line) for line in points.splitlines()]
    if len(rows) != int(fields["total_sv"]):
        raise ValueError("incorrect support vector count")
    return [
        f"inline constexpr std::array<double, 5> k{prefix}Slopes = {{" +
        ", ".join(map(number, model["slopes"])) + "};",
        f"inline constexpr std::array<double, 5> k{prefix}Intercepts = {{" +
        ", ".join(map(number, model["intercepts"])) + "};",
        f"inline constexpr double k{prefix}ChromaCorrection = " +
        number(model["chroma_correction_parameter"]) + ";",
        f"inline constexpr double k{prefix}Gamma = " + number(float(fields["gamma"])) +
        ";",
        f"inline constexpr double k{prefix}Rho = " + number(float(fields["rho"])) + ";",
        f"inline constexpr std::array<SupportVector, {len(rows)}> k{prefix}SupportVectors = {{{{",
        *rows, "}};"
    ]


def generate(source: Path) -> str:
    """Generate the C++ constants of the pinned SDR and HDR v1 predictors.

    :param source: Path to the standard model in the pinned model tree.
    :returns: Complete generated header text.
    :raises OSError: If a source model is unavailable.
    :raises ValueError: If a model differs from its pinned configuration.
    :raises KeyError: If a required model field is missing.
    """
    sdr = _load_model(source, MODEL_SHA256)
    hdr = _load_model(source.parent.with_name("hdr_vmaf_v1.1.8") / HDR_NAME, HDR_SHA256)
    return "\n".join(
        [
            "// Copyright (c) 2020 Netflix, Inc.",
            "// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.",
            "// Generated by tools/generate_vmaf_v1_model.py from Netflix/vmaf",
            "// SDR v1.0.16 and HDR v1.1.8 at 4611e47a7330cd342498dcc130e1008efbeab6e4.",
            "// Model file SHA-256 pins are in the generator. Do not edit by hand.",
            "inline constexpr std::array<VmafV1ModelParameters, 5> kModels = {{",
            *_model_rows(source, sdr),
            _model_row(hdr), "}};", *_predictor(sdr, "Sdr"), *_predictor(hdr, "Hdr"),
            "",
        ]
    )


def main() -> None:
    """Write the generated model table or check the existing table.

    :raises SystemExit: If generation or verification fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        result = generate(args.model)
        if not args.check:
            args.output.write_text(result)
            print(f"wrote {args.output}")
            return
        if args.output.read_text() != result:
            raise ValueError(f"{args.output}: generated constants differ")
        print(f"{args.output}: matches pinned model")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
