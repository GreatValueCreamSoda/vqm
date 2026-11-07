# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Validate complete normal-CLI clip results before accepting their evidence."""

from __future__ import annotations

import math

FEATURE_COUNTS = {
    "ssimulacra2": 0,
    "butteraugli": 0,
    "vmaf-v0": 6,
    "vmaf-v1": 4
}


def _finite(value: object) -> bool:
    """Recognize finite JSON numbers without accepting booleans or strings.

    :param value: Parsed JSON scalar or container.
    :returns: Whether the value is a finite JSON number.
    """
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _frame_score(value: object, index: int, features: int) -> float:
    """Require one finite frame and its model's complete feature vector.

    :param value: Parsed frame result.
    :param index: Expected zero-based position in the complete clip.
    :param features: Required model feature count, or zero for spatial metrics.
    :returns: Validated finite frame score.
    :raises ValueError: If the frame identity, score or features are invalid.
    """
    if not isinstance(value, dict):
        raise ValueError("candidate frame result must be an object")
    if type(value.get("index")) is not int or value["index"] != index:
        raise ValueError(
            "candidate frame results must have complete ordered indices"
        )
    if not _finite(value.get("score")):
        raise ValueError(
            "candidate returned a nonfinite or invalid frame score"
        )
    if features:
        values = value.get("features")
        if not isinstance(values, list) or len(values) != features:
            raise ValueError("candidate frame has an incomplete feature vector")
        if not all(_finite(item) for item in values):
            raise ValueError(
                "candidate frame has nonfinite or invalid features"
            )
    return float(value["score"])


def _validate_frames(
    value: dict[str, object], metric: str, frames: int
) -> None:
    """Require frames and their mean, or CVVDP's clip-only result.

    :param value: Clip result with a validated finite score.
    :param metric: Qualified metric name.
    :param frames: Expected positive frame count.
    :raises ValueError: If frame evidence is incomplete or pooling is
        inconsistent.
    """
    results = value.get("frame_results")
    if not isinstance(results, list):
        raise ValueError("candidate must include a frame_results array")
    if metric == "cvvdp":
        if results:
            raise ValueError(
                "CVVDP must return a clip score without frame scores"
            )
        return
    if len(results) != frames:
        raise ValueError("candidate returned incomplete frame results")
    scores = [
        _frame_score(frame, index, FEATURE_COUNTS[metric])
        for index, frame in enumerate(results)
    ]
    mean = sum(scores) / frames
    if not math.isfinite(mean) or not math.isclose(mean, float(
            value["score"]), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("candidate clip score does not match its frame mean")


def _validate_timing(value: dict[str, object]) -> None:
    """Require finite, nonnegative decode and positive core times.

    :param value: Parsed CLI clip result.
    :raises ValueError: If a duration is absent, nonnumeric, nonfinite or
        invalid.
    """
    for name in ("core_seconds", "decode_seconds"):
        duration = value.get(name)
        if not _finite(duration) or duration < 0:
            raise ValueError(
                f"candidate returned a nonfinite or invalid {name}"
            )
    if value["core_seconds"] == 0:
        raise ValueError("candidate core_seconds must be positive")


def validate_cli_result(
    value: object, metric: str, sample_id: str, frames: int
) -> None:
    """Require complete clip, frame and timing evidence for one input.

    :param value: Parsed JSON emitted by the normal CLI.
    :param metric: Selected metric in the established reference matrix.
    :param sample_id: Expected input association.
    :param frames: Expected positive complete-clip frame count.
    :raises ValueError: If any required result field is absent or inconsistent.
    """
    if metric not in FEATURE_COUNTS and metric != "cvvdp":
        raise ValueError(f"unsupported qualified metric: {metric}")
    if not isinstance(value, dict):
        raise ValueError("candidate clip result must be an object")
    if value.get("metric") != metric or value.get("sample_id") != sample_id:
        raise ValueError(
            "candidate returned mismatched metric or sample identity"
        )
    if type(value.get("frames")
            ) is not int or value["frames"] != frames or frames < 1:
        raise ValueError(
            "candidate returned mismatched complete-clip frame count"
        )
    if not _finite(value.get("score")):
        raise ValueError("candidate returned a nonfinite or invalid clip score")
    _validate_timing(value)
    _validate_frames(value, metric, frames)
