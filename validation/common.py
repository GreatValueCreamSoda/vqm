#!/usr/bin/env python3
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
"""Provide corpus pair enumeration, score parsing and rank correlations.

Everything here is standard library plus numpy. The correlation code is the
implementation the previous single-reference runner used, moved unchanged so
old and new reports mean the same thing.
"""

import csv
import math
import os
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

Pair = tuple[str, str]
ScoreResult = tuple[float | None, str | None]


def _codec_pairs(compressed: str, name: str, codec: str) -> list[Pair]:
    directory = f"{compressed}/{name}/{codec}"
    return [
        (f"original/{name}.png", f"compressed/{name}/{codec}/{entry}")
        for entry in sorted(os.listdir(directory))
        if entry.lower().endswith((".png", ".jpg"))
    ]


def enumerate_pairs(data_root: str | Path) -> list[Pair]:
    """List every original and distortion pair in the CID22 validation set.

    :param data_root: Extracted dataset directory.
    :returns: Dataset-relative image paths in source, codec and filename order.
    """
    pairs = []
    compressed = f"{data_root}/compressed"
    for name in sorted(os.listdir(compressed)):
        for codec in sorted(os.listdir(f"{compressed}/{name}")):
            pairs.extend(_codec_pairs(compressed, name, codec))
    return pairs


def _parse_mcos(rows: Iterable[dict[str, str]]) -> dict[Pair, float]:
    scores = {}
    for row in rows:
        value = float(row["MCOS"])
        if not math.isfinite(value):
            raise ValueError("MCOS contains a non-finite value")
        scores[(row["reference_img"], row["distorted_img"])] = value
    return scores


def load_mcos(path: str | Path) -> dict[Pair, float]:
    """Read subjective MCOS values indexed by the CSV's image pairs.

    :param path: CID22 annotations CSV.
    :returns: MCOS keyed by reference and distorted paths.
    :raises ValueError: If an MCOS value is nonfinite or cannot be parsed.
    """
    with open(path, newline="") as stream:
        return _parse_mcos(csv.DictReader(stream))


def parse_score(stdout: str, score_line: str = "last") -> float:
    """Read a finite score from the selected nonempty line of stdout.

    :param stdout: Tool output.
    :param score_line: Select the first line with "first" or the last otherwise.
    :returns: The parsed score.
    :raises ValueError: If the selected line does not contain a finite number.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty output")
    text = lines[0] if score_line == "first" else lines[-1]
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("non-finite score")
    return value


def run_score(
    command: Sequence[str], original: str, distorted: str, timeout: float = 120,
    score_line: str = "last"
) -> ScoreResult:
    """Run one metric command on an image pair.

    :param command: Argument vector with optional image path placeholders.
    :param original: Reference image path.
    :param distorted: Distorted image path.
    :param timeout: Maximum command runtime in seconds.
    :param score_line: Select the first or last nonempty line of stdout.
    :returns: The score or a diagnostic on failure.
    """
    argv = [
        part.replace("{original}", original).replace("{distorted}", distorted)
        for part in command
    ]
    if not any("{original}" in part for part in command):
        argv += [original, distorted]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout}s"
    except OSError as error:
        return None, f"could not run {argv[0]}: {error}"
    if result.returncode != 0:
        return None, f"exit {result.returncode}: {result.stderr.strip()[:200]}"
    try:
        return parse_score(result.stdout, score_line), None
    except ValueError as error:
        return None, (f"unparsable output: {error}: "
                      f"{result.stdout.strip()[:200]}")


def rankdata(values: ArrayLike) -> NDArray[np.float64]:
    """Assign ascending ranks, averaging tied ranks.

    :param values: One-dimensional numeric observations.
    :returns: Ranks in the same order as the input observations.
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(x: ArrayLike, y: ArrayLike) -> float:
    """Compute Pearson correlation for paired observations.

    :param x: First sequence of values.
    :param y: Second sequence of values.
    :returns: Correlation, or NaN when undefined.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denominator = math.sqrt(float(np.dot(xm, xm)) * float(np.dot(ym, ym)))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(xm, ym)) / denominator


def spearman(x: ArrayLike, y: ArrayLike) -> float:
    """Compute Spearman correlation from averaged ranks.

    :param x: First sequence of values.
    :param y: Second sequence of values.
    :returns: Rank correlation, or NaN when undefined.
    """
    return pearson(rankdata(x), rankdata(y))


def kendall_tau_b(x: ArrayLike, y: ArrayLike, block: int = 256) -> float:
    """Compute Kendall tau-b with ties and bounded scratch space.

    :param x: First sequence of values.
    :param y: Second sequence of values.
    :param block: Maximum number of rows per temporary comparison array.
    :returns: Tau-b, or NaN when undefined.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 2:
        return float("nan")
    concordance = 0.0
    ties_x = 0
    ties_y = 0
    columns = np.arange(n)
    for start in range(0, n, block):
        stop = min(start + block, n)
        rows = np.arange(start, stop)
        mask = columns[None, :] > rows[:, None]
        dx = np.where(mask, np.sign(x[start:stop, None] - x[None, :]), 0.0)
        dy = np.where(mask, np.sign(y[start:stop, None] - y[None, :]), 0.0)
        concordance += float(np.sum(dx * dy))
        ties_x += int(np.sum((dx == 0.0) & mask))
        ties_y += int(np.sum((dy == 0.0) & mask))
    pairs = n * (n - 1) / 2.0
    denominator = math.sqrt((pairs - ties_x) * (pairs - ties_y))
    if denominator == 0.0:
        return float("nan")
    return concordance / denominator


def correlation_triple(scores: ArrayLike,
                       mcos: ArrayLike) -> tuple[float, float, float]:
    """Compare one implementation's scores against subjective MCOS.

    :param scores: Metric values in corpus pair order.
    :param mcos: Subjective values in matching corpus pair order.
    :returns: Kendall tau-b, Spearman and Pearson correlations.
    """
    return (kendall_tau_b(scores, mcos), spearman(scores, mcos), pearson(scores, mcos))
