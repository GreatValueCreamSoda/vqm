#!/usr/bin/env python3
"""Check early-QR certificates against an unchanged FP32 rank-zero search.

This deterministic CPU diagnostic is not a GPU arithmetic emulator or oracle.
It includes arbitrary symmetric tridiagonal matrices; those are conditioning
stress examples, not claimed realizable covariance matrices from image inputs.
"""

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[3]
F32 = np.float32
THRESHOLD = F32(1e-6)
PIVOT_FLOOR = F32(1e-37)


def enclosing_bounds(diagonal: NDArray[np.float32],
                     edges: NDArray[np.float32]) -> tuple[np.float32, np.float32]:
    """Bound eigenvalues using the tridiagonal row sums.

    :param diagonal: Diagonal of the tridiagonal system.
    :param edges: Adjacent off-diagonal coefficients.
    :returns: Lower and upper padded binary32 eigenvalue bounds.
    """
    lower = F32(diagonal[0] - abs(edges[0]))
    upper = F32(diagonal[0] + abs(edges[0]))
    for row in range(1, 25):
        radius = abs(edges[row - 1])
        if row < 24:
            radius = F32(radius + abs(edges[row]))
        lower = min(lower, F32(diagonal[row] - radius))
        upper = max(upper, F32(diagonal[row] + radius))
    padding = F32(THRESHOLD * max(abs(lower), abs(upper)))
    return F32(lower - padding), F32(upper + padding)


def sturm_count(
    diagonal: NDArray[np.float32], edges: NDArray[np.float32], x: np.float32
) -> int:
    """Count eigenvalues below a trial point with the binary32 recurrence.

    :param diagonal: Diagonal of the tridiagonal system.
    :param edges: Adjacent off-diagonal coefficients.
    :param x: Trial eigenvalue used by the Sturm recurrence.
    :returns: Count below the trial point, or the invalid-count sentinel.
    """
    pivot = F32(diagonal[0] - x)
    if np.isnan(pivot):
        return 26
    if abs(pivot) < PIVOT_FLOOR:
        pivot = -PIVOT_FLOOR if pivot < 0 else PIVOT_FLOOR
    count = int(pivot < 0)
    for row in range(1, 25):
        edge = edges[row - 1]
        # Explicit noncontracted FP32 operations. The certificate argument does
        # not require this recurrence to be accurate or monotonic, only stable
        # between the prefix and the continuation of that same search.
        quotient = F32(edge / pivot)
        product = F32(edge * quotient)
        pivot = F32(F32(diagonal[row] - x) - product)
        if np.isnan(pivot):
            return 26
        if abs(pivot) < PIVOT_FLOOR:
            pivot = -PIVOT_FLOOR if pivot < 0 else PIVOT_FLOOR
        count += int(pivot < 0)
    return count


def midpoint(lo: np.float32, hi: np.float32) -> np.float32:
    """Compute the midpoint of two binary32 endpoints.

    :param lo: Lower interval endpoint.
    :param hi: Upper interval endpoint.
    :returns: Midpoint rounded at each binary32 operation.
    """
    return F32(F32(lo * F32(0.5)) + F32(hi * F32(0.5)))


def rank_zero(
    diagonal: NDArray[np.float32], edges: NDArray[np.float32], lo: np.float32,
    hi: np.float32
) -> tuple[np.float32, bool, int]:
    """Search for the smallest eigenvalue with the baseline stopping rule.

    :param diagonal: Diagonal of the tridiagonal system.
    :param edges: Adjacent off-diagonal coefficients.
    :param lo: Lower interval endpoint.
    :param hi: Upper interval endpoint.
    :returns: Final eigenvalue estimate, invalid-count flag and step count.
    """
    converged = False
    iterations = 0
    for step in range(1, 321):
        iterations = step
        mid = midpoint(lo, hi)
        if mid in (lo, hi):
            converged = True
            break
        count = sturm_count(diagonal, edges, mid)
        if count > 25:
            break
        if count == 0:
            lo = mid
        else:
            hi = mid
    value = midpoint(lo, hi)
    invalid = not converged or not np.isfinite(value)
    invalid = invalid or sturm_count(diagonal, edges, THRESHOLD) > 25
    return value, bool(invalid), iterations


def prefix_certificate(
    diagonal: NDArray[np.float32], edges: NDArray[np.float32], lo: np.float32,
    hi: np.float32, guard: np.float32, iterations: int
) -> tuple[bool, int, str]:
    """Check whether early interval refinement certifies the QR branch.

    :param diagonal: Diagonal of the tridiagonal system.
    :param edges: Adjacent off-diagonal coefficients.
    :param lo: Lower interval endpoint.
    :param hi: Upper interval endpoint.
    :param guard: Safety margin around the branch threshold.
    :param iterations: Maximum number of iterations.
    :returns: Certificate status, step count and reason.
    """
    for step in range(iterations):
        if hi <= guard:
            return True, step, "upper-endpoint"
        mid = midpoint(lo, hi)
        if mid in (lo, hi):
            return bool(mid <= guard), step + 1, "converged"
        count = sturm_count(diagonal, edges, mid)
        if count > 25:
            return True, step + 1, "rank-zero-failure"
        if count == 0:
            lo = mid
        else:
            hi = mid
    return bool(hi <= guard), iterations, "prefix-limit"


def cases(random_cases: int,
          seed: int) -> Iterator[tuple[str, NDArray[np.float32], NDArray[np.float32]]]:
    """Yield fixed and seeded random tridiagonal systems.

    :param random_cases: Number of additional random cases.
    :param seed: Seed for deterministic random generation.
    :returns: Test cases in reproducible order.
    """
    zero = np.zeros(24, dtype=np.float32)
    yield "zero", np.zeros(25, dtype=np.float32), zero
    yield "identity", np.ones(25, dtype=np.float32), zero
    yield "negative", -np.arange(1, 26, dtype=np.float32), zero
    for name, minimum in (("below-threshold", np.nextafter(THRESHOLD, F32(0))),
                          ("at-threshold", THRESHOLD), ("above-threshold", np.nextafter(
                              THRESHOLD, F32(1))), ("small-positive", F32(1e-30)),
                          ):
        diagonal = np.arange(1, 26, dtype=np.float32)
        diagonal[0] = minimum
        yield name, diagonal, zero
    for exponent in (-30, -20, -10, 0, 10, 20, 30):
        scale = 10.0**exponent
        yield f"toeplitz-{exponent}", np.full(25, 2 * scale, dtype=np.float32
                                              ), np.full(24, -scale, dtype=np.float32)
    rng = np.random.default_rng(seed)
    for index in range(random_cases):
        scale = 10.0**int(rng.integers(-30, 31))
        if index % 3 == 0:
            # Bidiagonal L gives a tridiagonal LL'; rounding may make its
            # smallest mode singular. Dynamic range stresses pivot behavior.
            roots = np.power(10.0, rng.uniform(-8, 0, 25))
            off = rng.uniform(-1, 1, 24)
            diagonal = roots * roots
            diagonal[1:] += off * off
            edges = roots[:-1] * off
        elif index % 3 == 1:
            diagonal = rng.normal(size=25)
            edges = rng.normal(size=24)
        else:
            diagonal = np.power(10.0, rng.uniform(-12, 0, 25))
            edges = rng.uniform(-1, 1, 24) * np.sqrt(diagonal[:-1] * diagonal[1:])
        yield f"random-{index}", F32(diagonal * scale), F32(edges * scale)


def _certificate_gate(
    gate_mode: str, diagonal_gate: bool, diagonal: NDArray[np.float32],
    edges: NDArray[np.float32], guard: np.float32
) -> tuple[bool, int | None]:
    """Decide whether the candidate should attempt an early certificate.

    :param gate_mode: Selected gate policy.
    :param diagonal_gate: Result of the minimum-diagonal test.
    :param diagonal: Diagonal matrix coefficients.
    :param edges: Off-diagonal matrix coefficients.
    :param guard: Threshold including the guard band.
    :returns: Gate decision and its Sturm count when evaluated.
    """
    guard_count = None
    if gate_mode == "diagonal":
        gate = diagonal_gate
    elif gate_mode == "sturm":
        guard_count = sturm_count(diagonal, edges, guard)
        gate = 0 < guard_count <= 25
    else:
        gate = True
    return gate, guard_count


def investigate(
    random_cases: int, seed: int, probe_iterations: int, margin_scale: float,
    gate_mode: str = "diagonal"
) -> dict[str, object]:
    """Compare the candidate early certificate with baseline rank-zero decisions.

    :param random_cases: Number of additional random cases.
    :param seed: Seed for deterministic random generation.
    :param probe_iterations: Maximum number of prefix iterations.
    :param margin_scale: Scale of the safety margin for certification.
    :param gate_mode: Rule that decides whether to attempt certification.
    :returns: Configuration, certificate counts and per-case decisions.
    :raises ValueError: If the certificate gate is unsupported.
    """
    if gate_mode not in ("diagonal", "sturm", "always"):
        raise ValueError("gate must be diagonal, sturm or always")
    records = []
    for name, diagonal, edges in cases(random_cases, seed):
        lo, hi = enclosing_bounds(diagonal, edges)
        record = {
            "case": name,
            "input_sha256":
            hashlib.sha256(diagonal.tobytes() + edges.tobytes()).hexdigest()
        }
        if not np.isfinite(lo) or not np.isfinite(hi):
            record.update(bounds_valid=False, certified=False, baseline_qr=True)
            records.append(record)
            continue
        assert lo <= hi
        scale = max(abs(lo), abs(hi))
        guard = F32(THRESHOLD + F32(F32(1e-3) * scale))
        margin = F32(F32(margin_scale) * scale)
        diagonal_gate = bool(np.min(diagonal) < F32(guard - margin))
        gate, guard_count = _certificate_gate(
            gate_mode, diagonal_gate, diagonal, edges, guard
        )
        value, invalid, steps = rank_zero(diagonal, edges, lo, hi)
        baseline_qr = invalid or value <= guard
        certified, prefix_steps, reason = (False, 0, "gate-false")
        if gate:
            certified, prefix_steps, reason = prefix_certificate(
                diagonal, edges, lo, hi, guard, probe_iterations
            )
        assert not certified or baseline_qr, (name, value, guard, reason)
        record.update(
            bounds_valid=True, lower=float(lo), upper=float(hi), guard=float(guard),
            minimum_diagonal=float(np.min(diagonal)), gate=gate,
            diagonal_gate=diagonal_gate, guard_count=guard_count,
            baseline_minimum=float(value) if np.isfinite(value) else None,
            baseline_qr=bool(baseline_qr), baseline_invalid=invalid,
            baseline_steps=steps, certified=certified, prefix_steps=prefix_steps,
            certificate_reason=reason
        )
        records.append(record)
    valid = [row for row in records if row["bounds_valid"]]
    return {
        "model": "CPU noncontracted FP32 rank-zero recurrence; not a GPU emulator",
        "scope": "Direct tridiagonal stress; not all cases are image covariances",
        "seed": seed,
        "numpy": np.__version__,
        "random_cases": random_cases,
        "probe_iterations": probe_iterations,
        "margin_scale": margin_scale,
        "gate_mode": gate_mode,
        "summary": {
            "cases":
            len(records),
            "valid_bounds":
            len(valid),
            "certified":
            sum(row["certified"] for row in records),
            "certificate_mismatches":
            0,
            "rayleigh_gate_rank_zero_mismatches":
            sum(row["diagonal_gate"] and not row["baseline_qr"] for row in valid),
            "baseline_qr":
            sum(row["baseline_qr"] for row in records),
            "gate_false_qr":
            sum(not row["gate"] and row["baseline_qr"] for row in valid)
        },
        "cases": records,
    }


def main() -> None:
    """Run the seeded QR certificate checks and write their report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--probe-iterations", type=int, default=16)
    parser.add_argument("--margin-scale", type=float, default=1e-6)
    parser.add_argument(
        "--gate", choices=("diagonal", "sturm", "always"), default="diagonal"
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "artifacts/reports/vmaf-speed-parallelism/earlyqr/numerics.json"
    )
    args = parser.parse_args()
    if args.cases < 0 or not 1 <= args.probe_iterations <= 320:
        parser.error(
            "--cases must be nonnegative; --probe-iterations must be in 1..320"
        )
    if not np.isfinite(args.margin_scale) or args.margin_scale < 0:
        parser.error("--margin-scale must be finite and nonnegative")
    with np.errstate(all="ignore"):
        result = investigate(
            args.cases, args.seed, args.probe_iterations, args.margin_scale, args.gate
        )
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
