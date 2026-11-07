#!/usr/bin/env python3
"""Compare 31-node speculative FP32 bisection with sequential midpoint decisions.

This CPU diagnostic checks interval traversal and stopping behavior, not GPU
Sturm precision, SIMD contraction, metric correctness or performance. The count
callback is monotone except in explicit invalid-count stress cases.
"""

import argparse
import hashlib
import json
import struct
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SEED = 2026110503


def bits(value: float) -> int:
    """Return the binary32 bit pattern of a scalar.

    :param value: Input sample value.
    :returns: Unsigned integer with the same binary32 bit pattern.
    """
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f32(value: float) -> float:
    """Round a scalar to binary32.

    :param value: Input sample value.
    :returns: Scalar rounded to binary32.
    """
    return struct.unpack("<f", struct.pack("<f", value))[0]


def midpoint_numpy(lo: float, hi: float) -> float:
    """Compute an interval midpoint using NumPy binary32 operations.

    :param lo: Lower interval endpoint.
    :param hi: Upper interval endpoint.
    :returns: Binary32 midpoint as a Python float.
    """
    return float(
        np.float32(np.float32(lo) * np.float32(0.5)) +
        np.float32(np.float32(hi) * np.float32(0.5))
    )


def midpoint_scalar(lo: float, hi: float) -> float:
    """Compute a scalar midpoint with separate binary32 rounding steps.

    :param lo: Lower interval endpoint.
    :param hi: Upper interval endpoint.
    :returns: Binary32 midpoint as a Python float.
    """
    # Binary64 exactly holds the relevant binary32 products/sum here; round
    # each multiplication and addition independently to binary32 afterward.
    return f32(f32(lo * 0.5) + f32(hi * 0.5))


def finish(lo: float, hi: float, converged: bool, invalid: bool,
           tests: int) -> dict[str, int | bool]:
    """Record the final interval, result bits and stopping state.

    :param lo: Lower interval endpoint.
    :param hi: Upper interval endpoint.
    :param converged: Whether the interval reached the stopping condition.
    :param invalid: Whether the selected count was invalid.
    :param tests: Number of midpoint decisions taken.
    :returns: Endpoint and midpoint bits, status and decision count.
    """
    return {
        "lo_bits": bits(lo),
        "hi_bits": bits(hi),
        "value_bits": bits(midpoint_scalar(lo, hi)),
        "converged": converged,
        "invalid": invalid,
        "tests": tests
    }


def sequential(lo: float, hi: float, count: Callable[[float],
                                                     int]) -> dict[str, int | bool]:
    """Run sequential midpoint decisions to convergence or the step limit.

    :param lo: Lower interval endpoint.
    :param hi: Upper interval endpoint.
    :param count: Callback evaluating the rank at a trial midpoint.
    :returns: Final interval and stopping state.
    """
    for iteration in range(320):
        mid = midpoint_numpy(lo, hi)
        if mid in (lo, hi):
            return finish(lo, hi, True, False, iteration + 1)
        result = count(mid)
        if result > 25:
            return finish(lo, hi, False, True, iteration + 1)
        if result <= 0:
            lo = mid
        else:
            hi = mid
    return finish(lo, hi, False, False, 320)


def _trial_midpoint(lo: float, hi: float, node: int) -> float:
    """Find a node midpoint by following the bits of its root path.

    :param lo: Current lower interval endpoint.
    :param hi: Current upper interval endpoint.
    :param node: One-based node index with the root at one.
    :returns: Node midpoint rounded to binary32.
    """
    depth = node.bit_length() - 1
    path = node - (1 << depth)
    left, right = lo, hi
    for step in range(depth):
        mid = midpoint_scalar(left, right)
        if (path >> (depth - 1 - step)) & 1:
            left = mid
        else:
            right = mid
    mid = midpoint_scalar(left, right)
    return mid


def _trial_nodes(lo: float, hi: float,
                 count: Callable[[float], int]) -> dict[int, tuple[float, int]]:
    """Evaluate the 31 trial midpoints and ranks in node order.

    :param lo: Current lower interval endpoint.
    :param hi: Current upper interval endpoint.
    :param count: Rank callback for one midpoint.
    :returns: Midpoint and rank by binary tree node index.
    """
    nodes = {}
    for node in range(1, 32):
        mid = _trial_midpoint(lo, hi, node)
        nodes[node] = mid, count(mid)
    return nodes


def _follow_trials(
    lo: float, hi: float, nodes: dict[int, tuple[float, int]], tests: int
) -> tuple[float, float, int, bool, bool]:
    """Apply up to five precomputed decisions, stopping on a terminal state.

    :param lo: Current lower interval endpoint.
    :param hi: Current upper interval endpoint.
    :param nodes: Trial midpoints and ranks from the current batch.
    :param tests: Number of earlier midpoint decisions.
    :returns: Endpoints, cumulative count and convergence/invalid flags.
    """
    selected = 1
    for _ in range(5):
        mid, result = nodes[selected]
        tests += 1
        if mid in (lo, hi):
            return lo, hi, tests, True, False
        if result > 25:
            return lo, hi, tests, False, True
        if result <= 0:
            lo = mid
            selected = 2 * selected + 1
        else:
            hi = mid
            selected = 2 * selected
    return lo, hi, tests, False, False


def trial_tree(lo: float, hi: float, count: Callable[[float],
                                                     int]) -> dict[str, int | bool]:
    """Traverse batches of speculative midpoint decisions.

    :param lo: Lower interval endpoint.
    :param hi: Upper interval endpoint.
    :param count: Callback evaluating the rank at a trial midpoint.
    :returns: Final interval and stopping state.
    """
    tests = 0
    for _ in range(64):
        nodes = _trial_nodes(lo, hi, count)
        lo, hi, tests, converged, invalid = _follow_trials(lo, hi, nodes, tests)
        if converged or invalid:
            return finish(lo, hi, converged, invalid, tests)
    return finish(lo, hi, False, False, tests)


def cases(random_count: int) -> Iterator[tuple[str, float, float, float, int | None]]:
    """Yield fixed edge cases and seeded random intervals.

    :param random_count: Number of additional random cases.
    :returns: Test cases in reproducible order.
    """
    maximum = float(np.finfo(np.float32).max)
    tiny = float(np.finfo(np.float32).tiny)
    subnormal = float(np.nextafter(np.float32(0), np.float32(1)))
    adjacent = float(np.nextafter(np.float32(1), np.float32(2)))
    examples = [
        ("wide-zero", -maximum, maximum, 0.0),
        ("wide-positive", 0.0, maximum, f32(maximum / 3)),
        ("wide-negative", -maximum, 0.0, f32(-maximum / 3)),
        ("adjacent-normal", 1.0, adjacent, 1.0),
        ("adjacent-subnormal", subnormal, 2 * subnormal, subnormal),
        ("cross-subnormal", -3 * subnormal, 3 * subnormal, subnormal),
        ("zero-to-normal", 0.0, tiny, subnormal), ("signed-zero", -0.0, 0.0, 0.0),
        ("equal-positive", 3.0, 3.0, 3.0), ("equal-negative", -3.0, -3.0, -3.0)
    ]
    for name, lo, hi, target in examples:
        yield name, f32(lo), f32(hi), f32(target), None
    yield "invalid-unused-node", 0.0, 32.0, 24.0, bits(8.0)
    yield "invalid-selected-node", 0.0, 32.0, 24.0, bits(24.0)
    rng = np.random.default_rng(SEED)
    for index in range(random_count):
        raw = rng.integers(0, 0x7f800000, size=2, dtype=np.uint32)
        values = raw.view(np.float32).astype(np.float64)
        values *= rng.choice([-1.0, 1.0], size=2)
        lo, hi = sorted(float(v) for v in values)
        target = f32(lo + (hi - lo) * float(rng.random()))
        yield f"random-{index}", lo, hi, target, None


def main() -> None:
    """Check speculative and sequential midpoint decisions and write JSON.

    :raises AssertionError: If the speculative search differs from the sequential
        search.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-cases", type=int, default=256)
    parser.add_argument(
        "--out", type=Path, default=ROOT /
        "artifacts/reports/vmaf-speed-parallelism/solver/tree-checks.json"
    )
    args = parser.parse_args()
    if args.random_cases < 0:
        parser.error("--random-cases must be nonnegative")
    records = []
    for name, lo, hi, target, invalid_at in cases(args.random_cases):

        def count(
            value: float, invalid_at: int | None = invalid_at, target: float = target
        ) -> int:
            """Return the trial's rank or its injected invalid count.

            :param value: Trial midpoint.
            :param invalid_at: Bit pattern that injects an invalid rank, if any.
            :param target: Monotone rank transition point.
            :returns: Rank count, including the configured invalid sentinel.
            """
            return 26 if invalid_at is not None and bits(value) == invalid_at else int(
                value > target
            )

        reference = sequential(lo, hi, count)
        candidate = trial_tree(lo, hi, count)
        if reference != candidate:
            raise AssertionError((name, reference, candidate))
        records.append(
            {
                "case": name,
                "lo_bits": bits(lo),
                "hi_bits": bits(hi),
                "target_bits": bits(target),
                "invalid_at_bits": invalid_at,
                "matching_result": reference
            }
        )
    source = Path(__file__).resolve()
    result = {
        "scope": "CPU interval traversal/stopping check, not a GPU or metric oracle",
        "seed": SEED,
        "numpy": np.__version__,
        "passed": len(records),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cases": records
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        f"All {len(records)} sequential/tree FP32 endpoint and stopping checks passed"
    )


if __name__ == "__main__":
    main()
