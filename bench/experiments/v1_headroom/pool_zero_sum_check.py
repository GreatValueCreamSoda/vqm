#!/usr/bin/env python3
"""CPU proof examples for CAMBI's exact zero-above-threshold scan shortcut.

Models the existing radix decisions and checks the new proof against an
independent sorted-key selected set. There are no timers or GPU operations.
"""

import argparse
import hashlib
import json
import struct
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SEED = 20261105


def _first_histogram(keys: list[int], prefix: int) -> tuple[list[int], int]:
    """Build the top-byte histogram and OR the keys from its zero bucket.

    :param keys: Binary32 keys in input order.
    :param prefix: Prefix accumulated before this phase.
    :returns: Bucket counts and the OR of nonzero keys in bucket zero.
    """
    histogram = [0] * 256
    for key in keys:
        top = key >> 24
        if top:
            histogram[top] += 1
        elif key:
            prefix |= key
    return histogram, prefix


def _prefix_histogram(keys: list[int], byte: int, prefix: int) -> list[int]:
    """Build one lower-byte histogram within the selected prefix.

    :param keys: Binary32 keys in input order.
    :param byte: Radix byte index, below the first phase.
    :param prefix: Previously selected high bytes.
    :returns: Counts by bucket for the current byte.
    """
    histogram = [0] * 256
    shift = byte * 8
    mask = (0xffffffff << (shift + 8)) & 0xffffffff
    for key in keys:
        if key & mask == prefix:
            histogram[(key >> shift) & 255] += 1
    return histogram


def _choose_bucket(histogram: list[int], rank: int) -> tuple[int, int]:
    """Locate the descending bucket containing the requested rank.

    :param histogram: Counts indexed by radix bucket.
    :param rank: One-based rank within the current prefix.
    :returns: Selected bucket and rank within that bucket.
    :raises AssertionError: If the histogram does not cover the rank.
    """
    for bucket in range(255, -1, -1):
        count = histogram[bucket]
        if rank <= count:
            return bucket, rank
        rank -= count
    raise AssertionError("rank not found")


def _radix_phase(
    keys: list[int], selected: int, shortcut: bool, prefix: int, rank: int, byte: int
) -> tuple[int, int, int]:
    """Advance one byte of radix selection and record the zero-sum proof.

    :param keys: Binary32 keys in input order.
    :param selected: Original selected count.
    :param shortcut: Whether to enable the zero-sum proof.
    :param prefix: Threshold prefix from the preceding phase.
    :param rank: Remaining rank in the current prefix.
    :param byte: Current byte index.
    :returns: Updated threshold, rank and zero-sum proof flag.
    """
    zero_bucket = False
    zero_sum = False
    if byte == 3:
        histogram, prefix = _first_histogram(keys, prefix)
        other = sum(histogram)
        histogram[0] = len(keys) - other
        zero_bucket = prefix == 0
        zero_sum = shortcut and other == 0 and zero_bucket
        prefix = 0
    else:
        histogram = _prefix_histogram(keys, byte, prefix)
    bucket, rank = _choose_bucket(histogram, rank)
    prefix |= bucket << (byte * 8)
    if shortcut and byte == 0 and rank == selected:
        zero_sum = True
    if (byte == 3 and bucket == 0 and zero_bucket) or (byte > 0
                                                       and rank == histogram[bucket]):
        prefix = 0 if prefix == 0 else prefix - 1
        rank = 0
    return prefix, rank, int(zero_sum)


def radix(keys: list[int], selected: int,
          shortcut: bool) -> tuple[int, int, int, list[dict[str, int]]]:
    """Simulate radix threshold selection and its optional zero-sum proof.

    :param keys: Binary32 bit keys ordered as unsigned integers.
    :param selected: Number of largest keys to select.
    :param shortcut: Whether to enable the zero-sum proof.
    :returns: Threshold key, remaining rank, proof flag and phase history.
    """
    prefix, rank, flag = 0, selected, 0
    history = []
    for byte in (3, 2, 1, 0):
        if rank:
            prefix, rank, flag = _radix_phase(
                keys, selected, shortcut, prefix, rank, byte
            )
        # A completed scale has no later histogram writers. Bin zero's
        # proof is preserved until the final partial scan consumes it.
        history.append(
            {
                "byte": byte,
                "threshold": prefix,
                "rank": rank,
                "zero_sum": flag
            }
        )
    return prefix, rank, flag, history


def check_case(keys: list[int], selected: int) -> dict[str, object]:
    """Check the radix selector against an independently sorted selected set.

    :param keys: Binary32 bit keys ordered as unsigned integers.
    :param selected: Number of largest keys to select.
    :returns: Input identity, selector result and per-byte history.
    """
    baseline = radix(keys, selected, False)
    candidate = radix(keys, selected, True)
    assert baseline[:2] == candidate[:2]
    threshold, rank, flag, history = candidate
    above = sum(key > threshold for key in keys)
    equal = sum(key == threshold for key in keys)
    zero = sum(key == 0 for key in keys)
    if rank:
        assert above + rank == selected and rank <= equal
    elif threshold == 0:
        assert above <= selected <= above + zero
    else:
        assert above == selected
    if flag:
        assert above == 0
        # Independent sorted-set check: every selected value is a tie, or
        # all inputs are zero and the old shortcut omits zero from summation.
        chosen = sorted(keys, reverse=True)[:selected]
        assert len(set(chosen)) == 1 and chosen[0] == threshold
    return {
        "count": len(keys),
        "selected": selected,
        "input_sha256": hashlib.sha256(struct.pack(f"<{len(keys)}I",
                                                   *keys)).hexdigest(),
        "threshold": threshold,
        "rank": rank,
        "zero_sum": bool(flag),
        "above": above,
        "history": history
    }


def run() -> dict[str, object]:
    """Evaluate fixed and seeded random selector cases.

    :returns: Probe result and all recorded case outcomes.
    """
    rng = np.random.default_rng(SEED)
    cases = []
    sizes = (1, 2, 3, 5, 7, 255, 256, 257, 1024, 8193)
    for count in sizes:
        samples = {
            "zero": [0] * count,
            "same_normal": [0x41200000] * count,
            "same_tiny": [0x00000001] * count,
            "normal_zero_mix": [0x41200000 if i % 3 else 0 for i in range(count)],
            "tiny_zero_mix":
            [0x00800000 + i % 17 if i % 4 else 0 for i in range(count)],
            "descending_ties": [0x3f800000 + (i % 17) for i in range(count)],
            "scattered_finite": [int(x) for x in rng.integers(0, 0x7f800000, count)],
        }
        for (label, keys), ratio in product(samples.items(), (0.01, 0.2, 0.6, 1.0)):
            selected = int(max(1, min(count, ratio * count)))
            case = check_case(keys, selected)
            case.update({"pattern": label, "ratio": ratio})
            cases.append(case)
    for _ in range(600):
        count = int(rng.integers(1, 1300))
        keys = [int(x) for x in rng.integers(0, 0x7f800000, count)]
        if rng.random() < 0.5:
            key = int(rng.integers(0, 0x7f800000))
            keys = [key if rng.random() < 0.75 else x for x in keys]
        case = check_case(keys, int(rng.integers(1, count + 1)))
        case["pattern"] = "seeded_finite_keys"
        cases.append(case)
    return {
        "scope": "CPU selector/proof equivalence, not GPU timing or qualification",
        "seed": SEED,
        "numpy": np.__version__,
        "cases": cases,
        "count": len(cases),
        "all_states_match": True,
        "flagged_cases": sum(case["zero_sum"] for case in cases),
        "all_flagged_above_counts_zero": True
    }


def main() -> None:
    """Run the CAMBI selector proof examples and write their report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "artifacts/reports/vmaf-v1-headroom/pool-zero-sum/numerics.json"
    )
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{result['count']} selector states match; {result['flagged_cases']} exact zero-sum proofs"
    )


if __name__ == "__main__":
    main()
