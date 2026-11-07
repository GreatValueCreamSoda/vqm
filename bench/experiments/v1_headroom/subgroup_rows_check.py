#!/usr/bin/env python3
"""CPU access-domain proof for the CAMBI subgroup-row experiment.

This checks coverage and independent row ownership under arbitrary physical
local-index permutations. It is not a Vulkan execution or performance check.
"""

import json
import random
from collections import Counter


def _check_coverage(logical: list[int]) -> None:
    """Check halo and window loads under one physical lane permutation.

    :param logical: Logical lane indices in physical order.
    """
    for radius in (0, 8, 16):
        extent, window, rows = 16 + 2 * radius, 1 + 2 * radius, 8 + 2 * radius
        # Center publication and cooperative halo loads partition the tile.
        center_indices = {
            (row + radius) * extent + x + radius
            for row in range(8)
            for x in range(16)
        }
        halo = Counter(
            at for i in logical for at in range(i, extent * rows, 256)
            if at not in center_indices
        )
        assert set(halo).isdisjoint(center_indices)
        assert all(count == 1 for count in halo.values())
        assert set(halo) | center_indices == set(range(extent * rows))
        # Initial dense construction retains all cross-row barriers.
        window_loads = Counter(
            at for i in logical for at in range(i, window * window, 256)
        )
        assert window_loads == Counter(range(window * window))


def _check_row_storage() -> None:
    """Verify disjoint histogram, term and output addresses for each row."""
    histogram = [set(range(row * 256, row * 256 + 249)) for row in range(8)]
    terms = [set(range(row * 4, row * 4 + 4)) for row in range(8)]
    outputs = [set(range(row * 16, row * 16 + 16)) for row in range(8)]
    for a in range(8):
        for b in range(a):
            assert histogram[a].isdisjoint(histogram[b])
            assert terms[a].isdisjoint(terms[b])
            assert outputs[a].isdisjoint(outputs[b])


def _check_slide(
    x: int, row: int, radius: int, extent: int, window: int, rows: int
) -> None:
    """Bound the center and sliding-window addresses for one output.

    :param x: Output column inside the tile.
    :param row: Output row inside the tile.
    :param radius: Window radius.
    :param extent: Tile width including the halo.
    :param window: Window width in samples.
    :param rows: Tile height including the halo.
    """
    center = (row + radius) * extent + x + radius
    assert 0 <= center < extent * rows
    if x == 15:
        return
    # Every entering/leaving tile sample remains in bounds.
    # These are reads of the already-published immutable tile.
    for k in range(window):
        at = (row + k) * extent + x
        assert 0 <= at < extent * rows
        assert 0 <= at + window < extent * rows


def _check_slides(radius: int) -> None:
    """Bound all center and update addresses for a window radius.

    :param radius: Window radius.
    """
    extent, window, rows = 16 + 2 * radius, 1 + 2 * radius, 8 + 2 * radius
    for row in range(8):
        for x in range(16):
            _check_slide(x, row, radius, extent, window, rows)


def main() -> None:
    """Check subgroup row ownership and print the JSON result."""
    rng = random.Random(20261105)
    local_indices = list(range(256))
    for _ in range(100):
        # Subgroup builtins need not agree with physical local index order.
        rng.shuffle(local_indices)
        logical_for_local = dict(zip(local_indices, range(256), strict=True))
        logical = list(logical_for_local.values())
        assert sorted(logical) == list(range(256))
        assert sum(i == 0 for i in logical) == 1
        centers = [(i // 32, i % 32) for i in logical if i % 32 < 16]
        assert sorted(centers) == [(row, x) for row in range(8) for x in range(16)]
        clear = Counter(at for i in logical for at in range(i, 8 * 256, 256))
        assert clear == Counter(range(8 * 256))
        clone = Counter(logical)
        assert clone == Counter(range(256))
        _check_coverage(logical)

    _check_row_storage()

    for radius in (0, 8, 16):
        _check_slides(radius)

    # Every active-row pattern is legal. Optional early returns are uniform
    # within a complete subgroup and occur after all cross-row barriers.
    for active in range(256):
        returning = {row for row in range(8) if not (active & (1 << row))}
        assert sum(32 for row in returning) % 32 == 0
        for row in range(8):
            predicates = [row in returning for _ in range(32)]
            assert all(value == predicates[0] for value in predicates)
    print(
        json.dumps(
            {
                "CPU_access_domains": "PASS",
                "physical_index_permutations": 100,
                "active_row_patterns": 256,
                "radii": [0, 8, 16],
                "rows": 8,
                "lanes_per_row": 32,
                "workgroup_invocations": 256,
                "GPU_correctness": "not tested",
                "performance": "not measured"
            }, indent=2
        )
    )


if __name__ == "__main__":
    main()
