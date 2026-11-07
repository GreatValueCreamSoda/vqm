#!/usr/bin/env python3
"""CPU contract checks and analytical sizes for the packed-mode-bank probe.

This checks representation, boundaries and disjoint addresses, not Vulkan
execution or performance. No external packages or generated fixtures are used.
"""

import json
import random
import struct
from array import array
from collections.abc import Callable


def mode3(a: float, b: float, c: float) -> float:
    """Select the majority code or the minimum when all three differ.

    :param a: First sample code.
    :param b: Second sample code.
    :param c: Third sample code.
    :returns: Selected sample code.
    """
    if a in (b, c):
        return a
    if b == c:
        return b
    return min(a, b, c)


def filter_reference(source: list[float], width: int,
                     height: int) -> tuple[list[float], list[float]]:
    """Compute the horizontal and vertical mode filter with unpacked codes.

    :param source: Input codes in row-major order.
    :param width: Width in samples or workgroup output columns.
    :param height: Height in samples.
    :returns: Horizontal and vertical filtered planes.
    """
    horizontal = list(source)
    for y in range(height):
        for x in range(1, width - 1):
            i = y * width + x
            horizontal[i] = mode3(source[i - 1], source[i], source[i + 1])
    filtered = list(source)
    for y in range(1, height - 1):
        for x in range(width):
            i = y * width + x
            filtered[i] = mode3(
                horizontal[i - width], horizontal[i], horizontal[i + width]
            )
    return horizontal, filtered


def _packed_vertical(
    source: list[float], width: int, height: int, horizontal_offset: int,
    filtered_offset: int, load: Callable[[int, int], float],
    store: Callable[[int, int, float], None]
) -> None:
    """Read the horizontal bank and write the vertical mode-filter output.

    :param source: Original row-major codes for the top and bottom boundaries.
    :param width: Image width in samples.
    :param height: Image height in samples.
    :param horizontal_offset: Horizontal bank address in words.
    :param filtered_offset: Output bank address in words.
    :param load: Function for reading one packed code.
    :param store: Function for writing one packed code.
    """
    for y in range(height):
        for x in range(width):
            i = y * width + x
            value = source[i] if y in (0, height - 1) else mode3(
                load(horizontal_offset, i - width), load(horizontal_offset, i),
                load(horizontal_offset, i + width)
            )
            store(filtered_offset, i, value)


def filter_packed(source: list[float], width: int,
                  height: int) -> tuple[list[float], list[float]]:
    """Simulate the mode filter in guarded packed-code storage.

    :param source: Input codes in row-major order.
    :param width: Width in samples or workgroup output columns.
    :param height: Height in samples.
    :returns: Horizontal and vertical filtered planes.
    """
    # Two full FP32-sized regions with odd word offsets and guard bytes. The
    # first halves hold codes, exactly as binding 2 indexes 2*word_offset+i.
    count = width * height
    horizontal_offset, filtered_offset = 7, 7 + count + 11
    memory = bytearray([0xA5] * (4 * (filtered_offset + count + 7)))

    def store(offset: int, i: int, value: float) -> None:
        """Write one code into a packed bank.

        :param offset: Bank base in words.
        :param i: Sample index in the bank.
        :param value: Input sample value.
        """
        struct.pack_into("=H", memory, 4 * offset + 2 * i, int(value))

    def load(offset: int, i: int) -> float:
        """Read one code from a packed bank.

        :param offset: Bank base in words.
        :param i: Sample index in the bank.
        :returns: Decoded packed code as a floating-point value.
        """
        return float(struct.unpack_from("=H", memory, 4 * offset + 2 * i)[0])

    for y in range(height):
        for x in range(width):
            i = y * width + x
            value = source[i] if x in (
                0, width - 1
            ) else mode3(source[i - 1], source[i], source[i + 1])
            store(horizontal_offset, i, value)
    _packed_vertical(
        source, width, height, horizontal_offset, filtered_offset, load, store
    )
    touched = set()
    for offset in (horizontal_offset, filtered_offset):
        touched.update(range(4 * offset, 4 * offset + 2 * count))
    assert all(value == 0xA5 for i, value in enumerate(memory) if i not in touched)
    return (
        [load(horizontal_offset, i)
         for i in range(count)], [load(filtered_offset, i) for i in range(count)]
    )


def scale_sizes(width: int, height: int) -> tuple[int, list[tuple[int, int]]]:
    """Plan the contrast window and five CAMBI scale geometries.

    :param width: Width in samples or workgroup output columns.
    :param height: Height in samples.
    :returns: Window width and ordered scale dimensions.
    """
    pixels = width * height
    speedup = pixels >= 1920 * 1080
    window = ((65 * (width + height) // 375) >> 4)
    if speedup:
        window = (window + 1) >> 1
        width, height = (width + 1) // 2, (height + 1) // 2
    window |= 1
    sizes = []
    for scale in range(5):
        if scale:
            width, height = (width + 1) // 2, (height + 1) // 2
        sizes.append((width, height))
    return window, sizes


def _mode_source(rng: random.Random, kind: int, width: int, count: int) -> list[float]:
    """Generate one fixture while preserving the seeded draw order.

    :param rng: Random generator shared across fixtures.
    :param kind: Sample-pattern selector.
    :param width: Image width in samples.
    :param count: Number of samples in the source plane.
    :returns: Source plane in row-major order.
    """
    if kind == 0:
        source = [float(rng.randrange(1025)) for _ in range(count)]
    elif kind == 1:
        source = [float(rng.choice((0, 67, 563, 1024))) for _ in range(count)]
    elif kind == 2:
        source = [float((i // width + i % width) % 1025) for i in range(count)]
    else:
        source = [float((0, 563, 1024)[kind - 3])] * count
    return source


def main() -> None:
    """Check packed CAMBI mode boundaries and print the JSON report."""
    # Source conversion can round an original 16-bit maximum up to code1024.
    # Every reachable code therefore round-trips exactly, including extremes.
    for code in range(1025):
        assert float(array("H", [code])[0]) == float(code)
    rng = random.Random(20261105)
    cases = 0
    for width, height in ((1, 1), (1, 9), (9, 1), (3, 3), (17, 33), (33, 17), (64, 35),
                          (129, 73)):
        count = width * height
        for kind in range(6):
            source = _mode_source(rng, kind, width, count)
            horizontal, filtered = filter_reference(source, width, height)
            actual_horizontal, actual_filtered = filter_packed(source, width, height)
            assert actual_horizontal == horizontal
            assert actual_filtered == filtered
            ow, oh = (width + 1) // 2, (height + 1) // 2
            # Packing changes only the source read. Output and spatial mask
            # remain FP32 and all odd-edge samples retain exact coordinates.
            expected = [
                filtered[(2 * y) * width + 2 * x] for y in range(oh) for x in range(ow)
            ]
            actual = [
                actual_filtered[(2 * y) * width + 2 * x] for y in range(oh)
                for x in range(ow)
            ]
            assert actual == expected
            cases += 1
    report = {
        "CPU": "PASS",
        "code_roundtrips": 1025,
        "mode_boundary_cases": cases,
        "GPU_correctness": "not tested",
        "performance": "not measured",
        "sizes": []
    }
    for width, height in ((1920, 1080), (3840, 2160), (1919, 1079)):
        window, sizes = scale_sizes(width, height)
        total = sum(w * h for w, h in sizes)
        maximum = sizes[0][0] * sizes[0][1]
        # Hypothetical all-active upper grid, including out-of-bounds halo
        # calls. packed_sample's bounds check avoids some of these loads.
        source_probes = sum(
            ((w + 15) // 16) * ((h + 7) // 8) * (16 + window - 1) * (8 + window - 1)
            for w, h in sizes
        )
        report["sizes"].append(
            {
                "input": [width, height],
                "window":
                window,
                "scales":
                sizes,
                "active_scale_pixels":
                total,
                "later_layout_saving_bytes_if_compacted":
                4 * maximum,
                "packed_variant_current_layout_saving_bytes":
                0,
                "mode_filter_logical_bytes_saved":
                sum(2 * w * h + 2 * w * h + 6 * w * max(0, h - 2) for w, h in sizes),
                "filtered_source_probe_bytes_saved_upper_bound":
                2 * source_probes,
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
