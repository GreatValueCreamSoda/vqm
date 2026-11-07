#!/usr/bin/env python3
"""CPU-only CAMBI tile/index proof and SPIR-V capacity checks; never uses a GPU.

Run from the repository with:
  python3 bench/experiments/v1_headroom/contrast_contracts.py

The histogram simulation compares every eligible center with an independent
clipped-window Counter, covering dense, sparse, empty and late-column-only
inputs. It checks each tile address, packed-counter bound, output ownership,
clone transition and horizontal update. Compiler checks require glslc,
spirv-val, spirv-dis and a C++20 compiler. Raw evidence is disposable; the
research note records the compact conclusions.
"""

import argparse
import hashlib
import json
import random
import re
import shutil
import struct
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

from contrast_variants import CAMBI, ROOT, SHADER, install


def _spirv_type_bytes(definitions: dict[str, tuple[str, list[str]]], ident: str) -> int:
    """Resolve the byte size of a SPIR-V type.

    :param definitions: Operations and operands indexed by SPIR-V identifier.
    :param ident: SPIR-V type identifier.
    :returns: Size in bytes.
    :raises ValueError: If the workgroup type is unsupported.
    """
    op, args = definitions[ident]
    if op in ("OpTypeFloat", "OpTypeInt"):
        return int(args[0]) // 8
    if op == "OpTypeArray":
        length = int(definitions[args[1]][1][1])
        return _spirv_type_bytes(definitions, args[0]) * length
    if op == "OpTypePointer":
        return _spirv_type_bytes(definitions, args[1])
    raise ValueError(f"unhandled workgroup type: {op} {args}")


def spirv_shared_bytes(assembly: str) -> int:
    """Count workgroup storage declared by the SPIR-V module.

    :param assembly: Disassembled SPIR-V text.
    :returns: Total declared workgroup storage in bytes.
    """
    definitions = {}
    variables = []
    for line in assembly.splitlines():
        match = re.match(r"\s*(%\S+) = (Op\w+)\s*(.*)", line)
        if not match:
            continue
        ident, op, args = match.groups()
        definitions[ident] = (op, args.split())
        if op == "OpVariable" and args.split()[1] == "Workgroup":
            variables.append(args.split()[0])

    return sum(_spirv_type_bytes(definitions, pointer) for pointer in variables)


def _check_shared_bytes(
    name: str, kwargs: dict[str, int | bool], shader_name: str, definitions: str,
    assembly: str, shared: int, half: bool
) -> None:
    """Check the compiler-reported workgroup storage against the tile geometry.

    :param name: Variant label for failure diagnostics.
    :param kwargs: Geometry and specialization options used by the installer.
    :param shader_name: Generated shader name.
    :param definitions: Command-line shader macro definitions.
    :param assembly: Disassembled SPIR-V.
    :param shared: Declared workgroup storage in bytes.
    :param half: Whether the module stores half-precision samples.
    """
    if shader_name == "vmaf_cambi_mode":
        assert shared == 0 and "OpCapability Float16" not in assembly
        expected = 0
    else:
        radius = 8 if "MAX_RADIUS=8" in definitions else 16
        selected_small = not kwargs.get(
            "small_window_only"
        ) or shader_name == "vmaf_cambi_contrast_small_half"
        width = kwargs.get("width", 16) if half and selected_small else 16
        rows = kwargs.get("rows", 8) if half and selected_small else 8
        expected = (
            (width + 2 * radius) * (rows + 2 * radius) * (2 if half else 4) +
            rows * 256 * 4 + 4 + rows * 4 * 4
        )
        if kwargs.get("reuse_tail"):
            expected -= 4 + rows * 4 * 4
        assert shared == expected, (name, shader_name, shared, expected)
        assert shared <= 16004


def _compile_module(
    out: Path, source: Path, name: str, kwargs: dict[str, int | bool], shader_name: str,
    definitions: str
) -> dict[str, object]:
    """Compile and validate a shader using its generated build rule.

    :param out: Directory for the compiled module.
    :param source: Isolated source copy with the variant installed.
    :param name: Variant label for artifacts and diagnostics.
    :param kwargs: Geometry and specialization options used by the installer.
    :param shader_name: Generated shader name.
    :param definitions: Command-line shader macro definitions.
    :returns: Shader identity, compiled hash and workgroup storage.
    """
    binary = out / (name + "-" + shader_name + ".spv")
    subprocess.run(
        [
            "glslc", "-O", "--target-env=vulkan1.1", "-I",
            str(ROOT / "src/vulkan/shaders"), *definitions.split(),
            str(source / SHADER), "-o",
            str(binary)
        ], check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["spirv-val", "--target-env", "vulkan1.1",
         str(binary)], check=True, capture_output=True, text=True
    )
    assembly = subprocess.run(
        ["spirv-dis", str(binary)], check=True, capture_output=True, text=True
    ).stdout
    shared = spirv_shared_bytes(assembly)
    local_size = 128 if "TILE_INVOCATIONS=128" in definitions else 256
    assert re.search(
        r"OpExecutionMode %\S+ LocalSize 16 " + str(local_size // 16) + r" 1", assembly
    )
    half = "HALF_SOURCE" in definitions
    _check_shared_bytes(name, kwargs, shader_name, definitions, assembly, shared, half)
    return {
        "arm": name,
        "shader": shader_name,
        "defines": definitions.split(),
        "shared_bytes": shared,
        "spirv_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()
    }


def _compile_variant(out: Path, source: Path, name: str,
                     kwargs: dict[str, int | bool]) -> list[dict[str, object]]:
    """Install and validate one variant's host code and shader modules.

    :param out: Directory for compiled modules.
    :param source: Destination for the isolated source copy.
    :param name: Variant label for artifacts and diagnostics.
    :param kwargs: Geometry and specialization options used by the installer.
    :returns: Validation records for every generated module.
    """
    files = [
        SHADER, CAMBI / "vmaf_cambi.cc", CAMBI / "vmaf_cambi.h",
        Path("CMakeLists.txt"),
        Path("src/metrics/vmaf/v1/vmaf_v1_pipeline.cc")
    ]
    for relative in files:
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, source / relative)
    old_host = (source / CAMBI / "vmaf_cambi.cc").read_bytes()
    old_header = (source / CAMBI / "vmaf_cambi.h").read_bytes()
    old_cmake = (source / "CMakeLists.txt").read_bytes()
    install(source, **kwargs)
    if kwargs.get("small_window_only"):
        assert (ROOT / files[-1]).read_bytes() == (source / files[-1]).read_bytes()
    if kwargs == {"reuse_tail": True}:
        assert old_host == (source / CAMBI / "vmaf_cambi.cc").read_bytes()
        assert old_header == (source / CAMBI / "vmaf_cambi.h").read_bytes()
        assert old_cmake == (source / "CMakeLists.txt").read_bytes()
    old_shader = (ROOT / SHADER).read_text()
    new_shader = (source / SHADER).read_text()
    # Mode ties, boundary policies, stores and floating arithmetic remain
    # byte-for-byte source-identical in the active mode body.
    marker = "    uvec2 p = gl_GlobalInvocationID.xy;"
    assert old_shader[old_shader.index(marker):] == new_shader[new_shader.
                                                               index(marker):]
    start, end = "float mode3(", "uint bin_count("
    assert old_shader[old_shader.index(start):old_shader.index(end)] == new_shader[
        new_shader.index(start):new_shader.index(end)]
    subprocess.run(
        [
            "c++", "-std=c++20", "-fsyntax-only", "-I",
            str(source / "src"), "-I",
            str(ROOT / "src"), "-I",
            str(ROOT / "include"),
            str(source / CAMBI / "vmaf_cambi.cc")
        ], check=True, capture_output=True, text=True
    )
    # Check the exact definitions emitted by the installer, including
    # uint fallbacks and both radius capacities.
    cmake = (source / "CMakeLists.txt").read_text()
    rules = re.findall(
        r"vqm_add_shader\((vmaf_cambi_(?:contrast\w*|mode))[ \t]+[^)\s]+([^)]*)\)",
        cmake
    )
    results = []
    for shader_name, definitions in rules:
        results.append(
            _compile_module(out, source, name, kwargs, shader_name, definitions)
        )
    return results


def compile_contracts(out: Path) -> list[dict[str, object]]:
    """Compile and inspect each CAMBI shader geometry.

    :param out: Directory for generated probe artifacts.
    :returns: Validated module hashes and capacity checks in variant order.
    """
    variants = [
        ("mode", {
            "mode_only": True
        }), ("radius8", {
            "radius8": True
        }), ("wide32", {
            "width": 32
        }), ("short4", {
            "rows": 4
        }), ("wide32-short4", {
            "width": 32,
            "rows": 4
        }), ("combined", {
            "mode_only": True,
            "radius8": True,
            "width": 32
        }), ("tail", {
            "reuse_tail": True
        }), ("radius8-tail", {
            "radius8": True,
            "reuse_tail": True
        }), ("half128-rows4", {
            "mode_only": True,
            "rows": 4,
            "invocations": 128
        }),
        (
            "half128-compact-tail", {
                "mode_only": True,
                "radius8": True,
                "rows": 4,
                "invocations": 128,
                "reuse_tail": True
            }
        ),
        (
            "small-only128", {
                "mode_only": True,
                "rows": 4,
                "invocations": 128,
                "small_window_only": True
            }
        ),
        (
            "small-only128-compact-tail", {
                "mode_only": True,
                "radius8": True,
                "rows": 4,
                "invocations": 128,
                "small_window_only": True,
                "reuse_tail": True
            }
        ),
    ]
    results = []
    with tempfile.TemporaryDirectory(prefix="v1-contrast-contracts-") as temporary:
        temporary = Path(temporary)
        for name, kwargs in variants:
            results.extend(_compile_variant(out, temporary / name, name, kwargs))
    return results


def _pattern_image(rng: random.Random, width: int, height: int,
                   pattern: str) -> list[list[int]]:
    """Generate packed sample codes in the original seeded draw order.

    :param rng: Generator shared by all image patterns.
    :param width: Image width in samples.
    :param height: Image height in samples.
    :param pattern: Masking pattern applied after drawing the codes.
    :returns: Packed eligible codes, with zero denoting a masked sample.
    """
    # Values 1..497 are exact halves. Zero is also used for samples outside
    # the image when constructing tile halos.
    image = [[rng.randrange(1, 498) for _ in range(width)] for _ in range(height)]
    for y, x in product(range(height), range(width)):
        if (pattern == "empty" or (pattern == "sparse" and rng.random() < 0.97)
                or (pattern == "late-column" and x % 32 < 16)):
            image[y][x] = 0
    return image


def _expected_histograms(
    sample: Callable[[int, int], int], width: int, height: int, radius: int
) -> dict[tuple[int, int], Counter[int]]:
    """Count each eligible center's clipped window independently of the tile.

    :param sample: Read a packed code or zero outside the image.
    :param width: Image width in samples.
    :param height: Image height in samples.
    :param radius: Window radius.
    :returns: Independent expected histograms keyed by center coordinates.
    """
    expected = {}
    for y, x in product(range(height), range(width)):
        if sample(x, y):
            expected[x, y] = Counter(
                v for yy in range(y - radius, y + radius + 1)
                for xx in range(x - radius, x + radius + 1) if (v := sample(xx, yy))
            )
    return expected


def _check_tile_coverage(
    tile_width: int, rows: int, invocations: int, radius: int, extent: int,
    capacity: int, window: int, checks: Counter[str]
) -> None:
    """Verify center probes and cooperative loops cover each slot once.

    :param tile_width: Number of output columns inside the workgroup.
    :param rows: Number of output rows inside the workgroup.
    :param invocations: Number of invocations in the workgroup.
    :param radius: Window radius.
    :param extent: Tile width including halo.
    :param capacity: Total sample count in the tile.
    :param window: Window width in samples.
    :param checks: Counters for successful checks.
    """
    # Includes inactive-group early-out stores and partial tiles.
    probes = Counter()
    lanes = invocations // rows
    for lid in range(invocations):
        lane, row = lid % lanes, lid // lanes
        for x in range(lane, tile_width, lanes):
            at = (row + radius) * extent + x + radius
            assert 0 <= at < capacity
            probes[x, row] += 1
    assert len(probes) == tile_width * rows and set(probes.values()) == {1}
    for count in (capacity, rows * 256, window * window, 256):
        cooperative = Counter(
            i for lid in range(invocations) for i in range(lid, count, invocations)
        )
        assert len(cooperative) == count and set(cooperative.values()) == {1}
        checks["cooperative_coverage_loops"] += 1


def _replace_sample(hist: Counter[int], leaving: int, entering: int) -> None:
    """Remove the outgoing code and add the incoming code in order.

    :param hist: Histogram for the current window.
    :param leaving: Outgoing packed code or zero.
    :param entering: Incoming packed code or zero.
    """
    if leaving:
        assert hist[leaving] > 0
        hist[leaving] -= 1
    if entering:
        hist[entering] += 1


def _clone_histogram(
    previous: Counter[int], tile: list[int], row: int, extent: int, window: int
) -> Counter[int]:
    """Clone the preceding dense row's histogram and update its boundary.

    :param previous: Histogram of the previous output row.
    :param tile: Tile samples including the halo.
    :param row: Current output row within the tile.
    :param extent: Tile width including halo.
    :param window: Window width in samples.
    :returns: Histogram for the current output row.
    """
    hist = previous.copy()
    for xx in range(window):
        leaving = tile[(row - 1) * extent + xx]
        entering = tile[(row + window - 1) * extent + xx]
        _replace_sample(hist, leaving, entering)
    return hist


def _initial_histograms(
    tile: list[int], active: list[bool], extent: int, window: int, checks: Counter[str]
) -> list[Counter[int]]:
    """Initialize each row's first window using the shader's clone rule.

    :param tile: Tile samples including the halo.
    :param active: Whether each output row has an eligible center.
    :param extent: Tile width including halo.
    :param window: Window width in samples.
    :param checks: Counters for successful checks.
    :returns: First histogram for each output row.
    """
    histograms = []
    for row, row_active in enumerate(active):
        if all(active) and row:
            hist = _clone_histogram(histograms[-1], tile, row, extent, window)
            checks["dense_row_clones"] += 1
        elif row_active:
            hist = Counter(
                tile[(row + yy) * extent + xx] for yy in range(window)
                for xx in range(window) if tile[(row + yy) * extent + xx]
            )
        else:
            hist = Counter()
        histograms.append(hist)
    return histograms


def _check_center(
    sample: Callable[[int, int], int], out_x: int, out_y: int, width: int, height: int,
    hist: Counter[int], window: int, expected: dict[tuple[int, int], Counter[int]],
    coverage: Counter[tuple[int, int]], checks: Counter[str]
) -> None:
    """Compare one in-image center with the independent clipped-window oracle.

    :param sample: Read a packed code or zero outside the image.
    :param out_x: Center column in the image.
    :param out_y: Center row in the image.
    :param width: Image width in samples.
    :param height: Image height in samples.
    :param hist: Sliding histogram at the current center.
    :param window: Window width in samples.
    :param expected: Expected clipped-window histograms.
    :param coverage: Counter of visited image coordinates.
    :param checks: Counters for successful checks.
    """
    if out_x >= width or out_y >= height:
        return
    coverage[out_x, out_y] += 1
    if sample(out_x, out_y):
        assert +hist == expected[out_x, out_y]
        # The 16-bit halves cannot carry or borrow into their packed neighbor.
        assert max(hist.values(), default=0) <= window * window <= 1089
        checks["exact_center_histograms"] += 1


def _slide_histogram(
    tile: list[int], hist: Counter[int], x: int, row: int, tile_width: int,
    row_active: bool, extent: int, capacity: int, window: int, checks: Counter[str]
) -> None:
    """Advance an active row by one column, checking both tile addresses.

    :param tile: Tile samples including the halo.
    :param hist: Sliding histogram at the current center.
    :param x: Current output column inside the workgroup.
    :param row: Current output row inside the workgroup.
    :param tile_width: Number of output columns inside the workgroup.
    :param row_active: Whether the row contains an eligible center.
    :param extent: Tile width including halo.
    :param capacity: Total sample count in the tile.
    :param window: Window width in samples.
    :param checks: Counters for successful checks.
    """
    if x + 1 == tile_width or not row_active:
        return
    for k in range(window):
        at = (row + k) * extent + x
        assert 0 <= at < capacity and 0 <= at + window < capacity
        leaving, entering = tile[at], tile[at + window]
        if leaving != entering:
            _replace_sample(hist, leaving, entering)
    checks["horizontal_slides"] += 1


def _check_tile(
    sample: Callable[[int, int], int], width: int, height: int, origin_x: int,
    origin_y: int, radius: int, geometry: tuple[int, int, int],
    expected: dict[tuple[int, int],
                   Counter[int]], coverage: Counter[tuple[int,
                                                          int]], checks: Counter[str]
) -> None:
    """Simulate a workgroup's tile loads, initialization and horizontal scan.

    :param sample: Read a packed code or zero outside the image.
    :param width: Image width in samples.
    :param height: Image height in samples.
    :param origin_x: Image column of the first output in the tile.
    :param origin_y: Image row of the first output in the tile.
    :param radius: Window radius.
    :param geometry: Output columns, rows and invocations for the workgroup.
    :param expected: Expected clipped-window histograms.
    :param coverage: Counter of visited image coordinates.
    :param checks: Counters for successful checks.
    """
    tile_width, rows, invocations = geometry
    extent = tile_width + 2 * radius
    capacity = extent * (rows + 2 * radius)
    window = 2 * radius + 1
    tile = [
        sample(origin_x + xx - radius, origin_y + yy - radius)
        for yy in range(rows + 2 * radius) for xx in range(extent)
    ]
    active = [
        any(sample(origin_x + x, origin_y + row) for x in range(tile_width))
        for row in range(rows)
    ]
    _check_tile_coverage(
        tile_width, rows, invocations, radius, extent, capacity, window, checks
    )
    histograms = _initial_histograms(tile, active, extent, window, checks)
    for x, row in product(range(tile_width), range(rows)):
        hist = histograms[row]
        _check_center(
            sample, origin_x + x, origin_y + row, width, height, hist, window, expected,
            coverage, checks
        )
        _slide_histogram(
            tile, hist, x, row, tile_width, active[row], extent, capacity, window,
            checks
        )


def _check_histogram_geometry(
    sample: Callable[[int, int], int], width: int, height: int, radius: int,
    geometry: tuple[int, int, int], expected: dict[tuple[int, int],
                                                   Counter[int]], checks: Counter[str]
) -> None:
    """Simulate all tiles and verify that each image output has one owner.

    :param sample: Read a packed code or zero outside the image.
    :param width: Image width in samples.
    :param height: Image height in samples.
    :param radius: Window radius.
    :param geometry: Output columns, rows and invocations for the workgroup.
    :param expected: Expected clipped-window histograms.
    :param checks: Counters for successful checks.
    """
    tile_width, rows, _ = geometry
    coverage = Counter()
    for gy in range((height + rows - 1) // rows):
        for gx in range((width + tile_width - 1) // tile_width):
            _check_tile(
                sample, width, height, gx * tile_width, gy * rows, radius, geometry,
                expected, coverage, checks
            )
    assert len(coverage) == width * height and set(coverage.values()) == {1}
    checks["cases"] += 1
    checks["outputs_owned_once"] += len(coverage)


def _histogram_case(
    rng: random.Random, width: int, height: int, radius: int, pattern: str,
    geometries: list[tuple[int, int, int]], checks: Counter[str]
) -> None:
    """Build one fixture and compare every geometry with its expected windows.

    :param rng: Generator shared by all image patterns.
    :param width: Image width in samples.
    :param height: Image height in samples.
    :param radius: Window radius.
    :param pattern: Masking pattern applied to the source.
    :param geometries: Workgroup layouts to test in order.
    :param checks: Counters for successful checks.
    """
    image = _pattern_image(rng, width, height, pattern)

    def sample(x: int, y: int) -> int:
        """Read a code or return zero outside the image.

        :param x: Image column.
        :param y: Image row.
        :returns: Packed code, or zero for a halo sample.
        """
        return image[y][x] if 0 <= x < width and 0 <= y < height else 0

    expected = _expected_histograms(sample, width, height, radius)
    for geometry in geometries:
        _check_histogram_geometry(
            sample, width, height, radius, geometry, expected, checks
        )


def histogram_contracts() -> dict[str, int]:
    """Verify sliding histograms and sample ownership on synthetic inputs.

    :returns: Counts of checked centers, updates, cooperative loads and outputs.
    """
    rng = random.Random(0xCA8B1)
    geometries = [
        (16, 8, 256), (32, 8, 256), (16, 4, 256), (32, 4, 256), (16, 4, 128),
        (32, 4, 128)
    ]
    checks = Counter()
    for (width, height), radius, pattern in product(
        ((35, 37), (65, 41)), (0, 8, 16), ("dense", "sparse", "empty", "late-column")):
        _histogram_case(rng, width, height, radius, pattern, geometries, checks)
    return dict(checks)


def _clear_lane(words: list[int], lane: int, flag_slot: int) -> None:
    """Clear one invocation's words while preserving the shared flag.

    :param words: Workgroup storage modeled by the probe.
    :param lane: Invocation index.
    :param flag_slot: Address of the shared activity flag.
    """
    for at in range(lane, len(words), 256):
        if at != flag_slot:
            words[at] = 0


def _check_term_roundtrips(rng: random.Random, words: list[int], rows: int) -> int:
    """Round-trip floating terms through their 32-bit storage words.

    :param rng: Generator shared across schedules.
    :param words: Workgroup storage after the histogram clear.
    :param rows: Number of rows in the workgroup.
    :returns: Number of exact term round trips.
    """
    terms = 0
    for row in range(rows):
        for lane in range(4):
            value = struct.unpack("<f", struct.pack("<f", rng.random() * 1089))[0]
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            words[row * 256 + 249 + lane] = bits
            recovered = struct.unpack(
                "<f", struct.pack("<I", words[row * 256 + 249 + lane])
            )[0]
            assert recovered == value
            terms += 1
    return terms


def _check_tail_schedule(
    rng: random.Random, rows: int, flag_slot: int, mask: int, used_bins: set[int],
    term_slots: set[int]
) -> int:
    """Check one interleaved read/clear schedule and subsequent storage reuse.

    :param rng: Generator shared across schedules.
    :param rows: Number of rows in the workgroup.
    :param flag_slot: Address of the shared activity flag.
    :param mask: Initial activity mask.
    :param used_bins: Addresses occupied by histogram bins.
    :param term_slots: Addresses reused for floating-point terms.
    :returns: Number of exact term round trips.
    """
    words = [0xDEADBEEF] * (rows * 256)
    words[flag_slot] = mask
    cached = [None] * 256
    # Arbitrary interleaving of each invocation's mask read and
    # subsequent initial-clear loop. Skipping the flag makes all
    # reads safe without adding a barrier.
    pending = list(range(256))
    clearing = []
    while pending or clearing:
        if pending and (not clearing or rng.randrange(2)):
            lane = pending.pop(rng.randrange(len(pending)))
            cached[lane] = words[flag_slot]
            clearing.append(lane)
        else:
            lane = clearing.pop(rng.randrange(len(clearing)))
            _clear_lane(words, lane, flag_slot)
    assert cached == [mask] * 256
    assert all(words[at] == 0 for at in used_bins | term_slots)
    # Existing halo barrier ends the flag lifetime before dense
    # clones can copy the full 256-word rows over the flag slot.
    if mask == (1 << rows) - 1:
        for row in range(1, rows):
            words[row * 256:(row + 1) * 256] = words[(row - 1) * 256:row * 256]
    terms = _check_term_roundtrips(rng, words, rows)
    assert all(words[at] == 0 for at in used_bins)
    return terms


def tail_lifetime_contracts() -> dict[str, int | bool]:
    """Check that live histogram tails survive reuse across phases.

    :returns: Interleaved schedule counts, term counts and slot-disjointness result.
    """
    rng = random.Random(0x7A11)
    schedules = terms = 0
    for rows in (4, 8):
        used_bins = {
            row * 256 + bin_index // 2
            for row in range(rows)
            for bin_index in range(497)
        }
        term_slots = {
            row * 256 + 249 + lane
            for row in range(rows)
            for lane in range(4)
        }
        flag_slot = rows * 256 - 1
        assert used_bins.isdisjoint(term_slots)
        assert flag_slot not in used_bins and flag_slot not in term_slots
        masks = (0, 1, (1 << rows) - 1, sum(1 << row for row in range(0, rows, 2)))
        for mask, _ in product(masks, range(100)):
            terms += _check_tail_schedule(
                rng, rows, flag_slot, mask, used_bins, term_slots
            )
            schedules += 1
    return {
        "interleaved_flag_clear_schedules": schedules,
        "exact_fp32_term_roundtrips": terms,
        "histogram_term_flag_slots_disjoint": True
    }


def _check_small_dispatch(
    height: int, groups4: int, half_enabled: bool, window: int, limit: int
) -> None:
    """Verify one small-window dispatch stays within the device limit.

    :param height: Image height in samples.
    :param groups4: Workgroup count with four output rows per group.
    :param half_enabled: Whether the half-precision path is available.
    :param window: Contrast window width.
    :param limit: Maximum workgroup count supported on the row axis.
    """
    small = half_enabled and window <= 17 and groups4 <= limit
    rows = 4 if small else 8
    assert (height + rows - 1) // rows <= limit
    if not half_enabled or window > 17 or groups4 > limit:
        assert rows == 8


def small_selection_contracts() -> dict[str, int | bool]:
    """Check the rank selection contract on short windows.

    :returns: Cases checked and dispatch-contract outcomes.
    """
    cases = 0
    for height in (33, 67, 1080):
        groups4, groups8 = (height + 3) // 4, (height + 7) // 8
        for half_enabled, window, limit in product((False, True), (1, 17, 19, 33),
                                                   (groups8, groups4, 65535)):
            _check_small_dispatch(height, groups4, half_enabled, window, limit)
            cases += 1
    return {
        "cases": cases,
        "baseline_dispatch_limits_preserved": True,
        "disabled_half_and_window33_keep_original_geometry": True
    }


def main() -> None:
    """Run CPU and compiler checks and save the CAMBI contract report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "artifacts/reports/vmaf-v1-headroom/contrast-contracts"
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "utc":
        datetime.now(UTC).isoformat(),
        "source_sha256": {
            str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in [
                SHADER, CAMBI / "vmaf_cambi.cc", CAMBI / "vmaf_cambi.h",
                Path("CMakeLists.txt")
            ]
        },
        "compiler":
        subprocess.run(
            ["glslc", "--version"], capture_output=True, text=True, check=True
        ).stdout,
        "histogram_contracts":
        histogram_contracts(),
        "tail_lifetime_contracts":
        tail_lifetime_contracts(),
        "small_selection_contracts":
        small_selection_contracts(),
        "shader_contracts":
        compile_contracts(args.out_dir)
    }
    output = args.out_dir / "contracts.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                "histogram_contracts": results["histogram_contracts"],
                "tail_lifetime_contracts": results["tail_lifetime_contracts"],
                "small_selection_contracts": results["small_selection_contracts"],
                "validated_shaders": len(results["shader_contracts"]),
                "evidence": str(output)
            }, indent=2
        )
    )


if __name__ == "__main__":
    main()
