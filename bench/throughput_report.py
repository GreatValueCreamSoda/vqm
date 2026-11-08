"""Validate aggregate throughput evidence and render managed README sections."""

import collections
import dataclasses
import datetime
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

from bench import throughput_workload as workload
from bench.inmemory_report import JsonObject
from bench.throughput_workload import METRICS
from bench.throughput_workload import PEERS
from bench.throughput_workload import SIZES
from bench.throughput_workload import Arm

START = "<!-- throughput-comparison:start -->"
END = "<!-- throughput-comparison:end -->"
LABELS = {
    "ssimulacra2": "SSIMULACRA 2",
    "butteraugli": "Butteraugli",
    "psnr-hvs": "PSNR-HVS",
    "cvvdp-image": "CVVDP images",
    "cvvdp-video": "CVVDP video"
}


def _arm_summary(arm: Arm, rows: list[JsonObject], rounds: int) -> JsonObject:
    if len(rows) != rounds or {row["round"]
                               for row in rows} != set(range(1, rounds + 1)):
        raise ValueError(f"incomplete or duplicate rounds: {arm.key}")
    if any(row["status"] != 0 or row["count"] != arm.count for row in rows):
        raise ValueError(f"failed process or incorrect count: {arm.key}")
    scores = [float(row["score"]) for row in rows]
    rates = [float(row["fps"]) for row in rows]
    if not all(math.isfinite(value)
               for value in [*scores, *rates]) or min(rates) <= 0:
        raise ValueError(f"invalid finite score/rate: {arm.key}")
    if max(scores) - min(scores) > 1e-6:
        raise ValueError(f"unstable scores: {arm.key}")
    return {
        "metric": arm.metric,
        "size": arm.size,
        "implementation": arm.implementation,
        "mode": arm.mode,
        "workers": arm.workers,
        "iterations": arm.iterations,
        "frames": arm.frames,
        "count": arm.count,
        "rounds": rounds,
        "median": statistics.median(rates),
        "min": min(rates),
        "max": max(rates),
        "score": statistics.median(scores)
    }


def summarize(arms: list[Arm], rows: list[JsonObject],
              rounds: int) -> list[JsonObject]:
    """Require every completed count and round, then check input-path scores.

    :param arms: Expected full matrix.
    :param rows: Successful per-process measurements.
    :param rounds: Required rounds per arm.
    :returns: Validated median, range and score for every arm.
    :raises ValueError: If coverage, accounting or score stability is invalid.
    """
    grouped: dict[tuple[str, ...],
                  list[JsonObject]] = collections.defaultdict(list)
    for row in rows:
        grouped[_key(row)].append(row)
    expected = {
        (arm.size, arm.metric, arm.implementation, arm.mode)
        for arm in arms
    }
    if set(grouped) != expected:
        raise ValueError("missing or unexpected benchmark arms")
    summary = [
        _arm_summary(
            arm, grouped[(arm.size, arm.metric, arm.implementation, arm.mode)],
            rounds
        ) for arm in arms
    ]
    for resident in (row for row in summary if row["mode"] == "resident"):
        streaming = find(
            summary, resident["size"], resident["metric"], "vqm", "streaming"
        )
        if abs(resident["score"] - streaming["score"]) > 1e-6:
            raise ValueError(
                f"streaming/resident scores disagree: {resident['metric']}"
            )
    return summary


def _key(row: JsonObject) -> tuple[str, ...]:
    return tuple(
        str(row[name]) for name in ("size", "metric", "implementation", "mode")
    )


def find(
    summary: list[JsonObject], size: str, metric: str, implementation: str,
    mode: str = "streaming"
) -> JsonObject:
    """Find one explicitly measured rate without substituting a missing peer.

    :param summary: Validated aggregate measurements.
    :param size: Pixel geometry.
    :param metric: Metric and still/video mode.
    :param implementation: Candidate or peer name.
    :param mode: Streaming or resident inputs.
    :returns: Matching row, or an empty dictionary for an unmeasured arm.
    """
    target = (size, metric, implementation, mode)
    return next((row for row in summary if _key(row) == target), {})


def _rate(row: JsonObject, spread: bool = False) -> str:
    if not row:
        return "—"
    value = f"{row['median']:,.1f}"
    return f"{value} ({row['min']:,.1f}–{row['max']:,.1f})" if spread else value


def _vqm_table(report: JsonObject, spread: bool = False) -> str:
    lines = [
        "| Metric | Workers | 1080p streaming | 1080p resident | 4K "
        "streaming | 4K resident |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"
    ]
    summary = report["summary"]
    for metric in METRICS:
        workers = next(
            row["workers"] for row in summary if row["metric"] == metric
        )
        values = [
            _rate(find(summary, size, metric, "vqm", mode), spread)
            for size in SIZES for mode in ("streaming", "resident")
        ]
        lines.append(
            f"| {LABELS[metric]} | {workers} | " + " | ".join(values) + " |"
        )
    return "\n".join(lines)


def _peer_table(report: JsonObject, spread: bool = False) -> str:
    lines = [
        "| Metric | Size | VQM Vulkan | Vship CUDA | Vship Vulkan | "
        "Vship-fast Vulkan |", "| --- | --- | ---: | ---: | ---: | ---: |"
    ]
    for metric in ("ssimulacra2", "butteraugli", "cvvdp-image"):
        for size in SIZES:
            values = [
                _rate(
                    find(report["summary"], size, metric, implementation),
                    spread
                ) for implementation in ("vqm", *PEERS)
            ]
            lines.append(
                f"| {LABELS[metric]} | {size.replace('x', '×')} | " +
                " | ".join(values) + " |"
            )
    return "\n".join(lines)


def _device(report: JsonObject) -> str:
    return str(report["environment"]["devices"]["ours"]["name"])


def headline(report: JsonObject) -> str:
    """Render the concise main README comparison from validated aggregates.

    :param report: Complete report with protocol, device and summary evidence.
    :returns: Markdown performance tables and essential timing limitations.
    """
    date = datetime.datetime.fromisoformat(report["created_at"]
                                           ).strftime("%B %-d, %Y")
    return (
        f"**Scoring throughput on {_device(report)}**, "
        f"measured {date}: medians\n"
        f"of {report['protocol']['rounds']} rotating, non-overlapping rounds. "
        "Rates are pair evaluations/s\n"
        "for stills and frames/s for temporal CVVDP video. Setup, decoding "
        "and\n"
        "warmups are excluded; dispatch, synchronization and score "
        "readback remain.\n\n"
        "### Streaming versus resident inputs\n\n"
        "Streaming uploads decoded host pixels for each call. Resident inputs\n"
        "are uploaded before timing. These are repeated-input benchmark "
        "rates,\n"
        "not file-to-score CLI FPS.\n\n"
        f"{_vqm_table(report)}\n\n"
        "### Competitor comparison: streaming inputs\n\n"
        "All columns include per-call uploads and exclude setup/decoding. "
        "Spatial\n"
        "metrics use three GPU workers; CVVDP images use one.\n\n"
        f"{_peer_table(report)}\n\n"
        "CVVDP video preserves temporal history and frame order on one "
        "worker; its\n"
        "resident arm preuploads a complete clip. CVVDP images are "
        "independent\n"
        "stills, not video. Peer scores differ; these rates do not establish\n"
        "numerical equivalence. `—` means not measured. VMAF throughput is "
        "not\n"
        "measured in this matrix. Benchmark workers do not change normal "
        "CLI scheduling.\n\n"
        "[Methodology, spread, scores and "
        "limitations](bench/README.md"
        "#streaming-and-resident-scoring-comparison) ·\n"
        "[Run and refresh this entire "
        "comparison](bench/README.md#refresh-the-entire-comparison)\n"
    )


def _score_table(report: JsonObject) -> str:
    lines = [
        "| Metric | Size | VQM | Vship CUDA | Vship Vulkan | Vship-fast "
        "Vulkan |", "| --- | --- | ---: | ---: | ---: | ---: |"
    ]
    for metric in ("ssimulacra2", "butteraugli", "cvvdp-image"):
        for size in SIZES:
            rows = [
                find(report["summary"], size, metric, name)
                for name in ("vqm", *PEERS)
            ]
            values = [f"{row['score']:.8f}" if row else "—" for row in rows]
            lines.append(
                f"| {LABELS[metric]} | {size.replace('x', '×')} | " +
                " | ".join(values) + " |"
            )
    return "\n".join(lines)


def detailed(report: JsonObject) -> str:
    """Render all current rates, spread, workload counts and score differences.

    :param report: Complete validated evidence.
    :returns: Standalone Markdown suitable for the report and benchmark guide.
    """
    if report["protocol"]["profile"] == "smoke":
        return (
            "# Smoke throughput check\n\n"
            "Not publishable performance evidence.\n\n"
        ) + "\n".join(
            f"- {row['metric']} {row['implementation']} {row['mode']}: "
            f"{_rate(row, True)} pairs/s" for row in report["summary"]
        ) + "\n"
    protocol = report["protocol"]
    references = report["environment"]["references"]
    revisions = "\n".join(
        f"- `{name}`: `{value['revision']}`."
        for name, value in references.items()
    )
    counts = "\n".join(
        f"| {LABELS[metric]} | " + " | ".join(
            str(find(report["summary"], size, metric, "vqm")["count"])
            for size in SIZES
        ) + " |" for metric in METRICS
    )
    return (
        f"Measured `{report['created_at']}` on **{_device(report)}**.\n"
        "Candidate core revision: "
        f"`{report['environment']['project']['revision']}`;\n"
        "the report preserves working-tree changes, input/binary/library "
        "hashes,\n"
        "build fingerprints, compiler, CPU, driver inventory and every "
        "process receipt.\n\n"
        f"{revisions}\n\n"
        f"{protocol['rounds']} rotating rounds, {protocol['warmup']} warmups "
        "per worker (whole clips for video).\n"
        "No benchmark processes overlap. Median pair evaluations/s (video "
        "frames/s),\n"
        "with minimum–maximum rates in parentheses:\n\n"
        f"{_vqm_table(report, True)}\n\n"
        "Host-input peer scoring, including per-call uploads:\n\n"
        f"{_peer_table(report, True)}\n\n"
        "Timed counts are **total across workers**, not per worker:\n\n"
        "| Metric | 1920×1080 | 3840×2160 |\n| --- | ---: | ---: |\n"
        f"{counts}\n\n"
        "SSIMULACRA 2 rebuilds both images by alternating resident "
        "reference slots;\n"
        "Butteraugli may reuse its prepared reference. RGB8 synthetic PNG "
        "pairs are\n"
        "testsrc2 versus bicubic half-size/downscale-upscale distortion. "
        "PSNR-HVS\n"
        "uses the first native YUV420p8 clip frame and always prepares "
        "both inputs.\n"
        "Temporal CVVDP uses complete 24 fps YUV clips: 32 frames at "
        "1080p, 16 at 4K,\n"
        "ten timed clips each. Every frame stays on the same sequential "
        "pipeline.\n"
        "The resident video arm preuploads all raw frames into separate "
        "external\n"
        "device-local buffers; pipeline scratch and private temporal "
        "history remain\n"
        "core-owned. This avoids Vulkan's per-buffer storage-range limit "
        "but uses\n"
        "more VRAM than streaming and is not an unbounded-video throughput "
        "claim.\n"
        "CVVDP uses fixed 75.402449345163049 PPD, standard_4k display, no "
        "resizing\n"
        "or heatmaps. Independent stills reset between calls; temporal "
        "clips reset\n"
        "only between clips and report finalized whole-clip JOD.\n\n"
        "All rates exclude setup, decoding and warmups; only streaming "
        "includes\n"
        "uploads. CVVDP stills sum scoring-call wall durations; other arms "
        "use a\n"
        "shared worker interval or whole-clip scoring durations. VQM still "
        "resets\n"
        "are outside the timer, peer still resets are inside its public "
        "API call.\n"
        "Inputs can remain GPU-cache-hot. Device identity matches names, "
        "not UUIDs.\n"
        "The timing-enabled core is unprofiled and uses FP32 planes. No "
        "PSNR-HVS,\n"
        "temporal-video or resident-input peer arm is measured. No VMAF "
        "throughput\n"
        "is claimed. These workloads do not measure normal CLI throughput.\n\n"
        "Scores are stable across rounds; VQM streaming/resident scores match\n"
        "within 1e-6. Peer score differences are not numerical "
        "qualification:\n\n"
        f"{_score_table(report)}\n\n"
        f"Raw evidence: `{protocol['out_dir']}`. "
        "The standalone `summary.md` and\n"
        "`summary.json` preserve these results even after later README "
        "refreshes.\n"
    )


def replace_section(text: str, markdown: str) -> str:
    """Replace exactly one managed section, preserving all unrelated content.

    :param text: Existing document.
    :param markdown: Validated generated Markdown.
    :returns: Document with just the managed section changed.
    :raises ValueError: If markers are missing, duplicated or out of order.
    """
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("README needs exactly one throughput marker pair")
    before, remainder = text.split(START)
    old, after = remainder.split(END)
    if START in old or END in before:
        raise ValueError("invalid throughput marker order")
    return before + START + "\n" + markdown.rstrip() + "\n" + END + after


def publish(report: JsonObject, root: Path) -> None:
    """Refresh both README tables only after validating complete evidence.

    :param report: Full, completed, device-matched comparison.
    :param root: Repository root containing both managed documents.
    :raises ValueError: If the run or document markers are not publishable.
    """
    protocol = report["protocol"]
    if (report["status"] != "complete" or protocol["profile"] != "full"
            or protocol["rounds"] < 5 or protocol["warmup"] < 3
            or protocol["workers"] != 3
            or set(protocol["references"]) != set(PEERS)
            or list(protocol["sizes"]) != list(SIZES)
            or report["environment"]["devices"]["status"]
            != "matched_by_name_only"):
        raise ValueError(
            "refusing to publish incomplete, smoke or unmatched results"
        )
    arms = [Arm(**row) for row in report["matrix"]]
    expected = workload.matrix(
        SimpleNamespace(**protocol),
        workload.entries(Path(protocol["build_dir"]), PEERS)
    )
    if [_contract(arm) for arm in arms] != [_contract(arm) for arm in expected]:
        raise ValueError("refusing to publish a reduced or modified matrix")
    if summarize(arms, report["measurements"],
                 protocol["rounds"]) != report["summary"]:
        raise ValueError("published aggregates disagree with measurements")
    replacements = [
        (root / "README.md", headline(report)),
        (root / "bench/README.md", detailed(report))
    ]
    prepared = [
        (path, replace_section(path.read_text(encoding="utf-8"), markdown))
        for path, markdown in replacements
    ]
    for path, text in prepared:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)


def _contract(arm: Arm) -> JsonObject:
    fields = dataclasses.asdict(arm)
    fields.pop("command")
    return fields
