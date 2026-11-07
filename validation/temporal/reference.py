"""Evaluate prepared clips with pinned independent CPU reference implementations."""

from __future__ import annotations

import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .corpus import ROOT, digest, json_digest
from .spatial_reference import PINS as SPATIAL_PINS
from .spatial_reference import spatial_provenance, spatial_score

PINS = {
    **SPATIAL_PINS, "vmaf-v0": "4611e47a7330cd342498dcc130e1008efbeab6e4",
    "vmaf-v1": "4611e47a7330cd342498dcc130e1008efbeab6e4",
    "cvvdp": "2a268bce8d56e2f3abde46df3927d8a633707a24"
}
MODELS = {
    "vmaf-v0": ROOT / ".reference/vmaf/model/vmaf_float_v0.6.1.json",
    "vmaf-v1": ROOT / ".reference/vmaf/model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json",
}
VMAF = ROOT / "build/vmaf-build/tools/vmaf"
_CVVDP: dict[str, object] = {}


def verify_references(metrics: list[str]) -> None:
    """Check selected original pins and cleanliness without modifying them.

    :param metrics: Requested temporal metric names.
    :raises RuntimeError: If a pinned source is missing or dirty.
    """
    names = {
        "ssimulacra2": "ssimulacra2",
        "butteraugli": "libjxl",
        "vmaf-v0": "vmaf",
        "vmaf-v1": "vmaf",
        "cvvdp": "ColorVideoVDP"
    }
    if any(metric not in names for metric in metrics):
        raise ValueError("unknown reference metric")
    selected = {names[metric] for metric in metrics}
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/fetch_references.py"), "--check", "--only",
            ",".join(sorted(selected))
        ], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


def provenance(metric: str) -> dict[str, object]:
    """Fingerprint the actual independent executable, assets and runtime.

    :param metric: Requested temporal metric.
    :returns: Original revision and execution provenance.
    """
    info: dict[str, object] = {
        "revision": PINS[metric],
        "numpy": np.__version__,
        "runner_sha256": digest(Path(__file__))
    }
    if metric in SPATIAL_PINS:
        return dict(
            info, dispatcher_sha256=info["runner_sha256"], **spatial_provenance(metric)
        )
    if metric != "cvvdp":
        info.update(
            {
                "executable_sha256": digest(VMAF),
                "model_sha256": digest(MODELS[metric]),
                "threads": 1
            }
        )
        return info
    torch, _ = cvvdp_runtime()
    info.update(
        {
            "torch": str(torch.__version__),
            "threads": 1,
            "precision": "CPU-FP32-highest",
            "display": "standard_4k",
            "source_colorspace": "sRGB",
            "temporal_padding": "replicate"
        }
    )
    return info


def cvvdp_runtime() -> tuple[object, object]:
    """Import the pinned original without writing into its checkout.

    :returns: PyTorch module and one reusable original metric instance.
    :raises ImportError: If the optional original runtime is unavailable.
    """
    if _CVVDP:
        return _CVVDP["torch"], _CVVDP["metric"]
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / ".reference/ColorVideoVDP"))
    torch = importlib.import_module("torch")
    upstream = importlib.import_module("pycvvdp")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    metric = upstream.cvvdp(
        display_name="standard_4k", device=torch.device("cpu"), quiet=True,
        temp_padding="replicate"
    )
    _CVVDP.update({"torch": torch, "metric": metric})
    return torch, metric


def vmaf_score(row: dict[str, object], metric: str,
               directory: Path) -> dict[str, object]:
    """Evaluate every frame through the original libvmaf CLI.

    :param row: Prepared clip description.
    :param metric: VMAF model name.
    :param directory: Raw reference-report directory.
    :returns: Complete-clip mean and ordered independent frame metrics.
    :raises RuntimeError: If the original fails or omits frames.
    """
    target = directory / "libvmaf.json"
    command = [
        str(VMAF), "--reference",
        str(Path(str(row["original"])).with_suffix(".yuv")), "--distorted",
        str(Path(str(row["distorted"])).with_suffix(".yuv")), "--width",
        str(row["width"]), "--height",
        str(row["height"]), "--pixel_format", "444", "--bitdepth", "8", "--frame_cnt",
        str(row["frames"]), "--model", f"path={MODELS[metric]}:name=original", "--json",
        "--output",
        str(target), "--threads", "1", "--quiet"
    ]
    result = subprocess.run(
        command, text=True, capture_output=True, timeout=300, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    frames = json.loads(target.read_text())["frames"]
    if len(frames) != row["frames"]:
        raise RuntimeError("original VMAF omitted frames")
    values = [float(frame["metrics"]["original"]) for frame in frames]
    return {
        "score": math.fsum(values) / len(values),
        "frames": len(values),
        "frame_results": frames,
        "command": command
    }


def cvvdp_score(row: dict[str, object]) -> dict[str, object]:
    """Evaluate shared serialized RGB samples through original ColorVideoVDP.

    :param row: Prepared clip geometry and shared RGB sidecars.
    :returns: Whole-clip JOD, latent features and frame count.
    """
    torch, metric = cvvdp_runtime()
    shape = (int(row["frames"]), 3, int(row["height"]), int(row["width"]))
    original = np.fromfile(str(row["original"]) + ".rgb", dtype="<f4").reshape(shape)
    distorted = np.fromfile(str(row["distorted"]) + ".rgb", dtype="<f4").reshape(shape)
    numerator, denominator = (int(value) for value in str(row["fps"]).split(":"))
    with torch.inference_mode():
        score, stats = metric.predict(
            torch.from_numpy(distorted.transpose(1, 0, 2, 3)[None]),
            torch.from_numpy(original.transpose(1, 0, 2, 3)[None]),
            frames_per_second=numerator / denominator
        )
    return {
        "score": float(score),
        "frames": int(row["frames"]),
        "Q_per_ch": stats["Q_per_ch"].tolist(),
        "rho_band": stats["rho_band"].tolist()
    }


def score_reference(
    row: dict[str, object], metric: str, cache: Path, identity: dict[str, object]
) -> dict[str, object]:
    """Cache a reference result only under complete input and runtime fingerprints.

    :param row: Prepared clip pair with representation hashes.
    :param metric: Temporal metric/model name.
    :param cache: Disposable reference-result cache.
    :param identity: Pinned executable/assets/runtime fingerprint.
    :returns: Original score and complete numerical evidence.
    :raises ValueError: If an original produces a nonfinite or incomplete result.
    """
    inputs = {
        key: value
        for key, value in row.items() if key not in ("original", "distorted")
    }
    fingerprint = json_digest({"inputs": inputs, "reference": identity, "protocol": 1})
    directory = cache / metric / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "result.json"
    if path.exists():
        result = json.loads(path.read_text())
        recorded_hash = result.pop("result_sha256", None)
        if (result.get("fingerprint") != fingerprint
                or recorded_hash != json_digest(result)
                or not math.isfinite(float(result["score"]))
                or result["frames"] != row["frames"]):
            raise ValueError(f"corrupt reference cache: {path}")
        result["result_sha256"] = recorded_hash
        return result
    started = time.perf_counter()
    result = _score_uncached(row, metric, directory)
    if not math.isfinite(float(result["score"])) or result["frames"] != row["frames"]:
        raise ValueError("original returned a nonfinite or incomplete clip result")
    result.update(
        {
            "fingerprint": fingerprint,
            "seconds": time.perf_counter() - started
        }
    )
    result["result_sha256"] = json_digest(result)
    path.write_text(json.dumps(result, allow_nan=False) + "\n")
    return result


def _score_uncached(row: dict[str, object], metric: str,
                    directory: Path) -> dict[str, object]:
    if metric in SPATIAL_PINS:
        return spatial_score(row, metric, directory)
    if metric == "cvvdp":
        return cvvdp_score(row)
    return vmaf_score(row, metric, directory)
