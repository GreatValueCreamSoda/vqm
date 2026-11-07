#!/usr/bin/env python3
# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
"""Test legacy adapters and explicit ranking gate selection on CPU."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from validation import run_corpus as corpus

ROOT = Path(__file__).resolve().parent.parent
PIN = "test-original-pin"


def _load_adapter(name: str, filename: str) -> types.ModuleType:
    """Load reporting code without optional reference runtimes."""
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "validation" / filename
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load adapter {filename}")
    module = importlib.util.module_from_spec(spec)
    torch = types.ModuleType("torch")
    torch.__version__ = "test-runtime"
    display = types.ModuleType("pycvvdp.display_model")
    display.vvdp_display_photo_eotf = object
    modules = {
        name: module,
        "torch": torch,
        "pycvvdp": types.ModuleType("pycvvdp"),
        "pycvvdp.display_model": display
    }
    with (mock.patch.dict(sys.modules, modules), mock.patch.object(sys, "path",
                                                                   sys.path[:]),
          mock.patch.object(sys, "dont_write_bytecode")):
        spec.loader.exec_module(module)
    return module


CVVDP = _load_adapter("cvvdp_agreement_adapter", "cvvdp.py")


def _policy(metric: str, revision: str, configuration: str,
            stratum: str) -> dict[str, object]:
    """Return a policy requiring informative within-source comparisons."""
    return {
        "schema_version": 1,
        "metric": metric,
        "reference_revision": revision,
        "margins": {
            configuration: .01
        },
        "required_strata": {
            stratum: {}
        },
        "numerical_max": .001,
        "numerical_mean": .001
    }


def _raw_inputs(
    root: Path
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], float]]:
    """Create three source files with four distinct variants each."""
    pairs = []
    mcos = {}
    for source in range(3):
        original = Path(f"original/source-{source}.png")
        (root / original).parent.mkdir(parents=True, exist_ok=True)
        (root / original).write_bytes(f"original bytes {source}".encode())
        for variant in range(4):
            distorted = Path(
                f"compressed/source-{source}/codec/variant-{variant}.png"
            )
            (root / distorted).parent.mkdir(parents=True, exist_ok=True)
            (root / distorted).write_bytes(
                f"distorted bytes {source}/{variant}".encode()
            )
            pair = (str(original), str(distorted))
            pairs.append(pair)
            mcos[pair] = float(1 + 10 * source + variant)
    return pairs, mcos


class CorpusAgreementAdapterTests(unittest.TestCase):
    """Retain old reporting while making requested gates fail closed."""

    def setUp(self) -> None:
        """Create raw input, executable and frozen-policy provenance."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.pairs, self.mcos = _raw_inputs(self.root)
        self.policy = self.root / "policy.json"
        self.policy.write_text(
            json.dumps(
                _policy("ssimulacra2", PIN, "ssimulacra2:cid22", "cid22")
            )
        )
        self.entries = []
        for name in ("cloudinary", "ours"):
            binary = self.root / name
            binary.write_bytes(f"measured binary {name}".encode())
            self.entries.append(
                {
                    "name": name,
                    "command": [str(binary)],
                    "batch": True
                }
            )
        self.spec = {
            "reference": "cloudinary",
            "direction": "higher",
            "implementations": self.entries
        }
        self.args = SimpleNamespace(
            data_root=str(self.root), csv=None,
            out_dir=str(self.root / "reports"), timeout=1, jobs=1,
            max_threshold=.01, mean_threshold=.001, accuracy_tolerance=.005,
            strict_parity=False, agreement_policy=self.policy, gate_mode=None
        )
        self.scores = [self.mcos[pair] for pair in self.pairs]

    def _run(self, offset: float = 3.0) -> bool:
        """Run reporting and gate logic with complete mocked score arms."""
        values = [
            [(score, None) for score in self.scores],
            [(score + offset, None) for score in self.scores]
        ]
        with (mock.patch.object(corpus, "_reference_revision",
                                return_value=PIN),
              mock.patch.object(corpus, "batch_scores", side_effect=values),
              redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            return corpus.run_metric(
                "ssimulacra2", self.spec, self.pairs, self.mcos, self.args, None
            )

    def test_unqualified_default_rejects_offset_and_ranking_accepts_it(
        self
    ) -> None:
        """Keep numerical defaults for unregistered domains."""
        self.assertFalse(self._run())
        path = Path(self.args.out_dir)
        result = json.loads((path / "ssimulacra2-agreement.json").read_text())
        self.assertEqual(result["gate_mode"], "both")
        self.args.gate_mode = "ranking"
        self.assertTrue(self._run())
        result = json.loads((path / "ssimulacra2-agreement.json").read_text())
        self.assertEqual(result["ranking"]["status"], "pass")
        self.assertEqual(result["numerical"]["status"], "fail")
        self.assertFalse(result["legacy_numerical_passed"])
        original = json.loads((path / "ssimulacra2-original.json").read_text())
        candidate = json.loads(
            (path / "ssimulacra2-candidate.json").read_text()
        )
        self.assertEqual(original["revision"], PIN)
        self.assertTrue(candidate["revision"].startswith("sha256:"))
        self.assertEqual(
            original["configuration_hash"], candidate["configuration_hash"]
        )
        self.assertEqual(len(original["samples"]), len(self.pairs))

    def test_qualified_default_accepts_offset_and_explicit_both_rejects_it(
        self
    ) -> None:
        """Use ranking defaults and retain failed raw errors for CID22."""
        directory = ROOT / "validation/fixtures/cid22-balanced"
        original = json.loads(
            (directory / "ssimulacra2-original.json").read_text()
        )
        policy = json.loads((directory / "ssimulacra2-policy.json").read_text())
        candidate = copy.deepcopy(original)
        for sample in candidate["samples"]:
            sample["score"] += 3
        context = {
            "metric": "ssimulacra2",
            "reference": "cloudinary",
            "policy": policy,
            "fingerprints": {},
            "inputs": [],
            "configuration": {}
        }
        Path(self.args.out_dir).mkdir(parents=True)
        with (mock.patch.object(corpus, "_agreement_report",
                                side_effect=[original, candidate, original,
                                             candidate]),
              redirect_stdout(io.StringIO())):
            self.assertTrue(
                corpus._write_agreement(context, {}, self.args, False)
            )
            path = Path(self.args.out_dir) / "ssimulacra2-agreement.json"
            result = json.loads(path.read_text())
            self.assertEqual(result["gate_mode"], "ranking")
            self.assertEqual(result["numerical"]["status"], "fail")
            self.assertFalse(result["legacy_numerical_passed"])
            self.assertIsNone(self.args.gate_mode)
            self.args.gate_mode = "both"
            self.assertFalse(
                corpus._write_agreement(context, {}, self.args, False)
            )

    def test_parser_keeps_automatic_selection_unresolved(self) -> None:
        """Match the actual reference and policy before choosing defaults."""
        for options in ([], ["--agreement-policy", str(self.policy)]):
            with mock.patch.object(sys, "argv", ["run_corpus.py", *options]):
                self.assertIsNone(corpus.parse_args().gate_mode)

    def test_numerical_mode_and_unchanged_scores(self) -> None:
        """Require explicit thresholds for numerical-only qualification."""
        self.args.gate_mode = "numerical"
        self.assertFalse(self._run())
        self.assertTrue(self._run(0.0))

    def test_missing_reference_never_qualifies(self) -> None:
        """Reject requested gates when legacy references are missing."""
        with (mock.patch.object(corpus, "batch_scores") as scored,
              redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            result = corpus.run_metric(
                "ssimulacra2", self.spec, self.pairs, self.mcos, self.args,
                {"ours"}
            )
        self.assertFalse(result)
        scored.assert_not_called()

    def test_mismatched_pin_rejects_before_scoring(self) -> None:
        """Match frozen policy pins to the locked original revision."""
        with (mock.patch.object(corpus, "_reference_revision",
                                return_value="other-pin"),
              mock.patch.object(corpus, "batch_scores") as scored,
              redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            result = corpus.run_metric(
                "ssimulacra2", self.spec, self.pairs, self.mcos, self.args, None
            )
        self.assertFalse(result)
        scored.assert_not_called()

    def test_changed_source_and_missing_scores_cannot_shrink_evidence(
        self
    ) -> None:
        """Recheck source bytes and reject missing canonical scores."""
        with mock.patch.object(corpus, "_reference_revision", return_value=PIN):
            context = corpus._agreement_context(
                "ssimulacra2", self.spec, self.entries, self.pairs, self.mcos,
                self.args
            )
        scores = {"cloudinary": self.scores, "ours": self.scores[:]}
        scores["ours"][0] = None
        self.assertRaises(
            ValueError, corpus._agreement_report, context, scores, "ours"
        )
        target = self.root / self.pairs[0][1]
        target.write_bytes(b"changed during scoring")
        self.assertRaises(
            ValueError, corpus._write_agreement, context, scores, self.args,
            True
        )

    def test_gate_mode_requires_explicit_policy(self) -> None:
        """Never switch acceptance semantics with a bare gate-mode flag."""
        with (mock.patch.object(sys, "argv",
                                ["run_corpus.py", "--gate-mode", "ranking"]),
              redirect_stderr(io.StringIO())):
            self.assertRaises(SystemExit, corpus.parse_args)


class CvvdpAgreementAdapterTests(unittest.TestCase):
    """Qualify decoded RGB identity without requiring PyTorch."""

    def setUp(self) -> None:
        """Create decoded records, a retained binary and a frozen policy."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.policy = self.root / "policy.json"
        self.policy.write_text(
            json.dumps(_policy("cvvdp", CVVDP.PIN, "standard_4k", "rgb8"))
        )
        (self.root / "candidate-binary").write_bytes(b"measured candidate")
        self.args = SimpleNamespace(
            out_dir=self.root, agreement_policy=self.policy, gate_mode=None,
            display="standard_4k"
        )
        self.rows = []
        for source in range(3):
            for variant in range(4):
                original = self._source(f"original-{source}")
                distorted = self._source(f"distorted-{source}-{variant}")
                self.rows.append((original, distorted))
        self.original = [
            {
                "score": float(1 + index)
            } for index in range(len(self.rows))
        ]
        self.candidate = [row["score"] + 3 for row in self.original]

    def _source(self, name: str) -> object:
        """Record independent decoded and raw input identities."""
        decoded = hashlib.sha256(f"decoded:{name}".encode()).hexdigest()
        raw = hashlib.sha256(f"encoded:{name}".encode()).hexdigest()
        return CVVDP.Source(
            str(self.root / name), raw, str(self.root / f"{name}.png"), decoded,
            32, 32
        )

    def _run(self, numerical: bool = False) -> bool:
        """Write canonical reports with a small stand-in configuration."""
        with (mock.patch.object(CVVDP, "_agreement_configuration",
                                return_value={"display": "standard_4k", "input":
                                              "RGB8"}),
              redirect_stdout(io.StringIO())):
            result = CVVDP.write_agreement(
                self.rows, self.original, self.candidate, self.args, numerical
            )
        return result["selected_gate_passed"]

    def test_offset_preserves_ranking_and_decoded_provenance(self) -> None:
        """Keep both gates and decoded provenance for unknown domains."""
        self.assertFalse(self._run())
        result = json.loads((self.root / "agreement.json").read_text())
        self.assertEqual(result["gate_mode"], "both")
        self.args.gate_mode = "ranking"
        self.assertTrue(self._run())
        original = json.loads((self.root / "original.json").read_text())
        candidate = json.loads((self.root / "candidate.json").read_text())
        first = original["samples"][0]
        self.assertEqual(first["source_id"], self.rows[0][0].rgb_sha256)
        self.assertNotEqual(first["source_id"], self.rows[0][0].source_sha256)
        self.assertEqual(
            first["input_hash"], candidate["samples"][0]["input_hash"]
        )
        result = json.loads((self.root / "agreement.json").read_text())
        self.assertFalse(result["legacy_numerical_passed"])
        self.assertEqual(result["ranking"]["status"], "pass")

    def test_qualified_default_is_resolved_from_the_exact_report(self) -> None:
        """Use hosted ranking defaults without changing requested modes."""
        directory = ROOT / "validation/temporal/oracles/qualification"
        self.args.agreement_policy = directory / "cvvdp-policy.json"
        original = json.loads((directory / "cvvdp.json").read_text())
        candidate = copy.deepcopy(original)
        for sample in candidate["samples"]:
            sample["score"] += 3
        with (mock.patch.object(CVVDP, "_agreement_reports",
                                return_value=(original, candidate)),
              redirect_stdout(io.StringIO())):
            result = CVVDP.write_agreement(
                self.rows, self.original, self.candidate, self.args, False
            )
            self.assertTrue(result["selected_gate_passed"])
            self.assertEqual(result["gate_mode"], "ranking")
            self.assertEqual(result["numerical"]["status"], "fail")
            self.assertIsNone(self.args.gate_mode)
            self.args.gate_mode = "both"
            result = CVVDP.write_agreement(
                self.rows, self.original, self.candidate, self.args, False
            )
            self.assertFalse(result["selected_gate_passed"])

    def test_legacy_report_records_the_resolved_gate_and_raw_failure(
        self
    ) -> None:
        """Keep raw parity separate from selected aggregate acceptance."""
        self.args.binary = self.root / "candidate-binary"
        self.args.tolerance = .001
        selected = {"gate_mode": "ranking", "selected_gate_passed": True}
        with (mock.patch.object(CVVDP, "write_agreement",
                                return_value=selected),
              mock.patch.object(CVVDP, "candidate_provenance", return_value={}),
              redirect_stdout(io.StringIO())):
            self.assertTrue(
                CVVDP.write_report(
                    self.rows, self.original, self.candidate, self.args
                )
            )
        result = json.loads((self.root / "results.json").read_text())
        self.assertEqual(result["gate_mode"], "ranking")
        self.assertFalse(result["passed"])
        self.assertTrue(result["selected_gate_passed"])
        self.assertEqual(result["max_absolute_error"], 3)
        self.assertIsNone(self.args.gate_mode)

    def test_parser_keeps_unqualified_selection_unresolved(self) -> None:
        """Resolve CVVDP still defaults after building original reports."""
        with mock.patch.object(sys, "argv", ["cvvdp.py"]):
            self.assertIsNone(CVVDP.arguments().gate_mode)

    def test_candidate_keeps_reference_display_ppd(self) -> None:
        """Keep each still display's PPD fixed across input sizes."""
        self.args.binary = self.root / "built-binary"
        self.args.binary.write_bytes(b"candidate binary")
        for display, ppd in (("standard_4k", "75.402449345163049"),
                             ("standard_fhd", "37.842503853602885")):
            self.args.display = display
            outputs = [
                subprocess.CompletedProcess([], 0, stdout="{}\n", stderr=""),
                subprocess.CompletedProcess([], 0, stdout="9.5\n", stderr=""),
            ]
            with (self.subTest(display=display),
                  mock.patch.object(CVVDP.subprocess, "run",
                                    side_effect=outputs) as run):
                self.assertEqual(
                    CVVDP.candidate_scores(self.rows[:1], self.args), [9.5]
                )
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--cvvdp-ppd") + 1], ppd)

    def test_policy_numerical_thresholds_cannot_be_ignored(self) -> None:
        """Preserve tighter frozen bounds despite legacy arguments."""
        self.args.gate_mode = "numerical"
        self.assertFalse(self._run(numerical=True))

    def test_original_only_and_wrong_pin_are_rejected(self) -> None:
        """Treat original-only runs as evidence generation."""
        self.assertRaises(
            ValueError, CVVDP.write_agreement, self.rows, self.original, None,
            self.args, False
        )
        self.args.reference_only = True
        self.args.gate_mode = "both"
        parser = CVVDP.argparse.ArgumentParser()
        with redirect_stderr(io.StringIO()):
            self.assertRaises(
                SystemExit, CVVDP._agreement_options, self.args, parser
            )
        self.args.reference_only = False
        self.policy.write_text(
            json.dumps(_policy("cvvdp", "wrong-pin", "standard_4k", "rgb8"))
        )
        with redirect_stderr(io.StringIO()):
            self.assertRaises(
                SystemExit, CVVDP._agreement_options, self.args, parser
            )

    def test_authoritative_cid22_mos_is_retained_when_available(self) -> None:
        """Bootstrap whole sources using corpus labels, not scores."""
        self.args.corpus = self.root
        (self.root / "CID22_validation_set.csv").write_text(
            "reference_img,distorted_img,MCOS\noriginal.png,distorted.png,77\n"
        )
        original = CVVDP.dataclasses.replace(
            self.rows[0][0], source=str(self.root / "original.png")
        )
        distorted = CVVDP.dataclasses.replace(
            self.rows[0][1], source=str(self.root / "distorted.png")
        )
        samples = [{"score": 1.0}]
        CVVDP._attach_mos(samples, [(original, distorted)], self.args)
        self.assertEqual(samples[0]["mos"], 77)


if __name__ == "__main__":
    unittest.main()
