# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the HLS co-simulation evidence run script
"""Tests for scripts/run_hls_cosimulation_evidence.py.

The handoff run is stubbed via a canned artifact so the CLI wiring (argument
parsing, config assembly, JSON output, console summary, pass/fail exit code)
is exercised without a compiler invocation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.benchmarks.hls_cosimulation_evidence import HLSCosimulationHandoff
from scripts import run_hls_cosimulation_evidence as script


def _canned_artifact(passed: bool) -> HLSCosimulationHandoff:
    return HLSCosimulationHandoff(
        evidence={
            "passed": passed,
            "samples_streamed": 256 if passed else 0,
            "compiler_version": "g++ (stub) 13.0.0",
        },
        bundle_meta={"target_sku": "zu3eg", "n_samples": 256},
        consumer_contract="sc-neurocore.hdl_gen.hls_ingest.v1",
        timing_grade="advisory_shared_host",
        host={"ready": False},
        config={"n_samples": 256},
        provenance={"git_commit": "abc"},
        notes=("no synthesis, no timing closure, no board execution",),
    )


class TestArgumentParsing:
    def test_defaults(self) -> None:
        args = script._parse_args([])
        assert (args.samples, args.target_sku, args.compiler) == (256, "zu3eg", "g++")
        assert Path(args.out_dir) == script.DEFAULT_OUT_DIR

    def test_custom_arguments(self) -> None:
        args = script._parse_args(
            ["--samples", "64", "--amplitude", "0.5", "--target-sku", "zu9eg", "--compiler", "cc"]
        )
        assert (args.samples, args.amplitude, args.target_sku, args.compiler) == (
            64,
            0.5,
            "zu9eg",
            "cc",
        )


class TestMain:
    def test_writes_artifact_and_reports_pass(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run(config: Any) -> Any:
            captured["config"] = config
            return _canned_artifact(True)

        monkeypatch.setattr(script, "run_hls_cosimulation_handoff", fake_run)
        code = script.main(["--out-dir", str(tmp_path), "--samples", "256"])
        assert code == 0

        config = captured["config"]
        assert (config.n_samples, config.target_sku) == (256, "zu3eg")

        artifacts = list(tmp_path.glob("hls_cosimulation_zu3eg_n256.json"))
        assert len(artifacts) == 1
        payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
        assert payload["evidence"]["passed"] is True
        assert payload["consumer_contract"] == "sc-neurocore.hdl_gen.hls_ingest.v1"

        out = capsys.readouterr().out
        assert "co-simulation passed: True (256 samples)" in out
        assert "timing_grade: advisory_shared_host" in out
        assert "no synthesis" in out
        assert "written:" in out

    def test_failure_evidence_still_written_with_nonzero_exit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            script, "run_hls_cosimulation_handoff", lambda config: _canned_artifact(False)
        )
        code = script.main(["--out-dir", str(tmp_path)])
        assert code == 1
        payload = json.loads(
            (tmp_path / "hls_cosimulation_zu3eg_n256.json").read_text(encoding="utf-8")
        )
        assert payload["evidence"]["passed"] is False

    def test_config_carries_cli_settings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run(config: Any) -> Any:
            captured["config"] = config
            return _canned_artifact(True)

        monkeypatch.setattr(script, "run_hls_cosimulation_handoff", fake_run)
        code = script.main(
            [
                "--out-dir",
                str(tmp_path),
                "--samples",
                "64",
                "--amplitude",
                "0.5",
                "--sample-rate-hz",
                "125e6",
                "--compiler",
                "g++-13",
                "--reserved-core",
                "1",
            ]
        )
        assert code == 0
        config = captured["config"]
        assert (
            config.n_samples,
            config.amplitude,
            config.sample_rate_hz,
            config.compiler,
            config.reserved_core,
        ) == (64, 0.5, 125e6, "g++-13", 1)
