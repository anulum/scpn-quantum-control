# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the decisive-advantage run script
"""Tests for scripts/run_decisive_advantage_benchmark.py.

The heavy classical solvers are stubbed via a canned artifact so the CLI wiring
(argument parsing, manifest vs run modes, JSON output, console summary) is
exercised without touching the NumPy coverage tracer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.benchmarks.decisive_run_harness import DecisiveRunArtifact
from scripts import run_decisive_advantage_benchmark as script


def _canned_artifact() -> DecisiveRunArtifact:
    return DecisiveRunArtifact(
        protocol_id="decisive_advantage_order_parameter_n12_2026-07-15",
        n_qubits=12,
        reference_baseline="dense_statevector_evolution",
        reference_order_parameter=0.574,
        timing_grade="advisory_shared_host",
        rows=({"baseline": "classical_ode", "status": "ok"},),
        validation={"valid": True},
        decision={"label": "inconclusive", "reasons": ["no qpu_hardware row"]},
        provenance={"git_commit": "abc"},
        host_readiness={"ready": False},
        claim_boundary="bounded",
    )


class TestPreregistrationMode:
    def test_manifest_written(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        code = script.main(["--out-dir", str(tmp_path)])
        assert code == 0
        manifests = list(tmp_path.glob("*.manifest.json"))
        assert len(manifests) == 1
        payload = json.loads(manifests[0].read_text(encoding="utf-8"))
        assert "protocol" in payload
        out = capsys.readouterr().out
        assert "preregistration only" in out


class TestRunMode:
    def test_run_writes_artifact_and_reports_decision(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def _fake_run(protocol: Any, config: Any) -> DecisiveRunArtifact:
            captured["include_mps"] = config.include_mps
            captured["t_max"] = config.t_max
            return _canned_artifact()

        monkeypatch.setattr(script, "run_decisive_benchmark", _fake_run)
        code = script.main(
            ["--run", "--no-mps", "--t-max", "0.5", "--dt", "0.1", "--out-dir", str(tmp_path)]
        )
        assert code == 0
        assert captured["include_mps"] is False
        assert captured["t_max"] == 0.5
        artifacts = list(tmp_path.glob("*.artifact.json"))
        assert len(artifacts) == 1
        payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
        assert payload["decision"]["label"] == "inconclusive"
        out = capsys.readouterr().out
        assert "decision: inconclusive" in out
        assert "timing_grade: advisory_shared_host" in out

    def test_run_defaults_enable_mps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def _fake_run(protocol: Any, config: Any) -> DecisiveRunArtifact:
            captured["include_mps"] = config.include_mps
            return _canned_artifact()

        monkeypatch.setattr(script, "run_decisive_benchmark", _fake_run)
        assert script.main(["--run", "--out-dir", str(tmp_path)]) == 0
        assert captured["include_mps"] is True
