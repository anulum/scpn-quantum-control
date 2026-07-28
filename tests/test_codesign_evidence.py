# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — co-design evidence tests
"""Measured workflow, artefact, and CLI tests for BL-33 evidence."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from scpn_quantum_control.codesign.evidence import (
    EVIDENCE_CLASSIFICATION,
    EVIDENCE_SCHEMA,
    main,
    run_functional_evidence,
    validate_functional_evidence,
    write_functional_evidence,
)


def test_functional_evidence_measures_and_replays_real_loop() -> None:
    """Measure a real local loop and verify deterministic replay."""
    evidence = run_functional_evidence(iterations=2)

    assert evidence.schema == EVIDENCE_SCHEMA
    assert evidence.classification == EVIDENCE_CLASSIFICATION
    assert evidence.isolated is False
    assert evidence.iterations == 2
    assert evidence.steps_per_iteration == 2
    assert len(evidence.elapsed_ms) == 2
    assert all(value > 0.0 for value in evidence.elapsed_ms)
    assert evidence.throughput_steps_per_second > 0.0
    assert evidence.replay_verified is True
    assert evidence.provider_execution is False
    assert evidence.hardware_execution is False
    assert validate_functional_evidence(evidence.to_dict()) == ()


def test_evidence_writer_creates_valid_json(tmp_path: Path) -> None:
    """Write a JSON-ready evidence payload to a nested destination."""
    destination = tmp_path / "nested" / "evidence.json"
    written = write_functional_evidence(destination, iterations=1)
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload == written.to_dict()
    assert destination.read_text(encoding="utf-8").endswith("\n")
    assert validate_functional_evidence(payload) == ()


def test_evidence_cli_executes_repository_script(tmp_path: Path) -> None:
    """Execute the repository script as an external CLI process."""
    destination = tmp_path / "cli-evidence.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src:oscillatools/src"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_codesign_loop_evidence.py",
            "--output",
            str(destination),
            "--iterations",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert destination.is_file()
    assert "classification=functional_non_isolated" in completed.stdout
    assert "No provider or QPU execution" in completed.stdout
    assert validate_functional_evidence(json.loads(destination.read_text(encoding="utf-8"))) == ()


def test_evidence_main_accepts_explicit_arguments(tmp_path: Path) -> None:
    """Invoke the public CLI function with an explicit destination."""
    destination = tmp_path / "main-evidence.json"

    assert main(["--output", str(destination), "--iterations", "1"]) == 0
    assert destination.exists()


def test_repository_script_entrypoint_delegates_to_public_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the checked-in script entry point under coverage."""
    destination = tmp_path / "runpy-evidence.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/run_codesign_loop_evidence.py",
            "--output",
            str(destination),
            "--iterations",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path("scripts/run_codesign_loop_evidence.py", run_name="__main__")
    assert exit_info.value.code == 0
    assert destination.exists()


def test_evidence_validator_fails_closed_on_drift() -> None:
    """Report every promotion-sensitive evidence-field drift."""
    assert validate_functional_evidence([]) == ("payload must be a JSON object",)
    payload = run_functional_evidence(iterations=1).to_dict()
    payload.update(
        {
            "schema": "wrong",
            "classification": "isolated_affinity",
            "isolated": True,
            "provider_execution": True,
            "hardware_execution": True,
            "replay_verified": False,
            "claim_boundary": "wrong",
            "iterations": 0,
            "steps_per_iteration": 0,
            "median_elapsed_ms": 0.0,
            "throughput_steps_per_second": 0.0,
            "trace_digest": "short",
            "elapsed_ms": [],
        }
    )
    findings = validate_functional_evidence(payload)

    assert len(findings) == 13
    assert "schema must equal" in findings[0]
    assert "elapsed_ms must be a non-empty array" in findings[-1]


def test_evidence_validator_rejects_non_numeric_timings() -> None:
    """Reject boolean timings and non-positive iteration counts."""
    payload = run_functional_evidence(iterations=1).to_dict()
    payload["elapsed_ms"] = [False]

    assert validate_functional_evidence(payload)[-1] == "elapsed_ms values must be positive"
    with pytest.raises(ValueError, match="iterations"):
        run_functional_evidence(iterations=0)
