# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Quantum Advantage Gap Audit Runner
"""Tests for the quantum advantage gap audit runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_quantum_advantage_gap_audit.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "_run_quantum_advantage_gap_audit",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit_module = _load_script_module()
build_audit_payload = audit_module.build_audit_payload
classify_advantage_status = audit_module.classify_advantage_status
evaluate_s2_matrix_readiness = audit_module.evaluate_s2_matrix_readiness


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_classify_advantage_status_keeps_broad_advantage_open():
    status = classify_advantage_status(
        crossover_qubits=11.6,
        max_hardware_n=16,
        fastest_ode_ms_at_max_n=12.0,
        exact_oom_at_max_n=True,
    )

    assert status["current_label"] == "exact_hilbert_space_crossover_only"
    assert status["broad_quantum_advantage_supported"] is False
    assert status["exact_simulation_crossover_supported"] is True
    assert status["requires_ibm_hardware"] is False


def test_build_audit_payload_records_committed_sources_and_guardrails():
    payload = build_audit_payload(command=["python", "scripts/run_quantum_advantage_gap_audit.py"])

    assert payload["audit"] == "quantum_advantage_gap"
    assert payload["decision"]["current_label"] == "exact_hilbert_space_crossover_only"
    assert payload["decision"]["broad_quantum_advantage_supported"] is False
    assert payload["decision"]["ready_for_ibm_advantage_run"] is False
    assert (
        payload["s2_matrix_readiness"]["decision"] == "blocked_until_full_matrix_and_hardware_rows"
    )
    assert payload["provenance"]["hardware_sources"]
    assert payload["acceptance_gates"]["classical_matrix"]
    assert len(payload["hardware_points"]) >= 3


def test_s2_matrix_readiness_blocks_incomplete_no_qpu_progress(tmp_path: Path) -> None:
    protocol = _write_json(
        tmp_path / "protocol.json",
        {
            "protocol_id": "s2",
            "sizes": [4, 6, 8],
            "required_baselines": ["classical_ode", "mps_tensor_network"],
        },
    )
    progress = _write_json(
        tmp_path / "progress.json",
        {
            "sizes": [4],
            "full_campaign_complete": False,
            "all_rows_ok": True,
            "hardware_submission": False,
            "advantage_claim": False,
            "total_executed_rows": 2,
            "total_ok_rows": 2,
        },
    )
    boundary = _write_json(
        tmp_path / "boundary.json",
        {"remaining_blockers": ["run full MPS/TN baseline rows"]},
    )

    readiness = evaluate_s2_matrix_readiness(
        protocol_path=protocol,
        progress_path=progress,
        claim_boundary_path=boundary,
    )

    assert readiness["ready_for_ibm_advantage_run"] is False
    assert readiness["decision"] == "blocked_until_full_matrix_and_hardware_rows"
    assert readiness["missing_sizes"] == [6, 8]
    assert "full protocol size grid is not executed" in readiness["blockers"]
    assert "no preregistered QPU hardware rows are present" in readiness["blockers"]
    assert "run full MPS/TN baseline rows" in readiness["blockers"]


def test_s2_matrix_readiness_accepts_complete_preregistered_matrix(tmp_path: Path) -> None:
    protocol = _write_json(
        tmp_path / "protocol.json",
        {
            "protocol_id": "s2",
            "sizes": [4, 6],
            "required_baselines": ["classical_ode", "mps_tensor_network"],
        },
    )
    progress = _write_json(
        tmp_path / "progress.json",
        {
            "sizes": [4, 6],
            "full_campaign_complete": True,
            "all_rows_ok": True,
            "hardware_submission": True,
            "advantage_claim": True,
            "total_executed_rows": 4,
            "total_ok_rows": 4,
            "max_memory_bytes": 1024,
            "max_hilbert_dim": 64,
        },
    )
    boundary = _write_json(tmp_path / "boundary.json", {"remaining_blockers": []})

    readiness = evaluate_s2_matrix_readiness(
        protocol_path=protocol,
        progress_path=progress,
        claim_boundary_path=boundary,
    )

    assert readiness["ready_for_ibm_advantage_run"] is True
    assert readiness["decision"] == "ready_for_preregistered_ibm_advantage_run"
    assert readiness["blockers"] == []


def test_s2_matrix_readiness_reports_missing_artifacts(tmp_path: Path) -> None:
    readiness = evaluate_s2_matrix_readiness(
        protocol_path=tmp_path / "missing_protocol.json",
        progress_path=tmp_path / "missing_progress.json",
        claim_boundary_path=None,
    )

    assert readiness["available"] is False
    assert readiness["ready_for_ibm_advantage_run"] is False
    assert readiness["decision"] == "blocked_missing_s2_matrix_artifacts"
