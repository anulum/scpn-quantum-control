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
    assert payload["provenance"]["hardware_sources"]
    assert payload["acceptance_gates"]["classical_matrix"]
    assert len(payload["hardware_points"]) >= 3
