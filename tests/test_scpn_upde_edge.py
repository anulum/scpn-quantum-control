# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for SPO UPDE edge
"""Tests for the bounded ``knm.scpn-upde`` SPO edge."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from scpn_quantum_control.bridge import (
    SCPN_UPDE_EDGE_SCHEMA,
    SCPN_UPDE_SCOPE_ENVELOPE,
    build_paper27_scpn_upde_edge,
    build_scpn_upde_edge,
    validate_scpn_upde_edge_payload,
)


def test_paper27_edge_emits_16_oscillator_computational_agreement_payload() -> None:
    edge = build_paper27_scpn_upde_edge(time=0.1, trotter_steps=1, trotter_order=1)
    payload = edge.to_payload()

    assert payload["schema"] == SCPN_UPDE_EDGE_SCHEMA
    assert payload["scope_envelope"] == SCPN_UPDE_SCOPE_ENVELOPE
    assert payload["producer"] == "scpn-quantum-control"
    assert payload["consumer"] == "scpn-phase-orchestrator"
    assert payload["n_oscillators"] == 16
    assert len(payload["K_nm"]) == 16
    assert len(payload["K_nm"][0]) == 16
    assert len(payload["omega"]) == 16
    assert payload["permissions"] == {
        "qpu_execution_permitted": False,
        "actuation_permitted": False,
    }
    assert payload["trotter"] == {"time": 0.1, "steps": 1, "order": 1, "dt": 0.1}
    assert payload["compiler"]["num_qubits"] == 16
    assert payload["compiler"]["depth"] >= 1
    assert payload["compiler"]["operation_counts"]
    assert len(payload["edge_sha256"]) == 64
    validate_scpn_upde_edge_payload(payload)


def test_edge_payload_digest_changes_when_knm_changes() -> None:
    edge = build_paper27_scpn_upde_edge()
    payload = edge.to_payload()
    tampered = copy.deepcopy(payload)
    tampered["K_nm"][0][1] += 0.001

    with pytest.raises(ValueError, match="K_nm_sha256"):
        validate_scpn_upde_edge_payload(tampered)


def test_edge_payload_rejects_broader_scope_or_permissions() -> None:
    payload = build_paper27_scpn_upde_edge().to_payload()
    broader = copy.deepcopy(payload)
    broader["scope_envelope"] = "physical-validation"

    with pytest.raises(ValueError, match="scope_envelope"):
        validate_scpn_upde_edge_payload(broader)

    executable = copy.deepcopy(payload)
    executable["permissions"]["qpu_execution_permitted"] = True

    with pytest.raises(ValueError, match="qpu_execution_permitted"):
        validate_scpn_upde_edge_payload(executable)


def test_custom_edge_validates_shapes_before_compile_surface() -> None:
    K_nm = np.array([[0.0, 0.25], [0.25, 0.0]], dtype=np.float64)
    omega = np.array([1.0, -0.5], dtype=np.float64)

    payload = build_scpn_upde_edge(
        K_nm,
        omega,
        time=0.2,
        trotter_steps=2,
        trotter_order=1,
    ).to_payload()

    assert payload["n_oscillators"] == 2
    assert payload["trotter"]["dt"] == pytest.approx(0.1)
    validate_scpn_upde_edge_payload(payload)


def test_edge_rejects_asymmetric_knm() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        build_scpn_upde_edge(
            np.array([[0.0, 0.25], [0.1, 0.0]], dtype=np.float64),
            np.array([1.0, -0.5], dtype=np.float64),
        )
