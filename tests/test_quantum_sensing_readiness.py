# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S11 quantum sensing readiness tests
"""Tests for the S11 sync-order quantum-sensing readiness gate."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.sensing import (
    QUANTUM_SENSING_SCHEMA,
    QuantumSensingReadinessConfig,
    metrological_gain_vs_k,
    optimal_sensing_k,
    quantum_sensing_markdown,
    quantum_sensing_payload,
)


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega = np.array([-0.20, 0.0, 0.25], dtype=np.float64)
    topology = np.ones((3, 3), dtype=np.float64) - np.eye(3, dtype=np.float64)
    k_grid = np.array([0.4, 0.8, 1.2], dtype=np.float64)
    return omega, topology, k_grid


def test_metrological_gain_scan_uses_qfi_and_classical_proxy_without_promotion() -> None:
    omega, topology, k_grid = _inputs()
    result = metrological_gain_vs_k(
        omega,
        topology,
        k_grid,
        config=QuantumSensingReadinessConfig(max_dense_gib=0.01),
    )

    assert result.schema == "s11_quantum_sensing_gain_scan_v1"
    assert any(result.peak_k == pytest.approx(float(k_value)) for k_value in k_grid)
    assert result.peak_qfi > 0.0
    assert result.best_gain_ratio > 0.0
    assert result.hardware_submission_allowed is False
    assert result.sensing_advantage_claim_allowed is False
    assert "classical Fisher" in result.falsifier


def test_optimal_sensing_k_selects_grid_row_with_largest_gain() -> None:
    omega, topology, k_grid = _inputs()
    result = optimal_sensing_k(
        omega,
        topology,
        k_grid,
        config=QuantumSensingReadinessConfig(max_dense_gib=0.01),
    )

    assert any(result.k_value == pytest.approx(float(k_value)) for k_value in k_grid)
    assert result.qfi_value > 0.0
    assert result.gain_ratio > 0.0
    assert result.claim_boundary == "readiness estimate only; not hardware evidence"


def test_quantum_sensing_payload_keeps_hardware_and_advantage_claims_blocked() -> None:
    payload = quantum_sensing_payload()

    assert payload["schema"] == QUANTUM_SENSING_SCHEMA
    assert payload["hardware_submission_allowed"] is False
    assert payload["sensing_advantage_claim_allowed"] is False
    assert payload["gain_scan"]["optimal_row"]["k_value"] in payload["gain_scan"]["k_values"]
    assert "pre-registered perturbation benchmark" in payload["falsifier"]


def test_quantum_sensing_markdown_records_gate_and_falsifier() -> None:
    markdown = quantum_sensing_markdown(quantum_sensing_payload())

    assert "scpn-bench s11-quantum-sensing-readiness" in markdown
    assert "sensing advantage claim allowed: `False`" in markdown
    assert "classical Fisher information" in markdown


def test_quantum_sensing_inputs_fail_closed() -> None:
    omega, topology, k_grid = _inputs()
    with pytest.raises(ValueError, match="k_grid"):
        metrological_gain_vs_k(omega, topology, np.array([0.8]))
    with pytest.raises(ValueError, match="strictly increasing"):
        metrological_gain_vs_k(omega, topology, np.array([0.8, 0.4, 1.2]))
    with pytest.raises(ValueError, match="square"):
        metrological_gain_vs_k(omega, topology[:2], k_grid)
    with pytest.raises(ValueError, match="omega length"):
        metrological_gain_vs_k(omega[:2], topology, k_grid)
