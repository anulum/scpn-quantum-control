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
from numpy.typing import NDArray

from scpn_quantum_control.analysis.sensing import (
    CRITICALITY_TAIL_SCHEMA,
    QUANTUM_SENSING_SCHEMA,
    CriticalitySensingTail,
    QuantumSensingReadinessConfig,
    metrological_gain_vs_k,
    optimal_sensing_k,
    qfi_criticality_sensing_tail,
    quantum_sensing_markdown,
    quantum_sensing_payload,
)


def _inputs() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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


def test_qfi_criticality_sensing_tail_selects_pair_probe_with_precision_bound() -> None:
    """The QFI peak is converted into a pair-level Cramer-Rao probe target."""
    omega, topology, k_grid = _inputs()

    tail = qfi_criticality_sensing_tail(
        omega,
        topology,
        k_grid,
        measurements=5000,
        config=QuantumSensingReadinessConfig(max_dense_gib=0.01),
    )

    assert isinstance(tail, CriticalitySensingTail)
    assert tail.schema == CRITICALITY_TAIL_SCHEMA
    assert any(tail.operating_k == pytest.approx(float(k_value)) for k_value in k_grid)
    assert tail.selected_pair in ((0, 1), (0, 2), (1, 2))
    assert tail.qfi_value > 0.0
    assert tail.qfi_trace >= tail.qfi_value
    assert tail.cramer_rao_variance_bound == pytest.approx(1.0 / (5000 * tail.qfi_value))
    assert tail.cramer_rao_std_bound == pytest.approx(np.sqrt(tail.cramer_rao_variance_bound))
    assert tail.geometric_crosscheck_agrees is True
    assert tail.geometric_crosscheck_max_rel_difference < 0.05
    assert tail.hardware_submission_allowed is False
    assert tail.sensing_advantage_claim_allowed is False


def test_qfi_criticality_sensing_tail_can_skip_the_geometric_crosscheck() -> None:
    """Offline planning can omit the slower QGT route without hiding that fact."""
    omega, topology, k_grid = _inputs()

    tail = qfi_criticality_sensing_tail(
        omega,
        topology,
        k_grid,
        run_geometric_crosscheck=False,
        config=QuantumSensingReadinessConfig(max_dense_gib=0.01),
    )

    assert tail.geometric_crosscheck_agrees is False
    assert np.isinf(tail.geometric_crosscheck_max_rel_difference)


def test_quantum_sensing_payload_keeps_hardware_and_advantage_claims_blocked() -> None:
    payload = quantum_sensing_payload()

    assert payload["schema"] == QUANTUM_SENSING_SCHEMA
    assert payload["hardware_submission_allowed"] is False
    assert payload["sensing_advantage_claim_allowed"] is False
    assert payload["gain_scan"]["optimal_row"]["k_value"] in payload["gain_scan"]["k_values"]
    assert payload["criticality_tail"]["schema"] == CRITICALITY_TAIL_SCHEMA
    assert payload["criticality_tail"]["hardware_submission_allowed"] is False
    assert payload["criticality_tail"]["sensing_advantage_claim_allowed"] is False
    assert "pre-registered perturbation benchmark" in payload["falsifier"]


def test_quantum_sensing_markdown_records_gate_and_falsifier() -> None:
    markdown = quantum_sensing_markdown(quantum_sensing_payload())

    assert "scpn-bench s11-quantum-sensing-readiness" in markdown
    assert "sensing advantage claim allowed: `False`" in markdown
    assert "## QFI-Criticality Tail" in markdown
    assert "Cramer-Rao variance bound" in markdown
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


def test_qfi_criticality_sensing_tail_inputs_fail_closed() -> None:
    """Criticality-tail validation rejects ambiguous or unphysical plans."""
    omega, topology, k_grid = _inputs()

    with pytest.raises(ValueError, match="measurements"):
        qfi_criticality_sensing_tail(omega, topology, k_grid, measurements=0)
    with pytest.raises(ValueError, match="geometric_epsilon"):
        qfi_criticality_sensing_tail(omega, topology, k_grid, geometric_epsilon=0.0)
    with pytest.raises(ValueError, match="nonzero coupling edge"):
        qfi_criticality_sensing_tail(omega, np.zeros_like(topology), k_grid)


def test_analysis_subpackage_exports_criticality_sensing_tail() -> None:
    """The public analysis namespace exposes the QWC-4.4 sensing API."""
    from scpn_quantum_control import analysis

    assert analysis.CRITICALITY_TAIL_SCHEMA == CRITICALITY_TAIL_SCHEMA
    assert analysis.CriticalitySensingTail is CriticalitySensingTail
    assert analysis.qfi_criticality_sensing_tail is qfi_criticality_sensing_tail
    assert "qfi_criticality_sensing_tail" in analysis.__all__
