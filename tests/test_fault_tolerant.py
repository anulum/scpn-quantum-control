# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Fault Tolerant
"""Tests for qec/fault_tolerant.py."""

import pytest

from scpn_quantum_control.qec.fault_tolerant import FaultTolerantUPDE, LogicalQubit


def test_logical_qubit_data_count():
    lq = LogicalQubit(code_distance=3, phase_angle=1.0)
    assert lq.data_qubits == 3


def test_init_basic():
    ft = FaultTolerantUPDE(n_osc=2, code_distance=3)
    assert ft.n_osc == 2
    assert ft.d == 3
    assert ft.total_qubits == 2 * (3 + 2)


def test_init_rejects_even_distance():
    with pytest.raises(ValueError, match="odd positive"):
        FaultTolerantUPDE(n_osc=2, code_distance=4)


def test_init_rejects_single_osc():
    with pytest.raises(ValueError, match="2"):
        FaultTolerantUPDE(n_osc=1)


def test_physical_qubit_count():
    ft = FaultTolerantUPDE(n_osc=3, code_distance=5)
    assert ft.physical_qubit_count() == 3 * (5 + 4)


def test_build_step_circuit():
    ft = FaultTolerantUPDE(n_osc=2, code_distance=3)
    qc = ft.build_step_circuit(dt=0.1)
    assert qc.num_qubits == ft.total_qubits
    assert qc.size() > 0


def test_step_with_qec():
    ft = FaultTolerantUPDE(n_osc=2, code_distance=3)
    result = ft.step_with_qec(dt=0.1)
    assert "syndromes" in result
    assert len(result["syndromes"]) == 2
    assert result["code_distance"] == 3


def test_step_no_errors_clean_syndromes():
    ft = FaultTolerantUPDE(n_osc=2, code_distance=3)
    result = ft.step_with_qec(dt=0.1)
    assert result["errors_detected"] == 0


# ---------------------------------------------------------------------------
# Qubit layout invariants
# ---------------------------------------------------------------------------


def test_qubit_formula_2d_minus_1():
    """qubits_per_osc = 2d - 1 (d data + d-1 ancilla)."""
    for d in (3, 5, 7):
        ft = FaultTolerantUPDE(n_osc=2, code_distance=d)
        assert ft.qubits_per_osc == 2 * d - 1


def test_total_qubits_scales_linearly():
    """total_qubits = n_osc * (2d - 1)."""
    for n in (2, 3, 4):
        ft = FaultTolerantUPDE(n_osc=n, code_distance=3)
        assert ft.total_qubits == n * 5


def test_data_ancilla_ranges_non_overlapping():
    ft = FaultTolerantUPDE(n_osc=3, code_distance=3)
    for osc in range(3):
        data = set(ft._osc_data_range(osc))
        ancilla = set(ft._osc_ancilla_range(osc))
        assert data.isdisjoint(ancilla)


# ---------------------------------------------------------------------------
# Circuit structure
# ---------------------------------------------------------------------------


def test_circuit_depth_positive():
    ft = FaultTolerantUPDE(n_osc=2, code_distance=3)
    qc = ft.build_step_circuit(dt=0.1)
    assert qc.depth() > 0


def test_circuit_contains_rzz():
    """Coupled oscillators should produce RZZ gates."""
    ft = FaultTolerantUPDE(n_osc=2, code_distance=3)
    qc = ft.build_step_circuit(dt=0.1)
    ops = qc.count_ops()
    assert ops.get("rzz", 0) > 0


def test_circuit_contains_cx():
    """Encoding + syndrome extraction use CNOT gates."""
    ft = FaultTolerantUPDE(n_osc=2, code_distance=3)
    qc = ft.build_step_circuit(dt=0.1)
    ops = qc.count_ops()
    assert ops.get("cx", 0) > 0


# ---------------------------------------------------------------------------
# Custom K and omega
# ---------------------------------------------------------------------------


def test_custom_K_omega():
    import numpy as np

    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([1.0, 2.0])
    ft = FaultTolerantUPDE(n_osc=2, code_distance=3, K=K, omega=omega)
    result = ft.step_with_qec(dt=0.1)
    assert result["n_osc"] == 2


# ---------------------------------------------------------------------------
# Pipeline: Knm → FT-UPDE → syndrome → wired end-to-end
# ---------------------------------------------------------------------------


def test_pipeline_knm_to_ft_syndrome():
    """Full pipeline: build_knm → FaultTolerantUPDE → syndrome extraction."""
    from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    ft = FaultTolerantUPDE(n_osc=3, code_distance=3, K=K, omega=omega)
    result = ft.step_with_qec(dt=0.05)
    assert result["n_osc"] == 3
    assert result["code_distance"] == 3
    assert len(result["syndromes"]) == 3
    for syn in result["syndromes"]:
        assert len(syn) == 2  # d-1 = 2 ancillae per oscillator
