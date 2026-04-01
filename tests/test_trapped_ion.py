# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Trapped Ion
"""Tests for hardware/trapped_ion.py."""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from scpn_quantum_control.hardware.trapped_ion import (
    MS_ERROR,
    MS_GATE_TIME_US,
    SQ_GATE_TIME_US,
    T1_US,
    T2_US,
    transpile_for_trapped_ion,
    trapped_ion_noise_model,
)


def test_noise_model_returns_model():
    model = trapped_ion_noise_model()
    assert model is not None
    assert len(model.noise_instructions) > 0


def test_noise_model_custom_params():
    model = trapped_ion_noise_model(ms_error=0.01, t1_us=50_000.0)
    assert model is not None


def test_constants_physical_ranges():
    assert 0 < MS_ERROR < 1
    assert T1_US > T2_US > 0
    assert MS_GATE_TIME_US > SQ_GATE_TIME_US > 0


def test_transpile_single_qubit():
    qc = QuantumCircuit(1)
    qc.h(0)
    result = transpile_for_trapped_ion(qc)
    assert result.num_qubits == 1


def test_transpile_removes_swaps():
    qc = QuantumCircuit(4)
    qc.cx(0, 3)
    qc.cx(1, 3)
    result = transpile_for_trapped_ion(qc)
    ops = result.count_ops()
    assert "swap" not in ops


def test_transpile_preserves_unitarity():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    original = Operator(qc)
    transpiled = transpile_for_trapped_ion(qc)
    result = Operator(transpiled)
    assert original.equiv(result)


def test_transpile_all_to_all_connectivity():
    qc = QuantumCircuit(3)
    qc.cx(0, 2)
    qc.cx(1, 0)
    result = transpile_for_trapped_ion(qc)
    ops = result.count_ops()
    assert "swap" not in ops


def test_transpile_empty_circuit():
    qc = QuantumCircuit(2)
    result = transpile_for_trapped_ion(qc)
    assert result.num_qubits == 2


def test_transpile_ghz_circuit():
    """GHZ state circuit should transpile without error."""
    qc = QuantumCircuit(4)
    qc.h(0)
    for i in range(1, 4):
        qc.cx(0, i)
    result = transpile_for_trapped_ion(qc)
    assert result.num_qubits == 4
    assert result.depth() > 0


def test_transpile_preserves_qubit_count():
    """Transpiled circuit should have same number of qubits."""
    for n in (2, 3, 4):
        qc = QuantumCircuit(n)
        qc.h(0)
        qc.cx(0, n - 1)
        result = transpile_for_trapped_ion(qc)
        assert result.num_qubits == n


def test_noise_model_has_error_instructions():
    model = trapped_ion_noise_model()
    assert len(model.noise_instructions) > 0


def test_pipeline_knm_to_trapped_ion():
    """Pipeline: build Kuramoto circuit → transpile to trapped-ion basis."""
    from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
    from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver

    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    solver = QuantumKuramotoSolver(3, K, omega)
    qc = solver.evolve(time=0.1, trotter_steps=1)
    result = transpile_for_trapped_ion(qc)
    assert result.num_qubits == 3
    assert result.depth() > 0
