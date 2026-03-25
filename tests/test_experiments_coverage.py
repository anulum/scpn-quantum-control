"""Coverage tests for hardware.experiments — statevector-only functions."""

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27, OMEGA_N_16
from scpn_quantum_control.hardware.experiments import (
    _build_evo_base,
    _build_xyz_circuits,
    _correlator_from_counts,
    _expectation_per_qubit,
    _qaoa_cost_from_counts,
    _R_from_xyz,
    _run_vqe,
)


class TestBuildEvoBase:
    def test_returns_circuit(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        qc = _build_evo_base(2, K, omega, t=0.1, trotter_reps=1)
        assert qc.num_qubits == 2

    def test_trotter_order_2(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        qc = _build_evo_base(2, K, omega, t=0.1, trotter_reps=1, trotter_order=2)
        assert qc.num_qubits == 2

    def test_4_qubit(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        qc = _build_evo_base(4, K, omega, t=0.2, trotter_reps=2)
        assert qc.num_qubits == 4


class TestBuildXYZCircuits:
    def test_returns_three_circuits(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        base = _build_evo_base(2, K, omega, t=0.1, trotter_reps=1)
        z_qc, x_qc, y_qc = _build_xyz_circuits(base, 2)
        assert z_qc.num_qubits == 2
        assert x_qc.num_qubits == 2
        assert y_qc.num_qubits == 2


class TestExpectationPerQubit:
    def test_all_zeros(self):
        counts = {"00": 1000}
        exp = _expectation_per_qubit(counts, 2)
        assert exp.shape[0] == 2

    def test_all_ones(self):
        counts = {"11": 1000}
        exp = _expectation_per_qubit(counts, 2)
        assert exp.shape[0] == 2

    def test_returns_array(self):
        counts = {"00": 500, "11": 500}
        exp = _expectation_per_qubit(counts, 2)
        assert isinstance(exp, np.ndarray)


class TestRFromXYZ:
    def test_coherent_state(self):
        z = {"00": 1000}
        x = {"00": 1000}
        y = {"00": 1000}
        R = _R_from_xyz(z, x, y, 2)
        assert R > 0.5

    def test_returns_float(self):
        z = {"00": 500, "11": 500}
        x = {"00": 500, "11": 500}
        y = {"00": 500, "11": 500}
        R = _R_from_xyz(z, x, y, 2)
        assert isinstance(R, float)
        assert 0 <= R <= 1.0 + 1e-6


class TestQAOACost:
    def test_returns_float(self):
        from qiskit.quantum_info import SparsePauliOp

        ham = SparsePauliOp.from_list([("ZZ", 1.0), ("IZ", 0.5), ("ZI", -0.3)])
        counts = {"00": 400, "01": 200, "10": 200, "11": 200}
        cost = _qaoa_cost_from_counts(counts, ham, 2)
        assert isinstance(cost, float)


class TestCorrelatorFromCounts:
    def test_perfect_correlation(self):
        counts = {"00": 500, "11": 500}
        c = _correlator_from_counts(counts, 0, 1)
        assert abs(c - 1.0) < 0.01

    def test_anti_correlation(self):
        counts = {"01": 500, "10": 500}
        c = _correlator_from_counts(counts, 0, 1)
        assert abs(c - (-1.0)) < 0.01


class TestRunVQE:
    def test_returns_result(self):
        result = _run_vqe(2, maxiter=30)
        assert "ground_energy" in result
        assert "exact_energy" in result
        assert "relative_error_pct" in result
