"""Validation guard tests for constructors and public API boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.control.qaoa_mpc import QAOA_MPC
from scpn_quantum_control.crypto.entanglement_qkd import bell_inequality_test
from scpn_quantum_control.crypto.percolation import best_entanglement_path
from scpn_quantum_control.hardware.classical import classical_kuramoto_reference
from scpn_quantum_control.phase.phase_vqe import PhaseVQE
from scpn_quantum_control.qsnn.qlif import QuantumLIFNeuron
from scpn_quantum_control.qsnn.qstdp import QuantumSTDP
from scpn_quantum_control.qsnn.qsynapse import QuantumSynapse


def test_qlif_rejects_equal_thresholds():
    with pytest.raises(ValueError, match="v_threshold.*must exceed.*v_rest"):
        QuantumLIFNeuron(v_rest=1.0, v_threshold=1.0)


def test_qlif_rejects_zero_tau_mem():
    with pytest.raises(ValueError, match="tau_mem must be positive"):
        QuantumLIFNeuron(tau_mem=0.0)


def test_qsynapse_rejects_equal_bounds():
    with pytest.raises(ValueError, match="w_max.*must exceed.*w_min"):
        QuantumSynapse(weight=0.5, w_min=1.0, w_max=1.0)


def test_qstdp_rejects_zero_sin_shift():
    with pytest.raises(ValueError, match="sin\\(shift\\)~0"):
        QuantumSTDP(shift=0.0)


def test_vqe_solve_returns_exact_energy():
    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)
    sol = vqe.solve(maxiter=50)
    assert "exact_energy" in sol
    assert isinstance(sol["exact_energy"], float)


def test_vqe_solve_returns_energy_gap():
    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)
    sol = vqe.solve(maxiter=50)
    assert "energy_gap" in sol
    assert sol["energy_gap"] >= 0.0
    assert "relative_error_pct" in sol
    assert "n_params" in sol
    assert sol["n_params"] == vqe.n_params


# --- QAOA-MPC validation ---


def test_qaoa_mpc_rejects_zero_horizon():
    with pytest.raises(ValueError, match="horizon must be positive"):
        QAOA_MPC(B_matrix=np.eye(2), target_state=np.ones(2), horizon=0)


def test_qaoa_mpc_rejects_negative_horizon():
    with pytest.raises(ValueError, match="horizon must be positive"):
        QAOA_MPC(B_matrix=np.eye(2), target_state=np.ones(2), horizon=-3)


# --- Entanglement QKD validation ---


def test_bell_rejects_out_of_range_qubits():
    from qiskit.quantum_info import Statevector

    sv = Statevector.from_label("00")
    with pytest.raises(ValueError, match="out of range"):
        bell_inequality_test(sv, qubit_a=5, qubit_b=0, n_total=2)


# --- Percolation validation ---


def test_percolation_rejects_out_of_range_source():
    K = build_knm_paper27(L=4)
    with pytest.raises(ValueError, match="out of range"):
        best_entanglement_path(K, source=10, target=0)


def test_percolation_rejects_negative_target():
    K = build_knm_paper27(L=4)
    with pytest.raises(ValueError, match="out of range"):
        best_entanglement_path(K, source=0, target=-1)


# --- Classical reference validation ---


def test_classical_kuramoto_rejects_zero_dt():
    with pytest.raises(ValueError, match="dt must be positive"):
        classical_kuramoto_reference(n_osc=4, t_max=1.0, dt=0.0)


def test_classical_kuramoto_rejects_negative_tmax():
    with pytest.raises(ValueError, match="t_max must be non-negative"):
        classical_kuramoto_reference(n_osc=4, t_max=-1.0, dt=0.1)
