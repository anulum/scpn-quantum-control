"""Validation guard tests for QSNN constructors and PhaseVQE enriched return."""

from __future__ import annotations

import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
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
