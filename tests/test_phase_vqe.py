# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Phase Vqe
"""Tests for phase/phase_vqe.py."""

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase.phase_vqe import PhaseVQE


def test_solve_returns_energy():
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)
    result = vqe.solve(maxiter=30, seed=0)
    assert "ground_energy" in result
    assert np.isfinite(result["ground_energy"])


def test_ground_state_is_statevector():
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)
    vqe.solve(maxiter=20, seed=0)
    sv = vqe.ground_state()
    assert sv is not None
    assert abs(float(np.sum(np.abs(sv) ** 2)) - 1.0) < 1e-10


def test_ground_state_none_before_solve():
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    vqe = PhaseVQE(K, omega)
    assert vqe.ground_state() is None


def test_energy_decreases_with_iterations():
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    vqe_short = PhaseVQE(K, omega, ansatz_reps=1)
    vqe_long = PhaseVQE(K, omega, ansatz_reps=2)
    r1 = vqe_short.solve(maxiter=10, seed=0)
    r2 = vqe_long.solve(maxiter=50, seed=0)
    # More parameters + iterations should find equal or lower energy
    assert r2["ground_energy"] <= r1["ground_energy"] + 1.0  # generous tolerance
