# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Tests for high-performance sparse classical evolution."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.hardware.classical import (
    _build_initial_state,
    _state_order_param,
    classical_exact_evolution,
)
from scpn_quantum_control.hardware.fast_classical import fast_sparse_evolution


class TestFastSparseEvolution:
    def test_fast_sparse_matches_exact_evolution(self):
        """Verify that the high-performance sparse engine matches Exact Diagonalization."""
        n = 3
        dt = 0.5
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]

        # Old classical implementation (eigh based)
        exact_res = classical_exact_evolution(n, dt, dt, K, omega)
        exact_r = exact_res["R"][-1]

        # New fast sparse implementation
        psi0 = _build_initial_state(n, omega)
        fast_res = fast_sparse_evolution(K, omega, t_total=dt, n_steps=1, initial_state=psi0)
        fast_state = fast_res["final_state"]
        fast_r = _state_order_param(fast_state, n)

        # Compare Order Parameter R
        assert abs(fast_r - exact_r) < 1e-10

    def test_n_steps_evolution(self):
        """Verify multiple time steps are stored correctly."""
        n = 2
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        res = fast_sparse_evolution(K, omega, t_total=1.0, n_steps=10)

        assert len(res["times"]) == 11
        assert len(res["states"]) == 11
        assert res["times"][0] == 0.0
        assert res["times"][-1] == 1.0

    def test_n_qubits(self):
        n = 2
        K = np.ones((n, n))
        omega = np.ones(n)
        res = fast_sparse_evolution(K, omega, t_total=1.0, n_steps=1)
        assert res["n_qubits"] == 2
