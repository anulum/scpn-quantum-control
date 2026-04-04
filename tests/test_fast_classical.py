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

    def test_unitarity_preserves_norm(self):
        """Evolution under Hermitian H must preserve state norm at every step."""
        n = 3
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        res = fast_sparse_evolution(K, omega, t_total=5.0, n_steps=50)
        for psi in res["states"]:
            assert abs(np.vdot(psi, psi).real - 1.0) < 1e-10

    def test_custom_initial_state(self):
        """Custom initial state is propagated, not overwritten."""
        n = 2
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        psi0 = np.array([0, 1, 0, 0], dtype=complex)
        res = fast_sparse_evolution(K, omega, t_total=0.0, n_steps=1, initial_state=psi0)
        np.testing.assert_allclose(res["states"][0], psi0)

    def test_superposition_initial_state_norm(self):
        """Superposition initial state stays normalised throughout evolution."""
        n = 2
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        psi0 = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
        res = fast_sparse_evolution(K, omega, t_total=2.0, n_steps=20, initial_state=psi0)
        for psi in res["states"]:
            assert abs(np.linalg.norm(psi) - 1.0) < 1e-10

    def test_zero_time_identity(self):
        """t=0 evolution returns the initial state unchanged."""
        n = 3
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        psi0 = _build_initial_state(n, omega)
        res = fast_sparse_evolution(K, omega, t_total=0.0, n_steps=1, initial_state=psi0)
        np.testing.assert_allclose(res["final_state"], psi0, atol=1e-12)

    def test_time_reversal_symmetry(self):
        """Evolving forward then backward recovers the initial state."""
        n = 3
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        psi0 = _build_initial_state(n, omega)
        res_fwd = fast_sparse_evolution(K, omega, t_total=2.0, n_steps=20, initial_state=psi0)
        psi_mid = res_fwd["final_state"]
        res_bwd = fast_sparse_evolution(K, omega, t_total=-2.0, n_steps=20, initial_state=psi_mid)
        np.testing.assert_allclose(np.abs(res_bwd["final_state"]), np.abs(psi0), atol=1e-8)

    def test_xxz_delta_zero_matches_xy(self):
        """delta=0 and default (no delta) produce identical evolution."""
        n = 3
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        res_default = fast_sparse_evolution(K, omega, t_total=1.0, n_steps=10)
        res_d0 = fast_sparse_evolution(K, omega, t_total=1.0, n_steps=10, delta=0.0)
        np.testing.assert_allclose(res_default["final_state"], res_d0["final_state"], atol=1e-12)

    def test_xxz_nonzero_delta_differs(self):
        """delta != 0 adds ZZ coupling, so results diverge from XY."""
        n = 2
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        res_xy = fast_sparse_evolution(K, omega, t_total=1.0, n_steps=10, delta=0.0)
        res_xxz = fast_sparse_evolution(K, omega, t_total=1.0, n_steps=10, delta=1.0)
        assert not np.allclose(res_xy["final_state"], res_xxz["final_state"])

    def test_scaling_n8(self):
        """N=8 (256-dim Hilbert space) completes and preserves norm."""
        n = 8
        rng = np.random.default_rng(42)
        K = rng.random((n, n)) * 0.2
        K = (K + K.T) / 2
        np.fill_diagonal(K, 0)
        omega = rng.uniform(0.5, 1.5, n)
        res = fast_sparse_evolution(K, omega, t_total=1.0, n_steps=5)
        assert res["final_state"].shape == (256,)
        assert abs(np.linalg.norm(res["final_state"]) - 1.0) < 1e-10


class TestFastClassicalRustParity:
    """Verify Rust-accelerated Hamiltonian matches Qiskit fallback."""

    def test_rust_hamiltonian_matches_qiskit(self):
        try:
            import scpn_quantum_engine as eng
        except ImportError:
            import pytest

            pytest.skip("Rust engine not available")

        from scipy.sparse import csc_matrix

        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_xxz_hamiltonian

        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        n = 4

        rows, cols, vals = eng.build_sparse_xy_hamiltonian(K.ravel(), omega, n)
        H_rust = csc_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))),
            shape=(2**n, 2**n),
        ).toarray()

        H_qiskit = knm_to_xxz_hamiltonian(K, omega, delta=0.0).to_matrix()
        np.testing.assert_allclose(H_rust, H_qiskit.real, atol=1e-12)
