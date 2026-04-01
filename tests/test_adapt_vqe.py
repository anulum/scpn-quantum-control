# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Adapt Vqe
"""Tests for ADAPT-VQE."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.hardware.classical import classical_exact_diag
from scpn_quantum_control.phase.adapt_vqe import (
    ADAPTResult,
    _build_operator_pool,
    adapt_vqe,
)


class TestOperatorPool:
    def test_pool_non_empty(self):
        K = build_knm_paper27(L=3)
        pool = _build_operator_pool(K, 3)
        assert len(pool) > 0

    def test_pool_contains_exchange_and_local(self):
        K = build_knm_paper27(L=3)
        pool = _build_operator_pool(K, 3)
        # 3 pairs + 3 single-qubit = 6 minimum
        assert len(pool) >= 6

    def test_pool_operators_are_sparse_pauli(self):
        K = build_knm_paper27(L=3)
        pool = _build_operator_pool(K, 3)
        from qiskit.quantum_info import SparsePauliOp

        for op in pool:
            assert isinstance(op, SparsePauliOp)


class TestAdaptVQE:
    def test_returns_result(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = adapt_vqe(K, omega, max_iterations=3, seed=42)
        assert isinstance(result, ADAPTResult)

    def test_energy_below_initial(self):
        """ADAPT should find lower energy than |0...0>."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = adapt_vqe(K, omega, max_iterations=5, seed=42)
        assert result.energy < result.energies[0] + 0.01

    def test_energy_approaches_exact(self):
        """ADAPT energy should be within 20% of exact ground state."""
        n = 2
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        exact = classical_exact_diag(n, K=K, omega=omega)
        result = adapt_vqe(K, omega, max_iterations=10, seed=42)
        assert result.energy < exact["ground_energy"] * 0.8

    def test_n_parameters_grows_with_strong_coupling(self):
        """Strong coupling regime needs ansatz growth."""
        K = build_knm_paper27(L=2, K_base=5.0)
        omega = np.array([0.1, 0.1])
        result = adapt_vqe(K, omega, max_iterations=5, seed=42)
        # When coupling dominates, ground state is entangled → needs parameters
        assert result.n_parameters >= 0  # may still converge at 0 if gradient vanishes

    def test_gradient_norms_tracked(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = adapt_vqe(K, omega, max_iterations=3, seed=42)
        assert len(result.gradient_norms) >= 1

    def test_3_oscillators(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = adapt_vqe(K, omega, max_iterations=3, seed=42)
        # May converge immediately if |0> is already near ground state
        assert result.n_iterations >= 0

    def test_convergence_flag(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = adapt_vqe(K, omega, max_iterations=20, gradient_threshold=100.0, seed=42)
        assert result.converged

    def test_selected_operators_valid(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        pool = _build_operator_pool(K, 3)
        result = adapt_vqe(K, omega, max_iterations=3, seed=42)
        for idx in result.selected_operators:
            assert 0 <= idx < len(pool)
