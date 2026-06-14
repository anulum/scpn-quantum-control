# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for adaptive layered VQE
"""Tests for the adaptive layered VQE: pool, ansatz action, ground-state convergence."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_dense_matrix,
)
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.hardware.classical import classical_exact_diag
from scpn_quantum_control.phase.adapt_vqe import (
    ADAPTResult,
    _ansatz_state,
    _build_operator_pool,
    _plus_reference,
    _pool_generators_dense,
    adapt_vqe,
)


def _exact_ground_energy(n: int, K: np.ndarray, omega: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(knm_to_dense_matrix(K, omega))[0])


class TestOperatorPool:
    def test_pool_non_empty(self):
        K = build_knm_paper27(L=3)
        assert len(_build_operator_pool(K, 3)) > 0

    def test_pool_contains_exchange_and_local(self):
        K = build_knm_paper27(L=3)
        # 3 pairs + 3 single-qubit = 6 minimum
        assert len(_build_operator_pool(K, 3)) >= 6

    def test_pool_operators_are_sparse_pauli(self):
        K = build_knm_paper27(L=3)
        for op in _build_operator_pool(K, 3):
            assert isinstance(op, SparsePauliOp)

    def test_dense_generators_are_hermitian(self):
        K = build_knm_paper27(L=3)
        for generator in _pool_generators_dense(K, 3):
            assert np.allclose(generator, generator.conj().T, atol=1e-12)


class TestAnsatzAction:
    def test_plus_reference_is_normalised_uniform(self):
        ref = _plus_reference(3)
        assert ref.shape == (8,)
        assert np.allclose(np.abs(ref), 1.0 / np.sqrt(8))
        assert np.isclose(np.vdot(ref, ref), 1.0)

    def test_ansatz_zero_angles_is_identity(self):
        K = build_knm_paper27(L=2)
        spectra = [np.linalg.eigh(g) for g in _pool_generators_dense(K, 2)]
        ref = _plus_reference(2)
        out = _ansatz_state(ref, spectra, np.zeros(len(spectra)))
        assert np.allclose(out, ref)

    def test_ansatz_preserves_norm(self):
        K = build_knm_paper27(L=2)
        generators = _pool_generators_dense(K, 2)
        spectra = [np.linalg.eigh(g) for g in generators]
        ref = _plus_reference(2)
        angles = np.linspace(-1.0, 1.0, len(generators))
        out = _ansatz_state(ref, spectra, angles)
        assert np.isclose(np.vdot(out, out).real, 1.0, atol=1e-12)


class TestGroundStateConvergence:
    def test_returns_result(self):
        K = build_knm_paper27(L=2)
        result = adapt_vqe(K, OMEGA_N_16[:2], seed=42)
        assert isinstance(result, ADAPTResult)

    def test_reaches_exact_ground_state_2q(self):
        n = 2
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        exact = _exact_ground_energy(n, K, omega)
        result = adapt_vqe(K, omega, seed=42)
        assert abs(result.energy - exact) < 1e-6

    def test_reaches_exact_ground_state_3q(self):
        n = 3
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        exact = _exact_ground_energy(n, K, omega)
        result = adapt_vqe(K, omega, seed=42)
        assert abs(result.energy - exact) < 1e-6

    def test_matches_classical_exact_diag(self):
        n = 3
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        exact = classical_exact_diag(n, K=K, omega=omega)["ground_energy"]
        result = adapt_vqe(K, omega, seed=42)
        assert abs(result.energy - exact) < 1e-6

    def test_energy_below_reference(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = adapt_vqe(K, omega, seed=42)
        # The reference |+><+| energy is energies[0]; the optimum must improve on it.
        assert result.energy < result.energies[0] - 1e-6

    def test_builds_a_nonempty_ansatz(self):
        # Regression: the gradient-selection ADAPT terminated at 0 operators from the
        # |0...0> eigenstate and falsely reported convergence. The layered scheme must
        # build a real ansatz.
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = adapt_vqe(K, omega, seed=42)
        assert result.n_iterations >= 1
        assert result.n_parameters > 0
        assert len(result.selected_operators) > 0

    def test_reproducible_for_fixed_seed(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        a = adapt_vqe(K, omega, seed=7)
        b = adapt_vqe(K, omega, seed=7)
        assert a.energy == b.energy
        assert a.n_parameters == b.n_parameters

    def test_convergence_flag_set(self):
        K = build_knm_paper27(L=2)
        result = adapt_vqe(K, OMEGA_N_16[:2], seed=42)
        assert result.converged

    def test_gradient_norms_tracked_per_layer(self):
        K = build_knm_paper27(L=2)
        result = adapt_vqe(K, OMEGA_N_16[:2], seed=42)
        assert len(result.gradient_norms) == result.n_iterations
        assert all(g >= 0.0 for g in result.gradient_norms)

    def test_energies_history_starts_at_reference(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = adapt_vqe(K, omega, seed=42)
        ref_energy = float(
            (_plus_reference(2).conj() @ knm_to_dense_matrix(K, omega) @ _plus_reference(2)).real
        )
        assert result.energies[0] == pytest.approx(ref_energy, abs=1e-10)
        assert len(result.energies) == result.n_iterations + 1

    def test_selected_operators_index_into_pool(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        pool = _build_operator_pool(K, 3)
        result = adapt_vqe(K, omega, seed=42)
        for idx in result.selected_operators:
            assert 0 <= idx < len(pool)


class TestContracts:
    def test_rejects_dense_budget_before_dense_allocation(self, monkeypatch):
        adapt_module = importlib.import_module("scpn_quantum_control.phase.adapt_vqe")
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        def fail_if_dense_requested(*args, **kwargs):  # noqa: ARG001
            raise AssertionError("dense allocation happened before the budget gate")

        monkeypatch.setattr(adapt_module, "knm_to_dense_matrix", fail_if_dense_requested)

        with pytest.raises(DenseAllocationError, match="ADAPT-VQE statevector"):
            adapt_vqe(K, omega, max_iterations=1, seed=42, max_dense_gib=1e-12)

    def test_rejects_oversized_system(self):
        K = build_knm_paper27(L=11)
        omega = OMEGA_N_16[:11]
        with pytest.raises(ValueError, match="too large"):
            adapt_vqe(K, omega, seed=42)
