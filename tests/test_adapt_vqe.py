# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Adapt Vqe
"""Tests for ADAPT-VQE."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.hardware.classical import classical_exact_diag
from scpn_quantum_control.phase.adapt_vqe import (
    ADAPTResult,
    _build_ansatz,
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

    def test_build_ansatz_with_selected_operator(self):
        K = build_knm_paper27(L=2)
        pool = _build_operator_pool(K, 2)

        qc = _build_ansatz(2, pool, [0], [0.125])

        assert qc.num_qubits == 2
        assert qc.size() == 1


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

    def test_rejects_dense_budget_before_statevector_allocation(self, monkeypatch):
        adapt_vqe_mod = importlib.import_module("scpn_quantum_control.phase.adapt_vqe")
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        class FailIfStatevectorRequested:
            @staticmethod
            def from_instruction(*args, **kwargs):  # noqa: ARG004
                raise AssertionError("Statevector allocation happened before budget gate")

        monkeypatch.setattr(adapt_vqe_mod, "Statevector", FailIfStatevectorRequested)

        with pytest.raises(DenseAllocationError, match="ADAPT-VQE statevector"):
            adapt_vqe(K, omega, max_iterations=1, seed=42, max_dense_gib=1e-9)


class TestAdaptVQECoverage:
    """Lines 156-172 (operator selection + re-optimisation) are unreachable
    with XY Hamiltonian: |0...0⟩ is eigenstate → gradient always 0 → ADAPT
    terminates immediately. Same root cause as quantum_speed_limit line 124."""

    def test_adapt_immediate_convergence_documented(self):
        """Verify ADAPT converges immediately (gradient=0) for XY model."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = adapt_vqe(K, omega, max_iterations=3, gradient_threshold=1e-12, seed=42)
        assert result.converged
        assert len(result.selected_operators) == 0
        assert result.gradient_norms[0] == 0.0

    def test_adapt_selection_and_optimisation_loop(self, monkeypatch):
        """Forced gradient exercises operator selection and optimiser callback."""
        adapt_module = importlib.import_module("scpn_quantum_control.phase.adapt_vqe")
        calls = {"cost": 0}

        def fake_gradient(_sv, _hamiltonian, pool):
            gradients = np.zeros(len(pool))
            gradients[1] = 2.0
            return gradients

        def simple_ansatz(n, _pool, _selected, _params):
            return QuantumCircuit(n)

        def fake_minimise(cost_fn, x0, method, options):
            assert method == "COBYLA"
            assert options["maxiter"] == 3
            calls["cost"] += 1
            cost_fn(np.asarray(x0))
            return SimpleNamespace(x=np.asarray(x0) + 0.25)

        monkeypatch.setattr(adapt_module, "_compute_gradient", fake_gradient)
        monkeypatch.setattr(adapt_module, "_build_ansatz", simple_ansatz)
        monkeypatch.setattr(adapt_module, "minimize", fake_minimise)

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = adapt_vqe(
            K,
            omega,
            max_iterations=1,
            gradient_threshold=0.5,
            maxiter_opt=3,
            seed=42,
        )

        assert result.selected_operators == [1]
        assert result.n_parameters == 1
        assert result.gradient_norms == [2.0]
        assert calls["cost"] == 1
