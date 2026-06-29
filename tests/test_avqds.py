# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Avqds
"""Tests for fixed-ansatz McLachlan variational real-time dynamics."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_quantum_control.phase.avqds as avqds_mod
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.phase.avqds import (
    AVQDSResult,
    _mclachlan_matrices,
    _qubits_from_state_length,
    avqds_simulate,
)


class _SparseLike:
    def __init__(self, matrix):
        self.matrix = matrix

    def toarray(self):
        return self.matrix


class _FakeHamiltonian:
    def __init__(self, matrix):
        self.matrix = matrix

    def to_matrix(self):
        return _SparseLike(self.matrix)


class _FakeAnsatz:
    num_parameters = 1

    def __init__(self, *, expose_num_qubits: bool = False):
        if expose_num_qubits:
            self.num_qubits = 1

    def assign_parameters(self, params):
        return params


class _FakeStatevector:
    def __init__(self, data):
        self.data = data

    @classmethod
    def from_instruction(cls, assigned):
        theta = float(np.asarray(assigned)[0])
        return cls(np.array([np.cos(theta), np.sin(theta)], dtype=np.complex128))

    def expectation_value(self, hamiltonian):
        matrix = hamiltonian.to_matrix().toarray()
        return complex(np.vdot(self.data, matrix @ self.data))


class TestAVQDS:
    def test_returns_result(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = avqds_simulate(K, omega, t_total=0.1, n_steps=3, seed=42)
        assert isinstance(result, AVQDSResult)

    def test_times_shape(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = avqds_simulate(K, omega, t_total=0.5, n_steps=5, seed=42)
        assert result.times.shape == (6,)
        assert result.energies.shape == (6,)
        assert result.fidelities.shape == (6,)

    def test_initial_fidelity_one(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = avqds_simulate(K, omega, t_total=0.1, n_steps=3, seed=42)
        assert result.fidelities[0] == pytest.approx(1.0, abs=1e-6)

    def test_energy_finite(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = avqds_simulate(K, omega, t_total=0.1, n_steps=3, seed=42)
        assert np.all(np.isfinite(result.energies))

    def test_parameters_history(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = avqds_simulate(K, omega, t_total=0.1, n_steps=3, seed=42)
        assert len(result.parameters_history) >= 1

    def test_n_params_positive(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = avqds_simulate(K, omega, t_total=0.1, n_steps=2, seed=42)
        assert result.n_params > 0

    def test_3_oscillators(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = avqds_simulate(K, omega, t_total=0.1, n_steps=2, seed=42)
        assert result.n_params > 0

    def test_ansatz_parameter_count_is_constant_not_adaptive(self):
        """The fixed ansatz must never grow: it is not the adaptive AVQDS.

        Guards the docstring claim against the behaviour. Adaptive operator-pool
        growth would lengthen the parameter vector on demand; here every entry
        of the trajectory has the same length ``n_params`` across all steps.
        """
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        n_steps = 6
        result = avqds_simulate(K, omega, t_total=0.2, n_steps=n_steps, seed=42)

        # n*2*reps = 3*2*2 = 12 fixed parameters of the physics-informed ansatz.
        assert result.n_params == 12
        assert len(result.parameters_history) == n_steps + 1
        sizes = {vector.size for vector in result.parameters_history}
        assert sizes == {result.n_params}

    def test_short_time_high_fidelity(self):
        """Very short evolution should maintain high fidelity."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = avqds_simulate(K, omega, t_total=0.01, n_steps=2, seed=42)
        assert result.final_fidelity > 0.9

    def test_rejects_dense_budget_before_hamiltonian_matrix_allocation(self, monkeypatch):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        class FailIfHamiltonianMatrixRequested:
            def to_matrix(self):
                raise AssertionError("Hamiltonian matrix allocation happened before budget gate")

        monkeypatch.setattr(
            avqds_mod,
            "knm_to_hamiltonian",
            lambda *args, **kwargs: FailIfHamiltonianMatrixRequested(),
        )

        with pytest.raises(DenseAllocationError, match="AVQDS dense Hamiltonian"):
            avqds_simulate(K, omega, t_total=0.1, n_steps=2, seed=42, max_dense_gib=1e-9)

    def test_rejects_non_power_of_two_statevector_length(self):
        with pytest.raises(ValueError, match="positive power of two"):
            _qubits_from_state_length(3)

    def test_mclachlan_uses_declared_ansatz_qubit_count(self, monkeypatch):
        monkeypatch.setattr(avqds_mod, "Statevector", _FakeStatevector)
        matrix = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.complex128)

        M, V = _mclachlan_matrices(
            _FakeAnsatz(expose_num_qubits=True),
            np.array([0.1], dtype=np.float64),
            _FakeHamiltonian(matrix),
            max_dense_gib=1e-6,
        )

        assert M.shape == (1, 1)
        assert V.shape == (1,)
        assert np.all(np.isfinite(M))
        assert np.all(np.isfinite(V))

    def test_simulation_contract_with_sparse_like_boundaries(self, monkeypatch):
        matrix = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.complex128)
        monkeypatch.setattr(
            avqds_mod, "knm_to_hamiltonian", lambda K, omega: _FakeHamiltonian(matrix)
        )
        monkeypatch.setattr(
            avqds_mod,
            "knm_to_ansatz",
            lambda K, reps: _FakeAnsatz(expose_num_qubits=False),
        )
        monkeypatch.setattr(avqds_mod, "Statevector", _FakeStatevector)

        result = avqds_simulate(
            np.zeros((1, 1), dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            t_total=0.1,
            n_steps=1,
            seed=0,
        )

        assert result.times.tolist() == [0.0, 0.1]
        assert len(result.parameters_history) == 2
        assert result.final_fidelity >= 0.0


def test_avqds_finite_energies():
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = avqds_simulate(K, omega, t_total=0.1, n_steps=3, seed=42)
    assert np.all(np.isfinite(result.energies))


def test_avqds_3q():
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    result = avqds_simulate(K, omega, t_total=0.1, n_steps=2, seed=0)
    assert result.n_params > 0


def test_avqds_fidelity_bounded():
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = avqds_simulate(K, omega, t_total=0.5, n_steps=5, seed=42)
    assert 0 <= result.final_fidelity <= 1.0 + 1e-10
