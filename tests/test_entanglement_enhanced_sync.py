# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Entanglement Enhanced Sync
"""Tests for entanglement-enhanced synchronization."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

import scpn_quantum_control.analysis.entanglement_enhanced_sync as sync_module
from scpn_quantum_control.analysis.entanglement_enhanced_sync import (
    InitialState,
    SyncTrajectory,
    compare_all_initial_states,
    entanglement_advantage,
    prepare_initial_state,
    simulate_sync_trajectory,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.dense_budget import DenseAllocationError


class TestPrepareInitialState:
    def test_product_state(self):
        omega = OMEGA_N_16[:4]
        qc = prepare_initial_state(4, InitialState.PRODUCT, omega)
        sv = Statevector.from_instruction(qc)
        assert sv.num_qubits == 4
        assert abs(sv.probabilities().sum() - 1.0) < 1e-10

    def test_bell_pairs(self):
        omega = OMEGA_N_16[:4]
        qc = prepare_initial_state(4, InitialState.BELL_PAIRS, omega)
        sv = Statevector.from_instruction(qc)
        # Bell pairs: should have entanglement between qubits 0-1 and 2-3
        probs = sv.probabilities_dict()
        # |0000⟩ and |1111⟩ should have nonzero probability (|00⟩|00⟩ + |11⟩|11⟩ cross terms)
        assert len(probs) > 1

    def test_ghz_state(self):
        omega = OMEGA_N_16[:4]
        qc = prepare_initial_state(4, InitialState.GHZ, omega)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities_dict()
        # GHZ: only |0000⟩ and |1111⟩ should have probability
        nonzero = {k: v for k, v in probs.items() if v > 1e-10}
        assert len(nonzero) == 2
        assert all(k in ["0000", "1111"] for k in nonzero)
        assert all(abs(v - 0.5) < 1e-10 for v in nonzero.values())

    def test_w_state(self):
        omega = OMEGA_N_16[:4]
        qc = prepare_initial_state(4, InitialState.W_STATE, omega)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities_dict()
        # W state: |1000⟩, |0100⟩, |0010⟩, |0001⟩ each with prob 1/4
        nonzero = {k: v for k, v in probs.items() if v > 1e-10}
        assert len(nonzero) == 4
        for v in nonzero.values():
            assert abs(v - 0.25) < 1e-10

    def test_odd_qubit_count_bell(self):
        omega = OMEGA_N_16[:3]
        qc = prepare_initial_state(3, InitialState.BELL_PAIRS, omega)
        sv = Statevector.from_instruction(qc)
        assert sv.num_qubits == 3

    def test_two_qubit_w(self):
        omega = OMEGA_N_16[:2]
        qc = prepare_initial_state(2, InitialState.W_STATE, omega)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities_dict()
        nonzero = {k: v for k, v in probs.items() if v > 1e-10}
        assert len(nonzero) == 2


class TestSimulateSyncTrajectory:
    def test_rejects_dense_budget_before_hamiltonian_allocation(self, monkeypatch):
        K = build_knm_paper27(L=12)
        omega = OMEGA_N_16[:12]

        def fail_if_dense_hamiltonian_is_requested(*args, **kwargs):  # noqa: ARG001
            raise AssertionError("dense Hamiltonian allocation happened before budget gate")

        monkeypatch.setattr(
            sync_module,
            "knm_to_dense_matrix",
            fail_if_dense_hamiltonian_is_requested,
        )

        with pytest.raises(DenseAllocationError, match="entanglement-enhanced dense"):
            simulate_sync_trajectory(
                K,
                omega,
                InitialState.PRODUCT,
                t_max=0.1,
                n_steps=1,
                max_dense_gib=1e-12,
            )

    def test_passes_dense_budget_to_bridge(self, monkeypatch):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        seen_budgets: list[float | None] = []

        def fake_dense_matrix(K_arg, omega_arg, **kwargs):  # noqa: ARG001
            seen_budgets.append(kwargs.get("max_dense_gib"))
            return np.zeros((4, 4), dtype=complex)

        monkeypatch.setattr(sync_module, "knm_to_dense_matrix", fake_dense_matrix)

        simulate_sync_trajectory(
            K,
            omega,
            InitialState.PRODUCT,
            t_max=0.1,
            n_steps=1,
            max_dense_gib=0.25,
        )

        assert seen_budgets == [0.25]

    def test_returns_trajectory(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        traj = simulate_sync_trajectory(K, omega, InitialState.PRODUCT, t_max=0.5, n_steps=5)
        assert isinstance(traj, SyncTrajectory)
        assert len(traj.times) == 6  # 5 steps + initial
        assert len(traj.R_values) == 6
        assert traj.n_qubits == 3

    def test_R_bounded(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        traj = simulate_sync_trajectory(K, omega, InitialState.GHZ, t_max=1.0, n_steps=10)
        for r in traj.R_values:
            assert 0.0 <= r <= 1.0 + 1e-10

    def test_product_has_finite_R(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        traj = simulate_sync_trajectory(K, omega, InitialState.PRODUCT, t_max=1.0, n_steps=5)
        assert traj.final_R > 0.0

    def test_ghz_has_finite_R(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        traj = simulate_sync_trajectory(K, omega, InitialState.GHZ, t_max=1.0, n_steps=5)
        assert traj.final_R >= 0.0


class TestCompareAllStates:
    def test_propagates_dense_budget_to_each_initial_state(self, monkeypatch):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        seen: list[tuple[str, float | None]] = []

        def fake_trajectory(
            K_arg,  # noqa: ARG001
            omega_arg,  # noqa: ARG001
            state_type,
            t_max=2.0,  # noqa: ARG001
            n_steps=20,  # noqa: ARG001
            *,
            max_dense_gib=None,
        ):
            seen.append((state_type.value, max_dense_gib))
            return SyncTrajectory(state_type.value, [0.0], [1.0], 1.0, 2)

        monkeypatch.setattr(sync_module, "simulate_sync_trajectory", fake_trajectory)

        compare_all_initial_states(K, omega, t_max=0.5, n_steps=2, max_dense_gib=0.5)

        assert seen == [(state.value, 0.5) for state in InitialState]

    def test_returns_four_trajectories(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        results = compare_all_initial_states(K, omega, t_max=0.5, n_steps=3)
        assert len(results) == 4
        assert "product" in results
        assert "bell_pairs" in results
        assert "ghz" in results
        assert "w_state" in results

    def test_all_valid(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        results = compare_all_initial_states(K, omega, t_max=0.5, n_steps=3)
        for _name, traj in results.items():
            assert len(traj.R_values) == 4
            assert all(0 <= r <= 1 + 1e-10 for r in traj.R_values)


class TestEntanglementAdvantage:
    def test_returns_three_comparisons(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        results = compare_all_initial_states(K, omega, t_max=1.0, n_steps=10)
        adv = entanglement_advantage(results)
        assert "bell_pairs" in adv
        assert "ghz" in adv
        assert "w_state" in adv

    def test_has_expected_keys(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        results = compare_all_initial_states(K, omega, t_max=0.5, n_steps=5)
        adv = entanglement_advantage(results)
        for _name, data in adv.items():
            assert "delta_R_final" in data
            assert "convergence_speedup" in data

    def test_empty_without_product(self):
        adv = entanglement_advantage({})
        assert adv == {}
