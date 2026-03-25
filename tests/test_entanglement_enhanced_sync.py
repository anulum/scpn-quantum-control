# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for entanglement-enhanced synchronization."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from scpn_quantum_control.analysis.entanglement_enhanced_sync import (
    InitialState,
    SyncTrajectory,
    compare_all_initial_states,
    entanglement_advantage,
    prepare_initial_state,
    simulate_sync_trajectory,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


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
        for name, traj in results.items():
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
        for name, data in adv.items():
            assert "delta_R_final" in data
            assert "convergence_speedup" in data

    def test_empty_without_product(self):
        adv = entanglement_advantage({})
        assert adv == {}
