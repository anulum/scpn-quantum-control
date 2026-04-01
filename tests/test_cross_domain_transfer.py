# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Cross Domain Transfer
"""Tests for cross-domain VQE transfer learning."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.phase.cross_domain_transfer import (
    PhysicalSystem,
    TransferResult,
    build_systems,
    run_transfer_matrix,
    summarize_transfer,
    transfer_experiment,
)


class TestBuildSystems:
    def test_returns_four_systems(self):
        systems = build_systems(n_qubits=3)
        assert len(systems) == 4

    def test_system_shapes(self):
        systems = build_systems(n_qubits=4)
        for s in systems:
            assert isinstance(s, PhysicalSystem)
            assert s.K.shape == (4, 4)
            assert len(s.omega) == 4

    def test_names_unique(self):
        systems = build_systems(n_qubits=3)
        names = [s.name for s in systems]
        assert len(names) == len(set(names))

    def test_K_symmetric(self):
        systems = build_systems(n_qubits=4)
        for s in systems:
            np.testing.assert_array_almost_equal(s.K, s.K.T)


class TestTransferExperiment:
    def test_returns_result(self):
        systems = build_systems(n_qubits=3)
        result = transfer_experiment(systems[0], systems[1], reps=1, maxiter=10)
        assert isinstance(result, TransferResult)
        assert result.source_system == systems[0].name
        assert result.target_system == systems[1].name

    def test_energies_finite(self):
        systems = build_systems(n_qubits=3)
        result = transfer_experiment(systems[0], systems[1], reps=1, maxiter=10)
        assert np.isfinite(result.random_init_energy)
        assert np.isfinite(result.transfer_init_energy)
        assert np.isfinite(result.exact_energy)

    def test_exact_is_lower_bound(self):
        systems = build_systems(n_qubits=3)
        result = transfer_experiment(systems[0], systems[1], reps=1, maxiter=20)
        # VQE energy should be >= exact ground state
        assert result.random_init_energy >= result.exact_energy - 0.1
        assert result.transfer_init_energy >= result.exact_energy - 0.1

    def test_speedup_positive(self):
        systems = build_systems(n_qubits=3)
        result = transfer_experiment(systems[0], systems[1], reps=1, maxiter=15)
        assert result.speedup > 0


class TestRunTransferMatrix:
    def test_all_pairs(self):
        results = run_transfer_matrix(n_qubits=2, reps=1, maxiter=5)
        # 4 systems × 3 targets each = 12 pairs
        assert len(results) == 12

    def test_no_self_transfer(self):
        results = run_transfer_matrix(n_qubits=2, reps=1, maxiter=5)
        for r in results:
            assert r.source_system != r.target_system


class TestSummarizeTransfer:
    def test_summary_keys(self):
        results = run_transfer_matrix(n_qubits=2, reps=1, maxiter=5)
        summary = summarize_transfer(results)
        assert "n_pairs" in summary
        assert "n_positive_transfer" in summary
        assert "best_speedup" in summary
        assert "mean_speedup" in summary
        assert summary["n_pairs"] == 12

    def test_empty_results(self):
        summary = summarize_transfer([])
        assert summary["n_pairs"] == 0
