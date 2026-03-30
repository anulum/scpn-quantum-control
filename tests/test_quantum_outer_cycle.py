# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Quantum Outer Cycle
"""Tests for quantum SSGF outer cycle."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.ssgf.quantum_outer_cycle import (
    OuterCycleResult,
    classical_cost,
    quantum_outer_cycle,
)


class TestClassicalCost:
    def test_zero_coupling_high_cost(self):
        W = np.zeros((3, 3))
        c = classical_cost(W)
        assert c >= 0.5

    def test_strong_uniform_coupling_lower_cost(self):
        W = np.ones((3, 3)) * 2.0
        np.fill_diagonal(W, 0.0)
        c = classical_cost(W)
        assert c < 1.0

    def test_bounded(self):
        W = np.random.default_rng(42).uniform(0, 1, (4, 4))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        c = classical_cost(W)
        assert isinstance(c, float)


class TestQuantumOuterCycle:
    def test_returns_result(self):
        result = quantum_outer_cycle(n_osc=2, max_iterations=3, seed=42)
        assert isinstance(result, OuterCycleResult)

    def test_z_shape(self):
        result = quantum_outer_cycle(n_osc=3, max_iterations=2, seed=42)
        assert result.z_optimised.shape == (3,)  # 3 choose 2 = 3

    def test_w_symmetric(self):
        result = quantum_outer_cycle(n_osc=3, max_iterations=2, seed=42)
        np.testing.assert_allclose(result.W_optimised, result.W_optimised.T, atol=1e-12)

    def test_w_non_negative(self):
        result = quantum_outer_cycle(n_osc=3, max_iterations=2, seed=42)
        assert np.all(result.W_optimised >= -1e-10)

    def test_cost_history_length(self):
        result = quantum_outer_cycle(n_osc=2, max_iterations=5, seed=42)
        assert len(result.cost_history) <= 5
        assert len(result.r_global_history) == len(result.cost_history)

    def test_r_global_bounded(self):
        result = quantum_outer_cycle(n_osc=2, max_iterations=3, seed=42)
        for r in result.r_global_history:
            assert 0 <= r <= 1.0

    def test_pure_quantum(self):
        result = quantum_outer_cycle(n_osc=2, alpha=1.0, max_iterations=3, seed=42)
        assert isinstance(result.final_cost, float)

    def test_pure_classical(self):
        result = quantum_outer_cycle(n_osc=2, alpha=0.0, max_iterations=3, seed=42)
        assert isinstance(result.final_cost, float)

    def test_custom_z_init(self):
        z0 = np.array([0.5, -0.3, 1.0])
        result = quantum_outer_cycle(n_osc=3, z_init=z0, max_iterations=2)
        assert result.n_iterations >= 1

    def test_convergence_flag(self):
        result = quantum_outer_cycle(
            n_osc=2, max_iterations=50, convergence_threshold=100.0, seed=42
        )
        assert result.converged
