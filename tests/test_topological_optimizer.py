# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Tests for Topological Quantum Reinforcement Learning / Optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.quantum_persistent_homology import _RIPSER_AVAILABLE
from scpn_quantum_control.control.topological_optimizer import TopologicalCouplingOptimizer


class TestTopologicalOptimizer:
    def test_topological_optimizer_step(self):
        """Verify one cycle of topological gradient descent on p_h1."""
        if not _RIPSER_AVAILABLE:
            pytest.skip("ripser not available")

        n = 3
        # Incoherent state (all decoupled) -> should yield high p_h1 eventually.
        initial_K = np.zeros((n, n))
        omega = np.array([5.0, 10.0, 15.0])

        opt = TopologicalCouplingOptimizer(
            n_qubits=n, initial_K=initial_K, omega=omega, learning_rate=0.5, dt=0.5
        )

        # We step it
        res = opt.step(n_samples=2)

        assert "K_updated" in res
        assert "p_h1_current" in res
        assert "gradient_norm" in res
        assert res["K_updated"].shape == (n, n)
        # Verify symmetry
        np.testing.assert_allclose(res["K_updated"], res["K_updated"].T)

    def test_optimize_loop(self):
        if not _RIPSER_AVAILABLE:
            pytest.skip("ripser not available")

        n = 2
        initial_K = np.ones((n, n)) * 0.1
        omega = np.array([5.0, 5.0])

        opt = TopologicalCouplingOptimizer(
            n_qubits=n, initial_K=initial_K, omega=omega, learning_rate=0.1, dt=0.5
        )

        history = opt.optimize(steps=2, n_samples=1)
        assert len(history) == 2

    def test_k_symmetry_preserved(self):
        """K must remain symmetric after optimization steps."""
        if not _RIPSER_AVAILABLE:
            pytest.skip("ripser not available")
        n = 3
        rng = np.random.default_rng(42)
        K0 = rng.random((n, n)) * 0.3
        K0 = (K0 + K0.T) / 2
        np.fill_diagonal(K0, 0)
        omega = np.array([5.0, 7.0, 9.0])
        opt = TopologicalCouplingOptimizer(
            n_qubits=n, initial_K=K0, omega=omega, learning_rate=0.1, dt=0.5
        )
        res = opt.step(n_samples=2)
        np.testing.assert_allclose(res["K_updated"], res["K_updated"].T, atol=1e-12)

    def test_k_non_negative(self):
        """K entries must remain >= 0."""
        if not _RIPSER_AVAILABLE:
            pytest.skip("ripser not available")
        n = 2
        K0 = np.array([[0, 0.01], [0.01, 0]])
        omega = np.array([5.0, 5.0])
        opt = TopologicalCouplingOptimizer(
            n_qubits=n, initial_K=K0, omega=omega, learning_rate=1.0, dt=0.5
        )
        res = opt.step(n_samples=3)
        assert np.all(res["K_updated"] >= -1e-15)

    def test_diagonal_stays_zero(self):
        """Self-coupling K[i,i] must remain zero."""
        if not _RIPSER_AVAILABLE:
            pytest.skip("ripser not available")
        n = 3
        K0 = np.ones((n, n)) * 0.2
        np.fill_diagonal(K0, 0)
        omega = np.ones(n) * 5.0
        opt = TopologicalCouplingOptimizer(
            n_qubits=n, initial_K=K0, omega=omega, learning_rate=0.1, dt=0.5
        )
        history = opt.optimize(steps=3, n_samples=1)
        for step in history:
            np.testing.assert_allclose(np.diag(step["K_updated"]), 0.0)

    def test_gradient_norm_finite(self):
        """Gradient norm must be a finite non-negative number."""
        if not _RIPSER_AVAILABLE:
            pytest.skip("ripser not available")
        n = 2
        K0 = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([5.0, 10.0])
        opt = TopologicalCouplingOptimizer(n_qubits=n, initial_K=K0, omega=omega)
        res = opt.step(n_samples=2)
        assert np.isfinite(res["gradient_norm"])
        assert res["gradient_norm"] >= 0
