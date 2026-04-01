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
