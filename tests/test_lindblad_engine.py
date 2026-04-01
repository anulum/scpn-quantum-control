# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Tests for Lindblad Master Equation Solver."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.phase.lindblad_engine import LindbladSyncEngine


class TestLindbladSyncEngine:
    def test_lindblad_evolution_preserves_trace(self):
        """Verify that the Lindbladian preserves trace of the density matrix."""
        K = np.array([[0.0, 1.0], [1.0, 0.0]])
        omega = np.array([5.0, 5.0])

        engine = LindbladSyncEngine(K, omega, gamma=0.1)
        res = engine.evolve(t_max=0.5, n_steps=2)

        for rho in res["states"]:
            tr = np.trace(rho)
            assert abs(tr - 1.0) < 1e-6

    def test_dissipator_construction(self):
        K = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        omega = np.array([1.0, 1.0, 1.0])
        engine = LindbladSyncEngine(K, omega)
        # Should have 4 jump operators (0->1, 1->0, 1->2, 2->1)
        assert len(engine.L_ops) == 4
