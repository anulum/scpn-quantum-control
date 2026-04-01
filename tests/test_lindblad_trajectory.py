# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Tests for Lindblad Quantum Trajectory vs Density Matrix path."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from scpn_quantum_control.phase.lindblad_engine import LindbladSyncEngine


class TestLindbladTrajectory:
    def test_trajectory_matches_density_matrix(self):
        """Verify trajectory average converges to density matrix solution for N=2."""

        K = np.array([[0.0, 1.0], [1.0, 0.0]])
        omega = np.array([0.5, 0.5])
        gamma = 0.2
        t_max = 1.0
        n_steps = 10

        obs = [SparsePauliOp("ZZ")]

        engine = LindbladSyncEngine(K, omega, gamma=gamma)

        # Density Matrix path
        res_dm = engine.evolve(t_max, n_steps, method="density_matrix", observables=obs)
        val_dm = res_dm["observables"][str(obs[0])]

        # Trajectory path (higher n_traj for better convergence)
        res_tj = engine.evolve(
            t_max, n_steps, method="trajectory", n_traj=200, seed=42, observables=obs
        )
        val_tj = res_tj["observables"][str(obs[0])]

        # Check end point
        assert abs(val_dm[-1] - val_tj[-1]) < 0.1

    def test_large_n_no_memory_error(self):
        """Verify that N=12 (trajectory) doesn't crash but N=12 (density_matrix) raises."""
        # We use N=12 instead of 16 for faster test
        n = 12
        K = np.eye(n)  # sparse-ish
        omega = np.ones(n)
        engine = LindbladSyncEngine(K, omega)

        # Trajectory should work (memory-wise)
        # We just step it once to verify it doesn't crash
        engine.evolve(t_max=0.1, n_steps=1, method="trajectory", n_traj=1)

        # Density Matrix should raise RuntimeError because N > 10
        with pytest.raises(RuntimeError, match="Density matrix path only supported for N <= 10"):
            engine.evolve(t_max=0.1, n_steps=1, method="density_matrix")
