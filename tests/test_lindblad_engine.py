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
        assert len(engine.L_ops_dense) == 4

    def test_invalid_method_raises(self):
        """Unknown method argument must raise ValueError."""
        K = np.array([[0.0, 1.0], [1.0, 0.0]])
        omega = np.array([1.0, 1.0])
        engine = LindbladSyncEngine(K, omega)
        import pytest

        with pytest.raises(ValueError, match="Unknown method"):
            engine.evolve(t_max=0.1, method="bogus")

    def test_trajectory_with_observables(self):
        """Trajectory path must propagate observable expectations."""
        from qiskit.quantum_info import SparsePauliOp

        K = np.array([[0.0, 1.0], [1.0, 0.0]])
        omega = np.array([1.0, 1.0])
        engine = LindbladSyncEngine(K, omega, gamma=0.05)
        obs = [SparsePauliOp.from_list([("ZI", 1.0)])]
        res = engine.evolve(t_max=0.2, n_steps=5, method="trajectory", n_traj=5, observables=obs)
        assert "observables" in res
        key = list(res["observables"].keys())[0]
        assert len(res["observables"][key]) == 6  # n_steps + 1

    def test_density_matrix_with_observables(self):
        """Density matrix path must track observable history."""
        from qiskit.quantum_info import SparsePauliOp

        K = np.array([[0.0, 1.0], [1.0, 0.0]])
        omega = np.array([1.0, 1.0])
        engine = LindbladSyncEngine(K, omega, gamma=0.05)
        obs = [SparsePauliOp.from_list([("ZZ", 1.0)])]
        res = engine.evolve(t_max=0.2, n_steps=5, method="density_matrix", observables=obs)
        assert "observables" in res
        key = list(res["observables"].keys())[0]
        assert len(res["observables"][key]) == 6
