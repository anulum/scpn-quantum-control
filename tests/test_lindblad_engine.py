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

    def test_density_matrix_stays_positive_semidefinite(self):
        """Density matrix eigenvalues must remain >= 0 throughout."""
        K = np.array([[0.0, 0.5], [0.5, 0.0]])
        omega = np.array([1.0, 2.0])
        engine = LindbladSyncEngine(K, omega, gamma=0.2)
        res = engine.evolve(t_max=1.0, n_steps=20, method="density_matrix")
        for rho in res["states"]:
            eigvals = np.linalg.eigvalsh(rho)
            assert np.all(eigvals > -1e-8), f"Negative eigenvalue: {eigvals.min()}"

    def test_gamma_zero_unitary(self):
        """gamma=0 should produce unitary evolution (pure state preserved)."""
        K = np.array([[0.0, 0.5], [0.5, 0.0]])
        omega = np.array([1.0, 1.0])
        engine = LindbladSyncEngine(K, omega, gamma=0.0)
        res = engine.evolve(t_max=1.0, n_steps=10, method="density_matrix")
        for rho in res["states"]:
            tr = np.trace(rho).real
            purity = np.trace(rho @ rho).real
            assert abs(tr - 1.0) < 1e-6
            assert abs(purity - 1.0) < 1e-4

    def test_strong_dissipation_approaches_steady_state(self):
        """Strong gamma drives system toward a steady state (lower purity)."""
        K = np.array([[0.0, 1.0], [1.0, 0.0]])
        omega = np.array([1.0, 1.0])
        engine = LindbladSyncEngine(K, omega, gamma=2.0)
        res = engine.evolve(t_max=5.0, n_steps=50, method="density_matrix")
        purity_start = np.trace(res["states"][0] @ res["states"][0]).real
        purity_end = np.trace(res["states"][-1] @ res["states"][-1]).real
        assert purity_end <= purity_start + 1e-6

    def test_three_qubit_chain_jump_operators(self):
        """3-qubit chain: 4 jump operators (bidirectional on 2 active edges)."""
        K = np.array([[0, 0.5, 0], [0.5, 0, 0.3], [0, 0.3, 0]])
        omega = np.ones(3)
        engine = LindbladSyncEngine(K, omega, gamma=0.1)
        assert len(engine.L_ops_sparse) == 4
        assert len(engine.L_ops_dense) == 4

    def test_trajectory_density_matrix_agreement(self):
        """Trajectory and density matrix methods should give similar final states."""
        K = np.array([[0.0, 0.5], [0.5, 0.0]])
        omega = np.array([1.0, 1.0])
        engine = LindbladSyncEngine(K, omega, gamma=0.1)
        res_dm = engine.evolve(t_max=0.5, n_steps=10, method="density_matrix")
        res_tr = engine.evolve(t_max=0.5, n_steps=10, method="trajectory", n_traj=200, seed=42)
        np.testing.assert_allclose(res_dm["final_state"], res_tr["final_state"], atol=0.15)

    def test_anti_hermitian_sum_values(self):
        """Anti-Hermitian diagonal must count active jump channels per state."""
        K = np.array([[0.0, 1.0], [1.0, 0.0]])
        omega = np.array([1.0, 1.0])
        engine = LindbladSyncEngine(K, omega, gamma=0.1)
        diag = engine.anti_hermitian_sum
        # |01> can jump to |10> (1->0 transfer) = 1 channel
        # |10> can jump to |01> (0->1 transfer) = 1 channel
        # |00> = 0 channels, |11> = 0 channels
        assert diag[0b00] == 0.0
        assert diag[0b01] == 1.0
        assert diag[0b10] == 1.0
        assert diag[0b11] == 0.0

    def test_rust_python_jump_ops_parity(self):
        """Rust-built jump operators match Python implementation."""
        try:
            import scpn_quantum_engine as eng
        except ImportError:
            import pytest

            pytest.skip("Rust engine not available")

        K = np.array([[0, 0.5, 0.1], [0.5, 0, 0.3], [0.1, 0.3, 0]])
        n = 3
        dim = 1 << n

        # Rust path
        rows, cols, starts, n_ops = eng.lindblad_jump_ops_coo(K.ravel(), n, 1e-5)
        rows = np.array(rows)
        cols = np.array(cols)
        starts = np.array(starts)

        # Python path (direct construction)
        py_ops = []
        for i in range(n):
            for j in range(n):
                if i != j and abs(K[i, j]) > 1e-5:
                    row_py, col_py = [], []
                    for idx in range(dim):
                        if ((idx >> i) & 1) == 1 and ((idx >> j) & 1) == 0:
                            row_py.append(idx ^ ((1 << i) | (1 << j)))
                            col_py.append(idx)
                    py_ops.append((row_py, col_py))

        assert n_ops == len(py_ops)
        for k in range(n_ops):
            s, e = int(starts[k]), int(starts[k + 1])
            np.testing.assert_array_equal(sorted(rows[s:e]), sorted(py_ops[k][0]))
            np.testing.assert_array_equal(sorted(cols[s:e]), sorted(py_ops[k][1]))
