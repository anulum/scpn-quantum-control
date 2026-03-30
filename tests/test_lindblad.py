# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
"""Tests for Lindblad open-system Kuramoto-XY solver."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.phase.lindblad import LindbladKuramotoSolver, _sigma


class TestSigmaOperators:
    def test_sigma_x_shape(self):
        sx = _sigma("X", 0, 2)
        assert sx.shape == (4, 4)

    def test_sigma_z_diagonal(self):
        sz = _sigma("Z", 0, 1)
        assert sz[0, 0] == 1.0
        assert sz[1, 1] == -1.0

    def test_sigma_plus_lowers(self):
        sp = _sigma("+", 0, 1)
        assert sp[0, 1] == 1.0
        assert sp[1, 0] == 0.0

    def test_pauli_anticommutation(self):
        sx = _sigma("X", 0, 1)
        sy = _sigma("Y", 0, 1)
        assert np.allclose(sx @ sy + sy @ sx, np.zeros((2, 2)))


class TestLindbladSolverBasic:
    def setup_method(self):
        self.n = 2
        self.K = np.array([[0, 0.5], [0.5, 0]])
        self.omega = np.array([1.0, 1.2])

    def test_unitary_preserves_purity(self):
        solver = LindbladKuramotoSolver(self.n, self.K, self.omega, gamma_amp=0.0, gamma_deph=0.0)
        result = solver.run(t_max=0.5, dt=0.05)
        assert all(p > 0.99 for p in result["purity"]), "Unitary evolution should preserve purity"

    def test_damping_reduces_purity(self):
        solver = LindbladKuramotoSolver(self.n, self.K, self.omega, gamma_amp=0.5, gamma_deph=0.0)
        result = solver.run(t_max=1.0, dt=0.1)
        assert result["purity"][-1] < result["purity"][0], "Damping should reduce purity"

    def test_dephasing_reduces_purity(self):
        solver = LindbladKuramotoSolver(self.n, self.K, self.omega, gamma_amp=0.0, gamma_deph=0.5)
        result = solver.run(t_max=1.0, dt=0.1)
        assert result["purity"][-1] < result["purity"][0], "Dephasing should reduce purity"

    def test_r_stays_bounded(self):
        solver = LindbladKuramotoSolver(self.n, self.K, self.omega, gamma_amp=0.1, gamma_deph=0.1)
        result = solver.run(t_max=1.0, dt=0.1)
        assert all(0 <= r <= 1.01 for r in result["R"]), "R should stay in [0, 1]"

    def test_strong_damping_kills_sync(self):
        solver = LindbladKuramotoSolver(self.n, self.K, self.omega, gamma_amp=5.0, gamma_deph=5.0)
        result = solver.run(t_max=2.0, dt=0.1)
        assert result["R"][-1] < 0.3, "Strong damping should destroy synchronisation"

    def test_output_keys(self):
        solver = LindbladKuramotoSolver(self.n, self.K, self.omega)
        result = solver.run(t_max=0.2, dt=0.1)
        assert set(result.keys()) == {"times", "R", "purity", "rho_final"}

    def test_rho_final_is_density_matrix(self):
        solver = LindbladKuramotoSolver(self.n, self.K, self.omega, gamma_amp=0.1)
        result = solver.run(t_max=0.5, dt=0.1)
        rho = result["rho_final"]
        assert np.allclose(np.trace(rho), 1.0, atol=1e-6), "Tr(rho) should be 1"
        eigvals = np.linalg.eigvalsh(rho)
        assert all(v > -1e-6 for v in eigvals), "rho should be positive semi-definite"


class TestLindbladVsUnitary:
    def test_zero_dissipation_matches_unitary(self):
        n = 2
        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, 1.1])

        lindblad = LindbladKuramotoSolver(n, K, omega, gamma_amp=0, gamma_deph=0)
        result_l = lindblad.run(t_max=0.5, dt=0.05)

        from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver

        unitary = QuantumKuramotoSolver(n, K, omega)
        result_u = unitary.run(t_max=0.5, dt=0.05)

        np.testing.assert_allclose(
            result_l["R"],
            result_u["R"],
            atol=0.05,
            err_msg="Zero-dissipation Lindblad should match unitary R(t)",
        )


class TestLindbladFourOscillators:
    def test_4osc_runs(self):
        n = 4
        K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
        omega = np.linspace(0.8, 1.2, n)
        solver = LindbladKuramotoSolver(n, K, omega, gamma_amp=0.05, gamma_deph=0.02)
        result = solver.run(t_max=0.5, dt=0.1)
        assert len(result["R"]) == 6
        assert result["purity"][-1] < 1.0
