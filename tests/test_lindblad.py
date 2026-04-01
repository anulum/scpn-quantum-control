# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Lindblad
"""Tests for Lindblad open-system Kuramoto-XY solver.

Multi-angle: Pauli operator algebra, density matrix invariants,
trace/Hermiticity preservation, energy conservation under unitary,
purity decay, dissipation regimes, parametrised system sizes,
zero-dissipation consistency, physical bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase.lindblad import LindbladKuramotoSolver, _sigma


# =====================================================================
# Pauli Operator Construction
# =====================================================================
class TestSigmaOperators:
    """Verify single-qubit Pauli operator construction."""

    def test_sigma_x_shape_and_values(self):
        sx = _sigma("X", 0, 2)
        assert sx.shape == (4, 4)
        assert sx[0, 2] == 1.0
        assert sx[2, 0] == 1.0

    def test_sigma_z_diagonal_and_eigenvalues(self):
        sz = _sigma("Z", 0, 1)
        assert sz[0, 0] == 1.0
        assert sz[1, 1] == -1.0
        eigvals = np.linalg.eigvalsh(sz)
        np.testing.assert_allclose(sorted(eigvals), [-1.0, 1.0])

    def test_sigma_plus_structure(self):
        sp = _sigma("+", 0, 1)
        assert sp[0, 1] == 1.0
        assert sp[1, 0] == 0.0
        assert np.count_nonzero(sp) == 1

    def test_pauli_anticommutation(self):
        """{X, Y} = 0 — fundamental Pauli algebra."""
        sx = _sigma("X", 0, 1)
        sy = _sigma("Y", 0, 1)
        anticomm = sx @ sy + sy @ sx
        np.testing.assert_allclose(anticomm, np.zeros((2, 2)), atol=1e-14)

    def test_pauli_commutation_xy(self):
        """[X, Y] = 2iZ."""
        sx = _sigma("X", 0, 1)
        sy = _sigma("Y", 0, 1)
        sz = _sigma("Z", 0, 1)
        comm = sx @ sy - sy @ sx
        np.testing.assert_allclose(comm, 2j * sz, atol=1e-14)

    def test_pauli_squared_is_identity(self):
        """X² = Y² = Z² = I."""
        for pauli in ["X", "Y", "Z"]:
            s = _sigma(pauli, 0, 1)
            np.testing.assert_allclose(s @ s, np.eye(2), atol=1e-14)

    @pytest.mark.parametrize("qubit", [0, 1, 2])
    def test_sigma_on_different_qubits(self, qubit):
        """Pauli on different qubits should be Hermitian and unitary."""
        sx = _sigma("X", qubit, 3)
        assert sx.shape == (8, 8)
        np.testing.assert_allclose(sx, sx.conj().T, atol=1e-14)
        np.testing.assert_allclose(sx @ sx, np.eye(8), atol=1e-14)

    def test_sigma_minus_is_adjoint_of_plus(self):
        sp = _sigma("+", 0, 1)
        sm = _sigma("-", 0, 1)
        np.testing.assert_allclose(sm, sp.conj().T, atol=1e-14)


# =====================================================================
# Density Matrix Invariants
# =====================================================================
class TestDensityMatrixInvariants:
    """Verify density matrix properties are preserved during evolution."""

    def setup_method(self):
        self.n = 3
        self.K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(self.n), range(self.n))))
        np.fill_diagonal(self.K, 0.0)
        self.omega = np.linspace(0.8, 1.2, self.n)

    def test_trace_preserved(self):
        """Tr(ρ) = 1 at all times."""
        solver = LindbladKuramotoSolver(
            self.n,
            self.K,
            self.omega,
            gamma_amp=0.1,
            gamma_deph=0.05,
        )
        result = solver.run(t_max=1.0, dt=0.05)
        rho = result["rho_final"]
        np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-6)

    def test_hermiticity_preserved(self):
        """ρ must be Hermitian: ρ = ρ†."""
        solver = LindbladKuramotoSolver(
            self.n,
            self.K,
            self.omega,
            gamma_amp=0.1,
        )
        result = solver.run(t_max=0.5, dt=0.05)
        rho = result["rho_final"]
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-8)

    def test_positive_semidefinite(self):
        """All eigenvalues of ρ must be ≥ 0."""
        solver = LindbladKuramotoSolver(
            self.n,
            self.K,
            self.omega,
            gamma_amp=0.1,
            gamma_deph=0.1,
        )
        result = solver.run(t_max=1.0, dt=0.05)
        eigvals = np.linalg.eigvalsh(result["rho_final"])
        assert all(v > -1e-8 for v in eigvals), f"Negative eigenvalue: {min(eigvals)}"

    def test_purity_bounded(self):
        """1/d ≤ Tr(ρ²) ≤ 1 for d-dimensional system."""
        dim = 2**self.n
        solver = LindbladKuramotoSolver(
            self.n,
            self.K,
            self.omega,
            gamma_amp=0.2,
        )
        result = solver.run(t_max=2.0, dt=0.1)
        for p in result["purity"]:
            assert 1.0 / dim - 1e-6 <= p <= 1.0 + 1e-6, f"Purity {p} out of [1/{dim}, 1]"


# =====================================================================
# Unitary Evolution (Zero Dissipation)
# =====================================================================
class TestUnitaryEvolution:
    """With γ=0, Lindblad reduces to von Neumann: dρ/dt = -i[H,ρ]."""

    def setup_method(self):
        self.n = 2
        self.K = np.array([[0, 0.5], [0.5, 0]])
        self.omega = np.array([1.0, 1.2])

    def test_purity_preserved_exactly(self):
        solver = LindbladKuramotoSolver(
            self.n,
            self.K,
            self.omega,
            gamma_amp=0.0,
            gamma_deph=0.0,
        )
        result = solver.run(t_max=1.0, dt=0.05)
        np.testing.assert_allclose(
            result["purity"],
            np.ones_like(result["purity"]),
            atol=1e-6,
        )

    def test_matches_unitary_solver(self):
        """Zero-dissipation Lindblad should match unitary R(t)."""
        from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver

        lindblad = LindbladKuramotoSolver(
            self.n,
            self.K,
            self.omega,
            gamma_amp=0,
            gamma_deph=0,
        )
        result_l = lindblad.run(t_max=0.5, dt=0.05)

        unitary = QuantumKuramotoSolver(self.n, self.K, self.omega)
        result_u = unitary.run(t_max=0.5, dt=0.05)

        np.testing.assert_allclose(
            result_l["R"],
            result_u["R"],
            atol=0.05,
            err_msg="Zero-dissipation Lindblad should match unitary R(t)",
        )

    def test_trace_exactly_one(self):
        solver = LindbladKuramotoSolver(
            self.n,
            self.K,
            self.omega,
            gamma_amp=0.0,
        )
        result = solver.run(t_max=0.5, dt=0.05)
        np.testing.assert_allclose(
            np.trace(result["rho_final"]),
            1.0,
            atol=1e-10,
        )


# =====================================================================
# Dissipation Regimes
# =====================================================================
class TestDissipationRegimes:
    """Test various dissipation strengths and channels."""

    def setup_method(self):
        self.n = 2
        self.K = np.array([[0, 0.5], [0.5, 0]])
        self.omega = np.array([1.0, 1.2])

    def test_damping_reduces_purity(self):
        solver = LindbladKuramotoSolver(
            self.n,
            self.K,
            self.omega,
            gamma_amp=0.5,
        )
        result = solver.run(t_max=1.0, dt=0.1)
        assert result["purity"][-1] < result["purity"][0]

    def test_dephasing_reduces_purity(self):
        solver = LindbladKuramotoSolver(
            self.n,
            self.K,
            self.omega,
            gamma_deph=0.5,
        )
        result = solver.run(t_max=1.0, dt=0.1)
        assert result["purity"][-1] < result["purity"][0]

    def test_strong_damping_kills_sync(self):
        solver = LindbladKuramotoSolver(
            self.n,
            self.K,
            self.omega,
            gamma_amp=5.0,
            gamma_deph=5.0,
        )
        result = solver.run(t_max=2.0, dt=0.1)
        assert result["R"][-1] < 0.3, (
            f"Strong damping should destroy sync, got R={result['R'][-1]:.3f}"
        )

    def test_purity_monotonically_decreases_under_damping(self):
        """Purity should not increase under dissipation (Markovian)."""
        solver = LindbladKuramotoSolver(
            self.n,
            self.K,
            self.omega,
            gamma_amp=0.2,
            gamma_deph=0.1,
        )
        result = solver.run(t_max=2.0, dt=0.05)
        purity = result["purity"]
        for i in range(1, len(purity)):
            assert purity[i] <= purity[i - 1] + 1e-6, (
                f"Purity increased at step {i}: {purity[i - 1]:.6f} → {purity[i]:.6f}"
            )

    @pytest.mark.parametrize("gamma_amp", [0.01, 0.05, 0.1, 0.5, 1.0])
    def test_r_bounded_across_damping_rates(self, gamma_amp):
        """R ∈ [0, 1] for all damping rates."""
        solver = LindbladKuramotoSolver(
            self.n,
            self.K,
            self.omega,
            gamma_amp=gamma_amp,
        )
        result = solver.run(t_max=0.5, dt=0.05)
        assert all(0 <= r <= 1.0 + 1e-6 for r in result["R"])


# =====================================================================
# Multiple System Sizes
# =====================================================================
class TestMultipleSizes:
    """Test across different oscillator counts."""

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_runs_and_produces_correct_shapes(self, n):
        K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
        np.fill_diagonal(K, 0.0)
        omega = np.linspace(0.8, 1.2, n)

        solver = LindbladKuramotoSolver(n, K, omega, gamma_amp=0.05, gamma_deph=0.02)
        result = solver.run(t_max=0.3, dt=0.1)

        dim = 2**n
        assert result["rho_final"].shape == (dim, dim)
        assert len(result["R"]) == len(result["times"])
        assert len(result["purity"]) == len(result["times"])

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_decoupled_system(self, n):
        """K=0: decoupled oscillators, purity preserved."""
        K = np.zeros((n, n))
        omega = np.linspace(0.8, 1.2, n)

        solver = LindbladKuramotoSolver(n, K, omega, gamma_amp=0.0)
        result = solver.run(t_max=0.5, dt=0.1)

        np.testing.assert_allclose(result["purity"][-1], 1.0, atol=1e-6)


# =====================================================================
# Output Keys and Types
# =====================================================================
class TestOutputFormat:
    def test_output_keys(self):
        solver = LindbladKuramotoSolver(
            2,
            np.array([[0, 0.3], [0.3, 0]]),
            np.array([1.0, 1.1]),
        )
        result = solver.run(t_max=0.2, dt=0.1)
        assert set(result.keys()) == {"times", "R", "purity", "rho_final"}

    def test_output_types(self):
        solver = LindbladKuramotoSolver(
            2,
            np.array([[0, 0.3], [0.3, 0]]),
            np.array([1.0, 1.1]),
        )
        result = solver.run(t_max=0.2, dt=0.1)
        assert isinstance(result["times"], np.ndarray)
        assert isinstance(result["R"], np.ndarray)
        assert isinstance(result["purity"], np.ndarray)
        assert isinstance(result["rho_final"], np.ndarray)

    def test_all_values_finite(self):
        K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(3), range(3))))
        np.fill_diagonal(K, 0.0)
        solver = LindbladKuramotoSolver(
            3,
            K,
            np.linspace(0.8, 1.2, 3),
            gamma_amp=0.1,
        )
        result = solver.run(t_max=1.0, dt=0.1)
        assert all(np.isfinite(result["R"]))
        assert all(np.isfinite(result["purity"]))
        assert np.all(np.isfinite(result["rho_final"]))
