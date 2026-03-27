# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Physics Invariant Tests
"""Systematic tests of physical invariants across all system sizes.

Tests conservation laws, symmetry properties, and mathematical bounds
that must hold for any valid quantum Kuramoto simulation:

  - Hamiltonian hermiticity
  - State normalisation under evolution
  - R order parameter bounded in [0, 1]
  - Spectral gap positivity
  - Energy conservation under unitary evolution
  - Sparse ↔ dense path agreement
  - Initial state preparation correctness
  - Coupling matrix symmetry
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    build_kuramoto_ring,
    knm_to_hamiltonian,
    knm_to_xxz_hamiltonian,
)
from scpn_quantum_control.hardware.classical import (
    _build_initial_state,
    _order_param,
    _state_order_param,
    _state_order_param_sparse,
    classical_exact_diag,
    classical_exact_evolution,
    classical_kuramoto_reference,
)

# ---------------------------------------------------------------------------
# Hamiltonian hermiticity
# ---------------------------------------------------------------------------


class TestHamiltonianHermiticity:
    @pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
    def test_paper27_hamiltonian_is_hermitian(self, n):
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        H = knm_to_hamiltonian(K, omega)
        H_mat = np.array(H.to_matrix())
        np.testing.assert_allclose(H_mat, H_mat.conj().T, atol=1e-14)

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_ring_hamiltonian_is_hermitian(self, n):
        K, omega = build_kuramoto_ring(n, coupling=1.0, rng_seed=0)
        H = knm_to_hamiltonian(K, omega)
        H_mat = np.array(H.to_matrix())
        np.testing.assert_allclose(H_mat, H_mat.conj().T, atol=1e-14)

    @pytest.mark.parametrize("delta", [0.0, 0.5, 1.0])
    def test_xxz_hamiltonian_is_hermitian(self, delta):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        H = knm_to_xxz_hamiltonian(K, omega, delta=delta)
        H_mat = np.array(H.to_matrix())
        np.testing.assert_allclose(H_mat, H_mat.conj().T, atol=1e-14)

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_zero_coupling_hamiltonian_diagonal(self, n):
        """H with K=0 is diagonal (only Z terms)."""
        K = np.zeros((n, n))
        omega = OMEGA_N_16[:n]
        H = knm_to_hamiltonian(K, omega)
        H_mat = np.array(H.to_matrix())
        off_diag = H_mat - np.diag(np.diag(H_mat))
        assert np.max(np.abs(off_diag)) < 1e-14

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_hamiltonian_eigenvalues_real(self, n):
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        H = knm_to_hamiltonian(K, omega)
        H_mat = np.array(H.to_matrix())
        eigenvalues = np.linalg.eigvalsh(H_mat)
        assert np.all(np.isreal(eigenvalues))

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_hamiltonian_trace(self, n):
        """XY Hamiltonian with only XX+YY terms has zero trace (traceless Paulis)."""
        K = build_knm_paper27(L=n)
        omega = np.zeros(n)
        H = knm_to_hamiltonian(K, omega)
        H_mat = np.array(H.to_matrix())
        assert abs(np.trace(H_mat)) < 1e-12


# ---------------------------------------------------------------------------
# State normalisation
# ---------------------------------------------------------------------------


class TestStateNormalisation:
    @pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
    def test_initial_state_normalised(self, n):
        omega = OMEGA_N_16[:n]
        psi = _build_initial_state(n, omega)
        np.testing.assert_allclose(np.linalg.norm(psi), 1.0, atol=1e-14)

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    @pytest.mark.parametrize("dt", [0.01, 0.05, 0.1])
    def test_evolution_preserves_norm(self, n, dt):
        """Unitary evolution preserves statevector norm."""
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        H = knm_to_hamiltonian(K, omega)
        H_mat = np.array(H.to_matrix())
        U = expm(-1j * H_mat * dt)

        psi = _build_initial_state(n, omega)
        for _ in range(10):
            psi = U @ psi
            np.testing.assert_allclose(np.linalg.norm(psi), 1.0, atol=1e-12)

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_propagator_unitarity(self, n):
        """U^dag U = I."""
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        H = knm_to_hamiltonian(K, omega)
        H_mat = np.array(H.to_matrix())
        U = expm(-1j * H_mat * 0.1)
        product = U.conj().T @ U
        np.testing.assert_allclose(product, np.eye(2**n), atol=1e-12)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_ground_state_normalised(self, n):
        result = classical_exact_diag(n)
        np.testing.assert_allclose(np.linalg.norm(result["ground_state"]), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# R order parameter bounds
# ---------------------------------------------------------------------------


class TestOrderParameterBounds:
    @pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
    @pytest.mark.parametrize("dt", [0.01, 0.05, 0.1])
    def test_quantum_R_in_01(self, n, dt):
        """R(t) must be in [0, 1] at every time step."""
        result = classical_exact_evolution(n, t_max=0.5, dt=dt)
        for r in result["R"]:
            assert -1e-10 <= r <= 1.0 + 1e-10, f"R={r} out of [0,1] for n={n}, dt={dt}"

    @pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
    @pytest.mark.parametrize("dt", [0.01, 0.05, 0.1])
    def test_classical_R_in_01(self, n, dt):
        result = classical_kuramoto_reference(n, t_max=0.5, dt=dt)
        for r in result["R"]:
            assert -1e-10 <= r <= 1.0 + 1e-10, f"R={r} out of [0,1] for n={n}, dt={dt}"

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_state_order_param_random_states(self, n):
        """R computed from random statevectors stays in [0, 1]."""
        rng = np.random.default_rng(123)
        for _ in range(20):
            dim = 2**n
            psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            psi /= np.linalg.norm(psi)
            R = _state_order_param(psi, n)
            assert -1e-10 <= R <= 1.0 + 1e-10

    def test_R_equals_1_for_synchronised_phases(self):
        """All phases identical -> R = 1."""
        theta = np.array([0.5, 0.5, 0.5, 0.5])
        assert abs(_order_param(theta) - 1.0) < 1e-14

    @pytest.mark.parametrize(
        "phases,expected_R",
        [
            (np.array([0.0, np.pi]), 0.0),
            (np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]), 0.0),
        ],
        ids=["antipodal_2", "uniform_4"],
    )
    def test_R_zero_for_uniform_distribution(self, phases, expected_R):
        """Uniformly distributed phases -> R ≈ 0."""
        assert abs(_order_param(phases) - expected_R) < 1e-14


# ---------------------------------------------------------------------------
# Spectral properties
# ---------------------------------------------------------------------------


class TestSpectralProperties:
    @pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
    def test_spectral_gap_positive(self, n):
        result = classical_exact_diag(n)
        assert result["spectral_gap"] > 0.0

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_eigenvalues_sorted_ascending(self, n):
        result = classical_exact_diag(n)
        evals = result["eigenvalues"]
        np.testing.assert_array_less(evals[:-1], evals[1:] + 1e-12)

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_ground_energy_is_minimum(self, n):
        result = classical_exact_diag(n)
        assert result["ground_energy"] == pytest.approx(result["eigenvalues"][0])

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_ground_state_is_eigenvector(self, n):
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        H = knm_to_hamiltonian(K, omega)
        H_mat = np.array(H.to_matrix())

        result = classical_exact_diag(n)
        gs = result["ground_state"]
        E0 = result["ground_energy"]

        residual = H_mat @ gs - E0 * gs
        assert np.linalg.norm(residual) < 1e-10

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_hilbert_space_dimension(self, n):
        result = classical_exact_diag(n)
        assert len(result["ground_state"]) == 2**n

    def test_spectral_gap_decreases_with_size(self):
        """Spectral gap generally decreases as system size grows."""
        gaps = []
        for n in [2, 4, 6, 8]:
            result = classical_exact_diag(n)
            gaps.append(result["spectral_gap"])
        assert gaps[0] > gaps[-1], "Spectral gap should decrease with system size"


# ---------------------------------------------------------------------------
# Energy conservation
# ---------------------------------------------------------------------------


class TestEnergyConservation:
    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.parametrize("dt", [0.01, 0.05])
    def test_energy_conserved_under_evolution(self, n, dt):
        """<psi(t)|H|psi(t)> is constant for unitary evolution."""
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        H = knm_to_hamiltonian(K, omega)
        H_mat = np.array(H.to_matrix())
        U = expm(-1j * H_mat * dt)

        psi = _build_initial_state(n, omega)
        E0 = float(np.real(psi.conj() @ H_mat @ psi))

        for step in range(20):
            psi = U @ psi
            E = float(np.real(psi.conj() @ H_mat @ psi))
            np.testing.assert_allclose(
                E, E0, atol=1e-10, err_msg=f"Energy drifted at step {step + 1}: {E} vs {E0}"
            )


# ---------------------------------------------------------------------------
# Sparse ↔ dense agreement
# ---------------------------------------------------------------------------


class TestSparseVsDense:
    @pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
    def test_state_order_param_sparse_matches_dense(self, n):
        """_state_order_param_sparse must match _state_order_param exactly."""
        omega = OMEGA_N_16[:n]
        psi = _build_initial_state(n, omega)
        R_dense = _state_order_param(psi, n)
        R_sparse = _state_order_param_sparse(psi, n)
        np.testing.assert_allclose(R_sparse, R_dense, atol=1e-14)

    @pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
    def test_sparse_dense_agree_random_states(self, n):
        """Sparse and dense R computation agree on random normalised states."""
        rng = np.random.default_rng(999)
        for _ in range(10):
            dim = 2**n
            psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            psi /= np.linalg.norm(psi)
            R_dense = _state_order_param(psi, n)
            R_sparse = _state_order_param_sparse(psi, n)
            np.testing.assert_allclose(R_sparse, R_dense, atol=1e-13)

    @pytest.mark.parametrize("n", [2, 4, 6, 8])
    def test_sparse_dense_evolution_agree(self, n):
        """Full evolution trajectory: sparse path matches dense path."""
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        H_op = knm_to_hamiltonian(K, omega)
        H_mat = np.array(H_op.to_matrix())
        U_dt = expm(-1j * H_mat * 0.05)
        psi = _build_initial_state(n, omega)

        for _ in range(5):
            psi = U_dt @ psi
            R_dense = _state_order_param(psi, n)
            R_sparse = _state_order_param_sparse(psi, n)
            np.testing.assert_allclose(R_sparse, R_dense, atol=1e-13)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_sparse_dense_on_basis_states(self, n):
        """R from computational basis states should agree between paths."""
        for idx in range(min(4, 2**n)):
            dim = 2**n
            psi = np.zeros(dim, dtype=complex)
            psi[idx] = 1.0
            R_dense = _state_order_param(psi, n)
            R_sparse = _state_order_param_sparse(psi, n)
            np.testing.assert_allclose(R_sparse, R_dense, atol=1e-14)


# ---------------------------------------------------------------------------
# Initial state preparation
# ---------------------------------------------------------------------------


class TestInitialStatePreparation:
    @pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
    def test_initial_state_dimension(self, n):
        omega = OMEGA_N_16[:n]
        psi = _build_initial_state(n, omega)
        assert len(psi) == 2**n

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_initial_state_deterministic(self, n):
        """Same frequencies -> same initial state."""
        omega = OMEGA_N_16[:n]
        psi1 = _build_initial_state(n, omega)
        psi2 = _build_initial_state(n, omega)
        np.testing.assert_array_equal(psi1, psi2)

    def test_initial_state_zero_frequencies(self):
        """All omega=0 -> all qubits in |0> -> |000...0>."""
        n = 4
        omega = np.zeros(n)
        psi = _build_initial_state(n, omega)
        expected = np.zeros(2**n, dtype=complex)
        expected[0] = 1.0
        np.testing.assert_allclose(psi, expected, atol=1e-14)

    def test_initial_state_pi_frequency(self):
        """omega=pi on qubit 0 -> Ry(pi)|0> = |1> on that qubit."""
        n = 1
        omega = np.array([np.pi])
        psi = _build_initial_state(n, omega)
        # Ry(pi)|0> = [cos(pi/2), sin(pi/2)] = [0, 1]
        np.testing.assert_allclose(psi, [0, 1], atol=1e-14)


# ---------------------------------------------------------------------------
# Coupling matrix properties
# ---------------------------------------------------------------------------


class TestCouplingMatrixProperties:
    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    def test_paper27_coupling_symmetric(self, n):
        K = build_knm_paper27(L=n)
        np.testing.assert_allclose(K, K.T, atol=1e-14)

    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    def test_paper27_coupling_nonneg(self, n):
        K = build_knm_paper27(L=n)
        assert np.all(K >= -1e-14)

    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    def test_paper27_diagonal_equals_kbase(self, n):
        """Diagonal K[i,i] = K_base * exp(0) = 0.45 from the formula.

        Not zeroed because knm_to_hamiltonian skips i==j (no self-coupling).
        """
        K = build_knm_paper27(L=n)
        np.testing.assert_allclose(np.diag(K), 0.45, atol=1e-14)

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_ring_coupling_symmetric(self, n):
        K, _ = build_kuramoto_ring(n, coupling=1.0, rng_seed=0)
        np.testing.assert_allclose(K, K.T, atol=1e-14)

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_ring_coupling_neighbours_only(self, n):
        K, _ = build_kuramoto_ring(n, coupling=1.0, rng_seed=0)
        for i in range(n):
            for j in range(n):
                if abs(i - j) == 1 or (i == 0 and j == n - 1) or (j == 0 and i == n - 1):
                    assert K[i, j] == pytest.approx(1.0)
                else:
                    assert K[i, j] == pytest.approx(0.0)

    @pytest.mark.parametrize("coupling", [0.0, 0.5, 1.0, 2.0])
    def test_ring_coupling_strength(self, coupling):
        K, _ = build_kuramoto_ring(4, coupling=coupling, rng_seed=0)
        for i in range(4):
            j = (i + 1) % 4
            assert K[i, j] == pytest.approx(coupling)


# ---------------------------------------------------------------------------
# Classical Kuramoto properties
# ---------------------------------------------------------------------------


class TestKuramotoProperties:
    @pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
    @pytest.mark.parametrize("dt", [0.01, 0.05, 0.1])
    def test_kuramoto_shapes(self, n, dt):
        result = classical_kuramoto_reference(n, t_max=0.5, dt=dt)
        n_steps = max(1, round(0.5 / dt))
        assert result["times"].shape == (n_steps + 1,)
        assert result["theta"].shape == (n_steps + 1, n)
        assert result["R"].shape == (n_steps + 1,)

    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_zero_coupling_free_rotation(self, n):
        """K=0: phases advance linearly at natural frequencies."""
        K = np.zeros((n, n))
        omega = OMEGA_N_16[:n]
        theta0 = np.zeros(n)
        result = classical_kuramoto_reference(
            n, t_max=0.1, dt=0.0001, K=K, omega=omega, theta0=theta0
        )
        np.testing.assert_allclose(result["theta"][-1], omega * 0.1, atol=0.01)

    def test_negative_dt_raises(self):
        with pytest.raises(ValueError, match="dt must be positive"):
            classical_kuramoto_reference(4, t_max=1.0, dt=-0.1)

    def test_negative_tmax_raises(self):
        with pytest.raises(ValueError, match="t_max must be non-negative"):
            classical_kuramoto_reference(4, t_max=-1.0, dt=0.1)

    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_identical_frequencies_preserve_sync(self, n):
        """Identical frequencies with coupling: synchronised start stays synchronised."""
        K = np.ones((n, n)) - np.eye(n)
        omega = np.ones(n)
        theta0 = np.ones(n) * 0.3
        result = classical_kuramoto_reference(
            n, t_max=0.5, dt=0.01, K=K, omega=omega, theta0=theta0
        )
        assert result["R"][-1] > 0.99


# ---------------------------------------------------------------------------
# Evolution output shapes
# ---------------------------------------------------------------------------


class TestEvolutionShapes:
    @pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
    @pytest.mark.parametrize("dt", [0.01, 0.05, 0.1])
    def test_exact_evolution_shapes(self, n, dt):
        result = classical_exact_evolution(n, t_max=0.3, dt=dt)
        n_steps = max(1, round(0.3 / dt))
        assert result["times"].shape == (n_steps + 1,)
        assert result["R"].shape == (n_steps + 1,)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_exact_evolution_initial_R_positive(self, n):
        result = classical_exact_evolution(n, t_max=0.1, dt=0.05)
        assert result["R"][0] > 0.0
