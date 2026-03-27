# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Hypothesis Property-Based Tests
"""Property-based tests using hypothesis for stochastic invariant checking.

These tests generate random inputs and verify that physical and mathematical
properties hold universally, not just for hand-picked examples.
"""

from __future__ import annotations

import numpy as np
from conftest import st_angles, st_coupling_matrix, st_dt, st_frequencies, st_statevector
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from scpn_quantum_control.bridge.knm_hamiltonian import (
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
    classical_brute_mpc,
    classical_kuramoto_reference,
)

# ---------------------------------------------------------------------------
# Kuramoto order parameter properties
# ---------------------------------------------------------------------------


class TestOrderParameterProperties:
    @given(st_angles(min_n=2, max_n=8))
    @settings(max_examples=100)
    def test_R_bounded_01(self, theta):
        """R(theta) in [0, 1] for any set of angles."""
        R = _order_param(theta)
        assert -1e-10 <= R <= 1.0 + 1e-10

    @given(st.integers(min_value=2, max_value=8))
    @settings(max_examples=20)
    def test_R_identical_angles(self, n):
        """All identical angles -> R = 1."""
        angle = 0.7
        theta = np.full(n, angle)
        assert abs(_order_param(theta) - 1.0) < 1e-14

    @given(st_angles(min_n=2, max_n=6))
    @settings(max_examples=50)
    def test_R_shift_invariant(self, theta):
        """R(theta + c) = R(theta) for any constant c."""
        c = 1.234
        R1 = _order_param(theta)
        R2 = _order_param(theta + c)
        np.testing.assert_allclose(R1, R2, atol=1e-14)


# ---------------------------------------------------------------------------
# Coupling matrix properties
# ---------------------------------------------------------------------------


class TestCouplingMatrixProperties:
    @given(st.integers(min_value=2, max_value=16))
    @settings(max_examples=30)
    def test_paper27_symmetric(self, n):
        K = build_knm_paper27(L=n)
        np.testing.assert_allclose(K, K.T, atol=1e-14)

    @given(st.integers(min_value=2, max_value=16))
    @settings(max_examples=30)
    def test_paper27_nonneg(self, n):
        K = build_knm_paper27(L=n)
        assert np.all(K >= -1e-14)

    @given(st.integers(min_value=2, max_value=16))
    @settings(max_examples=30)
    def test_paper27_constant_diagonal(self, n):
        """K_nm diagonal = K_base * exp(-alpha*0) = K_base for all n."""
        K = build_knm_paper27(L=n)
        diag = np.diag(K)
        assert np.all(diag > 0)
        np.testing.assert_allclose(diag, diag[0], atol=1e-14)

    @given(
        st.integers(min_value=2, max_value=10),
        st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_ring_symmetric(self, n, coupling):
        K, _ = build_kuramoto_ring(n, coupling=coupling, rng_seed=0)
        np.testing.assert_allclose(K, K.T, atol=1e-14)

    @given(
        st.integers(min_value=3, max_value=10),
        st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_ring_correct_nonzeros(self, n, coupling):
        """Ring graph has exactly 2*n non-zero entries (n >= 3, coupling > 0).

        n=2 excluded: ring of 2 nodes has only 2 entries (single bidirectional edge).
        """
        K, _ = build_kuramoto_ring(n, coupling=coupling, rng_seed=0)
        expected_nnz = 2 * n
        assert np.count_nonzero(K) == expected_nnz


# ---------------------------------------------------------------------------
# Hamiltonian properties
# ---------------------------------------------------------------------------


class TestHamiltonianProperties:
    @given(st.integers(min_value=2, max_value=6))
    @settings(max_examples=15)
    def test_hermiticity_paper27(self, n):
        K = build_knm_paper27(L=n)
        omega = np.ones(n)
        H = knm_to_hamiltonian(K, omega)
        H_mat = np.array(H.to_matrix())
        np.testing.assert_allclose(H_mat, H_mat.conj().T, atol=1e-14)

    @given(st_coupling_matrix(n=3), st_frequencies(n=3))
    @settings(max_examples=30)
    def test_hermiticity_random(self, K, omega):
        H = knm_to_hamiltonian(K, omega)
        H_mat = np.array(H.to_matrix())
        np.testing.assert_allclose(H_mat, H_mat.conj().T, atol=1e-12)

    @given(
        st_coupling_matrix(n=3),
        st_frequencies(n=3),
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_xxz_hermiticity_random(self, K, omega, delta):
        H = knm_to_xxz_hamiltonian(K, omega, delta=delta)
        H_mat = np.array(H.to_matrix())
        np.testing.assert_allclose(H_mat, H_mat.conj().T, atol=1e-12)

    @given(st_frequencies(n=3))
    @settings(max_examples=20)
    def test_zero_coupling_diagonal(self, omega):
        """K=0 -> H is diagonal (only Z terms)."""
        K = np.zeros((3, 3))
        H = knm_to_hamiltonian(K, omega)
        H_mat = np.array(H.to_matrix())
        off_diag = H_mat - np.diag(np.diag(H_mat))
        assert np.max(np.abs(off_diag)) < 1e-14

    @given(st_coupling_matrix(n=3))
    @settings(max_examples=20)
    def test_zero_field_traceless(self, K):
        """omega=0 -> H is traceless (all Paulis except I are traceless)."""
        assume(np.any(K != 0))
        omega = np.zeros(3)
        H = knm_to_hamiltonian(K, omega)
        H_mat = np.array(H.to_matrix())
        assert abs(np.trace(H_mat)) < 1e-12

    @given(st.integers(min_value=2, max_value=5))
    @settings(max_examples=10)
    def test_hamiltonian_dimension(self, n):
        K = build_knm_paper27(L=n)
        omega = np.ones(n)
        H = knm_to_hamiltonian(K, omega)
        assert H.num_qubits == n
        H_mat = np.array(H.to_matrix())
        assert H_mat.shape == (2**n, 2**n)


# ---------------------------------------------------------------------------
# State preparation properties
# ---------------------------------------------------------------------------


class TestStatePreparationProperties:
    @given(st.integers(min_value=1, max_value=8))
    @settings(max_examples=20)
    def test_normalised(self, n):
        omega = np.ones(n) * 0.5
        psi = _build_initial_state(n, omega)
        np.testing.assert_allclose(np.linalg.norm(psi), 1.0, atol=1e-14)

    @given(st.integers(min_value=1, max_value=8))
    @settings(max_examples=20)
    def test_dimension(self, n):
        omega = np.ones(n) * 0.5
        psi = _build_initial_state(n, omega)
        assert len(psi) == 2**n

    @given(st.integers(min_value=1, max_value=6), st_frequencies(n=None, min_n=1, max_n=1))
    @settings(max_examples=20)
    def test_deterministic(self, n, omega_base):
        omega = np.full(n, omega_base[0])
        psi1 = _build_initial_state(n, omega)
        psi2 = _build_initial_state(n, omega)
        np.testing.assert_array_equal(psi1, psi2)


# ---------------------------------------------------------------------------
# Sparse ↔ dense R agreement
# ---------------------------------------------------------------------------


class TestSparseVsDenseProperties:
    @given(st_statevector(min_n=2, max_n=4))
    @settings(max_examples=50)
    def test_agreement(self, psi_n):
        psi, n = psi_n
        R_dense = _state_order_param(psi, n)
        R_sparse = _state_order_param_sparse(psi, n)
        np.testing.assert_allclose(R_sparse, R_dense, atol=1e-12)

    @given(st_statevector(min_n=2, max_n=4))
    @settings(max_examples=50)
    def test_both_bounded(self, psi_n):
        psi, n = psi_n
        R_dense = _state_order_param(psi, n)
        R_sparse = _state_order_param_sparse(psi, n)
        assert -1e-10 <= R_dense <= 1.0 + 1e-10
        assert -1e-10 <= R_sparse <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# Brute-force MPC properties
# ---------------------------------------------------------------------------


class TestBruteMPCProperties:
    @given(
        st.integers(min_value=1, max_value=5),
        st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=30)
    def test_enumerates_all(self, dim, horizon):
        B = np.eye(dim)
        target = np.ones(dim)
        result = classical_brute_mpc(B, target, horizon)
        assert result["n_evaluated"] == 2**horizon
        assert result["all_costs"].shape == (2**horizon,)

    @given(
        st.integers(min_value=1, max_value=3),
        st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=20)
    def test_optimal_is_minimum(self, dim, horizon):
        B = np.eye(dim)
        target = np.ones(dim) * 0.5
        result = classical_brute_mpc(B, target, horizon)
        assert result["optimal_cost"] <= np.min(result["all_costs"]) + 1e-12

    @given(st.integers(min_value=1, max_value=4))
    @settings(max_examples=10)
    def test_binary_actions(self, horizon):
        B = np.eye(2)
        target = np.ones(2)
        result = classical_brute_mpc(B, target, horizon)
        assert set(np.unique(result["optimal_actions"])).issubset({0, 1})


# ---------------------------------------------------------------------------
# Classical Kuramoto properties
# ---------------------------------------------------------------------------


class TestKuramotoSimulationProperties:
    @given(
        st.integers(min_value=2, max_value=6),
        st_dt(min_val=0.01, max_val=0.1),
    )
    @settings(max_examples=30)
    def test_R_bounded(self, n, dt):
        result = classical_kuramoto_reference(n, t_max=0.3, dt=dt)
        for r in result["R"]:
            assert -1e-10 <= r <= 1.0 + 1e-10

    @given(st.integers(min_value=2, max_value=6))
    @settings(max_examples=15)
    def test_shapes_consistent(self, n):
        dt = 0.05
        result = classical_kuramoto_reference(n, t_max=0.5, dt=dt)
        n_steps = max(1, round(0.5 / dt))
        assert result["times"].shape == (n_steps + 1,)
        assert result["theta"].shape == (n_steps + 1, n)
        assert result["R"].shape == (n_steps + 1,)
