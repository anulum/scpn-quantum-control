# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Dynamical Lie Algebra
"""Tests for dynamical Lie algebra computation."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.dynamical_lie_algebra import (
    DLAResult,
    build_full_scpn_generators,
    build_ssgf_generators,
    build_tcbo_generators,
    build_xy_generators,
    compute_dla,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestDLAComputation:
    def test_single_qubit_z(self):
        """Single Z generator has DLA dimension 1."""
        from qiskit.quantum_info import SparsePauliOp

        gen = [SparsePauliOp("Z", 1.0)]
        result = compute_dla(gen)
        assert result.dimension == 1
        assert result.n_qubits == 1

    def test_two_qubit_xy(self):
        """XX + YY + Z_1 + Z_2 generates a finite DLA."""
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, 1.5])
        gens = build_xy_generators(K, omega)
        result = compute_dla(gens)
        assert result.dimension > len(gens)
        assert result.n_qubits == 2

    @pytest.mark.slow
    def test_xy_4qubit_dla_dimension(self):
        """4-qubit XY model DLA dimension: characterize the algebra.

        For all-to-all XY with heterogeneous frequencies (our K_nm),
        the DLA can be large — up to O(4^N) for N qubits.
        The g-sim polynomial regime requires specific symmetry.
        """
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        gens = build_xy_generators(K, omega)
        result = compute_dla(gens, max_iterations=30)
        # DLA dimension should be between N² and 4^N - 1
        assert result.dimension >= 4  # at least the generators
        assert result.dimension <= 255  # at most su(2^4)

    @pytest.mark.slow
    def test_tcbo_zz_expands_dla(self):
        """Adding ZZ terms (TCBO) should expand the DLA beyond pure XY."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        xy_gens = build_xy_generators(K, omega)
        xy_result = compute_dla(xy_gens, max_iterations=30)

        tcbo_gens = build_tcbo_generators(K, omega, kappa=1.0, connectivity="nearest")
        tcbo_result = compute_dla(tcbo_gens, max_iterations=30)

        assert tcbo_result.dimension >= xy_result.dimension

    @pytest.mark.slow
    def test_full_scpn_generators(self):
        """Full SCPN generators include XY + SSGF + PGBO + TCBO."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        W = K * 0.5  # arbitrary geometry
        rng = np.random.default_rng(42)
        u = rng.standard_normal(4)
        h_munu = np.outer(u, u)  # rank-1 PSD tensor
        h_munu = (h_munu + h_munu.T) / 2

        gens = build_full_scpn_generators(K, omega, W=W, h_munu=h_munu)
        result = compute_dla(gens, max_iterations=20)
        assert result.dimension > 0
        assert result.n_generators == len(gens)

    def test_dla_result_properties(self):
        """DLAResult properties work correctly."""
        result = DLAResult(
            dimension=16,
            n_qubits=4,
            n_generators=10,
            n_iterations=5,
            basis_labels=["H_0", "H_1"],
            is_polynomial=True,
            polynomial_degree=2.0,
            max_hilbert_dim=256,
        )
        assert result.classical_simulable is True
        assert result.dimension == 16

    @pytest.mark.slow
    def test_xy_vs_full_scpn_comparison(self):
        """Full SCPN should have equal or larger DLA than pure XY."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        W = np.abs(K) * 0.3
        np.fill_diagonal(W, 0)
        h = np.eye(3) * 0.1

        xy_gens = build_xy_generators(K, omega)
        xy_result = compute_dla(xy_gens, max_iterations=20)

        full_gens = build_full_scpn_generators(K, omega, W=W, h_munu=h, kappa=0.5)
        full_result = compute_dla(full_gens, max_iterations=20)

        assert full_result.dimension >= xy_result.dimension

    def test_ssgf_generators_include_geometry(self):
        """SSGF generators have more terms than base XY."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        W = np.ones((3, 3)) * 0.2
        np.fill_diagonal(W, 0)

        xy_gens = build_xy_generators(K, omega)
        ssgf_gens = build_ssgf_generators(K, omega, W, sigma_g=0.3)
        assert len(ssgf_gens) >= len(xy_gens)


def test_dla_dimension_2q():
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    gens = build_xy_generators(K, omega)
    result = compute_dla(gens)
    assert result.dimension >= 1
    assert result.n_qubits == 2


def test_dla_dimension_finite():
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    gens = build_xy_generators(K, omega)
    result = compute_dla(gens)
    assert np.isfinite(result.dimension)


def test_xy_generators_nonempty():
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    gens = build_xy_generators(K, omega)
    assert len(gens) > 0


# ---------------------------------------------------------------------------
# Coverage: internal helpers, Rust path, PGBO generators, edge cases
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    def test_commutator_antisymmetric(self):
        from scpn_quantum_control.analysis.dynamical_lie_algebra import _commutator

        A = np.array([[0, 1], [0, 0]], dtype=complex)
        B = np.array([[0, 0], [1, 0]], dtype=complex)
        ab = _commutator(A, B)
        ba = _commutator(B, A)
        np.testing.assert_allclose(ab, -ba, atol=1e-12)

    def test_commutator_diagonal_is_zero(self):
        from scpn_quantum_control.analysis.dynamical_lie_algebra import _commutator

        A = np.diag([1.0, 2.0, 3.0]).astype(complex)
        np.testing.assert_allclose(_commutator(A, A), 0.0, atol=1e-12)

    def test_is_independent_empty_basis(self):
        from scpn_quantum_control.analysis.dynamical_lie_algebra import _is_independent

        op = np.eye(2, dtype=complex)
        assert _is_independent(op, []) is True

    def test_is_independent_zero_op(self):
        from scpn_quantum_control.analysis.dynamical_lie_algebra import _is_independent

        op = np.zeros((2, 2), dtype=complex)
        assert _is_independent(op, []) is False

    def test_is_independent_duplicate(self):
        from scpn_quantum_control.analysis.dynamical_lie_algebra import _is_independent

        op = np.eye(2, dtype=complex)
        assert _is_independent(op, [op]) is False

    def test_is_independent_orthogonal(self):
        from scpn_quantum_control.analysis.dynamical_lie_algebra import _is_independent

        A = np.array([[1, 0], [0, 0]], dtype=complex)
        B = np.array([[0, 0], [0, 1]], dtype=complex)
        assert _is_independent(B, [A]) is True


class TestComputeDlaEdgeCases:
    def test_max_dimension_cap(self):
        """When max_dimension is reached, DLA stops expanding."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        gens = build_xy_generators(K, omega)
        # With cap=25, the first iteration adds all new ops (up to ~21)
        # but a second iteration would be stopped
        result_capped = compute_dla(gens, max_iterations=50, max_dimension=25)
        result_uncapped = compute_dla(gens, max_iterations=50, max_dimension=500)
        assert result_capped.dimension <= result_uncapped.dimension

    def test_polynomial_degree_n_cube_branch(self):
        """DLA between N² and N³ classified as polynomial degree 3."""
        result = DLAResult(
            dimension=50,
            n_qubits=4,
            n_generators=10,
            n_iterations=5,
            basis_labels=[],
            is_polynomial=True,
            polynomial_degree=3.0,
            max_hilbert_dim=256,
        )
        assert result.classical_simulable is True

    def test_exponential_regime(self):
        result = DLAResult(
            dimension=200,
            n_qubits=4,
            n_generators=10,
            n_iterations=5,
            basis_labels=[],
            is_polynomial=False,
            polynomial_degree=float("inf"),
            max_hilbert_dim=256,
        )
        assert result.classical_simulable is False


class TestComputeDlaRust:
    def test_rust_fallback_to_python(self):
        """compute_dla_rust falls back to Python when Rust unavailable."""
        from unittest.mock import patch

        from scpn_quantum_control.analysis.dynamical_lie_algebra import compute_dla_rust

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        gens = build_xy_generators(K, omega)

        with patch.dict("sys.modules", {"scpn_quantum_engine": None}):
            result = compute_dla_rust(gens)
            assert result.dimension >= 1
            assert result.n_qubits == 2

    def test_rust_vs_python_parity(self):
        """If Rust is available, results should match Python."""
        from scpn_quantum_control.analysis.dynamical_lie_algebra import compute_dla_rust

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        gens = build_xy_generators(K, omega)

        py = compute_dla(gens)
        rust = compute_dla_rust(gens)
        assert rust.dimension == py.dimension
        assert rust.n_qubits == py.n_qubits


class TestBuildPgboGenerators:
    def test_pgbo_more_generators(self):
        from scpn_quantum_control.analysis.dynamical_lie_algebra import build_pgbo_generators

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        rng = np.random.default_rng(42)
        u = rng.standard_normal(3)
        h = np.outer(u, u)
        h = (h + h.T) / 2

        xy_gens = build_xy_generators(K, omega)
        pgbo_gens = build_pgbo_generators(K, omega, h, pgbo_weight=0.1)
        assert len(pgbo_gens) >= len(xy_gens)

    def test_pgbo_zero_weight_equals_xy(self):
        from scpn_quantum_control.analysis.dynamical_lie_algebra import build_pgbo_generators

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        h = np.ones((2, 2)) * 0.5
        np.fill_diagonal(h, 0)

        xy_gens = build_xy_generators(K, omega)
        pgbo_gens = build_pgbo_generators(K, omega, h, pgbo_weight=0.0)
        assert len(pgbo_gens) == len(xy_gens)


class TestBuildTcboConnectivity:
    def test_full_connectivity(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        gens_near = build_tcbo_generators(K, omega, connectivity="nearest")
        gens_full = build_tcbo_generators(K, omega, connectivity="full")
        assert len(gens_full) >= len(gens_near)


class TestBuildFullScpnNoneInputs:
    def test_no_ssgf_no_pgbo(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        gens = build_full_scpn_generators(K, omega, W=None, h_munu=None)
        xy_gens = build_xy_generators(K, omega)
        # Only XY + TCBO (nearest), no SSGF/PGBO
        assert len(gens) >= len(xy_gens)
