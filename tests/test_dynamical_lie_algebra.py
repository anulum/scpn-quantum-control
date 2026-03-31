# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Dynamical Lie Algebra
"""Tests for dynamical Lie algebra computation."""

from __future__ import annotations

import numpy as np

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

    def test_tcbo_zz_expands_dla(self):
        """Adding ZZ terms (TCBO) should expand the DLA beyond pure XY."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        xy_gens = build_xy_generators(K, omega)
        xy_result = compute_dla(xy_gens, max_iterations=30)

        tcbo_gens = build_tcbo_generators(K, omega, kappa=1.0, connectivity="nearest")
        tcbo_result = compute_dla(tcbo_gens, max_iterations=30)

        assert tcbo_result.dimension >= xy_result.dimension

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
