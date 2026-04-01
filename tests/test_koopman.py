# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Koopman
"""Tests for Koopman linearisation of nonlinear Kuramoto."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.koopman import (
    KoopmanResult,
    build_koopman_generator,
    koopman_analysis,
    koopman_dimension,
    koopman_to_hamiltonian,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestKoopmanDimension:
    def test_formula(self):
        assert koopman_dimension(4) == 16
        assert koopman_dimension(16) == 256

    def test_scales_quadratically(self):
        d4 = koopman_dimension(4)
        d8 = koopman_dimension(8)
        assert d8 == 4 * d4


class TestBuildKoopmanGenerator:
    def test_shape(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        L, labels = build_koopman_generator(K, omega)
        assert L.shape == (16, 16)  # n + 2 × n(n-1)/2 = 4 + 12 = 16

    def test_labels_count(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        _L, labels = build_koopman_generator(K, omega)
        assert len(labels) == 16

    def test_labels_structure(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        _L, labels = build_koopman_generator(K, omega)
        # 3 theta + 3 cos + 3 sin = 9
        assert len(labels) == 9
        assert labels[0] == "θ_0"
        assert "cos" in labels[3]
        assert "sin" in labels[6]

    def test_zero_coupling_block_diagonal(self):
        """With K=0, pairs decouple: only frequency terms remain."""
        n = 3
        K = np.zeros((n, n))
        omega = np.array([1.0, 2.0, 3.0])
        L, _labels = build_koopman_generator(K, omega)
        # θ rows should be zero (no coupling)
        np.testing.assert_allclose(L[:n, :], 0.0, atol=1e-12)

    def test_custom_reference(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        theta_ref = np.array([0.1, 0.2, 0.3])
        L, _labels = build_koopman_generator(K, omega, theta_ref=theta_ref)
        assert L.shape[0] == 9


class TestKoopmanAnalysis:
    def test_returns_result(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = koopman_analysis(K, omega)
        assert isinstance(result, KoopmanResult)

    def test_n_observables(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = koopman_analysis(K, omega)
        assert result.n_observables == 16
        assert result.n_oscillators == 4

    def test_eigenvalue_count(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = koopman_analysis(K, omega)
        assert len(result.eigenvalues) == 16

    def test_eigenvalues_sorted_by_magnitude(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = koopman_analysis(K, omega)
        mags = np.abs(result.eigenvalues)
        for i in range(len(mags) - 1):
            assert mags[i] >= mags[i + 1] - 1e-10

    def test_scpn_koopman_spectrum(self):
        """Record Koopman spectrum at SCPN default."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = koopman_analysis(K, omega)
        top5 = result.eigenvalues[:5]
        print(f"\n  Koopman dim: {result.n_observables}")
        print(f"  Top 5 eigenvalues: {[f'{e:.4f}' for e in top5]}")
        assert result.n_observables == koopman_dimension(4)


class TestKoopmanToHamiltonian:
    def test_hermitian(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        L, _labels = build_koopman_generator(K, omega)
        H = koopman_to_hamiltonian(L)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_shape_preserved(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        L, _labels = build_koopman_generator(K, omega)
        H = koopman_to_hamiltonian(L)
        assert H.shape == L.shape

    def test_real_eigenvalues(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        L, _labels = build_koopman_generator(K, omega)
        H = koopman_to_hamiltonian(L)
        eigs = np.linalg.eigvalsh(H)
        assert np.all(np.isreal(eigs))
