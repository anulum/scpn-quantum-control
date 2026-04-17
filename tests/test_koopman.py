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
import pytest

from scpn_quantum_control.analysis.koopman import (
    MAX_OSCILLATORS_DEFAULT,
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


# ---------------------------------------------------------------------------
# Coverage: 2-oscillator, edge cases, generator structure
# ---------------------------------------------------------------------------


class TestKoopman2Oscillator:
    def test_2osc_dimension(self):
        assert koopman_dimension(2) == 4

    def test_2osc_generator(self):
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, 1.5])
        L, labels = build_koopman_generator(K, omega)
        # 2 theta + 1 cos + 1 sin = 4
        assert L.shape == (4, 4)
        assert len(labels) == 4

    def test_2osc_analysis(self):
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, 1.5])
        result = koopman_analysis(K, omega)
        assert result.n_oscillators == 2
        assert result.n_observables == 4

    def test_2osc_hamiltonian_size(self):
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, 1.5])
        L, _ = build_koopman_generator(K, omega)
        H = koopman_to_hamiltonian(L)
        assert H.shape == (4, 4)


class TestKoopmanGeneratorPhysics:
    def test_identical_frequencies_block(self):
        """Identical frequencies → Δω=0 → cos-sin coupling vanishes."""
        K = np.array([[0, 1.0], [1.0, 0]])
        omega = np.array([1.0, 1.0])
        L, _ = build_koopman_generator(K, omega)
        # cos-sin block: L[2,3] and L[3,2] should be 0 (Δω=0)
        assert abs(L[2, 3]) < 1e-10
        assert abs(L[3, 2]) < 1e-10

    def test_generator_finite(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        L, _ = build_koopman_generator(K, omega)
        assert np.all(np.isfinite(L))


class TestKoopmanInputValidation:
    """Guards against malformed or pathologically large input.

    Fixes the unbounded eigvals / dense allocation surface that the
    internal security audit (docs/internal/security_audit.md) flagged
    as VULN-SQC-001 on 2026-04-15.
    """

    @pytest.fixture
    def valid_inputs(self):
        return np.array([[0.0, 0.5], [0.5, 0.0]]), np.array([1.0, 1.5])

    def test_default_max_oscillators_is_sane(self):
        # 32 oscillators → dim=1024, ~8 MB. Higher would let a stray
        # caller silently allocate hundreds of MB and queue eigvals
        # for many minutes.
        assert MAX_OSCILLATORS_DEFAULT == 32

    def test_non_square_matrix_rejected(self):
        K = np.zeros((3, 4))
        omega = np.array([1.0, 1.5, 2.0])
        with pytest.raises(ValueError, match="square 2-D matrix"):
            build_koopman_generator(K, omega)

    def test_one_dimensional_K_rejected(self):
        K = np.array([0.0, 0.5, 0.0, 0.0])
        omega = np.array([1.0, 1.5])
        with pytest.raises(ValueError, match="square 2-D matrix"):
            build_koopman_generator(K, omega)

    def test_empty_K_rejected(self):
        K = np.zeros((0, 0))
        omega = np.array([])
        with pytest.raises(ValueError, match="at least one oscillator"):
            build_koopman_generator(K, omega)

    def test_K_with_nan_rejected(self, valid_inputs):
        K, omega = valid_inputs
        K = K.copy()
        K[0, 1] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            build_koopman_generator(K, omega)

    def test_K_with_inf_rejected(self, valid_inputs):
        K, omega = valid_inputs
        K = K.copy()
        K[1, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            build_koopman_generator(K, omega)

    def test_omega_length_mismatch_rejected(self, valid_inputs):
        K, _ = valid_inputs
        with pytest.raises(ValueError, match="omega"):
            build_koopman_generator(K, np.array([1.0, 1.5, 2.0]))

    def test_omega_with_nan_rejected(self, valid_inputs):
        K, omega = valid_inputs
        with pytest.raises(ValueError, match="non-finite"):
            build_koopman_generator(K, np.array([1.0, np.nan]))

    def test_theta_ref_length_mismatch_rejected(self, valid_inputs):
        K, omega = valid_inputs
        with pytest.raises(ValueError, match="theta_ref"):
            build_koopman_generator(K, omega, theta_ref=np.array([0.0, 0.0, 0.0]))

    def test_n_above_default_rejected(self):
        n = MAX_OSCILLATORS_DEFAULT + 1
        K = np.zeros((n, n))
        omega = np.zeros(n)
        with pytest.raises(ValueError, match="exceeds max_oscillators"):
            build_koopman_generator(K, omega)

    def test_explicit_max_oscillators_allows_larger_n(self):
        # Caller explicitly opts in. Use a modestly larger size that
        # still completes quickly so the test stays fast.
        n = MAX_OSCILLATORS_DEFAULT + 1
        K = np.zeros((n, n))
        omega = np.zeros(n)
        L, labels = build_koopman_generator(K, omega, max_oscillators=n)
        assert L.shape == (n * n, n * n)
        assert len(labels) == n * n

    def test_koopman_analysis_propagates_validation(self):
        K = np.zeros((3, 4))
        omega = np.array([1.0, 1.5, 2.0])
        with pytest.raises(ValueError, match="square 2-D matrix"):
            koopman_analysis(K, omega)

    def test_koopman_analysis_propagates_max_oscillators(self):
        n = MAX_OSCILLATORS_DEFAULT + 1
        K = np.zeros((n, n))
        omega = np.zeros(n)
        with pytest.raises(ValueError, match="exceeds max_oscillators"):
            koopman_analysis(K, omega)
