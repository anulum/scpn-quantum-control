# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Bkt Analysis
"""Tests for BKT phase transition analysis."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.bkt_analysis import (
    BKTResult,
    bkt_analysis,
    coupling_laplacian,
    estimate_t_bkt,
    fiedler_eigenvalue,
    scan_synchronization_transition,
)
from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27


class TestCouplingLaplacian:
    def test_laplacian_symmetric(self):
        K = build_knm_paper27(L=8)
        L = coupling_laplacian(K)
        np.testing.assert_allclose(L, L.T, atol=1e-12)

    def test_laplacian_row_sums_zero(self):
        K = build_knm_paper27(L=8)
        L = coupling_laplacian(K)
        np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-12)

    def test_laplacian_positive_semidefinite(self):
        K = build_knm_paper27(L=8)
        L = coupling_laplacian(K)
        eigenvalues = np.linalg.eigvalsh(L)
        assert np.all(eigenvalues >= -1e-12)

    def test_laplacian_zero_diagonal_coupling(self):
        K = np.array([[0.0, 0.5], [0.5, 0.0]])
        L = coupling_laplacian(K)
        expected = np.array([[0.5, -0.5], [-0.5, 0.5]])
        np.testing.assert_allclose(L, expected, atol=1e-12)


class TestFiedlerEigenvalue:
    def test_fiedler_positive_connected_graph(self):
        K = build_knm_paper27(L=8)
        lam2 = fiedler_eigenvalue(K)
        assert lam2 > 0, "Connected graph must have positive Fiedler value"

    def test_fiedler_increases_with_coupling(self):
        lam_weak = fiedler_eigenvalue(build_knm_paper27(L=8, K_base=0.1))
        lam_strong = fiedler_eigenvalue(build_knm_paper27(L=8, K_base=1.0))
        assert lam_strong > lam_weak

    def test_fiedler_uniform_coupling(self):
        """Complete graph with uniform coupling: λ_2 = n × k."""
        n = 5
        k = 0.3
        K = np.full((n, n), k)
        np.fill_diagonal(K, 0.0)
        lam2 = fiedler_eigenvalue(K)
        np.testing.assert_allclose(lam2, n * k, atol=1e-10)


class TestEstimateTBKT:
    def test_t_bkt_positive(self):
        K = build_knm_paper27(L=16)
        t = estimate_t_bkt(K)
        assert t > 0

    def test_t_bkt_scales_with_coupling(self):
        t_weak = estimate_t_bkt(build_knm_paper27(L=16, K_base=0.1))
        t_strong = estimate_t_bkt(build_knm_paper27(L=16, K_base=1.0))
        assert t_strong > t_weak

    def test_t_bkt_formula(self):
        """T_BKT = (π/2) × λ_2 / (2n)."""
        K = build_knm_paper27(L=8)
        t = estimate_t_bkt(K)
        lam2 = fiedler_eigenvalue(K)
        expected = (np.pi / 2.0) * lam2 / (2.0 * 8)
        np.testing.assert_allclose(t, expected, atol=1e-14)


class TestBKTAnalysis:
    def test_returns_bkt_result(self):
        K = build_knm_paper27(L=16)
        result = bkt_analysis(K)
        assert isinstance(result, BKTResult)

    def test_n_oscillators(self):
        K = build_knm_paper27(L=10)
        result = bkt_analysis(K)
        assert result.n_oscillators == 10

    def test_eta_universal(self):
        K = build_knm_paper27(L=16)
        result = bkt_analysis(K)
        assert result.eta_critical == 0.25

    def test_stiffness_jump_nelson_kosterlitz(self):
        """ρ_s = (2/π) × T_BKT."""
        K = build_knm_paper27(L=16)
        result = bkt_analysis(K)
        expected = (2.0 / np.pi) * result.t_bkt_estimate
        np.testing.assert_allclose(result.stiffness_jump, expected, atol=1e-14)

    def test_critical_ratio_positive(self):
        K = build_knm_paper27(L=16)
        result = bkt_analysis(K)
        assert result.critical_ratio > 0

    def test_p_h1_predicted_bounded(self):
        K = build_knm_paper27(L=16)
        result = bkt_analysis(K)
        assert result.p_h1_predicted is not None
        assert 0 < result.p_h1_predicted <= 1.0

    def test_p_h1_vs_target(self):
        """Record predicted p_h1 and compare to target 0.72.

        This is a MEASUREMENT, not a pass/fail assertion.
        The BKT bound-pair formula gives ~0.94 for n=16, which is above 0.72.
        The deviation quantifies how much finite-size/noise correction is needed.
        """
        K = build_knm_paper27(L=16)
        result = bkt_analysis(K)
        deviation = abs(result.p_h1_predicted - 0.72)
        print(f"\n  BKT predicted p_h1 = {result.p_h1_predicted:.4f}")
        print("  Target p_h1 = 0.72")
        print(f"  Deviation = {deviation:.4f}")
        assert isinstance(result.p_h1_predicted, float)

    def test_p_h1_none_for_single_oscillator(self):
        K = np.array([[0.0]])
        result = bkt_analysis(K)
        assert result.p_h1_predicted is None

    def test_fiedler_matches_direct(self):
        K = build_knm_paper27(L=16)
        result = bkt_analysis(K)
        direct = fiedler_eigenvalue(K)
        np.testing.assert_allclose(result.fiedler_value, direct, atol=1e-14)


class TestScanSynchronizationTransition:
    def test_scan_returns_all_keys(self):
        K_vals = np.linspace(0.1, 1.0, 5)
        results = scan_synchronization_transition(K_vals, n=8)
        for key in ["K_base", "T_BKT", "fiedler", "critical_ratio", "p_h1_predicted"]:
            assert key in results
            assert len(results[key]) == 5

    def test_t_bkt_monotonic(self):
        K_vals = np.linspace(0.1, 2.0, 10)
        results = scan_synchronization_transition(K_vals, n=8)
        t_bkt = results["T_BKT"]
        for i in range(1, len(t_bkt)):
            assert t_bkt[i] >= t_bkt[i - 1], "T_BKT must increase with K_base"

    def test_fiedler_monotonic(self):
        K_vals = np.linspace(0.1, 2.0, 10)
        results = scan_synchronization_transition(K_vals, n=8)
        fiedler = results["fiedler"]
        for i in range(1, len(fiedler)):
            assert fiedler[i] >= fiedler[i - 1]
