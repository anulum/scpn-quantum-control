# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Universality
"""Tests for BKT universality class analysis."""

from __future__ import annotations

import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.gauge.universality import (
    UniversalityResult,
    check_nelson_kosterlitz,
    correlation_vs_distance,
    fit_correlation_exponent,
    universality_analysis,
)


class TestCorrelationVsDistance:
    def test_returns_two_lists(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        distances, correlations = correlation_vs_distance(K, omega)
        assert len(distances) > 0
        assert len(correlations) == len(distances)

    def test_distances_positive(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        distances, _corr = correlation_vs_distance(K, omega)
        for d in distances:
            assert d > 0

    def test_correlations_bounded(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        _dist, correlations = correlation_vs_distance(K, omega)
        for c in correlations:
            assert abs(c) <= 2.0 + 1e-10  # XX+YY bounded by 2


class TestFitCorrelationExponent:
    def test_positive_correlations(self):
        distances = [1.0, 2.0, 3.0]
        correlations = [1.0, 0.5, 0.33]
        eta = fit_correlation_exponent(distances, correlations)
        assert eta is not None
        assert eta > 0

    def test_none_for_negative_correlations(self):
        distances = [1.0, 2.0]
        correlations = [-0.5, -0.3]
        eta = fit_correlation_exponent(distances, correlations)
        assert eta is None

    def test_known_exponent(self):
        """C(r) = r^{-0.25} should give η ≈ 0.25."""
        distances = [1.0, 2.0, 3.0, 4.0, 5.0]
        correlations = [d ** (-0.25) for d in distances]
        eta = fit_correlation_exponent(distances, correlations)
        assert eta is not None
        assert eta == pytest.approx(0.25, abs=0.01)


class TestCheckNelsonKosterlitz:
    def test_returns_ratio_and_deviation(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        ratio, dev = check_nelson_kosterlitz(K, omega)
        assert isinstance(ratio, float)
        assert dev >= 0

    def test_deviation_finite(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        _ratio, dev = check_nelson_kosterlitz(K, omega)
        assert dev < 100  # should be finite


class TestUniversalityAnalysis:
    def test_returns_result(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = universality_analysis(K, omega)
        assert isinstance(result, UniversalityResult)

    def test_n_qubits(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = universality_analysis(K, omega)
        assert result.n_qubits == 4

    def test_eta_type(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = universality_analysis(K, omega)
        assert result.eta_exponent is None or isinstance(result.eta_exponent, float)

    def test_scpn_universality(self):
        """Record universality check at SCPN defaults."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = universality_analysis(K, omega)
        print(f"\n  η exponent: {result.eta_exponent}")
        print(f"  Stiffness ratio: {result.stiffness_ratio:.4f}")
        print(f"  NK deviation: {result.nk_deviation:.4f}")
        print(f"  BKT consistent: {result.is_bkt_consistent}")
        assert isinstance(result.is_bkt_consistent, bool)
