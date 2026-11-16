# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Spectral Bridge
"""Tests for spectral bridge (Fiedler via QPE)."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.ssgf.quantum_spectral import (
    SpectralBridgeResult,
    entrainment_criterion,
    laplacian_spectrum,
    qpe_resource_estimate,
    spectral_bridge_analysis,
    spectral_bridge_vs_coupling,
)


class TestLaplacianSpectrum:
    """The Laplacian spectrum of the coupling graph."""

    def test_first_eigenvalue_zero(self) -> None:
        """The first eigenvalue is zero."""
        K = build_knm_paper27(L=4)
        spec = laplacian_spectrum(K)
        assert spec[0] == pytest.approx(0.0, abs=1e-10)

    def test_all_non_negative(self) -> None:
        """Every eigenvalue is non-negative."""
        K = build_knm_paper27(L=4)
        spec = laplacian_spectrum(K)
        assert np.all(spec >= -1e-10)

    def test_sorted(self) -> None:
        """Eigenvalues are in ascending order."""
        K = build_knm_paper27(L=4)
        spec = laplacian_spectrum(K)
        for i in range(len(spec) - 1):
            assert spec[i] <= spec[i + 1] + 1e-10

    def test_length(self) -> None:
        """There is one eigenvalue per node."""
        K = build_knm_paper27(L=6)
        spec = laplacian_spectrum(K)
        assert len(spec) == 6


class TestEntrainmentCriterion:
    """The entrainment stability criterion."""

    def test_strong_coupling_stable(self) -> None:
        """Strong coupling is stable with a positive margin."""
        K = build_knm_paper27(L=4, K_base=10.0)
        omega = OMEGA_N_16[:4]
        stable, margin = entrainment_criterion(K, omega)
        assert stable
        assert margin > 0

    def test_weak_coupling_unstable(self) -> None:
        """Weak coupling is unstable with a negative margin."""
        K = build_knm_paper27(L=4, K_base=0.001)
        omega = OMEGA_N_16[:4]
        stable, margin = entrainment_criterion(K, omega)
        assert not stable
        assert margin < 0


class TestQPEResourceEstimate:
    """Phase-estimation resource estimates."""

    def test_bits_increase_with_precision(self) -> None:
        """Higher precision needs more bits."""
        K = build_knm_paper27(L=4)
        b1, _ = qpe_resource_estimate(K, epsilon=0.1)
        b2, _ = qpe_resource_estimate(K, epsilon=0.001)
        assert b2 > b1

    def test_depth_positive(self) -> None:
        """The depth is positive."""
        K = build_knm_paper27(L=4)
        _, depth = qpe_resource_estimate(K)
        assert depth > 0


class TestSpectralBridgeAnalysis:
    """The spectral bridge analysis."""

    def test_returns_result(self) -> None:
        """The analysis returns a result record."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = spectral_bridge_analysis(K, omega)
        assert isinstance(result, SpectralBridgeResult)

    def test_fiedler_positive(self) -> None:
        """The Fiedler value is positive."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = spectral_bridge_analysis(K, omega)
        assert result.fiedler_value > 0

    def test_spectrum_length(self) -> None:
        """The spectrum has one value per node."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = spectral_bridge_analysis(K, omega)
        assert len(result.laplacian_spectrum) == 4

    def test_scpn_spectral_bridge(self) -> None:
        """Record spectral bridge at SCPN defaults."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = spectral_bridge_analysis(K, omega)
        print("\n  Spectral bridge (4 osc):")
        print(f"  λ_2 = {result.fiedler_value:.6f}")
        print(f"  Δω = {result.frequency_spread:.4f}")
        print(f"  Stable: {result.entrainment_stable}")
        print(f"  Margin: {result.stability_margin:.4f}")
        print(f"  QPE bits: {result.qpe_bits_needed}")
        assert isinstance(result.fiedler_value, float)


class TestSpectralBridgeVsCoupling:
    """Scanning the spectral bridge across couplings."""

    def test_returns_keys(self) -> None:
        """The Fiedler series has one value per coupling."""
        omega = OMEGA_N_16[:4]
        k_vals = np.array([0.1, 1.0, 3.0])
        results = spectral_bridge_vs_coupling(omega, k_vals)
        assert "fiedler" in results
        assert len(results["fiedler"]) == 3

    def test_fiedler_increases(self) -> None:
        """The Fiedler value grows with the coupling."""
        omega = OMEGA_N_16[:4]
        k_vals = np.array([0.1, 1.0, 5.0])
        results = spectral_bridge_vs_coupling(omega, k_vals)
        assert results["fiedler"][2] > results["fiedler"][0]
