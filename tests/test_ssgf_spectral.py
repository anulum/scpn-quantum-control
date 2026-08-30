# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — SSGF Spectral Bridge Tests
"""Tests for ssgf.quantum_spectral."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.ssgf.quantum_spectral import (
    SpectralBridgeResult,
    entrainment_criterion,
    laplacian_spectrum,
    qpe_resource_estimate,
    spectral_bridge_analysis,
)


def _small_system() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return build_knm_paper27(L=3), OMEGA_N_16[:3]


class TestLaplacianSpectrum:
    """Exercise weighted-Laplacian spectrum construction."""

    def test_first_eigenvalue_zero(self) -> None:
        """The graph Laplacian has a zero first eigenvalue."""
        K, _ = _small_system()
        spec = laplacian_spectrum(K)
        assert abs(spec[0]) < 1e-10

    def test_sorted_ascending(self) -> None:
        """The returned Laplacian spectrum is sorted ascending."""
        K, _ = _small_system()
        spec = laplacian_spectrum(K)
        assert all(spec[i] <= spec[i + 1] + 1e-12 for i in range(len(spec) - 1))

    def test_length_matches_n(self) -> None:
        """The spectrum contains one eigenvalue per oscillator."""
        K, _ = _small_system()
        spec = laplacian_spectrum(K)
        assert len(spec) == 3


class TestEntrainmentCriterion:
    """Exercise the Fiedler-versus-frequency-spread criterion."""

    def test_strong_coupling_stable(self) -> None:
        """Strong all-to-all coupling satisfies the stability criterion."""
        K = np.array([[0, 5.0, 5.0], [5.0, 0, 5.0], [5.0, 5.0, 0]])
        omega = OMEGA_N_16[:3]
        stable, margin = entrainment_criterion(K, omega)
        assert stable is True
        assert margin > 0

    def test_returns_tuple(self) -> None:
        """The criterion returns its decision and margin pair."""
        K, omega = _small_system()
        result = entrainment_criterion(K, omega)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestQPEResourceEstimate:
    """Exercise coarse asymptotic QPE resource estimates."""

    def test_returns_positive(self) -> None:
        """Default precision yields positive bit and depth estimates."""
        K, _ = _small_system()
        n_bits, depth = qpe_resource_estimate(K)
        assert n_bits >= 1
        assert depth > 0

    def test_higher_precision_more_bits(self) -> None:
        """Higher requested precision does not reduce the bit estimate."""
        K, _ = _small_system()
        n1, _ = qpe_resource_estimate(K, epsilon=0.1)
        n2, _ = qpe_resource_estimate(K, epsilon=0.001)
        assert n2 >= n1


class TestSpectralBridgeAnalysis:
    """Exercise the combined spectral bridge analysis."""

    def test_returns_result(self) -> None:
        """The analysis returns connectivity, spectrum, and QPE estimates."""
        K, omega = _small_system()
        result = spectral_bridge_analysis(K, omega)
        assert isinstance(result, SpectralBridgeResult)
        assert result.fiedler_value >= 0
        assert result.qpe_bits_needed >= 1
        assert len(result.laplacian_spectrum) == 3


# ---------------------------------------------------------------------------
# Spectral graph theory invariants
# ---------------------------------------------------------------------------


class TestSpectralInvariants:
    """Exercise spectral graph invariants."""

    def test_fiedler_positive_for_connected(self) -> None:
        """Fiedler value (λ_2) > 0 iff graph is connected."""
        K, omega = _small_system()
        result = spectral_bridge_analysis(K, omega)
        # paper27 coupling is fully connected → fiedler > 0
        assert result.fiedler_value > 0

    def test_laplacian_eigenvalues_nonnegative(self) -> None:
        """All Laplacian eigenvalues must be ≥ 0 (positive semi-definite)."""
        K, _ = _small_system()
        spec = laplacian_spectrum(K)
        assert np.all(np.array(spec) >= -1e-10)

    def test_laplacian_trace_positive(self) -> None:
        """tr(L) > 0 for non-trivial coupling (sum of eigenvalues)."""
        K, _ = _small_system()
        spec = laplacian_spectrum(K)
        assert np.sum(spec) > 0

    def test_zero_coupling_fiedler_zero(self) -> None:
        """Disconnected graph → fiedler = 0."""
        K_zero = np.zeros((3, 3))
        omega = OMEGA_N_16[:3]
        result = spectral_bridge_analysis(K_zero, omega)
        assert result.fiedler_value < 1e-10


# ---------------------------------------------------------------------------
# Pipeline: Knm → spectral analysis → QPE estimate → wired
# ---------------------------------------------------------------------------


class TestSpectralPipeline:
    """Exercise the K_nm-to-spectral diagnostic pipeline."""

    def test_pipeline_knm_to_qpe(self) -> None:
        """Full pipeline: build_knm → spectral analysis → QPE resource estimate.

        The result supplies diagnostic asymptotic parameters, not compiled
        hardware-resource evidence.
        """
        import time

        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        t0 = time.perf_counter()
        result = spectral_bridge_analysis(K, omega)
        dt = (time.perf_counter() - t0) * 1000

        assert result.fiedler_value > 0
        assert result.qpe_bits_needed >= 1
        print(f"\n  PIPELINE Knm→Spectral→QPE (4 osc): {dt:.1f} ms")
        print(f"  Fiedler = {result.fiedler_value:.4f}")
        print(f"  QPE bits = {result.qpe_bits_needed}")
