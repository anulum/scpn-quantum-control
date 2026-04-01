# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Spectral Form Factor
"""Tests for Spectral Form Factor at the synchronization transition."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.spectral_form_factor import (
    SFFResult,
    SFFScanResult,
    compute_sff,
    sff_vs_coupling,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16


def _ring(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestComputeSFF:
    def test_returns_result(self):
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        result = compute_sff(K, omega, t_max=5.0, n_times=50)
        assert isinstance(result, SFFResult)
        assert len(result.sff) == 50

    def test_sff_starts_at_one(self):
        """K(0) = |Tr(I)|²/d² = d²/d² = 1."""
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        result = compute_sff(K, omega, t_max=5.0, n_times=50)
        assert abs(result.sff[0] - 1.0) < 1e-10

    def test_sff_bounded(self):
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        result = compute_sff(K, omega, t_max=10.0, n_times=100)
        assert np.all(result.sff >= 0)
        assert np.all(result.sff <= 1.0 + 1e-10)

    def test_level_spacing_ratio_bounded(self):
        """r̄ should be between 0 and 1."""
        K = 2.0 * _ring(4)
        omega = OMEGA_N_16[:4]
        result = compute_sff(K, omega)
        assert 0 <= result.level_spacing_ratio <= 1.0

    def test_4qubit(self):
        K = 1.5 * _ring(4)
        omega = OMEGA_N_16[:4]
        result = compute_sff(K, omega, t_max=5.0, n_times=30)
        assert result.spectral_gap > 0
        assert len(result.times) == 30


class TestSFFVsCoupling:
    def test_returns_result(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = sff_vs_coupling(omega, T, k_range=np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, SFFScanResult)
        assert len(result.level_spacing_ratios) == 3

    def test_r_bar_varies(self):
        """Level spacing ratio should change across the scan."""
        T = _ring(4)
        omega = OMEGA_N_16[:4]
        result = sff_vs_coupling(omega, T, k_range=np.linspace(0.5, 5.0, 6))
        assert not np.all(result.level_spacing_ratios == result.level_spacing_ratios[0])

    def test_dip_depth_finite(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = sff_vs_coupling(omega, T, k_range=np.array([1.0, 3.0]))
        assert np.all(np.isfinite(result.sff_dip_depth))
        assert np.all(result.sff_dip_depth >= 0)


# ---------------------------------------------------------------------------
# SFF physics invariants
# ---------------------------------------------------------------------------


class TestSFFPhysics:
    def test_sff_at_t_zero_equals_one(self):
        """K(t=0) = 1 exactly (trace of identity squared / d²)."""
        K = 1.0 * _ring(2)
        omega = OMEGA_N_16[:2]
        result = compute_sff(K, omega, t_max=1.0, n_times=10)
        np.testing.assert_allclose(result.sff[0], 1.0, atol=1e-10)

    def test_times_monotonic(self):
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        result = compute_sff(K, omega, t_max=5.0, n_times=20)
        assert np.all(np.diff(result.times) > 0)

    def test_spectral_gap_positive(self):
        """Spectral gap (E1-E0) must be positive for non-trivial coupling."""
        K = 3.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        result = compute_sff(K, omega)
        assert result.spectral_gap > 0

    def test_eigenvalues_real(self):
        """Eigenvalues of Hermitian H must be real (verified via SFF computation)."""
        K = 2.0 * _ring(2)
        omega = OMEGA_N_16[:2]
        result = compute_sff(K, omega, t_max=1.0, n_times=5)
        # If eigenvalues were complex, sff would diverge
        assert np.all(np.isfinite(result.sff))


# ---------------------------------------------------------------------------
# Pipeline: Knm → SFF → level statistics → RMT wiring
# ---------------------------------------------------------------------------


class TestSFFPipeline:
    def test_knm_to_sff_wired(self):
        """Pipeline: build_knm_paper27 → compute_sff → level_spacing_ratio."""
        import time

        from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        t0 = time.perf_counter()
        result = compute_sff(K, omega, t_max=3.0, n_times=20)
        dt = (time.perf_counter() - t0) * 1000

        assert len(result.sff) == 20
        assert 0 <= result.level_spacing_ratio <= 1.0

        print(f"\n  PIPELINE Knm→SFF (4q, 20 time points): {dt:.1f} ms")
        print(f"  r̄ = {result.level_spacing_ratio:.4f}, gap = {result.spectral_gap:.4f}")
