# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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
