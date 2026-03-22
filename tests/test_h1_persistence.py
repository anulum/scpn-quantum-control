# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for H1 persistence at BKT transition."""

from __future__ import annotations

from scpn_quantum_control.analysis.h1_persistence import (
    H1PersistenceResult,
    scan_h1_persistence,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16


class TestScanH1Persistence:
    def test_returns_result(self):
        omega = OMEGA_N_16[:4]
        result = scan_h1_persistence(omega, n_points=8)
        assert isinstance(result, H1PersistenceResult)

    def test_k_critical_positive(self):
        omega = OMEGA_N_16[:4]
        result = scan_h1_persistence(omega, n_points=8)
        assert result.k_critical > 0

    def test_p_h1_bounded(self):
        omega = OMEGA_N_16[:4]
        result = scan_h1_persistence(omega, n_points=8)
        assert 0 <= result.p_h1_at_critical <= 1.0

    def test_arrays_length(self):
        omega = OMEGA_N_16[:4]
        result = scan_h1_persistence(omega, n_points=10)
        assert len(result.k_values) == 10
        assert len(result.vortex_densities) == 10
        assert len(result.derivative) == 10

    def test_deviation_non_negative(self):
        omega = OMEGA_N_16[:4]
        result = scan_h1_persistence(omega, n_points=8)
        assert result.deviation_from_target >= 0

    def test_gap3_measurement(self):
        """Record H1 persistence at BKT — Gap 3 data."""
        omega = OMEGA_N_16[:4]
        result = scan_h1_persistence(omega, n_points=15)
        print("\n  H1 persistence (4 osc):")
        print(f"  K_critical = {result.k_critical:.4f}")
        print(f"  p_h1 at K_c = {result.p_h1_at_critical:.4f}")
        print(f"  |p_h1 - 0.72| = {result.deviation_from_target:.4f}")
        assert isinstance(result.p_h1_at_critical, float)
