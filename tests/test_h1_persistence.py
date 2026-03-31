# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for H1 Persistence
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

    def test_3osc_scan(self):
        omega = OMEGA_N_16[:3]
        result = scan_h1_persistence(omega, n_points=6)
        assert isinstance(result, H1PersistenceResult)
        assert result.k_critical > 0

    def test_k_values_sorted(self):
        omega = OMEGA_N_16[:4]
        result = scan_h1_persistence(omega, n_points=10)
        import numpy as np

        assert np.all(np.diff(result.k_values) > 0)

    def test_vortex_densities_bounded(self):
        omega = OMEGA_N_16[:4]
        result = scan_h1_persistence(omega, n_points=8)
        for v in result.vortex_densities:
            assert 0.0 <= v <= 1.0 + 1e-10

    def test_derivative_length(self):
        omega = OMEGA_N_16[:4]
        result = scan_h1_persistence(omega, n_points=12)
        assert len(result.derivative) == 12


# ---------------------------------------------------------------------------
# Pipeline: Knm → H1 persistence → K_c → wired
# ---------------------------------------------------------------------------


class TestH1Pipeline:
    def test_pipeline_knm_to_h1(self):
        """Full pipeline: OMEGA → H1 persistence scan → K_critical.
        Verifies H1 persistence is wired and produces topological data.
        """
        import time

        omega = OMEGA_N_16[:4]

        t0 = time.perf_counter()
        result = scan_h1_persistence(omega, n_points=10)
        dt = (time.perf_counter() - t0) * 1000

        assert result.k_critical > 0
        assert 0 <= result.p_h1_at_critical <= 1.0

        print(f"\n  PIPELINE H1 persistence (4 osc, 10 pts): {dt:.1f} ms")
        print(f"  K_c={result.k_critical:.4f}, p_h1={result.p_h1_at_critical:.4f}")
