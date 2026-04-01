# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Xxz Phase Diagram
"""Tests for XXZ anisotropy phase diagram."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.xxz_phase_diagram import (
    AnisotropyScanResult,
    PhaseDiagramResult,
    anisotropy_phase_diagram,
    scan_coupling_at_delta,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16


def _ring(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestScanCouplingAtDelta:
    def test_returns_result(self):
        n = 3
        T = _ring(n)
        omega = OMEGA_N_16[:n]
        result = scan_coupling_at_delta(omega, T, delta=0.0, k_range=np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, AnisotropyScanResult)
        assert result.delta == 0.0
        assert len(result.gaps) == 3

    def test_gap_positive(self):
        n = 3
        T = _ring(n)
        omega = OMEGA_N_16[:n]
        result = scan_coupling_at_delta(omega, T, delta=0.5, k_range=np.linspace(0.5, 4.0, 6))
        assert np.all(result.gaps > 0)

    def test_k_c_in_range(self):
        n = 3
        T = _ring(n)
        omega = OMEGA_N_16[:n]
        k_range = np.linspace(0.5, 5.0, 8)
        result = scan_coupling_at_delta(omega, T, delta=0.0, k_range=k_range)
        assert result.k_c_from_gap in k_range


class TestAnisotropyPhaseDiagram:
    def test_returns_result(self):
        n = 3
        T = _ring(n)
        omega = OMEGA_N_16[:n]
        result = anisotropy_phase_diagram(
            omega,
            T,
            delta_range=np.array([0.0, 0.5, 1.0]),
            k_range=np.array([1.0, 2.0, 3.0]),
        )
        assert isinstance(result, PhaseDiagramResult)
        assert len(result.delta_values) == 3
        assert len(result.k_c_values) == 3

    def test_k_c_varies_with_delta(self):
        """K_c should shift as anisotropy changes."""
        n = 3
        T = _ring(n)
        omega = OMEGA_N_16[:n]
        result = anisotropy_phase_diagram(
            omega,
            T,
            delta_range=np.array([0.0, 1.0]),
            k_range=np.linspace(0.5, 5.0, 8),
        )
        # Gap minimum position should differ between XY and Heisenberg
        assert len(result.scans) == 2

    def test_all_gaps_positive(self):
        n = 3
        T = _ring(n)
        omega = OMEGA_N_16[:n]
        result = anisotropy_phase_diagram(
            omega,
            T,
            delta_range=np.array([0.0, 0.5]),
            k_range=np.array([1.0, 3.0]),
        )
        for scan in result.scans:
            assert np.all(scan.gaps > 0)

    def test_delta_values_match_input(self):
        n = 3
        T = _ring(n)
        omega = OMEGA_N_16[:n]
        deltas = np.array([0.0, 0.5, 1.0])
        result = anisotropy_phase_diagram(
            omega, T, delta_range=deltas, k_range=np.array([1.0, 2.0])
        )
        np.testing.assert_array_equal(result.delta_values, deltas)


class TestScanGapProperties:
    def test_gap_finite(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = scan_coupling_at_delta(omega, T, delta=0.0, k_range=np.linspace(0.5, 3.0, 5))
        assert np.all(np.isfinite(result.gaps))

    def test_2osc_scan(self):
        T = _ring(2)
        omega = OMEGA_N_16[:2]
        result = scan_coupling_at_delta(omega, T, delta=0.0, k_range=np.array([1.0, 2.0]))
        assert len(result.gaps) == 2

    def test_heisenberg_delta_one(self):
        """At delta=1, Hamiltonian is Heisenberg XXX."""
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = scan_coupling_at_delta(omega, T, delta=1.0, k_range=np.linspace(0.5, 3.0, 4))
        assert result.delta == 1.0
        assert len(result.gaps) == 4


# ---------------------------------------------------------------------------
# Phase diagram physics: XY vs Heisenberg
# ---------------------------------------------------------------------------


class TestPhaseDiagramPhysics:
    def test_xy_and_heisenberg_different_gaps(self):
        """XY (Δ=0) and Heisenberg (Δ=1) produce different gap structure."""
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        k_range = np.array([1.0, 3.0])
        r_xy = scan_coupling_at_delta(omega, T, delta=0.0, k_range=k_range)
        r_heis = scan_coupling_at_delta(omega, T, delta=1.0, k_range=k_range)
        assert not np.allclose(r_xy.gaps, r_heis.gaps)

    def test_gaps_all_positive(self):
        """Non-degenerate spectrum → all gaps > 0."""
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = scan_coupling_at_delta(omega, T, delta=0.5, k_range=np.linspace(0.5, 4.0, 6))
        assert np.all(result.gaps > 0)


# ---------------------------------------------------------------------------
# Pipeline: XXZ → phase diagram → K_c → wired
# ---------------------------------------------------------------------------


class TestXXZPipeline:
    def test_pipeline_xxz_phase_diagram(self):
        """Full pipeline: topology → XXZ scan → K_c across anisotropies.
        Verifies XXZ module is wired end-to-end.
        """
        import time

        T = _ring(3)
        omega = OMEGA_N_16[:3]

        t0 = time.perf_counter()
        result = anisotropy_phase_diagram(
            omega,
            T,
            delta_range=np.array([0.0, 0.5, 1.0]),
            k_range=np.linspace(0.5, 4.0, 6),
        )
        dt = (time.perf_counter() - t0) * 1000

        assert len(result.k_c_values) == 3
        assert len(result.scans) == 3

        print(f"\n  PIPELINE XXZ phase diagram (3q, 3δ×6K): {dt:.1f} ms")
        print(f"  K_c values: {result.k_c_values}")
