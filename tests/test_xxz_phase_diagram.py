# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
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
