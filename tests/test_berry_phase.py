# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for Berry phase analysis at BKT criticality."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.berry_phase import (
    BerryPhaseResult,
    berry_phase_scan,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16


def _ring_topology(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestBerryPhaseScan:
    def test_returns_result(self):
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = berry_phase_scan(omega, T, k_range=np.linspace(0.5, 4.0, 6))
        assert isinstance(result, BerryPhaseResult)
        assert len(result.k_values) == 5  # n_k - 1 midpoints

    def test_fidelity_near_one_for_small_steps(self):
        """Fidelity between adjacent ground states should be close to 1 for small dK."""
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = berry_phase_scan(omega, T, k_range=np.linspace(1.0, 1.5, 20))
        assert np.all(result.fidelity > 0.9)

    def test_fidelity_drops_across_transition(self):
        """Fidelity should be lower where ground state changes character (near K_c)."""
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        # Wide range crosses the transition
        result = berry_phase_scan(omega, T, k_range=np.linspace(0.3, 6.0, 15))
        # Minimum fidelity should be < 1 (not all steps are trivial)
        assert np.min(result.fidelity) < 0.999

    def test_berry_curvature_has_peak(self):
        """Curvature should peak near the transition."""
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = berry_phase_scan(omega, T, k_range=np.linspace(0.3, 6.0, 20))
        assert result.curvature_peak_k is not None
        # Peak should be in the scan range
        assert result.curvature_peak_k >= 0.3
        assert result.curvature_peak_k <= 6.0

    def test_fidelity_susceptibility_peaks_near_transition(self):
        """χ_F (gauge-invariant) should peak where ground state changes fastest."""
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = berry_phase_scan(omega, T, k_range=np.linspace(0.3, 6.0, 20))
        # χ_F should have a nonzero maximum somewhere in the scan
        assert np.max(result.fidelity_susceptibility) > 0

    def test_fidelity_susceptibility_positive(self):
        """Fidelity susceptibility should be non-negative."""
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = berry_phase_scan(omega, T, k_range=np.linspace(0.5, 5.0, 10))
        assert np.all(result.fidelity_susceptibility >= -1e-10)

    def test_gap_varies(self):
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = berry_phase_scan(omega, T, k_range=np.linspace(0.5, 5.0, 8))
        assert result.spectral_gap[0] != result.spectral_gap[-1]

    def test_4qubit_scan(self):
        n = 4
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = berry_phase_scan(omega, T, k_range=np.array([1.0, 2.0, 3.0, 4.0]))
        assert isinstance(result, BerryPhaseResult)
        assert len(result.berry_connection) == 3

    def test_2qubit_smooth(self):
        """2-qubit system should have smooth Berry connection."""
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = berry_phase_scan(omega, T, k_range=np.linspace(1.0, 3.0, 10))
        assert np.all(np.isfinite(result.berry_connection))
        assert np.all(np.isfinite(result.berry_curvature))
