# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for Floquet-Kuramoto time crystal module."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16
from scpn_quantum_control.phase.floquet_kuramoto import (
    FloquetResult,
    floquet_evolve,
    scan_drive_amplitude,
)


def _ring_topology(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestFloquetEvolve:
    def test_returns_result(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = floquet_evolve(
            T,
            omega,
            K_base=1.0,
            drive_amplitude=0.5,
            drive_frequency=2.0,
            n_periods=3,
            steps_per_period=8,
        )
        assert isinstance(result, FloquetResult)
        assert len(result.times) == 3 * 8 + 1
        assert len(result.R_values) == len(result.times)

    def test_R_bounded(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = floquet_evolve(
            T,
            omega,
            K_base=2.0,
            drive_amplitude=0.3,
            drive_frequency=1.5,
            n_periods=4,
            steps_per_period=10,
        )
        assert np.all(result.R_values >= 0)
        assert np.all(result.R_values <= 1.0 + 1e-10)

    def test_drive_signal_oscillates(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = floquet_evolve(
            T,
            omega,
            K_base=1.0,
            drive_amplitude=0.5,
            drive_frequency=2.0,
            n_periods=5,
            steps_per_period=10,
        )
        # Drive should vary between K_base*(1-amp) and K_base*(1+amp)
        assert np.min(result.drive_signal) < 1.0
        assert np.max(result.drive_signal) > 1.0

    def test_zero_amplitude_no_oscillation(self):
        """No drive → R should still evolve (unitary dynamics) but no drive signal variation."""
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = floquet_evolve(
            T,
            omega,
            K_base=1.0,
            drive_amplitude=0.0,
            drive_frequency=2.0,
            n_periods=3,
            steps_per_period=8,
        )
        # Drive signal should be constant
        assert np.std(result.drive_signal) < 1e-10

    def test_subharmonic_ratio_finite(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = floquet_evolve(
            T,
            omega,
            K_base=2.0,
            drive_amplitude=1.0,
            drive_frequency=3.0,
            n_periods=6,
            steps_per_period=12,
        )
        assert np.isfinite(result.subharmonic_ratio)

    def test_3qubit_evolves(self):
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = floquet_evolve(
            T,
            omega,
            K_base=1.5,
            drive_amplitude=0.5,
            drive_frequency=2.0,
            n_periods=3,
            steps_per_period=8,
        )
        assert len(result.R_values) == 3 * 8 + 1
        assert np.all(np.isfinite(result.R_values))


class TestScanDriveAmplitude:
    def test_returns_dict(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = scan_drive_amplitude(
            T,
            omega,
            K_base=1.0,
            drive_frequency=2.0,
            amplitudes=np.array([0.3, 0.8]),
            n_periods=3,
            steps_per_period=8,
        )
        assert "amplitude" in result
        assert "subharmonic_ratio" in result
        assert len(result["amplitude"]) == 2

    def test_all_values_finite(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = scan_drive_amplitude(
            T,
            omega,
            K_base=1.5,
            drive_frequency=2.0,
            amplitudes=np.array([0.2, 0.5, 1.0]),
            n_periods=4,
            steps_per_period=10,
        )
        for key in result:
            assert all(np.isfinite(v) for v in result[key])
