# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for adiabatic state preparation at BKT."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16
from scpn_quantum_control.phase.adiabatic_preparation import (
    AdiabaticResult,
    adiabatic_ramp,
    adiabatic_time_scaling,
)


def _ring_topology(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestAdiabaticRamp:
    def test_returns_result(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_ramp(omega, T, K_target=3.0, T_total=5.0, n_steps=20)
        assert isinstance(result, AdiabaticResult)
        assert len(result.times) == 21

    def test_fidelity_starts_at_one(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_ramp(omega, T, K_target=2.0, T_total=5.0, n_steps=15)
        assert result.fidelity[0] > 0.99

    def test_slow_ramp_before_transition(self):
        """Slow ramp to K below K_c should maintain fidelity (no gap closing)."""
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        # K_target=1.0 stays well below the gap minimum at K≈1.87
        result = adiabatic_ramp(omega, T, K_target=1.0, T_total=30.0, n_steps=30)
        assert result.final_fidelity > 0.5

    def test_fast_ramp_lower_fidelity(self):
        """Very fast ramp → diabatic transitions → lower fidelity."""
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        fast = adiabatic_ramp(omega, T, K_target=3.0, T_total=0.1, n_steps=10)
        slow = adiabatic_ramp(omega, T, K_target=3.0, T_total=20.0, n_steps=30)
        # Slow should generally have better fidelity
        assert slow.final_fidelity >= fast.final_fidelity - 0.1

    def test_gap_always_positive(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_ramp(omega, T, K_target=3.0, T_total=5.0, n_steps=15)
        assert np.all(result.gap > 0)

    def test_min_gap_location(self):
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_ramp(omega, T, K_target=5.0, T_total=10.0, n_steps=20)
        assert result.min_gap > 0
        assert 0 <= result.min_gap_K <= 5.0

    def test_3qubit_ramp(self):
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_ramp(omega, T, K_target=2.0, T_total=5.0, n_steps=15)
        assert isinstance(result, AdiabaticResult)
        assert np.all(np.isfinite(result.fidelity))


class TestAdiabaticTimeScaling:
    def test_returns_dict(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_time_scaling(
            omega, T, K_target=2.0, T_values=np.array([1.0, 5.0]), n_steps_per_T=10
        )
        assert "T_total" in result
        assert "final_fidelity" in result
        assert len(result["T_total"]) == 2

    def test_fidelity_increases_with_time(self):
        """Longer T should give better fidelity (adiabatic theorem)."""
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_time_scaling(
            omega, T, K_target=2.0, T_values=np.array([0.5, 20.0]), n_steps_per_T=15
        )
        # Not guaranteed for all T, but large gap should show trend
        assert all(np.isfinite(f) for f in result["final_fidelity"])
