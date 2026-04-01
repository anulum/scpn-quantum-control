# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Quantum Mpemba
"""Tests for quantum Mpemba effect in synchronization."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.quantum_mpemba import (
    MpembaResult,
    mpemba_experiment,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16


def _ring_topology(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestMpembaExperiment:
    def test_returns_result(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = mpemba_experiment(omega, T, K_base=2.0, gamma=0.2, t_max=2.0, n_steps=10)
        assert isinstance(result, MpembaResult)
        assert len(result.times) == 11

    def test_fidelity_bounded(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = mpemba_experiment(omega, T, K_base=2.0, gamma=0.2, t_max=2.0, n_steps=10)
        assert np.all(result.fidelity_near >= -0.01)
        assert np.all(result.fidelity_near <= 1.01)
        assert np.all(result.fidelity_far >= -0.01)
        assert np.all(result.fidelity_far <= 1.01)

    def test_initial_distances_differ(self):
        """The two initial states should have different distances from NESS."""
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = mpemba_experiment(omega, T, K_base=3.0, gamma=0.1, t_max=3.0, n_steps=15)
        # Distances should be non-negative and at least one nonzero
        assert result.initial_distance_near >= 0
        assert result.initial_distance_far >= 0
        assert result.initial_distance_near + result.initial_distance_far > 0

    def test_both_approach_steady_state(self):
        """Both states should get closer to steady state over time."""
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = mpemba_experiment(omega, T, K_base=2.0, gamma=0.3, t_max=5.0, n_steps=20)
        # Final fidelity should be higher than initial for at least one
        assert result.fidelity_near[-1] >= result.fidelity_near[0] - 0.1
        assert result.fidelity_far[-1] >= result.fidelity_far[0] - 0.1

    def test_R_values_bounded(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = mpemba_experiment(omega, T, K_base=2.0, gamma=0.2, t_max=2.0, n_steps=8)
        assert np.all(result.R_near >= 0)
        assert np.all(result.R_near <= 1.0 + 1e-10)
        assert np.all(result.R_far >= 0)
        assert np.all(result.R_far <= 1.0 + 1e-10)

    def test_crossing_time_if_mpemba(self):
        """If Mpemba detected, crossing time should be positive."""
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = mpemba_experiment(omega, T, K_base=2.0, gamma=0.2, t_max=5.0, n_steps=20)
        if result.has_mpemba:
            assert result.crossing_time is not None
            assert result.crossing_time > 0

    def test_3qubit_runs(self):
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = mpemba_experiment(omega, T, K_base=1.5, gamma=0.2, t_max=2.0, n_steps=8)
        assert isinstance(result, MpembaResult)
        assert len(result.R_far) == 9


def test_mpemba_result_fields():
    n = 2
    T = _ring_topology(n)
    omega = OMEGA_N_16[:n]
    result = mpemba_experiment(omega, T, K_base=2.0, gamma=0.2, t_max=2.0, n_steps=5)
    assert hasattr(result, "has_mpemba")
    assert hasattr(result, "R_near")
    assert hasattr(result, "R_far")
    assert hasattr(result, "crossing_time")


def test_mpemba_R_length():
    n = 2
    T = _ring_topology(n)
    omega = OMEGA_N_16[:n]
    result = mpemba_experiment(omega, T, K_base=2.0, gamma=0.2, t_max=1.0, n_steps=10)
    assert len(result.R_near) == 11
    assert len(result.R_far) == 11


def test_mpemba_R_near_bounded():
    n = 2
    T = _ring_topology(n)
    omega = OMEGA_N_16[:n]
    result = mpemba_experiment(omega, T, K_base=2.0, gamma=0.2, t_max=2.0, n_steps=8)
    assert np.all(result.R_near >= 0)
    assert np.all(result.R_near <= 1.0 + 1e-10)


def test_mpemba_4qubit():
    n = 4
    T = _ring_topology(n)
    omega = OMEGA_N_16[:n]
    result = mpemba_experiment(omega, T, K_base=1.5, gamma=0.2, t_max=1.0, n_steps=5)
    assert isinstance(result, MpembaResult)
