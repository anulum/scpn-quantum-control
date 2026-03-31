# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Avqds
"""Tests for AVQDS adaptive variational dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase.avqds import (
    AVQDSResult,
    avqds_simulate,
)


class TestAVQDS:
    def test_returns_result(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = avqds_simulate(K, omega, t_total=0.1, n_steps=3, seed=42)
        assert isinstance(result, AVQDSResult)

    def test_times_shape(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = avqds_simulate(K, omega, t_total=0.5, n_steps=5, seed=42)
        assert result.times.shape == (6,)
        assert result.energies.shape == (6,)
        assert result.fidelities.shape == (6,)

    def test_initial_fidelity_one(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = avqds_simulate(K, omega, t_total=0.1, n_steps=3, seed=42)
        assert result.fidelities[0] == pytest.approx(1.0, abs=1e-6)

    def test_energy_finite(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = avqds_simulate(K, omega, t_total=0.1, n_steps=3, seed=42)
        assert np.all(np.isfinite(result.energies))

    def test_parameters_history(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = avqds_simulate(K, omega, t_total=0.1, n_steps=3, seed=42)
        assert len(result.parameters_history) >= 1

    def test_n_params_positive(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = avqds_simulate(K, omega, t_total=0.1, n_steps=2, seed=42)
        assert result.n_params > 0

    def test_3_oscillators(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = avqds_simulate(K, omega, t_total=0.1, n_steps=2, seed=42)
        assert result.n_params > 0

    def test_short_time_high_fidelity(self):
        """Very short evolution should maintain high fidelity."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = avqds_simulate(K, omega, t_total=0.01, n_steps=2, seed=42)
        assert result.final_fidelity > 0.9


def test_avqds_finite_energies():
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = avqds_simulate(K, omega, t_total=0.1, n_steps=3, seed=42)
    assert np.all(np.isfinite(result.energies))


def test_avqds_3q():
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    result = avqds_simulate(K, omega, t_total=0.1, n_steps=2, seed=0)
    assert result.n_params > 0


def test_avqds_fidelity_bounded():
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = avqds_simulate(K, omega, t_total=0.5, n_steps=5, seed=42)
    assert 0 <= result.final_fidelity <= 1.0 + 1e-10
