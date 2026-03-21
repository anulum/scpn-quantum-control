# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for power grid synchronisation benchmark."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.applications.power_grid import (
    IEEE_5BUS_SUSCEPTANCE,
    PowerGridBenchmarkResult,
    ieee_5bus_coupling_matrix,
    power_grid_benchmark,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestIEEE5BusData:
    def test_susceptance_symmetric(self):
        np.testing.assert_allclose(IEEE_5BUS_SUSCEPTANCE, IEEE_5BUS_SUSCEPTANCE.T, atol=1e-10)

    def test_susceptance_5x5(self):
        assert IEEE_5BUS_SUSCEPTANCE.shape == (5, 5)

    def test_susceptance_non_negative(self):
        assert np.all(IEEE_5BUS_SUSCEPTANCE >= 0)


class TestIEEE5BusCouplingMatrix:
    def test_shape(self):
        K, omega = ieee_5bus_coupling_matrix()
        assert K.shape == (5, 5)
        assert omega.shape == (5,)

    def test_symmetric(self):
        K, _ = ieee_5bus_coupling_matrix()
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_non_negative(self):
        K, _ = ieee_5bus_coupling_matrix()
        assert np.all(K >= 0)

    def test_zero_diagonal(self):
        K, _ = ieee_5bus_coupling_matrix()
        np.testing.assert_allclose(np.diag(K), 0.0, atol=1e-10)

    def test_sparse_topology(self):
        """IEEE 5-bus is NOT fully connected — has zeros."""
        K, _ = ieee_5bus_coupling_matrix()
        assert np.any(K == 0)


class TestPowerGridBenchmark:
    def test_returns_result(self):
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        result = power_grid_benchmark(K, omega)
        assert isinstance(result, PowerGridBenchmarkResult)

    def test_n_generators(self):
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        result = power_grid_benchmark(K, omega)
        assert result.n_generators == 5

    def test_correlation_bounded(self):
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        result = power_grid_benchmark(K, omega)
        assert -1 <= result.topology_correlation <= 1

    def test_coupling_ratio_positive(self):
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        result = power_grid_benchmark(K, omega)
        assert result.coupling_ratio > 0

    def test_summary_string(self):
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        result = power_grid_benchmark(K, omega)
        assert "SCPN vs IEEE-5bus" in result.summary

    def test_fewer_oscillators(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = power_grid_benchmark(K, omega)
        assert result.n_generators == 3

    def test_scpn_vs_grid(self):
        """Record SCPN vs IEEE-5bus comparison — Gap 1 data."""
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        result = power_grid_benchmark(K, omega)
        print(f"\n  {result.summary}")
        assert isinstance(result.topology_correlation, float)
