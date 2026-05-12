# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Power Grid
"""Tests for power grid synchronisation benchmark."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

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
    def test_builtin_reference_requires_explicit_opt_in(self):
        with pytest.raises(RuntimeError, match="allow_builtin_reference"):
            ieee_5bus_coupling_matrix()

    def test_shape(self):
        K, omega = ieee_5bus_coupling_matrix(allow_builtin_reference=True)
        assert K.shape == (5, 5)
        assert omega.shape == (5,)

    def test_symmetric(self):
        K, _ = ieee_5bus_coupling_matrix(allow_builtin_reference=True)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_non_negative(self):
        K, _ = ieee_5bus_coupling_matrix(allow_builtin_reference=True)
        assert np.all(K >= 0)

    def test_zero_diagonal(self):
        K, _ = ieee_5bus_coupling_matrix(allow_builtin_reference=True)
        np.testing.assert_allclose(np.diag(K), 0.0, atol=1e-10)

    def test_sparse_topology(self):
        """IEEE 5-bus is NOT fully connected — has zeros."""
        K, _ = ieee_5bus_coupling_matrix(allow_builtin_reference=True)
        assert np.any(K == 0)


class TestPowerGridBenchmark:
    def test_returns_result(self):
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        result = power_grid_benchmark(K, omega, allow_builtin_reference=True)
        assert isinstance(result, PowerGridBenchmarkResult)

    def test_n_generators(self):
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        result = power_grid_benchmark(K, omega, allow_builtin_reference=True)
        assert result.n_generators == 5

    def test_correlation_bounded(self):
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        result = power_grid_benchmark(K, omega, allow_builtin_reference=True)
        assert -1 <= result.topology_correlation <= 1

    def test_coupling_ratio_positive(self):
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        result = power_grid_benchmark(K, omega, allow_builtin_reference=True)
        assert result.coupling_ratio > 0

    def test_summary_string(self):
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        result = power_grid_benchmark(K, omega, allow_builtin_reference=True)
        assert "SCPN vs IEEE-5bus" in result.summary

    def test_fewer_oscillators(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = power_grid_benchmark(K, omega, allow_builtin_reference=True)
        assert result.n_generators == 3

    def test_constant_frequency_vector_returns_zero_correlation(self):
        K = build_knm_paper27(L=5)
        omega = np.ones(5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = power_grid_benchmark(K, omega, allow_builtin_reference=True)
        assert result.frequency_correlation == 0.0
        assert result.n_generators == 5
        assert "freq r=0.000" in result.summary

    def test_benchmark_refuses_implicit_builtin_reference(self):
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        with pytest.raises(RuntimeError, match="allow_builtin_reference"):
            power_grid_benchmark(K, omega)

    def test_result_labels_builtin_reference_source_mode(self):
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        result = power_grid_benchmark(K, omega, allow_builtin_reference=True)
        assert result.source_mode == "curated"
        assert result.publication_safe is True

    def test_result_labels_artifact_reference_source_mode(self):
        K_grid, omega_grid = ieee_5bus_coupling_matrix(allow_builtin_reference=True)
        result = power_grid_benchmark(
            K_grid,
            omega_grid,
            grid_coupling=K_grid,
            grid_frequencies=omega_grid,
            reference_source_mode="curated",
        )
        assert result.source_mode == "curated"
        assert result.publication_safe is True

    def test_reference_matrix_and_frequency_vector_must_be_supplied_together(self):
        K_grid, omega_grid = ieee_5bus_coupling_matrix(allow_builtin_reference=True)
        with pytest.raises(ValueError, match="grid_coupling and grid_frequencies"):
            power_grid_benchmark(K_grid, omega_grid, grid_coupling=K_grid)

    def test_scpn_vs_grid(self):
        """Record SCPN vs IEEE-5bus comparison — Gap 1 data."""
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        result = power_grid_benchmark(K, omega, allow_builtin_reference=True)
        print(f"\n  {result.summary}")
        assert isinstance(result.topology_correlation, float)

    def test_rejects_non_square_scpn_coupling(self):
        K = np.ones((2, 3))
        omega = np.ones(2)
        with pytest.raises(ValueError, match="K_scpn must be a square"):
            power_grid_benchmark(K, omega, allow_builtin_reference=True)

    def test_rejects_scpn_frequency_shape_mismatch(self):
        K = build_knm_paper27(L=4)
        omega = np.ones(3)
        with pytest.raises(ValueError, match="omega_scpn must match"):
            power_grid_benchmark(K, omega, allow_builtin_reference=True)

    def test_rejects_non_finite_scpn_coupling(self):
        K = build_knm_paper27(L=4)
        K[0, 1] = np.nan
        omega = OMEGA_N_16[:4]
        with pytest.raises(ValueError, match="K_scpn must contain only finite"):
            power_grid_benchmark(K, omega, allow_builtin_reference=True)

    def test_rejects_too_small_comparison_system(self):
        K = np.array([[0.0]])
        omega = np.array([0.0])
        with pytest.raises(ValueError, match="at least two coupled grid nodes"):
            power_grid_benchmark(K, omega, allow_builtin_reference=True)

    def test_rejects_grid_coupling_asymmetry(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        grid_K = np.array(
            [
                [0.0, 0.8, 0.1],
                [0.2, 0.0, 0.3],
                [0.1, 0.3, 0.0],
            ]
        )
        grid_omega = np.array([0.0, 0.02, -0.01])
        with pytest.raises(ValueError, match="grid_coupling must be symmetric"):
            power_grid_benchmark(K, omega, grid_coupling=grid_K, grid_frequencies=grid_omega)

    def test_rejects_negative_grid_coupling(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        grid_K = np.array(
            [
                [0.0, -0.1, 0.1],
                [-0.1, 0.0, 0.3],
                [0.1, 0.3, 0.0],
            ]
        )
        grid_omega = np.array([0.0, 0.02, -0.01])
        with pytest.raises(ValueError, match="grid_coupling values must be non-negative"):
            power_grid_benchmark(K, omega, grid_coupling=grid_K, grid_frequencies=grid_omega)

    def test_rejects_grid_coupling_nonzero_diagonal(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        grid_K = np.array(
            [
                [1.0, 0.8, 0.1],
                [0.8, 0.0, 0.3],
                [0.1, 0.3, 0.0],
            ]
        )
        grid_omega = np.array([0.0, 0.02, -0.01])
        with pytest.raises(ValueError, match="grid_coupling diagonal must be zero"):
            power_grid_benchmark(K, omega, grid_coupling=grid_K, grid_frequencies=grid_omega)
