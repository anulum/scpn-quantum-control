# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Iter Benchmark
"""Tests for ITER synthetic data benchmark."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from scpn_quantum_control.applications.iter_benchmark import (
    ITER_MODE_COUPLING,
    ITERBenchmarkResult,
    iter_benchmark,
    iter_coupling_matrix,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestITERData:
    def test_coupling_symmetric(self):
        np.testing.assert_allclose(ITER_MODE_COUPLING, ITER_MODE_COUPLING.T, atol=1e-10)

    def test_coupling_8x8(self):
        assert ITER_MODE_COUPLING.shape == (8, 8)

    def test_coupling_non_negative(self):
        assert np.all(ITER_MODE_COUPLING >= 0)

    def test_zero_diagonal(self):
        np.testing.assert_allclose(np.diag(ITER_MODE_COUPLING), 0.0)


class TestITERCouplingMatrix:
    def test_builtin_synthetic_reference_requires_explicit_opt_in(self):
        with pytest.raises(RuntimeError, match="allow_synthetic_reference"):
            iter_coupling_matrix()

    def test_shape(self):
        K, omega = iter_coupling_matrix(allow_synthetic_reference=True)
        assert K.shape == (8, 8)
        assert omega.shape == (8,)


class TestITERBenchmark:
    def test_returns_result(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = iter_benchmark(K, omega, allow_synthetic_reference=True)
        assert isinstance(result, ITERBenchmarkResult)

    def test_n_modes(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = iter_benchmark(K, omega, allow_synthetic_reference=True)
        assert result.n_modes == 8

    def test_correlation_bounded(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = iter_benchmark(K, omega, allow_synthetic_reference=True)
        assert -1 <= result.topology_correlation <= 1

    def test_fewer_oscillators(self):
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        result = iter_benchmark(K, omega, allow_synthetic_reference=True)
        assert result.n_modes == 5

    def test_constant_frequency_vector_returns_zero_correlation(self):
        K = build_knm_paper27(L=5)
        omega = np.ones(5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = iter_benchmark(K, omega, allow_synthetic_reference=True)
        assert result.frequency_correlation == 0.0
        assert result.n_modes == 5
        assert "freq r=0.000" in result.summary

    def test_locking_risk_bounded(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = iter_benchmark(K, omega, allow_synthetic_reference=True)
        assert 0 <= result.mode_locking_risk <= 1.0

    def test_summary_string(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = iter_benchmark(K, omega, allow_synthetic_reference=True)
        assert "SCPN vs ITER" in result.summary

    def test_benchmark_refuses_implicit_synthetic_reference(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        with pytest.raises(RuntimeError, match="allow_synthetic_reference"):
            iter_benchmark(K, omega)

    def test_result_labels_synthetic_reference_source_mode(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = iter_benchmark(K, omega, allow_synthetic_reference=True)
        assert result.source_mode == "synthetic"
        assert result.publication_safe is False

    def test_result_labels_curated_reference_source_mode(self):
        K_iter, omega_iter = iter_coupling_matrix(allow_synthetic_reference=True)
        result = iter_benchmark(
            K_iter,
            omega_iter,
            iter_coupling=K_iter,
            iter_frequencies=omega_iter,
            reference_source_mode="curated",
        )
        assert result.source_mode == "curated"
        assert result.publication_safe is True

    def test_reference_matrix_and_frequency_vector_must_be_supplied_together(self):
        K_iter, omega_iter = iter_coupling_matrix(allow_synthetic_reference=True)
        with pytest.raises(ValueError, match="iter_coupling and iter_frequencies"):
            iter_benchmark(K_iter, omega_iter, iter_coupling=K_iter)

    def test_scpn_vs_iter(self):
        """Record SCPN vs ITER — Gap 1 fusion data."""
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = iter_benchmark(K, omega, allow_synthetic_reference=True)
        print(f"\n  {result.summary}")
        assert isinstance(result.topology_correlation, float)
