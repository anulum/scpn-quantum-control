# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Eeg Benchmark
"""Tests for EEG neural oscillator benchmark."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from scpn_quantum_control.applications.eeg_benchmark import (
    EEG_ALPHA_PLV,
    EEGBenchmarkResult,
    eeg_benchmark,
    eeg_coupling_matrix,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestEEGData:
    def test_plv_symmetric(self):
        np.testing.assert_allclose(EEG_ALPHA_PLV, EEG_ALPHA_PLV.T, atol=1e-10)

    def test_plv_8x8(self):
        assert EEG_ALPHA_PLV.shape == (8, 8)

    def test_plv_bounded(self):
        assert np.all(EEG_ALPHA_PLV >= 0)
        assert np.all(EEG_ALPHA_PLV <= 1.0)

    def test_plv_zero_diagonal(self):
        np.testing.assert_allclose(np.diag(EEG_ALPHA_PLV), 0.0)


class TestEEGCouplingMatrix:
    def test_builtin_alpha_requires_explicit_opt_in(self):
        with pytest.raises(RuntimeError, match="allow_builtin_reference"):
            eeg_coupling_matrix("alpha")

    def test_alpha_shape(self):
        K, omega = eeg_coupling_matrix("alpha", allow_builtin_reference=True)
        assert K.shape == (8, 8)
        assert omega.shape == (8,)


class TestEEGBenchmark:
    def test_returns_result(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = eeg_benchmark(K, omega, allow_builtin_reference=True)
        assert isinstance(result, EEGBenchmarkResult)

    def test_n_channels(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = eeg_benchmark(K, omega, allow_builtin_reference=True)
        assert result.n_channels == 8

    def test_correlation_bounded(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = eeg_benchmark(K, omega, allow_builtin_reference=True)
        assert -1 <= result.topology_correlation <= 1

    def test_fewer_oscillators(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = eeg_benchmark(K, omega, allow_builtin_reference=True)
        assert result.n_channels == 4

    def test_constant_frequency_vector_returns_zero_correlation(self):
        K = build_knm_paper27(L=4)
        omega = np.ones(4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = eeg_benchmark(K, omega, allow_builtin_reference=True)
        assert result.frequency_correlation == 0.0
        assert result.n_channels == 4
        assert "freq r=0.000" in result.summary

    def test_summary_string(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = eeg_benchmark(K, omega, allow_builtin_reference=True)
        assert "SCPN vs EEG" in result.summary

    def test_result_labels_builtin_reference_source_mode(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = eeg_benchmark(K, omega, allow_builtin_reference=True)
        assert result.source_mode == "builtin_literature_shape"
        assert result.publication_safe is False

    def test_scpn_vs_eeg(self):
        """Record SCPN vs EEG — Gap 1 data."""
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = eeg_benchmark(K, omega, allow_builtin_reference=True)
        print(f"\n  {result.summary}")
        assert isinstance(result.topology_correlation, float)

    def test_rejects_non_square_scpn_coupling(self):
        K = np.ones((2, 3))
        omega = np.ones(2)
        with pytest.raises(ValueError, match="K_scpn must be a square"):
            eeg_benchmark(K, omega, allow_builtin_reference=True)

    def test_rejects_scpn_frequency_shape_mismatch(self):
        K = build_knm_paper27(L=4)
        omega = np.ones(3)
        with pytest.raises(ValueError, match="omega_scpn must match"):
            eeg_benchmark(K, omega, allow_builtin_reference=True)

    def test_rejects_non_finite_scpn_coupling(self):
        K = build_knm_paper27(L=4)
        K[0, 1] = np.nan
        omega = OMEGA_N_16[:4]
        with pytest.raises(ValueError, match="K_scpn must contain only finite"):
            eeg_benchmark(K, omega, allow_builtin_reference=True)

    def test_rejects_too_small_comparison_system(self):
        K = np.array([[0.0]])
        omega = np.array([10.0])
        with pytest.raises(ValueError, match="at least two coupled channels"):
            eeg_benchmark(K, omega, allow_builtin_reference=True)

    def test_rejects_measured_eeg_asymmetry(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        eeg_K = np.array(
            [
                [0.0, 0.8, 0.1],
                [0.2, 0.0, 0.3],
                [0.1, 0.3, 0.0],
            ]
        )
        eeg_omega = np.array([9.0, 10.0, 11.0])
        with pytest.raises(ValueError, match="eeg_coupling must be symmetric"):
            eeg_benchmark(K, omega, eeg_coupling=eeg_K, eeg_frequencies=eeg_omega)

    def test_rejects_measured_eeg_outside_unit_interval(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        eeg_K = np.array(
            [
                [0.0, 1.2, 0.1],
                [1.2, 0.0, 0.3],
                [0.1, 0.3, 0.0],
            ]
        )
        eeg_omega = np.array([9.0, 10.0, 11.0])
        with pytest.raises(ValueError, match="eeg_coupling values must be in"):
            eeg_benchmark(K, omega, eeg_coupling=eeg_K, eeg_frequencies=eeg_omega)

    def test_rejects_measured_eeg_nonzero_diagonal(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        eeg_K = np.array(
            [
                [1.0, 0.8, 0.1],
                [0.8, 0.0, 0.3],
                [0.1, 0.3, 0.0],
            ]
        )
        eeg_omega = np.array([9.0, 10.0, 11.0])
        with pytest.raises(ValueError, match="eeg_coupling diagonal must be zero"):
            eeg_benchmark(K, omega, eeg_coupling=eeg_K, eeg_frequencies=eeg_omega)
