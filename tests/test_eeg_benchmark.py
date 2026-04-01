# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Eeg Benchmark
"""Tests for EEG neural oscillator benchmark."""

from __future__ import annotations

import numpy as np

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
    def test_alpha_shape(self):
        K, omega = eeg_coupling_matrix("alpha")
        assert K.shape == (8, 8)
        assert omega.shape == (8,)


class TestEEGBenchmark:
    def test_returns_result(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = eeg_benchmark(K, omega)
        assert isinstance(result, EEGBenchmarkResult)

    def test_n_channels(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = eeg_benchmark(K, omega)
        assert result.n_channels == 8

    def test_correlation_bounded(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = eeg_benchmark(K, omega)
        assert -1 <= result.topology_correlation <= 1

    def test_fewer_oscillators(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = eeg_benchmark(K, omega)
        assert result.n_channels == 4

    def test_summary_string(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = eeg_benchmark(K, omega)
        assert "SCPN vs EEG" in result.summary

    def test_scpn_vs_eeg(self):
        """Record SCPN vs EEG — Gap 1 data."""
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = eeg_benchmark(K, omega)
        print(f"\n  {result.summary}")
        assert isinstance(result.topology_correlation, float)
