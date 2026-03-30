# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
"""Tests for Neural Quantum State (RBM) ansatz."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase.nqs_ansatz import RBMWavefunction, vmc_ground_state


class TestRBMWavefunction:
    def test_log_psi_returns_complex(self):
        rbm = RBMWavefunction(4, seed=42)
        sigma = np.array([1, -1, 1, -1], dtype=np.float64)
        val = rbm.log_psi(sigma)
        assert isinstance(val, complex)

    def test_psi_nonzero(self):
        rbm = RBMWavefunction(4, seed=42)
        sigma = np.array([1, 1, 1, 1], dtype=np.float64)
        assert abs(rbm.psi(sigma)) > 0

    def test_all_amplitudes_normalised(self):
        rbm = RBMWavefunction(3, seed=42)
        amps = rbm.all_amplitudes()
        norm = np.sum(np.abs(amps) ** 2)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)

    def test_all_amplitudes_correct_size(self):
        rbm = RBMWavefunction(4, seed=42)
        amps = rbm.all_amplitudes()
        assert len(amps) == 16

    def test_n_params(self):
        rbm = RBMWavefunction(4, n_hidden=8, seed=42)
        assert rbm.n_params() == 4 + 8 + 8 * 4  # a + b + W

    def test_reproducible_with_seed(self):
        rbm1 = RBMWavefunction(3, seed=123)
        rbm2 = RBMWavefunction(3, seed=123)
        np.testing.assert_array_equal(rbm1.a, rbm2.a)
        np.testing.assert_array_equal(rbm1.W, rbm2.W)


class TestVMCGroundState:
    def test_vmc_returns_energy(self):
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, 1.2])
        result = vmc_ground_state(K, omega, n_iterations=10, seed=42)
        assert "energy" in result
        assert isinstance(result["energy"], float)

    def test_vmc_energy_decreases(self):
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, 1.2])
        result = vmc_ground_state(K, omega, n_iterations=30, learning_rate=0.005, seed=42)
        assert result["energy_history"][-1] <= result["energy_history"][0] + 0.1

    def test_vmc_output_keys(self):
        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, 1.1])
        result = vmc_ground_state(K, omega, n_iterations=5, seed=42)
        assert set(result.keys()) == {"energy", "energy_history", "wavefunction", "n_params"}

    def test_vmc_rejects_large_n(self):
        K = np.eye(14)
        omega = np.ones(14)
        with pytest.raises(ValueError, match="Exact NQS only for n<=12"):
            vmc_ground_state(K, omega, n_iterations=1)
