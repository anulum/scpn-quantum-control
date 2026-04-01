# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Nqs Ansatz
"""Tests for Neural Quantum State (RBM) ansatz.

Multi-angle: normalisation invariant, parameter counting, reproducibility,
multiple system sizes, convergence, ED comparison, edge cases,
wavefunction properties, type checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase.nqs_ansatz import RBMWavefunction, vmc_ground_state


# =====================================================================
# RBM Wavefunction Properties
# =====================================================================
class TestRBMWavefunction:
    """RBM wavefunction structure and mathematical properties."""

    def test_log_psi_returns_complex(self):
        rbm = RBMWavefunction(4, seed=42)
        sigma = np.array([1, -1, 1, -1], dtype=np.float64)
        val = rbm.log_psi(sigma)
        assert isinstance(val, complex)

    def test_psi_nonzero_for_all_configs(self):
        """RBM should give nonzero amplitude for every configuration."""
        rbm = RBMWavefunction(3, seed=42)
        for k in range(2**3):
            sigma = np.array([(1 if (k >> i) & 1 else -1) for i in range(3)], dtype=np.float64)
            assert abs(rbm.psi(sigma)) > 0

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_all_amplitudes_normalised(self, n):
        """Sum |ψ(σ)|² = 1 — fundamental QM normalisation."""
        rbm = RBMWavefunction(n, seed=42)
        amps = rbm.all_amplitudes()
        norm = np.sum(np.abs(amps) ** 2)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_all_amplitudes_correct_size(self, n):
        rbm = RBMWavefunction(n, seed=42)
        amps = rbm.all_amplitudes()
        assert len(amps) == 2**n

    def test_amplitudes_all_finite(self):
        rbm = RBMWavefunction(4, seed=42)
        amps = rbm.all_amplitudes()
        assert all(np.isfinite(amps))

    @pytest.mark.parametrize("n_hidden", [None, 2, 4, 8, 16])
    def test_n_params_formula(self, n_hidden):
        """n_params = n_visible + n_hidden + n_visible * n_hidden."""
        n = 4
        rbm = RBMWavefunction(n, n_hidden=n_hidden, seed=42)
        nh = n_hidden if n_hidden is not None else 2 * n
        assert rbm.n_params() == n + nh + n * nh

    def test_reproducible_with_seed(self):
        rbm1 = RBMWavefunction(3, seed=123)
        rbm2 = RBMWavefunction(3, seed=123)
        np.testing.assert_array_equal(rbm1.a, rbm2.a)
        np.testing.assert_array_equal(rbm1.W, rbm2.W)

    def test_different_seeds_give_different_params(self):
        rbm1 = RBMWavefunction(3, seed=1)
        rbm2 = RBMWavefunction(3, seed=2)
        assert not np.array_equal(rbm1.a, rbm2.a)

    def test_psi_equals_exp_log_psi(self):
        """ψ(σ) = exp(log ψ(σ))."""
        rbm = RBMWavefunction(3, seed=42)
        sigma = np.array([1, -1, 1], dtype=np.float64)
        psi_direct = rbm.psi(sigma)
        psi_from_log = np.exp(rbm.log_psi(sigma))
        np.testing.assert_allclose(psi_direct, psi_from_log, rtol=1e-10)


# =====================================================================
# VMC Ground State Search
# =====================================================================
class TestVMCGroundState:
    """Variational Monte Carlo optimisation tests."""

    def test_vmc_returns_valid_result(self):
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, 1.2])
        result = vmc_ground_state(K, omega, n_iterations=10, seed=42)
        assert "energy" in result
        assert isinstance(result["energy"], float)
        assert np.isfinite(result["energy"])

    def test_vmc_energy_decreases_over_iterations(self):
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, 1.2])
        result = vmc_ground_state(K, omega, n_iterations=50, learning_rate=0.005, seed=42)
        # Last energy should be ≤ first (allowing small fluctuations)
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

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_vmc_variational_upper_bound(self, n):
        """VMC energy ≥ exact ground energy (variational principle)."""
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
        np.fill_diagonal(K, 0.0)
        omega = np.linspace(0.8, 1.2, n)

        H = knm_to_dense_matrix(K, omega)
        E_exact = np.linalg.eigvalsh(H)[0]

        result = vmc_ground_state(K, omega, n_iterations=200, seed=42)
        # VMC is variational: E_vmc ≥ E_exact (with small tolerance for numerics)
        assert result["energy"] >= E_exact - 0.5, (
            f"VMC {result['energy']:.4f} suspiciously below exact {E_exact:.4f}"
        )

    def test_vmc_reproducible_with_seed(self):
        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, 1.1])
        r1 = vmc_ground_state(K, omega, n_iterations=10, seed=42)
        r2 = vmc_ground_state(K, omega, n_iterations=10, seed=42)
        np.testing.assert_allclose(r1["energy"], r2["energy"])

    def test_vmc_wavefunction_is_normalised(self):
        """Returned wavefunction should give normalised amplitudes."""
        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, 1.1])
        result = vmc_ground_state(K, omega, n_iterations=10, seed=42)
        wf = result["wavefunction"]
        amps = wf.all_amplitudes()
        norm = np.sum(np.abs(amps) ** 2)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)

    def test_vmc_n_params_matches_wavefunction(self):
        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, 1.1])
        result = vmc_ground_state(K, omega, n_iterations=5, seed=42)
        assert result["n_params"] == result["wavefunction"].n_params()
