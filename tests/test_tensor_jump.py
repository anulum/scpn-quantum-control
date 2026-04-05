# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Tensor Jump Method
"""Tests for Monte Carlo Wave Function (MCWF) open-system solver.

Covers:
    - _build_effective_hamiltonian non-Hermiticity
    - mcwf_trajectory output structure and physics
    - mcwf_ensemble averaging and statistics
    - _order_param_vec Rust and Python paths
    - Seed reproducibility
    - Zero damping = unitary evolution
    - Jump probability and jump counting
    - Edge cases: single step, n=2
"""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.phase.tensor_jump import (
    _build_effective_hamiltonian,
    _order_param_vec,
    mcwf_ensemble,
    mcwf_trajectory,
)


def _system(n: int = 3):
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return K, omega


class TestBuildEffectiveHamiltonian:
    def test_non_hermitian(self):
        """H_eff should NOT be Hermitian when L_ops are non-trivial."""
        from scpn_quantum_control.phase.lindblad import _sigma

        H = np.eye(4, dtype=np.complex128)
        L_ops = [np.sqrt(0.1) * _sigma("-", 0, 2)]
        H_eff = _build_effective_hamiltonian(H, L_ops)
        diff = np.linalg.norm(H_eff - H_eff.conj().T)
        assert diff > 1e-10

    def test_no_ops_returns_copy(self):
        H = np.eye(4, dtype=np.complex128) * 3.0
        H_eff = _build_effective_hamiltonian(H, [])
        np.testing.assert_allclose(H_eff, H)

    def test_shape_preserved(self):
        from scpn_quantum_control.phase.lindblad import _sigma

        H = np.zeros((8, 8), dtype=np.complex128)
        L_ops = [np.sqrt(0.05) * _sigma("-", i, 3) for i in range(3)]
        H_eff = _build_effective_hamiltonian(H, L_ops)
        assert H_eff.shape == (8, 8)


class TestMCWFTrajectory:
    def test_output_keys(self):
        K, omega = _system(2)
        result = mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.5, dt=0.1, seed=42)
        assert set(result.keys()) == {"times", "R", "psi_final", "n_jumps"}

    def test_r_bounded(self):
        K, omega = _system(2)
        result = mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.5, dt=0.1, seed=42)
        assert np.all(result["R"] >= 0)
        assert np.all(result["R"] <= 1.0 + 1e-10)

    def test_psi_normalised(self):
        K, omega = _system(3)
        result = mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.3, dt=0.05, seed=7)
        norm = np.linalg.norm(result["psi_final"])
        np.testing.assert_allclose(norm, 1.0, atol=1e-8)

    def test_seed_reproducibility(self):
        K, omega = _system(2)
        r1 = mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.3, dt=0.1, seed=42)
        r2 = mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.3, dt=0.1, seed=42)
        np.testing.assert_array_equal(r1["R"], r2["R"])

    def test_zero_damping_unitary(self):
        """Without damping, no jumps should occur."""
        K, omega = _system(2)
        result = mcwf_trajectory(
            K, omega, gamma_amp=0.0, gamma_deph=0.0, t_max=0.5, dt=0.1, seed=42
        )
        assert result["n_jumps"] == 0

    def test_dephasing_only(self):
        K, omega = _system(2)
        result = mcwf_trajectory(
            K, omega, gamma_amp=0.0, gamma_deph=0.1, t_max=0.3, dt=0.05, seed=42
        )
        assert result["psi_final"].shape == (4,)

    def test_single_step(self):
        K, omega = _system(2)
        result = mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.05, dt=0.05, seed=42)
        assert len(result["times"]) == 2


class TestMCWFEnsemble:
    def test_output_keys(self):
        K, omega = _system(2)
        result = mcwf_ensemble(
            K, omega, gamma_amp=0.05, t_max=0.3, dt=0.1, n_trajectories=5, seed=42
        )
        assert "R_mean" in result
        assert "R_std" in result
        assert "R_trajectories" in result
        assert result["n_trajectories"] == 5

    def test_r_mean_bounded(self):
        K, omega = _system(2)
        result = mcwf_ensemble(
            K, omega, gamma_amp=0.05, t_max=0.3, dt=0.1, n_trajectories=10, seed=42
        )
        assert np.all(result["R_mean"] >= 0)
        assert np.all(result["R_mean"] <= 1.0 + 1e-10)

    def test_trajectories_shape(self):
        K, omega = _system(2)
        result = mcwf_ensemble(
            K, omega, gamma_amp=0.05, t_max=0.2, dt=0.1, n_trajectories=3, seed=42
        )
        assert result["R_trajectories"].shape[0] == 3

    def test_total_jumps_nonneg(self):
        K, omega = _system(2)
        result = mcwf_ensemble(
            K, omega, gamma_amp=0.1, t_max=0.5, dt=0.05, n_trajectories=10, seed=42
        )
        assert result["total_jumps"] >= 0


class TestOrderParamVec:
    def test_all_up_r(self):
        """All spin up → R depends on the state structure."""
        psi = np.zeros(4, dtype=np.complex128)
        psi[0] = 1.0
        r = _order_param_vec(psi, 2)
        assert np.isfinite(r)
        assert r >= 0

    def test_bell_state(self):
        """Bell state |00⟩+|11⟩)/√2 has well-defined R."""
        psi = np.zeros(4, dtype=np.complex128)
        psi[0] = 1.0 / np.sqrt(2)
        psi[3] = 1.0 / np.sqrt(2)
        r = _order_param_vec(psi, 2)
        assert 0 <= r <= 1.0

    def test_python_fallback(self):
        """Force Python path and compare to Rust path."""
        from unittest.mock import patch

        psi = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex128)
        r_default = _order_param_vec(psi, 2)

        with patch.dict("sys.modules", {"scpn_quantum_engine": None}):
            r_python = _order_param_vec(psi, 2)

        np.testing.assert_allclose(r_python, r_default, atol=1e-10)
