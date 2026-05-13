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
import pytest

from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.phase import tensor_jump as tensor_jump_module
from scpn_quantum_control.phase.tensor_jump import (
    _build_effective_hamiltonian,
    _order_param_vec,
    _single_qubit_sparse,
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


class TestSparseJumpOperators:
    @pytest.mark.parametrize("pauli", ["X", "Y", "Z", "+", "-"])
    @pytest.mark.parametrize("qubit", [0, 1, 2])
    def test_single_qubit_sparse_matches_dense_sigma(self, pauli, qubit):
        from scpn_quantum_control.phase.lindblad import _sigma

        dense = _sigma(pauli, qubit, 3)
        sparse_matrix = _single_qubit_sparse(pauli, qubit, 3).toarray()

        np.testing.assert_allclose(sparse_matrix, dense)


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

    def test_zero_weight_selected_jump_falls_back_to_no_jump(self, monkeypatch):
        from scpn_quantum_control.phase import tensor_jump as module

        class FixedRng:
            def uniform(self):
                return 0.0

        monkeypatch.setattr(module.np.random, "default_rng", lambda seed=None: FixedRng())
        monkeypatch.setattr(
            module,
            "knm_to_sparse_matrix",
            lambda K, omega: module.sparse.csr_matrix((2, 2), dtype=np.complex128),
        )
        monkeypatch.setattr(
            module,
            "_build_sparse_lindblad_ops",
            lambda n, gamma_amp, gamma_deph: [
                module.sparse.csr_matrix((2, 2), dtype=np.complex128)
            ],
        )
        monkeypatch.setattr(module, "expm_multiply", lambda matrix, psi: 0.5 * psi)

        result = module.mcwf_trajectory(
            np.zeros((1, 1)),
            np.zeros(1),
            gamma_amp=0.1,
            gamma_deph=0.0,
            t_max=0.1,
            dt=0.1,
            seed=7,
        )

        assert result["n_jumps"] == 0
        np.testing.assert_allclose(np.linalg.norm(result["psi_final"]), 1.0)

    def test_trajectory_does_not_use_dense_hamiltonian_builder(self, monkeypatch):
        """MCWF trajectory path must remain sparse/statevector, not dense-Hamiltonian."""
        K, omega = _system(3)

        def fail_dense(*args, **kwargs):
            raise AssertionError("dense Hamiltonian builder must not be used by MCWF trajectory")

        monkeypatch.setattr(tensor_jump_module, "knm_to_dense_matrix", fail_dense, raising=False)

        result = mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.1, dt=0.1, seed=42)

        assert result["psi_final"].shape == (8,)
        np.testing.assert_allclose(np.linalg.norm(result["psi_final"]), 1.0, atol=1e-8)

    def test_rejects_statevector_budget_before_sparse_setup(self, monkeypatch):
        K, omega = _system(3)

        def fail_sparse(*args, **kwargs):
            raise AssertionError("sparse Hamiltonian must not be built after budget rejection")

        monkeypatch.setattr(tensor_jump_module, "knm_to_sparse_matrix", fail_sparse)

        with pytest.raises(DenseAllocationError, match="MCWF statevector workspace"):
            mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.1, dt=0.1, max_dense_gib=1e-12)

    @pytest.mark.parametrize(
        ("K", "omega", "kwargs", "match"),
        [
            (np.ones((2, 3)), np.ones(2), {}, "square"),
            (np.eye(3), np.ones(2), {}, "omega"),
            (np.eye(2), np.array([1.0, np.nan]), {}, "finite"),
            (np.eye(2), np.ones(2), {"gamma_amp": -0.1}, "gamma_amp"),
            (np.eye(2), np.ones(2), {"gamma_deph": -0.1}, "gamma_deph"),
            (np.eye(2), np.ones(2), {"t_max": -0.1}, "t_max"),
            (np.eye(2), np.ones(2), {"dt": 0.0}, "dt"),
        ],
    )
    def test_rejects_invalid_inputs(self, K, omega, kwargs, match):
        with pytest.raises(ValueError, match=match):
            mcwf_trajectory(K, omega, **kwargs)


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

    def test_ensemble_propagates_statevector_budget(self):
        K, omega = _system(3)

        with pytest.raises(DenseAllocationError, match="MCWF statevector workspace"):
            mcwf_ensemble(
                K,
                omega,
                gamma_amp=0.05,
                t_max=0.1,
                dt=0.1,
                n_trajectories=2,
                seed=42,
                max_dense_gib=1e-12,
            )


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
