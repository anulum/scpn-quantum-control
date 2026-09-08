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

from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pytest
from scipy import sparse

from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.phase import tensor_jump as tensor_jump_module
from scpn_quantum_control.phase.tensor_jump import (
    _build_effective_hamiltonian,
    _order_param_vec,
    _single_qubit_sparse,
    mcwf_ensemble,
    mcwf_trajectory,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _TrajectoryOverrides(TypedDict, total=False):
    """Floating-point trajectory fields exercised by the rejection table."""

    gamma_amp: float
    gamma_deph: float
    t_max: float
    dt: float


def _system(n: int = 3) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return K, omega


class TestBuildEffectiveHamiltonian:
    def test_non_hermitian(self) -> None:
        """H_eff should NOT be Hermitian when L_ops are non-trivial."""
        from scpn_quantum_control.phase.lindblad import _sigma

        H = np.eye(4, dtype=np.complex128)
        L_ops = [np.sqrt(0.1) * _sigma("-", 0, 2)]
        H_eff = _build_effective_hamiltonian(H, L_ops)
        diff = np.linalg.norm(H_eff - H_eff.conj().T)
        assert diff > 1e-10

    def test_no_ops_returns_copy(self) -> None:
        H = np.eye(4, dtype=np.complex128) * 3.0
        H_eff = _build_effective_hamiltonian(H, [])
        np.testing.assert_allclose(H_eff, H)

    def test_shape_preserved(self) -> None:
        from scpn_quantum_control.phase.lindblad import _sigma

        H = np.zeros((8, 8), dtype=np.complex128)
        L_ops = [np.sqrt(0.05) * _sigma("-", i, 3) for i in range(3)]
        H_eff = _build_effective_hamiltonian(H, L_ops)
        assert H_eff.shape == (8, 8)


class TestSparseJumpOperators:
    @pytest.mark.parametrize("pauli", ["X", "Y", "Z", "+", "-"])
    @pytest.mark.parametrize("qubit", [0, 1, 2])
    def test_single_qubit_sparse_matches_dense_sigma(self, pauli: str, qubit: int) -> None:
        from scpn_quantum_control.phase.lindblad import _sigma

        dense = _sigma(pauli, qubit, 3)
        sparse_matrix = _single_qubit_sparse(pauli, qubit, 3).toarray()

        np.testing.assert_allclose(sparse_matrix, dense)


class TestMCWFTrajectory:
    def test_output_keys(self) -> None:
        K, omega = _system(2)
        result = mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.5, dt=0.1, seed=42)
        assert set(result.keys()) == {"times", "R", "psi_final", "n_jumps"}

    def test_r_bounded(self) -> None:
        K, omega = _system(2)
        result = mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.5, dt=0.1, seed=42)
        assert np.all(result["R"] >= 0)
        assert np.all(result["R"] <= 1.0 + 1e-10)

    def test_psi_normalised(self) -> None:
        K, omega = _system(3)
        result = mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.3, dt=0.05, seed=7)
        norm = np.linalg.norm(result["psi_final"])
        np.testing.assert_allclose(norm, 1.0, atol=1e-8)

    def test_seed_reproducibility(self) -> None:
        K, omega = _system(2)
        r1 = mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.3, dt=0.1, seed=42)
        r2 = mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.3, dt=0.1, seed=42)
        np.testing.assert_array_equal(r1["R"], r2["R"])

    def test_zero_damping_unitary(self) -> None:
        """Without damping, no jumps should occur."""
        K, omega = _system(2)
        result = mcwf_trajectory(
            K, omega, gamma_amp=0.0, gamma_deph=0.0, t_max=0.5, dt=0.1, seed=42
        )
        assert result["n_jumps"] == 0

    def test_dephasing_only(self) -> None:
        K, omega = _system(2)
        result = mcwf_trajectory(
            K, omega, gamma_amp=0.0, gamma_deph=0.1, t_max=0.3, dt=0.05, seed=42
        )
        assert result["psi_final"].shape == (4,)

    def test_single_step(self) -> None:
        K, omega = _system(2)
        result = mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.05, dt=0.05, seed=42)
        assert len(result["times"]) == 2

    def test_time_grid_respects_requested_maximum_step(self) -> None:
        K, omega = _system(2)

        result = mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.25, dt=0.1, seed=42)

        assert result["times"][0] == pytest.approx(0.0)
        assert result["times"][-1] == pytest.approx(0.25)
        assert np.max(np.diff(result["times"])) <= 0.1 + 1e-12
        assert result["R"].shape == result["times"].shape

    def test_propagation_steps_match_reported_time_grid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scpn_quantum_control.phase import tensor_jump as module

        K = np.zeros((1, 1))
        omega = np.zeros(1)
        step_durations: list[float] = []

        monkeypatch.setattr(
            module,
            "knm_to_sparse_matrix",
            lambda K, omega: sparse.identity(2, dtype=np.complex128, format="csr"),
        )

        def record_step_duration(
            generator: sparse.spmatrix, psi: NDArray[np.complex128]
        ) -> NDArray[np.complex128]:
            step_durations.append(float(np.real(generator[0, 0] / -1j)))
            return psi

        monkeypatch.setattr(module, "expm_multiply", record_step_duration)

        result = module.mcwf_trajectory(
            K,
            omega,
            gamma_amp=0.0,
            gamma_deph=0.0,
            t_max=0.25,
            dt=0.1,
            seed=42,
        )

        np.testing.assert_allclose(step_durations, np.diff(result["times"]), atol=1e-15)
        assert sum(step_durations) == pytest.approx(0.25)
        assert max(step_durations) <= 0.1 + 1e-12

    def test_order_parameter_uses_kron_qubit_ordering(self) -> None:
        from scpn_quantum_control.phase.lindblad import _sigma

        psi = np.array([0.2 + 0.1j, 0.3 - 0.4j, -0.5 + 0.2j, 0.6 + 0.1j])
        psi = psi / np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())
        z = 0.0 + 0.0j
        for qubit in range(2):
            z += np.trace(_sigma("X", qubit, 2) @ rho)
            z += 1j * np.trace(_sigma("Y", qubit, 2) @ rho)
        expected = abs(z / 2)

        assert _order_param_vec(psi, 2) == pytest.approx(expected)

    def test_zero_horizon_returns_initial_state_without_propagation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        K, omega = _system(2)

        def fail_propagation(*args: object, **kwargs: object) -> None:
            raise AssertionError("zero-horizon MCWF must not propagate")

        monkeypatch.setattr(tensor_jump_module, "expm_multiply", fail_propagation)

        result = mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.0, dt=0.1, seed=42)

        assert result["times"].shape == (1,)
        assert result["times"][0] == 0.0
        assert result["R"].shape == (1,)
        assert result["psi_final"].shape == (4,)
        assert result["n_jumps"] == 0

    def test_zero_weight_selected_jump_falls_back_to_no_jump(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scpn_quantum_control.phase import tensor_jump as module

        class FixedRng:
            def uniform(self) -> float:
                return 0.0

        monkeypatch.setattr(np.random, "default_rng", lambda seed=None: FixedRng())
        monkeypatch.setattr(
            module,
            "knm_to_sparse_matrix",
            lambda K, omega: sparse.csr_matrix((2, 2), dtype=np.complex128),
        )
        monkeypatch.setattr(
            module,
            "_build_sparse_lindblad_ops",
            lambda n, gamma_amp, gamma_deph: [sparse.csr_matrix((2, 2), dtype=np.complex128)],
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

    def test_trajectory_does_not_use_dense_hamiltonian_builder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MCWF trajectory path must remain sparse/statevector, not dense-Hamiltonian."""
        K, omega = _system(3)

        def fail_dense(*args: object, **kwargs: object) -> None:
            raise AssertionError("dense Hamiltonian builder must not be used by MCWF trajectory")

        monkeypatch.setattr(tensor_jump_module, "knm_to_dense_matrix", fail_dense, raising=False)

        result = mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.1, dt=0.1, seed=42)

        assert result["psi_final"].shape == (8,)
        np.testing.assert_allclose(np.linalg.norm(result["psi_final"]), 1.0, atol=1e-8)

    def test_rejects_statevector_budget_before_sparse_setup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        K, omega = _system(3)

        def fail_sparse(*args: object, **kwargs: object) -> None:
            raise AssertionError("sparse Hamiltonian must not be built after budget rejection")

        monkeypatch.setattr(tensor_jump_module, "knm_to_sparse_matrix", fail_sparse)

        with pytest.raises(DenseAllocationError, match="MCWF statevector workspace"):
            mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=0.1, dt=0.1, max_dense_gib=1e-12)

    @pytest.mark.parametrize(
        ("K", "omega", "kwargs", "match"),
        [
            (np.ones((2, 3)), np.ones(2), {}, "square"),
            (np.empty((0, 0)), np.empty(0), {}, "at least one oscillator"),
            (np.eye(3), np.ones(2), {}, "omega"),
            (np.eye(2), np.array([1.0, np.nan]), {}, "finite"),
            (np.eye(2), np.ones(2), {"gamma_amp": -0.1}, "gamma_amp"),
            (np.eye(2), np.ones(2), {"gamma_deph": -0.1}, "gamma_deph"),
            (np.eye(2), np.ones(2), {"t_max": -0.1}, "t_max"),
            (np.eye(2), np.ones(2), {"dt": 0.0}, "dt"),
        ],
    )
    def test_rejects_invalid_inputs(
        self,
        K: NDArray[np.float64],
        omega: NDArray[np.float64],
        kwargs: _TrajectoryOverrides,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            mcwf_trajectory(K, omega, **kwargs)


class TestMCWFEnsemble:
    def test_output_keys(self) -> None:
        K, omega = _system(2)
        result = mcwf_ensemble(
            K, omega, gamma_amp=0.05, t_max=0.3, dt=0.1, n_trajectories=5, seed=42
        )
        assert "R_mean" in result
        assert "R_std" in result
        assert "R_trajectories" in result
        assert result["n_trajectories"] == 5

    def test_r_mean_bounded(self) -> None:
        K, omega = _system(2)
        result = mcwf_ensemble(
            K, omega, gamma_amp=0.05, t_max=0.3, dt=0.1, n_trajectories=10, seed=42
        )
        assert np.all(result["R_mean"] >= 0)
        assert np.all(result["R_mean"] <= 1.0 + 1e-10)

    def test_trajectories_shape(self) -> None:
        K, omega = _system(2)
        result = mcwf_ensemble(
            K, omega, gamma_amp=0.05, t_max=0.2, dt=0.1, n_trajectories=3, seed=42
        )
        assert result["R_trajectories"].shape[0] == 3

    def test_ensemble_time_grid_matches_trajectory_resolution(self) -> None:
        K, omega = _system(2)

        result = mcwf_ensemble(
            K, omega, gamma_amp=0.05, t_max=0.25, dt=0.1, n_trajectories=3, seed=42
        )

        assert result["times"][0] == pytest.approx(0.0)
        assert result["times"][-1] == pytest.approx(0.25)
        assert np.max(np.diff(result["times"])) <= 0.1 + 1e-12
        assert result["R_trajectories"].shape[1] == result["times"].shape[0]

    def test_total_jumps_nonneg(self) -> None:
        K, omega = _system(2)
        result = mcwf_ensemble(
            K, omega, gamma_amp=0.1, t_max=0.5, dt=0.05, n_trajectories=10, seed=42
        )
        assert result["total_jumps"] >= 0

    def test_ensemble_propagates_statevector_budget(self) -> None:
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

    @pytest.mark.parametrize("n_trajectories", [0, -1])
    def test_rejects_non_positive_trajectory_count(self, n_trajectories: int) -> None:
        K, omega = _system(2)

        with pytest.raises(ValueError, match="n_trajectories"):
            mcwf_ensemble(K, omega, n_trajectories=n_trajectories)


class TestOrderParamVec:
    def test_all_up_r(self) -> None:
        """All spin up → R depends on the state structure."""
        psi = np.zeros(4, dtype=np.complex128)
        psi[0] = 1.0
        r = _order_param_vec(psi, 2)
        assert np.isfinite(r)
        assert r >= 0

    def test_bell_state(self) -> None:
        """Bell state |00⟩+|11⟩)/√2 has well-defined R."""
        psi = np.zeros(4, dtype=np.complex128)
        psi[0] = 1.0 / np.sqrt(2)
        psi[3] = 1.0 / np.sqrt(2)
        r = _order_param_vec(psi, 2)
        assert 0 <= r <= 1.0

    def test_plus_product_state_has_unit_order(self) -> None:
        """The |++> product state has unit transverse Kuramoto order."""
        psi = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex128)
        assert _order_param_vec(psi, 2) == pytest.approx(1.0)

    def test_y_axis_product_state_has_unit_order(self) -> None:
        """The |+i,+i> product state has unit Pauli-Y Kuramoto order."""
        single = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2)
        psi = np.kron(single, single)
        assert _order_param_vec(psi, 2) == pytest.approx(1.0)

    def test_global_phase_invariant(self) -> None:
        """Global phase must not change the physical order parameter."""
        psi = np.array([0.2 + 0.1j, 0.3 - 0.4j, -0.5 + 0.2j, 0.6 + 0.1j])
        psi = psi / np.linalg.norm(psi)
        phase = np.exp(0.37j)
        assert _order_param_vec(phase * psi, 2) == pytest.approx(_order_param_vec(psi, 2))
