# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tensor-jump contract tests
"""Contract tests for MCWF tensor-jump trajectory and ensemble statistics."""

from __future__ import annotations

from typing import TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
System = tuple[int, FloatArray, FloatArray]


def _system(n: int = 4) -> System:
    """Build a standard heterogeneous Kuramoto-XY system."""
    K: FloatArray = np.asarray(
        0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n)))),
        dtype=np.float64,
    )
    np.fill_diagonal(K, 0.0)
    omega: FloatArray = np.asarray(np.linspace(0.8, 1.2, n), dtype=np.float64)
    return n, K, omega


def _zero_coupling(n: int = 4) -> System:
    """Build a decoupled system whose eigenstates are product states."""
    K: FloatArray = np.zeros((n, n), dtype=np.float64)
    omega: FloatArray = np.asarray(np.linspace(0.8, 1.2, n), dtype=np.float64)
    return n, K, omega


class TestTensorJump:
    """Tests for Monte Carlo Wave Function method."""

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_single_trajectory_runs_multiple_sizes(self, n: int) -> None:
        """Run public trajectories across several system sizes."""
        from scpn_quantum_control.phase.tensor_jump import mcwf_trajectory

        _, K, omega = _system(n)
        result = mcwf_trajectory(
            K,
            omega,
            gamma_amp=0.1,
            t_max=0.3,
            dt=0.1,
            seed=42,
        )
        assert "R" in result
        assert "psi_final" in result
        assert "n_jumps" in result
        assert len(result["R"]) >= 3  # at least 3 time steps
        assert result["n_jumps"] >= 0

    def test_r_strictly_bounded(self) -> None:
        """R must be in [0, 1] — physical invariant."""
        from scpn_quantum_control.phase.tensor_jump import mcwf_trajectory

        _, K, omega = _system(4)
        result = mcwf_trajectory(
            K,
            omega,
            gamma_amp=0.05,
            t_max=1.0,
            dt=0.05,
            seed=42,
        )
        assert all(0 <= r <= 1.0 + 1e-10 for r in result["R"]), (
            f"R out of [0,1]: {[r for r in result['R'] if r < 0 or r > 1.0 + 1e-10]}"
        )

    @pytest.mark.parametrize("n_traj", [5, 20, 50])
    def test_ensemble_shape_consistency(self, n_traj: int) -> None:
        """Keep ensemble statistics aligned with trajectory and time counts."""
        from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, 1.1])
        result = mcwf_ensemble(
            K,
            omega,
            gamma_amp=0.1,
            t_max=0.2,
            dt=0.1,
            n_trajectories=n_traj,
            seed=42,
        )
        n_steps = len(result["times"])
        assert result["R_mean"].shape == (n_steps,)
        assert result["R_std"].shape == (n_steps,)
        assert result["R_trajectories"].shape == (n_traj, n_steps)
        assert result["n_trajectories"] == n_traj

    def test_no_damping_preserves_norm_and_no_jumps(self) -> None:
        """Zero gamma → unitary evolution, norm preserved, no jumps."""
        from scpn_quantum_control.phase.tensor_jump import mcwf_trajectory

        _, K, omega = _system(3)
        result = mcwf_trajectory(
            K,
            omega,
            gamma_amp=0.0,
            t_max=0.5,
            dt=0.05,
            seed=42,
        )
        norm = float(np.linalg.norm(result["psi_final"]))
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)
        assert result["n_jumps"] == 0, "No jumps expected at zero damping"

    def test_strong_damping_reduces_R(self) -> None:
        """Strong damping should drive R toward 0."""
        from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

        _, K, omega = _system(3)
        result = mcwf_ensemble(
            K,
            omega,
            gamma_amp=2.0,
            t_max=2.0,
            dt=0.05,
            n_trajectories=30,
            seed=42,
        )
        assert result["R_mean"][-1] < 0.5, (
            f"Strong damping should reduce R, got {result['R_mean'][-1]:.3f}"
        )

    def test_ensemble_output_keys(self) -> None:
        """Return the documented ensemble mapping keys."""
        from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, 1.1])
        result = mcwf_ensemble(
            K,
            omega,
            t_max=0.1,
            dt=0.05,
            n_trajectories=5,
            seed=42,
        )
        expected = {
            "times",
            "R_mean",
            "R_std",
            "R_trajectories",
            "total_jumps",
            "n_trajectories",
        }
        assert set(result.keys()) == expected

    def test_reproducible_with_seed(self) -> None:
        """Same seed should give identical results."""
        from scpn_quantum_control.phase.tensor_jump import mcwf_trajectory

        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, 1.1])
        r1 = mcwf_trajectory(K, omega, gamma_amp=0.1, t_max=0.3, dt=0.1, seed=42)
        r2 = mcwf_trajectory(K, omega, gamma_amp=0.1, t_max=0.3, dt=0.1, seed=42)
        np.testing.assert_array_equal(r1["R"], r2["R"])
        assert r1["n_jumps"] == r2["n_jumps"]

    def test_psi_final_is_normalised(self) -> None:
        """Final state vector must be normalised."""
        from scpn_quantum_control.phase.tensor_jump import mcwf_trajectory

        _, K, omega = _system(4)
        result = mcwf_trajectory(
            K,
            omega,
            gamma_amp=0.05,
            t_max=0.5,
            dt=0.1,
            seed=42,
        )
        norm = np.linalg.norm(result["psi_final"])
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)

    def test_ensemble_r_mean_bounded(self) -> None:
        """Ensemble-averaged R must be in [0, 1]."""
        from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

        _, K, omega = _system(3)
        result = mcwf_ensemble(
            K,
            omega,
            gamma_amp=0.1,
            t_max=0.5,
            dt=0.05,
            n_trajectories=20,
            seed=42,
        )
        assert all(0 <= r <= 1.0 + 1e-10 for r in result["R_mean"])
        assert all(s >= 0 for s in result["R_std"])

    @pytest.mark.parametrize("invalid_count", [True, 1.5])
    def test_ensemble_rejects_non_integral_counts(self, invalid_count: object) -> None:
        """Reject boolean and fractional trajectory counts at the public boundary."""
        _, K, omega = _zero_coupling(2)

        with pytest.raises(ValueError, match="positive integer"):
            from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

            mcwf_ensemble(K, omega, n_trajectories=cast(int, invalid_count))
