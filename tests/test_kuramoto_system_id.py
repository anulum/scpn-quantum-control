# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Differentiable Kuramoto system identification tests
"""Multi-angle tests for differentiable Kuramoto system identification (learn K from data).

Covers the trajectory-match loss and its gradient against finite differences for both
integrators, the value-only companion, input/step/integrator validation, and the learn-coupling
optimiser (monotone loss, recovery of a planted coupling from synthetic observations — loss
driven down by orders of magnitude and the coupling error reduced — the converged flag, the
iteration budget, the symmetry constraint and reproducibility).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import scpn_quantum_control.accel.diff_kuramoto_rk4 as dk_rk4
import scpn_quantum_control.accel.kuramoto_system_id as sysid


def _planted(n: int, seed: int):
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-math.pi, math.pi, size=n)
    omega = rng.uniform(-1.0, 1.0, size=n)
    coupling = rng.uniform(0.0, 0.4, size=(n, n))
    coupling = 0.5 * (coupling + coupling.T)
    np.fill_diagonal(coupling, 0.0)
    return theta0, omega, coupling


def _observe(theta0, omega, coupling, steps, dt=0.05, n_steps=40):
    trajectory = dk_rk4._python_kuramoto_rk4_trajectory(theta0, omega, coupling, dt, n_steps)
    return trajectory[steps]


class TestTrajectoryMatchGradient:
    @pytest.mark.parametrize("integrator", ["rk4", "euler"])
    def test_gradient_matches_finite_difference(self, integrator: str) -> None:
        theta0, omega, coupling = _planted(6, 5)
        dt, n_steps = 0.05, 20
        steps = np.array([5, 10, 15, 20])
        observed = _observe(theta0, omega, coupling, steps, dt, n_steps)
        guess = coupling + 0.1
        guess = 0.5 * (guess + guess.T)
        np.fill_diagonal(guess, 0.0)
        loss, _, _, grad_coupling = sysid.trajectory_match_value_and_grad(
            theta0, omega, guess, observed, steps, dt, n_steps, integrator=integrator
        )

        def cost(k: np.ndarray) -> float:
            return sysid.trajectory_match_value(
                theta0, omega, k, observed, steps, dt, n_steps, integrator=integrator
            )

        assert abs(loss - cost(guess)) < 1e-9
        step = 1e-6
        fd = np.zeros_like(guess)
        for p in range(6):
            for q in range(6):
                plus = guess.copy()
                minus = guess.copy()
                plus[p, q] += step
                minus[p, q] -= step
                fd[p, q] = (cost(plus) - cost(minus)) / (2.0 * step)
        np.testing.assert_allclose(grad_coupling, fd, atol=1e-5)

    def test_value_matches_value_and_grad(self) -> None:
        theta0, omega, coupling = _planted(5, 7)
        steps = np.array([10, 20])
        observed = _observe(theta0, omega, coupling, steps)
        loss, *_ = sysid.trajectory_match_value_and_grad(
            theta0, omega, coupling, observed, steps, 0.05, 40
        )
        value = sysid.trajectory_match_value(theta0, omega, coupling, observed, steps, 0.05, 40)
        assert abs(loss - value) < 1e-12
        # At the true coupling the loss vanishes.
        assert loss < 1e-18

    def test_validation(self) -> None:
        theta0, omega, coupling = _planted(4, 1)
        steps = np.array([5, 10])
        observed = _observe(theta0, omega, coupling, steps)
        with pytest.raises(ValueError, match="omega must have shape"):
            sysid.trajectory_match_value(theta0, np.zeros(3), coupling, observed, steps, 0.05, 40)
        with pytest.raises(ValueError, match="coupling must have shape"):
            sysid.trajectory_match_value(
                theta0, omega, np.zeros((4, 3)), observed, steps, 0.05, 40
            )
        with pytest.raises(ValueError, match="observed must have shape"):
            sysid.trajectory_match_value(
                theta0, omega, coupling, np.zeros((3, 4)), steps, 0.05, 40
            )
        with pytest.raises(ValueError, match="observation_steps must be one-dimensional"):
            sysid.trajectory_match_value(
                theta0, omega, coupling, observed, np.zeros((2, 2), dtype=np.int64), 0.05, 40
            )
        with pytest.raises(ValueError, match="observation_steps must be non-empty within"):
            sysid.trajectory_match_value(
                theta0, omega, coupling, observed, np.array([5, 99]), 0.05, 40
            )
        with pytest.raises(ValueError, match="integrator must be 'rk4' or 'euler'"):
            sysid.trajectory_match_value_and_grad(
                theta0, omega, coupling, observed, steps, 0.05, 40, integrator="midpoint"
            )
        with pytest.raises(ValueError, match="integrator must be 'rk4' or 'euler'"):
            sysid.trajectory_match_value(
                theta0, omega, coupling, observed, steps, 0.05, 40, integrator="x"
            )


class TestLearnCoupling:
    def test_recovers_planted_coupling(self) -> None:
        theta0, omega, coupling = _planted(6, 1)
        dt, n_steps = 0.05, 40
        steps = np.array([5, 10, 15, 20, 25, 30, 35, 40])
        observed = _observe(theta0, omega, coupling, steps, dt, n_steps)
        rng = np.random.default_rng(99)
        guess = coupling + rng.uniform(-0.15, 0.15, size=(6, 6))
        guess = 0.5 * (guess + guess.T)
        np.fill_diagonal(guess, 0.0)
        error_before = float(np.max(np.abs(guess - coupling)))
        result = sysid.learn_coupling(
            theta0,
            omega,
            observed,
            steps,
            guess,
            dt,
            n_steps,
            max_iterations=300,
            learning_rate=2.0,
        )
        for earlier, later in zip(result.cost_history, result.cost_history[1:], strict=False):
            assert later <= earlier + 1e-12
        # The trajectory match improves by orders of magnitude and the coupling moves toward truth.
        assert result.cost_history[-1] < result.cost_history[0] * 1e-2
        error_after = float(np.max(np.abs(result.coupling - coupling)))
        assert error_after < error_before
        # The learned coupling stays symmetric with a zero diagonal.
        np.testing.assert_allclose(result.coupling, result.coupling.T, atol=1e-12)
        np.testing.assert_array_equal(np.diag(result.coupling), np.zeros(6))

    def test_converged_at_true_coupling(self) -> None:
        theta0, omega, coupling = _planted(5, 3)
        steps = np.array([10, 20, 30])
        observed = _observe(theta0, omega, coupling, steps)
        result = sysid.learn_coupling(
            theta0, omega, observed, steps, coupling, 0.05, 40, max_iterations=50
        )
        # Starting at the truth, the loss is already 0 and cannot improve.
        assert result.converged
        assert result.iterations == 1
        assert result.cost_history[0] < 1e-18

    def test_asymmetric_path_runs(self) -> None:
        theta0, omega, coupling = _planted(4, 13)
        steps = np.array([10, 20])
        observed = _observe(theta0, omega, coupling, steps)
        result = sysid.learn_coupling(
            theta0,
            omega,
            observed,
            steps,
            coupling + 0.1,
            0.05,
            40,
            max_iterations=5,
            symmetric=False,
        )
        assert isinstance(result, sysid.SystemIdentificationResult)
        assert result.coupling.shape == (4, 4)

    def test_iteration_budget_and_rejects_non_positive(self) -> None:
        theta0, omega, coupling = _planted(5, 2)
        steps = np.array([10, 20])
        observed = _observe(theta0, omega, coupling, steps)
        result = sysid.learn_coupling(
            theta0, omega, observed, steps, coupling + 0.1, 0.05, 40, max_iterations=3
        )
        assert result.iterations <= 3
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            sysid.learn_coupling(
                theta0, omega, observed, steps, coupling, 0.05, 40, max_iterations=0
            )

    def test_reproducible(self) -> None:
        theta0, omega, coupling = _planted(5, 4)
        steps = np.array([10, 20, 30])
        observed = _observe(theta0, omega, coupling, steps)
        guess = coupling + 0.1
        first = sysid.learn_coupling(
            theta0, omega, observed, steps, guess, 0.05, 40, max_iterations=20
        )
        second = sysid.learn_coupling(
            theta0, omega, observed, steps, guess, 0.05, 40, max_iterations=20
        )
        np.testing.assert_array_equal(first.coupling, second.coupling)
        assert first.cost_history == second.cost_history
