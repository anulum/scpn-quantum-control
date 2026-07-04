# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Differentiable Kuramoto control objectives and value-and-grad tests
"""Multi-angle tests for the differentiable Kuramoto control objectives and value-and-grad.

Covers each terminal objective (closed-form value, cotangent against the finite-difference of
the cost), the composed ``terminal_objective_value_and_grad`` against finite differences for
both integrators and every gradient channel, the convenience synchronisation wrapper, integrator
validation, the phase-target shape guard, and a multi-step gradient-descent optimisation that
drives the order parameter up (demonstrating optimal-coupling control end to end).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import oscillatools.accel.diff_kuramoto_euler as dk_euler
import oscillatools.accel.diff_kuramoto_rk4 as dk_rk4
import oscillatools.accel.kuramoto_control as kc
from oscillatools.accel import order_parameter

_GLOBAL_SETTINGS = settings(max_examples=25, deadline=None)


def _random_problem(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-math.pi, math.pi, size=n)
    omega = rng.uniform(-1.0, 1.0, size=n)
    coupling = rng.uniform(0.0, 0.4, size=(n, n))
    coupling = 0.5 * (coupling + coupling.T)
    np.fill_diagonal(coupling, 0.0)
    return theta0, omega, coupling


def _finite_difference(cost: object, base: np.ndarray, step: float = 1e-6) -> np.ndarray:
    out = np.zeros_like(base, dtype=np.float64)
    flat = base.reshape(-1)
    grad_flat = out.reshape(-1)
    for i in range(flat.size):
        plus = base.astype(np.float64).copy().reshape(-1)
        minus = base.astype(np.float64).copy().reshape(-1)
        plus[i] += step
        minus[i] -= step
        grad_flat[i] = (
            cost(plus.reshape(base.shape)) - cost(minus.reshape(base.shape))  # type: ignore[operator]
        ) / (2.0 * step)
    return out


class TestObjectives:
    def test_coherence_value_and_cotangent(self) -> None:
        rng = np.random.default_rng(1)
        theta = rng.uniform(-math.pi, math.pi, size=7)
        cost, cotangent = kc.coherence_objective(theta)
        radius = float(order_parameter(theta))
        assert abs(cost - (1.0 - radius**2)) < 1e-13
        fd = _finite_difference(lambda t: 1.0 - float(order_parameter(t)) ** 2, theta)
        np.testing.assert_allclose(cotangent, fd, atol=1e-6)

    def test_phase_target_value_and_cotangent(self) -> None:
        rng = np.random.default_rng(2)
        theta = rng.uniform(-math.pi, math.pi, size=6)
        target = rng.uniform(-math.pi, math.pi, size=6)
        objective = kc.phase_target_objective(target)
        cost, cotangent = objective(theta)
        assert abs(cost - float(np.mean(1.0 - np.cos(theta - target)))) < 1e-13
        fd = _finite_difference(lambda t: float(np.mean(1.0 - np.cos(t - target))), theta)
        np.testing.assert_allclose(cotangent, fd, atol=1e-6)

    def test_phase_target_rejects_shape_mismatch(self) -> None:
        objective = kc.phase_target_objective(np.zeros(4))
        with pytest.raises(ValueError, match="target must have shape"):
            objective(np.zeros(3))

    def test_interaction_energy_value_and_cotangent(self) -> None:
        rng = np.random.default_rng(3)
        theta = rng.uniform(-math.pi, math.pi, size=6)
        coupling = rng.uniform(0.0, 1.0, size=(6, 6))
        coupling = 0.5 * (coupling + coupling.T)
        np.fill_diagonal(coupling, 0.0)
        objective = kc.interaction_energy_objective(coupling)
        cost, cotangent = objective(theta)
        expected = -0.5 * float((coupling * np.cos(theta[:, None] - theta[None, :])).sum())
        assert abs(cost - expected) < 1e-11
        fd = _finite_difference(
            lambda t: -0.5 * float((coupling * np.cos(t[:, None] - t[None, :])).sum()), theta
        )
        np.testing.assert_allclose(cotangent, fd, atol=1e-6)


class TestValueAndGrad:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=7),
        n_steps=st.integers(min_value=1, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        integrator=st.sampled_from(["rk4", "euler"]),
    )
    def test_coherence_gradients_match_finite_difference(
        self, n: int, n_steps: int, seed: int, integrator: str
    ) -> None:
        theta0, omega, coupling = _random_problem(n, seed)
        dt = 0.05
        cost, grad_theta0, grad_omega, grad_coupling = kc.terminal_objective_value_and_grad(
            kc.coherence_objective, theta0, omega, coupling, dt, n_steps, integrator=integrator
        )
        forward = (
            dk_rk4._python_kuramoto_rk4_trajectory
            if integrator == "rk4"
            else dk_euler._python_kuramoto_euler_trajectory
        )

        def loss(t0: np.ndarray, om: np.ndarray, k: np.ndarray) -> float:
            final = forward(t0, om, k, dt, n_steps)[-1]
            return 1.0 - float(order_parameter(final)) ** 2

        assert abs(cost - loss(theta0, omega, coupling)) < 1e-9
        np.testing.assert_allclose(
            grad_theta0, _finite_difference(lambda t: loss(t, omega, coupling), theta0), atol=1e-5
        )
        np.testing.assert_allclose(
            grad_omega, _finite_difference(lambda o: loss(theta0, o, coupling), omega), atol=1e-5
        )
        np.testing.assert_allclose(
            grad_coupling,
            _finite_difference(lambda k: loss(theta0, omega, k), coupling),
            atol=1e-5,
        )

    def test_phase_target_gradients_match_finite_difference(self) -> None:
        theta0, omega, coupling = _random_problem(5, 9)
        target = np.zeros(5)
        dt, n_steps = 0.05, 12
        objective = kc.phase_target_objective(target)
        _, _, _, grad_coupling = kc.terminal_objective_value_and_grad(
            objective, theta0, omega, coupling, dt, n_steps, integrator="rk4"
        )

        def loss(k: np.ndarray) -> float:
            final = dk_rk4._python_kuramoto_rk4_trajectory(theta0, omega, k, dt, n_steps)[-1]
            return float(np.mean(1.0 - np.cos(final - target)))

        np.testing.assert_allclose(grad_coupling, _finite_difference(loss, coupling), atol=1e-5)

    def test_synchronisation_wrapper_matches_objective_path(self) -> None:
        theta0, omega, coupling = _random_problem(6, 4)
        dt, n_steps = 0.05, 10
        direct = kc.terminal_objective_value_and_grad(
            kc.coherence_objective, theta0, omega, coupling, dt, n_steps
        )
        wrapped = kc.synchronisation_value_and_grad(theta0, omega, coupling, dt, n_steps)
        assert abs(direct[0] - wrapped[0]) < 1e-15
        for a, b in zip(direct[1:], wrapped[1:], strict=True):
            np.testing.assert_array_equal(a, b)

    def test_rejects_unknown_integrator(self) -> None:
        theta0, omega, coupling = _random_problem(3, 1)
        with pytest.raises(ValueError, match="integrator must be 'rk4' or 'euler'"):
            kc.terminal_objective_value_and_grad(
                kc.coherence_objective, theta0, omega, coupling, 0.05, 5, integrator="midpoint"
            )

    def test_gradient_descent_drives_synchronisation(self) -> None:
        # End-to-end optimal-coupling control: gradient descent on K raises the order parameter.
        theta0, omega, coupling = _random_problem(10, 21)
        dt, n_steps, rate = 0.05, 30, 2.0
        radius_initial = float(
            order_parameter(
                dk_rk4._python_kuramoto_rk4_trajectory(theta0, omega, coupling, dt, n_steps)[-1]
            )
        )
        current = coupling.copy()
        for _ in range(15):
            _, _, _, grad_coupling = kc.synchronisation_value_and_grad(
                theta0, omega, current, dt, n_steps
            )
            current = np.clip(current - rate * grad_coupling, 0.0, None)
            np.fill_diagonal(current, 0.0)
        radius_final = float(
            order_parameter(
                dk_rk4._python_kuramoto_rk4_trajectory(theta0, omega, current, dt, n_steps)[-1]
            )
        )
        assert radius_final > radius_initial + 0.05


class TestTerminalObjectiveValue:
    @pytest.mark.parametrize("integrator", ["rk4", "euler"])
    def test_matches_value_and_grad_cost(self, integrator: str) -> None:
        theta0, omega, coupling = _random_problem(6, 31)
        dt, n_steps = 0.05, 12
        value = kc.terminal_objective_value(
            kc.coherence_objective, theta0, omega, coupling, dt, n_steps, integrator=integrator
        )
        cost, *_ = kc.terminal_objective_value_and_grad(
            kc.coherence_objective, theta0, omega, coupling, dt, n_steps, integrator=integrator
        )
        assert abs(value - cost) < 1e-12

    def test_rejects_unknown_integrator(self) -> None:
        theta0, omega, coupling = _random_problem(3, 1)
        with pytest.raises(ValueError, match="integrator must be 'rk4' or 'euler'"):
            kc.terminal_objective_value(
                kc.coherence_objective, theta0, omega, coupling, 0.05, 5, integrator="midpoint"
            )
