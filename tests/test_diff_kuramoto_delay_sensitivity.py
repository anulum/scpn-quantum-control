# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for the time-delayed Kuramoto delay sensitivity dtheta_N/dtau
r"""Tests for the delay sensitivity ``∂θ_N/∂τ`` of the time-delayed Kuramoto model.

Verify the forward-mode delay sensitivity against an independent central finite difference of the
terminal state over ``τ`` (the capability's own second witness), the objective delay gradient
``dL/dτ`` against a finite difference of the objective, and — by a first-order Taylor identity — the
forward map *and* the sensitivity together against the production method-of-steps integrator
:func:`~scpn_quantum_control.accel.kuramoto_delayed.integrate_delayed_kuramoto` evaluated at an
integer delay. Also cover the analytic zero-coupling case (no delay dependence), the internal
consistency of the objective gradient with the sensitivity, the rejection of a delay on a grid node
(where the derivative does not exist), and the input validation of every argument.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.accel.diff_kuramoto_delay_sensitivity import (
    delayed_delay_gradient,
    delayed_delay_sensitivity,
)
from scpn_quantum_control.accel.kuramoto_delayed import (
    delayed_networked_force,
    integrate_delayed_kuramoto,
)

_DT = 0.02
_N_STEPS = 100


def _problem(
    count: int = 4, seed: int = 20260703
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    history = rng.uniform(-math.pi, math.pi, count)
    omega = rng.uniform(-0.5, 0.5, count)
    coupling = rng.uniform(0.3, 0.8, (count, count))
    np.fill_diagonal(coupling, 0.0)
    return history, omega, coupling


def _terminal_phase(
    history: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    delay: float,
) -> NDArray[np.float64]:
    phases, _ = delayed_delay_sensitivity(
        history, omega, coupling, delay=delay, dt=_DT, n_steps=_N_STEPS
    )
    return phases


@pytest.mark.parametrize("delay", [0.366, 0.5123, 0.913])
@pytest.mark.parametrize("seed", [20260703, 7, 101])
def test_delay_sensitivity_matches_central_finite_difference(delay: float, seed: int) -> None:
    history, omega, coupling = _problem(seed=seed)
    _, dtheta_dtau = delayed_delay_sensitivity(
        history, omega, coupling, delay=delay, dt=_DT, n_steps=_N_STEPS
    )
    epsilon = 1e-6
    forward = _terminal_phase(history, omega, coupling, delay + epsilon)
    backward = _terminal_phase(history, omega, coupling, delay - epsilon)
    finite_difference = (forward - backward) / (2.0 * epsilon)
    assert np.allclose(dtheta_dtau, finite_difference, atol=1e-6, rtol=1e-5)


def test_objective_delay_gradient_matches_finite_difference() -> None:
    history, omega, coupling = _problem()
    delay = 0.417

    def objective(theta: NDArray[np.float64]) -> float:
        return float(np.sin(theta).sum())

    def objective_grad(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.cos(theta)

    value, d_value = delayed_delay_gradient(
        history,
        omega,
        coupling,
        delay=delay,
        dt=_DT,
        n_steps=_N_STEPS,
        objective=objective,
        objective_grad=objective_grad,
    )
    assert value == pytest.approx(
        float(np.sin(_terminal_phase(history, omega, coupling, delay)).sum())
    )
    epsilon = 1e-6
    forward = float(np.sin(_terminal_phase(history, omega, coupling, delay + epsilon)).sum())
    backward = float(np.sin(_terminal_phase(history, omega, coupling, delay - epsilon)).sum())
    assert d_value == pytest.approx((forward - backward) / (2.0 * epsilon), abs=1e-6)


def test_forward_map_and_sensitivity_reproduce_the_production_integrator_by_taylor() -> None:
    # At an integer delay the production method-of-steps integrator is the reference. The sensitivity
    # function refuses that grid node, so approach it: a first-order Taylor step from just above it,
    # theta_N(m*dt + eps) - eps * dtheta/dtau(m*dt + eps), must reproduce the production terminal
    # phase to O(eps^2) — validating the forward map AND the sensitivity against production at once.
    history, omega, coupling = _problem()
    delay_steps = 8
    node = delay_steps * _DT
    epsilon = 0.01 * _DT
    phases, dtheta_dtau = delayed_delay_sensitivity(
        history, omega, coupling, delay=node + epsilon, dt=_DT, n_steps=_N_STEPS
    )
    taylor = phases - epsilon * dtheta_dtau

    constant_history = np.tile(history, (delay_steps + 1, 1))
    production = integrate_delayed_kuramoto(
        constant_history,
        omega,
        lambda current, lagged: delayed_networked_force(current, lagged, coupling),
        delay=node,
        dt=_DT,
        n_steps=_N_STEPS,
    )
    assert np.allclose(taylor, production.phases[-1], atol=1e-5)


def test_zero_coupling_has_zero_delay_sensitivity() -> None:
    # With no coupling the delayed term vanishes, theta(t) = theta0 + omega t is independent of tau,
    # so the delay sensitivity is exactly zero.
    history, omega, _ = _problem()
    coupling = np.zeros((history.size, history.size), dtype=np.float64)
    phases, dtheta_dtau = delayed_delay_sensitivity(
        history, omega, coupling, delay=0.313, dt=_DT, n_steps=_N_STEPS
    )
    assert np.allclose(dtheta_dtau, 0.0, atol=1e-14)
    assert np.allclose(phases, history + omega * (_N_STEPS * _DT))


def test_gradient_is_consistent_with_the_sensitivity() -> None:
    history, omega, coupling = _problem()
    delay = 0.371
    weights = np.linspace(0.5, 2.0, history.size)

    def objective(theta: NDArray[np.float64]) -> float:
        return float(weights @ theta)

    def objective_grad(_theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return weights

    _, d_value = delayed_delay_gradient(
        history,
        omega,
        coupling,
        delay=delay,
        dt=_DT,
        n_steps=_N_STEPS,
        objective=objective,
        objective_grad=objective_grad,
    )
    _, dtheta_dtau = delayed_delay_sensitivity(
        history, omega, coupling, delay=delay, dt=_DT, n_steps=_N_STEPS
    )
    assert d_value == pytest.approx(float(weights @ dtheta_dtau))


@pytest.mark.parametrize("multiple", [1.0, 2.0, 3.0, 1.5, 2.5])
def test_rejects_delay_on_a_grid_node(multiple: float) -> None:
    history, omega, coupling = _problem()
    with pytest.raises(ValueError, match="half-integer multiple of dt"):
        delayed_delay_sensitivity(
            history, omega, coupling, delay=multiple * _DT, dt=_DT, n_steps=_N_STEPS
        )


def test_rejects_malformed_inputs() -> None:
    history, omega, coupling = _problem()
    good = {"dt": _DT, "n_steps": _N_STEPS}
    with pytest.raises(ValueError, match="one-dimensional constant history"):
        delayed_delay_sensitivity(history[:, None], omega, coupling, delay=0.31, **good)
    with pytest.raises(ValueError, match="at least one oscillator"):
        delayed_delay_sensitivity(np.empty(0), np.empty(0), np.empty((0, 0)), delay=0.31, **good)
    with pytest.raises(ValueError, match="omega must have shape"):
        delayed_delay_sensitivity(history, omega[:-1], coupling, delay=0.31, **good)
    with pytest.raises(ValueError, match="coupling must have shape"):
        delayed_delay_sensitivity(history, omega, coupling[:-1], delay=0.31, **good)
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        delayed_delay_sensitivity(history, omega, coupling, delay=0.31, dt=0.0, n_steps=_N_STEPS)
    with pytest.raises(ValueError, match="delay must be finite and positive"):
        delayed_delay_sensitivity(history, omega, coupling, delay=0.0, dt=_DT, n_steps=_N_STEPS)
    with pytest.raises(ValueError, match="delay must be at least dt"):
        delayed_delay_sensitivity(
            history, omega, coupling, delay=0.5 * _DT, dt=_DT, n_steps=_N_STEPS
        )
    with pytest.raises(ValueError, match="n_steps must be a positive integer"):
        delayed_delay_sensitivity(history, omega, coupling, delay=0.31, dt=_DT, n_steps=0)


def test_rejects_wrong_shape_cotangent() -> None:
    history, omega, coupling = _problem()
    with pytest.raises(ValueError, match="cotangent"):
        delayed_delay_gradient(
            history,
            omega,
            coupling,
            delay=0.313,
            dt=_DT,
            n_steps=_N_STEPS,
            objective=lambda theta: float(theta.sum()),
            objective_grad=lambda theta: theta[:-1],
        )
