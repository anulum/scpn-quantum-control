# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the differentiable inertial Kuramoto integrator
"""Module-specific tests for :mod:`diff_kuramoto_inertial`.

The gradient channels are checked against a central finite difference of the real
forward integrator :func:`integrate_inertial` — the production surface the module
differentiates — for both a linear and a nonlinear (order-parameter) terminal
objective, so the forward-mode sensitivity is validated against ground truth on
every input channel (initial phases, initial velocities, frequencies, coupling,
inertia, damping).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel import order_parameter, order_parameter_gradient
from oscillatools.accel.diff_kuramoto_inertial import (
    InertialGradients,
    inertial_state_sensitivity,
    inertial_terminal_value_and_grad,
)
from oscillatools.accel.kuramoto_inertial import integrate_inertial
from oscillatools.accel.networked_kuramoto import networked_kuramoto_force

_DT = 0.02
_N_STEPS = 30
_MASS = 1.7
_DAMPING = 0.8


def _problem(n: int = 4, seed: int = 0) -> dict[str, Any]:
    """Build a deterministic inertial problem."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=(n, n))
    coupling = 0.5 * (raw + raw.T)
    np.fill_diagonal(coupling, 0.0)
    return {
        "phases": rng.uniform(0.0, 2.0 * np.pi, size=n),
        "velocities": rng.standard_normal(n) * 0.3,
        "omega": rng.standard_normal(n),
        "coupling": coupling,
    }


def _terminal_phases_velocities(
    phases: NDArray[np.float64],
    velocities: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run the real forward integrator and return the terminal state."""
    trajectory = integrate_inertial(
        phases,
        velocities,
        omega,
        lambda t: networked_kuramoto_force(t, coupling),
        _MASS,
        damping=_DAMPING,
        dt=_DT,
        n_steps=_N_STEPS,
    )
    return trajectory.terminal_phases, trajectory.terminal_velocities


def _central_difference(base: NDArray[np.float64], scalar_loss: Any) -> NDArray[np.float64]:
    """Central finite difference of ``scalar_loss`` over every entry of ``base``."""
    eps = 1e-6
    grad = np.zeros_like(base)
    for index in np.ndindex(base.shape):
        high = base.copy()
        high[index] += eps
        low = base.copy()
        low[index] -= eps
        grad[index] = (scalar_loss(high) - scalar_loss(low)) / (2.0 * eps)
    return grad


def test_linear_objective_matches_finite_difference_on_every_channel() -> None:
    problem = _problem()
    rng = np.random.default_rng(99)
    weight_phase = rng.standard_normal(4)
    weight_velocity = rng.standard_normal(4)

    def objective(theta: NDArray[np.float64], velocity: NDArray[np.float64]) -> float:
        return float(weight_phase @ theta + weight_velocity @ velocity)

    def objective_grad(
        theta: NDArray[np.float64], velocity: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return weight_phase.copy(), weight_velocity.copy()

    value, grads = inertial_terminal_value_and_grad(
        problem["phases"],
        problem["velocities"],
        problem["omega"],
        problem["coupling"],
        _MASS,
        damping=_DAMPING,
        dt=_DT,
        n_steps=_N_STEPS,
        objective=objective,
        objective_grad=objective_grad,
    )

    def loss(
        phases: NDArray[np.float64],
        velocities: NDArray[np.float64],
        omega: NDArray[np.float64],
        coupling: NDArray[np.float64],
        mass: float,
        damping: float,
    ) -> float:
        trajectory = integrate_inertial(
            phases,
            velocities,
            omega,
            lambda t: networked_kuramoto_force(t, coupling),
            mass,
            damping=damping,
            dt=_DT,
            n_steps=_N_STEPS,
        )
        return objective(trajectory.terminal_phases, trajectory.terminal_velocities)

    p = problem
    assert value == pytest.approx(
        loss(p["phases"], p["velocities"], p["omega"], p["coupling"], _MASS, _DAMPING)
    )

    fd_theta0 = _central_difference(
        p["phases"], lambda x: loss(x, p["velocities"], p["omega"], p["coupling"], _MASS, _DAMPING)
    )
    fd_v0 = _central_difference(
        p["velocities"], lambda x: loss(p["phases"], x, p["omega"], p["coupling"], _MASS, _DAMPING)
    )
    fd_omega = _central_difference(
        p["omega"], lambda x: loss(p["phases"], p["velocities"], x, p["coupling"], _MASS, _DAMPING)
    )
    fd_coupling = _central_difference(
        p["coupling"], lambda x: loss(p["phases"], p["velocities"], p["omega"], x, _MASS, _DAMPING)
    )
    eps = 1e-6
    fd_mass = (
        loss(p["phases"], p["velocities"], p["omega"], p["coupling"], _MASS + eps, _DAMPING)
        - loss(p["phases"], p["velocities"], p["omega"], p["coupling"], _MASS - eps, _DAMPING)
    ) / (2.0 * eps)
    fd_damping = (
        loss(p["phases"], p["velocities"], p["omega"], p["coupling"], _MASS, _DAMPING + eps)
        - loss(p["phases"], p["velocities"], p["omega"], p["coupling"], _MASS, _DAMPING - eps)
    ) / (2.0 * eps)

    assert grads.initial_phases == pytest.approx(fd_theta0, abs=1e-7)
    assert grads.initial_velocities == pytest.approx(fd_v0, abs=1e-7)
    assert grads.omega == pytest.approx(fd_omega, abs=1e-7)
    assert grads.coupling == pytest.approx(fd_coupling, abs=1e-7)
    assert grads.mass == pytest.approx(fd_mass, abs=1e-7)
    assert grads.damping == pytest.approx(fd_damping, abs=1e-7)


def test_order_parameter_objective_matches_finite_difference() -> None:
    problem = _problem(seed=3)

    def objective(theta: NDArray[np.float64], velocity: NDArray[np.float64]) -> float:
        return float(order_parameter(theta))

    def objective_grad(
        theta: NDArray[np.float64], velocity: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return order_parameter_gradient(theta), np.zeros_like(velocity)

    _, grads = inertial_terminal_value_and_grad(
        problem["phases"],
        problem["velocities"],
        problem["omega"],
        problem["coupling"],
        _MASS,
        damping=_DAMPING,
        dt=_DT,
        n_steps=_N_STEPS,
        objective=objective,
        objective_grad=objective_grad,
    )

    def loss_coupling(coupling: NDArray[np.float64]) -> float:
        theta_n, _ = _terminal_phases_velocities(
            problem["phases"], problem["velocities"], problem["omega"], coupling
        )
        return float(order_parameter(theta_n))

    fd_coupling = _central_difference(problem["coupling"], loss_coupling)
    assert grads.coupling == pytest.approx(fd_coupling, abs=1e-7)


def test_state_sensitivity_shapes_and_terminal_match() -> None:
    problem = _problem(n=5, seed=1)
    theta_n, velocity_n, sensitivity = inertial_state_sensitivity(
        problem["phases"],
        problem["velocities"],
        problem["omega"],
        problem["coupling"],
        _MASS,
        damping=_DAMPING,
        dt=_DT,
        n_steps=_N_STEPS,
    )
    n = 5
    assert theta_n.shape == (n,)
    assert velocity_n.shape == (n,)
    assert sensitivity.shape == (2 * n, 3 * n + n * n + 2)
    ref_theta, ref_velocity = _terminal_phases_velocities(
        problem["phases"], problem["velocities"], problem["omega"], problem["coupling"]
    )
    assert theta_n == pytest.approx(ref_theta)
    assert velocity_n == pytest.approx(ref_velocity)


def test_gradients_are_deterministic() -> None:
    problem = _problem(seed=7)
    kwargs: dict[str, Any] = dict(
        mass=_MASS,
        damping=_DAMPING,
        dt=_DT,
        n_steps=_N_STEPS,
        objective=lambda t, v: float(order_parameter(t)),
        objective_grad=lambda t, v: (order_parameter_gradient(t), np.zeros_like(v)),
    )
    _, a = inertial_terminal_value_and_grad(
        problem["phases"], problem["velocities"], problem["omega"], problem["coupling"], **kwargs
    )
    _, b = inertial_terminal_value_and_grad(
        problem["phases"], problem["velocities"], problem["omega"], problem["coupling"], **kwargs
    )
    assert np.array_equal(a.coupling, b.coupling)
    assert np.array_equal(a.omega, b.omega)


def test_objective_grad_wrong_shape_raises() -> None:
    problem = _problem()

    def bad_grad(
        theta: NDArray[np.float64], velocity: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.zeros(theta.size + 1), np.zeros_like(velocity)

    with pytest.raises(ValueError, match="objective_grad must return"):
        inertial_terminal_value_and_grad(
            problem["phases"],
            problem["velocities"],
            problem["omega"],
            problem["coupling"],
            _MASS,
            damping=_DAMPING,
            dt=_DT,
            n_steps=_N_STEPS,
            objective=lambda t, v: 0.0,
            objective_grad=bad_grad,
        )


def test_returns_inertial_gradients_type() -> None:
    problem = _problem()
    _, grads = inertial_terminal_value_and_grad(
        problem["phases"],
        problem["velocities"],
        problem["omega"],
        problem["coupling"],
        _MASS,
        damping=_DAMPING,
        dt=_DT,
        n_steps=_N_STEPS,
        objective=lambda t, v: 0.0,
        objective_grad=lambda t, v: (np.zeros_like(t), np.zeros_like(v)),
    )
    assert isinstance(grads, InertialGradients)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"omega": np.zeros((2, 2))}, "omega must be a non-empty one-dimensional array"),
        ({"phases": np.zeros(3)}, "phases must have shape"),
        ({"velocities": np.zeros(3)}, "velocities must have shape"),
        ({"coupling": np.zeros((3, 3))}, "coupling must have shape"),
        ({"mass": 0.0}, "mass must be positive"),
        ({"damping": -1.0}, "damping must be non-negative"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"n_steps": 0}, "n_steps must be positive"),
    ],
)
def test_validation_errors(mutation: dict[str, Any], message: str) -> None:
    problem = _problem()
    call: dict[str, Any] = {
        "phases": problem["phases"],
        "velocities": problem["velocities"],
        "omega": problem["omega"],
        "coupling": problem["coupling"],
        "mass": _MASS,
        "damping": _DAMPING,
        "dt": _DT,
        "n_steps": _N_STEPS,
    }
    call.update(mutation)
    with pytest.raises(ValueError, match=message):
        inertial_state_sensitivity(
            call["phases"],
            call["velocities"],
            call["omega"],
            call["coupling"],
            call["mass"],
            damping=call["damping"],
            dt=call["dt"],
            n_steps=call["n_steps"],
        )
