# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the differentiable adaptive Kuramoto integrator
"""Module-specific tests for :mod:`diff_kuramoto_adaptive`.

The gradient channels are checked against a central finite difference of the real
forward integrator :func:`integrate_adaptive_kuramoto` — the production surface the
module differentiates — for both a linear and a nonlinear (order-parameter)
terminal objective, validating the forward-mode sensitivity against ground truth
on every input channel (initial phases, initial coupling, frequencies, and the
Hebbian plasticity rate).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.accel import order_parameter, order_parameter_gradient
from scpn_quantum_control.accel.diff_kuramoto_adaptive import (
    AdaptiveGradients,
    adaptive_state_sensitivity,
    adaptive_terminal_value_and_grad,
)
from scpn_quantum_control.accel.kuramoto_adaptive import (
    hebbian_plasticity_rate,
    integrate_adaptive_kuramoto,
)
from scpn_quantum_control.accel.networked_kuramoto import networked_kuramoto_force

_DT = 0.02
_N_STEPS = 30
_EPS = 0.6


def _problem(n: int = 4, seed: int = 0) -> dict[str, Any]:
    """Build a deterministic adaptive problem."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=(n, n))
    coupling = 0.5 * (raw + raw.T)
    np.fill_diagonal(coupling, 0.0)
    return {
        "phases": rng.uniform(0.0, 2.0 * np.pi, size=n),
        "coupling": coupling,
        "omega": rng.standard_normal(n),
    }


def _terminal(
    phases: NDArray[np.float64],
    coupling: NDArray[np.float64],
    omega: NDArray[np.float64],
    plasticity_rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run the real forward integrator and return the terminal (phases, coupling)."""
    trajectory = integrate_adaptive_kuramoto(
        phases,
        coupling,
        omega,
        lambda t, k: networked_kuramoto_force(t, k),
        lambda t, k: hebbian_plasticity_rate(t, k, plasticity_rate=plasticity_rate),
        dt=_DT,
        n_steps=_N_STEPS,
    )
    return trajectory.terminal_phases, trajectory.terminal_coupling


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
    weight_coupling = rng.standard_normal((4, 4))

    def objective(theta: NDArray[np.float64], coupling: NDArray[np.float64]) -> float:
        return float(weight_phase @ theta + np.sum(weight_coupling * coupling))

    def objective_grad(
        theta: NDArray[np.float64], coupling: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return weight_phase.copy(), weight_coupling.copy()

    value, grads = adaptive_terminal_value_and_grad(
        problem["phases"],
        problem["coupling"],
        problem["omega"],
        plasticity_rate=_EPS,
        dt=_DT,
        n_steps=_N_STEPS,
        objective=objective,
        objective_grad=objective_grad,
    )

    def loss(
        phases: NDArray[np.float64],
        coupling: NDArray[np.float64],
        omega: NDArray[np.float64],
        eps: float,
    ) -> float:
        theta_n, coupling_n = _terminal(phases, coupling, omega, eps)
        return objective(theta_n, coupling_n)

    p = problem
    assert value == pytest.approx(loss(p["phases"], p["coupling"], p["omega"], _EPS))

    fd_phases = _central_difference(
        p["phases"], lambda x: loss(x, p["coupling"], p["omega"], _EPS)
    )
    fd_coupling = _central_difference(
        p["coupling"], lambda x: loss(p["phases"], x, p["omega"], _EPS)
    )
    fd_omega = _central_difference(p["omega"], lambda x: loss(p["phases"], p["coupling"], x, _EPS))
    eps = 1e-6
    fd_rate = (
        loss(p["phases"], p["coupling"], p["omega"], _EPS + eps)
        - loss(p["phases"], p["coupling"], p["omega"], _EPS - eps)
    ) / (2.0 * eps)

    assert grads.initial_phases == pytest.approx(fd_phases, abs=1e-7)
    assert grads.initial_coupling == pytest.approx(fd_coupling, abs=1e-7)
    assert grads.omega == pytest.approx(fd_omega, abs=1e-7)
    assert grads.plasticity_rate == pytest.approx(fd_rate, abs=1e-7)


def test_order_parameter_objective_matches_finite_difference() -> None:
    problem = _problem(seed=3)

    def objective(theta: NDArray[np.float64], coupling: NDArray[np.float64]) -> float:
        return float(order_parameter(theta))

    def objective_grad(
        theta: NDArray[np.float64], coupling: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return order_parameter_gradient(theta), np.zeros_like(coupling)

    _, grads = adaptive_terminal_value_and_grad(
        problem["phases"],
        problem["coupling"],
        problem["omega"],
        plasticity_rate=_EPS,
        dt=_DT,
        n_steps=_N_STEPS,
        objective=objective,
        objective_grad=objective_grad,
    )

    def loss_omega(omega: NDArray[np.float64]) -> float:
        theta_n, _ = _terminal(problem["phases"], problem["coupling"], omega, _EPS)
        return float(order_parameter(theta_n))

    fd_omega = _central_difference(problem["omega"], loss_omega)
    assert grads.omega == pytest.approx(fd_omega, abs=1e-7)


def test_state_sensitivity_shapes_and_terminal_match() -> None:
    problem = _problem(n=5, seed=1)
    theta_n, coupling_n, sensitivity = adaptive_state_sensitivity(
        problem["phases"],
        problem["coupling"],
        problem["omega"],
        plasticity_rate=_EPS,
        dt=_DT,
        n_steps=_N_STEPS,
    )
    n = 5
    assert theta_n.shape == (n,)
    assert coupling_n.shape == (n, n)
    assert sensitivity.shape == (n + n * n, 2 * n + n * n + 1)
    ref_theta, ref_coupling = _terminal(
        problem["phases"], problem["coupling"], problem["omega"], _EPS
    )
    assert theta_n == pytest.approx(ref_theta)
    assert coupling_n == pytest.approx(ref_coupling)


def test_gradients_are_deterministic() -> None:
    problem = _problem(seed=7)
    kwargs: dict[str, Any] = dict(
        plasticity_rate=_EPS,
        dt=_DT,
        n_steps=_N_STEPS,
        objective=lambda t, k: float(order_parameter(t)),
        objective_grad=lambda t, k: (order_parameter_gradient(t), np.zeros_like(k)),
    )
    _, a = adaptive_terminal_value_and_grad(
        problem["phases"], problem["coupling"], problem["omega"], **kwargs
    )
    _, b = adaptive_terminal_value_and_grad(
        problem["phases"], problem["coupling"], problem["omega"], **kwargs
    )
    assert np.array_equal(a.initial_coupling, b.initial_coupling)
    assert np.array_equal(a.omega, b.omega)


def test_returns_adaptive_gradients_type() -> None:
    problem = _problem()
    _, grads = adaptive_terminal_value_and_grad(
        problem["phases"],
        problem["coupling"],
        problem["omega"],
        plasticity_rate=_EPS,
        dt=_DT,
        n_steps=_N_STEPS,
        objective=lambda t, k: 0.0,
        objective_grad=lambda t, k: (np.zeros_like(t), np.zeros_like(k)),
    )
    assert isinstance(grads, AdaptiveGradients)


def test_objective_grad_wrong_shape_raises() -> None:
    problem = _problem()

    def bad_grad(
        theta: NDArray[np.float64], coupling: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.zeros(theta.size + 1), np.zeros_like(coupling)

    with pytest.raises(ValueError, match="objective_grad must return"):
        adaptive_terminal_value_and_grad(
            problem["phases"],
            problem["coupling"],
            problem["omega"],
            plasticity_rate=_EPS,
            dt=_DT,
            n_steps=_N_STEPS,
            objective=lambda t, k: 0.0,
            objective_grad=bad_grad,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"phases": np.zeros((2, 2))}, "phases must be a non-empty one-dimensional array"),
        ({"coupling": np.zeros((3, 3))}, "coupling must have shape"),
        ({"omega": np.zeros(3)}, "omega must have shape"),
        ({"plasticity_rate": -1.0}, "plasticity_rate must be non-negative"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"n_steps": 0}, "n_steps must be positive"),
    ],
)
def test_validation_errors(mutation: dict[str, Any], message: str) -> None:
    problem = _problem()
    call: dict[str, Any] = {
        "phases": problem["phases"],
        "coupling": problem["coupling"],
        "omega": problem["omega"],
        "plasticity_rate": _EPS,
        "dt": _DT,
        "n_steps": _N_STEPS,
    }
    call.update(mutation)
    with pytest.raises(ValueError, match=message):
        adaptive_state_sensitivity(
            call["phases"],
            call["coupling"],
            call["omega"],
            plasticity_rate=call["plasticity_rate"],
            dt=call["dt"],
            n_steps=call["n_steps"],
        )
