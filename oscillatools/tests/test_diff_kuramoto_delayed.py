# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the differentiable time-delayed Kuramoto integrator
"""Module-specific tests for :mod:`diff_kuramoto_delayed`.

The method-of-steps gradients are checked against a central finite difference of the
real forward integrator :func:`integrate_delayed_kuramoto` for every input channel —
the full initial history on ``[-τ, 0]``, the frequencies, and the coupling — and a
separate test asserts the differentiable pass reproduces the integrator's exact
trajectory.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel import order_parameter, order_parameter_gradient
from oscillatools.accel.diff_kuramoto_delayed import (
    DelayedGradients,
    delayed_phase_sensitivity,
    delayed_terminal_value_and_grad,
)
from oscillatools.accel.kuramoto_delayed import (
    delayed_networked_force,
    integrate_delayed_kuramoto,
)

_DT = 0.05
_TAU = 0.15
_N_STEPS = 20
_DELAY_STEPS = int(round(_TAU / _DT))


def _problem(n: int = 3, seed: int = 0) -> dict[str, Any]:
    """Build a deterministic delayed problem with a full initial history."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=(n, n))
    coupling = 0.5 * (raw + raw.T)
    np.fill_diagonal(coupling, 0.0)
    return {
        "history": rng.uniform(0.0, 2.0 * np.pi, size=(_DELAY_STEPS + 1, n)),
        "omega": rng.standard_normal(n),
        "coupling": coupling,
    }


def _terminal_phases(
    history: NDArray[np.float64], omega: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Run the real forward integrator and return the terminal phases."""
    run = integrate_delayed_kuramoto(
        history,
        omega,
        lambda cur, lag: delayed_networked_force(cur, lag, coupling),
        delay=_TAU,
        dt=_DT,
        n_steps=_N_STEPS,
    )
    return run.terminal_phases


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


def test_sensitivity_reproduces_forward_trajectory() -> None:
    problem = _problem()
    theta_n, _ = delayed_phase_sensitivity(
        problem["history"],
        problem["omega"],
        problem["coupling"],
        delay=_TAU,
        dt=_DT,
        n_steps=_N_STEPS,
    )
    reference = _terminal_phases(problem["history"], problem["omega"], problem["coupling"])
    assert theta_n == pytest.approx(reference, abs=0.0)


def test_gradient_matches_finite_difference_on_every_channel() -> None:
    problem = _problem()

    def objective(theta: NDArray[np.float64]) -> float:
        return float(order_parameter(theta))

    def objective_grad(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return order_parameter_gradient(theta)

    value, grads = delayed_terminal_value_and_grad(
        problem["history"],
        problem["omega"],
        problem["coupling"],
        delay=_TAU,
        dt=_DT,
        n_steps=_N_STEPS,
        objective=objective,
        objective_grad=objective_grad,
    )

    def loss(
        history: NDArray[np.float64], omega: NDArray[np.float64], coupling: NDArray[np.float64]
    ) -> float:
        return float(order_parameter(_terminal_phases(history, omega, coupling)))

    p = problem
    assert value == pytest.approx(loss(p["history"], p["omega"], p["coupling"]))

    fd_history = _central_difference(p["history"], lambda x: loss(x, p["omega"], p["coupling"]))
    fd_omega = _central_difference(p["omega"], lambda x: loss(p["history"], x, p["coupling"]))
    fd_coupling = _central_difference(p["coupling"], lambda x: loss(p["history"], p["omega"], x))

    assert grads.initial_history == pytest.approx(fd_history, abs=1e-7)
    assert grads.omega == pytest.approx(fd_omega, abs=1e-7)
    assert grads.coupling == pytest.approx(fd_coupling, abs=1e-7)


def test_phase_sensitivity_shape() -> None:
    problem = _problem(n=4, seed=1)
    theta_n, sensitivity = delayed_phase_sensitivity(
        problem["history"],
        problem["omega"],
        problem["coupling"],
        delay=_TAU,
        dt=_DT,
        n_steps=_N_STEPS,
    )
    n = 4
    assert theta_n.shape == (n,)
    assert sensitivity.shape == (n, (_DELAY_STEPS + 1) * n + n + n * n)


def test_gradients_are_deterministic() -> None:
    problem = _problem(seed=7)
    kwargs: dict[str, Any] = dict(
        delay=_TAU,
        dt=_DT,
        n_steps=_N_STEPS,
        objective=lambda t: float(order_parameter(t)),
        objective_grad=lambda t: order_parameter_gradient(t),
    )
    _, a = delayed_terminal_value_and_grad(
        problem["history"], problem["omega"], problem["coupling"], **kwargs
    )
    _, b = delayed_terminal_value_and_grad(
        problem["history"], problem["omega"], problem["coupling"], **kwargs
    )
    assert np.array_equal(a.initial_history, b.initial_history)
    assert np.array_equal(a.coupling, b.coupling)


def test_returns_delayed_gradients_type() -> None:
    problem = _problem()
    _, grads = delayed_terminal_value_and_grad(
        problem["history"],
        problem["omega"],
        problem["coupling"],
        delay=_TAU,
        dt=_DT,
        n_steps=_N_STEPS,
        objective=lambda t: 0.0,
        objective_grad=lambda t: np.zeros_like(t),
    )
    assert isinstance(grads, DelayedGradients)


def test_objective_grad_wrong_shape_raises() -> None:
    problem = _problem()
    with pytest.raises(ValueError, match="objective_grad must return"):
        delayed_terminal_value_and_grad(
            problem["history"],
            problem["omega"],
            problem["coupling"],
            delay=_TAU,
            dt=_DT,
            n_steps=_N_STEPS,
            objective=lambda t: 0.0,
            objective_grad=lambda t: np.zeros(t.size + 1),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"dt": 0.0}, "dt must be positive"),
        ({"delay": 0.0}, "delay must be positive"),
        ({"n_steps": 0}, "n_steps must be positive"),
        ({"delay": 0.075}, "delay must be a positive integer multiple of dt"),
        ({"omega": np.zeros((2, 2))}, "omega must be a non-empty one-dimensional array"),
        ({"coupling": np.zeros((5, 5))}, "coupling must have shape"),
        ({"history": np.zeros((2, 3))}, "initial_history must have shape"),
    ],
)
def test_validation_errors(mutation: dict[str, Any], message: str) -> None:
    problem = _problem()
    call: dict[str, Any] = {
        "history": problem["history"],
        "omega": problem["omega"],
        "coupling": problem["coupling"],
        "delay": _TAU,
        "dt": _DT,
        "n_steps": _N_STEPS,
    }
    call.update(mutation)
    with pytest.raises(ValueError, match=message):
        delayed_phase_sensitivity(
            call["history"],
            call["omega"],
            call["coupling"],
            delay=call["delay"],
            dt=call["dt"],
            n_steps=call["n_steps"],
        )
