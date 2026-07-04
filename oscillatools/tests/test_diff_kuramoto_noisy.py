# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the differentiable noisy Kuramoto integrator
"""Module-specific tests for :mod:`diff_kuramoto_noisy`.

The pathwise gradients are checked against a central finite difference of the real
forward integrator :func:`integrate_noisy_kuramoto` *at a fixed seed* — the frozen
Brownian path under which the stochastic recursion is a deterministic map — for the
initial phases, frequencies, coupling, and the diffusion intensity. A separate test
asserts the differentiable pass reproduces the forward integrator's exact noise path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel import order_parameter, order_parameter_gradient
from oscillatools.accel.diff_kuramoto_noisy import (
    NoisyGradients,
    noisy_phase_sensitivity,
    noisy_terminal_value_and_grad,
)
from oscillatools.accel.kuramoto_noisy import integrate_noisy_kuramoto
from oscillatools.accel.networked_kuramoto import networked_kuramoto_force

_DT = 0.02
_N_STEPS = 25
_D = 0.5
_SEED = 123


def _problem(n: int = 4, seed: int = 0) -> dict[str, Any]:
    """Build a deterministic noisy problem (the RNG here seeds the *inputs*, not the noise)."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=(n, n))
    coupling = 0.5 * (raw + raw.T)
    np.fill_diagonal(coupling, 0.0)
    return {
        "phases": rng.uniform(0.0, 2.0 * np.pi, size=n),
        "omega": rng.standard_normal(n),
        "coupling": coupling,
    }


def _terminal_phases(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    diffusion: float,
) -> NDArray[np.float64]:
    """Run the real seeded forward integrator and return the terminal phases."""
    run = integrate_noisy_kuramoto(
        phases,
        omega,
        lambda t: networked_kuramoto_force(t, coupling),
        diffusion=diffusion,
        dt=_DT,
        n_steps=_N_STEPS,
        seed=_SEED,
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


def test_sensitivity_reproduces_forward_noise_path() -> None:
    problem = _problem()
    theta_n, _ = noisy_phase_sensitivity(
        problem["phases"],
        problem["omega"],
        problem["coupling"],
        diffusion=_D,
        dt=_DT,
        n_steps=_N_STEPS,
        seed=_SEED,
    )
    reference = _terminal_phases(problem["phases"], problem["omega"], problem["coupling"], _D)
    assert theta_n == pytest.approx(reference, abs=0.0)


def test_pathwise_gradient_matches_finite_difference_on_every_channel() -> None:
    problem = _problem()

    def objective(theta: NDArray[np.float64]) -> float:
        return float(order_parameter(theta))

    def objective_grad(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return order_parameter_gradient(theta)

    value, grads = noisy_terminal_value_and_grad(
        problem["phases"],
        problem["omega"],
        problem["coupling"],
        diffusion=_D,
        dt=_DT,
        n_steps=_N_STEPS,
        seed=_SEED,
        objective=objective,
        objective_grad=objective_grad,
    )

    def loss(
        phases: NDArray[np.float64],
        omega: NDArray[np.float64],
        coupling: NDArray[np.float64],
        diffusion: float,
    ) -> float:
        return float(order_parameter(_terminal_phases(phases, omega, coupling, diffusion)))

    p = problem
    assert value == pytest.approx(loss(p["phases"], p["omega"], p["coupling"], _D))

    fd_phases = _central_difference(p["phases"], lambda x: loss(x, p["omega"], p["coupling"], _D))
    fd_omega = _central_difference(p["omega"], lambda x: loss(p["phases"], x, p["coupling"], _D))
    fd_coupling = _central_difference(
        p["coupling"], lambda x: loss(p["phases"], p["omega"], x, _D)
    )
    eps = 1e-6
    fd_diffusion = (
        loss(p["phases"], p["omega"], p["coupling"], _D + eps)
        - loss(p["phases"], p["omega"], p["coupling"], _D - eps)
    ) / (2.0 * eps)

    assert grads.initial_phases == pytest.approx(fd_phases, abs=1e-7)
    assert grads.omega == pytest.approx(fd_omega, abs=1e-7)
    assert grads.coupling == pytest.approx(fd_coupling, abs=1e-7)
    assert grads.diffusion == pytest.approx(fd_diffusion, abs=1e-6)


def test_phase_sensitivity_shape() -> None:
    problem = _problem(n=5, seed=1)
    theta_n, sensitivity = noisy_phase_sensitivity(
        problem["phases"],
        problem["omega"],
        problem["coupling"],
        diffusion=_D,
        dt=_DT,
        n_steps=_N_STEPS,
        seed=_SEED,
    )
    n = 5
    assert theta_n.shape == (n,)
    assert sensitivity.shape == (n, 2 * n + n * n + 1)


def test_gradients_are_deterministic() -> None:
    problem = _problem(seed=7)
    kwargs: dict[str, Any] = dict(
        diffusion=_D,
        dt=_DT,
        n_steps=_N_STEPS,
        seed=_SEED,
        objective=lambda t: float(order_parameter(t)),
        objective_grad=lambda t: order_parameter_gradient(t),
    )
    _, a = noisy_terminal_value_and_grad(
        problem["phases"], problem["omega"], problem["coupling"], **kwargs
    )
    _, b = noisy_terminal_value_and_grad(
        problem["phases"], problem["omega"], problem["coupling"], **kwargs
    )
    assert np.array_equal(a.coupling, b.coupling)
    assert a.diffusion == b.diffusion


def test_returns_noisy_gradients_type() -> None:
    problem = _problem()
    _, grads = noisy_terminal_value_and_grad(
        problem["phases"],
        problem["omega"],
        problem["coupling"],
        diffusion=_D,
        dt=_DT,
        n_steps=_N_STEPS,
        seed=_SEED,
        objective=lambda t: 0.0,
        objective_grad=lambda t: np.zeros_like(t),
    )
    assert isinstance(grads, NoisyGradients)


def test_objective_grad_wrong_shape_raises() -> None:
    problem = _problem()
    with pytest.raises(ValueError, match="objective_grad must return"):
        noisy_terminal_value_and_grad(
            problem["phases"],
            problem["omega"],
            problem["coupling"],
            diffusion=_D,
            dt=_DT,
            n_steps=_N_STEPS,
            seed=_SEED,
            objective=lambda t: 0.0,
            objective_grad=lambda t: np.zeros(t.size + 1),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"omega": np.zeros((2, 2))}, "omega must be a non-empty one-dimensional array"),
        ({"phases": np.zeros(3)}, "phases must have shape"),
        ({"coupling": np.zeros((3, 3))}, "coupling must have shape"),
        ({"diffusion": 0.0}, "diffusion must be positive"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"n_steps": 0}, "n_steps must be positive"),
    ],
)
def test_validation_errors(mutation: dict[str, Any], message: str) -> None:
    problem = _problem()
    call: dict[str, Any] = {
        "phases": problem["phases"],
        "omega": problem["omega"],
        "coupling": problem["coupling"],
        "diffusion": _D,
        "dt": _DT,
        "n_steps": _N_STEPS,
        "seed": _SEED,
    }
    call.update(mutation)
    with pytest.raises(ValueError, match=message):
        noisy_phase_sensitivity(
            call["phases"],
            call["omega"],
            call["coupling"],
            diffusion=call["diffusion"],
            dt=call["dt"],
            n_steps=call["n_steps"],
            seed=call["seed"],
        )
