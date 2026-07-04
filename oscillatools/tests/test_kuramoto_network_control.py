# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for indirect-adjoint optimal control of the network
"""Module-specific tests for :mod:`kuramoto_network_control`.

The contracts: with zero control the integrator reproduces the shipped networked Kuramoto RK4
trajectory exactly; the discrete-adjoint gradient of the running desynchronisation objective matches
finite differences to machine precision; gradient descent genuinely desynchronises the network
(drives the mean order parameter down) with a monotone cost; and the trajectory container and input
contract behave.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel.diff_kuramoto_rk4 import kuramoto_rk4_trajectory
from oscillatools.accel.kuramoto_network_control import (
    ControlledNetworkTrajectory,
    integrate_controlled_network,
    network_control_value_and_grad,
    optimise_network_control,
)
from oscillatools.accel.order_parameter_observables import order_parameter

_N = 8
_DT = 0.05


def _problem(seed: int = 1) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=(_N, _N))
    coupling = 0.4 * (raw + raw.T)
    np.fill_diagonal(coupling, 0.0)
    return {
        "phases": rng.uniform(0.0, 2.0 * np.pi, size=_N),
        "omega": rng.standard_normal(_N) * 0.2,
        "coupling": coupling,
    }


def test_zero_control_reproduces_networked_rk4() -> None:
    problem = _problem()
    n_steps = 20
    trajectory = integrate_controlled_network(
        problem["phases"], np.zeros((n_steps, _N)), problem["omega"], problem["coupling"], _DT
    )
    reference = kuramoto_rk4_trajectory(
        problem["phases"], problem["omega"], problem["coupling"], _DT, n_steps
    )
    assert trajectory.phases == pytest.approx(reference, abs=1e-12)


def test_adjoint_gradient_matches_finite_differences() -> None:
    problem = _problem()
    rng = np.random.default_rng(3)
    n_steps = 15
    control = rng.standard_normal((n_steps, _N)) * 0.1
    weight = 0.05
    grads = network_control_value_and_grad(
        problem["phases"],
        control,
        problem["omega"],
        problem["coupling"],
        _DT,
        control_weight=weight,
    )

    eps = 1e-6
    finite = np.zeros((n_steps, _N), dtype=np.float64)

    def cost(series: NDArray[np.float64]) -> float:
        return network_control_value_and_grad(
            problem["phases"],
            series,
            problem["omega"],
            problem["coupling"],
            _DT,
            control_weight=weight,
        ).cost

    for step in range(n_steps):
        for node in range(_N):
            plus = control.copy()
            minus = control.copy()
            plus[step, node] += eps
            minus[step, node] -= eps
            finite[step, node] = (cost(plus) - cost(minus)) / (2.0 * eps)
    assert grads.control_gradient == pytest.approx(finite, abs=1e-7)


def test_gradient_descent_desynchronises_with_monotone_cost() -> None:
    problem = _problem()
    n_steps = 15
    control, history = optimise_network_control(
        problem["phases"],
        problem["omega"],
        problem["coupling"],
        _DT,
        n_steps,
        control_weight=1e-3,
        learning_rate=2.0,
        n_iterations=150,
    )
    uncontrolled = integrate_controlled_network(
        problem["phases"], np.zeros((n_steps, _N)), problem["omega"], problem["coupling"], _DT
    )
    controlled = integrate_controlled_network(
        problem["phases"], control, problem["omega"], problem["coupling"], _DT
    )
    mean_uncontrolled = float(
        np.mean([order_parameter(uncontrolled.phases[k]) for k in range(1, n_steps + 1)])
    )
    mean_controlled = float(
        np.mean([order_parameter(controlled.phases[k]) for k in range(1, n_steps + 1)])
    )
    assert mean_controlled < mean_uncontrolled
    assert np.all(np.diff(history) <= 1e-9)
    assert history[-1] < history[0]


def test_trajectory_container() -> None:
    problem = _problem()
    n_steps = 10
    trajectory = integrate_controlled_network(
        problem["phases"], np.full((n_steps, _N), 0.02), problem["omega"], problem["coupling"], _DT
    )
    assert isinstance(trajectory, ControlledNetworkTrajectory)
    assert trajectory.phases.shape == (n_steps + 1, _N)
    assert trajectory.times.shape == (n_steps + 1,)
    assert trajectory.times[-1] == pytest.approx(n_steps * _DT)
    assert trajectory.terminal_phases == pytest.approx(trajectory.phases[-1])


@pytest.mark.parametrize(
    ("call", "kwargs", "message"),
    [
        ("integrate", {"phases": np.zeros((2, 2))}, "phases must be a non-empty"),
        ("integrate", {"control": np.zeros((3, _N + 1))}, "control must have shape"),
        ("integrate", {"omega": np.zeros(_N + 1)}, "omega must have shape"),
        ("integrate", {"coupling": np.zeros((_N, _N + 1))}, "coupling must have shape"),
        ("integrate", {"dt": 0.0}, "dt must be positive"),
        ("grad", {"control_weight": -1.0}, "control_weight must be non-negative"),
        ("optimise", {"n_steps": 0}, "n_steps must be positive"),
        ("optimise", {"learning_rate": 0.0}, "learning_rate must be positive"),
        ("optimise", {"n_iterations": 0}, "n_iterations must be positive"),
        ("optimise", {"initial_control": np.zeros((2, _N))}, "initial_control must have shape"),
    ],
)
def test_validation_errors(call: str, kwargs: dict[str, Any], message: str) -> None:
    problem = _problem()
    with pytest.raises(ValueError, match=message):
        if call == "integrate":
            args: dict[str, Any] = {
                "phases": problem["phases"],
                "control": np.zeros((4, _N)),
                "omega": problem["omega"],
                "coupling": problem["coupling"],
                "dt": _DT,
            }
            args.update(kwargs)
            integrate_controlled_network(
                args["phases"], args["control"], args["omega"], args["coupling"], args["dt"]
            )
        elif call == "grad":
            network_control_value_and_grad(
                problem["phases"],
                np.zeros((4, _N)),
                problem["omega"],
                problem["coupling"],
                _DT,
                control_weight=kwargs["control_weight"],
            )
        else:
            opt: dict[str, Any] = {"n_steps": 4, "learning_rate": 0.5, "n_iterations": 5}
            opt.update(kwargs)
            optimise_network_control(
                problem["phases"],
                problem["omega"],
                problem["coupling"],
                _DT,
                opt["n_steps"],
                control_weight=0.1,
                learning_rate=opt["learning_rate"],
                n_iterations=opt["n_iterations"],
                initial_control=opt.get("initial_control"),
            )
