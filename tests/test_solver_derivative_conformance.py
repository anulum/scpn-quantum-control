# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — discrete versus continuous solver derivative conformance
"""Discrete solver derivatives stay distinct from continuous sensitivities."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable_implicit_sensitivity import (
    RK4_ORDER,
    SOLVER_DERIVATIVE_SUPPORT,
    SolverDerivativeRefusal,
    refine_solver_sensitivity,
    rk4_forward_sensitivity,
)

Vector = NDArray[np.float64]


def _linear() -> tuple[Any, Any, Any]:
    """Return ``y' = a y`` with ``theta = (a,)`` and its exact Jacobians."""
    return (
        lambda t, y, th: th[0] * y,
        lambda t, y, th: np.array([[th[0]]]),
        lambda t, y, th: y.reshape(1, 1),
    )


def _rk4_closed_form(a: float, y0: float, final_time: float, steps: int) -> tuple[float, float]:
    """Return ``y_N`` and ``dy_N/da`` of RK4 on ``y' = a y`` from its amplification factor."""
    h = final_time / steps
    z = a * h
    amplification = 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    derivative = h * (1 + z + z**2 / 2 + z**3 / 6)
    return y0 * amplification**steps, y0 * steps * amplification ** (steps - 1) * derivative


def test_solver_derivative_conformance_01() -> None:
    """The discrete derivative is exact for its mesh; the continuous one needs refinement."""
    a, y0, final_time = -0.7, 1.3, 2.0
    rhs, jy, jp = _linear()
    exact = final_time * y0 * math.exp(a * final_time)
    errors = []
    for steps in (8, 16, 32, 64):
        discrete = rk4_forward_sensitivity(rhs, jy, jp, [y0], [a], final_time, steps)
        state, derivative = _rk4_closed_form(a, y0, final_time, steps)
        assert discrete.kind == "discrete"
        assert discrete.final_state[0] == pytest.approx(state, rel=1e-13)
        assert discrete.sensitivity[0, 0] == pytest.approx(derivative, rel=1e-13)
        errors.append(abs(discrete.sensitivity[0, 0] - exact))
    ratios = [coarse / fine for coarse, fine in zip(errors, errors[1:], strict=False)]
    assert all(14.0 < ratio < 18.0 for ratio in ratios)
    assert errors[0] > 1e-6
    continuous = refine_solver_sensitivity(rhs, jy, jp, [y0], [a], final_time, base_steps=16)
    assert continuous.kind == "continuous_refined"
    assert continuous.meshes == (16, 32, 64)
    assert abs(continuous.observed_order - RK4_ORDER) < 0.1
    assert abs(continuous.sensitivity[0, 0] - exact) <= continuous.error_estimate
    assert continuous.error_estimate <= continuous.tolerance


def test_two_equal_discrete_derivatives_do_not_make_a_continuous_claim() -> None:
    """Agreement of two discrete implementations is not convergence to the continuous value."""
    a, y0, final_time, steps = 1.1, 0.4, 1.5, 4
    rhs, jy, jp = _linear()
    stage_wise = rk4_forward_sensitivity(rhs, jy, jp, [y0], [a], final_time, steps)
    _, closed_form = _rk4_closed_form(a, y0, final_time, steps)
    assert stage_wise.sensitivity[0, 0] == pytest.approx(closed_form, rel=1e-13)
    exact = final_time * y0 * math.exp(a * final_time)
    assert abs(closed_form - exact) > 1e-3
    with pytest.raises(SolverDerivativeRefusal) as refused:
        refine_solver_sensitivity(rhs, jy, jp, [y0], [a], final_time, base_steps=steps)
    assert refused.value.reason == "not_refined"


def test_solver_derivative_conformance_02() -> None:
    """A failed solve never returns a gradient, and refinement inherits the refusal."""
    blow_up = (
        lambda t, y, th: th[0] * y**2,
        lambda t, y, th: np.array([[2 * th[0] * y[0]]]),
        lambda t, y, th: (y**2).reshape(1, 1),
    )
    for function in (rk4_forward_sensitivity,):
        with pytest.raises(SolverDerivativeRefusal, match="diverged") as refused:
            function(*blow_up, [1.0], [1.0], 2.0, 200)
        assert refused.value.reason == "failed_solve"
    with pytest.raises(SolverDerivativeRefusal) as refined:
        refine_solver_sensitivity(*blow_up, [1.0], [1.0], 2.0, base_steps=100)
    assert refined.value.reason == "failed_solve"
    poisoned = (
        lambda t, y, th: np.array([np.nan]) if t > 0.5 else -y,
        lambda t, y, th: np.array([[-1.0]]),
        lambda t, y, th: np.zeros((1, 1)),
    )
    with pytest.raises(SolverDerivativeRefusal, match="non-finite") as nan_solve:
        rk4_forward_sensitivity(*poisoned, [1.0], [0.0], 1.0, 10)
    assert nan_solve.value.reason == "failed_solve"


def test_a_time_kink_off_the_mesh_is_not_refined() -> None:
    """A right-hand side with a kink between mesh points loses fourth order and is refused."""
    kinked = (
        lambda t, y, th: np.array([th[0] * abs(t - 0.3)]),
        lambda t, y, th: np.zeros((1, 1)),
        lambda t, y, th: np.array([[abs(t - 0.3)]]),
    )
    with pytest.raises(SolverDerivativeRefusal, match="no continuous claim") as refused:
        refine_solver_sensitivity(*kinked, [0.0], [1.0], 1.0, base_steps=16, tolerance=1e-12)
    assert refused.value.reason == "not_refined"
    with pytest.raises(SolverDerivativeRefusal) as tight:
        refine_solver_sensitivity(*_linear(), [1.0], [0.5], 1.0, base_steps=4, tolerance=1e-30)
    assert tight.value.reason == "not_refined"


def test_solver_derivative_conformance_03() -> None:
    """A crossed event is a typed refusal; the same solve before the event is supported."""
    falling = (
        lambda t, y, th: np.array([-th[0]]),
        lambda t, y, th: np.zeros((1, 1)),
        lambda t, y, th: np.array([[-1.0]]),
    )
    ground = lambda t, y, th: float(y[0])  # noqa: E731
    with pytest.raises(SolverDerivativeRefusal, match="event-time derivative") as crossed:
        rk4_forward_sensitivity(*falling, [1.0], [1.0], 2.0, 16, event=ground)
    assert crossed.value.reason == "event_crossing"
    with pytest.raises(SolverDerivativeRefusal) as refined:
        refine_solver_sensitivity(*falling, [1.0], [1.0], 2.0, event=ground)
    assert refined.value.reason == "event_crossing"
    with pytest.raises(SolverDerivativeRefusal, match="reached or changed sign"):
        rk4_forward_sensitivity(*falling, [1.0], [1.0], 1.0, 8, event=ground)
    with pytest.raises(SolverDerivativeRefusal, match="starts on the event surface"):
        rk4_forward_sensitivity(*falling, [0.0], [1.0], 1.0, 8, event=ground)
    before = rk4_forward_sensitivity(*falling, [1.0], [1.0], 0.5, 16, event=ground)
    assert before.sensitivity[0, 0] == pytest.approx(-0.5, rel=1e-14)
    assert SOLVER_DERIVATIVE_SUPPORT["event_time"].startswith("refused")
    assert SOLVER_DERIVATIVE_SUPPORT["adaptive_accept_reject"].startswith("excluded")
    assert SOLVER_DERIVATIVE_SUPPORT["delayed_history"].startswith("excluded")
    assert SOLVER_DERIVATIVE_SUPPORT["failed_solve"].startswith("refused")


def test_initial_sensitivity_propagates_through_the_flow() -> None:
    """A parameter-dependent initial state adds its own flowed sensitivity."""
    a, final_time = -0.4, 1.0
    rhs, jy, jp = _linear()
    seeded = refine_solver_sensitivity(
        rhs, jy, jp, [2.0], [a], final_time, initial_sensitivity=[[1.0]]
    )
    expected = math.exp(a * final_time) + final_time * 2.0 * math.exp(a * final_time)
    assert seeded.sensitivity[0, 0] == pytest.approx(expected, rel=1e-10)


def test_exactly_converged_meshes_report_full_order() -> None:
    """A solve RK4 integrates exactly is accepted with the nominal order."""
    constant = (
        lambda t, y, th: np.array([th[0]]),
        lambda t, y, th: np.zeros((1, 1)),
        lambda t, y, th: np.ones((1, 1)),
    )
    result = refine_solver_sensitivity(*constant, [0.0], [2.0], 3.0, base_steps=2)
    assert result.observed_order == RK4_ORDER
    assert result.sensitivity[0, 0] == pytest.approx(3.0, rel=1e-14)


def test_stalled_coarse_meshes_are_not_called_converged() -> None:
    """Identical coarse meshes followed by a change do not pass as fourth order."""

    def pulse(t: float) -> float:
        return float(abs(t - 0.125) < 1e-12)

    only_finest = (
        lambda t, y, th: np.array([th[0] * pulse(t)]),
        lambda t, y, th: np.zeros((1, 1)),
        lambda t, y, th: np.array([[pulse(t)]]),
    )
    coarse = [
        rk4_forward_sensitivity(*only_finest, [0.0], [1.0], 1.0, steps).sensitivity[0, 0]
        for steps in (1, 2, 4)
    ]
    assert coarse[0] == coarse[1] == 0.0 and coarse[2] != 0.0
    with pytest.raises(SolverDerivativeRefusal) as refused:
        refine_solver_sensitivity(*only_finest, [0.0], [1.0], 1.0, base_steps=1)
    assert refused.value.reason == "not_refined"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"initial_state": []}, "initial_state must be"),
        ({"initial_state": [np.inf]}, "initial_state must be"),
        ({"theta": [[1.0]]}, "theta must be"),
        ({"final_time": 0.0}, "final_time must be"),
        ({"final_time": True}, "final_time must be"),
        ({"steps": 0}, "steps must be"),
        ({"steps": True}, "steps must be"),
        ({"initial_sensitivity": [1.0, 2.0]}, "initial_sensitivity must be"),
        ({"divergence_bound": -1.0}, "divergence_bound must be"),
    ],
)
def test_malformed_solves_are_refused(change: dict[str, Any], message: str) -> None:
    """Malformed solve inputs are refused before any step is taken."""
    arguments: dict[str, Any] = {
        "initial_state": [1.0],
        "theta": [0.5],
        "final_time": 1.0,
        "steps": 4,
    }
    arguments.update(change)
    rhs, jy, jp = _linear()
    with pytest.raises(ValueError, match=message):
        rk4_forward_sensitivity(rhs, jy, jp, **arguments)


def test_wrong_returned_shapes_are_refused() -> None:
    """A right-hand side or Jacobian with the wrong shape is refused."""
    rhs, jy, jp = _linear()
    with pytest.raises(ValueError, match="rhs must return shape"):
        rk4_forward_sensitivity(lambda t, y, th: np.ones(2), jy, jp, [1.0], [0.5], 1.0, 2)
    with pytest.raises(ValueError, match="parameter_jacobian must return shape"):
        rk4_forward_sensitivity(rhs, jy, lambda t, y, th: np.ones(1), [1.0], [0.5], 1.0, 2)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"base_steps": 0}, "base_steps must be"),
        ({"base_steps": True}, "base_steps must be"),
        ({"refinements": 2}, "refinements must be"),
        ({"refinements": True}, "refinements must be"),
        ({"tolerance": 0.0}, "tolerance must be"),
        ({"order_tolerance": float("nan")}, "order_tolerance must be"),
    ],
)
def test_malformed_refinements_are_refused(change: dict[str, Any], message: str) -> None:
    """Refinement settings that cannot measure an order are refused."""
    rhs, jy, jp = _linear()
    with pytest.raises(ValueError, match=message):
        refine_solver_sensitivity(rhs, jy, jp, [1.0], [0.5], 1.0, **change)


def test_inputs_are_not_modified() -> None:
    """A solve leaves its caller's arrays unchanged."""
    state: Vector = np.array([1.0])
    theta: Vector = np.array([0.5])
    seed = np.array([[0.25]])
    rhs, jy, jp = _linear()
    rk4_forward_sensitivity(rhs, jy, jp, state, theta, 1.0, 4, initial_sensitivity=seed)
    assert state.tolist() == [1.0] and theta.tolist() == [0.5] and seed.tolist() == [[0.25]]
