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
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable_implicit_sensitivity import (
    DELAYED_METHOD_OF_STEPS_ORDERS,
    RK4_ORDER,
    SOLVER_DERIVATIVE_SUPPORT,
    SolverDerivativeRefusal,
    qualify_adaptive_dopri_gradient,
    qualify_delayed_sensitivity,
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
    assert continuous.meshes == (16, 32, 64, 128)
    assert len(continuous.observed_orders) == 2
    assert all(abs(order - RK4_ORDER) < 0.1 for order in continuous.observed_orders)
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
    assert SOLVER_DERIVATIVE_SUPPORT["adaptive_accept_reject"].startswith("qualified only")
    assert SOLVER_DERIVATIVE_SUPPORT["delayed_history"].startswith("qualified only")
    assert SOLVER_DERIVATIVE_SUPPORT["failed_solve"].startswith("refused")
    assert SOLVER_DERIVATIVE_SUPPORT["open_system_objectives"].startswith("discrete only")


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


def test_non_finite_sensitivity_is_a_failed_solve() -> None:
    """A state that stays finite while its sensitivity overflows yields no derivative."""
    stiff = (
        lambda t, y, th: 100.0 * y + th[0],
        lambda t, y, th: np.array([[100.0]]),
        lambda t, y, th: np.array([[1.0]]),
    )
    with pytest.raises(SolverDerivativeRefusal, match="sensitivity became non-finite") as refused:
        rk4_forward_sensitivity(*stiff, [0.0], [0.0], 8.0, 1000)
    assert refused.value.reason == "failed_solve"


def _rotation() -> tuple[Any, Any, Any]:
    """Return ``y' = theta * (y1, -y0)``, the unit circle at angular speed ``theta``."""
    return (
        lambda t, y, th: th[0] * np.array([y[1], -y[0]]),
        lambda t, y, th: th[0] * np.array([[0.0, 1.0], [-1.0, 0.0]]),
        lambda t, y, th: np.array([[y[1]], [-y[0]]]),
    )


def test_event_near_zero_between_mesh_points_refuses_a_continuous_claim() -> None:
    """``cos t + 0.9999999`` nearly touches zero at ``t = pi`` without changing sign at nodes."""
    grazing = refine_solver_sensitivity
    with pytest.raises(SolverDerivativeRefusal) as refused:
        grazing(
            *_rotation(),
            [1.0, 0.0],
            [1.0],
            4.0,
            base_steps=128,
            event=lambda t, y, th: float(y[0]) + 0.9999999,
        )
    assert refused.value.reason == "event_crossing"
    distant = refine_solver_sensitivity(
        *_rotation(),
        [1.0, 0.0],
        [1.0],
        4.0,
        base_steps=128,
        event=lambda t, y, th: float(y[0]) + 2.0,
    )
    exact = np.array([[-4.0 * math.sin(4.0)], [-4.0 * math.cos(4.0)]])
    assert np.all(np.abs(distant.sensitivity - exact) <= 10.0 * distant.error_estimate + 1e-12)


def test_non_finite_event_value_is_a_failed_solve() -> None:
    """``nan * side <= 0`` is false, so a NaN event must be refused explicitly."""
    with pytest.raises(SolverDerivativeRefusal) as refused:
        rk4_forward_sensitivity(*_linear(), [1.0], [0.5], 1.0, 4, event=lambda t, y, th: math.nan)
    assert refused.value.reason == "failed_solve"


@pytest.mark.parametrize("amplitude", [1e-4, 1e-16], ids=["unit-scale", "tiny-scale"])
def test_pre_asymptotic_kink_is_not_refined(amplitude: float) -> None:
    """A kink between mesh points shows inconsistent orders; one ratio near four is not enough.

    Parameters
    ----------
    amplitude
        Size of the kinked forcing; the tiny case once fabricated order four.

    """
    kinked = (
        lambda t, y, th: np.array([th[0] * amplitude * abs(t - 0.154)]),
        lambda t, y, th: np.zeros((1, 1)),
        lambda t, y, th: np.array([[amplitude * abs(t - 0.154)]]),
    )
    with pytest.raises(SolverDerivativeRefusal) as refused:
        refine_solver_sensitivity(*kinked, [0.0], [1.0], 1.0, tolerance=1.0)
    assert refused.value.reason == "not_refined"


def test_divergence_bound_and_numpy_steps_are_forwarded() -> None:
    """Large but bounded states refine with a matching bound; ``np.int64`` steps are counts."""
    rhs, jy, jp = _linear()
    result = refine_solver_sensitivity(
        rhs, jy, jp, [1e13], [-0.5], 1.0, base_steps=16, tolerance=1e6, divergence_bound=1e15
    )
    assert result.sensitivity[0, 0] == pytest.approx(1e13 * math.exp(-0.5), rel=1e-8)
    counted = rk4_forward_sensitivity(rhs, jy, jp, [1.0], [0.5], 1.0, cast(int, np.int64(8)))
    assert counted.steps == 8


def _van_der_pol() -> tuple[Any, Any, Any]:
    """Return the Van der Pol field with ``theta = (mu, omega)`` and its Jacobians."""
    return (
        lambda t, y, th: np.array([y[1], th[0] * (1 - y[0] ** 2) * y[1] - th[1] ** 2 * y[0]]),
        lambda t, y, th: np.array(
            [[0.0, 1.0], [-2 * th[0] * y[0] * y[1] - th[1] ** 2, th[0] * (1 - y[0] ** 2)]]
        ),
        lambda t, y, th: np.array([[0.0, 0.0], [(1 - y[0] ** 2) * y[1], -2 * th[1] * y[0]]]),
    )


def test_nonlinear_two_parameter_sensitivity_matches_a_variational_oracle() -> None:
    """Van der Pol with two states and two parameters, against DOP853 at ``rtol=1e-12``."""
    from scipy.integrate import solve_ivp

    theta, start, final_time = np.array([0.8, 1.3]), np.array([0.4, -0.2]), 3.0
    rhs, jy, jp = _van_der_pol()

    def augmented(t: float, packed: Vector) -> Vector:
        state, sens = packed[:2], packed[2:].reshape(2, 2)
        return np.concatenate(
            (rhs(t, state, theta), (jy(t, state, theta) @ sens + jp(t, state, theta)).ravel())
        )

    oracle = (
        solve_ivp(
            augmented,
            (0.0, final_time),
            np.concatenate((start, np.zeros(4))),
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        )
        .y[2:, -1]
        .reshape(2, 2)
    )
    refined = refine_solver_sensitivity(rhs, jy, jp, start, theta, final_time, base_steps=64)
    assert np.all(np.abs(refined.sensitivity - oracle) <= 10.0 * refined.error_estimate + 1e-10)
    discrete = rk4_forward_sensitivity(rhs, jy, jp, start, theta, final_time, 64)
    step = 1e-6
    for column in range(2):
        shift = np.zeros(2)
        shift[column] = step
        plus = rk4_forward_sensitivity(
            rhs, jy, jp, start, theta + shift, final_time, 64
        ).final_state
        minus = rk4_forward_sensitivity(
            rhs, jy, jp, start, theta - shift, final_time, 64
        ).final_state
        assert np.allclose(discrete.sensitivity[:, column], (plus - minus) / (2 * step), atol=1e-8)


def _kuramoto_terminal(
    theta0: Vector, omega: Vector, coupling: Vector, final_time: float
) -> Vector:
    """Integrate the Kuramoto flow with DOP853 at ``rtol=1e-12``."""
    from scipy.integrate import solve_ivp

    def field(t: float, phases: Vector) -> Vector:
        drift: Vector = omega + np.sum(
            coupling * np.sin(phases[None, :] - phases[:, None]), axis=1
        )
        return drift

    result: Vector = solve_ivp(
        field, (0.0, final_time), theta0, method="DOP853", rtol=1e-12, atol=1e-14
    ).y[:, -1]
    return result


def test_adaptive_dopri_gradient_is_qualified_by_tolerance_refinement() -> None:
    """The fixed-grid adjoint is issued only from refinement evidence, close to the flow."""
    theta0 = np.array([0.1, 0.5, -0.3])
    omega = np.array([0.4, -0.2, 0.9])
    coupling = 0.5 * np.array([[0.0, 1.0, 0.3], [0.7, 0.0, 0.2], [0.1, 0.9, 0.0]])
    cotangent = np.ones(3)
    qualified = qualify_adaptive_dopri_gradient(
        theta0, omega, coupling, cotangent, final_time=2.0, tolerance=1e-6
    )
    assert qualified.kind == "continuous_refined" and len(qualified.differences) == 2
    step = 1e-6
    for index in range(3):
        shift = np.zeros(3)
        shift[index] = step
        flow = (
            (
                _kuramoto_terminal(theta0, omega + shift, coupling, 2.0)
                - _kuramoto_terminal(theta0, omega - shift, coupling, 2.0)
            )
            @ cotangent
            / (2 * step)
        )
        assert abs(qualified.omega[index] - flow) <= 10.0 * qualified.error_estimate + 1e-7
    with pytest.raises(SolverDerivativeRefusal) as refused:
        qualify_adaptive_dopri_gradient(
            theta0, omega, coupling, cotangent, final_time=2.0, tolerance=1e-15
        )
    assert refused.value.reason == "not_refined"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"rtols": (1e-6, 1e-8)}, "at least three"),
        ({"rtols": (1e-6, 1e-6, 1e-8)}, "strictly decreasing"),
        ({"atol_scale": 0.0}, "atol_scale"),
        ({"final_time": -1.0}, "final_time"),
    ],
    ids=["two-tolerances", "not-decreasing", "zero-atol", "negative-time"],
)
def test_adaptive_qualification_refuses_malformed_settings(
    changes: dict[str, Any], message: str
) -> None:
    """Malformed tolerance ladders and times raise before any solve.

    Parameters
    ----------
    changes
        Keyword arguments replacing the valid defaults.
    message
        Expected refusal text.

    """
    options: dict[str, Any] = {"final_time": 1.0}
    options.update(changes)
    with pytest.raises(ValueError, match=message):
        qualify_adaptive_dopri_gradient(
            np.zeros(2), np.ones(2), np.zeros((2, 2)), np.ones(2), **options
        )


def test_delayed_sensitivity_is_qualified_against_a_closed_form() -> None:
    """With ``K = 0`` and a free-running history, ``dtheta_j/dK_jk`` integrates a sinusoid.

    Then ``theta_k(t - tau) - theta_j(t) = a + b t`` with
    ``a = theta_k(0) - theta_j(0) - omega_k tau`` and ``b = omega_k - omega_j``,
    so ``dtheta_j(T)/dK_jk = (cos(a) - cos(a + b T)) / b``, which is
    ``T sin(a)`` for the self-coupling ``b = 0``, and ``dtheta(T)/domega = T * I``.
    """
    start = np.array([0.2, -0.4])
    omega = np.array([0.7, 1.5])
    delay, final_time = 0.5, 2.0

    def history(times: Vector) -> Vector:
        return start[None, :] + omega[None, :] * times[:, None]

    result = qualify_delayed_sensitivity(
        history, omega, np.zeros((2, 2)), delay=delay, final_time=final_time, tolerance=1e-6
    )
    assert result.observed_orders
    assert (
        min(abs(allowed - result.observed_order) for allowed in DELAYED_METHOD_OF_STEPS_ORDERS)
        < 0.5
    )
    expected = np.zeros((2, 6))
    expected[:, :2] = final_time * np.eye(2)
    for j in range(2):
        for k in range(2):
            a = start[k] - start[j] - omega[k] * delay
            b = omega[k] - omega[j]
            # The self-coupling term has b = 0: a constant integrand sin(a).
            expected[j, 2 + 2 * j + k] = (
                final_time * math.sin(a)
                if b == 0.0
                else (math.cos(a) - math.cos(a + b * final_time)) / b
            )
    assert np.all(np.abs(result.sensitivity - expected) <= 10.0 * result.error_estimate + 1e-9)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"final_time": 2.1}, "integer multiple"),
        ({"delay": 0.0}, "delay"),
        ({"history": lambda times: np.zeros((len(times), 3))}, "history must return"),
        ({"refinements": 3}, "refinements"),
    ],
    ids=["off-mesh-time", "zero-delay", "wrong-history-shape", "three-meshes"],
)
def test_delayed_qualification_refuses_malformed_settings(
    changes: dict[str, Any], message: str
) -> None:
    """Off-mesh times, non-positive delays, wrong histories and short ladders raise.

    Parameters
    ----------
    changes
        Keyword arguments replacing the valid defaults.
    message
        Expected refusal text.

    """
    options: dict[str, Any] = {
        "history": lambda times: np.zeros((len(times), 2)),
        "delay": 0.5,
        "final_time": 2.0,
    }
    options.update(changes)
    history = options.pop("history")
    with pytest.raises(ValueError, match=message):
        qualify_delayed_sensitivity(history, np.ones(2), np.zeros((2, 2)), **options)


def test_delayed_qualification_refuses_an_unmet_tolerance() -> None:
    """A tolerance below the refined error estimate is not a continuous claim."""
    with pytest.raises(SolverDerivativeRefusal) as refused:
        qualify_delayed_sensitivity(
            lambda times: np.stack((np.sin(times), np.cos(times)), axis=1),
            np.array([0.3, -0.5]),
            np.array([[0.0, 0.8], [0.6, 0.0]]),
            delay=0.5,
            final_time=2.0,
            tolerance=1e-14,
        )
    assert refused.value.reason == "not_refined"


def test_qualifiers_refuse_non_finite_gradients() -> None:
    """An overflowing adjoint or delayed sensitivity is a failed solve, never a claim."""
    with np.errstate(all="ignore"), pytest.raises(SolverDerivativeRefusal) as adaptive:
        qualify_adaptive_dopri_gradient(
            np.zeros(2), np.ones(2), 0.1 * np.ones((2, 2)), np.full(2, 1e308), final_time=10.0
        )
    assert adaptive.value.reason == "failed_solve"
    with np.errstate(all="ignore"), pytest.raises(SolverDerivativeRefusal) as delayed:
        qualify_delayed_sensitivity(
            lambda times: np.stack((np.sin(times), np.cos(times)), axis=1),
            np.ones(2),
            1e200 * np.array([[0.0, 1.0], [1.0, 0.0]]),
            delay=0.5,
            final_time=1.0,
        )
    assert delayed.value.reason == "failed_solve"


def test_event_narrower_than_a_step_is_outside_the_continuous_guarantee() -> None:
    """A time pulse in ``g`` between mesh points crosses zero unseen: the documented limit.

    ``g(t) = 1 - 2 exp(-((t - 0.5003) / 1e-4)**2)`` dips to -1 for about 2e-4 of
    time, far narrower than the finest step 1/128. No mesh point sees it, so a
    continuous claim is issued; ``SOLVER_DERIVATIVE_SUPPORT`` states that the
    claim assumes the finest mesh resolves ``g``.
    """

    def pulse(t: float, y: Vector, th: Vector) -> float:
        return 1.0 - 2.0 * math.exp(-(((t - 0.5003) / 1e-4) ** 2))

    result = refine_solver_sensitivity(*_linear(), [1.0], [-0.5], 1.0, base_steps=16, event=pulse)
    assert result.kind == "continuous_refined"
    assert pulse(0.5003, np.zeros(1), np.zeros(1)) < 0.0
    assert "resolved by the" in SOLVER_DERIVATIVE_SUPPORT["event_time"]
