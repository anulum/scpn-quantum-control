# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — public solver time, refinement and termination regressions
"""Qualify existing solver paths against independent equations and real SciPy."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools import (
    KuramotoParameters,
    KuramotoSystem,
    adaptive_state_sensitivity,
    integrate_adaptive_kuramoto,
    integrate_delayed_kuramoto,
    kuramoto_dopri_trajectory,
    solve_kuramoto_ivp,
)
from oscillatools.accel.kuramoto_adaptive import (
    adaptive_vector_field,
    hebbian_adaptive_jacobian,
    hebbian_plasticity_rate,
)
from oscillatools.accel.networked_kuramoto import networked_kuramoto_force


def _zero_force(current: NDArray[np.float64], delayed: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return zero coupling for the exactly uncoupled delayed equation."""
    return np.zeros_like(current)


@pytest.mark.parametrize("scheme", ["euler", "rk4"])
def test_kuramoto_solver_trajectories_01(scheme: str) -> None:
    """Every actual returned timestamp identifies its uncoupled unwrapped state."""
    initial = np.array([8.0, -9.0])
    omega = np.array([0.3, -0.7])
    system = KuramotoSystem.mean_field(initial, omega, 0.0, dt=0.03, scheme=scheme)
    system.reinit(time=2.5)
    times, phases = system.trajectory_with_times(17)
    np.testing.assert_allclose(phases, initial + (times[:, None] - 2.5) * omega, atol=2e-14)
    assert times[0] == 2.5 and times[-1] == system.current_time
    np.testing.assert_array_equal(phases[-1], system.current_state)
    system.reinit()
    np.testing.assert_array_equal(system.trajectory(17), phases)

    scipy = solve_kuramoto_ivp(system, (0.0, 0.51), t_eval=np.linspace(0.0, 0.51, 18))
    np.testing.assert_allclose(scipy.phases, phases[-1] + scipy.times[:, None] * omega, atol=2e-14)
    assert scipy.termination == "completed" and scipy.success
    dopri = kuramoto_dopri_trajectory(
        initial, omega, np.zeros((2, 2)), t_end=0.51, first_step=0.03
    )
    np.testing.assert_allclose(dopri.phases, initial + dopri.times[:, None] * omega, atol=2e-14)
    np.testing.assert_allclose(np.diff(dopri.times), dopri.steps, atol=2e-16)
    delayed = integrate_delayed_kuramoto(
        np.tile(initial, (5, 1)), omega, _zero_force, delay=0.12, dt=0.03, n_steps=17
    )
    np.testing.assert_allclose(
        delayed.phases, initial + delayed.times[:, None] * omega, atol=2e-14
    )
    np.testing.assert_array_equal(initial, [8.0, -9.0])
    np.testing.assert_array_equal(omega, [0.3, -0.7])


@pytest.mark.parametrize(("scheme", "min_ratio"), [("euler", 1.8), ("rk4", 14.0)])
def test_kuramoto_solver_trajectories_02(scheme: str, min_ratio: float) -> None:
    """Two-node errors reach their analytic first/fourth-order refinement regime."""
    k, delta0, omega = 0.7, 0.8, 0.2
    initial = np.array([-delta0 / 2, delta0 / 2])
    coupling = np.array([[0.0, k], [k, 0.0]])
    errors: list[float] = []
    for steps in (16, 32, 64):
        system = KuramotoSystem.networked(
            initial, np.full(2, omega), coupling, dt=1 / steps, scheme=scheme
        )
        times, phases = system.trajectory_with_times(steps)
        delta = 2 * np.arctan(np.tan(delta0 / 2) * np.exp(-2 * k * times))
        exact = omega * times[:, None] + np.column_stack((-delta / 2, delta / 2))
        errors.append(float(np.max(np.abs(phases - exact))))
        shifted = KuramotoSystem.networked(
            initial + 8, np.full(2, omega), coupling, dt=1 / steps, scheme=scheme
        )
        np.testing.assert_allclose(shifted.trajectory(steps) - 8, phases, atol=3e-14)
        reversed_system = KuramotoSystem.networked(
            initial[::-1], np.full(2, omega), coupling, dt=1 / steps, scheme=scheme
        )
        np.testing.assert_allclose(reversed_system.trajectory(steps)[:, ::-1], phases, atol=2e-15)
    assert errors[0] / errors[1] > min_ratio and errors[1] / errors[2] > min_ratio


def test_kuramoto_solver_trajectories_03() -> None:
    """Invalid history and a real accepted-step limit cannot appear converged."""
    history = np.zeros((3, 2))
    history[1, 0] = np.nan
    saved = history.copy()
    with pytest.raises(ValueError, match="finite"):
        integrate_delayed_kuramoto(history, np.ones(2), _zero_force, delay=0.2, dt=0.1, n_steps=3)
    np.testing.assert_array_equal(history, saved)
    with pytest.raises(ValueError, match="max_steps"):
        kuramoto_dopri_trajectory(
            np.zeros(2), np.ones(2), np.zeros((2, 2)), t_end=1.0, first_step=0.01, max_steps=1
        )


def test_delayed_history_refinement() -> None:
    """Linear lag interpolation sets the delay method's second-order ceiling."""
    errors: list[float] = []
    for steps in (16, 32, 64):
        dt = 1 / steps
        history = np.exp(np.linspace(-0.5, 0.0, steps // 2 + 1))[:, None]
        saved = history.copy()

        def exponential_force(
            current: NDArray[np.float64], lagged: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            return float(np.exp(0.5)) * lagged

        result = integrate_delayed_kuramoto(
            history, np.zeros(1), exponential_force, delay=0.5, dt=dt, n_steps=steps
        )
        errors.append(float(np.max(np.abs(result.phases[:, 0] - np.exp(result.times)))))
        np.testing.assert_array_equal(history, saved)
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2


def test_delayed_history_derivative_jump() -> None:
    """A constant history and linear DDE retain the propagated derivative jump."""

    def lagged_force(
        current: NDArray[np.float64], lagged: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return lagged.copy()

    result = integrate_delayed_kuramoto(
        np.ones((9, 1)), np.zeros(1), lagged_force, delay=0.5, dt=0.0625, n_steps=16
    )
    times = result.times
    exact = np.where(times <= 0.5, 1 + times, 1 + times + (times - 0.5) ** 2 / 2)
    np.testing.assert_allclose(result.phases[:, 0], exact, atol=2e-15)
    # theta'' jumps from zero to one at t=tau; finite samples do not establish global smoothness.
    assert (result.phases[8, 0] - result.phases[7, 0]) == pytest.approx(0.0625)
    assert (result.phases[9, 0] - result.phases[8, 0]) > 0.0625


def test_adaptive_plastic_refinement() -> None:
    """Joint phases, plastic weights and sensitivities match independent equations."""
    initial, omega, coupling = np.full(2, 8.0), np.full(2, 0.2), np.full((2, 2), 0.3)
    epsilon = 0.8

    def plasticity(
        phases: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return hebbian_plasticity_rate(phases, weights, plasticity_rate=epsilon)

    errors: list[float] = []
    for steps in (8, 16, 32):
        result = integrate_adaptive_kuramoto(
            initial,
            coupling,
            omega,
            networked_kuramoto_force,
            plasticity,
            dt=1 / steps,
            n_steps=steps,
        )
        np.testing.assert_allclose(
            result.phases, initial + result.times[:, None] * omega, atol=2e-14
        )
        exact = 1 + (coupling - 1) * np.exp(-epsilon * result.times[:, None, None])
        errors.append(float(np.max(np.abs(result.couplings - exact))))
        phases, weights, sensitivity = adaptive_state_sensitivity(
            initial, coupling, omega, plasticity_rate=epsilon, dt=1 / steps, n_steps=steps
        )
        np.testing.assert_allclose(phases, result.terminal_phases, atol=1e-14)
        np.testing.assert_allclose(weights, result.terminal_coupling, atol=1e-14)
        np.testing.assert_allclose(
            sensitivity[2:, -1], (1 - coupling).ravel() * np.exp(-epsilon), atol=2e-6
        )
        np.testing.assert_allclose(sensitivity[2:, 2:6], np.eye(4) * np.exp(-epsilon), atol=4e-7)
    assert errors[0] / errors[1] > 15 and errors[1] / errors[2] > 15
    np.testing.assert_array_equal(initial, [8.0, 8.0])
    np.testing.assert_array_equal(coupling, np.full((2, 2), 0.3))


class PhaseThreshold:
    """A real SciPy event with direction and terminal attributes."""

    terminal = True
    direction = 1.0

    def __call__(self, time: float, phases: NDArray[np.float64]) -> float:
        """Locate the upward crossing of theta=0.3."""
        return float(phases[0] - 0.3)


def test_scipy_events_and_failure() -> None:
    """Event termination, interpolation and actual singular failure stay distinct."""
    system = KuramotoSystem.mean_field(np.zeros(1), np.ones(1), 0.0, dt=0.1)
    event = solve_kuramoto_ivp(
        system, (0.0, 1.0), events=PhaseThreshold(), t_eval=[0, 0.1, 0.2, 0.4]
    )
    assert event.success and event.status == 1 and event.termination == "event"
    np.testing.assert_allclose(event.event_times[0], [0.3], atol=2e-15)
    np.testing.assert_allclose(event.event_phases[0], [[0.3]], atol=2e-15)
    np.testing.assert_array_equal(event.times, [0.0, 0.1, 0.2])
    early = solve_kuramoto_ivp(system, (0.0, 1.0), events=[PhaseThreshold()], t_eval=[0.9])
    assert early.phases.shape == (0, 1) and early.times.size == 0
    np.testing.assert_allclose(early.event_phases[0], [[0.3]], atol=2e-15)
    with pytest.raises(ValueError, match="samples"):
        _ = early.terminal_phases
    np.testing.assert_array_equal(system.current_state, [0.0])
    assert system.current_time == 0.0

    def singular(
        phases: NDArray[np.float64], parameters: KuramotoParameters, time: float
    ) -> NDArray[np.float64]:
        return 1 + phases * phases

    failure_system = KuramotoSystem(
        singular, np.zeros(1), KuramotoParameters(np.zeros(1), 0.0), dt=0.01
    )
    failed = solve_kuramoto_ivp(failure_system, (0, 2), rtol=1e-9, atol=1e-12)
    assert not failed.success and failed.status == -1 and failed.termination == "failed"
    assert failed.times[-1] < 2 and np.all(np.isfinite(failed.phases))
    assert failed.message and failed.function_evaluations > 0
    np.testing.assert_array_equal(failure_system.current_state, [0.0])


@pytest.mark.parametrize("method", ["Radau", "BDF"])
def test_stiff_solver_reference(method: str) -> None:
    """Existing implicit SciPy methods and analytic Jacobians track a stiff equation."""

    def stiff(
        phases: NDArray[np.float64], parameters: KuramotoParameters, time: float
    ) -> NDArray[np.float64]:
        return -1000 * (phases - 1)

    def jacobian(
        phases: NDArray[np.float64], parameters: KuramotoParameters, time: float
    ) -> NDArray[np.float64]:
        return np.array([[-1000.0]])

    system = KuramotoSystem(
        stiff, np.zeros(1), KuramotoParameters(np.zeros(1), 0.0), dt=0.01, jacobian=jacobian
    )
    result = solve_kuramoto_ivp(
        system,
        (0, 0.05),
        method=method,
        use_jacobian=True,
        t_eval=np.linspace(0, 0.05, 51),
        rtol=1e-9,
        atol=1e-12,
    )
    assert result.success and result.termination == "completed"
    np.testing.assert_allclose(result.phases[:, 0], 1 - np.exp(-1000 * result.times), atol=5e-9)
    assert result.jacobian_evaluations >= 1
    np.testing.assert_array_equal(system.current_state, [0.0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_fixed_rejection_preserves_state(bad: float) -> None:
    """Invalid domains are refused before modifying the accepted state or clock."""
    with pytest.raises(ValueError, match="finite"):
        KuramotoSystem.mean_field(np.zeros(2), np.ones(2), 0.0, dt=bad)
    system = KuramotoSystem.mean_field(np.zeros(2), np.ones(2), 0.0, dt=0.1)
    for operation in (
        lambda: system.set_state(np.array([bad, 0])),
        lambda: system.reinit(np.ones(2), time=bad),
        lambda: system.step(dt=bad),
        lambda: system.trajectory_with_times(2, dt=bad),
        lambda: system.set_parameter("coupling", bad),
    ):
        with pytest.raises(ValueError, match="finite"):
            operation()
        np.testing.assert_array_equal(system.current_state, [0, 0])
        assert system.current_time == 0


@pytest.mark.parametrize("count", [True, False, 0, -1, 1.5])
def test_fixed_count_refusal(count: object) -> None:
    """The public discrete step budget must be an integer, excluding booleans."""
    system = KuramotoSystem.mean_field(np.zeros(1), np.ones(1), 0.0, dt=0.1)
    with pytest.raises(ValueError, match="integer"):
        system.step(n=cast(int, count))
    with pytest.raises(ValueError, match="integer"):
        system.trajectory_with_times(cast(int, count))
    assert system.current_time == 0


@pytest.mark.parametrize("scheme", ["euler", "rk4"])
def test_failed_fixed_batch_is_atomic(scheme: str) -> None:
    """A real callback fault after an accepted local step preserves pre-call state."""

    def failing(
        phases: NDArray[np.float64], parameters: KuramotoParameters, time: float
    ) -> NDArray[np.float64]:
        if time >= 0.15:
            return np.full_like(phases, np.nan)
        return np.ones_like(phases)

    system = KuramotoSystem(
        failing, np.zeros(2), KuramotoParameters(np.zeros(2), 0.0), dt=0.1, scheme=scheme
    )
    with pytest.raises(ValueError, match="finite"):
        system.step(n=4)
    np.testing.assert_array_equal(system.current_state, [0, 0])
    with pytest.raises(ValueError, match="finite"):
        system.trajectory_with_times(4)
    np.testing.assert_array_equal(system.current_state, [0, 0])
    assert system.current_time == 0


def test_scipy_malformed_rule_refuses_broadcast() -> None:
    """A two-state solver cannot treat a one-element callback as a valid field."""

    def malformed(
        phases: NDArray[np.float64], parameters: KuramotoParameters, time: float
    ) -> NDArray[np.float64]:
        return np.ones(1)

    system = KuramotoSystem(malformed, np.zeros(2), KuramotoParameters(np.zeros(2), 0.0), dt=0.1)
    with pytest.raises(ValueError, match="finite.*vector"):
        solve_kuramoto_ivp(system, (0, 0.3))
    np.testing.assert_array_equal(system.current_state, [0, 0])


def test_adaptive_time_tolerance_refinement() -> None:
    """The original Dormand-Prince controller converges to a nonlinear analytic flow."""
    coupling = np.array([[0.0, 0.7], [0.7, 0.0]])
    errors: list[float] = []
    for tolerance in (1e-4, 1e-6, 1e-8):
        result = kuramoto_dopri_trajectory(
            np.array([-0.8, 0.8]),
            np.full(2, 0.2),
            coupling,
            t_end=4.0,
            rtol=tolerance,
            atol=tolerance * 0.01,
            first_step=0.1,
        )
        delta = 2 * np.arctan(np.tan(0.8) * np.exp(-1.4 * result.times))
        exact = 0.2 * result.times[:, None] + np.column_stack((-delta / 2, delta / 2))
        errors.append(float(np.max(np.abs(result.phases - exact))))
        assert result.times[-1] == 4.0
    assert errors[0] > 10 * errors[1] and errors[1] > 10 * errors[2]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_plastic_nonfinite_refusal(bad: float) -> None:
    """Both original plastic solvers refuse invalid source arrays and scalars."""
    phases, weights, omega = np.zeros(2), np.zeros((2, 2)), np.zeros(2)
    invalid_phases = np.array([bad, 0.0])

    def plasticity(
        theta: NDArray[np.float64], coupling: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.zeros_like(coupling)

    with pytest.raises(ValueError, match="finite"):
        integrate_adaptive_kuramoto(
            invalid_phases, weights, omega, networked_kuramoto_force, plasticity, dt=0.1, n_steps=2
        )
    with pytest.raises(ValueError, match="finite"):
        adaptive_state_sensitivity(phases, weights, omega, plasticity_rate=bad, dt=0.1, n_steps=2)
    with pytest.raises(ValueError, match="finite"):
        adaptive_state_sensitivity(
            invalid_phases, weights, omega, plasticity_rate=0.1, dt=0.1, n_steps=2
        )
    np.testing.assert_array_equal(phases, [0.0, 0.0])
    np.testing.assert_array_equal(weights, np.zeros((2, 2)))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_all_solver_nonfinite_domains(bad: float) -> None:
    """Finite domains cover each source array, grid scalar and SciPy request."""
    initial, omega, weights = np.zeros(2), np.ones(2), np.zeros((2, 2))
    for frequencies, coupling, frustration in (
        (np.array([bad, 0.0]), weights, 0.0),
        (omega, bad, 0.0),
        (omega, np.full((2, 2), bad), 0.0),
        (omega, weights, bad),
    ):
        with pytest.raises(ValueError, match="finite"):
            KuramotoParameters(frequencies, coupling, frustration)
    with pytest.raises(ValueError, match="finite"):
        KuramotoSystem.mean_field(np.array([bad, 0]), omega, 0.0, dt=0.1)
    system = KuramotoSystem.mean_field(initial, omega, 0.0, dt=0.1)
    for span, times, rtol, atol in (
        ((0.0, bad), None, 1e-6, 1e-9),
        ((0.0, 1.0), [bad], 1e-6, 1e-9),
        ((0.0, 1.0), None, bad, 1e-9),
        ((0.0, 1.0), None, 1e-6, bad),
    ):
        with pytest.raises(ValueError, match="finite"):
            solve_kuramoto_ivp(system, span, t_eval=times, rtol=rtol, atol=atol)
    for delay, dt, tolerance, frequencies in (
        (bad, 0.1, 1e-9, omega),
        (0.2, bad, 1e-9, omega),
        (0.2, 0.1, bad, omega),
        (0.2, 0.1, 1e-9, np.array([bad, 0])),
    ):
        with pytest.raises(ValueError, match="finite"):
            integrate_delayed_kuramoto(
                np.zeros((3, 2)),
                frequencies,
                _zero_force,
                delay=delay,
                dt=dt,
                n_steps=2,
                delay_tolerance=tolerance,
            )

    def zero_plasticity(
        theta: NDArray[np.float64], coupling: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.zeros_like(coupling)

    with pytest.raises(ValueError, match="finite"):
        integrate_adaptive_kuramoto(
            initial, weights, omega, networked_kuramoto_force, zero_plasticity, dt=bad, n_steps=2
        )
    with pytest.raises(ValueError, match="finite"):
        adaptive_vector_field(
            initial, weights, np.array([bad, 0]), networked_kuramoto_force, zero_plasticity
        )
    with pytest.raises(ValueError, match="finite"):
        adaptive_state_sensitivity(initial, weights, omega, plasticity_rate=0.1, dt=bad, n_steps=2)
    with pytest.raises(ValueError, match="finite"):
        adaptive_state_sensitivity(
            initial, np.full((2, 2), bad), omega, plasticity_rate=0.1, dt=0.1, n_steps=2
        )
    with pytest.raises(ValueError, match="finite"):
        adaptive_state_sensitivity(
            initial, weights, np.array([bad, 0]), plasticity_rate=0.1, dt=0.1, n_steps=2
        )
    with pytest.raises(ValueError, match="finite"):
        hebbian_plasticity_rate(initial, weights, plasticity_rate=bad)
    with pytest.raises(ValueError, match="finite"):
        hebbian_adaptive_jacobian(initial, weights, plasticity_rate=bad)
    np.testing.assert_array_equal(system.current_state, initial)
    assert system.current_time == 0


@pytest.mark.parametrize("count", [True, False, 1.5, 0, -1])
def test_delay_and_plastic_count_refusal(count: object) -> None:
    """Each public discrete solver refuses malformed step budgets before allocation."""
    with pytest.raises(ValueError, match="integer"):
        integrate_delayed_kuramoto(
            np.zeros((3, 1)), np.zeros(1), _zero_force, delay=0.2, dt=0.1, n_steps=cast(int, count)
        )
    with pytest.raises(ValueError, match="integer"):
        adaptive_state_sensitivity(
            np.zeros(1),
            np.zeros((1, 1)),
            np.zeros(1),
            plasticity_rate=0.1,
            dt=0.1,
            n_steps=cast(int, count),
        )

    def plasticity(
        theta: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.zeros_like(weights)

    with pytest.raises(ValueError, match="integer"):
        integrate_adaptive_kuramoto(
            np.zeros(1),
            np.zeros((1, 1)),
            np.zeros(1),
            networked_kuramoto_force,
            plasticity,
            dt=0.1,
            n_steps=cast(int, count),
        )


@pytest.mark.parametrize("kind", ["shape", "finite"])
def test_malformed_vector_fields_refuse(kind: Literal["shape", "finite"]) -> None:
    """Public callback paths cannot broadcast a malformed or nonfinite field."""

    def invalid_force(
        current: NDArray[np.float64], lagged: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.zeros(1) if kind == "shape" else np.full_like(current, np.nan)

    def invalid_rule(
        phases: NDArray[np.float64], parameters: KuramotoParameters, time: float
    ) -> NDArray[np.float64]:
        return invalid_force(phases, phases)

    system = KuramotoSystem(
        invalid_rule, np.zeros(2), KuramotoParameters(np.zeros(2), 0.0), dt=0.1
    )
    with pytest.raises(ValueError, match="finite.*vector"):
        system.step()
    with pytest.raises(ValueError, match="finite.*vector"):
        integrate_delayed_kuramoto(
            np.zeros((3, 2)), np.zeros(2), invalid_force, delay=0.2, dt=0.1, n_steps=2
        )

    def plasticity(
        phases: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.zeros_like(weights)

    with pytest.raises(ValueError, match="shapes|finite"):
        integrate_adaptive_kuramoto(
            np.zeros(2),
            np.zeros((2, 2)),
            np.zeros(2),
            invalid_force,
            plasticity,
            dt=0.1,
            n_steps=2,
        )
    np.testing.assert_array_equal(system.current_state, [0, 0])
    assert system.current_time == 0


def test_fixed_clock_resolution_refusal() -> None:
    """A step smaller than the current clock's ULP cannot claim time evolution."""
    system = KuramotoSystem.mean_field(np.zeros(1), np.ones(1), 0.0, dt=0.1)
    system.reinit(time=1e20)
    with pytest.raises(ValueError, match="representable"):
        system.step()
    with pytest.raises(ValueError, match="finite"):
        system.trajectory_with_times(2, dt=1e308)
    np.testing.assert_array_equal(system.current_state, [0])
    assert system.current_time == 1e20


def test_finite_inputs_with_overflow_refuse_outputs() -> None:
    """Representable input scalars cannot legitimize an overflowed proposal."""

    def huge_rule(
        phases: NDArray[np.float64], parameters: KuramotoParameters, time: float
    ) -> NDArray[np.float64]:
        return np.full_like(phases, 1e308)

    def huge_force(
        current: NDArray[np.float64], lagged: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.full_like(current, 1e308)

    system = KuramotoSystem(
        huge_rule, np.zeros(1), KuramotoParameters(np.zeros(1), 0.0), dt=2.0, scheme="euler"
    )
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="evolved state.*finite"):
            system.step()
        with pytest.raises(ValueError, match="evolved phases.*finite"):
            integrate_delayed_kuramoto(
                np.zeros((2, 1)), np.zeros(1), huge_force, delay=1, dt=1, n_steps=1
            )
        with pytest.raises(ValueError, match="sensitivity.*finite"):
            adaptive_state_sensitivity(
                np.zeros(1), np.ones((1, 1)), np.zeros(1), plasticity_rate=1e308, dt=0.1, n_steps=1
            )
    np.testing.assert_array_equal(system.current_state, [0])
    assert system.current_time == 0


def test_all_grid_overflow_and_negative_tolerances_refuse() -> None:
    """Grid representability is checked before allocating or interpreting history."""

    def plasticity(
        phases: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.zeros_like(weights)

    with pytest.raises(ValueError, match="end time.*finite"):
        integrate_adaptive_kuramoto(
            np.zeros(1),
            np.zeros((1, 1)),
            np.zeros(1),
            networked_kuramoto_force,
            plasticity,
            dt=1e308,
            n_steps=2,
        )
    with pytest.raises(ValueError, match="end time.*finite"):
        adaptive_state_sensitivity(
            np.zeros(1), np.zeros((1, 1)), np.zeros(1), plasticity_rate=0, dt=1e308, n_steps=2
        )
    with pytest.raises(ValueError, match="grid.*finite"):
        integrate_delayed_kuramoto(
            np.zeros((2, 1)), np.zeros(1), _zero_force, delay=1e308, dt=1e-308, n_steps=2
        )
    with pytest.raises(ValueError, match="non-negative"):
        integrate_delayed_kuramoto(
            np.zeros((2, 1)),
            np.zeros(1),
            _zero_force,
            delay=1,
            dt=1,
            n_steps=1,
            delay_tolerance=-1,
        )
    system = KuramotoSystem.mean_field(np.zeros(1), np.ones(1), 0.0, dt=0.1)
    with pytest.raises(ValueError, match="non-negative"):
        solve_kuramoto_ivp(system, (0, 1), rtol=-1)
    with pytest.raises(ValueError, match="non-negative"):
        solve_kuramoto_ivp(system, (0, 1), atol=-1)


@pytest.mark.parametrize("malformed", ["shape", "finite"])
def test_plasticity_callback_refusal(malformed: str) -> None:
    """Coupling callbacks have their own matrix shape and finite-output contract."""

    def invalid_plasticity(
        phases: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.zeros((1, 1)) if malformed == "shape" else np.full_like(weights, np.nan)

    with pytest.raises(ValueError, match="shapes|finite"):
        integrate_adaptive_kuramoto(
            np.zeros(2),
            np.zeros((2, 2)),
            np.zeros(2),
            networked_kuramoto_force,
            invalid_plasticity,
            dt=0.1,
            n_steps=1,
        )


def test_scipy_event_direction_nonterminal_and_backward() -> None:
    """SciPy retains direction, nonterminal roots and backward interval semantics."""
    system = KuramotoSystem.mean_field(np.zeros(1), np.ones(1), 0.0, dt=0.1)
    rising = PhaseThreshold()
    rising.terminal = False
    completed = solve_kuramoto_ivp(system, (0, 1), events=[rising])
    assert completed.termination == "completed" and completed.times[-1] == 1
    np.testing.assert_allclose(completed.event_times[0], [0.3], atol=2e-15)
    rising.direction = -1
    ignored = solve_kuramoto_ivp(system, (0, 1), events=rising)
    assert ignored.termination == "completed" and ignored.event_times[0].size == 0
    assert ignored.event_phases[0].shape == (0, 1)
    system.set_state(np.array([0.6]))
    rising.terminal = True
    backward = solve_kuramoto_ivp(system, (1, 0), events=rising)
    assert backward.termination == "event"
    np.testing.assert_allclose(backward.event_times[0], [0.7], atol=2e-15)
    np.testing.assert_allclose(backward.event_phases[0], [[0.3]], atol=2e-15)
    np.testing.assert_array_equal(system.current_state, [0.6])


def test_scipy_nonfinite_field_refusal() -> None:
    """A nonfinite callback is rejected before SciPy can attempt integration."""

    def nonfinite(
        phases: NDArray[np.float64], parameters: KuramotoParameters, time: float
    ) -> NDArray[np.float64]:
        return np.full_like(phases, np.nan)

    system = KuramotoSystem(nonfinite, np.zeros(2), KuramotoParameters(np.zeros(2), 0.0), dt=0.1)
    with pytest.raises(ValueError, match="finite.*vector"):
        solve_kuramoto_ivp(system, (0, 1))
    np.testing.assert_array_equal(system.current_state, [0, 0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_dopri_invalid_controller_refusal(bad: float) -> None:
    """The original adaptive public boundary refuses nonfinite controller inputs."""
    initial, omega, coupling = np.zeros(1), np.ones(1), np.zeros((1, 1))
    for horizon, rtol, atol, first, safety, minimum, maximum in (
        (bad, 1e-6, 1e-9, 0.01, 0.9, 0.2, 5.0),
        (1.0, bad, 1e-9, 0.01, 0.9, 0.2, 5.0),
        (1.0, 1e-6, bad, 0.01, 0.9, 0.2, 5.0),
        (1.0, 1e-6, 1e-9, bad, 0.9, 0.2, 5.0),
        (1.0, 1e-6, 1e-9, 0.01, bad, 0.2, 5.0),
        (1.0, 1e-6, 1e-9, 0.01, 0.9, bad, 5.0),
        (1.0, 1e-6, 1e-9, 0.01, 0.9, 0.2, bad),
    ):
        with pytest.raises(ValueError, match="finite"):
            kuramoto_dopri_trajectory(
                initial,
                omega,
                coupling,
                t_end=horizon,
                rtol=rtol,
                atol=atol,
                first_step=first,
                safety=safety,
                min_factor=minimum,
                max_factor=maximum,
            )
    for phases, frequencies, weights in (
        (np.array([bad]), omega, coupling),
        (initial, np.array([bad]), coupling),
        (initial, omega, np.array([[bad]])),
    ):
        with pytest.raises(ValueError, match="finite"):
            kuramoto_dopri_trajectory(phases, frequencies, weights, t_end=1.0, first_step=0.01)
    np.testing.assert_array_equal(initial, [0])
    np.testing.assert_array_equal(omega, [1])


@pytest.mark.parametrize("count", [True, False, 0, -1, 1.5])
def test_dopri_count_refusal(count: object) -> None:
    """Adaptive accepted-step limits retain the same positive integer contract."""
    with pytest.raises(ValueError, match="integer"):
        kuramoto_dopri_trajectory(
            np.zeros(1),
            np.ones(1),
            np.zeros((1, 1)),
            t_end=1,
            first_step=0.01,
            max_steps=cast(int, count),
        )


def test_dopri_underresolved_horizon_refusal() -> None:
    """The unchanged integrator's clock floor cannot count as a completed horizon."""
    with pytest.raises(ValueError, match="before reaching"):
        kuramoto_dopri_trajectory(
            np.zeros(1), np.ones(1), np.zeros((1, 1)), t_end=5e-15, first_step=1e-15
        )


@pytest.mark.parametrize(
    ("safety", "minimum", "maximum"),
    [(0.0, 0.2, 5.0), (1.0, 0.2, 5.0), (0.9, 1.0, 5.0), (0.9, 0.0, 5.0), (0.9, 0.2, 0.5)],
)
def test_dopri_inconsistent_controller_refusal(
    safety: float, minimum: float, maximum: float
) -> None:
    """A controller unable to shrink rejected steps is refused before integration."""
    with pytest.raises(ValueError, match="controller"):
        kuramoto_dopri_trajectory(
            np.zeros(1),
            np.ones(1),
            np.zeros((1, 1)),
            t_end=1,
            first_step=0.01,
            safety=safety,
            min_factor=minimum,
            max_factor=maximum,
        )
