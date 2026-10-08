# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — public solver refusal and recovery contracts
"""Exercise finite-state admission, callback results and representable solver clocks."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel.diff_kuramoto_adaptive import adaptive_state_sensitivity
from oscillatools.accel.diff_kuramoto_dopri import kuramoto_dopri_trajectory
from oscillatools.accel.kuramoto_adaptive import (
    adaptive_vector_field,
    integrate_adaptive_kuramoto,
)
from oscillatools.accel.kuramoto_delayed import integrate_delayed_kuramoto
from oscillatools.accel.kuramoto_scipy_interop import solve_kuramoto_ivp
from oscillatools.accel.kuramoto_system import KuramotoParameters, KuramotoSystem

FloatArray = NDArray[np.float64]


def _zero_phase_force(phases: FloatArray, coupling: FloatArray) -> FloatArray:
    """Supply an actual uncoupled phase-force callback."""
    return np.zeros_like(phases)


def _zero_plasticity(phases: FloatArray, coupling: FloatArray) -> FloatArray:
    """Keep the supplied coupling matrix constant."""
    return np.zeros_like(coupling)


def _zero_delayed_force(current: FloatArray, delayed: FloatArray) -> FloatArray:
    """Supply the uncoupled delayed model."""
    return np.zeros_like(current)


def _free_system() -> KuramotoSystem:
    """Construct the public model with an independently known linear trajectory."""
    return KuramotoSystem.mean_field(np.array([0.2, -0.1]), np.array([0.3, -0.4]), 0.0, dt=0.1)


@pytest.mark.parametrize("field", ["phases", "coupling", "omega"])
def test_adaptive_sensitivity_refuses_nonfinite_inputs_and_recovers(field: str) -> None:
    """Bad input cannot contaminate the next uncoupled trajectory or its gradient."""
    arrays = [np.zeros(2), np.zeros((2, 2)), np.array([0.3, -0.4])]
    bad = [value.copy() for value in arrays]
    bad[["phases", "coupling", "omega"].index(field)].flat[0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        adaptive_state_sensitivity(*bad, plasticity_rate=0.0, dt=0.1, n_steps=1)
    phases, coupling, sensitivity = adaptive_state_sensitivity(
        *arrays, plasticity_rate=0.0, dt=0.1, n_steps=1
    )
    np.testing.assert_allclose(phases, arrays[2] * 0.1, atol=1e-12)
    np.testing.assert_array_equal(coupling, arrays[1])
    np.testing.assert_allclose(sensitivity[:2, 6:8], np.eye(2) * 0.1, atol=1e-12)


def test_adaptive_sensitivity_refuses_time_and_state_overflow() -> None:
    """Finite inputs whose clock or evolved buffers overflow never produce a result."""
    with pytest.raises(ValueError, match="end time must be finite"):
        adaptive_state_sensitivity(
            np.zeros(1),
            np.zeros((1, 1)),
            np.ones(1),
            plasticity_rate=0.0,
            dt=1e308,
            n_steps=2,
        )
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="state and sensitivity must be finite"),
    ):
        adaptive_state_sensitivity(
            np.zeros(1),
            np.zeros((1, 1)),
            np.full(1, 1e308),
            plasticity_rate=0.0,
            dt=1.0,
            n_steps=1,
        )
    result, _, gradient = adaptive_state_sensitivity(
        np.zeros(1),
        np.zeros((1, 1)),
        np.ones(1),
        plasticity_rate=0.0,
        dt=0.1,
        n_steps=1,
    )
    np.testing.assert_allclose(result, [0.1], atol=1e-12)
    assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize("field", ["phases", "omega", "coupling"])
def test_dopri_refuses_nonfinite_initial_data_and_recovers(field: str) -> None:
    """Admission rejects corrupted inputs before a real error-controlled solve."""
    arrays = [np.zeros(2), np.array([0.3, -0.4]), np.zeros((2, 2))]
    bad = [value.copy() for value in arrays]
    bad[["phases", "omega", "coupling"].index(field)].flat[0] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        kuramoto_dopri_trajectory(*bad, t_end=0.1)
    result = kuramoto_dopri_trajectory(*arrays, t_end=0.1, first_step=0.05)
    np.testing.assert_allclose(result.phases[-1], arrays[1] * 0.1, atol=1e-10)


@pytest.mark.parametrize("end,relative,absolute", [(np.inf, 1e-6, 1e-9), (0.1, np.nan, 1e-9)])
def test_dopri_refuses_nonfinite_controller_scalars(
    end: float, relative: float, absolute: float
) -> None:
    """A nonfinite endpoint or tolerance is not an integration request."""
    with pytest.raises(ValueError, match="controller scalars must be finite"):
        kuramoto_dopri_trajectory(
            np.zeros(1),
            np.ones(1),
            np.zeros((1, 1)),
            t_end=end,
            rtol=relative,
            atol=absolute,
        )


@pytest.mark.parametrize("max_steps", [0, True])
def test_dopri_refuses_invalid_discrete_budget(max_steps: int) -> None:
    """Zero and boolean budgets cannot masquerade as an accepted step count."""
    with pytest.raises(ValueError, match="max_steps must be a positive integer"):
        kuramoto_dopri_trajectory(
            np.zeros(1), np.ones(1), np.zeros((1, 1)), t_end=0.1, max_steps=max_steps
        )


@pytest.mark.parametrize(
    "safety,minimum,maximum", [(1.0, 0.2, 5.0), (0.9, 1.0, 5.0), (0.9, 0.2, 0.5)]
)
def test_dopri_refuses_invalid_controller_bounds(
    safety: float, minimum: float, maximum: float
) -> None:
    """The adaptive controller must have a bounded shrinking and growth policy."""
    with pytest.raises(ValueError, match="controller requires"):
        kuramoto_dopri_trajectory(
            np.zeros(1),
            np.ones(1),
            np.zeros((1, 1)),
            t_end=0.1,
            safety=safety,
            min_factor=minimum,
            max_factor=maximum,
        )


@pytest.mark.parametrize("field", ["phases", "coupling", "omega"])
def test_adaptive_field_refuses_nonfinite_state(field: str) -> None:
    """The public field cannot admit nonfinite phase, weight or frequency data."""
    arrays = [np.zeros(2), np.zeros((2, 2)), np.ones(2)]
    arrays[["phases", "coupling", "omega"].index(field)].flat[0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        adaptive_vector_field(arrays[0], arrays[1], arrays[2], _zero_phase_force, _zero_plasticity)


@pytest.mark.parametrize(
    "fault", ["force_shape", "plasticity_shape", "force_finite", "plasticity_finite"]
)
def test_adaptive_field_refuses_invalid_callback_results_and_recovers(fault: str) -> None:
    """Actual user callbacks cannot publish malformed joint velocities."""
    phases, coupling, omega = np.zeros(2), np.zeros((2, 2)), np.array([0.3, -0.4])

    def force(state: FloatArray, weights: FloatArray) -> FloatArray:
        if fault == "force_shape":
            return np.zeros(3)
        if fault == "force_finite":
            return np.full_like(state, np.inf)
        return np.zeros_like(state)

    def plasticity(state: FloatArray, weights: FloatArray) -> FloatArray:
        if fault == "plasticity_shape":
            return np.zeros((1, 2))
        if fault == "plasticity_finite":
            return np.full_like(weights, np.nan)
        return np.zeros_like(weights)

    with pytest.raises(ValueError, match="shapes|velocities must be finite"):
        adaptive_vector_field(phases, coupling, omega, force, plasticity)
    velocity, weights_velocity = adaptive_vector_field(
        phases, coupling, omega, _zero_phase_force, _zero_plasticity
    )
    np.testing.assert_array_equal(velocity, omega)
    np.testing.assert_array_equal(weights_velocity, coupling)


def test_adaptive_trajectory_refuses_unrepresentable_endpoint_and_recovers() -> None:
    """A rejected clock leaves caller-owned initial arrays available for another solve."""
    phases, coupling, omega = np.zeros(2), np.zeros((2, 2)), np.array([0.3, -0.4])
    with pytest.raises(ValueError, match="end time must be finite"):
        integrate_adaptive_kuramoto(
            phases,
            coupling,
            omega,
            _zero_phase_force,
            _zero_plasticity,
            dt=1e308,
            n_steps=2,
        )
    trajectory = integrate_adaptive_kuramoto(
        phases, coupling, omega, _zero_phase_force, _zero_plasticity, dt=0.1, n_steps=2
    )
    np.testing.assert_allclose(trajectory.phases[-1], omega * 0.2, atol=1e-12)
    np.testing.assert_array_equal(phases, np.zeros(2))
    np.testing.assert_array_equal(coupling, np.zeros((2, 2)))


@pytest.mark.parametrize("tolerance", [np.nan, -1.0])
def test_delayed_trajectory_refuses_invalid_grid_tolerance(tolerance: float) -> None:
    """A delay grid requires a finite nonnegative matching tolerance."""
    with pytest.raises(ValueError, match="delay_tolerance"):
        integrate_delayed_kuramoto(
            np.zeros((2, 1)),
            np.ones(1),
            _zero_delayed_force,
            delay=0.1,
            dt=0.1,
            n_steps=1,
            delay_tolerance=tolerance,
        )


@pytest.mark.parametrize("delay,step,count", [(1e308, 1e-308, 1), (1e308, 1e308, 2)])
def test_delayed_trajectory_refuses_unrepresentable_grid(
    delay: float, step: float, count: int
) -> None:
    """The delay ratio and endpoint are bounded before constructing history storage."""
    with (
        np.errstate(over="ignore"),
        pytest.raises(ValueError, match="delay grid and end time must be finite"),
    ):
        integrate_delayed_kuramoto(
            np.zeros((2, 1)),
            np.ones(1),
            _zero_delayed_force,
            delay=delay,
            dt=step,
            n_steps=count,
        )


@pytest.mark.parametrize("field", ["history", "omega"])
def test_delayed_trajectory_refuses_nonfinite_input_and_recovers(field: str) -> None:
    """A fresh finite history still follows the analytic uncoupled delayed solution."""
    history, omega = np.zeros((2, 1)), np.array([0.3])
    bad_history, bad_omega = history.copy(), omega.copy()
    (bad_history if field == "history" else bad_omega).flat[0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        integrate_delayed_kuramoto(
            bad_history, bad_omega, _zero_delayed_force, delay=0.1, dt=0.1, n_steps=1
        )
    result = integrate_delayed_kuramoto(
        history, omega, _zero_delayed_force, delay=0.1, dt=0.1, n_steps=2
    )
    np.testing.assert_allclose(result.phases[-1], omega * 0.2, atol=1e-12)
    np.testing.assert_array_equal(history, np.zeros((2, 1)))


@pytest.mark.parametrize("fault", ["shape", "finite"])
def test_delayed_trajectory_refuses_invalid_force_result(fault: str) -> None:
    """The method-of-steps integrator refuses malformed real callback outputs."""
    history = np.zeros((2, 1))

    def force(current: FloatArray, delayed: FloatArray) -> FloatArray:
        return np.zeros(2) if fault == "shape" else np.full_like(current, np.nan)

    with pytest.raises(ValueError, match="force must return a finite vector"):
        integrate_delayed_kuramoto(history, np.ones(1), force, delay=0.1, dt=0.1, n_steps=1)
    np.testing.assert_array_equal(history, np.zeros((2, 1)))


def test_delayed_trajectory_refuses_numeric_overflow_and_recovers() -> None:
    """Finite velocities that overflow RK accumulation cannot escape as a trajectory."""
    history = np.zeros((2, 1))
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="evolved phases must be finite"),
    ):
        integrate_delayed_kuramoto(
            history, np.full(1, 1e308), _zero_delayed_force, delay=1.0, dt=1.0, n_steps=1
        )
    result = integrate_delayed_kuramoto(
        history, np.ones(1), _zero_delayed_force, delay=1.0, dt=1.0, n_steps=1
    )
    np.testing.assert_allclose(result.phases[-1], [1.0], atol=1e-12)


@pytest.mark.parametrize("field", ["omega", "coupling", "frustration", "state"])
def test_system_construction_refuses_nonfinite_model_data(field: str) -> None:
    """A malformed model has no observable accepted state."""
    phases, omega, coupling, frustration = np.zeros(2), np.ones(2), 0.0, 0.0
    if field == "state":
        phases[0] = np.nan
    elif field == "omega":
        omega[0] = np.inf
    elif field == "coupling":
        coupling = np.nan
    else:
        frustration = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        KuramotoSystem.mean_field(phases, omega, coupling, frustration=frustration, dt=0.1)


@pytest.mark.parametrize("operation", ["set_state", "reinit_state", "reinit_time"])
def test_system_invalid_replacement_preserves_accepted_state(operation: str) -> None:
    """Invalid replacement data cannot partially update the accepted state or clock."""
    system = _free_system()
    before = system.current_state
    with pytest.raises(ValueError, match="must be finite"):
        if operation == "set_state":
            system.set_state(np.full(2, np.nan))
        elif operation == "reinit_state":
            system.reinit(np.full(2, np.inf))
        else:
            system.reinit(time=np.inf)
    np.testing.assert_array_equal(system.current_state, before)
    assert system.current_time == 0.0
    np.testing.assert_allclose(system.step(), before + np.array([0.3, -0.4]) * 0.1, atol=1e-12)


@pytest.mark.parametrize("fault", ["shape", "finite"])
def test_system_and_scipy_refuse_bad_user_rules_without_mutating_state(fault: str) -> None:
    """Both public solvers check callback results before committing state."""

    def rule(state: FloatArray, parameters: KuramotoParameters, time: float) -> FloatArray:
        return np.zeros(3) if fault == "shape" else np.full_like(state, np.nan)

    system = KuramotoSystem(rule, np.zeros(2), KuramotoParameters(np.ones(2), 0.0), dt=0.1)
    for solve in (lambda: system.step(), lambda: solve_kuramoto_ivp(system, (0.0, 0.1))):
        with pytest.raises(ValueError, match="rule must return a finite vector"):
            solve()
        np.testing.assert_array_equal(system.current_state, np.zeros(2))
        assert system.current_time == 0.0


@pytest.mark.parametrize("fault", ["endpoint", "rounded_clock", "state_overflow"])
def test_system_refusal_preserves_state_and_allows_recovery(fault: str) -> None:
    """Clock and arithmetic refusal retain the previous state until a valid public step."""
    system = _free_system()
    if fault == "rounded_clock":
        system.reinit(time=1e308)
    elif fault == "state_overflow":
        system.set_parameter("natural_frequencies", np.full(2, 1e308))
    before, before_time = system.current_state, system.current_time
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="end time|step must advance|evolved state"),
    ):
        if fault == "endpoint":
            system.step(n=2, dt=1e308)
        else:
            system.step(dt=1.0)
    np.testing.assert_array_equal(system.current_state, before)
    assert system.current_time == before_time
    system.set_parameter("natural_frequencies", np.array([0.3, -0.4]))
    system.reinit()
    np.testing.assert_allclose(
        system.step(), system.initial_state + np.array([0.3, -0.4]) * 0.1, atol=1e-12
    )


@pytest.mark.parametrize("fault", ["span", "tolerance", "samples"])
def test_scipy_refuses_nonfinite_request_without_advancing_system(fault: str) -> None:
    """Invalid solve requests leave the real system reusable for a finite interval."""
    system = _free_system()
    with pytest.raises(ValueError, match="must be finite|finite and non-negative"):
        if fault == "span":
            solve_kuramoto_ivp(system, (0.0, np.inf))
        elif fault == "tolerance":
            solve_kuramoto_ivp(system, (0.0, 0.1), rtol=np.nan)
        else:
            solve_kuramoto_ivp(system, (0.0, 0.1), t_eval=[0.0, np.inf])
    solution = solve_kuramoto_ivp(system, (0.0, 0.1), t_eval=[0.0, 0.1])
    assert solution.termination == "completed"
    np.testing.assert_allclose(
        solution.terminal_phases, system.initial_state + np.array([0.3, -0.4]) * 0.1, atol=1e-10
    )
    np.testing.assert_array_equal(system.current_state, system.initial_state)
    assert system.current_time == 0.0


def test_scipy_reports_actual_solver_failure_without_mutating_system() -> None:
    """A finite Riccati field blows up before the requested endpoint and is reported failed."""

    def riccati(state: FloatArray, parameters: KuramotoParameters, time: float) -> FloatArray:
        return state * state

    system = KuramotoSystem(riccati, np.ones(1), KuramotoParameters(np.zeros(1), 0.0), dt=0.01)
    solution = solve_kuramoto_ivp(system, (0.0, 2.0), rtol=1e-8, atol=1e-10)
    assert not solution.success
    assert solution.status == -1
    assert solution.termination == "failed"
    assert solution.times[-1] < 2.0
    assert np.all(np.isfinite(solution.terminal_phases))
    np.testing.assert_array_equal(system.current_state, [1.0])
    assert system.current_time == 0.0


def test_scipy_terminal_event_preserves_unsampled_event_state() -> None:
    """An event before the first sample retains its actual state without inventing samples."""
    system = KuramotoSystem.mean_field(np.zeros(1), np.ones(1), 0.0, dt=0.1)

    class StopAtQuarter:
        """Use the public SciPy terminal-event callable protocol."""

        terminal = True

        def __call__(self, time: float, state: FloatArray) -> float:
            """Locate the first quarter-radian crossing in the real trajectory."""
            return float(state[0] - 0.25)

    solution = solve_kuramoto_ivp(system, (0.0, 1.0), t_eval=[0.8, 1.0], events=StopAtQuarter())
    assert solution.success
    assert solution.termination == "event"
    assert solution.phases.shape == (0, 1)
    np.testing.assert_allclose(solution.event_times[0], [0.25], atol=1e-10)
    np.testing.assert_allclose(solution.event_phases[0], [[0.25]], atol=1e-10)
    with pytest.raises(ValueError, match="no stored samples"):
        _ = solution.terminal_phases
    np.testing.assert_array_equal(system.current_state, [0.0])
    assert system.current_time == 0.0
