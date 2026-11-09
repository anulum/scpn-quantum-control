# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Parameter-Shift Gradient Descent
"""Tests for phase/gradient_descent.py parameter-shift training semantics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scpn_quantum_control.differentiable import ParameterBounds
from scpn_quantum_control.phase import (
    ParameterShiftTrainingResult,
    multi_frequency_parameter_shift_rule,
    parameter_shift_gradient_descent,
    validate_parameter_shift_training,
)


def test_parameter_shift_gradient_descent_converges_on_rotation_cost() -> None:
    """Converge on the analytic single-frequency rotation minimum."""

    def objective(params: np.ndarray) -> float:
        return float(1.0 - np.cos(params[0]))

    result = parameter_shift_gradient_descent(
        objective,
        np.array([0.8], dtype=float),
        learning_rate=0.5,
        max_steps=80,
        gradient_tolerance=1e-7,
    )
    certificate = validate_parameter_shift_training(
        result,
        gradient_tolerance=1e-7,
        target_value=0.0,
        target_value_tolerance=1e-10,
        min_decrease=0.1,
    )

    assert isinstance(result, ParameterShiftTrainingResult)
    assert result.converged
    assert result.reason == "gradient_tolerance"
    assert result.accepted_steps > 0
    assert result.rejected_steps == 0
    assert result.best_value < 1e-10
    assert result.final_gradient_norm <= 1e-7
    assert result.backend_plan.method == "parameter_shift"
    assert certificate.monotone_accepted_values
    assert certificate.within_gradient_tolerance
    assert certificate.within_target_value_tolerance
    assert certificate.min_decrease_satisfied


def test_parameter_shift_gradient_descent_records_multi_frequency_rule() -> None:
    """Retain the multi-frequency method and shift accounting through descent."""
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(params: np.ndarray) -> float:
        return float(1.0 - np.cos(params[0]) + 0.05 * (1.0 - np.cos(2.0 * params[0])))

    result = parameter_shift_gradient_descent(
        objective,
        np.array([0.7], dtype=float),
        rule=rule,
        learning_rate=0.4,
        max_steps=80,
        gradient_tolerance=1e-7,
    )
    payload = result.to_dict()

    assert result.converged
    assert result.method == "multi_frequency_parameter_shift"
    assert result.shift_terms == len(rule.terms)
    assert result.backend_plan.shift_terms == len(rule.terms)
    assert result.best_value < 1e-10
    assert payload["shift_terms"] == len(rule.terms)
    assert all(step.shift_terms == len(rule.terms) for step in result.steps)


def test_parameter_shift_gradient_descent_fails_closed_for_hardware() -> None:
    """Refuse an unapproved hardware backend before any objective evaluation."""

    def objective(params: np.ndarray) -> float:
        return float(1.0 - np.cos(params[0]))

    reference = parameter_shift_gradient_descent(objective, np.array([0.4]), max_steps=1)
    assert np.isfinite(reference.final_value)
    with pytest.raises(ValueError, match="hardware gradient execution requires"):
        parameter_shift_gradient_descent(
            objective,
            np.array([0.4], dtype=float),
            backend="hardware",
            max_steps=2,
        )


def test_parameter_shift_gradient_descent_rejects_invalid_training_controls() -> None:
    """Reject nonpositive rates, invalid backtracking and nonscalar parameter axes."""

    def objective(params: np.ndarray) -> float:
        return float(1.0 - np.cos(params[0]))

    reference = parameter_shift_gradient_descent(objective, np.array([0.4]), max_steps=1)
    assert np.isfinite(reference.final_value)
    with pytest.raises(ValueError, match="learning_rate"):
        parameter_shift_gradient_descent(
            objective,
            np.array([0.4], dtype=float),
            learning_rate=0.0,
        )

    with pytest.raises(ValueError, match="backtracking_factor"):
        parameter_shift_gradient_descent(
            objective,
            np.array([0.4], dtype=float),
            backtracking_factor=1.0,
        )

    with pytest.raises(ValueError, match="initial_params"):
        parameter_shift_gradient_descent(objective, np.array([[0.4]], dtype=float))


def test_parameter_shift_gradient_descent_rejects_nonfinite_objectives() -> None:
    """Refuse a non-finite value from the actual scalar objective."""

    def objective(params: np.ndarray) -> float:
        if abs(params[0] - 0.4) < 1e-12:
            return float("nan")
        return float(1.0 - np.cos(params[0]))

    reference = parameter_shift_gradient_descent(objective, np.array([0.5]), max_steps=1)
    assert np.isfinite(reference.final_value)
    with pytest.raises(ValueError, match="objective"):
        parameter_shift_gradient_descent(
            objective,
            np.array([0.4], dtype=float),
            max_steps=2,
        )


def test_projected_descent_keeps_probes_free_and_reports_a_feasible_minimum() -> None:
    """Project iterates onto a box while differentiating the original periodic cost."""
    probes: list[float] = []

    def objective(params: np.ndarray) -> float:
        value = float(params[0])
        probes.append(value)
        return float(1.0 - np.cos(value))

    result = parameter_shift_gradient_descent(
        objective,
        np.array([0.5]),
        bounds=[ParameterBounds(lower=0.2, upper=0.6)],
        learning_rate=10.0,
        max_steps=3,
        gradient_tolerance=1e-12,
    )
    np.testing.assert_allclose(result.final_params, [0.2], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(result.final_gradient, [np.sin(0.2)], rtol=1e-12, atol=1e-14)
    assert result.converged and result.reason == "projected_gradient_tolerance"
    assert all(0.2 <= step.params[0] <= 0.6 for step in result.steps)
    assert min(probes) < 0.2 and max(probes) > 0.6
    assert result.to_dict()["bounds"] == [{"lower": 0.2, "upper": 0.6}]


@pytest.mark.parametrize("max_steps", [1, 3])
@pytest.mark.parametrize("upper", [False, True])
def test_box_descent_records_projected_stationarity_after_the_final_step(
    max_steps: int, upper: bool
) -> None:
    """Keep both boundary optima and both convergence exits observable.

    Parameters
    ----------
    max_steps
        End immediately after the accepted step or enter the next gradient iteration.
    upper
        Whether the objective is minimised at the upper rather than lower bound.

    """
    sign = 1.0 if upper else -1.0

    def objective(params: np.ndarray) -> float:
        return float(1.0 + sign * np.cos(params[0]))

    result = parameter_shift_gradient_descent(
        objective,
        np.array([0.5]),
        bounds=[ParameterBounds(lower=0.2, upper=0.6)],
        learning_rate=10.0,
        max_steps=max_steps,
        gradient_tolerance=1e-12,
    )
    expected = 0.6 if upper else 0.2
    np.testing.assert_allclose(result.final_params, [expected], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(result.final_gradient, [-sign * np.sin(expected)], atol=1e-14)
    assert result.converged and result.reason == "projected_gradient_tolerance"
    assert result.final_gradient_norm > 0.1
    assert result.bounds == (ParameterBounds(lower=0.2, upper=0.6),)
    certificate = validate_parameter_shift_training(result, gradient_tolerance=1e-12)
    assert certificate.converged and certificate.within_gradient_tolerance is False


@pytest.mark.parametrize(
    "bounds,initial,message",
    [
        ([], 0.5, "bounds length"),
        ([ParameterBounds(lower=0.2, upper=0.6)], 0.1, "within the box bounds"),
        ([ParameterBounds(lower=0.0, upper=1.0, periodic=True)], 0.5, "not periodic"),
    ],
)
def test_invalid_box_bounds_refuse_before_the_objective_runs(
    bounds: list[ParameterBounds], initial: float, message: str
) -> None:
    """Reject unsupported bounds before consuming an objective or altering its caller.

    Parameters
    ----------
    bounds
        Missing, infeasible or periodic intervals.
    initial
        Initial parameter supplied to the actual optimiser.
    message
        Observable rejection contract.

    """
    calls: list[float] = []

    def objective(params: np.ndarray) -> float:
        calls.append(float(params[0]))
        return float(1.0 - np.cos(params[0]))

    reference = parameter_shift_gradient_descent(objective, np.array([0.5]), max_steps=1)
    assert calls and np.isfinite(reference.final_value)
    calls.clear()
    values = np.array([initial])
    with pytest.raises(ValueError, match=message):
        parameter_shift_gradient_descent(objective, values, bounds=bounds)
    assert calls == []
    np.testing.assert_array_equal(values, [initial])


@pytest.mark.parametrize("corruption", ["shape", "nonfinite", "norm"])
def test_training_certificate_refuses_corrupted_gradient_evidence(corruption: str) -> None:
    """Do not certify a caller-modified derivative using only its claimed norm.

    Parameters
    ----------
    corruption
        Wrong gradient length, a non-finite derivative or a false zero norm.

    """

    def objective(params: np.ndarray) -> float:
        return float(1.0 - np.cos(params[0]))

    result = parameter_shift_gradient_descent(objective, np.array([0.5]), max_steps=1)
    if corruption == "shape":
        result = replace(result, final_gradient=np.array([0.0, 0.0]))
        message = "gradient shape"
    elif corruption == "nonfinite":
        result = replace(result, final_gradient=np.array([np.nan]))
        message = "gradient must contain only finite"
    else:
        result = replace(result, final_gradient_norm=0.0)
        message = "gradient norm"
    with pytest.raises(ValueError, match=message):
        validate_parameter_shift_training(result, gradient_tolerance=1e-12)


@pytest.mark.parametrize("field", ["initial_value", "final_value", "best_value"])
def test_training_certificate_refuses_nonfinite_objective_records(field: str) -> None:
    """Refuse non-finite trace values before issuing any convergence certificate.

    Parameters
    ----------
    field
        Public trace scalar corrupted after a real completed descent.

    """

    def objective(params: np.ndarray) -> float:
        return float(1.0 - np.cos(params[0]))

    result = parameter_shift_gradient_descent(objective, np.array([0.5]), max_steps=1)
    if field == "initial_value":
        invalid = replace(result, initial_value=float("nan"))
    elif field == "final_value":
        invalid = replace(result, final_value=float("nan"))
    else:
        invalid = replace(result, best_value=float("nan"))
    with pytest.raises(ValueError, match="objective must return a finite"):
        validate_parameter_shift_training(invalid)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_descent_refuses_an_objective_that_changes_at_identical_parameters(
    direction: float,
) -> None:
    """Do not promote a changing loss as deterministic parameter-shift evidence.

    Parameters
    ----------
    direction
        Whether repeated evaluation of the same input drifts downward or upward.

    """
    seen: dict[tuple[float, ...], int] = {}

    def objective(params: np.ndarray) -> float:
        key = tuple(float(value) for value in params)
        previous = seen.get(key, 0)
        seen[key] = previous + 1
        return float(1.0 - np.cos(params[0]) + direction * previous * 0.01)

    initial = np.array([0.5])
    with pytest.raises(ValueError, match="objective changed at final parameters"):
        parameter_shift_gradient_descent(objective, initial, max_steps=1)
    np.testing.assert_array_equal(initial, [0.5])


@pytest.mark.parametrize(
    "control", ["gradient", "value", "steps", "backtracks", "decrease", "nonfinite", "initial"]
)
def test_invalid_optimizer_controls_refuse_without_a_new_objective_call(control: str) -> None:
    """Reject invalid numeric controls after the same cost has a real positive run.

    Parameters
    ----------
    control
        The invalid tolerance, iteration budget or line-search factor.

    """
    calls: list[float] = []

    def objective(params: np.ndarray) -> float:
        calls.append(float(params[0]))
        return float(1.0 - np.cos(params[0]))

    parameter_shift_gradient_descent(objective, np.array([0.5]), max_steps=1)
    assert calls
    calls.clear()
    with pytest.raises(ValueError):
        if control == "gradient":
            parameter_shift_gradient_descent(objective, [0.5], gradient_tolerance=-1.0)
        elif control == "value":
            parameter_shift_gradient_descent(objective, [0.5], value_tolerance=-1.0)
        elif control == "steps":
            parameter_shift_gradient_descent(objective, [0.5], max_steps=0)
        elif control == "backtracks":
            parameter_shift_gradient_descent(objective, [0.5], max_backtracks=0)
        elif control == "decrease":
            parameter_shift_gradient_descent(objective, [0.5], sufficient_decrease=0.0)
        elif control == "nonfinite":
            parameter_shift_gradient_descent(objective, [0.5], backtracking_factor=float("nan"))
        else:
            parameter_shift_gradient_descent(objective, [float("nan")])
    assert calls == []


@pytest.mark.parametrize("rate,backtracks", [(4.0, 2), (8.0, 1)])
def test_backtracking_records_accepted_and_exhausted_rotation_steps(
    rate: float, backtracks: int
) -> None:
    """Retain genuine line-search refusal and backtracking cost for a periodic objective.

    Parameters
    ----------
    rate
        Initial step requiring reduction or exhausting both admissible trials.
    backtracks
        Frozen line-search reduction budget.

    """

    def objective(params: np.ndarray) -> float:
        return float(1.0 - np.cos(params[0]))

    result = parameter_shift_gradient_descent(
        objective, [0.8], learning_rate=rate, max_backtracks=backtracks, max_steps=1
    )
    assert result.steps[0].backtracks > 0
    if rate == 8.0:
        assert result.rejected_steps == 1 and result.reason == "line_search_failed"
        assert not result.converged
        np.testing.assert_array_equal(result.final_params, result.initial_params)
    else:
        assert result.accepted_steps == 1 and result.final_value < result.initial_value
    certificate = validate_parameter_shift_training(result)
    assert certificate.within_gradient_tolerance is None


@pytest.mark.parametrize("value_tolerance", [None, 1.0])
def test_gradient_and_value_convergence_are_reported_separately(
    value_tolerance: float | None,
) -> None:
    """Observe final-iteration gradient convergence and a distinct value-change exit.

    Parameters
    ----------
    value_tolerance
        Enable value convergence or require the analytic zero-gradient endpoint.

    """

    def objective(params: np.ndarray) -> float:
        return float(1.0 - np.cos(params[0]))

    result = parameter_shift_gradient_descent(
        objective,
        [float(np.pi / 2.0)],
        learning_rate=float(np.pi / 2.0),
        max_steps=1,
        gradient_tolerance=1e-12,
        value_tolerance=value_tolerance,
    )
    assert result.converged
    assert result.reason == (
        "gradient_tolerance" if value_tolerance is None else "value_tolerance"
    )
    assert result.final_value == pytest.approx(0.0, abs=1e-14)


def test_unresolvable_small_step_does_not_claim_a_loss_decrease() -> None:
    """Keep machine-resolution-limited progress distinct from a convergence claim."""

    def objective(params: np.ndarray) -> float:
        return float(1.0 - np.cos(params[0]))

    result = parameter_shift_gradient_descent(
        objective, [0.5], learning_rate=1e-300, max_steps=1, gradient_tolerance=0.0
    )
    assert not result.converged
    assert result.initial_value == result.final_value == result.best_value
    np.testing.assert_array_equal(result.initial_params, result.final_params)
