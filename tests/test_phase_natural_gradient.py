# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Natural Gradient
"""Tests for phase/natural_gradient.py metric-aware training semantics."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    ParameterShiftNaturalGradientResult,
    parameter_shift_gradient_descent,
    parameter_shift_natural_gradient_descent,
    solve_natural_gradient_direction,
    validate_natural_gradient_training,
)


def test_parameter_shift_natural_gradient_converges_with_callable_metric() -> None:
    def objective(params: np.ndarray) -> float:
        return float((1.0 - np.cos(params[0])) + 0.05 * (1.0 - np.cos(params[1])))

    def metric(params: np.ndarray) -> np.ndarray:
        del params
        return np.diag(np.array([1.0, 0.05], dtype=float))

    result = parameter_shift_natural_gradient_descent(
        objective,
        np.array([0.8, -0.7], dtype=float),
        metric_tensor=metric,
        learning_rate=0.5,
        max_steps=80,
        gradient_tolerance=1e-7,
        natural_gradient_tolerance=1e-7,
    )
    certificate = validate_natural_gradient_training(
        result,
        gradient_tolerance=1e-7,
        target_value=0.0,
        target_value_tolerance=1e-10,
        min_decrease=0.1,
    )

    assert isinstance(result, ParameterShiftNaturalGradientResult)
    assert result.converged
    assert result.reason == "gradient_tolerance"
    assert result.metric_source == "callable"
    assert result.accepted_steps > 0
    assert result.rejected_steps == 0
    assert result.best_value < 1e-10
    assert result.final_gradient_norm <= 1e-7
    assert "caller-supplied metric" in result.claim_boundary
    assert certificate.monotone_accepted_values
    assert certificate.within_gradient_tolerance
    assert certificate.within_target_value_tolerance
    assert certificate.min_decrease_satisfied


def test_parameter_shift_natural_gradient_preconditions_slow_phase_axis() -> None:
    def objective(params: np.ndarray) -> float:
        return float((1.0 - np.cos(params[0])) + 0.05 * (1.0 - np.cos(params[1])))

    initial = np.array([0.8, 0.8], dtype=float)
    euclidean = parameter_shift_gradient_descent(
        objective,
        initial,
        learning_rate=0.4,
        max_steps=8,
        gradient_tolerance=1e-12,
    )
    natural = parameter_shift_natural_gradient_descent(
        objective,
        initial,
        metric_tensor=np.diag(np.array([1.0, 0.05], dtype=float)),
        learning_rate=0.4,
        max_steps=8,
        gradient_tolerance=1e-12,
        natural_gradient_tolerance=1e-12,
    )

    assert natural.best_value < euclidean.best_value
    assert natural.accepted_steps == 8
    assert natural.rejected_steps == 0
    assert natural.steps[-1].metric_condition_number > 1.0
    assert natural.to_dict()["metric_source"] == "array"


def test_solve_natural_gradient_direction_rejects_bad_metric_boundaries() -> None:
    gradient = np.array([1.0, -0.5], dtype=float)

    with pytest.raises(ValueError, match="symmetric"):
        solve_natural_gradient_direction(
            gradient,
            np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float),
        )

    with pytest.raises(ValueError, match="shape"):
        solve_natural_gradient_direction(gradient, np.eye(3, dtype=float))

    with pytest.raises(ValueError, match="positive definite"):
        solve_natural_gradient_direction(
            gradient,
            -np.eye(2, dtype=float),
            damping=0.0,
        )


def test_parameter_shift_natural_gradient_fails_closed_for_hardware() -> None:
    def objective(params: np.ndarray) -> float:
        return float(1.0 - np.cos(params[0]))

    with pytest.raises(ValueError, match="hardware gradient execution requires"):
        parameter_shift_natural_gradient_descent(
            objective,
            np.array([0.4], dtype=float),
            backend="hardware",
            max_steps=2,
        )


def test_validate_natural_gradient_training_rejects_unbound_target_tolerance() -> None:
    def objective(params: np.ndarray) -> float:
        return float(1.0 - np.cos(params[0]))

    result = parameter_shift_natural_gradient_descent(
        objective,
        np.array([0.4], dtype=float),
        max_steps=2,
    )

    with pytest.raises(ValueError, match="target_value_tolerance"):
        validate_natural_gradient_training(result, target_value_tolerance=1e-6)
