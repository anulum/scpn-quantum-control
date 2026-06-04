# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Parameter-Shift Gradient Descent
"""Tests for phase/gradient_descent.py parameter-shift training semantics."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    ParameterShiftTrainingResult,
    multi_frequency_parameter_shift_rule,
    parameter_shift_gradient_descent,
    validate_parameter_shift_training,
)


def test_parameter_shift_gradient_descent_converges_on_rotation_cost() -> None:
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
    def objective(params: np.ndarray) -> float:
        return float(1.0 - np.cos(params[0]))

    with pytest.raises(ValueError, match="hardware gradient execution requires"):
        parameter_shift_gradient_descent(
            objective,
            np.array([0.4], dtype=float),
            backend="hardware",
            max_steps=2,
        )


def test_parameter_shift_gradient_descent_rejects_invalid_training_controls() -> None:
    def objective(params: np.ndarray) -> float:
        return float(1.0 - np.cos(params[0]))

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
    def objective(params: np.ndarray) -> float:
        if abs(params[0] - 0.4) < 1e-12:
            return float("nan")
        return float(1.0 - np.cos(params[0]))

    with pytest.raises(ValueError, match="objective"):
        parameter_shift_gradient_descent(
            objective,
            np.array([0.4], dtype=float),
            max_steps=2,
        )
