# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Natural Gradient
"""Tests for phase/natural_gradient.py metric-aware training semantics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.phase.natural_gradient as natural_gradient_module
from scpn_quantum_control.differentiable import GradientResult, Parameter, ParameterShiftRule
from scpn_quantum_control.phase import (
    NaturalGradientRegularizationPolicy,
    ParameterShiftNaturalGradientResult,
    PhaseQNodeCircuit,
    parameter_shift_gradient_descent,
    parameter_shift_natural_gradient_descent,
    phase_qnode_natural_gradient_metric,
    solve_natural_gradient_direction,
    validate_natural_gradient_training,
)


def test_parameter_shift_natural_gradient_converges_with_callable_metric() -> None:
    def objective(params: NDArray[np.float64]) -> float:
        return float((1.0 - np.cos(params[0])) + 0.05 * (1.0 - np.cos(params[1])))

    def metric(params: NDArray[np.float64]) -> NDArray[np.float64]:
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
        natural_gradient_tolerance=1e-7,
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
    assert certificate.within_natural_gradient_tolerance
    assert certificate.within_target_value_tolerance
    assert certificate.min_decrease_satisfied
    assert result.value_history[0] == pytest.approx(result.initial_value)
    assert result.to_dict()["final_regularization_reason"] == "damped"
    assert certificate.to_dict()["metric_source"] == "callable"


def test_parameter_shift_natural_gradient_preconditions_slow_phase_axis() -> None:
    def objective(params: NDArray[np.float64]) -> float:
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


def test_parameter_shift_natural_gradient_accepts_phase_qnode_metric_provider() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable="pauli_z",
    )

    def objective(params: NDArray[np.float64]) -> float:
        return float(1.0 - np.cos(params[0]))

    result = parameter_shift_natural_gradient_descent(
        objective,
        np.array([0.8], dtype=float),
        metric_tensor=phase_qnode_natural_gradient_metric(circuit),
        learning_rate=0.1,
        max_steps=30,
        gradient_tolerance=1e-7,
        natural_gradient_tolerance=1e-7,
        value_tolerance=1e-10,
    )

    assert result.converged
    assert result.reason == "value_tolerance"
    assert result.metric_source == "callable"
    assert result.best_value < 1e-10
    assert result.accepted_steps > 0


def test_solve_natural_gradient_direction_rejects_bad_metric_boundaries() -> None:
    gradient = np.array([1.0, -0.5], dtype=float)

    with pytest.raises(ValueError, match="one-dimensional"):
        solve_natural_gradient_direction(np.array([[1.0, 2.0]], dtype=float), np.eye(2))

    with pytest.raises(ValueError, match="finite values"):
        solve_natural_gradient_direction(np.array([np.nan, 1.0], dtype=float), np.eye(2))

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

    with pytest.raises(ValueError, match="finite values"):
        solve_natural_gradient_direction(
            gradient,
            np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=float),
        )

    with pytest.raises(ValueError, match="greater than one"):
        solve_natural_gradient_direction(gradient, np.eye(2), max_condition_number=1.0)

    with pytest.raises(ValueError, match="positive definite"):
        solve_natural_gradient_direction(
            gradient,
            np.zeros((2, 2), dtype=float),
            damping=0.0,
            eigenvalue_floor=0.0,
            max_condition_number=1.0e12,
        )


def test_solve_natural_gradient_direction_records_unshifted_metric_policy() -> None:
    direction = solve_natural_gradient_direction(
        np.array([0.5, -0.25], dtype=float),
        np.eye(2, dtype=float),
        damping=0.0,
        eigenvalue_floor=0.0,
        max_condition_number=1.0e6,
    )

    assert direction.regularization_reason == "none"
    assert direction.diagonal_shift == pytest.approx(0.0)
    assert direction.metric_rank == 2
    assert direction.metric_nullity == 0


def test_solve_natural_gradient_direction_records_degenerate_unregularized_metric() -> None:
    direction = solve_natural_gradient_direction(
        np.array([1.0, 1.0e-12], dtype=float),
        np.diag(np.array([1.0, 1.0e-12], dtype=float)),
        damping=0.0,
        eigenvalue_floor=0.0,
        max_condition_number=1.0e13,
        degeneracy_tolerance=1.0e-10,
    )

    assert direction.regularization_reason == "degenerate_unregularized"
    assert direction.metric_rank == 1
    assert direction.metric_nullity == 1


def test_solve_natural_gradient_direction_fails_closed_on_linalg_backend_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gradient = np.array([1.0], dtype=float)
    metric = np.eye(1, dtype=float)

    def nonfinite_eigenvalues(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        del matrix
        return np.array([np.nan], dtype=float)

    monkeypatch.setattr(np.linalg, "eigvalsh", nonfinite_eigenvalues)
    with pytest.raises(ValueError, match="eigenvalues must be finite"):
        solve_natural_gradient_direction(gradient, metric)

    monkeypatch.undo()

    def nonfinite_condition_number(matrix: NDArray[np.float64]) -> float:
        del matrix
        return float("inf")

    monkeypatch.setattr(np.linalg, "cond", nonfinite_condition_number)
    with pytest.raises(ValueError, match="condition number must be finite"):
        solve_natural_gradient_direction(gradient, metric)

    monkeypatch.undo()

    def excessive_condition_number(matrix: NDArray[np.float64]) -> float:
        del matrix
        return 2.0

    monkeypatch.setattr(np.linalg, "cond", excessive_condition_number)
    with pytest.raises(ValueError, match="exceeds max_condition_number"):
        solve_natural_gradient_direction(gradient, metric, max_condition_number=1.5)

    monkeypatch.undo()

    def singular_solve(
        matrix: NDArray[np.float64],
        vector: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        del matrix, vector
        raise np.linalg.LinAlgError("synthetic singular backend")

    monkeypatch.setattr(np.linalg, "solve", singular_solve)
    with pytest.raises(ValueError, match="must be invertible"):
        solve_natural_gradient_direction(gradient, metric)

    monkeypatch.undo()

    def nonfinite_solve(
        matrix: NDArray[np.float64],
        vector: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        del matrix, vector
        return np.array([np.nan], dtype=float)

    monkeypatch.setattr(np.linalg, "solve", nonfinite_solve)
    with pytest.raises(ValueError, match="direction must contain only finite values"):
        solve_natural_gradient_direction(gradient, metric)

    monkeypatch.undo()

    def negative_energy_solve(
        matrix: NDArray[np.float64],
        vector: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        del matrix, vector
        return np.array([-1.0], dtype=float)

    monkeypatch.setattr(np.linalg, "solve", negative_energy_solve)
    with pytest.raises(ValueError, match="positive descent direction"):
        solve_natural_gradient_direction(gradient, metric)


def test_solve_natural_gradient_direction_regularizes_singular_metric() -> None:
    gradient = np.array([1.0, 0.0], dtype=float)

    direction = solve_natural_gradient_direction(
        gradient,
        np.diag(np.array([0.0, 1.0], dtype=float)),
        damping=0.0,
        eigenvalue_floor=1.0e-3,
        max_condition_number=1.0e6,
    )

    assert isinstance(direction.regularization_policy, NaturalGradientRegularizationPolicy)
    assert direction.metric_rank == 1
    assert direction.metric_nullity == 1
    assert direction.regularization_reason == "eigenvalue_floor"
    assert direction.diagonal_shift == pytest.approx(1.0e-3)
    assert direction.regularized_minimum_eigenvalue >= 1.0e-3
    assert direction.condition_number <= 1.0e6
    assert direction.to_dict()["metric_nullity"] == 1


def test_solve_natural_gradient_direction_caps_ill_conditioned_metric() -> None:
    gradient = np.array([1.0, 1.0], dtype=float)

    direction = solve_natural_gradient_direction(
        gradient,
        np.diag(np.array([1.0, 1.0e-12], dtype=float)),
        damping=0.0,
        max_condition_number=1.0e6,
    )

    assert direction.regularization_reason == "condition_number_limited"
    assert direction.metric_nullity == 1
    assert direction.diagonal_shift > 0.0
    assert direction.condition_number <= 1.0e6 * (1.0 + 1.0e-10)


def test_parameter_shift_natural_gradient_records_degenerate_qfi_policy() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("rz", (0,), 0),),
        observable="pauli_z",
    )
    metric = phase_qnode_natural_gradient_metric(circuit)
    np.testing.assert_allclose(metric(np.array([0.4], dtype=float)), [[0.0]], atol=1e-12)

    result = parameter_shift_natural_gradient_descent(
        lambda params: float(1.0 - np.cos(params[0])),
        np.array([0.8], dtype=float),
        metric_tensor=metric,
        learning_rate=0.05,
        max_steps=2,
        damping=0.0,
        eigenvalue_floor=0.25,
        max_condition_number=1.0e6,
    )

    assert result.metric_source == "callable"
    assert result.eigenvalue_floor == pytest.approx(0.25)
    assert result.final_metric_nullity == 1
    assert result.final_regularization_reason == "eigenvalue_floor"
    assert result.steps
    assert result.steps[0].metric_nullity == 1
    assert result.steps[0].diagonal_shift == pytest.approx(0.25)


def test_parameter_shift_natural_gradient_rejects_invalid_public_inputs() -> None:
    def objective(params: NDArray[np.float64]) -> float:
        return float(1.0 - np.cos(params[0]))

    with pytest.raises(ValueError, match="initial_params"):
        parameter_shift_natural_gradient_descent(objective, np.array([], dtype=float))

    with pytest.raises(ValueError, match="learning_rate"):
        parameter_shift_natural_gradient_descent(
            objective,
            np.array([0.4], dtype=float),
            learning_rate=0.0,
        )

    with pytest.raises(ValueError, match="max_steps"):
        parameter_shift_natural_gradient_descent(
            objective,
            np.array([0.4], dtype=float),
            max_steps=True,
        )

    with pytest.raises(ValueError, match="damping"):
        parameter_shift_natural_gradient_descent(
            objective,
            np.array([0.4], dtype=float),
            damping=float("nan"),
        )

    with pytest.raises(ValueError, match="backtracking_factor"):
        parameter_shift_natural_gradient_descent(
            objective,
            np.array([0.4], dtype=float),
            backtracking_factor=1.0,
        )

    with pytest.raises(ValueError, match="differentiable objective must be finite"):
        parameter_shift_natural_gradient_descent(
            lambda params: float("nan"),
            np.array([0.4], dtype=float),
            max_steps=1,
        )


def test_parameter_shift_natural_gradient_records_multi_term_rule() -> None:
    def objective(params: NDArray[np.float64]) -> float:
        return float(1.0 - np.cos(params[0]))

    result = parameter_shift_natural_gradient_descent(
        objective,
        np.array([0.2], dtype=float),
        rule=ParameterShiftRule(
            shifts=(float(np.pi / 2.0), float(np.pi / 4.0)),
            coefficients=(0.5, 0.25),
        ),
        max_steps=1,
        gradient_tolerance=1.0e-12,
        natural_gradient_tolerance=1.0e-12,
    )

    assert result.shift_terms == 2
    assert result.steps[0].shift_terms == 2


def test_parameter_shift_natural_gradient_fails_closed_on_gradient_contract_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def objective(params: NDArray[np.float64]) -> float:
        return float(1.0 - np.cos(params[0]))

    def wrong_shape_gradient(
        objective_fn: Callable[[NDArray[np.float64]], float],
        params: NDArray[np.float64],
        *,
        parameters: Sequence[Parameter] | None,
        rule: ParameterShiftRule | None,
    ) -> GradientResult:
        del objective_fn, params, parameters, rule
        return GradientResult(
            value=0.0,
            gradient=np.array([1.0, 2.0], dtype=float),
            method="parameter_shift",
            shift=float(np.pi / 2.0),
            coefficient=0.5,
            evaluations=2,
            parameter_names=("a", "b"),
            trainable=(True, True),
        )

    monkeypatch.setattr(natural_gradient_module, "_value_and_grad", wrong_shape_gradient)
    with pytest.raises(ValueError, match="gradient shape"):
        parameter_shift_natural_gradient_descent(
            objective,
            np.array([0.4], dtype=float),
            max_steps=1,
        )

    def nonfinite_gradient(
        objective_fn: Callable[[NDArray[np.float64]], float],
        params: NDArray[np.float64],
        *,
        parameters: Sequence[Parameter] | None,
        rule: ParameterShiftRule | None,
    ) -> GradientResult:
        del objective_fn, params, parameters, rule
        return cast(
            GradientResult,
            SimpleNamespace(
                value=0.0,
                gradient=np.array([np.nan], dtype=float),
                method="parameter_shift",
                evaluations=2,
            ),
        )

    monkeypatch.setattr(natural_gradient_module, "_value_and_grad", nonfinite_gradient)
    with pytest.raises(ValueError, match="gradient must contain"):
        parameter_shift_natural_gradient_descent(
            objective,
            np.array([0.4], dtype=float),
            max_steps=1,
        )

    def nonfinite_value(
        objective_fn: Callable[[NDArray[np.float64]], float],
        params: NDArray[np.float64],
        *,
        parameters: Sequence[Parameter] | None,
        rule: ParameterShiftRule | None,
    ) -> GradientResult:
        del objective_fn, params, parameters, rule
        return cast(
            GradientResult,
            SimpleNamespace(
                value=float("nan"),
                gradient=np.array([0.0], dtype=float),
                method="parameter_shift",
                evaluations=2,
            ),
        )

    monkeypatch.setattr(natural_gradient_module, "_value_and_grad", nonfinite_value)
    with pytest.raises(ValueError, match="objective must return"):
        parameter_shift_natural_gradient_descent(
            objective,
            np.array([0.4], dtype=float),
            max_steps=1,
        )


def test_parameter_shift_natural_gradient_records_line_search_failure() -> None:
    def discontinuous_objective(params: NDArray[np.float64]) -> float:
        value = float(params[0])
        if np.isclose(value, float(np.pi / 2.0)):
            return -1.0
        if np.isclose(value, float(-np.pi / 2.0)):
            return 1.0
        if np.isclose(value, 0.0):
            return 0.0
        return 1.0

    result = parameter_shift_natural_gradient_descent(
        discontinuous_objective,
        np.array([0.0], dtype=float),
        learning_rate=0.5,
        max_steps=3,
        max_backtracks=2,
        damping=1.0e-6,
    )

    assert result.reason == "line_search_failed"
    assert result.rejected_steps == 1
    assert result.steps[-1].accepted is False
    assert result.steps[-1].backtracks == 3


def test_parameter_shift_natural_gradient_stops_on_initial_natural_norm() -> None:
    def objective(params: NDArray[np.float64]) -> float:
        return float(1.0 - np.cos(params[0]))

    result = parameter_shift_natural_gradient_descent(
        objective,
        np.array([0.8], dtype=float),
        metric_tensor=np.array([[1.0e18]], dtype=float),
        max_steps=3,
        gradient_tolerance=1.0e-12,
        natural_gradient_tolerance=1.0e-6,
    )

    assert result.converged
    assert result.reason == "natural_gradient_tolerance"
    assert result.accepted_steps == 0
    assert result.steps == ()


def test_parameter_shift_natural_gradient_converges_after_final_step_by_gradient() -> None:
    def objective(params: NDArray[np.float64]) -> float:
        return float(1.0 - np.cos(params[0]))

    result = parameter_shift_natural_gradient_descent(
        objective,
        np.array([0.1], dtype=float),
        learning_rate=1.0,
        max_steps=1,
        gradient_tolerance=1.0e-3,
        natural_gradient_tolerance=1.0e-12,
    )

    assert result.converged
    assert result.reason == "gradient_tolerance"
    assert result.accepted_steps == 1


def test_parameter_shift_natural_gradient_converges_after_final_step_by_natural_norm() -> None:
    def objective(params: NDArray[np.float64]) -> float:
        return float(1.0 - np.cos(params[0]))

    def metric(params: NDArray[np.float64]) -> NDArray[np.float64]:
        if float(params[0]) < 0.8:
            return np.array([[1.0e18]], dtype=float)
        return np.eye(1, dtype=float)

    result = parameter_shift_natural_gradient_descent(
        objective,
        np.array([0.8], dtype=float),
        metric_tensor=metric,
        learning_rate=0.5,
        max_steps=1,
        gradient_tolerance=1.0e-12,
        natural_gradient_tolerance=1.0e-6,
    )

    assert result.converged
    assert result.reason == "natural_gradient_tolerance"
    assert result.accepted_steps == 1


def test_parameter_shift_natural_gradient_converges_after_final_step_by_value() -> None:
    def objective(params: NDArray[np.float64]) -> float:
        return float(1.0 - np.cos(params[0]))

    result = parameter_shift_natural_gradient_descent(
        objective,
        np.array([0.8], dtype=float),
        learning_rate=0.5,
        max_steps=1,
        gradient_tolerance=1.0e-12,
        natural_gradient_tolerance=1.0e-12,
        value_tolerance=0.1,
    )

    assert result.converged
    assert result.reason == "value_tolerance"
    assert result.accepted_steps == 1


def test_parameter_shift_natural_gradient_fails_closed_for_hardware() -> None:
    def objective(params: NDArray[np.float64]) -> float:
        return float(1.0 - np.cos(params[0]))

    with pytest.raises(ValueError, match="hardware gradient execution requires"):
        parameter_shift_natural_gradient_descent(
            objective,
            np.array([0.4], dtype=float),
            backend="hardware",
            max_steps=2,
        )


def test_validate_natural_gradient_training_rejects_unbound_target_tolerance() -> None:
    def objective(params: NDArray[np.float64]) -> float:
        return float(1.0 - np.cos(params[0]))

    result = parameter_shift_natural_gradient_descent(
        objective,
        np.array([0.4], dtype=float),
        max_steps=2,
    )

    with pytest.raises(ValueError, match="target_value_tolerance"):
        validate_natural_gradient_training(result, target_value_tolerance=1e-6)
