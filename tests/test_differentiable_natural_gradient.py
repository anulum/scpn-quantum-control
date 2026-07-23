# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable natural gradient tests
# scpn-quantum-control -- natural-gradient helper tests
"""Tests for extracted natural-gradient and line-search helpers."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control as scpn
import scpn_quantum_control.differentiable as differentiable
from scpn_quantum_control.differentiable import (
    ArmijoLineSearchResult,
    GradientResult,
    NaturalGradientResult,
    Parameter,
    ParameterBounds,
    WeightedGradientResult,
    batch_value_and_parameter_shift_grad,
    value_and_finite_difference_grad,
    value_and_parameter_shift_grad,
)
from scpn_quantum_control.differentiable_natural_gradient import (
    armijo_backtracking_line_search,
    natural_gradient,
    weighted_gradient_sum,
)
from tools import differentiable_natural_gradient_quality_gates as natural_quality_gates


def test_natural_gradient_quality_gate_spec_is_exact_and_focused() -> None:
    """The natural-gradient owner gate must mirror strict branch checks."""
    static_gates = dict(natural_quality_gates.build_static_quality_gates("python"))
    cohort = natural_quality_gates.NATURAL_GRADIENT_QUALITY_RATCHET
    assert (
        static_gates["mypy-strict-differentiable-natural-gradient-quality"][-len(cohort) :]
        == cohort
    )
    assert (
        static_gates["ruff D differentiable-natural-gradient quality ratchet"][-len(cohort) :]
        == cohort
    )
    coverage_gates = natural_quality_gates.build_coverage_gates("python")
    assert "--branch" in coverage_gates[0][1]
    assert "--fail-under=100" in coverage_gates[1][1]
    assert "--include=*/differentiable_natural_gradient.py" in coverage_gates[1][1]


def _assert_allclose(
    actual: object,
    expected: object,
    *,
    rtol: float = 1.0e-7,
    atol: float = 0.0,
) -> None:
    """Assert NumPy-close equality while preserving strict test typing."""
    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def _gradient_result(
    gradient: NDArray[np.float64],
    *,
    trainable: tuple[bool, ...] | None = None,
    parameter_names: tuple[str, ...] | None = None,
) -> GradientResult:
    """Build a validated gradient result for natural-gradient unit tests."""
    names = parameter_names or tuple(f"theta_{index}" for index in range(gradient.size))
    mask = trainable or tuple(True for _ in range(gradient.size))
    return GradientResult(
        value=1.0,
        gradient=gradient,
        method="finite_difference_central",
        shift=1.0e-6,
        coefficient=5.0e5,
        evaluations=7,
        parameter_names=names,
        trainable=mask,
    )


def test_facade_and_package_root_reuse_extracted_natural_gradient_helpers() -> None:
    """Facade and package-root exports should point at the extracted helpers."""
    assert differentiable.armijo_backtracking_line_search is armijo_backtracking_line_search
    assert differentiable.natural_gradient is natural_gradient
    assert differentiable.weighted_gradient_sum is weighted_gradient_sum
    assert scpn.armijo_backtracking_line_search is armijo_backtracking_line_search
    assert scpn.natural_gradient is natural_gradient
    assert scpn.weighted_gradient_sum is weighted_gradient_sum


def test_natural_gradient_solves_trainable_metric_system() -> None:
    """Natural gradient should precondition only trainable parameters."""
    gradient = _gradient_result(
        np.array([2.0, 4.0, 0.0]),
        parameter_names=("x", "y", "frozen"),
        trainable=(True, True, False),
    )
    result = natural_gradient(gradient, np.diag([2.0, 4.0, 99.0]))

    assert isinstance(result, NaturalGradientResult)
    _assert_allclose(result.natural_gradient, [1.0, 1.0, 0.0])
    assert result.condition_number == pytest.approx(2.0)


def test_natural_gradient_damping_repairs_semidefinite_metric() -> None:
    """Damping should make semidefinite trainable metrics solvable."""
    gradient = _gradient_result(np.array([2.0]), parameter_names=("x",), trainable=(True,))
    result = natural_gradient(gradient, np.array([[0.0]]), damping=0.5)

    _assert_allclose(result.natural_gradient, [4.0])


def test_natural_gradient_handles_fully_frozen_gradient() -> None:
    """A fully frozen gradient should return a zero natural-gradient direction."""
    gradient = _gradient_result(
        np.array([0.0, 0.0]),
        parameter_names=("x", "y"),
        trainable=(False, False),
    )
    result = natural_gradient(gradient, np.eye(2), damping=0.25)

    assert result.condition_number == pytest.approx(1.0)
    _assert_allclose(result.natural_gradient, [0.0, 0.0])


def test_natural_gradient_rejects_invalid_metric() -> None:
    """Natural-gradient metrics must be finite, symmetric, and well conditioned."""
    gradient = _gradient_result(np.array([1.0, 2.0]), parameter_names=("x", "y"))

    with pytest.raises(ValueError, match="shape"):
        natural_gradient(gradient, np.eye(3))
    with pytest.raises(ValueError, match="finite"):
        natural_gradient(gradient, np.array([[1.0, 0.0], [0.0, np.inf]]))
    with pytest.raises(ValueError, match="symmetric"):
        natural_gradient(gradient, np.array([[1.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="damping"):
        natural_gradient(gradient, np.eye(2), damping=-1.0)
    with pytest.raises(ValueError, match="positive definite"):
        natural_gradient(gradient, np.diag([1.0, 0.0]))
    with pytest.raises(ValueError, match="ill-conditioned"):
        natural_gradient(gradient, np.diag([1.0, 1.0e-8]), rcond=1.0e-6)
    with pytest.raises(ValueError, match="rcond"):
        natural_gradient(gradient, np.eye(2), rcond=0.0)


def test_armijo_backtracking_line_search_accepts_sufficient_decrease() -> None:
    """Armijo search should accept a descent step with explicit provenance."""
    gradient = value_and_finite_difference_grad(lambda values: float(values[0] ** 2), [2.0])
    result = armijo_backtracking_line_search(
        lambda values: float(values[0] ** 2),
        [2.0],
        gradient,
        -gradient.gradient,
        initial_step=1.0,
        contraction=0.5,
    )

    assert isinstance(result, ArmijoLineSearchResult)
    assert result.accepted
    assert result.reason == "accepted"
    assert result.step_size == pytest.approx(0.5)
    assert result.value < gradient.value
    assert result.value_history[0] == pytest.approx(4.0)
    _assert_allclose(result.values, [0.0], atol=1.0e-10)


def test_armijo_backtracking_line_search_respects_bounds_and_frozen_parameters() -> None:
    """Line search should project bounds and remove frozen direction components."""
    gradient = value_and_finite_difference_grad(
        lambda values: values[0] ** 2 + values[1] ** 2,
        [2.0, 10.0],
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
    )
    result = armijo_backtracking_line_search(
        lambda values: values[0] ** 2 + values[1] ** 2,
        [2.0, 10.0],
        gradient,
        [-10.0, -999.0],
        bounds=[ParameterBounds(lower=-1.0, upper=1.0), ParameterBounds()],
    )

    assert result.accepted
    _assert_allclose(result.direction, [-10.0, 0.0])
    _assert_allclose(result.values, [-1.0, 10.0])


def test_armijo_backtracking_line_search_rejects_invalid_controls() -> None:
    """Line-search controls and directions must fail closed."""
    gradient = value_and_finite_difference_grad(lambda values: float(values[0] ** 2), [2.0])
    invalid_gradient = cast(GradientResult, gradient.gradient)
    with pytest.raises(ValueError, match="GradientResult"):
        armijo_backtracking_line_search(
            lambda values: float(values[0] ** 2),
            [2.0],
            invalid_gradient,
            [-1.0],
        )
    with pytest.raises(ValueError, match="values length"):
        armijo_backtracking_line_search(
            lambda values: float(values[0] ** 2), [2.0, 3.0], gradient, [-1.0, 0.0]
        )
    with pytest.raises(ValueError, match="direction length"):
        armijo_backtracking_line_search(
            lambda values: float(values[0] ** 2), [2.0], gradient, [-1.0, 0.0]
        )
    with pytest.raises(ValueError, match="initial_step"):
        armijo_backtracking_line_search(
            lambda values: float(values[0] ** 2), [2.0], gradient, [-1.0], initial_step=0.0
        )
    with pytest.raises(ValueError, match="contraction"):
        armijo_backtracking_line_search(
            lambda values: float(values[0] ** 2), [2.0], gradient, [-1.0], contraction=1.0
        )
    with pytest.raises(ValueError, match="sufficient_decrease"):
        armijo_backtracking_line_search(
            lambda values: float(values[0] ** 2), [2.0], gradient, [-1.0], sufficient_decrease=0.0
        )
    with pytest.raises(ValueError, match="max_steps"):
        armijo_backtracking_line_search(
            lambda values: float(values[0] ** 2), [2.0], gradient, [-1.0], max_steps=0
        )


def test_armijo_backtracking_line_search_rejects_non_descent_and_max_steps() -> None:
    """Line search should report both non-descent and exhausted-trial rejections."""
    gradient = value_and_finite_difference_grad(lambda values: float(values[0] ** 2), [2.0])
    non_descent = armijo_backtracking_line_search(
        lambda values: float(values[0] ** 2),
        [2.0],
        gradient,
        gradient.gradient,
    )
    exhausted = armijo_backtracking_line_search(
        lambda values: float(values[0] ** 2),
        [2.0],
        gradient,
        -gradient.gradient,
        initial_step=1.0,
        max_steps=1,
    )

    assert non_descent.accepted is False
    assert non_descent.reason == "non_descent_direction"
    assert non_descent.step_size == pytest.approx(0.0)
    assert non_descent.evaluations == 1
    _assert_allclose(non_descent.values, [2.0])
    assert exhausted.accepted is False
    assert exhausted.reason == "max_steps"
    assert exhausted.evaluations == 2
    _assert_allclose(exhausted.values, [2.0])


def test_weighted_gradient_sum_preserves_component_provenance() -> None:
    """Weighted multi-objective aggregation should keep component metadata."""
    components = batch_value_and_parameter_shift_grad(
        [
            lambda values: math.sin(values[0]),
            lambda values: math.cos(values[0]),
        ],
        [0.25],
        parameters=[Parameter("theta")],
    )
    result = weighted_gradient_sum(components, np.array([0.75, 0.25]))

    assert isinstance(result, WeightedGradientResult)
    assert result.components == components
    assert result.evaluations == sum(component.evaluations for component in components)
    assert result.parameter_names == ("theta",)
    assert result.value == pytest.approx(0.75 * math.sin(0.25) + 0.25 * math.cos(0.25))
    _assert_allclose(
        result.gradient,
        [0.75 * math.cos(0.25) - 0.25 * math.sin(0.25)],
    )


def test_weighted_gradient_sum_rejects_invalid_components() -> None:
    """Weighted aggregation must fail closed on malformed component contracts."""
    first = value_and_parameter_shift_grad(
        lambda values: math.sin(values[0]),
        [0.1],
        parameters=[Parameter("theta")],
    )
    second = value_and_parameter_shift_grad(
        lambda values: math.sin(values[0]),
        [0.1],
        parameters=[Parameter("phi")],
    )
    third = GradientResult(
        value=1.0,
        gradient=np.array([1.0, 2.0]),
        method="finite_difference_central",
        shift=1.0e-6,
        coefficient=5.0e5,
        evaluations=3,
        parameter_names=("theta", "phi"),
        trainable=(True, True),
    )
    frozen_mismatch = GradientResult(
        value=1.0,
        gradient=np.array([0.0]),
        method="finite_difference_central",
        shift=1.0e-6,
        coefficient=5.0e5,
        evaluations=3,
        parameter_names=("theta",),
        trainable=(False,),
    )
    invalid_component = cast(GradientResult, np.array([1.0]))

    with pytest.raises(ValueError, match="at least one"):
        weighted_gradient_sum([], np.array([]))
    with pytest.raises(ValueError, match="GradientResult"):
        weighted_gradient_sum([invalid_component], np.array([1.0]))
    with pytest.raises(ValueError, match="weights length"):
        weighted_gradient_sum([first], np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        weighted_gradient_sum([first], np.array([np.inf]))
    with pytest.raises(ValueError, match="matching shapes"):
        weighted_gradient_sum([first, third], np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="parameter_names"):
        weighted_gradient_sum([first, second], np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="trainable"):
        weighted_gradient_sum([first, frozen_mismatch], np.array([0.5, 0.5]))
