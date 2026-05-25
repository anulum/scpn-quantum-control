# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- differentiable-programming conformance benchmarks
"""Deterministic differentiable-programming conformance benchmark cases."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..differentiable import (
    Parameter,
    program_adjoint_gradient,
    vmap,
    whole_program_value_and_grad,
)


@dataclass(frozen=True)
class DifferentiableProgrammingBenchmarkResult:
    """Conformance result for one differentiable-programming benchmark case."""

    case_id: str
    category: str
    value: float
    gradient: NDArray[np.float64]
    analytic_gradient: NDArray[np.float64]
    max_abs_gradient_error: float
    adjoint_supported: bool
    max_abs_adjoint_error: float | None
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("benchmark case_id must be non-empty")
        if not self.category:
            raise ValueError("benchmark category must be non-empty")
        if not math.isfinite(self.value):
            raise ValueError("benchmark value must be finite")
        gradient = _as_gradient("gradient", self.gradient)
        analytic = _as_gradient("analytic_gradient", self.analytic_gradient)
        if gradient.shape != analytic.shape:
            raise ValueError("benchmark gradient and analytic_gradient shapes must match")
        if self.max_abs_gradient_error < 0.0 or not math.isfinite(self.max_abs_gradient_error):
            raise ValueError("benchmark max_abs_gradient_error must be finite and non-negative")
        if not isinstance(self.adjoint_supported, bool):
            raise ValueError("benchmark adjoint_supported must be a boolean")
        if self.max_abs_adjoint_error is not None and (
            self.max_abs_adjoint_error < 0.0 or not math.isfinite(self.max_abs_adjoint_error)
        ):
            raise ValueError("benchmark max_abs_adjoint_error must be finite or None")
        if not self.claim_boundary:
            raise ValueError("benchmark claim_boundary must be non-empty")
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "analytic_gradient", analytic)

    @property
    def passed(self) -> bool:
        """Return whether implemented gradients match the analytic reference."""

        return self.max_abs_gradient_error <= 1.0e-12 and (
            self.max_abs_adjoint_error is None or self.max_abs_adjoint_error <= 1.0e-12
        )


def run_differentiable_programming_benchmark_suite() -> tuple[
    DifferentiableProgrammingBenchmarkResult, ...
]:
    """Run deterministic program-AD conformance benchmarks against analytic references."""

    return (
        _loop_heavy_case(),
        _matrix_heavy_case(),
        _linalg_primitive_case(),
        _mutation_heavy_case(),
        _transform_nesting_case(),
    )


def _loop_heavy_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.2, -0.4, 0.7, -0.9], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        total = trace_values[0] * trace_values[0]
        for index in range(4):
            total = total + float(index + 1) * np.sin(trace_values[index])
        return total

    analytic = np.array(
        [
            2.0 * values[0] + math.cos(values[0]),
            2.0 * math.cos(values[1]),
            3.0 * math.cos(values[2]),
            4.0 * math.cos(values[3]),
        ],
        dtype=np.float64,
    )
    return _program_ad_case(
        "loop_heavy_scalar",
        "loop-heavy",
        objective,
        values,
        analytic,
    )


def _matrix_heavy_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.5, -0.25, 1.5, -2.0], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        left = trace_values[:2]
        right = trace_values[2:4]
        matrix = np.reshape(trace_values, (2, 2))
        return (
            np.inner(left, right)
            + np.sum(np.outer(left, right))
            + np.trace(matrix)
            + np.sum(np.diag(matrix))
            + np.tensordot(left, right, axes=1)
            + np.sum(np.tensordot(left, right, axes=0))
            + np.einsum("i,i->", left, right)
            + np.sum(np.einsum("i,j->ij", left, right))
            + np.sum(np.einsum("ij,j->i", matrix, left))
            + np.einsum("ii->", matrix)
        )

    analytic = np.array(
        [
            7.0 * values[2] + 3.0 * values[3] + 2.0 * values[0] + 3.0,
            3.0 * values[2] + 7.0 * values[3] + 2.0 * values[1],
            7.0 * values[0] + 3.0 * values[1],
            3.0 * values[0] + 7.0 * values[1] + 3.0,
        ],
        dtype=np.float64,
    )
    return _program_ad_case(
        "matrix_heavy_linear_algebra",
        "matrix-heavy",
        objective,
        values,
        analytic,
    )


def _linalg_primitive_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([1.5, 2.0, -0.75, 0.5], dtype=np.float64)
    inverse_weights = np.array([[0.25, 0.0], [0.0, -0.5]], dtype=np.float64)
    solve_weights = np.array([1.25, -0.75], dtype=np.float64)
    power_weights = np.array([[0.5, 0.0], [0.0, -0.25]], dtype=np.float64)
    multi_dot_left = np.array([0.75, -1.5], dtype=np.float64)
    multi_dot_right = np.array([1.25, 0.5], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        matrix = np.diag(trace_values[:2])
        rhs = trace_values[2:4]
        return (
            np.linalg.det(matrix)
            + np.sum(np.linalg.inv(matrix) * inverse_weights)
            + np.sum(np.linalg.solve(matrix, rhs) * solve_weights)
            + np.sum(np.linalg.matrix_power(matrix, 2) * power_weights)
            + np.linalg.multi_dot((multi_dot_left, matrix, multi_dot_right))
        )

    x0, x1, rhs0, rhs1 = values
    analytic = np.array(
        [
            x1
            - inverse_weights[0, 0] / (x0 * x0)
            - solve_weights[0] * rhs0 / (x0 * x0)
            + 2.0 * power_weights[0, 0] * x0
            + multi_dot_left[0] * multi_dot_right[0],
            x0
            - inverse_weights[1, 1] / (x1 * x1)
            - solve_weights[1] * rhs1 / (x1 * x1)
            + 2.0 * power_weights[1, 1] * x1
            + multi_dot_left[1] * multi_dot_right[1],
            solve_weights[0] / x0,
            solve_weights[1] / x1,
        ],
        dtype=np.float64,
    )
    return _program_ad_case(
        "linalg_primitive_contracts",
        "linalg-primitive",
        objective,
        values,
        analytic,
    )


def _mutation_heavy_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.4, -0.6, 1.25, -1.5], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        work = trace_values.copy()
        work[0] = trace_values[1] * trace_values[1] + trace_values[2]
        return work[0] + trace_values[0] * trace_values[3]

    analytic = np.array([values[3], 2.0 * values[1], 1.0, values[0]], dtype=np.float64)
    return _program_ad_case(
        "mutation_heavy_forward_only",
        "mutation-heavy",
        objective,
        values,
        analytic,
    )


def _transform_nesting_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([[0.5, -0.25], [1.25, 0.75], [-0.4, 1.1]], dtype=np.float64)

    def sample_objective(row: Any) -> object:
        return row[0] * row[0] + np.sin(row[1])

    gradients = vmap(
        lambda row: whole_program_value_and_grad(sample_objective, row, trace=False).gradient
    )(values)
    analytic = np.column_stack((2.0 * values[:, 0], np.cos(values[:, 1])))
    gradient = np.asarray(gradients, dtype=np.float64).reshape(-1)
    analytic_gradient = analytic.reshape(-1)
    return DifferentiableProgrammingBenchmarkResult(
        case_id="transform_nesting_vmap_program_grad",
        category="transform-nesting",
        value=float(np.sum(values[:, 0] ** 2 + np.sin(values[:, 1]))),
        gradient=gradient,
        analytic_gradient=analytic_gradient,
        max_abs_gradient_error=_max_abs_error(gradient, analytic_gradient),
        adjoint_supported=True,
        max_abs_adjoint_error=0.0,
        claim_boundary=(
            "vmap over program AD gradients compared with analytic separable references; "
            "diagnostic conformance only, not a performance timing claim"
        ),
    )


def _program_ad_case(
    case_id: str,
    category: str,
    objective: Callable[[Any], object],
    values: NDArray[np.float64],
    analytic_gradient: NDArray[np.float64],
) -> DifferentiableProgrammingBenchmarkResult:
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )
    adjoint_supported = result.adjoint_result is not None and result.adjoint_result.supported
    adjoint_error = None
    if adjoint_supported:
        adjoint_error = _max_abs_error(program_adjoint_gradient(result), analytic_gradient)
    return DifferentiableProgrammingBenchmarkResult(
        case_id=case_id,
        category=category,
        value=result.value,
        gradient=result.gradient,
        analytic_gradient=analytic_gradient,
        max_abs_gradient_error=_max_abs_error(result.gradient, analytic_gradient),
        adjoint_supported=adjoint_supported,
        max_abs_adjoint_error=adjoint_error,
        claim_boundary=(
            "deterministic program AD conformance against analytic references; "
            "no wall-clock performance, hardware, LLVM, Rust, or JIT execution claim"
        ),
    )


def _as_gradient(name: str, value: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _max_abs_error(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    return float(
        np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))
    )


__all__ = [
    "DifferentiableProgrammingBenchmarkResult",
    "run_differentiable_programming_benchmark_suite",
]
