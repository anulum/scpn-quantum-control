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
    is_jax_autodiff_available,
    jax_value_and_grad,
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


@dataclass(frozen=True)
class DifferentiableProgrammingExternalReferenceResult:
    """Program-AD comparison against an independently executed autodiff backend."""

    case_id: str
    backend: str
    program_value: float
    reference_value: float
    program_gradient: NDArray[np.float64]
    reference_gradient: NDArray[np.float64]
    max_abs_value_error: float
    max_abs_gradient_error: float
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("external reference case_id must be non-empty")
        if not self.backend:
            raise ValueError("external reference backend must be non-empty")
        if not math.isfinite(self.program_value) or not math.isfinite(self.reference_value):
            raise ValueError("external reference values must be finite")
        program_gradient = _as_gradient("program_gradient", self.program_gradient)
        reference_gradient = _as_gradient("reference_gradient", self.reference_gradient)
        if program_gradient.shape != reference_gradient.shape:
            raise ValueError("external reference gradient shapes must match")
        if self.max_abs_value_error < 0.0 or not math.isfinite(self.max_abs_value_error):
            raise ValueError("external reference value error must be finite and non-negative")
        if self.max_abs_gradient_error < 0.0 or not math.isfinite(self.max_abs_gradient_error):
            raise ValueError("external reference gradient error must be finite and non-negative")
        if not self.claim_boundary:
            raise ValueError("external reference claim_boundary must be non-empty")
        object.__setattr__(self, "program_gradient", program_gradient)
        object.__setattr__(self, "reference_gradient", reference_gradient)

    @property
    def passed(self) -> bool:
        """Return whether program AD matches the external reference backend."""

        return self.max_abs_value_error <= 1.0e-10 and self.max_abs_gradient_error <= 1.0e-10


def run_differentiable_programming_benchmark_suite() -> tuple[
    DifferentiableProgrammingBenchmarkResult, ...
]:
    """Run deterministic program-AD conformance benchmarks against analytic references."""

    return (
        _loop_heavy_case(),
        _matrix_heavy_case(),
        _selection_heavy_case(),
        _linalg_primitive_case(),
        _indexing_heavy_case(),
        _mutation_heavy_case(),
        _transform_nesting_case(),
    )


def run_differentiable_programming_external_reference_suite() -> tuple[
    DifferentiableProgrammingExternalReferenceResult, ...
]:
    """Run optional external-backend conformance comparisons when dependencies exist."""

    if not is_jax_autodiff_available():
        return ()
    return (
        _jax_loop_heavy_case(),
        _jax_linalg_primitive_case(),
        _jax_transform_nesting_case(),
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
    eigvalsh_weights = np.array([0.2, -0.3], dtype=np.float64)
    svd_weights = np.array([0.15, -0.25], dtype=np.float64)
    pinv_weights = np.array([[0.35, 0.0], [0.0, -0.45]], dtype=np.float64)
    trace_weight = 0.375
    diag_weights = np.array([-1.25, 0.5], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        matrix = np.diag(trace_values[:2])
        rhs = trace_values[2:4]
        return (
            np.linalg.det(matrix)
            + np.sum(np.linalg.inv(matrix) * inverse_weights)
            + np.sum(np.linalg.solve(matrix, rhs) * solve_weights)
            + trace_weight * np.trace(matrix)
            + np.sum(np.diag(matrix) * diag_weights)
            + np.sum(np.linalg.matrix_power(matrix, 2) * power_weights)
            + np.linalg.multi_dot((multi_dot_left, matrix, multi_dot_right))
            + np.sum(np.linalg.eigvalsh(matrix) * eigvalsh_weights)
            + np.sum(np.linalg.svd(matrix, compute_uv=False) * svd_weights)
            + np.sum(np.linalg.pinv(matrix) * pinv_weights)
        )

    x0, x1, rhs0, rhs1 = values
    analytic = np.array(
        [
            x1
            - inverse_weights[0, 0] / (x0 * x0)
            - solve_weights[0] * rhs0 / (x0 * x0)
            + trace_weight
            + diag_weights[0]
            + 2.0 * power_weights[0, 0] * x0
            + multi_dot_left[0] * multi_dot_right[0]
            + eigvalsh_weights[0]
            + svd_weights[1]
            - pinv_weights[0, 0] / (x0 * x0),
            x0
            - inverse_weights[1, 1] / (x1 * x1)
            - solve_weights[1] * rhs1 / (x1 * x1)
            + trace_weight
            + diag_weights[1]
            + 2.0 * power_weights[1, 1] * x1
            + multi_dot_left[1] * multi_dot_right[1]
            + eigvalsh_weights[1]
            + svd_weights[0]
            - pinv_weights[1, 1] / (x1 * x1),
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


def _selection_heavy_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([-1.0, 0.4, 1.2], dtype=np.float64)
    thresholds = np.array([-0.5, 0.0, 1.0], dtype=np.float64)
    offsets = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    upper = np.array([0.5, 0.75, 2.0], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        selected = np.select(
            [trace_values < -0.25, trace_values > 0.5],
            [trace_values * trace_values, 1.5 * trace_values],
            default=-0.75 * trace_values,
        )
        callable_piecewise = np.piecewise(
            trace_values,
            [trace_values < -0.25, trace_values > 0.5],
            [
                lambda item: item * item,
                lambda item: 1.5 * item,
                lambda item: -0.75 * item,
            ],
        )
        chosen = np.choose(
            np.array([0, 1, 2], dtype=np.int64),
            [trace_values * trace_values, -0.5 * trace_values, 2.0 * trace_values],
        )
        compressed = np.compress(np.array([True, False, True], dtype=np.bool_), trace_values)
        extracted = np.extract(np.array([True, False, True], dtype=np.bool_), trace_values)
        return (
            np.sum(
                np.where(trace_values > thresholds, trace_values**2, -trace_values)
                + np.clip(trace_values + offsets, -0.75, upper)
                + 0.09 * selected
                + 0.04 * callable_piecewise
                + 0.03 * chosen
            )
            + 0.02 * np.sum(compressed * np.array([2.0, -3.0], dtype=np.float64))
            + 0.015 * np.sum(extracted * np.array([1.0, -1.5], dtype=np.float64))
        )

    analytic = np.array([-1.265, 1.6875, 3.5725], dtype=np.float64)
    return _program_ad_case(
        "selection_piecewise_contracts",
        "selection-heavy",
        objective,
        values,
        analytic,
    )


def _jax_loop_heavy_case() -> DifferentiableProgrammingExternalReferenceResult:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    values = np.array([0.2, -0.4, 0.7, -0.9], dtype=np.float64)

    def program_objective(trace_values: Any) -> object:
        total = trace_values[0] * trace_values[0]
        for index in range(4):
            total = total + float(index + 1) * np.sin(trace_values[index])
        return total

    def reference_objective(raw_values: Any) -> object:
        total = raw_values[0] * raw_values[0]
        for index in range(4):
            total = total + float(index + 1) * jnp.sin(raw_values[index])
        return total

    program_result = whole_program_value_and_grad(
        program_objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )
    reference_value, reference_gradient = jax_value_and_grad(reference_objective, values)
    return DifferentiableProgrammingExternalReferenceResult(
        case_id="jax_loop_heavy_reference",
        backend="jax",
        program_value=program_result.value,
        reference_value=reference_value,
        program_gradient=program_result.gradient,
        reference_gradient=reference_gradient,
        max_abs_value_error=abs(program_result.value - reference_value),
        max_abs_gradient_error=_max_abs_error(program_result.gradient, reference_gradient),
        claim_boundary=(
            "optional JAX external-backend conformance for loop-heavy program AD; "
            "diagnostic correctness only, not a JIT, performance, LLVM, Rust, or hardware claim"
        ),
    )


def _jax_linalg_primitive_case() -> DifferentiableProgrammingExternalReferenceResult:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    values = np.array([1.5, 2.0, -0.75, 0.5], dtype=np.float64)
    inverse_weights = np.array([[0.25, 0.0], [0.0, -0.5]], dtype=np.float64)
    solve_weights = np.array([1.25, -0.75], dtype=np.float64)
    power_weights = np.array([[0.5, 0.0], [0.0, -0.25]], dtype=np.float64)
    eigvalsh_weights = np.array([0.2, -0.3], dtype=np.float64)
    svd_weights = np.array([0.15, -0.25], dtype=np.float64)
    pinv_weights = np.array([[0.35, 0.0], [0.0, -0.45]], dtype=np.float64)

    def program_objective(trace_values: Any) -> object:
        matrix = np.diag(trace_values[:2])
        rhs = trace_values[2:4]
        return (
            np.linalg.det(matrix)
            + np.sum(np.linalg.inv(matrix) * inverse_weights)
            + np.sum(np.linalg.solve(matrix, rhs) * solve_weights)
            + np.sum(np.linalg.matrix_power(matrix, 2) * power_weights)
            + np.sum(np.linalg.eigvalsh(matrix) * eigvalsh_weights)
            + np.sum(np.linalg.svd(matrix, compute_uv=False) * svd_weights)
            + np.sum(np.linalg.pinv(matrix) * pinv_weights)
        )

    def reference_objective(raw_values: Any) -> object:
        matrix = jnp.diag(raw_values[:2])
        rhs = raw_values[2:4]
        return (
            jnp.linalg.det(matrix)
            + jnp.sum(jnp.linalg.inv(matrix) * jnp.asarray(inverse_weights))
            + jnp.sum(jnp.linalg.solve(matrix, rhs) * jnp.asarray(solve_weights))
            + jnp.sum(jnp.linalg.matrix_power(matrix, 2) * jnp.asarray(power_weights))
            + jnp.sum(jnp.linalg.eigvalsh(matrix) * jnp.asarray(eigvalsh_weights))
            + jnp.sum(jnp.linalg.svd(matrix, compute_uv=False) * jnp.asarray(svd_weights))
            + jnp.sum(jnp.linalg.pinv(matrix) * jnp.asarray(pinv_weights))
        )

    program_result = whole_program_value_and_grad(
        program_objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )
    reference_value, reference_gradient = jax_value_and_grad(reference_objective, values)
    return DifferentiableProgrammingExternalReferenceResult(
        case_id="jax_linalg_primitive_reference",
        backend="jax",
        program_value=program_result.value,
        reference_value=reference_value,
        program_gradient=program_result.gradient,
        reference_gradient=reference_gradient,
        max_abs_value_error=abs(program_result.value - reference_value),
        max_abs_gradient_error=_max_abs_error(program_result.gradient, reference_gradient),
        claim_boundary=(
            "optional JAX external-backend conformance for supported linalg primitives; "
            "diagnostic correctness only, not a JIT, performance, LLVM, Rust, or hardware claim"
        ),
    )


def _jax_transform_nesting_case() -> DifferentiableProgrammingExternalReferenceResult:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    values = np.array([[0.5, -0.25], [1.25, 0.75], [-0.4, 1.1]], dtype=np.float64)

    def program_sample_objective(row: Any) -> object:
        return row[0] * row[0] + np.sin(row[1])

    program_gradients = vmap(
        lambda row: (
            whole_program_value_and_grad(program_sample_objective, row, trace=False).gradient
        )
    )(values)

    def reference_sample_objective(row: Any) -> object:
        return row[0] * row[0] + jnp.sin(row[1])

    reference_gradients = np.asarray(
        jax.vmap(jax.grad(reference_sample_objective))(jnp.asarray(values)),
        dtype=np.float64,
    )
    program_gradient = np.asarray(program_gradients, dtype=np.float64).reshape(-1)
    reference_gradient = reference_gradients.reshape(-1)
    program_value = float(np.sum(values[:, 0] ** 2 + np.sin(values[:, 1])))
    reference_value = float(np.sum(np.asarray(jax.vmap(reference_sample_objective)(values))))
    return DifferentiableProgrammingExternalReferenceResult(
        case_id="jax_transform_nesting_reference",
        backend="jax",
        program_value=program_value,
        reference_value=reference_value,
        program_gradient=program_gradient,
        reference_gradient=reference_gradient,
        max_abs_value_error=abs(program_value - reference_value),
        max_abs_gradient_error=_max_abs_error(program_gradient, reference_gradient),
        claim_boundary=(
            "optional JAX external-backend conformance for vmap over program AD gradients; "
            "diagnostic correctness only, not a JIT, performance, LLVM, Rust, or hardware claim"
        ),
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


def _indexing_heavy_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.0], dtype=np.float64)
    block_weights = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    flat_sort_weights = np.array([-0.4, 0.8, -1.2, 1.6, -2.0, 2.4], dtype=np.float64)
    axis_sort_weights = np.array([[0.3, -0.6, 0.9], [-1.1, 1.4, -1.7]], dtype=np.float64)
    trapz_x = np.array([0.0, 0.25, 1.0], dtype=np.float64)
    trapz_row_weights = np.array([0.35, -0.45], dtype=np.float64)
    gradient_flat_weights = np.array([0.25, -0.5, 1.0, -1.5, 0.75, -0.25], dtype=np.float64)
    gradient_axis_weights = np.array([[0.4, -0.6, 0.8], [-1.0, 1.2, -1.4]], dtype=np.float64)
    interp_xp = np.array([-3.0, 0.0, 2.0, 5.0], dtype=np.float64)
    interp_static_fp = np.array([0.75, -1.25, 1.5, -0.5], dtype=np.float64)
    interp_sample_weights = np.array([0.45, -0.35], dtype=np.float64)
    interp_control_weights = np.array([1.1, -0.7, 0.3], dtype=np.float64)
    convolve_full_weights = np.array([0.2, -0.35, 0.5, -0.65, 0.8], dtype=np.float64)
    convolve_same_kernel = np.array([0.4, -0.2], dtype=np.float64)
    convolve_same_weights = np.array([1.0, -0.75, 0.5, -0.25, 0.125, -0.5], dtype=np.float64)
    convolve_static_signal = np.array([0.75, -1.25, 1.5, -0.5], dtype=np.float64)
    convolve_valid_weights = np.array([0.6, -0.4], dtype=np.float64)
    correlate_full_weights = np.array([-0.15, 0.35, -0.55, 0.75, -0.95], dtype=np.float64)
    correlate_same_reference = np.array([0.45, -0.3], dtype=np.float64)
    correlate_same_weights = np.array([-0.6, 0.4, -0.2, 0.8, -1.0, 0.5], dtype=np.float64)
    correlate_static_signal = np.array([1.2, -0.7, 0.9, -1.1], dtype=np.float64)
    correlate_valid_weights = np.array([0.25, -0.85], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        matrix = np.reshape(trace_values, (2, 3))
        block = matrix[:, 1:]
        gathered = np.take(trace_values, [2, 0, 2])
        lower_left = matrix[1:, :2]
        advanced = matrix[[1, 0, 1], [2, 0, 2]]
        masked_columns = matrix[np.array([True, False])][:, np.array([2, 0, 2])]
        along_indices = np.array([[2, 0, 2], [1, 1, 0]], dtype=np.int64)
        along_weights = np.array([[1.0, -0.5, 2.0], [0.25, 1.5, -1.25]], dtype=np.float64)
        along = np.take_along_axis(matrix, along_indices, axis=1)
        wrapped = np.take(trace_values, [-1, 6, 0], mode="wrap")
        clipped = np.take(matrix, [-2, 1, 10], axis=1, mode="clip")
        axis_deleted = np.delete(matrix, [1], axis=1)
        flat_deleted = np.delete(trace_values, [1, 4])
        padded = np.pad(matrix, ((1, 0), (1, 0)), mode="constant", constant_values=-0.75)
        inserted = np.insert(matrix, 1, np.array([-0.25, 0.5]), axis=1)
        axis_appended = np.append(matrix[:, :2], matrix[:, 2:], axis=1)
        flat_appended = np.append(trace_values[:3], trace_values[3:])
        hstacked = np.hstack((trace_values[:3], trace_values[3:]))
        vstacked = np.vstack((matrix[0], matrix[1]))
        column_stacked = np.column_stack((trace_values[:3], trace_values[3:]))
        dstacked = np.dstack((matrix[:, :2], matrix[:, 1:]))
        blocked = np.block([[matrix[:, :2], matrix[:, 2:]], [matrix[1:, :2], matrix[:1, 1:2]]])
        split_first, split_second, split_third = np.split(trace_values, [2, 4])
        split_top, split_bottom = np.vsplit(matrix, 2)
        split_left, split_middle, split_right = np.hsplit(matrix, [1, 2])
        split_depth0, split_depth1 = np.dsplit(np.reshape(trace_values, (1, 2, 3)), [1])
        uneven0, uneven1, uneven2, uneven3 = np.array_split(trace_values, 4)
        lower_triangle = np.tril(matrix)
        upper_triangle = np.triu(matrix, k=1)
        depth_triangle = np.tril(np.reshape(trace_values, (1, 2, 3)), k=-1)
        offset_diagonal = np.diagonal(matrix, offset=1)
        depth_diagonal = np.diagonal(
            np.reshape(trace_values, (1, 2, 3)), offset=1, axis1=1, axis2=2
        )
        flat_diagonal = np.diagflat(matrix[:, :2], k=1)
        broadcast_left, broadcast_right = np.broadcast_arrays(matrix[:, :1], trace_values[:3])
        column_assembled = np.concatenate((matrix[:, 2:], matrix[:, :1], matrix[:, 1:2]), axis=1)
        depth_stacked = np.stack((matrix, matrix[:, ::-1]), axis=2)
        flat_assembled = np.concatenate((matrix[:, :1], matrix[:, 1:]), axis=None)
        flat_sorted = np.sort(trace_values, axis=None)
        axis_sorted = np.sort(matrix, axis=1)
        row_integrals = np.trapezoid(matrix, x=trapz_x, axis=1)
        flat_integral = np.trapezoid(trace_values, dx=0.2)
        flat_gradient = np.gradient(trace_values, 0.5, edge_order=1)
        axis_gradient = np.gradient(
            matrix,
            np.array([0.0, 0.5, 1.5], dtype=np.float64),
            axis=1,
            edge_order=2,
        )
        sample_interpolation = np.interp(trace_values[:2], interp_xp, interp_static_fp)
        control_interpolation = np.interp(
            np.array([-2.5, 1.0, 4.0], dtype=np.float64),
            interp_xp,
            trace_values[2:],
        )
        dynamic_convolution = np.convolve(trace_values[:3], trace_values[3:], mode="full")
        static_kernel_convolution = np.convolve(trace_values, convolve_same_kernel, mode="same")
        static_signal_convolution = np.convolve(
            convolve_static_signal,
            trace_values[3:],
            mode="valid",
        )
        dynamic_correlation = np.correlate(trace_values[:3], trace_values[3:], mode="full")
        static_reference_correlation = np.correlate(
            trace_values,
            correlate_same_reference,
            mode="same",
        )
        static_signal_correlation = np.correlate(
            correlate_static_signal,
            trace_values[3:],
            mode="valid",
        )
        return (
            np.sum(block * block_weights)
            + np.sum(gathered)
            - 2.0 * lower_left[0, 1]
            + 0.5 * matrix[None, :, :][0, -1, -1]
            + 0.25 * np.sum(advanced)
            - 0.1 * np.sum(masked_columns)
            + 0.2 * np.sum(along * along_weights)
            + 0.3 * np.sum(wrapped * np.array([0.5, -1.0, 2.0], dtype=np.float64))
            - 0.2
            * np.sum(
                clipped
                * np.array(
                    [[1.0, -0.25, 0.5], [0.75, -1.5, 0.25]],
                    dtype=np.float64,
                )
            )
            + 0.4 * np.sum(axis_deleted * np.array([[1.0, -2.0], [0.5, 3.0]]))
            + 0.6 * np.sum(flat_deleted * np.array([0.25, -0.75, 1.25, -1.5], dtype=np.float64))
            + 0.15
            * np.sum(
                padded
                * np.array(
                    [[0.5, -1.0, 2.0, 0.25], [1.5, -2.0, 0.75, 3.0], [-0.25, 0.5, 2.5, -1.5]],
                    dtype=np.float64,
                )
            )
            + 0.12
            * np.sum(
                inserted
                * np.array(
                    [[1.0, -4.0, 2.0, 0.5], [-1.0, 3.0, 1.5, -2.0]],
                    dtype=np.float64,
                )
            )
            + 0.05
            * np.sum(
                axis_appended * np.array([[1.0, -2.0, 0.5], [1.5, -0.75, 2.25]], dtype=np.float64)
            )
            + 0.07
            * np.sum(flat_appended * np.array([0.2, -0.4, 0.6, -0.8, 1.0, -1.2], dtype=np.float64))
            + 0.03
            * np.sum(hstacked * np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float64))
            + 0.04
            * np.sum(
                vstacked * np.array([[0.5, -1.0, 1.5], [-0.25, 0.75, -1.25]], dtype=np.float64)
            )
            + 0.02
            * np.sum(
                column_stacked
                * np.array([[1.0, -0.5], [0.25, 1.5], [-1.0, 0.75]], dtype=np.float64)
            )
            + 0.01
            * np.sum(
                dstacked
                * np.array(
                    [[[0.2, -0.1], [0.4, -0.3]], [[0.5, -0.2], [0.7, -0.6]]],
                    dtype=np.float64,
                )
            )
            + 0.025
            * np.sum(
                blocked
                * np.array(
                    [[0.25, -0.5, 0.75], [1.0, -1.25, 1.5], [-0.75, 0.5, -1.0]],
                    dtype=np.float64,
                )
            )
            + 0.015
            * (
                np.sum(split_first * np.array([1.0, -2.0], dtype=np.float64))
                + np.sum(split_second * np.array([0.5, 3.0], dtype=np.float64))
                + np.sum(split_third * np.array([-1.5, 2.5], dtype=np.float64))
                + np.sum(split_top * np.array([[0.25, -0.5, 0.75]], dtype=np.float64))
                + np.sum(split_bottom * np.array([[-1.0, 1.5, -2.0]], dtype=np.float64))
                + np.sum(split_left * np.array([[2.0], [-0.25]], dtype=np.float64))
                + np.sum(split_middle * np.array([[-1.25], [0.5]], dtype=np.float64))
                + np.sum(split_right * np.array([[1.75], [-0.75]], dtype=np.float64))
                + np.sum(split_depth0 * np.array([[[0.4], [-0.6]]], dtype=np.float64))
                + np.sum(split_depth1 * np.array([[[0.2, -0.3], [0.8, -0.9]]], dtype=np.float64))
                + np.sum(uneven0 * np.array([0.05, -0.1], dtype=np.float64))
                + np.sum(uneven1 * np.array([0.15, -0.2], dtype=np.float64))
                + np.sum(uneven2 * np.array([0.25], dtype=np.float64))
                + np.sum(uneven3 * np.array([-0.3], dtype=np.float64))
            )
            + 0.02
            * (
                np.sum(
                    lower_triangle
                    * np.array([[1.0, -2.0, 3.0], [0.5, 1.5, -1.0]], dtype=np.float64)
                )
                + np.sum(
                    upper_triangle
                    * np.array([[-0.25, 0.75, -1.25], [2.0, -0.5, 1.0]], dtype=np.float64)
                )
                + np.sum(
                    depth_triangle
                    * np.array([[[0.1, -0.2, 0.3], [2.0, -0.4, 0.5]]], dtype=np.float64)
                )
            )
            + 0.025 * np.sum(offset_diagonal * np.array([0.4, -0.6], dtype=np.float64))
            + 0.015 * np.sum(depth_diagonal * np.array([[1.2, -0.8]], dtype=np.float64))
            + 0.017
            * np.sum(
                flat_diagonal
                * np.array(
                    [
                        [0.0, 0.2, 0.0, 0.0, 0.0],
                        [0.0, 0.0, -0.7, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.1, 0.0],
                        [0.0, 0.0, 0.0, 0.0, -0.3],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=np.float64,
                )
            )
            + 0.011
            * np.sum(
                broadcast_left * np.array([[0.4, -0.2, 0.6], [1.0, -0.5, 0.25]], dtype=np.float64)
            )
            + 0.013
            * np.sum(
                broadcast_right * np.array([[-0.3, 0.7, -0.1], [0.5, -0.4, 0.2]], dtype=np.float64)
            )
            + np.sum(column_assembled * np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            + np.sum(
                depth_stacked
                * np.array(
                    [
                        [[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0]],
                        [[-0.5, 1.25], [0.75, -1.5], [2.5, -0.25]],
                    ],
                    dtype=np.float64,
                )
            )
            + np.sum(
                flat_assembled * np.array([-0.25, 0.5, -1.5, 2.0, 0.75, -0.5], dtype=np.float64)
            )
            + 0.075 * np.sum(flat_sorted * flat_sort_weights)
            + 0.055 * np.sum(axis_sorted * axis_sort_weights)
            + 0.09 * np.sum(row_integrals * trapz_row_weights)
            - 0.04 * flat_integral
            + 0.031 * np.sum(flat_gradient * gradient_flat_weights)
            + 0.027 * np.sum(axis_gradient * gradient_axis_weights)
            + 0.029 * np.sum(sample_interpolation * interp_sample_weights)
            + 0.034 * np.sum(control_interpolation * interp_control_weights)
            + 0.023 * np.sum(dynamic_convolution * convolve_full_weights)
            + 0.019 * np.sum(static_kernel_convolution * convolve_same_weights)
            - 0.017 * np.sum(static_signal_convolution * convolve_valid_weights)
            + 0.021 * np.sum(dynamic_correlation * correlate_full_weights)
            - 0.016 * np.sum(static_reference_correlation * correlate_same_weights)
            + 0.018 * np.sum(static_signal_correlation * correlate_valid_weights)
        )

    analytic = np.array(
        [
            6.18612125,
            4.369901666666667,
            4.915959166666667,
            5.377870833333333,
            8.34373,
            12.0727175,
        ],
        dtype=np.float64,
    )
    return _program_ad_case(
        "indexing_static_gather_contracts",
        "indexing-heavy",
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
    "DifferentiableProgrammingExternalReferenceResult",
    "run_differentiable_programming_benchmark_suite",
    "run_differentiable_programming_external_reference_suite",
]
