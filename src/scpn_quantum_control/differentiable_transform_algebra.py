# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable transform algebra gate.
"""Metamorphic transform-algebra gate for differentiable local routes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_canonical_api import grad, value_and_grad
from .differentiable_finite_difference import (
    hessian,
    jacfwd,
    jacobian,
    jacrev,
    jvp,
    value_and_jacobian,
    vjp,
)
from .differentiable_parameter_contracts import Parameter
from .differentiable_result_contracts import FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY
from .differentiable_sparse_derivatives import sparse_jacobian
from .differentiable_vmap import vmap

FloatArray: TypeAlias = NDArray[np.float64]
VectorObjective: TypeAlias = Callable[[FloatArray], ArrayLike]
ScalarObjective: TypeAlias = Callable[[FloatArray], float | int | np.floating[Any]]
TransformAlgebraStatus = Literal["passed", "failed", "blocked"]

TRANSFORM_ALGEBRA_TOLERANCE: Final[float] = 5.0e-5
TRANSFORM_ALGEBRA_CLAIM_BOUNDARY: Final[str] = (
    "bounded local transform-algebra audit over deterministic NumPy-backed routes; "
    "finite differences remain diagnostic evidence and blocked rows are not promoted "
    "to analytic, framework-native, compiler, provider, hardware, or performance claims"
)
REQUIRED_TRANSFORM_ALGEBRA_CATEGORIES: Final[tuple[str, ...]] = (
    "grad_vmap_composition",
    "vmap_grad_composition",
    "jacrev_jacfwd_composition",
    "hessian_symmetry",
    "jvp_vjp_duality",
    "linearity",
    "chain_rule",
    "finite_difference_diagnostic_boundary",
    "nondifferentiable_boundary",
    "dtype_promotion",
    "broadcasting",
    "deterministic_replay",
    "masked_parameters",
    "sparse_parameters",
    "complex_step_boundary",
    "custom_jvp_vjp_boundary",
    "structured_container_boundary",
    "batched_observables",
)


@dataclass(frozen=True)
class TransformAlgebraCase:
    """One transform-algebra metamorphic or fail-closed boundary check."""

    case_id: str
    category: str
    status: TransformAlgebraStatus
    lhs: tuple[float, ...]
    rhs: tuple[float, ...]
    residual: float | None
    tolerance: float
    evidence: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    claim_boundary: str = TRANSFORM_ALGEBRA_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate case metadata and pass/fail/block invariants."""
        if not self.case_id:
            raise ValueError("transform algebra case_id must be non-empty")
        if not self.category:
            raise ValueError("transform algebra category must be non-empty")
        if self.status not in {"passed", "failed", "blocked"}:
            raise ValueError("transform algebra status is unknown")
        if self.status == "blocked":
            if self.residual is not None:
                raise ValueError("blocked transform algebra cases must not carry residuals")
            if not self.blocked_reasons:
                raise ValueError("blocked transform algebra cases require blocked_reasons")
        else:
            if self.residual is None:
                raise ValueError("executed transform algebra cases require a residual")
            if self.residual < 0.0:
                raise ValueError("transform algebra residual must be non-negative")
            if self.status == "passed" and self.residual > self.tolerance:
                raise ValueError("passed transform algebra case exceeds tolerance")
            if self.status == "failed" and self.residual <= self.tolerance:
                raise ValueError("failed transform algebra case is within tolerance")
        if self.tolerance < 0.0:
            raise ValueError("transform algebra tolerance must be non-negative")
        if any(not item for item in self.evidence):
            raise ValueError("transform algebra evidence entries must be non-empty")
        if any(not item for item in self.blocked_reasons):
            raise ValueError("transform algebra blocked reasons must be non-empty")
        if not self.claim_boundary:
            raise ValueError("transform algebra claim_boundary must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready case metadata."""
        return {
            "case_id": self.case_id,
            "category": self.category,
            "status": self.status,
            "lhs": list(self.lhs),
            "rhs": list(self.rhs),
            "residual": self.residual,
            "tolerance": self.tolerance,
            "evidence": list(self.evidence),
            "blocked_reasons": list(self.blocked_reasons),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class TransformAlgebraAudit:
    """Executable transform-algebra audit over supported and blocked routes."""

    cases: tuple[TransformAlgebraCase, ...]
    required_categories: tuple[str, ...] = REQUIRED_TRANSFORM_ALGEBRA_CATEGORIES
    claim_boundary: str = TRANSFORM_ALGEBRA_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate audit coverage and category uniqueness expectations."""
        if not self.cases:
            raise ValueError("transform algebra audit must contain cases")
        if any(not category for category in self.required_categories):
            raise ValueError("transform algebra required_categories must be non-empty")
        if not self.claim_boundary:
            raise ValueError("transform algebra audit claim_boundary must be non-empty")

    @property
    def categories(self) -> tuple[str, ...]:
        """Return sorted categories covered by this audit."""
        return tuple(sorted({case.category for case in self.cases}))

    @property
    def missing_categories(self) -> tuple[str, ...]:
        """Return required categories missing from the audit."""
        present = set(self.categories)
        return tuple(category for category in self.required_categories if category not in present)

    @property
    def passed_cases(self) -> tuple[TransformAlgebraCase, ...]:
        """Return cases whose executed residuals are within tolerance."""
        return tuple(case for case in self.cases if case.status == "passed")

    @property
    def failed_cases(self) -> tuple[TransformAlgebraCase, ...]:
        """Return cases whose executed residuals exceeded tolerance."""
        return tuple(case for case in self.cases if case.status == "failed")

    @property
    def blocked_cases(self) -> tuple[TransformAlgebraCase, ...]:
        """Return explicit fail-closed transform-algebra boundaries."""
        return tuple(case for case in self.cases if case.status == "blocked")

    @property
    def passed(self) -> bool:
        """Return whether all executed checks passed and every category is covered."""
        return not self.failed_cases and not self.missing_categories

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready audit metadata."""
        return {
            "passed": self.passed,
            "case_count": len(self.cases),
            "passed_count": len(self.passed_cases),
            "blocked_count": len(self.blocked_cases),
            "failed_count": len(self.failed_cases),
            "categories": list(self.categories),
            "missing_categories": list(self.missing_categories),
            "cases": [case.to_dict() for case in self.cases],
            "claim_boundary": self.claim_boundary,
        }


def run_transform_algebra_audit(
    *,
    tolerance: float = TRANSFORM_ALGEBRA_TOLERANCE,
) -> TransformAlgebraAudit:
    """Run the bounded transform-algebra metamorphic audit."""
    if tolerance <= 0.0:
        raise ValueError("transform algebra tolerance must be positive")
    cases = (
        _grad_vmap_case(tolerance),
        _vmap_grad_case(tolerance),
        _jacrev_jacfwd_case(tolerance),
        _hessian_symmetry_case(tolerance),
        _jvp_vjp_duality_case(tolerance),
        _linearity_case(tolerance),
        _chain_rule_case(tolerance),
        _finite_difference_boundary_case(tolerance),
        _blocked_case(
            "nondifferentiable_abs_zero_boundary",
            "nondifferentiable_boundary",
            (
                "abs at zero has a cusp; central finite difference may produce a number "
                "but cannot promote differentiability",
            ),
            ("finite_difference_diagnostic_only", "fail_closed_boundary_row"),
        ),
        _dtype_promotion_case(tolerance),
        _broadcasting_case(tolerance),
        _deterministic_replay_case(tolerance),
        _masked_parameter_case(tolerance),
        _sparse_parameter_case(tolerance),
        _complex_step_case(tolerance),
        _blocked_case(
            "complex_valued_objective_boundary",
            "complex_step_boundary",
            (
                "complex-step support is limited to real-valued analytic objectives; "
                "complex-valued objectives need Wirtinger-specific contracts",
            ),
            ("complex_step_real_analytic_route", "wirtinger_boundary"),
        ),
        _blocked_case(
            "custom_jvp_vjp_unregistered_boundary",
            "custom_jvp_vjp_boundary",
            (
                "custom JVP/VJP composition requires registered exact rules and adjoint "
                "identity evidence before promotion",
            ),
            ("custom_rule_registry_required", "finite_difference_not_promotion_evidence"),
        ),
        _blocked_case(
            "structured_parameter_container_boundary",
            "structured_container_boundary",
            (
                "structured parameter containers need explicit PyTree/container metadata "
                "before transform composition can be promoted",
            ),
            ("first_path_flat_parameter_vector", "framework_parity_lane_required"),
        ),
        _batched_observables_case(tolerance),
    )
    return TransformAlgebraAudit(cases=cases)


def assert_transform_algebra_audit_passes(
    audit: TransformAlgebraAudit | None = None,
) -> TransformAlgebraAudit:
    """Return the audit or raise with actionable failures."""
    candidate = run_transform_algebra_audit() if audit is None else audit
    if candidate.passed:
        return candidate
    details: list[str] = []
    if candidate.missing_categories:
        details.append("missing categories: " + ", ".join(candidate.missing_categories))
    for case in candidate.failed_cases:
        details.append(f"{case.case_id} residual={case.residual} tolerance={case.tolerance}")
    raise AssertionError("; ".join(details))


def _grad_vmap_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([[0.2, -0.1], [0.4, 0.3], [-0.5, 0.7]], dtype=np.float64)

    def row_objective(row: FloatArray) -> float:
        return float(np.sin(row[0]) + row[0] * row[1] + row[1] ** 2)

    def reduced(flat_values: FloatArray) -> float:
        matrix = flat_values.reshape(values.shape)
        mapped = cast(FloatArray, vmap(row_objective)(matrix))
        return float(np.sum(mapped))

    lhs = grad(reduced, values.ravel(), method="finite_difference").reshape(values.shape)
    rhs = np.vstack(
        [grad(row_objective, row, method="finite_difference") for row in values]
    ).astype(np.float64)
    return _executed_case(
        "grad_of_reduced_vmap_matches_row_gradients",
        "grad_vmap_composition",
        lhs,
        rhs,
        tolerance,
        ("grad", "vmap", "finite_difference_diagnostic"),
    )


def _vmap_grad_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([[0.1, 0.2], [0.3, -0.4], [-0.2, 0.6]], dtype=np.float64)

    def row_objective(row: FloatArray) -> float:
        return float(row[0] ** 2 + np.cos(row[1]) + row[0] * row[1])

    vectorized_grad = cast(
        FloatArray,
        vmap(lambda row: grad(row_objective, row, method="finite_difference"))(values),
    )
    expected = np.column_stack(
        (
            2.0 * values[:, 0] + values[:, 1],
            -np.sin(values[:, 1]) + values[:, 0],
        )
    )
    return _executed_case(
        "vmap_of_grad_matches_analytic_rows",
        "vmap_grad_composition",
        vectorized_grad,
        expected,
        tolerance,
        ("vmap", "grad", "analytic_reference"),
    )


def _jacrev_jacfwd_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([0.35, -0.25], dtype=np.float64)

    def vector_objective(x: FloatArray) -> FloatArray:
        return np.array([x[0] ** 2 + x[1], np.sin(x[0] - x[1])], dtype=np.float64)

    lhs = jacfwd(vector_objective, values)
    rhs = jacrev(vector_objective, values)
    return _executed_case(
        "jacfwd_matches_jacrev_for_local_vector_objective",
        "jacrev_jacfwd_composition",
        lhs,
        rhs,
        tolerance,
        ("jacfwd", "jacrev", "finite_difference_diagnostic"),
    )


def _hessian_symmetry_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([0.25, -0.55], dtype=np.float64)

    def objective(x: FloatArray) -> float:
        return float(np.exp(0.1 * x[0]) + x[0] * x[1] + x[1] ** 3)

    matrix = hessian(objective, values)
    return _executed_case(
        "hessian_is_symmetric_for_smooth_scalar_objective",
        "hessian_symmetry",
        matrix,
        matrix.T,
        tolerance,
        ("hessian", "finite_difference_diagnostic"),
    )


def _jvp_vjp_duality_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([0.2, -0.4], dtype=np.float64)
    tangent = np.array([0.7, -0.3], dtype=np.float64)
    cotangent = np.array([1.2, -0.8], dtype=np.float64)

    def vector_objective(x: FloatArray) -> FloatArray:
        return np.array([x[0] ** 2 + x[1], x[0] - np.sin(x[1])], dtype=np.float64)

    jvp_value = jvp(vector_objective, values, tangent)
    vjp_value = vjp(vector_objective, values, cotangent)
    lhs = np.array([float(np.dot(cotangent, jvp_value))], dtype=np.float64)
    rhs = np.array([float(np.dot(vjp_value, tangent))], dtype=np.float64)
    return _executed_case(
        "jvp_vjp_adjoint_inner_product_identity",
        "jvp_vjp_duality",
        lhs,
        rhs,
        tolerance,
        ("jvp", "vjp", "adjoint_identity"),
    )


def _linearity_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([0.45, -0.2], dtype=np.float64)

    def first(x: FloatArray) -> float:
        return float(np.sin(x[0]) + x[1] ** 2)

    def second(x: FloatArray) -> float:
        return float(x[0] * x[1] + np.cos(x[1]))

    def combined(x: FloatArray) -> float:
        return 2.0 * first(x) - 0.5 * second(x)

    lhs = grad(combined, values, method="finite_difference")
    rhs = 2.0 * grad(first, values, method="finite_difference") - 0.5 * grad(
        second,
        values,
        method="finite_difference",
    )
    return _executed_case(
        "gradient_linearity_over_scalar_objectives",
        "linearity",
        lhs,
        rhs,
        tolerance,
        ("grad", "linear_combination", "finite_difference_diagnostic"),
    )


def _chain_rule_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([0.2, -0.3], dtype=np.float64)

    def inner(x: FloatArray) -> FloatArray:
        return np.array([x[0] ** 2 + x[1], np.sin(x[0] - x[1])], dtype=np.float64)

    def outer(y: FloatArray) -> FloatArray:
        return np.array([y[0] + y[1] ** 2, y[0] * y[1]], dtype=np.float64)

    def composed(x: FloatArray) -> FloatArray:
        return outer(inner(x))

    lhs = jacobian(composed, values)
    rhs = jacobian(outer, inner(values)) @ jacobian(inner, values)
    return _executed_case(
        "jacobian_chain_rule_for_smooth_vector_composition",
        "chain_rule",
        lhs,
        rhs,
        tolerance,
        ("jacobian", "chain_rule", "finite_difference_diagnostic"),
    )


def _finite_difference_boundary_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([0.2, -0.4], dtype=np.float64)

    def objective(x: FloatArray) -> float:
        return float(x[0] ** 2 + x[1] ** 2)

    result = value_and_grad(objective, values, method="finite_difference")
    lhs = np.array(
        [1.0 if result.claim_boundary == FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY else 0.0]
    )
    rhs = np.array([1.0], dtype=np.float64)
    return _executed_case(
        "finite_difference_result_keeps_diagnostic_claim_boundary",
        "finite_difference_diagnostic_boundary",
        lhs,
        rhs,
        tolerance,
        ("value_and_grad", "claim_boundary", "finite_difference_diagnostic_only"),
    )


def _dtype_promotion_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([1, -2], dtype=np.int64)

    def objective(x: FloatArray) -> float:
        return float(x[0] ** 2 + 3.0 * x[1])

    lhs = grad(objective, values, method="finite_difference")
    rhs = np.array([2.0, 3.0], dtype=np.float64)
    return _executed_case(
        "integer_input_promotes_to_float_diagnostic_gradient",
        "dtype_promotion",
        lhs,
        rhs,
        tolerance,
        ("grad", "dtype_promotion", "finite_input_validation"),
    )


def _broadcasting_case(tolerance: float) -> TransformAlgebraCase:
    rows = np.array([[0.2, 0.1], [0.4, -0.3]], dtype=np.float64)
    offset = np.array([0.5, -0.25], dtype=np.float64)

    def shifted_row(row: FloatArray, shift: FloatArray) -> FloatArray:
        return row + shift

    lhs = cast(FloatArray, vmap(shifted_row, in_axes=(0, None))(rows, offset))
    rhs = rows + offset
    return _executed_case(
        "vmap_broadcasts_none_axis_arguments",
        "broadcasting",
        lhs,
        rhs,
        tolerance,
        ("vmap", "in_axes_none", "broadcasting"),
    )


def _deterministic_replay_case(tolerance: float) -> TransformAlgebraCase:
    first = _linearity_case(tolerance)
    second = _linearity_case(tolerance)
    lhs = np.array([float(first.residual or 0.0), float(len(first.evidence))], dtype=np.float64)
    rhs = np.array([float(second.residual or 0.0), float(len(second.evidence))], dtype=np.float64)
    return _executed_case(
        "transform_algebra_case_replays_deterministically",
        "deterministic_replay",
        lhs,
        rhs,
        tolerance,
        ("linearity_case", "deterministic_replay"),
    )


def _masked_parameter_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([0.3, -0.7], dtype=np.float64)
    parameters = (Parameter("theta"), Parameter("frozen_bias", trainable=False))

    def objective(x: FloatArray) -> float:
        return float(x[0] ** 2 + 10.0 * x[1] ** 2)

    lhs = grad(objective, values, parameters=parameters, method="finite_difference")
    rhs = np.array([2.0 * values[0], 0.0], dtype=np.float64)
    return _executed_case(
        "masked_non_trainable_parameter_has_zero_gradient",
        "masked_parameters",
        lhs,
        rhs,
        tolerance,
        ("grad", "trainable_mask", "Parameter"),
    )


def _sparse_parameter_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([0.2, -0.1, 0.4], dtype=np.float64)
    parameters = (Parameter("x"), Parameter("frozen_y", trainable=False), Parameter("z"))

    def vector_objective(x: FloatArray) -> FloatArray:
        return np.array([x[0] ** 2, 3.0 * x[2]], dtype=np.float64)

    dense = value_and_jacobian(vector_objective, values, parameters=parameters)
    sparse = sparse_jacobian(dense, tolerance=1.0e-12)
    lhs = sparse.to_dense()
    rhs = np.array([[2.0 * values[0], 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float64)
    return _executed_case(
        "sparse_jacobian_preserves_masked_parameter_columns",
        "sparse_parameters",
        lhs,
        rhs,
        tolerance,
        ("value_and_jacobian", "sparse_jacobian", "trainable_mask"),
    )


def _complex_step_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([0.15, -0.2], dtype=np.float64)

    def objective(x: NDArray[np.complex128]) -> object:
        return np.exp(x[0]) + x[1] ** 2

    result = value_and_grad(objective, values, method="complex_step")
    lhs = result.gradient
    rhs = np.array([np.exp(values[0]), 2.0 * values[1]], dtype=np.float64)
    return _executed_case(
        "complex_step_real_analytic_gradient_matches_reference",
        "complex_step_boundary",
        lhs,
        rhs,
        tolerance,
        ("value_and_grad", "complex_step", "real_analytic_objective"),
    )


def _batched_observables_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([0.3, -0.2], dtype=np.float64)

    def vector_objective(x: FloatArray) -> FloatArray:
        return np.array([np.sin(x[0]), x[0] * x[1], x[1] ** 2], dtype=np.float64)

    lhs = jacobian(vector_objective, values)
    rhs = np.array(
        [[np.cos(values[0]), 0.0], [values[1], values[0]], [0.0, 2.0 * values[1]]],
        dtype=np.float64,
    )
    return _executed_case(
        "batched_observable_jacobian_matches_analytic_reference",
        "batched_observables",
        lhs,
        rhs,
        tolerance,
        ("jacobian", "batched_vector_output", "analytic_reference"),
    )


def _executed_case(
    case_id: str,
    category: str,
    lhs: ArrayLike,
    rhs: ArrayLike,
    tolerance: float,
    evidence: Sequence[str],
) -> TransformAlgebraCase:
    lhs_array = _as_flat_float_tuple(lhs)
    rhs_array = _as_flat_float_tuple(rhs)
    residual = _max_abs(lhs_array, rhs_array)
    return TransformAlgebraCase(
        case_id=case_id,
        category=category,
        status="passed" if residual <= tolerance else "failed",
        lhs=tuple(float(value) for value in lhs_array),
        rhs=tuple(float(value) for value in rhs_array),
        residual=residual,
        tolerance=tolerance,
        evidence=tuple(evidence),
        blocked_reasons=(),
    )


def _blocked_case(
    case_id: str,
    category: str,
    blocked_reasons: Sequence[str],
    evidence: Sequence[str],
) -> TransformAlgebraCase:
    return TransformAlgebraCase(
        case_id=case_id,
        category=category,
        status="blocked",
        lhs=(),
        rhs=(),
        residual=None,
        tolerance=TRANSFORM_ALGEBRA_TOLERANCE,
        evidence=tuple(evidence),
        blocked_reasons=tuple(blocked_reasons),
    )


def _as_flat_float_tuple(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("transform algebra values must be finite")
    return cast(FloatArray, array.reshape(-1).astype(np.float64, copy=True))


def _max_abs(lhs: FloatArray, rhs: FloatArray) -> float:
    if lhs.shape != rhs.shape:
        return float("inf")
    if lhs.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs - rhs)))


__all__ = [
    "REQUIRED_TRANSFORM_ALGEBRA_CATEGORIES",
    "TRANSFORM_ALGEBRA_CLAIM_BOUNDARY",
    "TRANSFORM_ALGEBRA_TOLERANCE",
    "TransformAlgebraAudit",
    "TransformAlgebraCase",
    "TransformAlgebraStatus",
    "assert_transform_algebra_audit_passes",
    "run_transform_algebra_audit",
]
