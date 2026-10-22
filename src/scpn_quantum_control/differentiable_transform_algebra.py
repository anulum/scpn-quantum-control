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
from .differentiable_custom_derivatives import custom_jvp, custom_vjp
from .differentiable_finite_difference import (
    hessian,
    jacfwd,
    jacobian,
    jacrev,
    jvp,
    value_and_jacobian,
    vjp,
)
from .differentiable_local_transform_composition import (
    adjoint_inner_product_pair,
    chain_rule_jacobian_pair,
    gradient_linear_difference_pair,
    reduced_mapped_gradient_pair,
)
from .differentiable_local_transform_primitives import LocalTransformPrimitives
from .differentiable_parameter_contracts import Parameter
from .differentiable_result_contracts import FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY
from .differentiable_sparse_derivatives import sparse_jacobian
from .differentiable_transform_comparison import (
    TransformComparisonPrimitives,
    as_finite_flat_array,
    maximum_absolute_residual,
)
from .differentiable_transform_support_matrix import (
    NATIVE_TRANSFORM_ALGEBRA_CATEGORIES,
    REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS,
    TransformAlgebraLane,
    TransformAlgebraSupportMatrixRow,
    build_transform_algebra_support_matrix,
)
from .differentiable_vmap import vmap
from .phase.qnode_vector_transforms import execute_phase_qnode_vmap_grad
from .program_ad_registry import CustomDerivativeRule
from .whole_program_ad_api import whole_program_value_and_grad

FloatArray: TypeAlias = NDArray[np.float64]
VectorObjective: TypeAlias = Callable[[FloatArray], ArrayLike]
ScalarObjective: TypeAlias = Callable[[FloatArray], float | int | np.floating[Any]]
TransformAlgebraStatus = Literal["passed", "failed", "blocked"]

_LOCAL_TRANSFORM_PRIMITIVES = LocalTransformPrimitives(
    gradient=lambda: grad,
    jacobian=lambda: jacobian,
    jvp=lambda: jvp,
    vjp=lambda: vjp,
    numpy=lambda: np,
    scalar=lambda: float,
)

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
    "periodicity",
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
    "registered_custom_rule_composition",
    "program_ad_jvp_vjp_composition",
    "program_ad_hessian_composition",
    "quantum_gradient_nesting",
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
        """Sorted categories covered by this audit."""
        return tuple(sorted({case.category for case in self.cases}))

    @property
    def missing_categories(self) -> tuple[str, ...]:
        """Required categories missing from the audit."""
        present = set(self.categories)
        return tuple(category for category in self.required_categories if category not in present)

    @property
    def passed_cases(self) -> tuple[TransformAlgebraCase, ...]:
        """Cases whose executed residuals are within tolerance."""
        return tuple(case for case in self.cases if case.status == "passed")

    @property
    def failed_cases(self) -> tuple[TransformAlgebraCase, ...]:
        """Cases whose executed residuals exceeded tolerance."""
        return tuple(case for case in self.cases if case.status == "failed")

    @property
    def blocked_cases(self) -> tuple[TransformAlgebraCase, ...]:
        """Explicit fail-closed transform-algebra boundaries."""
        return tuple(case for case in self.cases if case.status == "blocked")

    @property
    def support_matrix(self) -> tuple[TransformAlgebraSupportMatrixRow, ...]:
        """Support rows generated from executable and blocked cases."""
        return build_transform_algebra_support_matrix(self.cases)

    @property
    def missing_support_rows(self) -> tuple[str, ...]:
        """Required support-matrix rows missing from the generated matrix."""
        present = {row.row_id for row in self.support_matrix}
        if self.required_categories == NATIVE_TRANSFORM_ALGEBRA_CATEGORIES:
            return tuple(case.case_id for case in self.cases if case.case_id not in present)
        return tuple(
            row_id for row_id in REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS if row_id not in present
        )

    @property
    def failed_support_rows(self) -> tuple[TransformAlgebraSupportMatrixRow, ...]:
        """Generated support rows whose source cases failed."""
        return tuple(row for row in self.support_matrix if row.status == "failed")

    @property
    def passed(self) -> bool:
        """Whether all executed checks passed and every category is covered."""
        return (
            not self.failed_cases
            and not self.missing_categories
            and not self.missing_support_rows
            and not self.failed_support_rows
        )

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
            "support_matrix": [row.to_dict() for row in self.support_matrix],
            "missing_support_rows": list(self.missing_support_rows),
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
        _registered_custom_rule_case(tolerance),
        _program_ad_jvp_vjp_case(tolerance),
        _program_ad_hessian_case(tolerance),
        _quantum_gradient_native_vmap_case(tolerance),
        _linearity_case(tolerance),
        _chain_rule_case(tolerance),
        _periodicity_case(tolerance),
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
    if candidate.missing_support_rows:
        details.append("missing support rows: " + ", ".join(candidate.missing_support_rows))
    for row in candidate.failed_support_rows:
        details.append(
            f"support row {row.row_id} residual={row.residual} tolerance={row.tolerance}"
        )
    for case in candidate.failed_cases:
        details.append(f"{case.case_id} residual={case.residual} tolerance={case.tolerance}")
    raise AssertionError("; ".join(details))


def run_native_transform_algebra_audit(
    framework: str,
    values: ArrayLike,
    *,
    batch_axis: int = 0,
) -> TransformAlgebraAudit:
    """Qualify named CPU framework compositions against independent derivatives.

    Parameters
    ----------
    framework:
        Exact runtime selector: ``jax``, ``pytorch`` or ``tensorflow``. The
        selected dependency is required; no diagnostic fallback is substituted.
    values:
        Dimensionless finite real parameters, either a vector or a matrix of
        independent samples. Each dimension is between one and sixteen. Inputs
        are copied and promoted to float64 before invoking the selected runtime.
    batch_axis:
        Matrix axis containing independent samples, zero or one. Vector inputs
        require zero. Returned batched evidence always orders samples first.

    Returns
    -------
    TransformAlgebraAudit
        Executed native compositions and explicit unsupported project bridge
        boundaries, with generated support rows. This read-only audit does not
        submit provider jobs, persist results or qualify a physical device.

    Raises
    ------
    ValueError
        If the selector, numeric domain, shape or batch axis is inadmissible.
    ImportError
        If the selected required framework cannot be imported.
    RuntimeError
        If a required native transform is unavailable or cannot execute.

    """
    if framework not in {"jax", "pytorch", "tensorflow"}:
        raise ValueError("framework must be jax, pytorch or tensorflow")
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.ndim not in {1, 2}:
        raise ValueError("values must be a real numeric vector or matrix")
    if type(batch_axis) is not int or batch_axis not in {0, 1}:
        raise ValueError("batch_axis must be zero or one")
    if raw.ndim == 1 and batch_axis != 0:
        raise ValueError("vector values require batch_axis zero")
    if any(size < 1 or size > 16 for size in raw.shape):
        raise ValueError("each input dimension must be between one and sixteen")
    if not np.all(np.isfinite(raw)):
        raise ValueError("values must contain only finite entries")
    matrix = np.array(raw, dtype=np.float64, copy=True)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    elif batch_axis == 1:
        matrix = matrix.T.copy()
    if framework == "jax":
        from .differentiable_jax_transform_algebra import run_jax_transform_algebra_cases

        cases = run_jax_transform_algebra_cases(matrix, batch_axis=batch_axis)
    elif framework == "pytorch":
        from .differentiable_torch_transform_algebra import run_torch_transform_algebra_cases

        cases = run_torch_transform_algebra_cases(matrix, batch_axis=batch_axis)
    else:
        from .differentiable_tensorflow_transform_algebra import (
            run_tensorflow_transform_algebra_cases,
        )

        cases = run_tensorflow_transform_algebra_cases(matrix, batch_axis=batch_axis)
    return TransformAlgebraAudit(
        cases=cases,
        required_categories=NATIVE_TRANSFORM_ALGEBRA_CATEGORIES,
        claim_boundary=(
            f"framework-native {framework} CPU float64 composition qualification; "
            "each support row names its executed transform and domain; project "
            "host bridges retain separate refusal boundaries; no provider, "
            "hardware, accelerator or performance qualification"
        ),
    )


def _framework_transform_case(
    framework: str,
    version: str,
    category: str,
    actual: ArrayLike,
    expected: ArrayLike,
    transforms: Sequence[str],
) -> TransformAlgebraCase:
    """Compare an executed CPU transform with its independent float64 oracle."""
    return _executed_case(
        f"{framework}_{category}",
        category,
        actual,
        expected,
        1.0e-10,
        (
            f"framework={framework}",
            f"version={version}",
            "device=cpu",
            "dtype=float64",
            "manual_batching=False",
            "host_callback=False",
            *(f"transform={transform}" for transform in transforms),
        ),
        claim_boundary=(
            f"framework-native {framework} {version} CPU float64; "
            "only the executed named composition and smooth deterministic fixture; "
            "no project host-bridge, provider, hardware or performance promotion"
        ),
    )


def _framework_quantum_bridge_boundary_case(framework: str) -> TransformAlgebraCase:
    """Retain the generic quantum bridge's pre-execution nesting refusal."""
    from .phase.transform_nesting import plan_gradient_transform_nesting

    plan = plan_gradient_transform_nesting(("grad", "grad", "grad"), adapter=framework)
    if plan.supported:
        raise RuntimeError("undeclared third-order quantum bridge unexpectedly accepted")
    return TransformAlgebraCase(
        case_id=f"{framework}_framework_quantum_bridge_boundary",
        category="framework_quantum_bridge_boundary",
        status="blocked",
        lhs=(),
        rhs=(),
        residual=None,
        tolerance=0.0,
        evidence=("plan_gradient_transform_nesting", "no_derivative_execution"),
        blocked_reasons=plan.blocked_reasons,
        claim_boundary="generic quantum bridge nesting remains unsupported before execution",
    )


def _grad_vmap_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([[0.2, -0.1], [0.4, 0.3], [-0.5, 0.7]], dtype=np.float64)

    def row_objective(row: FloatArray) -> float:
        return float(np.sin(row[0]) + row[0] * row[1] + row[1] ** 2)

    def reduced(flat_values: FloatArray) -> float:
        matrix = flat_values.reshape(values.shape)
        mapped = cast(FloatArray, vmap(row_objective)(matrix))
        return float(np.sum(mapped))

    lhs, rhs = reduced_mapped_gradient_pair(
        reduced, row_objective, values, primitives=_LOCAL_TRANSFORM_PRIMITIVES
    )
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

    lhs, rhs = adjoint_inner_product_pair(
        vector_objective, values, tangent, cotangent, primitives=_LOCAL_TRANSFORM_PRIMITIVES
    )
    return _executed_case(
        "jvp_vjp_adjoint_inner_product_identity",
        "jvp_vjp_duality",
        lhs,
        rhs,
        tolerance,
        ("jvp", "vjp", "adjoint_identity"),
    )


def _registered_custom_rule_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([[0.25, -0.4], [0.6, 0.2], [-0.1, 0.5]], dtype=np.float64)
    tangents = np.array([[0.3, -0.2], [-0.5, 0.4], [0.7, 0.1]], dtype=np.float64)
    cotangents = np.array([[1.0, -0.25], [0.5, 0.75], [-0.4, 1.2]], dtype=np.float64)

    def value_fn(row: FloatArray) -> FloatArray:
        return np.array([row[0] ** 2 + np.sin(row[1]), row[0] * row[1]], dtype=np.float64)

    def jvp_rule(row: FloatArray, tangent: FloatArray) -> FloatArray:
        return np.array(
            [
                2.0 * row[0] * tangent[0] + np.cos(row[1]) * tangent[1],
                tangent[0] * row[1] + row[0] * tangent[1],
            ],
            dtype=np.float64,
        )

    def vjp_rule(row: FloatArray, cotangent: FloatArray) -> FloatArray:
        return np.array(
            [
                2.0 * row[0] * cotangent[0] + row[1] * cotangent[1],
                np.cos(row[1]) * cotangent[0] + row[0] * cotangent[1],
            ],
            dtype=np.float64,
        )

    rule = CustomDerivativeRule(
        name="transform_algebra_registered_custom_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
        parameter_names=("x0", "x1"),
    )
    mapped_jvp = cast(
        FloatArray,
        vmap(lambda row, tangent: custom_jvp(rule, row, tangent))(values, tangents),
    )
    mapped_vjp = cast(
        FloatArray,
        vmap(lambda row, cotangent: custom_vjp(rule, row, cotangent).vjp)(
            values,
            cotangents,
        ),
    )
    expected_jvp = np.asarray(
        [jvp_rule(row, tangent) for row, tangent in zip(values, tangents, strict=True)],
        dtype=np.float64,
    )
    expected_vjp = np.asarray(
        [vjp_rule(row, cotangent) for row, cotangent in zip(values, cotangents, strict=True)],
        dtype=np.float64,
    )
    lhs = np.concatenate([mapped_jvp.reshape(-1), mapped_vjp.reshape(-1)])
    rhs = np.concatenate([expected_jvp.reshape(-1), expected_vjp.reshape(-1)])
    return _executed_case(
        "registered_custom_rule_vmap_jvp_vjp_matches_reference",
        "registered_custom_rule_composition",
        lhs,
        rhs,
        tolerance,
        ("vmap", "custom_jvp", "custom_vjp", "CustomDerivativeRule", "analytic_reference"),
    )


def _program_ad_jvp_vjp_case(tolerance: float) -> TransformAlgebraCase:
    rows = np.array([[0.45, 0.2], [-0.35, 0.55]], dtype=np.float64)
    values = rows.reshape(-1)
    tangent = np.array([0.25, -0.1, 0.35, 0.2], dtype=np.float64)
    cotangent = np.array([0.9, -0.25, 0.4, 0.75], dtype=np.float64)
    row_hessian = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    block_hessian = np.zeros((values.size, values.size), dtype=np.float64)
    block_hessian[0:2, 0:2] = row_hessian
    block_hessian[2:4, 2:4] = row_hessian

    def row_loss(row: Any) -> object:
        return row[0] * row[0] + row[0] * row[1] + 0.5 * row[1] * row[1]

    def mapped_program_gradient(candidate: FloatArray) -> FloatArray:
        matrix = candidate.reshape(rows.shape)
        mapped = vmap(
            lambda row: whole_program_value_and_grad(row_loss, row, trace=False).gradient
        )(matrix)
        return np.asarray(mapped, dtype=np.float64).reshape(-1)

    lhs = np.concatenate(
        [
            jvp(mapped_program_gradient, values, tangent, step=1.0e-2),
            vjp(mapped_program_gradient, values, cotangent, step=1.0e-2),
        ]
    )
    rhs = np.concatenate([block_hessian @ tangent, block_hessian.T @ cotangent])
    return _executed_case(
        "program_ad_vmap_gradient_jvp_vjp_matches_block_hessian",
        "program_ad_jvp_vjp_composition",
        lhs,
        rhs,
        tolerance,
        ("whole_program_value_and_grad", "vmap", "jvp", "vjp", "analytic_reference"),
    )


def _program_ad_hessian_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([0.4, -0.65], dtype=np.float64)
    expected = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=np.float64)

    def row_loss(row: Any) -> object:
        return row[0] * row[0] + row[0] * row[1] + 0.5 * row[1] * row[1]

    def program_value(candidate: FloatArray) -> float:
        return float(whole_program_value_and_grad(row_loss, candidate, trace=False).value)

    lhs = hessian(program_value, values, step=1.0e-1)
    return _executed_case(
        "program_ad_hessian_matches_quadratic_curvature",
        "program_ad_hessian_composition",
        lhs,
        expected,
        tolerance,
        ("whole_program_value_and_grad", "hessian", "analytic_reference"),
    )


def _quantum_gradient_native_vmap_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([[0.2, -0.4], [0.7, 0.1], [-0.3, 0.6]], dtype=np.float64)

    def objective(params: FloatArray) -> float:
        return float(np.cos(params[0]) + 0.25 * np.sin(params[1]))

    result = execute_phase_qnode_vmap_grad(objective, values)
    if not result.supported or result.batched_gradients is None:
        return _blocked_case(
            "phase_qnode_native_vmap_grad_is_fail_closed",
            "quantum_gradient_nesting",
            result.plan.blocked_reasons or (result.failure_reason,),
            ("execute_phase_qnode_vmap_grad", "phase_qnode_parameter_shift"),
        )

    expected = np.column_stack((-np.sin(values[:, 0]), 0.25 * np.cos(values[:, 1]))).astype(
        np.float64
    )
    return _executed_case(
        "phase_qnode_native_vmap_grad_matches_parameter_shift_reference",
        "quantum_gradient_nesting",
        result.batched_gradients,
        expected,
        tolerance,
        ("execute_phase_qnode_vmap_grad", "parameter_shift", "phase_qnode", result.transform),
    )


def _linearity_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([0.45, -0.2], dtype=np.float64)

    def first(x: FloatArray) -> float:
        return float(np.sin(x[0]) + x[1] ** 2)

    def second(x: FloatArray) -> float:
        return float(x[0] * x[1] + np.cos(x[1]))

    def combined(x: FloatArray) -> float:
        return 2.0 * first(x) - 0.5 * second(x)

    lhs, rhs = gradient_linear_difference_pair(
        combined, first, second, values, 2.0, 0.5, primitives=_LOCAL_TRANSFORM_PRIMITIVES
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

    lhs, rhs = chain_rule_jacobian_pair(
        composed, outer, inner, values, primitives=_LOCAL_TRANSFORM_PRIMITIVES
    )
    return _executed_case(
        "jacobian_chain_rule_for_smooth_vector_composition",
        "chain_rule",
        lhs,
        rhs,
        tolerance,
        ("jacobian", "chain_rule", "finite_difference_diagnostic"),
    )


def _periodicity_case(tolerance: float) -> TransformAlgebraCase:
    values = np.array([np.pi - 0.02, -np.pi + 0.04], dtype=np.float64)
    periods = np.array([2.0 * np.pi, -2.0 * np.pi], dtype=np.float64)

    def objective(x: FloatArray) -> float:
        return float(np.sin(x[0]) + 0.25 * np.cos(x[1]) + 0.1 * np.sin(x[0] - x[1]))

    lhs = grad(objective, values, method="parameter_shift")
    rhs = grad(objective, values + periods, method="parameter_shift")
    return _executed_case(
        "parameter_shift_gradient_is_phase_periodic",
        "periodicity",
        lhs,
        rhs,
        tolerance,
        ("grad", "parameter_shift", "phase_wraparound"),
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
    *,
    claim_boundary: str = TRANSFORM_ALGEBRA_CLAIM_BOUNDARY,
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
        claim_boundary=claim_boundary,
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
    """Admit comparison values through the AD-owned numerical implementation."""
    return as_finite_flat_array(values, primitives=_comparison_primitives)


def _max_abs(lhs: FloatArray, rhs: FloatArray) -> float:
    """Compare admitted values through the AD-owned numerical implementation."""
    return maximum_absolute_residual(lhs, rhs, primitives=_comparison_primitives)


_comparison_primitives = TransformComparisonPrimitives(
    numpy=lambda: np,
    scalar=lambda: float,
    error=lambda: ValueError,
    array_cast=lambda: cast,
    array_type=lambda: FloatArray,
)


__all__ = [
    "REQUIRED_TRANSFORM_ALGEBRA_CATEGORIES",
    "REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS",
    "TRANSFORM_ALGEBRA_CLAIM_BOUNDARY",
    "TRANSFORM_ALGEBRA_TOLERANCE",
    "TransformAlgebraAudit",
    "TransformAlgebraCase",
    "TransformAlgebraLane",
    "TransformAlgebraSupportMatrixRow",
    "TransformAlgebraStatus",
    "assert_transform_algebra_audit_passes",
    "run_transform_algebra_audit",
    "run_native_transform_algebra_audit",
]
