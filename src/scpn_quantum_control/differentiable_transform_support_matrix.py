# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable transform-algebra support matrix.
"""Generated support-matrix rows for the transform-algebra audit."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol

TransformAlgebraSupportStatus = Literal["passed", "failed", "blocked"]
TransformAlgebraLane = Literal[
    "native",
    "custom_rules",
    "program_ad",
    "quantum_gradients",
    "unsupported_boundary",
]

TRANSFORM_ALGEBRA_SUPPORT_MATRIX_CLAIM_BOUNDARY: Final[str] = (
    "generated from bounded local transform-algebra audit rows; supported rows "
    "are local conformance evidence only, and blocked rows are not promoted to "
    "analytic, framework-native, compiler, provider, hardware, or performance claims"
)
REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS: Final[tuple[str, ...]] = (
    "native_grad_vmap",
    "native_vmap_grad",
    "native_jacfwd_jacrev",
    "native_hessian",
    "native_jvp_vjp",
    "registered_custom_rules",
    "program_ad_jvp_vjp",
    "program_ad_hessian",
    "quantum_gradient_native_nesting",
    "unsupported_custom_rule_registration",
    "unsupported_complex_valued_objective",
    "unsupported_structured_container",
    "unsupported_nondifferentiable_boundary",
)


class TransformAlgebraCaseLike(Protocol):
    """Structural view of an audit case consumed by matrix generation."""

    @property
    def case_id(self) -> str:
        """Return the source audit case identifier."""

    @property
    def status(self) -> TransformAlgebraSupportStatus:
        """Return the source audit case status."""

    @property
    def residual(self) -> float | None:
        """Return the source audit case residual when it executed."""

    @property
    def tolerance(self) -> float:
        """Return the source audit case tolerance."""

    @property
    def evidence(self) -> tuple[str, ...]:
        """Return evidence labels emitted by the source audit case."""

    @property
    def blocked_reasons(self) -> tuple[str, ...]:
        """Return fail-closed reasons emitted by the source audit case."""


@dataclass(frozen=True)
class TransformAlgebraSupportMatrixRow:
    """One generated reviewer-facing support-matrix row.

    The row is derived from one or more executable or fail-closed audit cases.
    It therefore mirrors the test battery instead of maintaining a separate
    hand-written capability claim.
    """

    row_id: str
    lane: TransformAlgebraLane
    transform_stack: tuple[str, ...]
    supported: bool
    status: TransformAlgebraSupportStatus
    case_ids: tuple[str, ...]
    residual: float | None
    tolerance: float
    evidence: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    notes: tuple[str, ...]
    claim_boundary: str = TRANSFORM_ALGEBRA_SUPPORT_MATRIX_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate matrix-row invariants."""
        if not self.row_id:
            raise ValueError("transform algebra support row_id must be non-empty")
        if self.lane not in {
            "native",
            "custom_rules",
            "program_ad",
            "quantum_gradients",
            "unsupported_boundary",
        }:
            raise ValueError("transform algebra support row lane is unknown")
        if not self.transform_stack:
            raise ValueError("transform algebra support row transform_stack must be non-empty")
        if any(not transform for transform in self.transform_stack):
            raise ValueError("transform algebra support row transforms must be non-empty")
        if self.status not in {"passed", "failed", "blocked"}:
            raise ValueError("transform algebra support row status is unknown")
        if not self.case_ids:
            raise ValueError("transform algebra support row must reference audit cases")
        if any(not case_id for case_id in self.case_ids):
            raise ValueError("transform algebra support row case_ids must be non-empty")
        if self.supported and self.status != "passed":
            raise ValueError("supported transform algebra rows must have passed status")
        if self.status == "passed" and not self.supported:
            raise ValueError("passed transform algebra rows must be supported")
        if self.status == "blocked" and self.supported:
            raise ValueError("blocked transform algebra rows must not be supported")
        if self.status == "blocked":
            if self.residual is not None:
                raise ValueError("blocked transform algebra rows must not carry residuals")
            if not self.blocked_reasons:
                raise ValueError("blocked transform algebra rows require blocked_reasons")
        else:
            if self.residual is None:
                raise ValueError("executed transform algebra rows require a residual")
            if self.residual < 0.0:
                raise ValueError("transform algebra support row residual must be non-negative")
        if self.tolerance < 0.0:
            raise ValueError("transform algebra support row tolerance must be non-negative")
        if any(not item for item in self.evidence):
            raise ValueError("transform algebra support row evidence entries must be non-empty")
        if any(not item for item in self.blocked_reasons):
            raise ValueError("transform algebra support row blockers must be non-empty")
        if any(not item for item in self.notes):
            raise ValueError("transform algebra support row notes must be non-empty")
        if not self.claim_boundary:
            raise ValueError("transform algebra support row claim_boundary must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready support-row metadata."""
        return {
            "row_id": self.row_id,
            "lane": self.lane,
            "transform_stack": list(self.transform_stack),
            "supported": self.supported,
            "status": self.status,
            "case_ids": list(self.case_ids),
            "residual": self.residual,
            "tolerance": self.tolerance,
            "evidence": list(self.evidence),
            "blocked_reasons": list(self.blocked_reasons),
            "notes": list(self.notes),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class _SupportMatrixRowSpec:
    row_id: str
    lane: TransformAlgebraLane
    transform_stack: tuple[str, ...]
    case_ids: tuple[str, ...]
    notes: tuple[str, ...]


_SUPPORT_MATRIX_ROW_SPECS: Final[tuple[_SupportMatrixRowSpec, ...]] = (
    _SupportMatrixRowSpec(
        "native_grad_vmap",
        "native",
        ("grad", "vmap"),
        ("grad_of_reduced_vmap_matches_row_gradients",),
        ("native local finite-difference diagnostic transform composition",),
    ),
    _SupportMatrixRowSpec(
        "native_vmap_grad",
        "native",
        ("vmap", "grad"),
        ("vmap_of_grad_matches_analytic_rows",),
        ("native local row-wise gradient vectorisation",),
    ),
    _SupportMatrixRowSpec(
        "native_jacfwd_jacrev",
        "native",
        ("jacfwd", "jacrev"),
        ("jacfwd_matches_jacrev_for_local_vector_objective",),
        ("forward and reverse finite-difference Jacobian routes agree",),
    ),
    _SupportMatrixRowSpec(
        "native_hessian",
        "native",
        ("hessian",),
        ("hessian_is_symmetric_for_smooth_scalar_objective",),
        ("smooth scalar-objective Hessian symmetry",),
    ),
    _SupportMatrixRowSpec(
        "native_jvp_vjp",
        "native",
        ("jvp", "vjp"),
        ("jvp_vjp_adjoint_inner_product_identity",),
        ("directional transform adjoint identity",),
    ),
    _SupportMatrixRowSpec(
        "registered_custom_rules",
        "custom_rules",
        ("vmap", "custom_jvp", "custom_vjp"),
        ("registered_custom_rule_vmap_jvp_vjp_matches_reference",),
        ("registered exact custom JVP/VJP rules under native vmap",),
    ),
    _SupportMatrixRowSpec(
        "program_ad_jvp_vjp",
        "program_ad",
        ("jvp", "vjp", "vmap", "whole_program_grad"),
        ("program_ad_vmap_gradient_jvp_vjp_matches_block_hessian",),
        ("directional transforms over vmap of whole-program AD gradients",),
    ),
    _SupportMatrixRowSpec(
        "program_ad_hessian",
        "program_ad",
        ("hessian", "whole_program_value_and_grad"),
        ("program_ad_hessian_matches_quadratic_curvature",),
        ("Hessian over a whole-program AD scalar objective",),
    ),
    _SupportMatrixRowSpec(
        "quantum_gradient_native_nesting",
        "quantum_gradients",
        ("vmap", "grad", "parameter_shift"),
        ("phase_qnode_native_vmap_grad_matches_parameter_shift_reference",),
        ("deterministic native local phase-QNode manual vmap(grad)",),
    ),
    _SupportMatrixRowSpec(
        "unsupported_custom_rule_registration",
        "unsupported_boundary",
        ("custom_jvp", "custom_vjp"),
        ("custom_jvp_vjp_unregistered_boundary",),
        ("unregistered custom rules remain fail-closed",),
    ),
    _SupportMatrixRowSpec(
        "unsupported_complex_valued_objective",
        "unsupported_boundary",
        ("complex_step", "wirtinger"),
        ("complex_valued_objective_boundary",),
        ("complex-valued objectives need Wirtinger-specific contracts",),
    ),
    _SupportMatrixRowSpec(
        "unsupported_structured_container",
        "unsupported_boundary",
        ("pytree", "structured_container"),
        ("structured_parameter_container_boundary",),
        ("structured containers need explicit metadata before promotion",),
    ),
    _SupportMatrixRowSpec(
        "unsupported_nondifferentiable_boundary",
        "unsupported_boundary",
        ("grad", "nondifferentiable"),
        ("nondifferentiable_abs_zero_boundary",),
        ("nondifferentiable cusps are diagnostic-only boundaries",),
    ),
)


def build_transform_algebra_support_matrix(
    cases: Sequence[TransformAlgebraCaseLike],
) -> tuple[TransformAlgebraSupportMatrixRow, ...]:
    """Return support rows generated from transform-algebra audit cases."""
    case_map = {case.case_id: case for case in cases}
    return tuple(
        _support_matrix_row_from_spec(spec, case_map) for spec in _SUPPORT_MATRIX_ROW_SPECS
    )


def _support_matrix_row_from_spec(
    spec: _SupportMatrixRowSpec,
    case_map: dict[str, TransformAlgebraCaseLike],
) -> TransformAlgebraSupportMatrixRow:
    selected = tuple(case_map[case_id] for case_id in spec.case_ids if case_id in case_map)
    missing = tuple(case_id for case_id in spec.case_ids if case_id not in case_map)
    if missing:
        return TransformAlgebraSupportMatrixRow(
            row_id=spec.row_id,
            lane=spec.lane,
            transform_stack=spec.transform_stack,
            supported=False,
            status="failed",
            case_ids=spec.case_ids,
            residual=float("inf"),
            tolerance=0.0,
            evidence=("support_matrix_generation",),
            blocked_reasons=("support matrix source case missing: " + ", ".join(missing),),
            notes=spec.notes,
        )
    if any(case.status == "failed" for case in selected):
        return TransformAlgebraSupportMatrixRow(
            row_id=spec.row_id,
            lane=spec.lane,
            transform_stack=spec.transform_stack,
            supported=False,
            status="failed",
            case_ids=spec.case_ids,
            residual=_max_case_residual(selected),
            tolerance=max(case.tolerance for case in selected),
            evidence=_unique_strings(item for case in selected for item in case.evidence),
            blocked_reasons=("one or more source audit cases failed",),
            notes=spec.notes,
        )
    if all(case.status == "blocked" for case in selected):
        return TransformAlgebraSupportMatrixRow(
            row_id=spec.row_id,
            lane=spec.lane,
            transform_stack=spec.transform_stack,
            supported=False,
            status="blocked",
            case_ids=spec.case_ids,
            residual=None,
            tolerance=max(case.tolerance for case in selected),
            evidence=_unique_strings(item for case in selected for item in case.evidence),
            blocked_reasons=_unique_strings(
                reason for case in selected for reason in case.blocked_reasons
            ),
            notes=spec.notes,
        )
    return TransformAlgebraSupportMatrixRow(
        row_id=spec.row_id,
        lane=spec.lane,
        transform_stack=spec.transform_stack,
        supported=True,
        status="passed",
        case_ids=spec.case_ids,
        residual=_max_case_residual(selected),
        tolerance=max(case.tolerance for case in selected),
        evidence=_unique_strings(item for case in selected for item in case.evidence),
        blocked_reasons=(),
        notes=spec.notes,
    )


def _max_case_residual(cases: Sequence[TransformAlgebraCaseLike]) -> float:
    residuals = [case.residual for case in cases if case.residual is not None]
    if not residuals:
        return 0.0
    return float(max(residuals))


def _unique_strings(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


__all__ = [
    "REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS",
    "TRANSFORM_ALGEBRA_SUPPORT_MATRIX_CLAIM_BOUNDARY",
    "TransformAlgebraCaseLike",
    "TransformAlgebraLane",
    "TransformAlgebraSupportMatrixRow",
    "TransformAlgebraSupportStatus",
    "build_transform_algebra_support_matrix",
]
