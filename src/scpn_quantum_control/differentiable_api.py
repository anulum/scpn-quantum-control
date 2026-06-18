# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- unified differentiable API facade
"""Unified differentiable-programming facade over supported local routes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .benchmarks.differentiable_programming import (
    run_differentiable_programming_benchmark_suite,
    run_quantum_gradient_benchmark_suite,
)
from .compiler.mlir import (
    build_compiler_ad_transform_plan,
    compile_compiler_ad_transform_plan_to_mlir,
)
from .differentiable import (
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRegistry,
    PrimitiveIdentity,
    ScalarObjective,
    VectorObjective,
    value_and_grad,
    value_and_hessian,
    value_and_jacobian,
)
from .phase.gradient_backend import (
    plan_quantum_gradient_backend,
    quantum_gradient_backend_capability,
)
from .phase.gradient_support_matrix import plan_gradient_support, run_gradient_support_matrix_audit
from .phase.qnn_framework_bridge_matrix import run_bounded_qnn_framework_bridge_matrix

FloatArray: TypeAlias = NDArray[np.float64]
UnifiedDifferentiableOperation = Literal[
    "value",
    "gradient",
    "jacobian",
    "hessian",
    "support_report",
    "diagnostic_report",
    "compile_report",
    "benchmark_report",
    "dashboard_status",
]
DifferentiableDashboardCapabilityState = Literal[
    "planned",
    "metadata_only",
    "diagnostic",
    "conformance_backed",
    "executable",
    "blocked",
    "unsupported",
]

CLAIM_BOUNDARY = (
    "unified differentiable API facade over already-supported local routes; "
    "finite-difference paths remain diagnostic, support and compile reports "
    "are fail-closed, and no hardware execution or performance claim is implied"
)


@dataclass(frozen=True)
class UnifiedDifferentiableAPIResult:
    """Stable JSON evidence envelope returned by the unified facade."""

    operation: UnifiedDifferentiableOperation
    supported: bool
    method: str
    value: float | None
    gradient: FloatArray | None
    jacobian: FloatArray | None
    hessian: FloatArray | None
    payload: Mapping[str, Any]
    claim_boundary: str = CLAIM_BOUNDARY

    @property
    def fail_closed(self) -> bool:
        """Return true when the requested operation is intentionally unsupported."""
        return not self.supported

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready unified differentiable evidence."""
        return {
            "operation": self.operation,
            "supported": self.supported,
            "fail_closed": self.fail_closed,
            "method": self.method,
            "value": self.value,
            "gradient": None if self.gradient is None else self.gradient.tolist(),
            "jacobian": None if self.jacobian is None else self.jacobian.tolist(),
            "hessian": None if self.hessian is None else self.hessian.tolist(),
            "payload": dict(self.payload),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiabilityDiagnosticReport:
    """JSON-ready explanation for a differentiability support decision."""

    request: Mapping[str, object]
    supported: bool
    blocked_reasons: tuple[str, ...]
    suggested_alternatives: tuple[str, ...]
    dependency_matrix: tuple[Mapping[str, object], ...]
    device_matrix: tuple[Mapping[str, object], ...]
    backend_matrix: tuple[Mapping[str, object], ...]
    support_payload: Mapping[str, object]
    claim_boundary: str

    @property
    def fail_closed(self) -> bool:
        """Return true when the requested route is intentionally blocked."""
        return not self.supported

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready differentiability diagnostics."""
        return {
            "request": dict(self.request),
            "supported": self.supported,
            "fail_closed": self.fail_closed,
            "blocked_reasons": list(self.blocked_reasons),
            "suggested_alternatives": list(self.suggested_alternatives),
            "dependency_matrix": [dict(row) for row in self.dependency_matrix],
            "device_matrix": [dict(row) for row in self.device_matrix],
            "backend_matrix": [dict(row) for row in self.backend_matrix],
            "support_payload": dict(self.support_payload),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiableDashboardCapabilityRow:
    """One claim-bounded row for differentiable dashboard consumers."""

    surface: str
    state: DifferentiableDashboardCapabilityState
    backing_api: str
    evidence: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.surface:
            raise ValueError("dashboard status surface must be non-empty")
        if not self.backing_api:
            raise ValueError("dashboard status backing_api must be non-empty")
        if any(not item for item in self.evidence):
            raise ValueError("dashboard status evidence entries must be non-empty")
        if any(not item for item in self.blocked_reasons):
            raise ValueError("dashboard status blocked reasons must be non-empty")
        if not self.claim_boundary:
            raise ValueError("dashboard status claim_boundary must be non-empty")

    @property
    def fail_closed(self) -> bool:
        """Return true when the dashboard row is intentionally non-executable."""
        return self.state in {"planned", "metadata_only", "diagnostic", "blocked", "unsupported"}

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready dashboard row."""
        return {
            "surface": self.surface,
            "state": self.state,
            "backing_api": self.backing_api,
            "evidence": list(self.evidence),
            "blocked_reasons": list(self.blocked_reasons),
            "fail_closed": self.fail_closed,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiableDashboardStatus:
    """Machine-readable differentiable status for GUI/audit-dashboard layers."""

    rows: tuple[DifferentiableDashboardCapabilityRow, ...]
    status_api_ready: bool
    generated_from: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("dashboard status rows must be non-empty")
        if any(not isinstance(row, DifferentiableDashboardCapabilityRow) for row in self.rows):
            raise ValueError("dashboard status rows must contain dashboard row entries")
        if not isinstance(self.status_api_ready, bool):
            raise ValueError("dashboard status status_api_ready must be boolean")
        if any(not item for item in self.generated_from):
            raise ValueError("dashboard status generated_from entries must be non-empty")
        if not self.claim_boundary:
            raise ValueError("dashboard status claim_boundary must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready dashboard status payload."""
        return {
            "status_api_ready": self.status_api_ready,
            "generated_from": list(self.generated_from),
            "rows": [row.to_dict() for row in self.rows],
            "claim_boundary": self.claim_boundary,
        }


def differentiable_value(
    objective: Callable[[Any], Any],
    values: ArrayLike,
    *,
    method: str = "parameter_shift",
    step: float | None = None,
) -> UnifiedDifferentiableAPIResult:
    """Evaluate a scalar objective through the unified differentiable facade."""
    result = value_and_grad(objective, values, method=method, step=step)
    return UnifiedDifferentiableAPIResult(
        operation="value",
        supported=True,
        method=result.method,
        value=float(result.value),
        gradient=None,
        jacobian=None,
        hessian=None,
        payload={
            "evaluations": result.evaluations,
            "parameter_names": list(result.parameter_names),
            "trainable": list(result.trainable),
        },
    )


def differentiable_gradient(
    objective: Callable[[Any], Any],
    values: ArrayLike,
    *,
    method: str = "parameter_shift",
    step: float | None = None,
) -> UnifiedDifferentiableAPIResult:
    """Evaluate a scalar objective and gradient through the unified facade."""
    result = value_and_grad(objective, values, method=method, step=step)
    return UnifiedDifferentiableAPIResult(
        operation="gradient",
        supported=True,
        method=result.method,
        value=float(result.value),
        gradient=result.gradient.astype(np.float64, copy=True),
        jacobian=None,
        hessian=None,
        payload={
            "evaluations": result.evaluations,
            "parameter_names": list(result.parameter_names),
            "trainable": list(result.trainable),
        },
    )


def differentiable_jacobian(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> UnifiedDifferentiableAPIResult:
    """Evaluate a vector objective and Jacobian through the unified facade."""
    result = value_and_jacobian(objective, values, method=method, step=step)
    return UnifiedDifferentiableAPIResult(
        operation="jacobian",
        supported=True,
        method=result.method,
        value=None,
        gradient=None,
        jacobian=result.jacobian.astype(np.float64, copy=True),
        hessian=None,
        payload={
            "objective_value": result.value.tolist(),
            "evaluations": result.evaluations,
            "parameter_names": list(result.parameter_names),
            "trainable": list(result.trainable),
        },
    )


def differentiable_hessian(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    method: str = "finite_difference",
    step: float = 1.0e-4,
) -> UnifiedDifferentiableAPIResult:
    """Evaluate a scalar objective and Hessian through the unified facade."""
    result = value_and_hessian(objective, values, method=method, step=step)
    return UnifiedDifferentiableAPIResult(
        operation="hessian",
        supported=True,
        method=result.method,
        value=float(result.value),
        gradient=None,
        jacobian=None,
        hessian=result.hessian.astype(np.float64, copy=True),
        payload={
            "evaluations": result.evaluations,
            "parameter_names": list(result.parameter_names),
            "trainable": list(result.trainable),
        },
    )


def differentiable_support_report(
    *,
    gate: str,
    observable: str,
    backend: str = "statevector",
    transform: str = "grad",
    adapter: str = "native",
    n_params: int = 1,
    shift_terms: int = 1,
    shots: int | None = None,
    allow_hardware: bool = False,
) -> UnifiedDifferentiableAPIResult:
    """Return a fail-closed support report for a quantum-gradient route."""
    plan = plan_gradient_support(
        gate=gate,
        observable=observable,
        backend=backend,
        transform=transform,
        adapter=adapter,
        n_params=n_params,
        shift_terms=shift_terms,
        shots=shots,
        allow_hardware=allow_hardware,
    )
    return UnifiedDifferentiableAPIResult(
        operation="support_report",
        supported=plan.supported,
        method=plan.recommended_method,
        value=None,
        gradient=None,
        jacobian=None,
        hessian=None,
        payload=plan.to_dict(),
        claim_boundary=plan.claim_boundary,
    )


def explain_differentiability(
    *,
    gate: str,
    observable: str,
    backend: str = "statevector",
    transform: str = "grad",
    adapter: str = "native",
    n_params: int = 1,
    shift_terms: int = 1,
    shots: int | None = None,
    allow_hardware: bool = False,
) -> DifferentiabilityDiagnosticReport:
    """Explain whether a differentiable route can run and why it may fail closed."""
    plan = plan_gradient_support(
        gate=gate,
        observable=observable,
        backend=backend,
        transform=transform,
        adapter=adapter,
        n_params=n_params,
        shift_terms=shift_terms,
        shots=shots,
        allow_hardware=allow_hardware,
    )
    support_payload = plan.to_dict()
    backend_names = _diagnostic_backend_names(backend)
    backend_matrix = tuple(
        _backend_plan_row(
            selected_backend,
            n_params=n_params,
            shift_terms=shift_terms,
            shots=shots,
            allow_hardware=allow_hardware,
        )
        for selected_backend in backend_names
    )
    alternatives = _unique_strings(
        plan.alternatives
        + plan.backend_plan.alternatives
        + tuple(
            str(alternative)
            for row in backend_matrix
            for alternative in cast(Sequence[object], row["alternatives"])
        )
    )
    return DifferentiabilityDiagnosticReport(
        request={
            "gate": gate,
            "observable": observable,
            "backend": backend,
            "transform": transform,
            "adapter": adapter,
            "n_params": n_params,
            "shift_terms": shift_terms,
            "shots": shots,
            "allow_hardware": allow_hardware,
        },
        supported=plan.supported,
        blocked_reasons=plan.blocked_reasons,
        suggested_alternatives=alternatives,
        dependency_matrix=_dependency_matrix_rows(),
        device_matrix=tuple(_device_capability_row(name) for name in backend_names),
        backend_matrix=backend_matrix,
        support_payload=support_payload,
        claim_boundary=(
            "differentiability diagnostic report only; support, dependency, "
            "device, and backend rows are planning evidence and do not execute "
            "objectives, provider callbacks, hardware jobs, or performance benchmarks"
        ),
    )


def differentiable_compile_report(
    *,
    primitive_identities: Sequence[str | PrimitiveIdentity] | None = None,
    registry: CustomDerivativeRegistry = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    transform: str = "jvp_vjp_adjoint",
) -> UnifiedDifferentiableAPIResult:
    """Return compiler-AD planning evidence for registered primitives."""
    plan = build_compiler_ad_transform_plan(registry, transform=transform)
    selected = _selected_primitive_keys(primitive_identities)
    statuses = (
        plan.statuses
        if selected is None
        else tuple(status for status in plan.statuses if status.identity.key in selected)
    )
    if selected is not None and len(statuses) != len(selected):
        present = {status.identity.key for status in statuses}
        missing = ", ".join(sorted(selected - present))
        raise ValueError(f"unknown primitive identities for compile report: {missing}")
    filtered_plan = type(plan)(
        statuses=statuses,
        dialect=plan.dialect,
        transform=plan.transform,
        executable_backend=plan.executable_backend,
        claim_boundary=plan.claim_boundary,
    )
    mlir_module = compile_compiler_ad_transform_plan_to_mlir(filtered_plan)
    return UnifiedDifferentiableAPIResult(
        operation="compile_report",
        supported=bool(statuses),
        method=f"compiler_ad_{filtered_plan.transform}",
        value=None,
        gradient=None,
        jacobian=None,
        hessian=None,
        payload={
            "primitive_count": len(statuses),
            "primitive_identities": [status.identity.key for status in statuses],
            "executable_backend": filtered_plan.executable_backend,
            "mlir": mlir_module.text,
        },
        claim_boundary=filtered_plan.claim_boundary,
    )


def differentiable_benchmark_report() -> UnifiedDifferentiableAPIResult:
    """Return local non-performance benchmark/conformance evidence."""
    program_rows = run_differentiable_programming_benchmark_suite()
    quantum_rows = run_quantum_gradient_benchmark_suite()
    support_audit = run_gradient_support_matrix_audit()
    passed = (
        all(row.passed for row in program_rows)
        and all(row.passed for row in quantum_rows)
        and support_audit.passed
    )
    return UnifiedDifferentiableAPIResult(
        operation="benchmark_report",
        supported=passed,
        method="local_conformance_bundle",
        value=None,
        gradient=None,
        jacobian=None,
        hessian=None,
        payload={
            "program_ad_case_count": len(program_rows),
            "quantum_gradient_case_count": len(quantum_rows),
            "support_audit_passed": support_audit.passed,
            "program_ad_cases": [_dataclass_payload(row) for row in program_rows],
            "quantum_gradient_cases": [_dataclass_payload(row) for row in quantum_rows],
        },
        claim_boundary=(
            "local deterministic conformance benchmark bundle; not isolated "
            "performance, hardware, or provider execution evidence"
        ),
    )


def differentiable_dashboard_status(
    *,
    include_conformance: bool = False,
) -> DifferentiableDashboardStatus:
    """Return the claim-bounded status contract for a future GUI/dashboard."""
    conformance_passed: bool | None = None
    if include_conformance:
        conformance_passed = differentiable_benchmark_report().supported

    rows = [
        DifferentiableDashboardCapabilityRow(
            surface="unified_differentiable_api",
            state="executable",
            backing_api="differentiable_api",
            evidence=(
                "UnifiedDifferentiableAPIResult",
                "differentiable_value",
                "differentiable_gradient",
                "differentiable_jacobian",
                "differentiable_hessian",
            ),
            blocked_reasons=(),
            claim_boundary=CLAIM_BOUNDARY,
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_ir",
            state="metadata_only",
            backing_api="whole_program_value_and_grad",
            evidence=("ProgramADEffectIR", "WholeProgramADResult.program_ir"),
            blocked_reasons=("complete bytecode/source compiler frontend remains open",),
            claim_boundary="metadata and executed-trace evidence only; not full arbitrary Python AD",
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_ir_roundtrip",
            state="metadata_only",
            backing_api="parse_program_ad_effect_ir",
            evidence=("parse_program_ad_effect_ir", "program_ad_effect_ir.v1"),
            blocked_reasons=(
                "parser is metadata-only and not a bytecode/source compiler frontend",
            ),
            claim_boundary=(
                "bounded Program AD IR JSON metadata round-trip only; not a full "
                "compiler frontend, alias lattice, Rust interpreter, LLVM/JIT "
                "lowering, provider, or hardware evidence"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_alias_effects",
            state="metadata_only",
            backing_api="analyze_program_ad_alias_effects",
            evidence=(
                "ProgramADAliasEffectAnalysis",
                "ProgramADAliasSet",
                "shape_view_alias_metadata_contracts",
                "slice_mutation_alias_metadata_contracts",
                "loop_carried_state_alias_metadata_contracts",
            ),
            blocked_reasons=("full static alias lattice remains open",),
            claim_boundary="metadata_only_no_general_alias_lattice",
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_python_semantics",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="whole_program_value_and_grad",
            evidence=(
                "WholeProgramSemanticsReport",
                "python_semantics_list_comprehension",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded plain list-comprehension whole-program AD semantics only; "
                "filtered, set, and dict comprehensions remain fail-closed; no "
                "compiler, Rust, LLVM, JIT, hardware, or performance claim"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_array_indexing",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "scpn.program_ad.array:{getitem,take,take_along_axis,delete,pad,insert}",
                "indexing_static_gather_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded static array indexing, gather, delete, constant-pad, and "
                "constant-insert Program AD semantics only; dynamic indices, "
                "dynamic insertion values, Rust, LLVM/JIT, hardware, and "
                "performance promotion remain blocked"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_linalg_primitives",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "scpn.program_ad.linalg:{trace,diag,diagflat,det,inv,solve,matrix_power,multi_dot}",
                "linalg_primitive_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded local Program AD linalg primitive conformance only; "
                "spectral multiplicity, rank-threshold crossings, wider native "
                "LLVM/JIT kernels, Rust interpreter promotion, hardware, and "
                "performance promotion remain blocked"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_structured_primitives",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "scpn.program_ad.product:{inner,outer,matmul,tensordot,einsum}",
                "scpn.program_ad.interpolation:interp",
                "scpn.program_ad.signal:{convolve,correlate}",
                "scpn.program_ad.stencil:gradient",
                "structured_numeric_primitive_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "structured numeric Program AD primitive conformance only; "
                "dynamic interpolation grids, singular stencil spacing, Rust/LLVM "
                "executable lowering, hardware, and performance promotion remain blocked"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_cumulative_primitives",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "scpn.program_ad.cumulative:{cumsum,cumprod,diff}",
                "cumulative_primitive_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded cumsum, cumprod, and diff Program AD primitive "
                "conformance only; dynamic axis promotion, Rust/LLVM executable "
                "lowering, hardware, and performance promotion remain blocked"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_assembly_primitives",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "scpn.program_ad.assembly:{zeros_like,ones_like,full_like,hstack,vstack,column_stack,dstack}",
                "assembly_primitive_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded like-constructor and stack assembly Program AD "
                "primitive conformance only; dynamic shape assembly, "
                "Rust/LLVM executable lowering, hardware, and performance "
                "promotion remain blocked"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_reduction_primitives",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "scpn.program_ad.reduction:{sum,prod,mean,var,std,trapezoid,max,min,median,quantile,percentile}",
                "reduction_primitive_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded Program AD reduction primitive conformance only; dynamic "
                "axes, dynamic q, strict-order selectors, zero-variance standard "
                "deviation, Rust/LLVM executable lowering, hardware, and "
                "performance promotion remain blocked; scalar q order-statistics "
                "are covered only under local deterministic conformance"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_shape_primitives",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "scpn.program_ad.shape:{reshape,ravel,transpose,expand_dims,squeeze,swapaxes,moveaxis,repeat,atleast_1d,atleast_2d,atleast_3d,tile,roll,rot90,flip,flipud,fliplr}",
                "shape_primitive_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded Program AD shape primitive conformance only; dynamic "
                "shape arguments, invalid axes, Rust/LLVM executable lowering, "
                "hardware, and performance promotion remain blocked; rank "
                "promotion is covered only under local deterministic conformance"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="primitive_contracts",
            state="executable",
            backing_api="primitive_complete_contract_for",
            evidence=("PrimitiveContract", "PrimitiveTransformRule"),
            blocked_reasons=(),
            claim_boundary="registered primitive contracts only; unknown primitives fail closed",
        ),
        DifferentiableDashboardCapabilityRow(
            surface="higher_order_transform_algebra",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "transform_nesting_vmap_program_grad",
                "transform_nesting_custom_rule_vmap_jvp_vjp",
                "transform_nesting_program_ad_vmap_jvp_vjp",
                "transform_nesting_whole_program_higher_order",
                "transform_nesting_program_ad_hessian",
                "transform_nesting_program_ad_hessian_jvp_vjp",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "local transform-algebra conformance for vmap, exact custom JVP/VJP "
                "rules, whole-program grad, JVP/VJP, jacfwd, jacrev, and local "
                "Hessian transforms including JVP/VJP over Hessian transforms only; "
                "no compiler, JIT, hardware, or performance claim"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="nondifferentiability_diagnostics",
            state="diagnostic",
            backing_api="primitive_contract_for",
            evidence=(
                "ProgramADLinalgConditioningDiagnostic",
                "program_ad_elementwise:sign",
                "program_ad_elementwise:heaviside",
            ),
            blocked_reasons=("diagnostic rows do not execute or promote derivative kernels",),
            claim_boundary="local diagnostic evidence only; no provider, hardware, or benchmark claim",
        ),
        DifferentiableDashboardCapabilityRow(
            surface="benchmark_conformance",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="differentiable_benchmark_report",
            evidence=("run_differentiable_programming_benchmark_suite",),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "local deterministic conformance evidence; not isolated performance, "
                "hardware, or provider execution evidence"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="polyglot_compiler_chain",
            state="blocked",
            backing_api="differentiable_compile_report",
            evidence=("compile_compiler_ad_transform_plan_to_mlir",),
            blocked_reasons=(
                "native Rust Program AD interpreter is not promoted",
                "native LLVM/JIT differentiated kernels remain blocked until runtime verified",
            ),
            claim_boundary="compiler/interchange planning evidence only unless executable kernels pass runtime verification",
        ),
        DifferentiableDashboardCapabilityRow(
            surface="provider_and_hardware_gradients",
            state="blocked",
            backing_api="explain_differentiability",
            evidence=("DifferentiabilityDiagnosticReport",),
            blocked_reasons=(
                "live provider and hardware gradient execution require explicit policy evidence",
            ),
            claim_boundary="planning and diagnostic evidence only; no hardware job submission",
        ),
        DifferentiableDashboardCapabilityRow(
            surface="gui_frontend",
            state="planned",
            backing_api="differentiable_dashboard_status",
            evidence=("DifferentiableDashboardStatus",),
            blocked_reasons=("frontend implementation is planned after this status contract",),
            claim_boundary="dashboard backing contract only; no user-interface implementation claim",
        ),
    ]
    return DifferentiableDashboardStatus(
        rows=tuple(rows),
        status_api_ready=True,
        generated_from=(
            "differentiable_api",
            "program_ad_capability_contracts",
            "program_ad_effect_ir.v1",
            "compiler_ad_transform_plan",
            "gradient_support_matrix",
        ),
        claim_boundary=(
            "machine-readable status for audit dashboards; row states must be displayed "
            "without upgrading planned, metadata-only, diagnostic, blocked, or unsupported routes"
        ),
    )


def differentiable_api(
    operation: UnifiedDifferentiableOperation,
    *,
    objective: Callable[[Any], Any] | None = None,
    values: ArrayLike | None = None,
    method: str | None = None,
    step: float | None = None,
    gate: str = "ry",
    observable: str = "pauli_expectation",
    backend: str = "statevector",
    transform: str = "grad",
    adapter: str = "native",
    n_params: int = 1,
    shift_terms: int = 1,
    shots: int | None = None,
    allow_hardware: bool = False,
    primitive_identities: Sequence[str | PrimitiveIdentity] | None = None,
    registry: CustomDerivativeRegistry = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
) -> UnifiedDifferentiableAPIResult:
    """Dispatch one supported unified differentiable operation."""
    if operation == "value":
        return differentiable_value(
            _require_objective(objective),
            _require_values(values),
            method="parameter_shift" if method is None else method,
            step=step,
        )
    if operation == "gradient":
        return differentiable_gradient(
            _require_objective(objective),
            _require_values(values),
            method="parameter_shift" if method is None else method,
            step=step,
        )
    if operation == "jacobian":
        return differentiable_jacobian(
            cast(VectorObjective, _require_objective(objective)),
            _require_values(values),
            method="finite_difference" if method is None else method,
            step=1.0e-6 if step is None else step,
        )
    if operation == "hessian":
        return differentiable_hessian(
            cast(ScalarObjective, _require_objective(objective)),
            _require_values(values),
            method="finite_difference" if method is None else method,
            step=1.0e-4 if step is None else step,
        )
    if operation == "support_report":
        return differentiable_support_report(
            gate=gate,
            observable=observable,
            backend=backend,
            transform=transform,
            adapter=adapter,
            n_params=n_params,
            shift_terms=shift_terms,
            shots=shots,
            allow_hardware=allow_hardware,
        )
    if operation == "diagnostic_report":
        report = explain_differentiability(
            gate=gate,
            observable=observable,
            backend=backend,
            transform=transform,
            adapter=adapter,
            n_params=n_params,
            shift_terms=shift_terms,
            shots=shots,
            allow_hardware=allow_hardware,
        )
        return UnifiedDifferentiableAPIResult(
            operation="diagnostic_report",
            supported=report.supported,
            method="differentiability_diagnostic",
            value=None,
            gradient=None,
            jacobian=None,
            hessian=None,
            payload=report.to_dict(),
            claim_boundary=report.claim_boundary,
        )
    if operation == "compile_report":
        return differentiable_compile_report(
            primitive_identities=primitive_identities,
            registry=registry,
            transform="jvp_vjp_adjoint" if method is None else method,
        )
    if operation == "benchmark_report":
        return differentiable_benchmark_report()
    if operation == "dashboard_status":
        status = differentiable_dashboard_status()
        return UnifiedDifferentiableAPIResult(
            operation="dashboard_status",
            supported=status.status_api_ready,
            method="claim_bounded_dashboard_status",
            value=None,
            gradient=None,
            jacobian=None,
            hessian=None,
            payload=status.to_dict(),
            claim_boundary=status.claim_boundary,
        )
    raise ValueError(f"unsupported unified differentiable operation: {operation!r}")


def _require_objective(objective: Callable[[Any], Any] | None) -> Callable[[Any], Any]:
    if objective is None:
        raise ValueError("objective is required for this differentiable operation")
    return objective


def _require_values(values: ArrayLike | None) -> ArrayLike:
    if values is None:
        raise ValueError("values are required for this differentiable operation")
    return values


def _selected_primitive_keys(
    primitive_identities: Sequence[str | PrimitiveIdentity] | None,
) -> set[str] | None:
    if primitive_identities is None:
        return None
    keys = {PrimitiveIdentity.parse(identity).key for identity in primitive_identities}
    if not keys:
        raise ValueError("primitive_identities must be non-empty when provided")
    return keys


def _dataclass_payload(value: object) -> dict[str, object]:
    if not is_dataclass(value):
        raise TypeError("benchmark payload values must be dataclass instances")
    payload: dict[str, object] = {}
    for field in fields(value):
        payload[field.name] = _json_ready(getattr(value, field.name))
    passed = getattr(value, "passed", None)
    if isinstance(passed, bool):
        payload["passed"] = passed
    return payload


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _dependency_matrix_rows() -> tuple[Mapping[str, object], ...]:
    bridge_matrix = run_bounded_qnn_framework_bridge_matrix()
    return tuple(
        {
            "framework": capability.framework,
            "optional_dependency": capability.optional_dependency,
            "runtime_dependency_required": capability.runtime_dependency_required,
            "implemented": capability.implemented,
            "supported": capability.supported,
            "public_api": capability.public_api,
            "gradient_route": capability.gradient_route,
            "native_framework_autodiff": capability.native_framework_autodiff,
            "tensor_output": capability.tensor_output,
            "host_boundary": capability.host_boundary,
            "fail_closed_reason": capability.fail_closed_reason,
        }
        for capability in bridge_matrix.capabilities
    )


def _diagnostic_backend_names(requested_backend: str) -> tuple[str, ...]:
    return _unique_strings(
        (
            requested_backend,
            "statevector_simulator",
            "finite_shot_simulator",
            "hardware_qpu",
        )
    )


def _device_capability_row(backend: str) -> Mapping[str, object]:
    capability = quantum_gradient_backend_capability(backend)
    return {
        "backend": capability.backend,
        "family": capability.family,
        "hardware": capability.hardware,
        "supports_parameter_shift": capability.supports_parameter_shift,
        "supports_finite_shot": capability.supports_finite_shot,
        "supports_adjoint": capability.supports_adjoint,
        "supports_spsa": capability.supports_spsa,
        "default_shots": capability.default_shots,
        "notes": list(capability.notes),
    }


def _backend_plan_row(
    backend: str,
    *,
    n_params: int,
    shift_terms: int,
    shots: int | None,
    allow_hardware: bool,
) -> Mapping[str, object]:
    plan = plan_quantum_gradient_backend(
        backend,
        n_params=n_params,
        shift_terms=shift_terms,
        shots=shots,
        finite_shot=shots is not None,
        allow_hardware=allow_hardware,
    )
    return {
        "backend": plan.backend,
        "family": plan.family,
        "method": plan.method,
        "supported": plan.supported,
        "fail_closed": plan.fail_closed,
        "evaluations": plan.evaluations,
        "shots": plan.shots,
        "finite_shot": plan.finite_shot,
        "confidence_level": plan.confidence_level,
        "requires_hardware_approval": plan.requires_hardware_approval,
        "reasons": list(plan.reasons),
        "alternatives": list(plan.alternatives),
    }


def _unique_strings(values: Sequence[object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))


__all__ = [
    "DifferentiableDashboardCapabilityRow",
    "DifferentiableDashboardCapabilityState",
    "DifferentiableDashboardStatus",
    "DifferentiabilityDiagnosticReport",
    "UnifiedDifferentiableAPIResult",
    "UnifiedDifferentiableOperation",
    "differentiable_api",
    "differentiable_benchmark_report",
    "differentiable_compile_report",
    "differentiable_dashboard_status",
    "differentiable_gradient",
    "differentiable_hessian",
    "differentiable_jacobian",
    "differentiable_support_report",
    "differentiable_value",
    "explain_differentiability",
]
