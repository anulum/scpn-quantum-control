# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Objective Execution Planner
"""Fail-closed execution planning for composed phase objectives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objectives import ComposedPhaseObjective, build_phase_control_objective


@dataclass(frozen=True)
class ComposedObjectiveExecutionPlan:
    """Support decision for a composed phase objective execution route."""

    objective_name: str
    backend: str
    supported: bool
    mode: str
    recommended_entrypoint: str | None
    reason: str
    parameter_shift_compatible: bool
    require_parameter_shift: bool
    term_count: int
    parameter_shift_terms: tuple[str, ...]
    analytic_terms: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready execution-plan metadata."""
        return {
            "objective_name": self.objective_name,
            "backend": self.backend,
            "supported": self.supported,
            "mode": self.mode,
            "recommended_entrypoint": self.recommended_entrypoint,
            "reason": self.reason,
            "parameter_shift_compatible": self.parameter_shift_compatible,
            "require_parameter_shift": self.require_parameter_shift,
            "term_count": self.term_count,
            "parameter_shift_terms": list(self.parameter_shift_terms),
            "analytic_terms": list(self.analytic_terms),
            "blocked_reasons": list(self.blocked_reasons),
            "warnings": list(self.warnings),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class ComposedObjectivePlannerAuditResult:
    """Built-in planner audit over supported and unsupported objective routes."""

    pure_plan: ComposedObjectiveExecutionPlan
    hybrid_plan: ComposedObjectiveExecutionPlan
    forced_parameter_shift_plan: ComposedObjectiveExecutionPlan
    hardware_plan: ComposedObjectiveExecutionPlan
    unknown_backend_plan: ComposedObjectiveExecutionPlan
    passed: bool
    claim_boundary: str

    @property
    def plans(self) -> tuple[ComposedObjectiveExecutionPlan, ...]:
        """Return all planner records in audit order."""
        return (
            self.pure_plan,
            self.hybrid_plan,
            self.forced_parameter_shift_plan,
            self.hardware_plan,
            self.unknown_backend_plan,
        )

    @property
    def blocked_plans(self) -> tuple[ComposedObjectiveExecutionPlan, ...]:
        """Return unsupported planner records."""
        return tuple(plan for plan in self.plans if not plan.supported)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready planner-audit metadata."""
        return {
            "pure_plan": self.pure_plan.to_dict(),
            "hybrid_plan": self.hybrid_plan.to_dict(),
            "forced_parameter_shift_plan": self.forced_parameter_shift_plan.to_dict(),
            "hardware_plan": self.hardware_plan.to_dict(),
            "unknown_backend_plan": self.unknown_backend_plan.to_dict(),
            "passed": self.passed,
            "claim_boundary": self.claim_boundary,
        }


_LOCAL_BACKENDS = frozenset(
    {
        "statevector",
        "statevector_simulator",
        "local",
        "local_statevector",
        "term_gradient",
        "analytic",
    }
)
_HARDWARE_BACKENDS = frozenset(
    {
        "hardware",
        "hardware_qpu",
        "qpu",
        "ibm",
        "ibm_quantum",
        "provider",
    }
)


def _normalise_backend(backend: str) -> str:
    normalised = backend.strip().lower()
    if not normalised:
        raise ValueError("backend must be non-empty")
    return normalised


def _term_names(objective: ComposedPhaseObjective, *, parameter_shift: bool) -> tuple[str, ...]:
    return tuple(
        term.name for term in objective.terms if term.parameter_shift_compatible is parameter_shift
    )


def _unsupported_plan(
    objective: ComposedPhaseObjective,
    *,
    backend: str,
    require_parameter_shift: bool,
    reason: str,
    blocked_reasons: tuple[str, ...],
    warnings: tuple[str, ...] = (),
) -> ComposedObjectiveExecutionPlan:
    return ComposedObjectiveExecutionPlan(
        objective_name=objective.name,
        backend=backend,
        supported=False,
        mode="unsupported",
        recommended_entrypoint=None,
        reason=reason,
        parameter_shift_compatible=objective.parameter_shift_compatible,
        require_parameter_shift=require_parameter_shift,
        term_count=len(objective.terms),
        parameter_shift_terms=_term_names(objective, parameter_shift=True),
        analytic_terms=_term_names(objective, parameter_shift=False),
        blocked_reasons=blocked_reasons,
        warnings=warnings,
        claim_boundary=(
            "planning decision only; unsupported routes must not be executed "
            "or promoted as differentiable objective support"
        ),
    )


def plan_composed_objective_execution(
    objective: ComposedPhaseObjective,
    *,
    backend: str = "statevector",
    require_parameter_shift: bool = False,
    allow_hardware: bool = False,
) -> ComposedObjectiveExecutionPlan:
    """Plan a safe execution route for a composed phase objective."""
    normalised_backend = _normalise_backend(backend)
    parameter_shift_terms = _term_names(objective, parameter_shift=True)
    analytic_terms = _term_names(objective, parameter_shift=False)
    if normalised_backend in _HARDWARE_BACKENDS:
        hardware_reason = (
            "composed-objective hardware execution is not implemented"
            if allow_hardware
            else "hardware objective execution requires an explicit provider-gradient policy"
        )
        return _unsupported_plan(
            objective,
            backend=normalised_backend,
            require_parameter_shift=require_parameter_shift,
            reason=hardware_reason,
            blocked_reasons=(hardware_reason,),
            warnings=("use local statevector or exact term-gradient routes first",),
        )
    if normalised_backend not in _LOCAL_BACKENDS:
        reason = f"unsupported composed-objective backend: {normalised_backend}"
        return _unsupported_plan(
            objective,
            backend=normalised_backend,
            require_parameter_shift=require_parameter_shift,
            reason=reason,
            blocked_reasons=(reason,),
        )
    if require_parameter_shift and analytic_terms:
        reason = "objective contains analytic terms that are not parameter-shift compatible"
        return _unsupported_plan(
            objective,
            backend=normalised_backend,
            require_parameter_shift=True,
            reason=reason,
            blocked_reasons=(reason,),
            warnings=("remove analytic terms or use train_composed_phase_objective",),
        )
    if objective.parameter_shift_compatible:
        return ComposedObjectiveExecutionPlan(
            objective_name=objective.name,
            backend=normalised_backend,
            supported=True,
            mode="pure_parameter_shift",
            recommended_entrypoint="parameter_shift_gradient_descent",
            reason="all objective terms are parameter-shift compatible",
            parameter_shift_compatible=True,
            require_parameter_shift=require_parameter_shift,
            term_count=len(objective.terms),
            parameter_shift_terms=parameter_shift_terms,
            analytic_terms=analytic_terms,
            blocked_reasons=(),
            warnings=(
                "local simulator route only; hardware execution still requires provider policy",
            ),
            claim_boundary=(
                "pure periodic composed objective may use local parameter-shift "
                "training; no hardware or global optimality claim is implied"
            ),
        )
    return ComposedObjectiveExecutionPlan(
        objective_name=objective.name,
        backend=normalised_backend,
        supported=True,
        mode="hybrid_term_gradient",
        recommended_entrypoint="train_composed_phase_objective",
        reason="objective mixes parameter-shift-compatible terms with analytic terms",
        parameter_shift_compatible=False,
        require_parameter_shift=require_parameter_shift,
        term_count=len(objective.terms),
        parameter_shift_terms=parameter_shift_terms,
        analytic_terms=analytic_terms,
        blocked_reasons=(),
        warnings=("analytic terms are trained with exact term gradients, not parameter-shift",),
        claim_boundary=(
            "hybrid composed objective uses local exact term-gradient training; "
            "analytic penalties are not promoted to quantum parameter-shift terms"
        ),
    )


def assert_composed_objective_execution_supported(
    plan: ComposedObjectiveExecutionPlan,
) -> ComposedObjectiveExecutionPlan:
    """Return a supported plan or raise with its fail-closed reason."""
    if not plan.supported:
        raise ValueError(plan.reason)
    return plan


def run_composed_objective_planner_audit() -> ComposedObjectivePlannerAuditResult:
    """Run the built-in objective planner support audit."""
    pure_objective = build_phase_control_objective(
        2,
        energy_weight=1.0,
        fidelity_target=np.zeros(2, dtype=np.float64),
        fidelity_weight=0.2,
        symmetry_pairs=((0, 1),),
        symmetry_weight=0.1,
    )
    hybrid_objective = build_phase_control_objective(
        2,
        energy_weight=1.0,
        fidelity_target=np.zeros(2, dtype=np.float64),
        fidelity_weight=0.2,
        safety_bounds=(-1.0, 1.0),
        safety_weight=0.2,
    )
    pure_plan = plan_composed_objective_execution(pure_objective)
    hybrid_plan = plan_composed_objective_execution(hybrid_objective)
    forced_parameter_shift_plan = plan_composed_objective_execution(
        hybrid_objective,
        require_parameter_shift=True,
    )
    hardware_plan = plan_composed_objective_execution(hybrid_objective, backend="hardware")
    unknown_backend_plan = plan_composed_objective_execution(
        pure_objective,
        backend="unknown_backend",
    )
    passed = (
        pure_plan.supported
        and pure_plan.mode == "pure_parameter_shift"
        and hybrid_plan.supported
        and hybrid_plan.mode == "hybrid_term_gradient"
        and not forced_parameter_shift_plan.supported
        and not hardware_plan.supported
        and not unknown_backend_plan.supported
    )
    return ComposedObjectivePlannerAuditResult(
        pure_plan=pure_plan,
        hybrid_plan=hybrid_plan,
        forced_parameter_shift_plan=forced_parameter_shift_plan,
        hardware_plan=hardware_plan,
        unknown_backend_plan=unknown_backend_plan,
        passed=passed,
        claim_boundary=(
            "local objective execution planning audit; support decisions do not "
            "execute hardware jobs or certify global optimisation"
        ),
    )
