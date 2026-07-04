# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Catalyst compiler workflow comparison boundaries.
"""Catalyst compiler-workflow evidence boundaries for differentiable comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

CatalystRunnerStatus = Literal["dependency_gap", "runtime_gap", "correctness_gap", "success"]

CATALYST_UNSUPPORTED_PROVIDER_ROUTES: tuple[str, ...] = (
    "finite_shot_provider_jobs",
    "hardware_qpu_execution",
    "cloud_provider_submission",
)

_ALLOWED_CATALYST_RUNNER_STATUSES: frozenset[str] = frozenset(
    ("dependency_gap", "runtime_gap", "correctness_gap", "success")
)

_COMPILED_WORKFLOW_BY_STATUS: dict[CatalystRunnerStatus, str] = {
    "dependency_gap": (
        "not_evaluated: PennyLane Catalyst qjit/MLIR/QIR runner is absent, so "
        "compiled quantum-classical workflow parity is not established."
    ),
    "runtime_gap": (
        "hard_gap: a configured Catalyst qjit/MLIR/QIR runner failed before "
        "validated compiled quantum-classical workflow evidence was produced."
    ),
    "correctness_gap": (
        "hard_gap: a configured Catalyst qjit/MLIR/QIR runner executed but did "
        "not match the SCPN reference objective and gradient."
    ),
    "success": (
        "bounded: a configured Catalyst qjit/MLIR/QIR runner matched one CPU "
        "float64 phase objective value and gradient; this is not arbitrary "
        "workflow evidence."
    ),
}

_COMPILED_DIFFERENTIATION_SCOPE = (
    "compiled differentiation is bounded to first-order value/gradient agreement "
    "for one CPU float64 phase objective; higher-order and arbitrary-program "
    "differentiation are not evaluated."
)

_CONTROL_FLOW_SCOPE = (
    "control flow is not evaluated; loops, conditionals, dynamic circuits, and "
    "hybrid quantum-classical branching require separate Catalyst artefacts."
)

_FINITE_SHOT_SCOPE = (
    "finite-shot limitations: no shot-noise gradients, sampler jobs, finite-shot "
    "provider executions, or statistical confidence intervals are evaluated."
)

_PROVIDER_ROUTE_SCOPE = (
    "provider route support is unsupported; no cloud provider, QPU, hardware, "
    "or remote job submission occurs in this comparison row."
)

_CATALYST_CLAIM_BOUNDARY = (
    "Dedicated Catalyst compiler-workflow comparison only; not production, "
    "provider, finite-shot, arbitrary-workflow, higher-order-AD, or performance "
    "promotion evidence."
)


@dataclass(frozen=True)
class CatalystCompilerWorkflowComparison:
    """Dedicated Catalyst compiler-workflow evidence attached to an external row."""

    runner_status: CatalystRunnerStatus
    compiled_quantum_classical_workflows: str
    compiled_differentiation: str
    control_flow: str
    finite_shot_limitations: str
    provider_route_support: str
    unsupported_provider_routes: tuple[str, ...]
    claim_boundary: str
    promotion_ready: bool = False

    def __post_init__(self) -> None:
        """Validate Catalyst comparison boundaries before JSON serialization."""
        if self.runner_status not in _ALLOWED_CATALYST_RUNNER_STATUSES:
            raise ValueError("runner_status must be a known Catalyst runner status")
        string_fields = {
            "compiled_quantum_classical_workflows": self.compiled_quantum_classical_workflows,
            "compiled_differentiation": self.compiled_differentiation,
            "control_flow": self.control_flow,
            "finite_shot_limitations": self.finite_shot_limitations,
            "provider_route_support": self.provider_route_support,
            "claim_boundary": self.claim_boundary,
        }
        for field_name, field_value in string_fields.items():
            if not field_value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.promotion_ready and self.unsupported_provider_routes:
            raise ValueError(
                "promotion_ready Catalyst rows cannot list unsupported_provider_routes"
            )
        if not self.promotion_ready and not self.unsupported_provider_routes:
            raise ValueError(
                "unsupported_provider_routes must be non-empty unless promotion_ready"
            )
        if len(set(self.unsupported_provider_routes)) != len(
            self.unsupported_provider_routes
        ) or any(not route.strip() for route in self.unsupported_provider_routes):
            raise ValueError("unsupported_provider_routes must be non-empty unique strings")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready Catalyst workflow comparison payload."""
        return {
            "runner_status": self.runner_status,
            "compiled_quantum_classical_workflows": self.compiled_quantum_classical_workflows,
            "compiled_differentiation": self.compiled_differentiation,
            "control_flow": self.control_flow,
            "finite_shot_limitations": self.finite_shot_limitations,
            "provider_route_support": self.provider_route_support,
            "unsupported_provider_routes": list(self.unsupported_provider_routes),
            "claim_boundary": self.claim_boundary,
            "promotion_ready": self.promotion_ready,
        }


def catalyst_compiler_workflow_comparison(
    *,
    runner_status: CatalystRunnerStatus,
) -> CatalystCompilerWorkflowComparison:
    """Build the standard Catalyst compiler-workflow comparison profile."""
    return CatalystCompilerWorkflowComparison(
        runner_status=runner_status,
        compiled_quantum_classical_workflows=_COMPILED_WORKFLOW_BY_STATUS[runner_status],
        compiled_differentiation=_COMPILED_DIFFERENTIATION_SCOPE,
        control_flow=_CONTROL_FLOW_SCOPE,
        finite_shot_limitations=_FINITE_SHOT_SCOPE,
        provider_route_support=_PROVIDER_ROUTE_SCOPE,
        unsupported_provider_routes=CATALYST_UNSUPPORTED_PROVIDER_ROUTES,
        claim_boundary=_CATALYST_CLAIM_BOUNDARY,
    )


__all__ = [
    "CATALYST_UNSUPPORTED_PROVIDER_ROUTES",
    "CatalystCompilerWorkflowComparison",
    "CatalystRunnerStatus",
    "catalyst_compiler_workflow_comparison",
]
