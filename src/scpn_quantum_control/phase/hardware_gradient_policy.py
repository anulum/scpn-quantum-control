# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hardware Gradient Policy
"""Fail-closed policy checks for hardware quantum-gradient preparation.

The policy layer is deliberately separate from provider execution. It approves
only bounded dry-run or ticketed-preparation records and never submits QPU jobs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .gradient_backend import QuantumGradientPlan, plan_quantum_gradient_backend

DEFAULT_ALLOWED_HARDWARE_PROVIDERS = (
    "ibm_quantum",
    "ibm",
    "aws_braket",
    "braket",
    "rigetti",
    "ionq",
    "quantinuum",
    "quera",
    "pasqal",
    "quandela",
    "pennylane_device",
)

DEFAULT_HARDWARE_BACKEND_ALIASES = (
    "hardware",
    "qpu",
    "ibm",
    "ibm_quantum",
    "ibm_kingston",
    "ibm_brisbane",
    "ibm_fez",
    "aws_braket",
    "braket",
    "rigetti",
    "ionq",
    "quantinuum",
    "quera",
    "pasqal",
    "quandela",
    "pennylane_device",
)

DEFAULT_REQUIRED_EVIDENCE_IDS = (
    "backend_calibration_id",
    "no_qpu_gate_id",
    "claim_boundary_id",
    "cost_budget_id",
)


def _normalise_identifier(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _normalise_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_normalise_identifier(value) for value in values)


def _clean_evidence_ids(evidence_ids: Mapping[str, str] | None) -> dict[str, str]:
    if evidence_ids is None:
        return {}
    return {
        str(key): str(value).strip()
        for key, value in evidence_ids.items()
        if str(key).strip() and str(value).strip()
    }


@dataclass(frozen=True)
class HardwareGradientRequest:
    """A proposed hardware-gradient preparation request."""

    provider: str
    backend: str
    n_params: int
    shots: int | None = None
    shift_terms: int = 1
    allow_hardware: bool = False
    evidence_ids: Mapping[str, str] | None = None
    dry_run_only: bool = True
    live_execution_ticket: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class HardwareGradientPolicy:
    """Bounds for hardware-gradient preparation approval."""

    allowed_providers: tuple[str, ...] = DEFAULT_ALLOWED_HARDWARE_PROVIDERS
    hardware_backend_aliases: tuple[str, ...] = DEFAULT_HARDWARE_BACKEND_ALIASES
    required_evidence_ids: tuple[str, ...] = DEFAULT_REQUIRED_EVIDENCE_IDS
    default_shots: int = 1024
    max_shots: int = 4096
    max_total_shots: int = 65_536
    max_params: int = 32
    max_shift_terms: int = 4
    confidence_level: float = 0.95

    @property
    def normalised_allowed_providers(self) -> tuple[str, ...]:
        """Return provider identifiers in planner-normalised form."""

        return _normalise_tuple(self.allowed_providers)

    @property
    def normalised_hardware_backend_aliases(self) -> tuple[str, ...]:
        """Return hardware backend aliases in planner-normalised form."""

        return _normalise_tuple(self.hardware_backend_aliases)


@dataclass(frozen=True)
class HardwareGradientPolicyDecision:
    """Decision record returned by the hardware-gradient policy."""

    provider: str
    backend: str
    approved: bool
    mode: str
    n_params: int
    shift_terms: int
    shots_per_evaluation: int
    evaluations: int
    estimated_total_shots: int
    requires_hardware_approval: bool
    missing_evidence: tuple[str, ...]
    reasons: tuple[str, ...]
    alternatives: tuple[str, ...]
    evidence_ids: dict[str, str]
    claim_boundary: str
    has_live_execution_ticket: bool
    backend_plan_method: str
    backend_plan_family: str
    scenario: str = "custom"

    @property
    def fail_closed(self) -> bool:
        """Whether the request is blocked by policy."""

        return not self.approved

    @property
    def failure_reason(self) -> str:
        """Human-readable blocked reason string."""

        return "; ".join(self.reasons)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready decision payload."""

        return {
            "scenario": self.scenario,
            "provider": self.provider,
            "backend": self.backend,
            "approved": self.approved,
            "fail_closed": self.fail_closed,
            "mode": self.mode,
            "n_params": self.n_params,
            "shift_terms": self.shift_terms,
            "shots_per_evaluation": self.shots_per_evaluation,
            "evaluations": self.evaluations,
            "estimated_total_shots": self.estimated_total_shots,
            "requires_hardware_approval": self.requires_hardware_approval,
            "missing_evidence": list(self.missing_evidence),
            "reasons": list(self.reasons),
            "failure_reason": self.failure_reason,
            "alternatives": list(self.alternatives),
            "evidence_ids": dict(self.evidence_ids),
            "claim_boundary": self.claim_boundary,
            "has_live_execution_ticket": self.has_live_execution_ticket,
            "backend_plan_method": self.backend_plan_method,
            "backend_plan_family": self.backend_plan_family,
        }


@dataclass(frozen=True)
class HardwareGradientReadinessSuiteResult:
    """Readiness-suite summary for hardware-gradient policy scenarios."""

    records: tuple[HardwareGradientPolicyDecision, ...]

    @property
    def record_count(self) -> int:
        """Number of readiness records in the suite."""

        return len(self.records)

    @property
    def approved_count(self) -> int:
        """Number of policy decisions approved by the suite."""

        return sum(record.approved for record in self.records)

    @property
    def blocked_count(self) -> int:
        """Number of policy decisions blocked fail-closed by the suite."""

        return sum(record.fail_closed for record in self.records)

    @property
    def live_execution_approved_count(self) -> int:
        """Number of approved records that require a live-execution ticket."""

        return sum(record.approved and record.mode == "live_ticketed" for record in self.records)

    @property
    def passed(self) -> bool:
        """Whether the built-in suite preserved every expected boundary."""

        return (
            self.record_count == 6
            and self.approved_count == 1
            and self.blocked_count == 5
            and self.live_execution_approved_count == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready readiness payload."""

        return {
            "passed": self.passed,
            "record_count": self.record_count,
            "approved_count": self.approved_count,
            "blocked_count": self.blocked_count,
            "live_execution_approved_count": self.live_execution_approved_count,
            "records": [record.to_dict() for record in self.records],
        }


def evaluate_hardware_gradient_policy(
    request: HardwareGradientRequest,
    *,
    policy: HardwareGradientPolicy | None = None,
    scenario: str = "custom",
) -> HardwareGradientPolicyDecision:
    """Evaluate a hardware-gradient request without submitting any QPU job."""

    active_policy = policy or HardwareGradientPolicy()
    provider = _normalise_identifier(request.provider)
    backend = _normalise_identifier(request.backend)
    evidence_ids = _clean_evidence_ids(request.evidence_ids)
    shots_per_evaluation = (
        request.shots if request.shots is not None else active_policy.default_shots
    )
    evaluations = 2 * max(request.n_params, 0) * max(request.shift_terms, 0)
    estimated_total_shots = evaluations * max(shots_per_evaluation, 0)
    backend_plan = _plan_backend_safely(request, shots_per_evaluation)

    reasons: list[str] = []
    alternatives = [
        "statevector simulator parameter-shift",
        "finite-shot simulator with variance metadata",
        "provider callback dry-run readiness audit",
    ]

    if not request.allow_hardware:
        reasons.append("allow_hardware=True is required for hardware-gradient policy approval")
    if provider not in active_policy.normalised_allowed_providers:
        reasons.append(f"provider '{request.provider}' is not allowlisted")
    if backend not in active_policy.normalised_hardware_backend_aliases:
        reasons.append(f"backend '{request.backend}' is not a registered hardware alias")
    if request.n_params <= 0:
        reasons.append("n_params must be positive")
    if request.n_params > active_policy.max_params:
        reasons.append(
            f"n_params {request.n_params} exceed policy maximum {active_policy.max_params}"
        )
    if request.shift_terms <= 0:
        reasons.append("shift_terms must be positive")
    if request.shift_terms > active_policy.max_shift_terms:
        reasons.append(
            f"shift_terms {request.shift_terms} exceed policy maximum "
            f"{active_policy.max_shift_terms}"
        )
    if shots_per_evaluation <= 0:
        reasons.append("shots must be positive for hardware-gradient policy approval")
    if shots_per_evaluation > active_policy.max_shots:
        reasons.append(
            f"shots per evaluation {shots_per_evaluation} exceed policy maximum "
            f"{active_policy.max_shots}"
        )
    if estimated_total_shots > active_policy.max_total_shots:
        reasons.append(
            f"estimated total shots {estimated_total_shots} exceed policy maximum "
            f"{active_policy.max_total_shots}"
        )

    missing_evidence = tuple(
        sorted(
            evidence_key
            for evidence_key in active_policy.required_evidence_ids
            if evidence_key not in evidence_ids
        )
    )
    if missing_evidence:
        reasons.append("missing required evidence IDs: " + ", ".join(missing_evidence))

    if backend_plan is None:
        reasons.append("backend planner could not produce a hardware-gradient plan")
    elif not backend_plan.supported:
        reasons.extend(backend_plan.reasons)

    has_live_execution_ticket = bool(
        request.live_execution_ticket and request.live_execution_ticket.strip()
    )
    if not request.dry_run_only and not has_live_execution_ticket:
        reasons.append("live hardware-gradient execution requires live_execution_ticket")

    approved = not reasons
    mode = "blocked"
    if approved:
        mode = "dry_run" if request.dry_run_only else "live_ticketed"

    return HardwareGradientPolicyDecision(
        provider=provider,
        backend=backend,
        approved=approved,
        mode=mode,
        n_params=request.n_params,
        shift_terms=request.shift_terms,
        shots_per_evaluation=shots_per_evaluation,
        evaluations=evaluations,
        estimated_total_shots=estimated_total_shots,
        requires_hardware_approval=_requires_hardware_approval(
            backend_plan, backend, active_policy
        ),
        missing_evidence=missing_evidence,
        reasons=tuple(reasons),
        alternatives=tuple(alternatives),
        evidence_ids=evidence_ids,
        claim_boundary=_claim_boundary(approved, request.dry_run_only),
        has_live_execution_ticket=has_live_execution_ticket,
        backend_plan_method=backend_plan.method if backend_plan is not None else "unsupported",
        backend_plan_family=backend_plan.family if backend_plan is not None else "unsupported",
        scenario=scenario,
    )


def assert_hardware_gradient_policy_approved(decision: HardwareGradientPolicyDecision) -> None:
    """Raise when a hardware-gradient policy decision is blocked."""

    if decision.fail_closed:
        raise ValueError("hardware gradient policy blocked: " + decision.failure_reason)


def run_hardware_gradient_policy_readiness_suite() -> HardwareGradientReadinessSuiteResult:
    """Run built-in hardware-gradient policy scenarios."""

    evidence_ids = {
        "backend_calibration_id": "cal-readiness-ibm-quantum",
        "no_qpu_gate_id": "no-qpu-readiness-gate",
        "claim_boundary_id": "claim-boundary-readiness",
        "cost_budget_id": "cost-budget-readiness",
    }
    scenarios = (
        (
            "bounded_dry_run",
            HardwareGradientRequest(
                provider="ibm_quantum",
                backend="ibm_quantum",
                n_params=2,
                shots=512,
                allow_hardware=True,
                evidence_ids=evidence_ids,
            ),
            HardwareGradientPolicy(),
        ),
        (
            "missing_hardware_approval",
            HardwareGradientRequest(
                provider="ibm_quantum",
                backend="ibm_quantum",
                n_params=2,
                shots=512,
                evidence_ids=evidence_ids,
            ),
            HardwareGradientPolicy(),
        ),
        (
            "unknown_provider_backend",
            HardwareGradientRequest(
                provider="unregistered_qpu",
                backend="mystery_backend",
                n_params=2,
                shots=512,
                allow_hardware=True,
                evidence_ids=evidence_ids,
            ),
            HardwareGradientPolicy(),
        ),
        (
            "shot_budget_exceeded",
            HardwareGradientRequest(
                provider="ibm_quantum",
                backend="ibm_quantum",
                n_params=4,
                shots=1_000,
                allow_hardware=True,
                evidence_ids=evidence_ids,
            ),
            HardwareGradientPolicy(max_total_shots=4_000),
        ),
        (
            "missing_evidence",
            HardwareGradientRequest(
                provider="ibm_quantum",
                backend="ibm_quantum",
                n_params=2,
                shots=512,
                allow_hardware=True,
                evidence_ids={"backend_calibration_id": "cal-readiness-ibm-quantum"},
            ),
            HardwareGradientPolicy(),
        ),
        (
            "live_execution_without_ticket",
            HardwareGradientRequest(
                provider="ibm_quantum",
                backend="ibm_quantum",
                n_params=1,
                shots=256,
                allow_hardware=True,
                evidence_ids=evidence_ids,
                dry_run_only=False,
            ),
            HardwareGradientPolicy(),
        ),
    )

    records = tuple(
        evaluate_hardware_gradient_policy(request, policy=policy, scenario=name)
        for name, request, policy in scenarios
    )
    return HardwareGradientReadinessSuiteResult(records=records)


def _plan_backend_safely(
    request: HardwareGradientRequest,
    shots_per_evaluation: int,
) -> QuantumGradientPlan | None:
    if request.n_params <= 0 or request.shift_terms <= 0 or shots_per_evaluation <= 0:
        return None
    try:
        return plan_quantum_gradient_backend(
            request.backend,
            n_params=request.n_params,
            shots=shots_per_evaluation,
            shift_terms=request.shift_terms,
            allow_hardware=request.allow_hardware,
        )
    except ValueError:
        return None


def _requires_hardware_approval(
    backend_plan: QuantumGradientPlan | None,
    backend: str,
    policy: HardwareGradientPolicy,
) -> bool:
    if backend_plan is not None:
        return backend_plan.requires_hardware_approval
    return backend in policy.normalised_hardware_backend_aliases


def _claim_boundary(approved: bool, dry_run_only: bool) -> str:
    if not approved:
        return (
            "hardware-gradient policy fail-closed before provider job preparation; "
            "no hardware execution occurred"
        )
    if dry_run_only:
        return (
            "dry-run hardware-gradient policy approval for provider job preparation only; "
            "no hardware execution occurred"
        )
    return (
        "ticketed hardware-gradient policy approval for controlled provider job preparation; "
        "execution remains outside this policy record"
    )


__all__ = [
    "DEFAULT_ALLOWED_HARDWARE_PROVIDERS",
    "DEFAULT_HARDWARE_BACKEND_ALIASES",
    "DEFAULT_REQUIRED_EVIDENCE_IDS",
    "HardwareGradientPolicy",
    "HardwareGradientPolicyDecision",
    "HardwareGradientReadinessSuiteResult",
    "HardwareGradientRequest",
    "assert_hardware_gradient_policy_approved",
    "evaluate_hardware_gradient_policy",
    "run_hardware_gradient_policy_readiness_suite",
]
