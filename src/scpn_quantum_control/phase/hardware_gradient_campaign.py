# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hardware Gradient Campaign Specs
"""No-submit hardware-gradient campaign specifications for XY Hamiltonians."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .hardware_gradient_policy import (
    HardwareGradientPolicy,
    HardwareGradientPolicyDecision,
    HardwareGradientRequest,
    evaluate_hardware_gradient_policy,
)

HardwareGradientCampaignMethod = Literal["parameter_shift_vqe", "spsa"]

DEFAULT_HERON_R2_BACKENDS = ("ibm_fez", "ibm_kingston", "ibm_brisbane")
DEFAULT_CAMPAIGN_EVIDENCE_IDS = {
    "backend_calibration_id": "calibration-snapshot-required",
    "no_qpu_gate_id": "no-qpu-default-dry-run-gate",
    "claim_boundary_id": "hardware-gradient-claim-boundary",
    "cost_budget_id": "hardware-gradient-cost-budget",
}
CAMPAIGN_SCHEMA_VERSION = "scpn.hardware_gradient_campaign.v1"


@dataclass(frozen=True)
class HardwareGradientReplaySchema:
    """Required replay artefact fields for one hardware-gradient campaign."""

    required_fields: tuple[str, ...]
    raw_count_fields: tuple[str, ...]
    reference_fields: tuple[str, ...]
    uncertainty_fields: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready replay schema metadata."""
        return {
            "required_fields": list(self.required_fields),
            "raw_count_fields": list(self.raw_count_fields),
            "reference_fields": list(self.reference_fields),
            "uncertainty_fields": list(self.uncertainty_fields),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class HardwareGradientCampaignSpec:
    """No-submit campaign specification for hardware-gradient validation."""

    name: str
    method: HardwareGradientCampaignMethod
    provider: str
    backend: str
    n_params: int
    shots_per_evaluation: int
    shift_terms: int
    spsa_repetitions: int
    perturbation_radius: float | None
    seed: int | None
    evidence_ids: Mapping[str, str]
    backend_allowlist: tuple[str, ...]
    dry_run_only: bool = True
    live_execution_ticket: str | None = None
    calibration_snapshot_required: bool = True
    statevector_reference_required: bool = True
    raw_counts_required: bool = True
    publication_claim: str = "no_hardware_execution"

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("campaign name must be non-empty")
        if self.method not in {"parameter_shift_vqe", "spsa"}:
            raise ValueError("method must be parameter_shift_vqe or spsa")
        provider = self.provider.strip()
        backend = self.backend.strip()
        if not provider:
            raise ValueError("provider must be non-empty")
        if not backend:
            raise ValueError("backend must be non-empty")
        if isinstance(self.n_params, bool) or self.n_params <= 0:
            raise ValueError("n_params must be a positive integer")
        if isinstance(self.shots_per_evaluation, bool) or self.shots_per_evaluation <= 0:
            raise ValueError("shots_per_evaluation must be a positive integer")
        if isinstance(self.shift_terms, bool) or self.shift_terms <= 0:
            raise ValueError("shift_terms must be a positive integer")
        if isinstance(self.spsa_repetitions, bool) or self.spsa_repetitions <= 0:
            raise ValueError("spsa_repetitions must be a positive integer")
        if self.method == "parameter_shift_vqe" and self.spsa_repetitions != 1:
            raise ValueError("parameter_shift_vqe specs must set spsa_repetitions=1")
        if self.method == "spsa" and self.shift_terms != 1:
            raise ValueError("spsa specs must set shift_terms=1")
        if self.method == "spsa":
            radius = _as_positive_float("perturbation_radius", self.perturbation_radius)
        else:
            radius = None
        seed = _as_optional_non_negative_int("seed", self.seed)
        evidence_ids = _clean_evidence_ids(self.evidence_ids)
        missing = tuple(key for key in DEFAULT_CAMPAIGN_EVIDENCE_IDS if key not in evidence_ids)
        if missing:
            raise ValueError("missing campaign evidence IDs: " + ", ".join(missing))
        allowlist = tuple(item.strip() for item in self.backend_allowlist if item.strip())
        if not allowlist:
            raise ValueError("backend_allowlist must contain at least one backend")
        if backend not in allowlist:
            raise ValueError(f"backend {backend!r} is not in the campaign allowlist")
        if not self.dry_run_only and not (
            self.live_execution_ticket and self.live_execution_ticket.strip()
        ):
            raise ValueError("live campaign specs require live_execution_ticket")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "perturbation_radius", radius)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "backend_allowlist", allowlist)

    @property
    def evaluations(self) -> int:
        """Return shifted hardware evaluations required by the method."""
        if self.method == "spsa":
            return 2 * self.spsa_repetitions
        return 2 * self.n_params * self.shift_terms

    @property
    def estimated_total_shots(self) -> int:
        """Return total shots implied by the no-submit campaign budget."""
        return self.evaluations * self.shots_per_evaluation

    def replay_schema(self) -> HardwareGradientReplaySchema:
        """Return the required replay artefact schema for this campaign."""
        common = (
            "schema_version",
            "campaign_name",
            "method",
            "provider",
            "backend",
            "backend_calibration_id",
            "parameters",
            "shots_per_evaluation",
            "evaluation_records",
            "statevector_reference",
            "claim_boundary",
            "hardware_execution",
        )
        method_fields: tuple[str, ...]
        if self.method == "spsa":
            method_fields = (
                "seed",
                "perturbation_radius",
                "spsa_repetitions",
                "perturbation_records",
            )
        else:
            method_fields = ("shift_terms", "parameter_shift_records")
        return HardwareGradientReplaySchema(
            required_fields=common + method_fields,
            raw_count_fields=("bitstrings", "counts", "shots", "quasi_distribution"),
            reference_fields=("statevector_value", "statevector_gradient", "max_abs_error"),
            uncertainty_fields=(
                "standard_error",
                "confidence_radius",
                "finite_shot_variance",
            ),
            claim_boundary=(
                "hardware-gradient replay schema for no-submit campaign planning; "
                "not live hardware evidence until raw counts are captured under an approved ticket"
            ),
        )

    def policy_request(self) -> HardwareGradientRequest:
        """Return the hardware-gradient policy request for this campaign."""
        return HardwareGradientRequest(
            provider=self.provider,
            backend=self.backend,
            n_params=self.n_params if self.method == "parameter_shift_vqe" else 1,
            shots=self.shots_per_evaluation,
            shift_terms=self.shift_terms
            if self.method == "parameter_shift_vqe"
            else self.spsa_repetitions,
            allow_hardware=True,
            evidence_ids=self.evidence_ids,
            dry_run_only=self.dry_run_only,
            live_execution_ticket=self.live_execution_ticket,
            metadata={
                "campaign_name": self.name,
                "campaign_method": self.method,
                "estimated_total_shots": self.estimated_total_shots,
            },
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready no-submit campaign metadata."""
        return {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "name": self.name,
            "method": self.method,
            "provider": self.provider,
            "backend": self.backend,
            "n_params": self.n_params,
            "shots_per_evaluation": self.shots_per_evaluation,
            "shift_terms": self.shift_terms,
            "spsa_repetitions": self.spsa_repetitions,
            "perturbation_radius": self.perturbation_radius,
            "seed": self.seed,
            "evaluations": self.evaluations,
            "estimated_total_shots": self.estimated_total_shots,
            "evidence_ids": dict(self.evidence_ids),
            "backend_allowlist": list(self.backend_allowlist),
            "dry_run_only": self.dry_run_only,
            "has_live_execution_ticket": bool(self.live_execution_ticket),
            "calibration_snapshot_required": self.calibration_snapshot_required,
            "statevector_reference_required": self.statevector_reference_required,
            "raw_counts_required": self.raw_counts_required,
            "publication_claim": self.publication_claim,
            "replay_schema": self.replay_schema().to_dict(),
        }


@dataclass(frozen=True)
class HardwareGradientCampaignPlan:
    """Policy-evaluated no-submit plan for one hardware-gradient campaign."""

    spec: HardwareGradientCampaignSpec
    policy_decision: HardwareGradientPolicyDecision
    hardware_execution: bool
    gradient_available: bool
    claim_boundary: str

    @property
    def approved_for_preparation(self) -> bool:
        """Return whether policy approves preparation metadata."""
        return self.policy_decision.approved

    @property
    def fail_closed(self) -> bool:
        """Return whether the policy blocks this campaign."""
        return self.policy_decision.fail_closed

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready plan metadata."""
        return {
            "spec": self.spec.to_dict(),
            "policy_decision": self.policy_decision.to_dict(),
            "approved_for_preparation": self.approved_for_preparation,
            "fail_closed": self.fail_closed,
            "hardware_execution": self.hardware_execution,
            "gradient_available": self.gradient_available,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class HardwareGradientCampaignSuite:
    """Collection of no-submit hardware-gradient campaign plans."""

    plans: tuple[HardwareGradientCampaignPlan, ...]
    claim_boundary: str

    @property
    def plan_count(self) -> int:
        """Return total campaign plans."""
        return len(self.plans)

    @property
    def approved_count(self) -> int:
        """Return policy-approved preparation plans."""
        return sum(plan.approved_for_preparation for plan in self.plans)

    @property
    def blocked_count(self) -> int:
        """Return blocked plans."""
        return sum(plan.fail_closed for plan in self.plans)

    @property
    def hardware_execution_count(self) -> int:
        """Return plans that performed live hardware execution."""
        return sum(plan.hardware_execution for plan in self.plans)

    @property
    def gradient_available_count(self) -> int:
        """Return plans that contain hardware-gradient values."""
        return sum(plan.gradient_available for plan in self.plans)

    @property
    def passed(self) -> bool:
        """Return whether the suite preserves no-submit boundaries."""
        return (
            self.plan_count >= 2
            and self.approved_count >= 2
            and self.hardware_execution_count == 0
            and self.gradient_available_count == 0
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready suite metadata."""
        return {
            "passed": self.passed,
            "plan_count": self.plan_count,
            "approved_count": self.approved_count,
            "blocked_count": self.blocked_count,
            "hardware_execution_count": self.hardware_execution_count,
            "gradient_available_count": self.gradient_available_count,
            "claim_boundary": self.claim_boundary,
            "plans": [plan.to_dict() for plan in self.plans],
        }


def default_hardware_gradient_campaign_specs() -> tuple[HardwareGradientCampaignSpec, ...]:
    """Return default no-submit campaign specs for XY hardware-gradient validation."""
    evidence_ids = dict(DEFAULT_CAMPAIGN_EVIDENCE_IDS)
    allowlist = DEFAULT_HERON_R2_BACKENDS
    return (
        HardwareGradientCampaignSpec(
            name="xy_parameter_shift_vqe_heron_r2_dry_run",
            method="parameter_shift_vqe",
            provider="ibm_quantum",
            backend="ibm_fez",
            n_params=6,
            shots_per_evaluation=512,
            shift_terms=1,
            spsa_repetitions=1,
            perturbation_radius=None,
            seed=None,
            evidence_ids=evidence_ids,
            backend_allowlist=allowlist,
        ),
        HardwareGradientCampaignSpec(
            name="xy_spsa_heron_r2_dry_run",
            method="spsa",
            provider="ibm_quantum",
            backend="ibm_kingston",
            n_params=6,
            shots_per_evaluation=512,
            shift_terms=1,
            spsa_repetitions=4,
            perturbation_radius=0.08,
            seed=17,
            evidence_ids=evidence_ids,
            backend_allowlist=allowlist,
        ),
    )


def plan_hardware_gradient_campaign(
    spec: HardwareGradientCampaignSpec,
    *,
    policy: HardwareGradientPolicy | None = None,
) -> HardwareGradientCampaignPlan:
    """Evaluate one no-submit hardware-gradient campaign spec."""
    decision = evaluate_hardware_gradient_policy(
        spec.policy_request(),
        policy=policy or HardwareGradientPolicy(),
        scenario=spec.name,
    )
    return HardwareGradientCampaignPlan(
        spec=spec,
        policy_decision=decision,
        hardware_execution=False,
        gradient_available=False,
        claim_boundary=(
            "no-submit hardware-gradient campaign plan; records backend, budget, "
            "calibration, raw-count, and statevector-reference requirements only"
        ),
    )


def run_hardware_gradient_campaign_readiness_suite(
    specs: Sequence[HardwareGradientCampaignSpec] | None = None,
    *,
    policy: HardwareGradientPolicy | None = None,
) -> HardwareGradientCampaignSuite:
    """Run no-submit readiness planning for hardware-gradient campaigns."""
    campaign_specs = (
        tuple(specs) if specs is not None else default_hardware_gradient_campaign_specs()
    )
    if not campaign_specs:
        raise ValueError("at least one campaign spec is required")
    plans = tuple(plan_hardware_gradient_campaign(spec, policy=policy) for spec in campaign_specs)
    return HardwareGradientCampaignSuite(
        plans=plans,
        claim_boundary=(
            "hardware-gradient campaign readiness only; no provider sampler calls, "
            "no QPU submission, no raw hardware counts, and no hardware-gradient result"
        ),
    )


def _clean_evidence_ids(values: Mapping[str, str]) -> dict[str, str]:
    return {
        str(key): str(value).strip()
        for key, value in values.items()
        if str(key).strip() and str(value).strip()
    }


def _as_positive_float(name: str, value: float | None) -> float:
    if value is None:
        raise ValueError(f"{name} must be finite and positive")
    scalar = float(value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if scalar != scalar or scalar in {float("inf"), float("-inf")}:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _as_optional_non_negative_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return int(value)


__all__ = [
    "CAMPAIGN_SCHEMA_VERSION",
    "DEFAULT_CAMPAIGN_EVIDENCE_IDS",
    "DEFAULT_HERON_R2_BACKENDS",
    "HardwareGradientCampaignMethod",
    "HardwareGradientCampaignPlan",
    "HardwareGradientCampaignSpec",
    "HardwareGradientCampaignSuite",
    "HardwareGradientReplaySchema",
    "default_hardware_gradient_campaign_specs",
    "plan_hardware_gradient_campaign",
    "run_hardware_gradient_campaign_readiness_suite",
]
