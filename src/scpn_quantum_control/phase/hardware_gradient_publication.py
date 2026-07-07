# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hardware Gradient Publication Package
"""Publication package scaffold for no-submit XY hardware-gradient campaigns."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .hardware_gradient_campaign import (
    CAMPAIGN_SCHEMA_VERSION,
    HardwareGradientCampaignPlan,
    HardwareGradientCampaignSpec,
    default_hardware_gradient_campaign_specs,
    plan_hardware_gradient_campaign,
)

HARDWARE_GRADIENT_PUBLICATION_SCHEMA_VERSION = "scpn.hardware_gradient_publication.v1"
HARDWARE_GRADIENT_PUBLICATION_TITLE = "Hardware-Validated Quantum Gradients for XY Hamiltonians"
HardwareGradientPublicationClaimStatus = Literal["pre_registered_no_submit_scaffold"]

_REQUIRED_PROMOTION_EVIDENCE = (
    "approved live execution ticket",
    "backend calibration snapshot",
    "raw hardware count artefact",
    "statevector reference gradient",
    "same-circuit competitor comparison",
    "claim-ledger artefact ID",
    "benchmark evidence ID",
)


@dataclass(frozen=True)
class HardwareGradientPreregistration:
    """Pre-execution registration record for the publication package."""

    title: str
    research_question: str
    primary_endpoint: str
    secondary_endpoints: tuple[str, ...]
    exclusion_rules: tuple[str, ...]
    statistical_plan: str
    claim_boundary: str

    def __post_init__(self) -> None:
        _require_non_empty("title", self.title)
        _require_non_empty("research_question", self.research_question)
        _require_non_empty("primary_endpoint", self.primary_endpoint)
        _require_non_empty("statistical_plan", self.statistical_plan)
        _require_non_empty("claim_boundary", self.claim_boundary)
        if not self.secondary_endpoints:
            raise ValueError("secondary_endpoints must not be empty")
        if not self.exclusion_rules:
            raise ValueError("exclusion_rules must not be empty")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready preregistration metadata."""
        return {
            "title": self.title,
            "research_question": self.research_question,
            "primary_endpoint": self.primary_endpoint,
            "secondary_endpoints": list(self.secondary_endpoints),
            "exclusion_rules": list(self.exclusion_rules),
            "statistical_plan": self.statistical_plan,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class HardwareGradientMethodSection:
    """Methods section scaffold derived from one campaign plan."""

    campaign_name: str
    method: str
    provider: str
    backend: str
    n_params: int
    shifted_evaluations: int
    shots_per_evaluation: int
    estimated_total_shots: int
    calibration_snapshot_required: bool
    raw_counts_required: bool
    statevector_reference_required: bool
    policy_approved_for_preparation: bool
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready methods-section metadata."""
        return {
            "campaign_name": self.campaign_name,
            "method": self.method,
            "provider": self.provider,
            "backend": self.backend,
            "n_params": self.n_params,
            "shifted_evaluations": self.shifted_evaluations,
            "shots_per_evaluation": self.shots_per_evaluation,
            "estimated_total_shots": self.estimated_total_shots,
            "calibration_snapshot_required": self.calibration_snapshot_required,
            "raw_counts_required": self.raw_counts_required,
            "statevector_reference_required": self.statevector_reference_required,
            "policy_approved_for_preparation": self.policy_approved_for_preparation,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class HardwareGradientArtifactMapEntry:
    """Required artefact slots for one campaign in the publication package."""

    campaign_name: str
    method: str
    backend: str
    required_replay_fields: tuple[str, ...]
    raw_count_fields: tuple[str, ...]
    reference_fields: tuple[str, ...]
    uncertainty_fields: tuple[str, ...]
    backend_calibration_status: str
    raw_counts_status: str
    statevector_reference_status: str
    artifact_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready artefact-map metadata."""
        return {
            "campaign_name": self.campaign_name,
            "method": self.method,
            "backend": self.backend,
            "required_replay_fields": list(self.required_replay_fields),
            "raw_count_fields": list(self.raw_count_fields),
            "reference_fields": list(self.reference_fields),
            "uncertainty_fields": list(self.uncertainty_fields),
            "backend_calibration_status": self.backend_calibration_status,
            "raw_counts_status": self.raw_counts_status,
            "statevector_reference_status": self.statevector_reference_status,
            "artifact_id": self.artifact_id,
        }


@dataclass(frozen=True)
class HardwareGradientClaimLedgerRow:
    """Draft claim-ledger row for a hardware-gradient publication campaign."""

    claim_id: str
    campaign_name: str
    method: str
    promoted: bool
    claim_boundary: str
    required_before_promotion: tuple[str, ...]
    artifact_id: str | None = None
    benchmark_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready claim-ledger metadata."""
        return {
            "claim_id": self.claim_id,
            "campaign_name": self.campaign_name,
            "method": self.method,
            "promoted": self.promoted,
            "claim_boundary": self.claim_boundary,
            "required_before_promotion": list(self.required_before_promotion),
            "artifact_id": self.artifact_id,
            "benchmark_id": self.benchmark_id,
        }


@dataclass(frozen=True)
class HardwareGradientBenchmarkPlaceholder:
    """Benchmark comparison slot that must be filled before promotion."""

    route: str
    status: str
    same_circuit_required: bool
    same_parameters_required: bool
    same_observable_required: bool
    same_shots_or_exact_mode_required: bool
    artifact_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready benchmark-placeholder metadata."""
        return {
            "route": self.route,
            "status": self.status,
            "same_circuit_required": self.same_circuit_required,
            "same_parameters_required": self.same_parameters_required,
            "same_observable_required": self.same_observable_required,
            "same_shots_or_exact_mode_required": self.same_shots_or_exact_mode_required,
            "artifact_id": self.artifact_id,
        }


@dataclass(frozen=True)
class HardwareGradientPublicationPackage:
    """No-submit publication package scaffold for XY hardware gradients."""

    title: str
    preregistration: HardwareGradientPreregistration
    method_sections: tuple[HardwareGradientMethodSection, ...]
    artifact_map: tuple[HardwareGradientArtifactMapEntry, ...]
    claim_ledger_rows: tuple[HardwareGradientClaimLedgerRow, ...]
    benchmark_placeholders: tuple[HardwareGradientBenchmarkPlaceholder, ...]
    hardware_execution_count: int
    gradient_available_count: int
    claim_status: HardwareGradientPublicationClaimStatus
    claim_boundary: str

    @property
    def submission_ready(self) -> bool:
        """Return whether the package contains enough evidence for submission."""
        return (
            self.hardware_execution_count > 0
            and self.gradient_available_count > 0
            and all(row.promoted for row in self.claim_ledger_rows)
            and all(entry.artifact_id for entry in self.artifact_map)
            and all(entry.artifact_id for entry in self.benchmark_placeholders)
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready publication package metadata."""
        return {
            "schema_version": HARDWARE_GRADIENT_PUBLICATION_SCHEMA_VERSION,
            "campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
            "title": self.title,
            "preregistration": self.preregistration.to_dict(),
            "method_sections": [section.to_dict() for section in self.method_sections],
            "artifact_map": [entry.to_dict() for entry in self.artifact_map],
            "claim_ledger_rows": [row.to_dict() for row in self.claim_ledger_rows],
            "benchmark_placeholders": [
                placeholder.to_dict() for placeholder in self.benchmark_placeholders
            ],
            "hardware_execution_count": self.hardware_execution_count,
            "gradient_available_count": self.gradient_available_count,
            "claim_status": self.claim_status,
            "submission_ready": self.submission_ready,
            "claim_boundary": self.claim_boundary,
        }

    def to_markdown(self) -> str:
        """Return a compact Markdown scaffold for reviewer handoff."""
        method_lines = [
            (
                f"- `{section.campaign_name}`: `{section.method}` on "
                f"`{section.backend}`, {section.shifted_evaluations} shifted "
                f"evaluations, {section.estimated_total_shots} planned shots."
            )
            for section in self.method_sections
        ]
        artefact_lines = [
            (
                f"- `{entry.campaign_name}`: raw counts "
                f"`{entry.raw_counts_status}`, statevector reference "
                f"`{entry.statevector_reference_status}`."
            )
            for entry in self.artifact_map
        ]
        benchmark_lines = [
            f"- `{placeholder.route}`: `{placeholder.status}`."
            for placeholder in self.benchmark_placeholders
        ]
        return "\n".join(
            (
                f"# {self.title}",
                "",
                "## Preregistration",
                f"- Question: {self.preregistration.research_question}",
                f"- Primary endpoint: {self.preregistration.primary_endpoint}",
                "",
                "## Methods",
                *method_lines,
                "",
                "## Artefact Map",
                *artefact_lines,
                "",
                "## Benchmark Placeholders",
                *benchmark_lines,
                "",
                f"Claim boundary: {self.claim_boundary}",
            )
        )


def build_hardware_gradient_publication_package(
    plans: Sequence[HardwareGradientCampaignPlan] | None = None,
) -> HardwareGradientPublicationPackage:
    """Build the no-submit publication package for XY hardware gradients."""
    campaign_plans = (
        tuple(plans)
        if plans is not None
        else tuple(
            plan_hardware_gradient_campaign(spec)
            for spec in default_hardware_gradient_campaign_specs()
        )
    )
    if not campaign_plans:
        raise ValueError("at least one hardware-gradient campaign plan is required")
    _reject_live_results(campaign_plans)
    method_sections = tuple(_method_section(plan) for plan in campaign_plans)
    artifact_map = tuple(_artifact_entry(plan.spec) for plan in campaign_plans)
    claim_rows = tuple(_claim_row(plan.spec) for plan in campaign_plans)
    return HardwareGradientPublicationPackage(
        title=HARDWARE_GRADIENT_PUBLICATION_TITLE,
        preregistration=_default_preregistration(),
        method_sections=method_sections,
        artifact_map=artifact_map,
        claim_ledger_rows=claim_rows,
        benchmark_placeholders=_default_benchmark_placeholders(),
        hardware_execution_count=sum(plan.hardware_execution for plan in campaign_plans),
        gradient_available_count=sum(plan.gradient_available for plan in campaign_plans),
        claim_status="pre_registered_no_submit_scaffold",
        claim_boundary=(
            "no-submit publication scaffold; contains preregistration, methods, "
            "artefact-map, claim-ledger, and benchmark-comparison slots only"
        ),
    )


def _default_preregistration() -> HardwareGradientPreregistration:
    return HardwareGradientPreregistration(
        title=HARDWARE_GRADIENT_PUBLICATION_TITLE,
        research_question=(
            "Do raw hardware count estimates for XY Hamiltonian gradients agree "
            "with statevector parameter-shift references within preregistered "
            "finite-shot uncertainty?"
        ),
        primary_endpoint=(
            "maximum absolute gradient error against the statevector reference "
            "for each campaign method"
        ),
        secondary_endpoints=(
            "finite-shot confidence interval coverage",
            "optimizer-step direction agreement",
            "backend-calibration sensitivity notes",
        ),
        exclusion_rules=(
            "exclude runs without backend calibration snapshot identifiers",
            "exclude runs without raw count artefacts for every shifted evaluation",
            "exclude runs whose circuit, parameters, observable, or shots differ "
            "from the preregistered comparison contract",
        ),
        statistical_plan=(
            "Report per-parameter absolute error, finite-shot standard error, "
            "confidence radius, and pass/fail status without replacing missing "
            "hardware counts by simulator values."
        ),
        claim_boundary=(
            "pre-execution registration only; no live hardware-gradient claim "
            "exists until approved raw-count artefacts are attached"
        ),
    )


def _method_section(plan: HardwareGradientCampaignPlan) -> HardwareGradientMethodSection:
    spec = plan.spec
    return HardwareGradientMethodSection(
        campaign_name=spec.name,
        method=spec.method,
        provider=spec.provider,
        backend=spec.backend,
        n_params=spec.n_params,
        shifted_evaluations=spec.evaluations,
        shots_per_evaluation=spec.shots_per_evaluation,
        estimated_total_shots=spec.estimated_total_shots,
        calibration_snapshot_required=spec.calibration_snapshot_required,
        raw_counts_required=spec.raw_counts_required,
        statevector_reference_required=spec.statevector_reference_required,
        policy_approved_for_preparation=plan.approved_for_preparation,
        claim_boundary=plan.claim_boundary,
    )


def _artifact_entry(spec: HardwareGradientCampaignSpec) -> HardwareGradientArtifactMapEntry:
    replay = spec.replay_schema()
    return HardwareGradientArtifactMapEntry(
        campaign_name=spec.name,
        method=spec.method,
        backend=spec.backend,
        required_replay_fields=replay.required_fields,
        raw_count_fields=replay.raw_count_fields,
        reference_fields=replay.reference_fields,
        uncertainty_fields=replay.uncertainty_fields,
        backend_calibration_status="required_not_captured",
        raw_counts_status="required_not_captured",
        statevector_reference_status="required_not_captured",
    )


def _claim_row(spec: HardwareGradientCampaignSpec) -> HardwareGradientClaimLedgerRow:
    return HardwareGradientClaimLedgerRow(
        claim_id=f"hardware_gradient_publication::{spec.name}",
        campaign_name=spec.name,
        method=spec.method,
        promoted=False,
        claim_boundary="planned_publication_row_no_hardware_evidence",
        required_before_promotion=_REQUIRED_PROMOTION_EVIDENCE,
    )


def _default_benchmark_placeholders() -> tuple[HardwareGradientBenchmarkPlaceholder, ...]:
    return tuple(
        HardwareGradientBenchmarkPlaceholder(
            route=route,
            status="placeholder_not_executed",
            same_circuit_required=True,
            same_parameters_required=True,
            same_observable_required=True,
            same_shots_or_exact_mode_required=True,
        )
        for route in (
            "scpn_statevector_reference",
            "pennylane_same_circuit",
            "qiskit_same_circuit",
        )
    )


def _reject_live_results(plans: Sequence[HardwareGradientCampaignPlan]) -> None:
    if any(plan.hardware_execution or plan.gradient_available for plan in plans):
        raise ValueError(
            "publication scaffold cannot contain live hardware execution or "
            "available hardware-gradient values"
        )


def _require_non_empty(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")


__all__ = [
    "HARDWARE_GRADIENT_PUBLICATION_SCHEMA_VERSION",
    "HARDWARE_GRADIENT_PUBLICATION_TITLE",
    "HardwareGradientArtifactMapEntry",
    "HardwareGradientBenchmarkPlaceholder",
    "HardwareGradientClaimLedgerRow",
    "HardwareGradientMethodSection",
    "HardwareGradientPreregistration",
    "HardwareGradientPublicationClaimStatus",
    "HardwareGradientPublicationPackage",
    "build_hardware_gradient_publication_package",
]
