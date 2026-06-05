# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Provider Hardware Gradient Audit
"""Executable audit for provider hardware-gradient preparation readiness."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .hardware_gradient_policy import HardwareGradientPolicy
from .provider_gradient import (
    ProviderHardwareGradientPreparationResult,
    prepare_provider_hardware_parameter_shift_gradient,
)

DEFAULT_PROVIDER_HARDWARE_EVIDENCE_IDS = {
    "backend_calibration_id": "cal-readiness-provider-hardware",
    "no_qpu_gate_id": "no-qpu-provider-hardware-gate",
    "claim_boundary_id": "claim-boundary-provider-hardware",
    "cost_budget_id": "cost-budget-provider-hardware",
}


@dataclass(frozen=True)
class ProviderHardwareGradientPreparationScenario:
    """One provider hardware-gradient preparation readiness scenario."""

    name: str
    provider: str
    backend: str
    values: ArrayLike
    shots: int
    evidence_ids: Mapping[str, str] | None
    expected_approved: bool
    description: str
    dry_run_only: bool = True
    live_execution_ticket: str | None = None
    policy: HardwareGradientPolicy | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("scenario name must be non-empty")
        if not self.provider.strip():
            raise ValueError("scenario provider must be non-empty")
        if not self.backend.strip():
            raise ValueError("scenario backend must be non-empty")
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("scenario values must be a non-empty finite vector")
        if isinstance(self.shots, bool) or self.shots <= 0:
            raise ValueError("scenario shots must be positive")
        evidence_ids = _normalise_evidence_ids(self.evidence_ids)
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "provider", self.provider.strip())
        object.__setattr__(self, "backend", self.backend.strip())
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "description", self.description.strip())

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready scenario metadata."""

        return {
            "name": self.name,
            "provider": self.provider,
            "backend": self.backend,
            "values": np.asarray(self.values, dtype=np.float64).tolist(),
            "shots": self.shots,
            "evidence_ids": dict(self.evidence_ids or {}),
            "expected_approved": self.expected_approved,
            "description": self.description,
            "dry_run_only": self.dry_run_only,
            "has_live_execution_ticket": bool(self.live_execution_ticket),
            "policy": None if self.policy is None else _policy_to_dict(self.policy),
        }


@dataclass(frozen=True)
class ProviderHardwareGradientPreparationRecord:
    """Observed result for one provider hardware-gradient preparation scenario."""

    scenario: ProviderHardwareGradientPreparationScenario
    result: ProviderHardwareGradientPreparationResult
    passed: bool

    @property
    def approved(self) -> bool:
        """Whether the scenario result passed policy approval."""

        return self.result.approved

    @property
    def blocked(self) -> bool:
        """Whether the scenario result failed closed."""

        return self.result.fail_closed

    @property
    def failure_reason(self) -> str:
        """Human-readable fail-closed reason for blocked scenarios."""

        return self.result.failure_reason

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready record metadata."""

        return {
            "scenario": self.scenario.to_dict(),
            "result": self.result.to_dict(),
            "passed": self.passed,
            "approved": self.approved,
            "blocked": self.blocked,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class ProviderHardwareGradientPreparationAuditResult:
    """Executable support matrix for provider hardware-gradient preparation."""

    records: tuple[ProviderHardwareGradientPreparationRecord, ...]
    claim_boundary: str

    @property
    def record_count(self) -> int:
        """Number of preparation audit records."""

        return len(self.records)

    @property
    def approved_count(self) -> int:
        """Number of approved preparation records."""

        return sum(record.approved for record in self.records)

    @property
    def blocked_count(self) -> int:
        """Number of blocked preparation records."""

        return sum(record.blocked for record in self.records)

    @property
    def hardware_execution_count(self) -> int:
        """Number of records that submitted or executed a hardware job."""

        return sum(record.result.hardware_execution for record in self.records)

    @property
    def gradient_available_count(self) -> int:
        """Number of records that produced hardware gradients."""

        return sum(record.result.gradient_available for record in self.records)

    @property
    def blocked_records(self) -> tuple[ProviderHardwareGradientPreparationRecord, ...]:
        """Return records that failed closed."""

        return tuple(record for record in self.records if record.blocked)

    @property
    def failing_records(self) -> tuple[ProviderHardwareGradientPreparationRecord, ...]:
        """Return records whose observed state missed the expected boundary."""

        return tuple(record for record in self.records if not record.passed)

    @property
    def passed(self) -> bool:
        """Whether every built-in preparation scenario preserved its boundary."""

        return (
            self.record_count == 6
            and self.approved_count == 2
            and self.blocked_count == 4
            and self.hardware_execution_count == 0
            and self.gradient_available_count == 0
            and not self.failing_records
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready provider hardware-preparation audit metadata."""

        return {
            "passed": self.passed,
            "record_count": self.record_count,
            "approved_count": self.approved_count,
            "blocked_count": self.blocked_count,
            "hardware_execution_count": self.hardware_execution_count,
            "gradient_available_count": self.gradient_available_count,
            "claim_boundary": self.claim_boundary,
            "records": [record.to_dict() for record in self.records],
        }


def default_provider_hardware_gradient_preparation_scenarios() -> tuple[
    ProviderHardwareGradientPreparationScenario, ...
]:
    """Return built-in provider hardware-gradient preparation scenarios."""

    values = np.array([0.2, -0.4], dtype=np.float64)
    single_value = np.array([0.2], dtype=np.float64)
    evidence_ids = dict(DEFAULT_PROVIDER_HARDWARE_EVIDENCE_IDS)
    return (
        ProviderHardwareGradientPreparationScenario(
            name="bounded_dry_run_preparation",
            provider="ibm_quantum",
            backend="ibm_quantum",
            values=values,
            shots=512,
            evidence_ids=evidence_ids,
            expected_approved=True,
            description="bounded dry-run hardware-gradient preparation with full evidence",
        ),
        ProviderHardwareGradientPreparationScenario(
            name="ticketed_live_preparation",
            provider="ibm_quantum",
            backend="ibm_quantum",
            values=single_value,
            shots=256,
            evidence_ids=evidence_ids,
            expected_approved=True,
            description="ticketed live-preparation record that still performs no hardware execution",
            dry_run_only=False,
            live_execution_ticket="QPU-LIVE-PREP-READINESS-001",
        ),
        ProviderHardwareGradientPreparationScenario(
            name="missing_evidence_preparation",
            provider="ibm_quantum",
            backend="ibm_quantum",
            values=values,
            shots=512,
            evidence_ids={"backend_calibration_id": evidence_ids["backend_calibration_id"]},
            expected_approved=False,
            description="hardware-gradient preparation must block missing evidence IDs",
        ),
        ProviderHardwareGradientPreparationScenario(
            name="shot_budget_exceeded_preparation",
            provider="ibm_quantum",
            backend="ibm_quantum",
            values=values,
            shots=1_000,
            evidence_ids=evidence_ids,
            expected_approved=False,
            description="hardware-gradient preparation must block excessive total shots",
            policy=HardwareGradientPolicy(max_total_shots=2_000),
        ),
        ProviderHardwareGradientPreparationScenario(
            name="unknown_provider_backend_preparation",
            provider="unregistered_qpu",
            backend="mystery_backend",
            values=values,
            shots=512,
            evidence_ids=evidence_ids,
            expected_approved=False,
            description="unknown provider/backend aliases must fail closed",
        ),
        ProviderHardwareGradientPreparationScenario(
            name="live_without_ticket_preparation",
            provider="ibm_quantum",
            backend="ibm_quantum",
            values=single_value,
            shots=256,
            evidence_ids=evidence_ids,
            expected_approved=False,
            description="live hardware-gradient preparation requires an explicit ticket",
            dry_run_only=False,
        ),
    )


def run_provider_hardware_gradient_preparation_audit(
    scenarios: tuple[ProviderHardwareGradientPreparationScenario, ...] | None = None,
) -> ProviderHardwareGradientPreparationAuditResult:
    """Run provider hardware-gradient preparation readiness checks."""

    active_scenarios = scenarios or default_provider_hardware_gradient_preparation_scenarios()
    records = tuple(_run_scenario(scenario) for scenario in active_scenarios)
    return ProviderHardwareGradientPreparationAuditResult(
        records=records,
        claim_boundary=(
            "provider hardware-gradient preparation readiness only; no sampler calls, "
            "no hardware execution, and no hardware-gradient result"
        ),
    )


def _run_scenario(
    scenario: ProviderHardwareGradientPreparationScenario,
) -> ProviderHardwareGradientPreparationRecord:
    result = prepare_provider_hardware_parameter_shift_gradient(
        scenario.values,
        provider=scenario.provider,
        backend=scenario.backend,
        shots=scenario.shots,
        policy=scenario.policy,
        evidence_ids=scenario.evidence_ids,
        dry_run_only=scenario.dry_run_only,
        live_execution_ticket=scenario.live_execution_ticket,
    )
    passed = (
        result.approved == scenario.expected_approved
        and not result.hardware_execution
        and not result.gradient_available
    )
    return ProviderHardwareGradientPreparationRecord(
        scenario=scenario,
        result=result,
        passed=passed,
    )


def _normalise_evidence_ids(evidence_ids: Mapping[str, str] | None) -> dict[str, str]:
    if evidence_ids is None:
        return {}
    return {
        str(key): str(value).strip()
        for key, value in evidence_ids.items()
        if str(key).strip() and str(value).strip()
    }


def _policy_to_dict(policy: HardwareGradientPolicy) -> dict[str, Any]:
    return {
        "allowed_providers": list(policy.allowed_providers),
        "hardware_backend_aliases": list(policy.hardware_backend_aliases),
        "required_evidence_ids": list(policy.required_evidence_ids),
        "default_shots": policy.default_shots,
        "max_shots": policy.max_shots,
        "max_total_shots": policy.max_total_shots,
        "max_params": policy.max_params,
        "max_shift_terms": policy.max_shift_terms,
        "confidence_level": policy.confidence_level,
    }


__all__ = [
    "DEFAULT_PROVIDER_HARDWARE_EVIDENCE_IDS",
    "ProviderHardwareGradientPreparationAuditResult",
    "ProviderHardwareGradientPreparationRecord",
    "ProviderHardwareGradientPreparationScenario",
    "default_provider_hardware_gradient_preparation_scenarios",
    "run_provider_hardware_gradient_preparation_audit",
]
